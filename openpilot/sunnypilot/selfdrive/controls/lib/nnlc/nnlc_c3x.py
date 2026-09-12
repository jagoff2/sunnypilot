"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and contributors.
This file is licensed under the MIT License; see LICENSE.md.
"""
from collections import deque
import math
import numpy as np

from opendbc.sunnypilot.car.lateral_ext import get_friction
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.selfdrive.controls.lib.drive_helpers import MAX_LATERAL_ACCEL_NO_ROLL, MAX_LATERAL_JERK
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext_base import LatControlTorqueExtBase
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.helpers import MOCK_MODEL_PATH
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.model import NNTorqueModel
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.sigmoid_map_tuner import SigmoidMapTuner

FRICTION_THRESHOLD = 0.3
NN_SLOPE_DELTA = 0.05  # m/s^2, local inverse response around the requested acceleration
NN_GAIN_BOUNDS = (0.5, 1.5)  # relative to the physical torque calibration
JERK_PREVIEW_SECONDS = 0.3


def roll_pitch_adjust(roll, pitch):
  return roll * math.cos(pitch)


class NeuralNetworkLateralControl(LatControlTorqueExtBase):
  def __init__(self, lac_torque, CP, CP_SP, CI):
    super().__init__(lac_torque, CP, CP_SP, CI)
    self.params = Params()
    self.enabled = self.params.get_bool("NeuralNetworkLateralControl")
    model_path = CP_SP.neuralNetworkLateralControl.model.path
    self.has_nn_model = model_path not in (MOCK_MODEL_PATH, "")
    self.model = NNTorqueModel(model_path) if self.has_nn_model else None
    self.future_times = [0.3, 0.6, 1.0, 1.5]
    self.nn_future_times = [t + self.desired_lat_jerk_time for t in self.future_times]
    self.past_times = [-0.3, -0.2, -0.1]
    history_frames = [max(1, round(abs(t) / lac_torque.dt)) for t in self.past_times]
    self.history_frame_offsets = [history_frames[0] - frames for frames in history_frames]
    # The current sample is appended before lookup; include it in capacity so
    # index zero is exactly .3 seconds old, rather than one control tick newer.
    self.lateral_accel_desired_deque = deque(maxlen=history_frames[0] + 1)
    self.roll_deque = deque(maxlen=history_frames[0] + 1)
    self.error_deque = deque(maxlen=history_frames[0] + 1)
    self.past_future_len = len(self.past_times) + len(self.future_times)
    self.reset()
    self.sigmoid_map_tuner = SigmoidMapTuner(lac_torque, self.torque_params,
                                             self.torque_from_lateral_accel_in_torque_space, CP.carFingerprint)

  @property
  def _nnlc_enabled(self):
    return self.enabled and self.model_valid and self.has_nn_model

  def update_model_v2(self, model_v2):
    self.model_v2 = model_v2
    self.model_valid = False
    if model_v2 is not None:
      arrays = (model_v2.acceleration.y, model_v2.orientation.x, model_v2.orientation.y)
      self.model_valid = all(len(values) == len(ModelConstants.T_IDXS) and np.isfinite(values).all() for values in arrays)

  def update_limits(self):
    self.lac_torque.update_limits()

  def reset(self):
    self.lateral_accel_desired_deque.clear()
    self.roll_deque.clear()
    self.error_deque.clear()
    self.pitch = FirstOrderFilter(0.0, 0.5, self.lac_torque.dt, initialized=False)
    self.pitch_last = 0.0
    self.jerk_filter = FirstOrderFilter(0.0, 0.15, self.lac_torque.dt)
    self.actual_lateral_jerk = self.lateral_jerk_setpoint = self.lateral_jerk_measurement = self.lookahead_lateral_jerk = 0.0
    self.lat_accel_friction_factor = 0.7

  def update_lateral_lag(self, lag):
    super().update_lateral_lag(lag)
    self.nn_future_times = [t + self.desired_lat_jerk_time for t in self.future_times]

  def update_calculations(self, CS, VM, desired_lateral_accel):
    # Measured wheel jerk does not belong in acceleration feedback. Planned
    # jerk is a continuous, bounded feedforward input, without a sign gate.
    self.actual_lateral_jerk = self.lateral_jerk_measurement = 0.0
    planned_jerk = 0.0
    if self.model_valid:
      t = self.desired_lat_jerk_time
      accel_now = np.interp(t, ModelConstants.T_IDXS, self.model_v2.acceleration.y)
      accel_next = np.interp(t + JERK_PREVIEW_SECONDS, ModelConstants.T_IDXS, self.model_v2.acceleration.y)
      planned_jerk = float(np.clip((accel_next - accel_now) / JERK_PREVIEW_SECONDS, -MAX_LATERAL_JERK, MAX_LATERAL_JERK))
    self.lookahead_lateral_jerk = self.jerk_filter.update(planned_jerk)
    self.lateral_jerk_setpoint = self.lat_jerk_friction_factor * self.lookahead_lateral_jerk

  def acceleration_error(self, speed, setpoint, measurement, desired_lateral_accel, roll, past_future_rolls):
    # Estimate the inverse-model slope at the target, independent of the
    # measurement. Holding jerk and all other inputs equal removes NN bias and
    # prevents derivative sign/gate changes from reversing proportional feedback.
    # Clamp to a positive calibrated range even outside the NN's training data.
    def at_accel(accel):
      return self.model.evaluate([speed, accel, 0.0, roll] + [accel] * self.past_future_len + past_future_rolls)

    center = desired_lateral_accel - self.torque_params.latAccelOffset
    slope = (at_accel(center + NN_SLOPE_DELTA) - at_accel(center - NN_SLOPE_DELTA)) / (2.0 * NN_SLOPE_DELTA)
    nominal = 1.0 / self.torque_params.latAccelFactor
    if not math.isfinite(slope):
      slope = nominal
    slope = float(np.clip(slope, NN_GAIN_BOUNDS[0] * nominal, NN_GAIN_BOUNDS[1] * nominal))
    return slope * (setpoint - measurement)

  def calculate_neural_network(self, CS, params, calibrated_pose):
    roll = params.roll
    if calibrated_pose is not None:
      self.pitch_last = self.pitch.update(calibrated_pose.orientation.pitch)
      roll = roll_pitch_adjust(roll, self.pitch_last)
    self.roll_deque.append(roll)
    self.lateral_accel_desired_deque.append(self._desired_lateral_accel)

    # Model trajectories are already parameterized by future time, including
    # planned longitudinal motion. Warping them again with current aEgo both
    # counts that motion twice and changes the declared NN feature ages.
    future_model_times = self.nn_future_times
    past_rolls = [self.roll_deque[min(len(self.roll_deque) - 1, i)] for i in self.history_frame_offsets]
    future_rolls = [roll_pitch_adjust(np.interp(t, ModelConstants.T_IDXS, self.model_v2.orientation.x) + params.roll,
                                      np.interp(t, ModelConstants.T_IDXS, self.model_v2.orientation.y) + self.pitch_last)
                    for t in future_model_times]
    offset = self.torque_params.latAccelOffset
    past_accels = [self.lateral_accel_desired_deque[min(len(self.lateral_accel_desired_deque) - 1, i)] - offset
                   for i in self.history_frame_offsets]
    # Constrain every adjacent preview interval, not merely its distance from
    # the origin. Independent origin clamps allow alternating samples to
    # exceed the jerk envelope. Use the same bank-adjusted bounds as controlsd.
    minimum = -MAX_LATERAL_ACCEL_NO_ROLL + self._roll_compensation
    maximum = MAX_LATERAL_ACCEL_NO_ROLL + self._roll_compensation
    previous_accel = float(np.clip(self._desired_lateral_accel, minimum, maximum))
    previous_time = 0.0
    future_accels = []
    for t, relative_t in zip(future_model_times, self.future_times, strict=True):
      max_change = MAX_LATERAL_JERK * (relative_t - previous_time)
      accel = float(np.clip(np.interp(t, ModelConstants.T_IDXS, self.model_v2.acceleration.y),
                            max(minimum, previous_accel - max_change), min(maximum, previous_accel + max_change)))
      future_accels.append(accel - offset)
      previous_accel, previous_time = accel, relative_t
    error = self.acceleration_error(CS.vEgo, self._setpoint, self._measurement, self._desired_lateral_accel,
                                    roll, past_rolls + future_rolls)
    friction_input = self.update_friction_input(self._setpoint, self._measurement)
    nn_input = [CS.vEgo, self._desired_lateral_accel - offset, friction_input, roll] + past_accels + future_accels + past_rolls + future_rolls
    feedforward = self.model.evaluate(nn_input)
    if self.model.friction_override:
      # This helper returns normalized torque; the base-car helper returns m/s^2.
      feedforward += get_friction(friction_input, self._lateral_accel_deadzone, FRICTION_THRESHOLD, self.torque_params)
    return error, feedforward

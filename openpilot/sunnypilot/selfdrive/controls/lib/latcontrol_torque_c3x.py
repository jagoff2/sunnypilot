import math

import numpy as np

from openpilot.cereal import log
from opendbc.car.lateral import get_friction
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.pid import PIDController
from openpilot.selfdrive.controls.lib.drive_helpers import MAX_LATERAL_JERK
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_c3x_ext import LatControlTorqueExt

FRICTION_THRESHOLD = 0.3
LOW_SPEED_X = [0, 10, 20, 30]
LOW_SPEED_Y = [10, 8, 6, 3]


class LatControlTorque(LatControl):
  """One torque-domain PI with present-response feedback and future feedforward."""

  def __init__(self, CP, CP_SP, CI, dt):
    super().__init__(CP, CP_SP, CI, dt)
    self.torque_params = CP.lateralTuning.torque.as_builder()
    self.torque_from_lateral_accel = CI.torque_from_lateral_accel()
    self.lateral_accel_from_torque = CI.lateral_accel_from_torque()
    # Preserve physical calibration before the optional sigmoid tuner wraps it.
    # A speed-dependent fitted map is not a physical slew metric.
    self._torque_from_accel = self.torque_from_lateral_accel
    self._accel_from_torque = self.lateral_accel_from_torque
    self.feedforward_gain = 0.7
    self.pid = PIDController(1.0, 0.2, rate=1.0 / self.dt)
    self.update_limits()
    self.steering_angle_deadzone_deg = self.torque_params.steeringAngleDeadzoneDeg
    self.extension = LatControlTorqueExt(self, CP, CP_SP, CI)
    self.reset()

  def update_torque_parameters(self, latAccelFactor, latAccelOffset, friction):
    if not all(math.isfinite(value) for value in (latAccelFactor, latAccelOffset, friction)) or latAccelFactor <= 0 or friction < 0:
      return
    self.torque_params.latAccelFactor = latAccelFactor
    self.torque_params.latAccelOffset = latAccelOffset
    self.torque_params.friction = friction
    self.update_limits()

  def update_limits(self):
    # NN and scalar feedback and I are always normalized torque.
    self.pid.set_limits(self.steer_max, -self.steer_max)

  def reset(self):
    super().reset()
    self.pid.reset()
    self.extension.reset()
    self._last_output_torque = 0.0
    self._last_output_lataccel = 0.0
    self._last_actual_lataccel = None
    self._last_desired_accel = None
    self._planned_jerk = FirstOrderFilter(0.0, 0.15, self.dt)
    self._last_error = 0.0
    self._using_nnlc = None
    self._output_limited = False

  @staticmethod
  def _deadzone(error, deadzone):
    return math.copysign(max(abs(error) - deadzone, 0.0), error)

  def _limit_output(self, requested_torque):
    # Re-express the previous final request in this frame's live calibration.
    # Parameter updates and scalar/NN switches cannot move that torque anchor.
    previous_accel = self._accel_from_torque(self._last_output_torque, self.torque_params)
    requested_accel = self._accel_from_torque(float(np.clip(requested_torque, -self.steer_max, self.steer_max)), self.torque_params)
    limited_accel = float(np.clip(requested_accel, previous_accel - MAX_LATERAL_JERK * self.dt,
                                  previous_accel + MAX_LATERAL_JERK * self.dt))
    return float(np.clip(self._torque_from_accel(limited_accel, self.torque_params), -self.steer_max, self.steer_max))

  @staticmethod
  def _applied_torque(car_output):
    if car_output is None:
      return None
    value = -float(car_output.actuatorsOutput.torque)  # external steering sign
    return value if math.isfinite(value) and abs(value) <= 1.001 else None

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, calibrated_pose, curvature_limited, lat_delay,
             car_output=None):
    prior_params = (self.torque_params.latAccelFactor, self.torque_params.latAccelOffset, self.torque_params.friction)
    self.extension.update_override_torque_params(self.torque_params)
    if not all(math.isfinite(value) for value in (self.torque_params.latAccelFactor, self.torque_params.latAccelOffset, self.torque_params.friction)) \
        or self.torque_params.latAccelFactor <= 0.0 or self.torque_params.friction < 0.0:
      self.torque_params.latAccelFactor, self.torque_params.latAccelOffset, self.torque_params.friction = prior_params
    self.update_limits()
    pid_log = log.ControlsState.LateralTorqueState.new_message()
    pid_log.version = 3
    observations = (desired_curvature, CS.vEgo, CS.aEgo, CS.steeringAngleDeg, CS.steeringRateDeg, params.angleOffsetDeg, params.roll)
    if not active or not all(math.isfinite(value) for value in observations) or CS.vEgo < 0.0:
      self.reset()
      return 0.0, 0.0, pid_log

    speed_squared = CS.vEgo * CS.vEgo
    if not math.isfinite(speed_squared):
      self.reset()
      return 0.0, 0.0, pid_log
    actual_curvature = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
    desired_accel = desired_curvature * speed_squared
    actual_accel = actual_curvature * speed_squared
    if not all(math.isfinite(value) for value in (desired_accel, actual_accel)):
      self.reset()
      return 0.0, 0.0, pid_log
    if calibrated_pose is not None and not math.isfinite(calibrated_pose.orientation.pitch):
      calibrated_pose = None
    roll_compensation = params.roll * ACCELERATION_DUE_TO_GRAVITY
    # Track the current accepted target. Re-delaying its declining portion can
    # command back into a turn while the planner is already unwinding. Physical
    # lag belongs to the separately maintained NN feedforward preview.
    feedback_accel = desired_accel
    desired_jerk = 0.0 if self._last_desired_accel is None else (desired_accel - self._last_desired_accel) / self.dt
    planned_jerk = self._planned_jerk.update(float(np.clip(desired_jerk, -MAX_LATERAL_JERK, MAX_LATERAL_JERK)))

    # Curvature excess is not lane-position error. Center/cut/apex additions
    # counted tracking error repeatedly and cancelled S-bend targets. Use one
    # symmetric error with only the smooth low-speed curvature boost.
    low_speed_factor = float(np.interp(CS.vEgo, LOW_SPEED_X, LOW_SPEED_Y)) ** 2
    low_speed_factor *= 1.0 - float(np.clip((CS.vEgo - 8.0) / 4.0, 0.0, 1.0))
    low_speed_gain = 1.0 + low_speed_factor / max(CS.vEgo ** 2, 1.0)
    curvature_deadzone = abs(VM.calc_curvature(math.radians(self.steering_angle_deadzone_deg), CS.vEgo, 0.0))
    accel_deadzone = curvature_deadzone * CS.vEgo ** 2
    accel_error = self._deadzone(feedback_accel - actual_accel, accel_deadzone) * low_speed_gain
    setpoint = actual_accel + accel_error
    measurement = actual_accel

    # Offset is a learned acceleration bias. Remove it once in feedforward;
    # it cancels from feedback differences and must not masquerade as error.
    scalar_ff_accel = desired_accel - roll_compensation - self.torque_params.latAccelOffset
    # Friction pickup follows requested motion, not tracking error. Feeding
    # error through friction adds a second high feedback gain around zero and
    # produces a limit cycle with a delayed low-friction steering response.
    scalar_ff_accel += get_friction(0.3 * planned_jerk, accel_deadzone, FRICTION_THRESHOLD, self.torque_params)
    error = self._torque_from_accel(accel_error, self.torque_params) - self._torque_from_accel(0.0, self.torque_params)
    feedforward = self._torque_from_accel(scalar_ff_accel, self.torque_params)

    using_nnlc = self.extension._nnlc_enabled
    if self._using_nnlc is not None and using_nnlc != self._using_nnlc:
      self.pid.reset()
      self.extension.reset()
      self._last_error = 0.0
    if using_nnlc:
      error, feedforward = self.extension.calculate(CS, VM, params, setpoint, measurement, calibrated_pose, roll_compensation,
                                                    desired_accel, actual_accel, accel_deadzone, desired_curvature, actual_curvature)
      if not math.isfinite(error) or not math.isfinite(feedforward):
        self.pid.reset()
        self.extension.reset()
        using_nnlc = False
        error = self._torque_from_accel(accel_error, self.torque_params) - self._torque_from_accel(0.0, self.torque_params)
        feedforward = self._torque_from_accel(scalar_ff_accel, self.torque_params)
    self._using_nnlc = using_nnlc
    if not math.isfinite(error) or not math.isfinite(feedforward):
      self.reset()
      return 0.0, 0.0, pid_log

    applied = self._applied_torque(car_output)
    # Freeze growth into the actuator's current limit, but permit I to unwind.
    applied_gap = 0.0 if applied is None else self._last_output_torque - applied
    actuator_limited = abs(applied_gap) > 0.01 if applied is not None else steer_limited_by_safety
    growing_i = error * self.pid.i >= 0.0
    freeze_integrator = CS.steeringPressed or CS.vEgo < 5.0 or (
      actuator_limited and growing_i and (applied is None or error * applied_gap > 0.0))
    # A zero steering deadzone must not make sign-change cleanup unreachable.
    small_error = max(0.02, accel_deadzone) / self.torque_params.latAccelFactor
    if self._last_error * error < 0.0 and abs(error) < 2.0 * small_error:
      self.pid.i *= 0.9
    previous_i = self.pid.i
    # Own integration here: the generic PID only knows its amplitude clip,
    # and can otherwise prevent opposing I from unwinding during a saturated
    # reversal. This one candidate increment is accepted against final limits.
    self.pid.speed = CS.vEgo
    if not freeze_integrator:
      self.pid.i = float(np.clip(previous_i + self.pid.k_i * self.dt * error, -self.steer_max, self.steer_max))
    self.pid.update(error, speed=CS.vEgo, feedforward=self.feedforward_gain * feedforward, freeze_integrator=True)
    requested_torque = self.pid.p + self.pid.i + self.pid.f
    output_torque = self._limit_output(requested_torque)
    # Include the FINAL slew/amplitude limit in same-frame anti-windup.
    # An opposing I increment may unwind immediately.
    if error * previous_i >= 0.0 and (self.pid.i - previous_i) * (requested_torque - output_torque) > 0.0:
      self.pid.i = previous_i
      requested_torque = self.pid.p + self.pid.i + self.pid.f
      output_torque = self._limit_output(requested_torque)
    self.pid.control = output_torque
    self._output_limited = bool(abs(requested_torque - output_torque) > 1e-6)

    if using_nnlc:
      self.extension.sigmoid_map_tuner.observe(True, CS, setpoint, measurement, desired_accel, output_torque,
                                               actuator_limited or self._output_limited, roll_compensation)

    actual_jerk = 0.0 if self._last_actual_lataccel is None else (actual_accel - self._last_actual_lataccel) / self.dt
    pid_log.active = True
    pid_log.error = float(error)
    pid_log.p = float(self.pid.p)
    pid_log.i = float(self.pid.i)
    pid_log.d = float(self.pid.d)
    pid_log.f = float(self.pid.f)
    pid_log.output = -output_torque
    pid_log.actualLateralAccel = float(actual_accel)
    pid_log.desiredLateralAccel = float(desired_accel)
    pid_log.rawDesiredLateralAccel = float(desired_accel)
    pid_log.feedbackLateralAccel = float(feedback_accel)
    pid_log.actualLateralJerk = float(actual_jerk)
    pid_log.desiredLateralJerk = float(self.extension.lookahead_lateral_jerk if using_nnlc else planned_jerk)
    pid_log.steeringRateDeg = float(CS.steeringRateDeg)
    angular_velocity = getattr(calibrated_pose, "angular_velocity", None)
    if angular_velocity is not None and math.isfinite(angular_velocity.yaw):
      pid_log.yawLateralAccel = float(CS.vEgo * angular_velocity.yaw)
    pid_log.preLimitOutput = -float(requested_torque)
    pid_log.appliedOutputValid = applied is not None
    pid_log.appliedOutput = 0.0 if applied is None else -applied
    pid_log.outputLimited = bool(self._output_limited or actuator_limited)
    pid_log.feedbackDelay = 0.0
    pid_log.saturated = bool(self._check_saturation(self.steer_max - abs(output_torque) < 1e-3,
                                                    CS, steer_limited_by_safety, curvature_limited))
    self._last_output_torque = output_torque
    self._last_output_lataccel = self._accel_from_torque(output_torque, self.torque_params)
    self._last_actual_lataccel = actual_accel
    self._last_desired_accel = desired_accel
    self._last_error = error
    return -output_torque, 0.0, pid_log

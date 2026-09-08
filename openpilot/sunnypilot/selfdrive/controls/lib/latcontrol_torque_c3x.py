import math
import numpy as np

from openpilot.cereal import log
from opendbc.car.lateral import get_friction
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.selfdrive.controls.lib.drive_helpers import MAX_LATERAL_JERK
from openpilot.common.pid import PIDController

from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_c3x_ext import LatControlTorqueExt

FRICTION_THRESHOLD = 0.3

# Low-speed curvature blending (will be faded to ~0 above ~12 m/s)
LOW_SPEED_X = [0, 10, 20, 30]          # m/s
LOW_SPEED_Y = [10, 8, 6, 3]            # unitless base, but we fade it out at speed

# Damping on measured lateral jerk to reduce snap-in and overshoot
BASE_JERK_DAMP_GAIN = 0.05  # unitless

# Lane-center feedback (previewed geometric error from curvature mismatch)
# Use only at low curvature/lat-acc where it helped; auto-fade elsewhere.
CENTER_PREVIEW_A = 4.5      # meters (base lookahead)
CENTER_PREVIEW_B = 0.25     # sec (meters per m/s)
K_Y   = 0.06                # s^-2  (reduced)
K_PSI = 0.025               # s^-1  (reduced)

# Apex guard + symmetric cut-prevention (nudges outward when tighter than plan)
K_APEX = 0.40               # dimensionless (stronger; symmetric)
K_CUT  = 0.60               # m/s^0 per (v^2 * delta_kappa) -> lat-acc reduction

# Aggressiveness scaling of PID error at high lat-acc (soften response)
AGGR_GAIN = 0.30            # 1/(1 + AGGR_GAIN*|a_lat_des|)

class LatControlTorque(LatControl):
  def __init__(self, CP, CP_SP, CI, dt):
    super().__init__(CP, CP_SP, CI, dt)
    self.torque_params = CP.lateralTuning.torque.as_builder()
    self.torque_from_lateral_accel = CI.torque_from_lateral_accel()
    self.lateral_accel_from_torque = CI.lateral_accel_from_torque()
    # Keep a stable calibration for the common limiter. The optional sigmoid
    # tuner wraps the controller's conversion methods with speed-dependent maps.
    self._slew_torque_from_lataccel = self.torque_from_lateral_accel
    self._slew_lataccel_from_torque = self.lateral_accel_from_torque
    self.feedforward_gain = 0.7
    self.pid = PIDController(1.0, 0.2, rate=1.0 / self.dt)
    self.update_limits()
    self.steering_angle_deadzone_deg = self.torque_params.steeringAngleDeadzoneDeg

    # sunnypilot extension remains fully supported
    self.extension = LatControlTorqueExt(self, CP, CP_SP, CI)

    # Internal state
    self._last_output_lataccel = 0.0
    self._last_output_torque = 0.0
    self._last_actual_lataccel = None
    self._last_error = 0.0
    self._output_limited = False
    self._limit_error = 0.0
    self._using_nnlc = None

  def update_torque_parameters(self, latAccelFactor, latAccelOffset, friction):
    self.torque_params.latAccelFactor = latAccelFactor
    self.torque_params.latAccelOffset = latAccelOffset
    self.torque_params.friction = friction
    self.update_limits()

  def update_limits(self):
    # NN feedback is in normalized torque; the scalar controller uses lat accel.
    if hasattr(self, "extension") and self.extension._nnlc_enabled:
      self.pid.set_limits(self.steer_max, -self.steer_max)
    else:
      self.pid.set_limits(self.lateral_accel_from_torque(self.steer_max, self.torque_params),
                          self.lateral_accel_from_torque(-self.steer_max, self.torque_params))

  def reset(self):
    super().reset()
    self.pid.reset()
    self.extension.reset()
    self._last_output_lataccel = 0.0
    self._last_output_torque = 0.0
    self._last_actual_lataccel = None
    self._last_error = 0.0
    self._output_limited = False
    self._limit_error = 0.0
    self._using_nnlc = None

  def _apply_jerk_limit(self, target_lataccel):
    """Use the planner's command envelope in calibrated acceleration units.

    This bounds the request, not measured vehicle jerk. The previous custom
    speed table was bypassed by NNLC and imposed an incompatible new tracking
    limit when applied to its final output.
    """
    max_delta = MAX_LATERAL_JERK * self.dt
    delta = np.clip(target_lataccel - self._last_output_lataccel, -max_delta, max_delta)
    return self._last_output_lataccel + delta

  def freeze_for_output_limit(self, error):
    # Stop integrating into a slew limit, but permit I to unwind immediately
    # when the tracking error reverses while the output is still rate limited.
    return self._output_limited and error * self._limit_error > 0.0 and error * self.pid.i >= 0.0

  @staticmethod
  def _deadzone(x, dz):
    if abs(x) <= dz:
      return 0.0
    return x - math.copysign(dz, x)

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, calibrated_pose, curvature_limited, lat_delay,
             car_output=None):
    if self.extension.update_override_torque_params(self.torque_params):
      self.update_limits()

    pid_log = log.ControlsState.LateralTorqueState.new_message()
    pid_log.version = 2
    if not active:
      output_torque = 0.0
      pid_log.active = False
      self.reset()
    else:
      using_nnlc = self.extension._nnlc_enabled
      if self._using_nnlc is not None and using_nnlc != self._using_nnlc:
        # Do not carry an integrator across controllers with different units.
        # Retain the last final request so the common limiter bridges the switch.
        self.pid.reset()
        self.extension.reset()
        self._last_error = 0.0
      self._using_nnlc = using_nnlc
      # The scalar fallback may use a previously learned sigmoid conversion.
      # Its limits and conversion must use this frame's speed/roll, not the last
      # frame on which NNLC happened to be enabled.
      self.extension.sigmoid_map_tuner.update_context(CS, params.roll * ACCELERATION_DUE_TO_GRAVITY)
      self.update_limits()

      # Curvature/lat-acc measurement
      actual_curvature = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
      roll_compensation = params.roll * ACCELERATION_DUE_TO_GRAVITY
      curvature_deadzone = abs(VM.calc_curvature(math.radians(self.steering_angle_deadzone_deg), CS.vEgo, 0.0))

      # Base desired/actual lateral accelerations
      base_desired_latacc = desired_curvature * CS.vEgo ** 2
      actual_lateral_accel = actual_curvature * CS.vEgo ** 2
      lateral_accel_deadzone = curvature_deadzone * CS.vEgo ** 2

      # ---------- Lane-center preview (ONLY in easy regions) ----------
      # Use desired - actual so we pull toward the planner.
      Lp = CENTER_PREVIEW_A + CENTER_PREVIEW_B * CS.vEgo  # meters
      Lp = float(np.clip(Lp, 4.0, 20.0))

      e_kappa = (desired_curvature - actual_curvature)    # 1/m
      e_psi = e_kappa * Lp                                # rad (small-angle approx)
      e_y_preview = 0.5 * e_kappa * (Lp ** 2)             # m (previewed cross-track)

      # Fade preview strongly if the requested turn is tight or speed is high.
      # Thresholds chosen to keep preview out of high-lat regions.
      a_des_mag = abs(base_desired_latacc)
      tight_curve = abs(desired_curvature) * CS.vEgo > 0.25     # ~turn radius < ~40m at 10 m/s
      preview_enable = (a_des_mag < 2.5) and (not tight_curve)  # only when gentle

      if preview_enable:
        speed_scale = np.clip((CS.vEgo - 3.0) / 4.0, 0.0, 1.0)  # 0 @3 m/s ? 1 @7 m/s
        center_term = (K_Y * e_y_preview + K_PSI * CS.vEgo * e_psi) * speed_scale
        # = 20% of planner accel (+floor) to keep it truly "shaping"
        max_center = 0.40 * (a_des_mag + 0.5)
        center_term = float(np.clip(center_term, -max_center, max_center))
      else:
        center_term = 0.0

      desired_lateral_accel = base_desired_latacc + center_term

      # ---------- Symmetric cut-prevention (works both left & right) ----------
      # If |actual| > |desired|, reduce commanded lat-acc in the turn direction.
      # Stronger effect at higher speed (v^2) and larger inside excess.
      same_sign = desired_curvature * actual_curvature > 0.0
      inside_excess = max(0.0, abs(actual_curvature) - abs(desired_curvature)) if same_sign else 0.0
      if inside_excess > 0.0:
        cut_guard = -K_CUT * (CS.vEgo ** 2) * inside_excess * np.sign(desired_curvature)
        # Clamp guard to = 40% of planner accel (+floor)
        max_guard = 0.50 * (a_des_mag + 0.5)
        cut_guard = float(np.clip(cut_guard, -max_guard, max_guard))
        desired_lateral_accel += cut_guard

      # ---------- Apex guard (directional; only when tighter with same sign) ----------
      cutting_inside = same_sign and (abs(actual_curvature) > abs(desired_curvature))
      if cutting_inside:
        delta_k_in = (abs(actual_curvature) - abs(desired_curvature))  # =0
        apex_guard = -K_APEX * (CS.vEgo ** 2) * delta_k_in * np.sign(desired_curvature)
        guard_limit = 0.40 * (a_des_mag + 0.5)  # = 30% (apex guard is secondary to cut_guard)
        apex_guard = float(np.clip(apex_guard, -guard_limit, guard_limit))
        desired_lateral_accel += apex_guard

      # ---------- Low-speed curvature blending with fade-out ----------
      # Original low-speed helper caused bias at speed. Fade it to zero above ~12 m/s.
      low_speed_base = np.interp(CS.vEgo, LOW_SPEED_X, LOW_SPEED_Y) ** 2
      fade = 1.0 - np.clip((CS.vEgo - 8.0) / 4.0, 0.0, 1.0)  # 1 @=8 m/s ? 0 @=12 m/s
      low_speed_factor = low_speed_base * fade

      setpoint = desired_lateral_accel + low_speed_factor * desired_curvature
      measurement = actual_lateral_accel + low_speed_factor * actual_curvature
      gravity_adjusted_lateral_accel = desired_lateral_accel - roll_compensation

      # One-step latency prediction on measurement (no extra signals required)
      measured_jerk = 0.0 if self._last_actual_lataccel is None else (actual_lateral_accel - self._last_actual_lataccel) / self.dt
      pred_measurement = measurement + measured_jerk * self.dt

      # ---------- Aggressiveness scaling of PID error at high lat-acc ----------
      raw_error = float(setpoint - pred_measurement)
      aggr_scale = 1.0 / (1.0 + AGGR_GAIN * a_des_mag)
      raw_error *= aggr_scale

      # Error deadzone to avoid small oscillations near center
      error_dz = 0.5 * lateral_accel_deadzone
      pid_error = self._deadzone(raw_error, error_dz)

      # Feedforward: gravity + friction; add small jerk damping (acts on measurement)
      ff = gravity_adjusted_lateral_accel
      ff += get_friction(desired_lateral_accel - actual_lateral_accel,
                         lateral_accel_deadzone, FRICTION_THRESHOLD, self.torque_params)
      ff -= BASE_JERK_DAMP_GAIN * measured_jerk

      # Select one controller before advancing its PID. Updating the shared PID
      # twice mixed acceleration and torque errors into the same integrator.
      if using_nnlc:
        pid_log, output_torque = self.extension.update(
          CS, VM, self.pid, params, ff, pid_log,
          setpoint, pred_measurement, calibrated_pose, roll_compensation,
          desired_lateral_accel, actual_lateral_accel, lateral_accel_deadzone, gravity_adjusted_lateral_accel,
          desired_curvature, actual_curvature, steer_limited_by_safety, 0.0
        )
      else:
        freeze_integrator = steer_limited_by_safety or CS.steeringPressed or CS.vEgo < 5 or self.freeze_for_output_limit(pid_error)
        if abs(desired_curvature) > 0.02 and CS.vEgo > 15.0:
          self.pid.i = float(np.clip(self.pid.i, -0.15, 0.15))
        if np.sign(self._last_error) != np.sign(pid_error) and abs(pid_error) < 2.0 * lateral_accel_deadzone:
          self.pid.i *= 0.9
        output_lataccel = self.pid.update(pid_error, feedforward=self.feedforward_gain * ff,
                                          speed=CS.vEgo, freeze_integrator=freeze_integrator)
        output_torque = self.torque_from_lateral_accel(output_lataccel, self.torque_params)
        pid_log.error = float(pid_error)

      # Limit the selected final request. Previously NNLC overwrote this limiter,
      # and scaling only the target before limiting could amplify its old state.
      # Re-express the previous request using today's calibration so a live
      # torque-parameter update cannot itself create a command discontinuity.
      self._last_output_lataccel = self._slew_lataccel_from_torque(self._last_output_torque, self.torque_params)
      requested_torque = output_torque
      requested_lataccel = self._slew_lataccel_from_torque(output_torque, self.torque_params)
      output_lataccel = self._apply_jerk_limit(requested_lataccel)
      output_torque = float(np.clip(self._slew_torque_from_lataccel(output_lataccel, self.torque_params), -self.steer_max, self.steer_max))
      self._limit_error = requested_torque - output_torque
      self._output_limited = abs(self._limit_error) > 1e-6
      if using_nnlc:
        self.extension.sigmoid_map_tuner.observe(True, CS, self.extension._setpoint, self.extension._measurement,
                                                 desired_lateral_accel, output_torque,
                                                 steer_limited_by_safety or self._output_limited, roll_compensation)

      # Logging
      pid_log.active = True
      pid_log.p = float(self.pid.p)
      pid_log.i = float(self.pid.i)
      pid_log.d = float(self.pid.d)
      pid_log.f = float(self.pid.f)
      pid_log.output = float(-output_torque)  # TODO: log lat accel?
      pid_log.actualLateralAccel = float(actual_lateral_accel)
      pid_log.desiredLateralAccel = float(desired_lateral_accel)
      pid_log.saturated = bool(self._check_saturation(self.steer_max - abs(output_torque) < 1e-3,
                                                      CS, steer_limited_by_safety, curvature_limited))

      # State update
      self._last_output_lataccel = self._slew_lataccel_from_torque(output_torque, self.torque_params)
      self._last_output_torque = output_torque
      self._last_actual_lataccel = actual_lateral_accel
      self._last_error = pid_log.error

    # TODO left is positive in this convention
    return -output_torque, 0.0, pid_log

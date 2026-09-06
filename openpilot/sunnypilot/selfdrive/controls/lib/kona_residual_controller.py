"""Driving integration for the Kona residual torque policy.

Configuration and model loading happen at controller startup, never in the
control loop. Restart controlsd after changing the residual parameters.
"""
import logging
import math
from pathlib import Path

from openpilot.sunnypilot.selfdrive.controls.lib.kona_residual_policy import (
  ResidualPolicyState, ResidualTorquePolicy, build_observation,
)


DEFAULT_POLICY_PATH = Path(__file__).with_name("models") / "kona_residual_policy.npz"
POLICY_DT = 0.01
logger = logging.getLogger(__name__)


class KonaResidualController:
  def __init__(self, CP, params, dt: float):
    self.dt = dt
    self.state = ResidualPolicyState()
    self.policy = ResidualTorquePolicy.disabled()
    # The observation normalization and artifact were developed for this Kona
    # platform at 100 Hz. Do not silently apply them to another vehicle/rate.
    if CP.carFingerprint != "HYUNDAI_KONA_EV" or not math.isclose(dt, POLICY_DT, rel_tol=0.0, abs_tol=1e-9):
      return

    try:
      if not params.get("KonaResidualPolicyEnabled", return_default=True):
        return
      path = params.get("KonaResidualPolicyPath", return_default=True)
      path = str(path).strip() if path is not None else ""
      max_delta = params.get("KonaResidualPolicyMaxDelta", return_default=True)
      max_delta = 0.0 if max_delta is None else float(max_delta)
      if not math.isfinite(max_delta) or not 0.0 <= max_delta <= 1.0:
        raise ValueError("KonaResidualPolicyMaxDelta must be between 0 and 1")
      # Zero retains the artifact's cap; the Enabled parameter explicitly
      # disables inference. A blank path selects the packaged runtime asset.
      self.policy = ResidualTorquePolicy.from_path(path or str(DEFAULT_POLICY_PATH), max_delta)
    except Exception:
      logger.warning("Kona residual policy unavailable; using base lateral control", exc_info=True)

  def reset(self):
    self.state.reset()

  def update(self, CS, desired_curvature: float, output_torque: float, calibrated_pose, car_output=None) -> float:
    if not math.isfinite(output_torque):
      self.reset()
      return 0.0
    if not self.policy.enabled or CS.steeringPressed:
      self.reset()
      return output_torque

    # carOutput is the previous command after vehicle-specific limiting. Do
    # not fabricate applied torque if that feedback is missing or invalid.
    try:
      speed = float(CS.vEgo)
      steer_angle = float(CS.steeringAngleDeg)
      applied_torque = float(car_output.actuatorsOutput.torque)
      values = (speed, steer_angle, desired_curvature, applied_torque)
      if not all(math.isfinite(v) for v in values):
        raise ValueError("nonfinite residual observation")
      bearing = None if calibrated_pose is None else math.degrees(calibrated_pose.orientation.yaw)
      if bearing is not None and not math.isfinite(bearing):
        raise ValueError("nonfinite vehicle heading")
    except (AttributeError, TypeError, ValueError, OverflowError):
      self.reset()
      return output_torque

    try:
      # Internal controller torque has the opposite sign to carControl and
      # carOutput. Compose with the final NNLC/jerk-aware result supplied here.
      requested_torque = -float(output_torque)
      observation = build_observation(self.state, speed, steer_angle, desired_curvature,
                                      requested_torque, applied_torque, dt_s=self.dt, vehicle_bearing_deg=bearing)
      delta = self.policy.predict_delta(observation)
      if not math.isfinite(delta):
        raise ValueError("nonfinite residual prediction")
      delta = self.policy.rate_limit_delta(delta, self.state)
      delta = self.policy.stabilize_applied_delta(delta, observation, self.state)
      torque = requested_torque + delta
      if not math.isfinite(torque):
        raise ValueError("nonfinite residual torque")
      return -max(-1.0, min(1.0, torque))
    except Exception:
      self.policy = ResidualTorquePolicy.disabled()
      self.reset()
      logger.warning("Kona residual inference failed; using base lateral control", exc_info=True)
      return output_torque

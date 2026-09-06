#!/usr/bin/env python3
"""Runtime residual torque policy for the Hyundai Kona EV.

The policy is deliberately narrow: stock openpilot still produces the base
torque, and this module can add a bounded learned delta. Inputs mirror the
controller-observable feature definitions embedded in this module.
"""

from __future__ import annotations

import math

from dataclasses import dataclass
from pathlib import Path

import numpy as np

KONA_MAX_STEER_ANGLE_DEG = 89.6
STEER_SATURATION_FRACTION = 0.98
DESIRED_CURVATURE_TURN_THRESHOLD = 0.002
OBSERVATION_FEATURE_NAMES = [
  "speed_mps",
  "steer_angle_deg",
  "desired_curvature",
  "stock_requested_torque",
  "stock_applied_torque",
  "prev_speed_mps",
  "prev_steer_angle_deg",
  "prev_desired_curvature",
  "prev_stock_requested_torque",
  "prev_stock_applied_torque",
  "abs_steer_angle_ratio",
  "desired_lateral_accel_mps2",
  "abs_desired_curvature",
  "requested_applied_torque_error",
  "abs_stock_requested_torque",
  "abs_stock_applied_torque",
  "stock_torque_sign_mismatch",
  "speed_delta_mps",
  "steer_angle_delta_deg",
  "desired_curvature_delta",
  "stock_requested_torque_delta",
  "stock_applied_torque_delta",
  "requested_applied_torque_error_delta",
  "abs_steer_angle_delta_deg",
  "abs_stock_applied_torque_delta",
  "steer_saturation_pressure",
  "near_steer_saturation",
  "model_desired_heading_integral_deg",
  "abs_model_desired_heading_integral_deg",
  "model_curvature_sign_run_s",
  "model_turn_progress_ratio",
  "model_heading_integral_aligned_curvature",
  "vehicle_heading_change_deg",
  "abs_vehicle_heading_change_deg",
]


DEFAULT_MAX_DELTA = 0.0


@dataclass
class ResidualPolicyState:
  initialized: bool = False
  speed_mps: float = 0.0
  steer_angle_deg: float = 0.0
  desired_curvature: float = 0.0
  stock_requested_torque: float = 0.0
  stock_applied_torque: float = 0.0
  residual_torque_delta: float = 0.0
  applied_torque_sign: float = 0.0
  pending_applied_torque_sign: float = 0.0
  pending_applied_torque_sign_frames: int = 0
  model_desired_heading_integral_deg: float = 0.0
  model_curvature_sign_run_s: float = 0.0
  model_curvature_sign: float = 0.0
  vehicle_bearing_initialized: bool = False
  initial_vehicle_bearing_deg: float = 0.0

  def reset(self) -> None:
    self.initialized = False
    self.speed_mps = 0.0
    self.steer_angle_deg = 0.0
    self.desired_curvature = 0.0
    self.stock_requested_torque = 0.0
    self.stock_applied_torque = 0.0
    self.residual_torque_delta = 0.0
    self.applied_torque_sign = 0.0
    self.pending_applied_torque_sign = 0.0
    self.pending_applied_torque_sign_frames = 0
    self.model_desired_heading_integral_deg = 0.0
    self.model_curvature_sign_run_s = 0.0
    self.model_curvature_sign = 0.0
    self.vehicle_bearing_initialized = False
    self.initial_vehicle_bearing_deg = 0.0


def build_observation(
  state: ResidualPolicyState,
  speed_mps: float,
  steer_angle_deg: float,
  desired_curvature: float,
  stock_requested_torque: float,
  stock_applied_torque: float,
  dt_s: float = 0.01,
  vehicle_bearing_deg: float | None = None,
) -> np.ndarray:
  inputs = [speed_mps, steer_angle_deg, desired_curvature, stock_requested_torque, stock_applied_torque, dt_s]
  if vehicle_bearing_deg is not None:
    inputs.append(vehicle_bearing_deg)
  if not all(math.isfinite(value) for value in inputs) or dt_s <= 0.0:
    raise ValueError("residual policy observation inputs must be finite with positive dt_s")
  if state.initialized:
    prev_speed = state.speed_mps
    prev_steer = state.steer_angle_deg
    prev_curvature = state.desired_curvature
    prev_requested = state.stock_requested_torque
    prev_applied = state.stock_applied_torque
  else:
    prev_speed = speed_mps
    prev_steer = steer_angle_deg
    prev_curvature = desired_curvature
    prev_requested = stock_requested_torque
    prev_applied = stock_applied_torque
  torque_sign_mismatch = (
    abs(stock_requested_torque) > 1e-4
    and abs(stock_applied_torque) > 1e-4
    and stock_requested_torque * stock_applied_torque < 0.0
  )
  torque_error = stock_requested_torque - stock_applied_torque
  prev_torque_error = prev_requested - prev_applied
  abs_steer_ratio = abs(steer_angle_deg) / KONA_MAX_STEER_ANGLE_DEG
  if desired_curvature > DESIRED_CURVATURE_TURN_THRESHOLD:
    curvature_sign = 1.0
  elif desired_curvature < -DESIRED_CURVATURE_TURN_THRESHOLD:
    curvature_sign = -1.0
  else:
    curvature_sign = 0.0
  if curvature_sign == 0.0:
    state.model_desired_heading_integral_deg = 0.0
    state.model_curvature_sign_run_s = 0.0
    state.model_curvature_sign = 0.0
  else:
    if state.model_curvature_sign != curvature_sign:
      state.model_desired_heading_integral_deg = 0.0
      state.model_curvature_sign_run_s = 0.0
    state.model_desired_heading_integral_deg += math.degrees(desired_curvature * max(speed_mps, 0.0) * max(dt_s, 0.0))
    state.model_curvature_sign_run_s += max(dt_s, 0.0)
    state.model_curvature_sign = curvature_sign
  abs_model_desired_heading_integral_deg = abs(state.model_desired_heading_integral_deg)
  vehicle_heading_change_deg = 0.0
  if vehicle_bearing_deg is not None:
    if not state.vehicle_bearing_initialized:
      state.initial_vehicle_bearing_deg = float(vehicle_bearing_deg)
      state.vehicle_bearing_initialized = True
    vehicle_heading_change_deg = math.degrees(math.atan2(
      math.sin(math.radians(float(vehicle_bearing_deg) - state.initial_vehicle_bearing_deg)),
      math.cos(math.radians(float(vehicle_bearing_deg) - state.initial_vehicle_bearing_deg)),
    ))

  observation = np.asarray([
    speed_mps,
    steer_angle_deg,
    desired_curvature,
    stock_requested_torque,
    stock_applied_torque,
    prev_speed,
    prev_steer,
    prev_curvature,
    prev_requested,
    prev_applied,
    abs_steer_ratio,
    desired_curvature * speed_mps * speed_mps,
    abs(desired_curvature),
    torque_error,
    abs(stock_requested_torque),
    abs(stock_applied_torque),
    float(torque_sign_mismatch),
    speed_mps - prev_speed,
    steer_angle_deg - prev_steer,
    desired_curvature - prev_curvature,
    stock_requested_torque - prev_requested,
    stock_applied_torque - prev_applied,
    torque_error - prev_torque_error,
    abs(steer_angle_deg - prev_steer),
    abs(stock_applied_torque - prev_applied),
    max(0.0, (abs_steer_ratio - STEER_SATURATION_FRACTION) / (1.0 - STEER_SATURATION_FRACTION)),
    float(abs_steer_ratio >= STEER_SATURATION_FRACTION),
    state.model_desired_heading_integral_deg,
    abs_model_desired_heading_integral_deg,
    state.model_curvature_sign_run_s,
    max(0.0, min(1.0, (abs_model_desired_heading_integral_deg - 60.0) / 30.0)),
    state.model_desired_heading_integral_deg * curvature_sign,
    vehicle_heading_change_deg,
    abs(vehicle_heading_change_deg),
  ], dtype=np.float32)

  if not np.isfinite(observation).all():
    raise ValueError("residual policy observation must be finite")

  state.initialized = True
  state.speed_mps = float(speed_mps)
  state.steer_angle_deg = float(steer_angle_deg)
  state.desired_curvature = float(desired_curvature)
  state.stock_requested_torque = float(stock_requested_torque)
  state.stock_applied_torque = float(stock_applied_torque)
  return observation


class ResidualTorquePolicy:
  def __init__(
    self,
    feature_mu: np.ndarray | None = None,
    feature_sigma: np.ndarray | None = None,
    weight: np.ndarray | None = None,
    bias: float = 0.0,
    max_delta: float = DEFAULT_MAX_DELTA,
    path: str = "",
    enabled: bool = False,
    policy_type: str = "linear",
    mlp_weights: list[np.ndarray] | None = None,
    mlp_biases: list[np.ndarray] | None = None,
    policy_feature_names: list[str] | None = None,
    non_opposing_torque_guard: bool = False,
    guard_min_abs_desired_curvature: float = 0.0,
    guard_min_abs_steer_ratio: float = 0.0,
    guard_min_abs_stock_torque: float = 0.0,
    guard_max_opposing_delta: float = 0.0,
    residual_delta_rate_limit: float = 0.0,
    residual_activation_min_speed_mps: float = 0.0,
    residual_activation_max_speed_mps: float = 0.0,
    residual_activation_min_abs_desired_curvature: float = 0.0,
    residual_activation_min_abs_model_heading_deg: float = 0.0,
    residual_activation_min_abs_vehicle_heading_deg: float = 0.0,
    policy_output_max_delta: float | None = None,
    post_turn_policy_output_max_delta: float = 0.0,
    post_turn_policy_output_min_abs_vehicle_heading_deg: float = 0.0,
    post_turn_policy_output_max_abs_vehicle_heading_deg: float = 0.0,
    post_turn_policy_output_max_abs_desired_curvature: float = 0.0,
    turn_assist_applied_torque_floor: float = 0.0,
    turn_assist_max_speed_mps: float = 0.0,
    turn_assist_min_abs_model_heading_deg: float = 0.0,
    turn_assist_max_abs_model_heading_deg: float = 0.0,
    turn_assist_min_abs_vehicle_heading_deg: float = 0.0,
    turn_assist_max_abs_vehicle_heading_deg: float = 0.0,
    turn_assist_min_abs_desired_curvature: float = 0.0,
    turn_assist_max_abs_desired_curvature: float = 0.0,
    late_exit_suppress_max_abs_delta: float = 0.0,
    late_exit_suppress_max_speed_mps: float = 0.0,
    late_exit_suppress_min_abs_vehicle_heading_deg: float = 0.0,
    late_exit_suppress_fast_min_speed_mps: float = 0.0,
    late_exit_suppress_fast_min_abs_vehicle_heading_deg: float = 0.0,
    late_exit_suppress_min_abs_desired_curvature: float = 0.0,
    residual_saturation_fade_start_steer_ratio: float = 0.0,
    residual_saturation_fade_end_steer_ratio: float = 1.0,
    residual_saturation_fade_min_scale: float = 0.0,
    residual_saturation_fade_min_speed_mps: float = 0.0,
    residual_saturation_fade_max_speed_mps: float = 0.0,
    residual_saturation_fade_min_abs_stock_torque: float = 0.0,
    residual_saturation_fade_min_abs_desired_curvature: float = 0.0,
    residual_saturation_fade_min_abs_model_heading_deg: float = 0.0,
    residual_saturation_fade_min_abs_vehicle_heading_deg: float = 0.0,
    saturation_torque_limit: float = 0.0,
    saturation_torque_limit_min_speed_mps: float = 0.0,
    saturation_torque_limit_max_speed_mps: float = 0.0,
    saturation_torque_limit_min_steer_ratio: float = 1.0,
    saturation_torque_limit_min_abs_stock_torque: float = 0.0,
    saturation_torque_limit_min_abs_desired_curvature: float = 0.0,
    saturation_torque_limit_min_abs_model_heading_deg: float = 0.0,
    saturation_torque_limit_min_abs_vehicle_heading_deg: float = 0.0,
    applied_torque_sign_hold_frames: int = 0,
    applied_torque_sign_hold_min_abs_torque: float = 0.02,
    applied_torque_sign_hold_min_abs_vehicle_heading_deg: float = 0.0,
    applied_torque_sign_hold_max_abs_desired_curvature: float = 0.0,
  ):
    for name, value in locals().copy().items():
      if isinstance(value, (int, float, np.number)) and not math.isfinite(value):
        raise ValueError(f"residual policy {name} must be finite")
    self.policy_feature_names = list(policy_feature_names) if policy_feature_names is not None else list(OBSERVATION_FEATURE_NAMES)
    self.feature_indices = self._feature_indices(self.policy_feature_names)
    feature_dim = len(self.policy_feature_names)
    self.feature_mu = np.asarray(
      feature_mu if feature_mu is not None else np.zeros(feature_dim),
      dtype=np.float32,
    )
    self.feature_sigma = np.asarray(
      feature_sigma if feature_sigma is not None else np.ones(feature_dim),
      dtype=np.float32,
    )
    self.weight = np.asarray(
      weight if weight is not None else np.zeros(feature_dim),
      dtype=np.float32,
    ).reshape(-1)
    self.bias = float(bias)
    self.max_delta = abs(float(max_delta))
    self.path = str(path)
    self.enabled = bool(enabled)
    self.policy_type = str(policy_type)
    self.mlp_weights = [np.asarray(weight, dtype=np.float32) for weight in (mlp_weights or [])]
    self.mlp_biases = [np.asarray(bias, dtype=np.float32).reshape(-1) for bias in (mlp_biases or [])]
    self.non_opposing_torque_guard = bool(non_opposing_torque_guard)
    self.guard_min_abs_desired_curvature = max(0.0, float(guard_min_abs_desired_curvature))
    self.guard_min_abs_steer_ratio = max(0.0, float(guard_min_abs_steer_ratio))
    self.guard_min_abs_stock_torque = max(0.0, float(guard_min_abs_stock_torque))
    self.guard_max_opposing_delta = max(0.0, float(guard_max_opposing_delta))
    self.residual_delta_rate_limit = max(0.0, float(residual_delta_rate_limit))
    self.residual_activation_min_speed_mps = max(0.0, float(residual_activation_min_speed_mps))
    self.residual_activation_max_speed_mps = max(0.0, float(residual_activation_max_speed_mps))
    self.residual_activation_min_abs_desired_curvature = max(0.0, float(residual_activation_min_abs_desired_curvature))
    self.residual_activation_min_abs_model_heading_deg = max(0.0, float(residual_activation_min_abs_model_heading_deg))
    self.residual_activation_min_abs_vehicle_heading_deg = max(0.0, float(residual_activation_min_abs_vehicle_heading_deg))
    self.policy_output_max_delta = self.max_delta if policy_output_max_delta is None else abs(float(policy_output_max_delta))
    self.policy_output_max_delta = min(self.policy_output_max_delta, self.max_delta)
    self.post_turn_policy_output_max_delta = min(abs(float(post_turn_policy_output_max_delta)), self.max_delta)
    self.post_turn_policy_output_min_abs_vehicle_heading_deg = max(0.0, float(post_turn_policy_output_min_abs_vehicle_heading_deg))
    self.post_turn_policy_output_max_abs_vehicle_heading_deg = max(0.0, float(post_turn_policy_output_max_abs_vehicle_heading_deg))
    self.post_turn_policy_output_max_abs_desired_curvature = max(0.0, float(post_turn_policy_output_max_abs_desired_curvature))
    self.turn_assist_applied_torque_floor = max(0.0, min(1.0, float(turn_assist_applied_torque_floor)))
    self.turn_assist_max_speed_mps = max(0.0, float(turn_assist_max_speed_mps))
    self.turn_assist_min_abs_model_heading_deg = max(0.0, float(turn_assist_min_abs_model_heading_deg))
    self.turn_assist_max_abs_model_heading_deg = max(0.0, float(turn_assist_max_abs_model_heading_deg))
    self.turn_assist_min_abs_vehicle_heading_deg = max(0.0, float(turn_assist_min_abs_vehicle_heading_deg))
    self.turn_assist_max_abs_vehicle_heading_deg = max(0.0, float(turn_assist_max_abs_vehicle_heading_deg))
    self.turn_assist_min_abs_desired_curvature = max(0.0, float(turn_assist_min_abs_desired_curvature))
    self.turn_assist_max_abs_desired_curvature = max(0.0, float(turn_assist_max_abs_desired_curvature))
    self.late_exit_suppress_max_abs_delta = max(0.0, float(late_exit_suppress_max_abs_delta))
    self.late_exit_suppress_max_speed_mps = max(0.0, float(late_exit_suppress_max_speed_mps))
    self.late_exit_suppress_min_abs_vehicle_heading_deg = max(0.0, float(late_exit_suppress_min_abs_vehicle_heading_deg))
    self.late_exit_suppress_fast_min_speed_mps = max(0.0, float(late_exit_suppress_fast_min_speed_mps))
    self.late_exit_suppress_fast_min_abs_vehicle_heading_deg = max(0.0, float(late_exit_suppress_fast_min_abs_vehicle_heading_deg))
    self.late_exit_suppress_min_abs_desired_curvature = max(0.0, float(late_exit_suppress_min_abs_desired_curvature))
    self.residual_saturation_fade_start_steer_ratio = max(0.0, float(residual_saturation_fade_start_steer_ratio))
    self.residual_saturation_fade_end_steer_ratio = max(self.residual_saturation_fade_start_steer_ratio, float(residual_saturation_fade_end_steer_ratio))
    self.residual_saturation_fade_min_scale = max(0.0, min(1.0, float(residual_saturation_fade_min_scale)))
    self.residual_saturation_fade_min_speed_mps = max(0.0, float(residual_saturation_fade_min_speed_mps))
    self.residual_saturation_fade_max_speed_mps = max(0.0, float(residual_saturation_fade_max_speed_mps))
    self.residual_saturation_fade_min_abs_stock_torque = max(0.0, float(residual_saturation_fade_min_abs_stock_torque))
    self.residual_saturation_fade_min_abs_desired_curvature = max(0.0, float(residual_saturation_fade_min_abs_desired_curvature))
    self.residual_saturation_fade_min_abs_model_heading_deg = max(0.0, float(residual_saturation_fade_min_abs_model_heading_deg))
    self.residual_saturation_fade_min_abs_vehicle_heading_deg = max(0.0, float(residual_saturation_fade_min_abs_vehicle_heading_deg))
    self.saturation_torque_limit = max(0.0, min(1.0, float(saturation_torque_limit)))
    self.saturation_torque_limit_min_speed_mps = max(0.0, float(saturation_torque_limit_min_speed_mps))
    self.saturation_torque_limit_max_speed_mps = max(0.0, float(saturation_torque_limit_max_speed_mps))
    self.saturation_torque_limit_min_steer_ratio = max(0.0, float(saturation_torque_limit_min_steer_ratio))
    self.saturation_torque_limit_min_abs_stock_torque = max(0.0, float(saturation_torque_limit_min_abs_stock_torque))
    self.saturation_torque_limit_min_abs_desired_curvature = max(0.0, float(saturation_torque_limit_min_abs_desired_curvature))
    self.saturation_torque_limit_min_abs_model_heading_deg = max(0.0, float(saturation_torque_limit_min_abs_model_heading_deg))
    self.saturation_torque_limit_min_abs_vehicle_heading_deg = max(0.0, float(saturation_torque_limit_min_abs_vehicle_heading_deg))
    self.applied_torque_sign_hold_frames = max(0, int(applied_torque_sign_hold_frames))
    self.applied_torque_sign_hold_min_abs_torque = max(0.0, float(applied_torque_sign_hold_min_abs_torque))
    self.applied_torque_sign_hold_min_abs_vehicle_heading_deg = max(0.0, float(applied_torque_sign_hold_min_abs_vehicle_heading_deg))
    self.applied_torque_sign_hold_max_abs_desired_curvature = max(0.0, float(applied_torque_sign_hold_max_abs_desired_curvature))

    if self.feature_mu.shape != (feature_dim,):
      raise ValueError("residual policy feature_mu has wrong shape")
    if self.feature_sigma.shape != (feature_dim,):
      raise ValueError("residual policy feature_sigma has wrong shape")
    if self.policy_type == "linear":
      if self.weight.shape != (feature_dim,):
        raise ValueError("residual policy weight has wrong shape")
    elif self.policy_type == "mlp_tanh":
      self._validate_mlp()
    else:
      raise ValueError(f"unsupported residual policy type: {self.policy_type}")
    arrays = [self.feature_mu, self.feature_sigma, self.weight, *self.mlp_weights, *self.mlp_biases]
    if not all(np.isfinite(array).all() for array in arrays):
      raise ValueError("residual policy arrays must be finite")
    self.feature_sigma = np.where(np.abs(self.feature_sigma) < 1e-6, 1.0, self.feature_sigma).astype(np.float32)

  @staticmethod
  def _feature_indices(policy_feature_names: list[str]) -> list[int]:
    if not policy_feature_names or len(set(policy_feature_names)) != len(policy_feature_names):
      raise ValueError("residual policy feature names must be nonempty and unique")
    indices = []
    for name in policy_feature_names:
      if name not in OBSERVATION_FEATURE_NAMES:
        raise ValueError(f"residual policy feature is not available at runtime: {name}")
      indices.append(OBSERVATION_FEATURE_NAMES.index(name))
    return indices

  def _validate_mlp(self) -> None:
    if not self.mlp_weights or len(self.mlp_weights) != len(self.mlp_biases):
      raise ValueError("mlp residual policy requires matched weights and biases")
    in_dim = len(self.policy_feature_names)
    for idx, (weight, bias) in enumerate(zip(self.mlp_weights, self.mlp_biases, strict=True)):
      if weight.ndim != 2:
        raise ValueError(f"mlp residual policy weight {idx} must be 2D")
      if weight.shape[0] != in_dim:
        raise ValueError(f"mlp residual policy weight {idx} input dimension mismatch")
      if bias.shape != (weight.shape[1],):
        raise ValueError(f"mlp residual policy bias {idx} shape mismatch")
      in_dim = weight.shape[1]
    if in_dim != 1:
      raise ValueError("mlp residual policy output dimension must be 1")

  @classmethod
  def disabled(cls) -> "ResidualTorquePolicy":
    return cls(enabled=False)

  @classmethod
  def zero(cls, max_delta: float = DEFAULT_MAX_DELTA) -> "ResidualTorquePolicy":
    return cls(max_delta=max_delta, path="zero", enabled=True)

  @classmethod
  def from_path(cls, path: str | None, max_delta: float = DEFAULT_MAX_DELTA) -> "ResidualTorquePolicy":
    if not math.isfinite(max_delta):
      raise ValueError("residual policy maximum delta must be finite")
    if path is None or str(path).strip() == "":
      return cls.disabled()
    if str(path) == "zero":
      return cls.zero(max_delta)

    policy_path = Path(path)
    with np.load(policy_path, allow_pickle=False) as policy:
      for name in policy.files:
        value = np.asarray(policy[name])
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
          raise ValueError(f"residual policy artifact {name} must be finite")
      if "observation_feature_names" in policy:
        policy_feature_names = [str(name) for name in np.asarray(policy["observation_feature_names"]).reshape(-1)]
      elif "feature_mu" in policy:
        policy_feature_names = list(OBSERVATION_FEATURE_NAMES[:len(np.asarray(policy["feature_mu"]).reshape(-1))])
      else:
        policy_feature_names = list(OBSERVATION_FEATURE_NAMES)
      feature_dim = len(policy_feature_names)
      feature_mu = policy["feature_mu"] if "feature_mu" in policy else np.zeros(feature_dim)
      feature_sigma = policy["feature_sigma"] if "feature_sigma" in policy else np.ones(feature_dim)
      policy_type = str(np.asarray(policy["policy_type"]).reshape(-1)[0]) if "policy_type" in policy else "linear"
      weight = policy["weight"] if "weight" in policy else np.zeros(feature_dim, dtype=np.float32)
      bias = float(np.asarray(policy["bias"]).reshape(-1)[0]) if "bias" in policy else 0.0
      policy_max_delta = float(np.asarray(policy["max_delta"]).reshape(-1)[0]) if "max_delta" in policy else max_delta
      mlp_weights = None
      mlp_biases = None
      if policy_type == "mlp_tanh":
        layer_count = int(np.asarray(policy["mlp_layer_count"]).reshape(-1)[0])
        mlp_weights = [policy[f"mlp_weight_{idx}"] for idx in range(layer_count)]
        mlp_biases = [policy[f"mlp_bias_{idx}"] for idx in range(layer_count)]
      non_opposing_torque_guard = bool(np.asarray(policy["non_opposing_torque_guard"]).reshape(-1)[0]) if "non_opposing_torque_guard" in policy else False
      guard_min_abs_desired_curvature = float(np.asarray(policy["guard_min_abs_desired_curvature"]).reshape(-1)[0]) if "guard_min_abs_desired_curvature" in policy else 0.0
      guard_min_abs_steer_ratio = float(np.asarray(policy["guard_min_abs_steer_ratio"]).reshape(-1)[0]) if "guard_min_abs_steer_ratio" in policy else 0.0
      guard_min_abs_stock_torque = float(np.asarray(policy["guard_min_abs_stock_torque"]).reshape(-1)[0]) if "guard_min_abs_stock_torque" in policy else 0.0
      guard_max_opposing_delta = float(np.asarray(policy["guard_max_opposing_delta"]).reshape(-1)[0]) if "guard_max_opposing_delta" in policy else 0.0
      residual_delta_rate_limit = float(np.asarray(policy["residual_delta_rate_limit"]).reshape(-1)[0]) if "residual_delta_rate_limit" in policy else 0.0
      residual_activation_min_speed_mps = float(np.asarray(policy["residual_activation_min_speed_mps"]).reshape(-1)[0]) if "residual_activation_min_speed_mps" in policy else 0.0
      residual_activation_max_speed_mps = float(np.asarray(policy["residual_activation_max_speed_mps"]).reshape(-1)[0]) if "residual_activation_max_speed_mps" in policy else 0.0
      residual_activation_min_abs_desired_curvature = float(np.asarray(policy["residual_activation_min_abs_desired_curvature"]).reshape(-1)[0]) if "residual_activation_min_abs_desired_curvature" in policy else 0.0
      residual_activation_min_abs_model_heading_deg = float(np.asarray(policy["residual_activation_min_abs_model_heading_deg"]).reshape(-1)[0]) if "residual_activation_min_abs_model_heading_deg" in policy else 0.0
      residual_activation_min_abs_vehicle_heading_deg = float(np.asarray(policy["residual_activation_min_abs_vehicle_heading_deg"]).reshape(-1)[0]) if "residual_activation_min_abs_vehicle_heading_deg" in policy else 0.0
      policy_output_max_delta = float(np.asarray(policy["policy_output_max_delta"]).reshape(-1)[0]) if "policy_output_max_delta" in policy else policy_max_delta
      post_turn_policy_output_max_delta = float(np.asarray(policy["post_turn_policy_output_max_delta"]).reshape(-1)[0]) if "post_turn_policy_output_max_delta" in policy else 0.0
      post_turn_policy_output_min_abs_vehicle_heading_deg = float(np.asarray(policy["post_turn_policy_output_min_abs_vehicle_heading_deg"]).reshape(-1)[0]) if "post_turn_policy_output_min_abs_vehicle_heading_deg" in policy else 0.0
      post_turn_policy_output_max_abs_vehicle_heading_deg = float(np.asarray(policy["post_turn_policy_output_max_abs_vehicle_heading_deg"]).reshape(-1)[0]) if "post_turn_policy_output_max_abs_vehicle_heading_deg" in policy else 0.0
      post_turn_policy_output_max_abs_desired_curvature = float(np.asarray(policy["post_turn_policy_output_max_abs_desired_curvature"]).reshape(-1)[0]) if "post_turn_policy_output_max_abs_desired_curvature" in policy else 0.0
      turn_assist_applied_torque_floor = float(np.asarray(policy["turn_assist_applied_torque_floor"]).reshape(-1)[0]) if "turn_assist_applied_torque_floor" in policy else 0.0
      turn_assist_max_speed_mps = float(np.asarray(policy["turn_assist_max_speed_mps"]).reshape(-1)[0]) if "turn_assist_max_speed_mps" in policy else 0.0
      turn_assist_min_abs_model_heading_deg = float(np.asarray(policy["turn_assist_min_abs_model_heading_deg"]).reshape(-1)[0]) if "turn_assist_min_abs_model_heading_deg" in policy else 0.0
      turn_assist_max_abs_model_heading_deg = float(np.asarray(policy["turn_assist_max_abs_model_heading_deg"]).reshape(-1)[0]) if "turn_assist_max_abs_model_heading_deg" in policy else 0.0
      turn_assist_min_abs_vehicle_heading_deg = float(np.asarray(policy["turn_assist_min_abs_vehicle_heading_deg"]).reshape(-1)[0]) if "turn_assist_min_abs_vehicle_heading_deg" in policy else 0.0
      turn_assist_max_abs_vehicle_heading_deg = float(np.asarray(policy["turn_assist_max_abs_vehicle_heading_deg"]).reshape(-1)[0]) if "turn_assist_max_abs_vehicle_heading_deg" in policy else 0.0
      turn_assist_min_abs_desired_curvature = float(np.asarray(policy["turn_assist_min_abs_desired_curvature"]).reshape(-1)[0]) if "turn_assist_min_abs_desired_curvature" in policy else 0.0
      turn_assist_max_abs_desired_curvature = float(np.asarray(policy["turn_assist_max_abs_desired_curvature"]).reshape(-1)[0]) if "turn_assist_max_abs_desired_curvature" in policy else 0.0
      late_exit_suppress_max_abs_delta = float(np.asarray(policy["late_exit_suppress_max_abs_delta"]).reshape(-1)[0]) if "late_exit_suppress_max_abs_delta" in policy else 0.0
      late_exit_suppress_max_speed_mps = float(np.asarray(policy["late_exit_suppress_max_speed_mps"]).reshape(-1)[0]) if "late_exit_suppress_max_speed_mps" in policy else 0.0
      late_exit_suppress_min_abs_vehicle_heading_deg = float(np.asarray(policy["late_exit_suppress_min_abs_vehicle_heading_deg"]).reshape(-1)[0]) if "late_exit_suppress_min_abs_vehicle_heading_deg" in policy else 0.0
      late_exit_suppress_fast_min_speed_mps = float(np.asarray(policy["late_exit_suppress_fast_min_speed_mps"]).reshape(-1)[0]) if "late_exit_suppress_fast_min_speed_mps" in policy else 0.0
      late_exit_suppress_fast_min_abs_vehicle_heading_deg = float(np.asarray(policy["late_exit_suppress_fast_min_abs_vehicle_heading_deg"]).reshape(-1)[0]) if "late_exit_suppress_fast_min_abs_vehicle_heading_deg" in policy else 0.0
      late_exit_suppress_min_abs_desired_curvature = float(np.asarray(policy["late_exit_suppress_min_abs_desired_curvature"]).reshape(-1)[0]) if "late_exit_suppress_min_abs_desired_curvature" in policy else 0.0
      residual_saturation_fade_start_steer_ratio = float(np.asarray(policy["residual_saturation_fade_start_steer_ratio"]).reshape(-1)[0]) if "residual_saturation_fade_start_steer_ratio" in policy else 0.0
      residual_saturation_fade_end_steer_ratio = float(np.asarray(policy["residual_saturation_fade_end_steer_ratio"]).reshape(-1)[0]) if "residual_saturation_fade_end_steer_ratio" in policy else 1.0
      residual_saturation_fade_min_scale = float(np.asarray(policy["residual_saturation_fade_min_scale"]).reshape(-1)[0]) if "residual_saturation_fade_min_scale" in policy else 0.0
      residual_saturation_fade_min_speed_mps = float(np.asarray(policy["residual_saturation_fade_min_speed_mps"]).reshape(-1)[0]) if "residual_saturation_fade_min_speed_mps" in policy else 0.0
      residual_saturation_fade_max_speed_mps = float(np.asarray(policy["residual_saturation_fade_max_speed_mps"]).reshape(-1)[0]) if "residual_saturation_fade_max_speed_mps" in policy else 0.0
      residual_saturation_fade_min_abs_stock_torque = float(np.asarray(policy["residual_saturation_fade_min_abs_stock_torque"]).reshape(-1)[0]) if "residual_saturation_fade_min_abs_stock_torque" in policy else 0.0
      residual_saturation_fade_min_abs_desired_curvature = float(np.asarray(policy["residual_saturation_fade_min_abs_desired_curvature"]).reshape(-1)[0]) if "residual_saturation_fade_min_abs_desired_curvature" in policy else 0.0
      residual_saturation_fade_min_abs_model_heading_deg = float(np.asarray(policy["residual_saturation_fade_min_abs_model_heading_deg"]).reshape(-1)[0]) if "residual_saturation_fade_min_abs_model_heading_deg" in policy else 0.0
      residual_saturation_fade_min_abs_vehicle_heading_deg = float(np.asarray(policy["residual_saturation_fade_min_abs_vehicle_heading_deg"]).reshape(-1)[0]) if "residual_saturation_fade_min_abs_vehicle_heading_deg" in policy else 0.0
      saturation_torque_limit = float(np.asarray(policy["saturation_torque_limit"]).reshape(-1)[0]) if "saturation_torque_limit" in policy else 0.0
      saturation_torque_limit_min_speed_mps = float(np.asarray(policy["saturation_torque_limit_min_speed_mps"]).reshape(-1)[0]) if "saturation_torque_limit_min_speed_mps" in policy else 0.0
      saturation_torque_limit_max_speed_mps = float(np.asarray(policy["saturation_torque_limit_max_speed_mps"]).reshape(-1)[0]) if "saturation_torque_limit_max_speed_mps" in policy else 0.0
      saturation_torque_limit_min_steer_ratio = float(np.asarray(policy["saturation_torque_limit_min_steer_ratio"]).reshape(-1)[0]) if "saturation_torque_limit_min_steer_ratio" in policy else 1.0
      saturation_torque_limit_min_abs_stock_torque = float(np.asarray(policy["saturation_torque_limit_min_abs_stock_torque"]).reshape(-1)[0]) if "saturation_torque_limit_min_abs_stock_torque" in policy else 0.0
      saturation_torque_limit_min_abs_desired_curvature = float(np.asarray(policy["saturation_torque_limit_min_abs_desired_curvature"]).reshape(-1)[0]) if "saturation_torque_limit_min_abs_desired_curvature" in policy else 0.0
      saturation_torque_limit_min_abs_model_heading_deg = float(np.asarray(policy["saturation_torque_limit_min_abs_model_heading_deg"]).reshape(-1)[0]) if "saturation_torque_limit_min_abs_model_heading_deg" in policy else 0.0
      saturation_torque_limit_min_abs_vehicle_heading_deg = float(np.asarray(policy["saturation_torque_limit_min_abs_vehicle_heading_deg"]).reshape(-1)[0]) if "saturation_torque_limit_min_abs_vehicle_heading_deg" in policy else 0.0
      applied_torque_sign_hold_frames = int(np.asarray(policy["applied_torque_sign_hold_frames"]).reshape(-1)[0]) if "applied_torque_sign_hold_frames" in policy else 0
      applied_torque_sign_hold_min_abs_torque = float(np.asarray(policy["applied_torque_sign_hold_min_abs_torque"]).reshape(-1)[0]) if "applied_torque_sign_hold_min_abs_torque" in policy else 0.02
      applied_torque_sign_hold_min_abs_vehicle_heading_deg = float(np.asarray(policy["applied_torque_sign_hold_min_abs_vehicle_heading_deg"]).reshape(-1)[0]) if "applied_torque_sign_hold_min_abs_vehicle_heading_deg" in policy else 0.0
      applied_torque_sign_hold_max_abs_desired_curvature = float(np.asarray(policy["applied_torque_sign_hold_max_abs_desired_curvature"]).reshape(-1)[0]) if "applied_torque_sign_hold_max_abs_desired_curvature" in policy else 0.0
    if max_delta > 0.0:
      policy_max_delta = min(abs(policy_max_delta), abs(float(max_delta)))
      policy_output_max_delta = min(abs(policy_output_max_delta), policy_max_delta)
      post_turn_policy_output_max_delta = min(abs(post_turn_policy_output_max_delta), policy_max_delta)
    return cls(
      feature_mu,
      feature_sigma,
      weight,
      bias,
      policy_max_delta,
      str(policy_path),
      enabled=True,
      policy_type=policy_type,
      mlp_weights=mlp_weights,
      mlp_biases=mlp_biases,
      policy_feature_names=policy_feature_names,
      non_opposing_torque_guard=non_opposing_torque_guard,
      guard_min_abs_desired_curvature=guard_min_abs_desired_curvature,
      guard_min_abs_steer_ratio=guard_min_abs_steer_ratio,
      guard_min_abs_stock_torque=guard_min_abs_stock_torque,
      guard_max_opposing_delta=guard_max_opposing_delta,
      residual_delta_rate_limit=residual_delta_rate_limit,
      residual_activation_min_speed_mps=residual_activation_min_speed_mps,
      residual_activation_max_speed_mps=residual_activation_max_speed_mps,
      residual_activation_min_abs_desired_curvature=residual_activation_min_abs_desired_curvature,
      residual_activation_min_abs_model_heading_deg=residual_activation_min_abs_model_heading_deg,
      residual_activation_min_abs_vehicle_heading_deg=residual_activation_min_abs_vehicle_heading_deg,
      policy_output_max_delta=policy_output_max_delta,
      post_turn_policy_output_max_delta=post_turn_policy_output_max_delta,
      post_turn_policy_output_min_abs_vehicle_heading_deg=post_turn_policy_output_min_abs_vehicle_heading_deg,
      post_turn_policy_output_max_abs_vehicle_heading_deg=post_turn_policy_output_max_abs_vehicle_heading_deg,
      post_turn_policy_output_max_abs_desired_curvature=post_turn_policy_output_max_abs_desired_curvature,
      turn_assist_applied_torque_floor=turn_assist_applied_torque_floor,
      turn_assist_max_speed_mps=turn_assist_max_speed_mps,
      turn_assist_min_abs_model_heading_deg=turn_assist_min_abs_model_heading_deg,
      turn_assist_max_abs_model_heading_deg=turn_assist_max_abs_model_heading_deg,
      turn_assist_min_abs_vehicle_heading_deg=turn_assist_min_abs_vehicle_heading_deg,
      turn_assist_max_abs_vehicle_heading_deg=turn_assist_max_abs_vehicle_heading_deg,
      turn_assist_min_abs_desired_curvature=turn_assist_min_abs_desired_curvature,
      turn_assist_max_abs_desired_curvature=turn_assist_max_abs_desired_curvature,
      late_exit_suppress_max_abs_delta=late_exit_suppress_max_abs_delta,
      late_exit_suppress_max_speed_mps=late_exit_suppress_max_speed_mps,
      late_exit_suppress_min_abs_vehicle_heading_deg=late_exit_suppress_min_abs_vehicle_heading_deg,
      late_exit_suppress_fast_min_speed_mps=late_exit_suppress_fast_min_speed_mps,
      late_exit_suppress_fast_min_abs_vehicle_heading_deg=late_exit_suppress_fast_min_abs_vehicle_heading_deg,
      late_exit_suppress_min_abs_desired_curvature=late_exit_suppress_min_abs_desired_curvature,
      residual_saturation_fade_start_steer_ratio=residual_saturation_fade_start_steer_ratio,
      residual_saturation_fade_end_steer_ratio=residual_saturation_fade_end_steer_ratio,
      residual_saturation_fade_min_scale=residual_saturation_fade_min_scale,
      residual_saturation_fade_min_speed_mps=residual_saturation_fade_min_speed_mps,
      residual_saturation_fade_max_speed_mps=residual_saturation_fade_max_speed_mps,
      residual_saturation_fade_min_abs_stock_torque=residual_saturation_fade_min_abs_stock_torque,
      residual_saturation_fade_min_abs_desired_curvature=residual_saturation_fade_min_abs_desired_curvature,
      residual_saturation_fade_min_abs_model_heading_deg=residual_saturation_fade_min_abs_model_heading_deg,
      residual_saturation_fade_min_abs_vehicle_heading_deg=residual_saturation_fade_min_abs_vehicle_heading_deg,
      saturation_torque_limit=saturation_torque_limit,
      saturation_torque_limit_min_speed_mps=saturation_torque_limit_min_speed_mps,
      saturation_torque_limit_max_speed_mps=saturation_torque_limit_max_speed_mps,
      saturation_torque_limit_min_steer_ratio=saturation_torque_limit_min_steer_ratio,
      saturation_torque_limit_min_abs_stock_torque=saturation_torque_limit_min_abs_stock_torque,
      saturation_torque_limit_min_abs_desired_curvature=saturation_torque_limit_min_abs_desired_curvature,
      saturation_torque_limit_min_abs_model_heading_deg=saturation_torque_limit_min_abs_model_heading_deg,
      saturation_torque_limit_min_abs_vehicle_heading_deg=saturation_torque_limit_min_abs_vehicle_heading_deg,
      applied_torque_sign_hold_frames=applied_torque_sign_hold_frames,
      applied_torque_sign_hold_min_abs_torque=applied_torque_sign_hold_min_abs_torque,
      applied_torque_sign_hold_min_abs_vehicle_heading_deg=applied_torque_sign_hold_min_abs_vehicle_heading_deg,
      applied_torque_sign_hold_max_abs_desired_curvature=applied_torque_sign_hold_max_abs_desired_curvature,
    )

  def _residual_activation_allowed(self, observation: np.ndarray | None) -> bool:
    if observation is None or observation.shape[0] != len(OBSERVATION_FEATURE_NAMES):
      return True

    speed = float(observation[OBSERVATION_FEATURE_NAMES.index("speed_mps")])
    if self.residual_activation_min_speed_mps > 0.0 and speed < self.residual_activation_min_speed_mps:
      return False
    if self.residual_activation_max_speed_mps > 0.0 and speed > self.residual_activation_max_speed_mps:
      return False

    configured = False
    allowed = False
    if self.residual_activation_min_abs_desired_curvature > 0.0:
      configured = True
      desired_curvature = float(observation[OBSERVATION_FEATURE_NAMES.index("desired_curvature")])
      allowed = allowed or abs(desired_curvature) >= self.residual_activation_min_abs_desired_curvature
    if self.residual_activation_min_abs_model_heading_deg > 0.0:
      configured = True
      model_heading = float(observation[OBSERVATION_FEATURE_NAMES.index("model_desired_heading_integral_deg")])
      allowed = allowed or abs(model_heading) >= self.residual_activation_min_abs_model_heading_deg
    if self.residual_activation_min_abs_vehicle_heading_deg > 0.0:
      configured = True
      vehicle_heading = float(observation[OBSERVATION_FEATURE_NAMES.index("vehicle_heading_change_deg")])
      allowed = allowed or abs(vehicle_heading) >= self.residual_activation_min_abs_vehicle_heading_deg
    return True if not configured else allowed

  def _active_policy_output_max_delta(self, observation: np.ndarray | None) -> float:
    if (
      observation is None
      or observation.shape[0] != len(OBSERVATION_FEATURE_NAMES)
      or self.post_turn_policy_output_max_delta <= self.policy_output_max_delta
    ):
      return self.policy_output_max_delta
    vehicle_heading = float(observation[OBSERVATION_FEATURE_NAMES.index("vehicle_heading_change_deg")])
    abs_vehicle_heading = abs(vehicle_heading)
    if abs_vehicle_heading < self.post_turn_policy_output_min_abs_vehicle_heading_deg:
      return self.policy_output_max_delta
    if self.post_turn_policy_output_max_abs_vehicle_heading_deg > 0.0 and abs_vehicle_heading > self.post_turn_policy_output_max_abs_vehicle_heading_deg:
      return self.policy_output_max_delta
    if self.post_turn_policy_output_max_abs_desired_curvature > 0.0:
      desired_curvature = float(observation[OBSERVATION_FEATURE_NAMES.index("desired_curvature")])
      if abs(desired_curvature) > self.post_turn_policy_output_max_abs_desired_curvature:
        return self.policy_output_max_delta
    return self.post_turn_policy_output_max_delta

  def _turn_assisted_delta(self, delta: float, observation: np.ndarray) -> float:
    if self.turn_assist_applied_torque_floor <= 0.0 or observation.shape[0] != len(OBSERVATION_FEATURE_NAMES):
      return delta
    speed = float(observation[OBSERVATION_FEATURE_NAMES.index("speed_mps")])
    if self.turn_assist_max_speed_mps > 0.0 and speed > self.turn_assist_max_speed_mps:
      return delta
    desired_curvature = float(observation[OBSERVATION_FEATURE_NAMES.index("desired_curvature")])
    abs_desired_curvature = abs(desired_curvature)
    if abs_desired_curvature < self.turn_assist_min_abs_desired_curvature:
      return delta
    if self.turn_assist_max_abs_desired_curvature > 0.0 and abs_desired_curvature > self.turn_assist_max_abs_desired_curvature:
      return delta
    vehicle_heading = abs(float(observation[OBSERVATION_FEATURE_NAMES.index("vehicle_heading_change_deg")]))
    model_heading = abs(float(observation[OBSERVATION_FEATURE_NAMES.index("model_desired_heading_integral_deg")]))
    model_heading_gate = (
      self.turn_assist_min_abs_model_heading_deg > 0.0
      and model_heading >= self.turn_assist_min_abs_model_heading_deg
    )
    if self.turn_assist_max_abs_model_heading_deg > 0.0 and model_heading > self.turn_assist_max_abs_model_heading_deg:
      model_heading_gate = False
    if vehicle_heading < self.turn_assist_min_abs_vehicle_heading_deg and not model_heading_gate:
      return delta
    if self.turn_assist_max_abs_vehicle_heading_deg > 0.0 and vehicle_heading > self.turn_assist_max_abs_vehicle_heading_deg:
      return delta

    stock_applied = float(observation[OBSERVATION_FEATURE_NAMES.index("stock_applied_torque")])
    turn_torque_sign = -math.copysign(1.0, desired_curvature)
    target_applied = turn_torque_sign * self.turn_assist_applied_torque_floor
    if abs(stock_applied) >= self.turn_assist_applied_torque_floor and stock_applied * target_applied > 0.0:
      return delta

    floor_delta = target_applied - stock_applied
    if abs(floor_delta) <= abs(delta):
      return delta
    return float(np.clip(floor_delta, -self.max_delta, self.max_delta))

  def _late_exit_suppressed_delta(self, delta: float, observation: np.ndarray) -> float:
    if self.late_exit_suppress_max_abs_delta <= 0.0 or observation.shape[0] != len(OBSERVATION_FEATURE_NAMES):
      return delta
    speed = float(observation[OBSERVATION_FEATURE_NAMES.index("speed_mps")])
    if self.late_exit_suppress_max_speed_mps > 0.0 and speed > self.late_exit_suppress_max_speed_mps:
      return delta
    desired_curvature = float(observation[OBSERVATION_FEATURE_NAMES.index("desired_curvature")])
    if abs(desired_curvature) < self.late_exit_suppress_min_abs_desired_curvature:
      return delta
    vehicle_heading = float(observation[OBSERVATION_FEATURE_NAMES.index("vehicle_heading_change_deg")])
    min_vehicle_heading = self.late_exit_suppress_min_abs_vehicle_heading_deg
    if (
      self.late_exit_suppress_fast_min_abs_vehicle_heading_deg > 0.0
      and speed >= self.late_exit_suppress_fast_min_speed_mps
    ):
      min_vehicle_heading = self.late_exit_suppress_fast_min_abs_vehicle_heading_deg
    if abs(vehicle_heading) < min_vehicle_heading:
      return delta
    if desired_curvature * vehicle_heading <= 0.0:
      return delta
    return float(np.clip(delta, -self.late_exit_suppress_max_abs_delta, self.late_exit_suppress_max_abs_delta))

  def _saturation_faded_delta(self, delta: float, observation: np.ndarray) -> float:
    if self.residual_saturation_fade_start_steer_ratio <= 0.0 or observation.shape[0] != len(OBSERVATION_FEATURE_NAMES):
      return delta
    speed = float(observation[OBSERVATION_FEATURE_NAMES.index("speed_mps")])
    if self.residual_saturation_fade_min_speed_mps > 0.0 and speed < self.residual_saturation_fade_min_speed_mps:
      return delta
    if self.residual_saturation_fade_max_speed_mps > 0.0 and speed > self.residual_saturation_fade_max_speed_mps:
      return delta
    abs_steer_ratio = float(observation[OBSERVATION_FEATURE_NAMES.index("abs_steer_angle_ratio")])
    if abs_steer_ratio < self.residual_saturation_fade_start_steer_ratio:
      return delta
    desired_curvature = float(observation[OBSERVATION_FEATURE_NAMES.index("desired_curvature")])
    if abs(desired_curvature) < self.residual_saturation_fade_min_abs_desired_curvature:
      return delta
    model_heading = float(observation[OBSERVATION_FEATURE_NAMES.index("model_desired_heading_integral_deg")])
    if abs(model_heading) < self.residual_saturation_fade_min_abs_model_heading_deg:
      return delta
    vehicle_heading = float(observation[OBSERVATION_FEATURE_NAMES.index("vehicle_heading_change_deg")])
    if abs(vehicle_heading) < self.residual_saturation_fade_min_abs_vehicle_heading_deg:
      return delta
    stock_applied = float(observation[OBSERVATION_FEATURE_NAMES.index("stock_applied_torque")])
    stock_requested = float(observation[OBSERVATION_FEATURE_NAMES.index("stock_requested_torque")])
    stock_reference = stock_applied if abs(stock_applied) >= self.residual_saturation_fade_min_abs_stock_torque else stock_requested
    if abs(stock_reference) < self.residual_saturation_fade_min_abs_stock_torque:
      return delta

    fade_range = self.residual_saturation_fade_end_steer_ratio - self.residual_saturation_fade_start_steer_ratio
    pressure = 1.0 if fade_range <= 1e-6 else (abs_steer_ratio - self.residual_saturation_fade_start_steer_ratio) / fade_range
    pressure = max(0.0, min(1.0, pressure))
    scale = 1.0 - pressure * (1.0 - self.residual_saturation_fade_min_scale)
    return float(delta * scale)

  def _guarded_delta(self, delta: float, observation: np.ndarray) -> float:
    if not self.non_opposing_torque_guard or observation.shape[0] != len(OBSERVATION_FEATURE_NAMES):
      return delta
    desired_curvature = float(observation[OBSERVATION_FEATURE_NAMES.index("desired_curvature")])
    abs_steer_ratio = float(observation[OBSERVATION_FEATURE_NAMES.index("abs_steer_angle_ratio")])
    stock_requested = float(observation[OBSERVATION_FEATURE_NAMES.index("stock_requested_torque")])
    stock_applied = float(observation[OBSERVATION_FEATURE_NAMES.index("stock_applied_torque")])
    if abs(desired_curvature) < self.guard_min_abs_desired_curvature:
      return delta
    if abs_steer_ratio < self.guard_min_abs_steer_ratio:
      return delta
    torque_reference = stock_applied if abs(stock_applied) >= self.guard_min_abs_stock_torque else stock_requested
    if abs(torque_reference) < self.guard_min_abs_stock_torque:
      return delta
    if delta * torque_reference < 0.0:
      return math.copysign(min(abs(delta), self.guard_max_opposing_delta), delta)
    return delta

  def _saturation_limited_delta(self, delta: float, observation: np.ndarray) -> float:
    if self.saturation_torque_limit <= 0.0 or observation.shape[0] != len(OBSERVATION_FEATURE_NAMES):
      return delta
    speed = float(observation[OBSERVATION_FEATURE_NAMES.index("speed_mps")])
    if self.saturation_torque_limit_min_speed_mps > 0.0 and speed < self.saturation_torque_limit_min_speed_mps:
      return delta
    if self.saturation_torque_limit_max_speed_mps > 0.0 and speed > self.saturation_torque_limit_max_speed_mps:
      return delta
    abs_steer_ratio = float(observation[OBSERVATION_FEATURE_NAMES.index("abs_steer_angle_ratio")])
    if abs_steer_ratio < self.saturation_torque_limit_min_steer_ratio:
      return delta
    desired_curvature = float(observation[OBSERVATION_FEATURE_NAMES.index("desired_curvature")])
    if abs(desired_curvature) < self.saturation_torque_limit_min_abs_desired_curvature:
      return delta
    model_heading = float(observation[OBSERVATION_FEATURE_NAMES.index("model_desired_heading_integral_deg")])
    if abs(model_heading) < self.saturation_torque_limit_min_abs_model_heading_deg:
      return delta
    vehicle_heading = float(observation[OBSERVATION_FEATURE_NAMES.index("vehicle_heading_change_deg")])
    if abs(vehicle_heading) < self.saturation_torque_limit_min_abs_vehicle_heading_deg:
      return delta
    stock_applied = float(observation[OBSERVATION_FEATURE_NAMES.index("stock_applied_torque")])
    stock_requested = float(observation[OBSERVATION_FEATURE_NAMES.index("stock_requested_torque")])
    stock_reference = stock_applied if abs(stock_applied) >= self.saturation_torque_limit_min_abs_stock_torque else stock_requested
    if abs(stock_reference) < self.saturation_torque_limit_min_abs_stock_torque:
      return delta

    applied = stock_applied + delta
    if abs(applied) <= self.saturation_torque_limit:
      return delta
    limited_applied = math.copysign(self.saturation_torque_limit, applied)
    return float(np.clip(limited_applied - stock_applied, -self.max_delta, self.max_delta))

  def rate_limit_delta(self, desired_delta: float, state: ResidualPolicyState) -> float:
    if not math.isfinite(desired_delta) or not math.isfinite(state.residual_torque_delta):
      raise ValueError("residual policy delta must be finite")
    if self.residual_delta_rate_limit <= 0.0:
      state.residual_torque_delta = float(desired_delta)
      return float(desired_delta)
    previous_delta = float(state.residual_torque_delta)
    limited_delta = previous_delta + float(np.clip(
      float(desired_delta) - previous_delta,
      -self.residual_delta_rate_limit,
      self.residual_delta_rate_limit,
    ))
    state.residual_torque_delta = limited_delta
    return limited_delta

  def stabilize_applied_delta(self, desired_delta: float, observation: np.ndarray, state: ResidualPolicyState) -> float:
    if not math.isfinite(desired_delta) or not np.isfinite(observation).all():
      raise ValueError("residual policy delta and observation must be finite")
    if self.applied_torque_sign_hold_frames <= 0 or observation.shape[0] != len(OBSERVATION_FEATURE_NAMES):
      return float(desired_delta)

    stock_applied = float(observation[OBSERVATION_FEATURE_NAMES.index("stock_applied_torque")])
    desired_applied = max(-1.0, min(1.0, stock_applied + float(desired_delta)))
    vehicle_heading = abs(float(observation[OBSERVATION_FEATURE_NAMES.index("vehicle_heading_change_deg")]))
    desired_curvature = abs(float(observation[OBSERVATION_FEATURE_NAMES.index("desired_curvature")]))
    hold_active = True
    if (
      self.applied_torque_sign_hold_min_abs_vehicle_heading_deg > 0.0
      and vehicle_heading < self.applied_torque_sign_hold_min_abs_vehicle_heading_deg
    ):
      hold_active = False
    if (
      self.applied_torque_sign_hold_max_abs_desired_curvature > 0.0
      and desired_curvature > self.applied_torque_sign_hold_max_abs_desired_curvature
    ):
      hold_active = False

    def torque_sign(torque: float) -> float:
      if abs(torque) <= self.applied_torque_sign_hold_min_abs_torque:
        return 0.0
      return 1.0 if torque > 0.0 else -1.0

    sign = torque_sign(desired_applied)
    if sign == 0.0:
      state.pending_applied_torque_sign = 0.0
      state.pending_applied_torque_sign_frames = 0
      return float(desired_delta)

    if not hold_active:
      state.applied_torque_sign = sign
      state.pending_applied_torque_sign = 0.0
      state.pending_applied_torque_sign_frames = 0
      return float(desired_delta)

    previous_sign = float(state.applied_torque_sign)
    if previous_sign == 0.0 or sign == previous_sign:
      state.applied_torque_sign = sign
      state.pending_applied_torque_sign = 0.0
      state.pending_applied_torque_sign_frames = 0
      return float(desired_delta)

    if state.pending_applied_torque_sign != sign:
      state.pending_applied_torque_sign = sign
      state.pending_applied_torque_sign_frames = 1
    else:
      state.pending_applied_torque_sign_frames += 1

    if state.pending_applied_torque_sign_frames <= self.applied_torque_sign_hold_frames:
      held_delta = float(np.clip(-stock_applied, -self.max_delta, self.max_delta))
      state.residual_torque_delta = held_delta
      return held_delta

    state.applied_torque_sign = sign
    state.pending_applied_torque_sign = 0.0
    state.pending_applied_torque_sign_frames = 0
    return float(desired_delta)

  def predict_delta(self, observation: np.ndarray) -> float:
    if not self.enabled:
      return 0.0
    observation = np.asarray(observation, dtype=np.float32).reshape(-1)
    if not np.isfinite(observation).all():
      raise ValueError("residual policy observation must be finite")
    if observation.shape[0] == len(OBSERVATION_FEATURE_NAMES):
      full_observation = observation
      policy_observation = observation[self.feature_indices]
    elif observation.shape[0] == len(self.policy_feature_names):
      full_observation = None
      policy_observation = observation
    else:
      raise ValueError("residual policy observation has wrong shape")
    if not self._residual_activation_allowed(full_observation):
      return 0.0
    output_max_delta = self._active_policy_output_max_delta(full_observation)
    x = (policy_observation - self.feature_mu) / self.feature_sigma
    if self.policy_type == "mlp_tanh":
      hidden = x
      for weight, bias in zip(self.mlp_weights[:-1], self.mlp_biases[:-1], strict=True):
        hidden = np.maximum(hidden @ weight + bias, 0.0)
      delta = float(np.tanh((hidden @ self.mlp_weights[-1] + self.mlp_biases[-1]).reshape(-1)[0]) * output_max_delta)
    else:
      delta = float(x @ self.weight + self.bias)
    if not math.isfinite(delta):
      raise ValueError("residual policy prediction must be finite")
    if self.max_delta <= 0.0:
      return 0.0
    clipped_delta = float(np.clip(delta, -output_max_delta, output_max_delta))
    guarded_delta = self._guarded_delta(clipped_delta, observation)
    turn_assisted_delta = self._turn_assisted_delta(guarded_delta, observation)
    suppressed_delta = self._late_exit_suppressed_delta(turn_assisted_delta, observation)
    saturation_faded_delta = self._saturation_faded_delta(suppressed_delta, observation)
    saturation_limited_delta = self._saturation_limited_delta(saturation_faded_delta, observation)
    return float(np.clip(saturation_limited_delta, -self.max_delta, self.max_delta))

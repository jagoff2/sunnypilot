from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import (
  MAX_CURVATURE,
  MAX_LATERAL_ACCEL_NO_ROLL,
  get_curvature_from_plan,
  smooth_value,
)
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan


CAMERA_OFFSET = 0.04
EGO_HALF_WIDTH = 1.08

MIN_LANE_WIDTH = 2.45
HOLD_MIN_LANE_WIDTH = 2.25
MAX_LANE_WIDTH = 4.4
MAX_LANE_WIDTH_SPAN = 0.75
MAX_CENTER_STEP = 0.20
MAX_CENTER_SHAPE_INNOVATION = 0.35
MAX_WIDTH_STEP = 0.30

ENTRY_LANE_PROB = 0.60
HOLD_LANE_PROB = 0.50
ENTRY_STD_MEDIAN = 0.15
ENTRY_STD_MAX = 0.30
HOLD_STD_MEDIAN = 0.30
HOLD_STD_MAX = 0.50
# The model has no road-edge probability head. This is the same confidence
# proxy used by the UI: clip(1 - road_edge_std, 0, 1). Requiring confidence
# strictly above 0.80 is therefore equivalent to requiring std strictly below
# 0.20 at every geometry sample used by the controller.
ROAD_EDGE_CONFIDENCE = 0.60

ENTRY_TIME = 0.75
# Anchored entry-confidence dropouts in the retained route corpus are 300 ms at p95.
ENTRY_CONFIDENCE_GRACE_TIME = 0.30
# Bound evidence age to 15 valid frames plus at most one full p95 confidence burst.
ENTRY_WINDOW_TIME = ENTRY_TIME + ENTRY_CONFIDENCE_GRACE_TIME
RAMP_IN_TIME = 1.00
INVALID_GRACE_TIME = 0.30
RAMP_OUT_TIME = 1.00
HARD_RAMP_OUT_TIME = 0.20

LINE_REFERENCE_TIME = 1.00
LINE_REFERENCE_MAX_AGE = 2.50
LINE_PAIR_WIDTH_TOLERANCE = 0.30
LINE_PAIR_SHAPE_TOLERANCE = 0.25
LINE_PAIR_MAX_SHIFT_FRACTION = 0.80
ACQUISITION_PAIR_MAX_SHIFT = 0.35

SOURCE_SWITCH_DWELL_TIME = 1.00
SOURCE_SWITCH_CONFIDENCE_ADVANTAGE = 0.10
SOURCE_HANDOFF_BLEND_TIME = 1.00
# Accommodate the single float32 rounding gap in decimal confidence values
# while keeping a genuine 0.0999999 advantage below the inclusive threshold.
CONFIDENCE_EPSILON = 5e-8

POLICY_FULL_AGREEMENT = 0.25
POLICY_ZERO_AGREEMENT = 0.75
AVOIDANCE_DIVERGENCE = 0.45
AVOIDANCE_GROWTH = 0.20

CENTERLINE_FILTER_TAU = 0.10
MIN_MERGE_DISTANCE = 12.0
NOMINAL_MERGE_TIME = 2.75
MAX_MERGE_DISTANCE = 70.0
MAX_INITIAL_CURVATURE = MAX_CURVATURE
MAX_LANE_PATH_JERK = 5.0
MAX_LANE_PATH_ACCEL = MAX_LATERAL_ACCEL_NO_ROLL
SPATIAL_DYNAMICS_MARGIN = 0.999
ACTION_SMOOTH_SECONDS = 0.10

MAX_CURVATURE_CORRECTION = 0.01
MAX_LATERAL_ACCEL_CORRECTION = 0.75
CAPPED_PATH_WEIGHT_SLEW_TIME = 1.00

VALID_MODES = ("off", "capped", "absolute")
TIME_EPSILON = 1e-9
GEOMETRY_X = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 30.0], dtype=np.float64)
CENTERLINE_FIT_WEIGHTS = np.array([2.0, 2.0, 1.5, 1.0, 0.75, 0.40], dtype=np.float64)
CENTERLINE_FIT_DESIGN = np.column_stack((np.ones_like(GEOMETRY_X), GEOMETRY_X, GEOMETRY_X * GEOMETRY_X))
CENTERLINE_FIT_OPERATOR = (
  np.linalg.pinv(CENTERLINE_FIT_WEIGHTS[:, None] * CENTERLINE_FIT_DESIGN) * CENTERLINE_FIT_WEIGHTS
)
LINE_PAIR_POSE_DESIGN = np.column_stack((np.ones_like(GEOMETRY_X), GEOMETRY_X))
LINE_PAIR_POSE_OPERATOR = np.linalg.pinv(LINE_PAIR_POSE_DESIGN)
POLICY_X = np.array([10.0, 15.0, 20.0, 25.0, 30.0], dtype=np.float64)
MODEL_X = np.asarray(ModelConstants.X_IDXS, dtype=np.float64)
LANE_CENTERING_INPUT_SERVICES = ("carState", "carControl")


@dataclass(frozen=True)
class LaneCenteringStatus:
  state: str
  source: str
  reason: str
  authority: float
  lane_width: float
  center_offset: float
  policy_disagreement: float
  curvature_correction: float
  lane_path_feasibility: float
  path_weight: float
  requested_lateral_jerk: float
  line_gate: str
  edge_gate: str
  entry_gate: str
  policy_gate: str


@dataclass(frozen=True)
class LaneCenteringInputStatus:
  ready: bool
  calibration_seen: bool
  services_alive: bool
  services_valid: bool
  services_frequency_ok: bool


def get_lane_centering_input_status(sm, live_calib_seen: bool) -> LaneCenteringInputStatus:
  services_alive = sm.all_alive(LANE_CENTERING_INPUT_SERVICES)
  services_valid = sm.all_valid(LANE_CENTERING_INPUT_SERVICES)
  services_frequency_ok = sm.all_freq_ok(LANE_CENTERING_INPUT_SERVICES)
  return LaneCenteringInputStatus(
    ready=bool(live_calib_seen and services_alive and services_valid),
    calibration_seen=bool(live_calib_seen),
    services_alive=bool(services_alive),
    services_valid=bool(services_valid),
    services_frequency_ok=bool(services_frequency_ok),
  )


@dataclass(frozen=True)
class BoundaryGeometry:
  left_y: np.ndarray
  right_y: np.ndarray
  center_y: np.ndarray
  median_width: float
  center_at_lookahead: float
  continuity_center: float
  lateral_offset: float
  heading_error: float
  path_curvature: float
  target_curvature: float


@dataclass(frozen=True)
class BoundaryCandidate:
  geometry: BoundaryGeometry
  source: str
  entry_valid: bool
  temporal_entry_valid: bool
  recovery_valid: bool
  confidence: float


@dataclass(frozen=True)
class CandidatePolicy:
  disagreement: float
  weight: float
  avoidance_shape: bool
  gate: str
  entry_allowed: bool
  hard_veto: bool


class LaneCenteringController:
  def __init__(self, mode: str):
    if mode not in VALID_MODES:
      raise ValueError(f"unsupported lane centering mode: {mode!r}")

    self.mode = mode
    self.reset()

  def reset(self) -> None:
    self.state = "inactive"
    self.source = "none"
    self.reason = "inactive"
    self.authority = 0.0
    self.acquire_time = 0.0
    self.acquire_wall_time = 0.0
    self.entry_confidence_invalid_time = 0.0
    self.acquisition_pair_center: np.ndarray | None = None
    self.acquisition_pair_width: np.ndarray | None = None
    self.acquisition_pair_source = "none"
    self.invalid_time = 0.0
    self.recovery_acquiring = False
    self.lines_stable_time = 0.0
    self.line_reference_ready = False
    self.line_reference_age = np.inf
    self.line_pair_recovery_armed = False
    self.line_pair_center: np.ndarray | None = None
    self.line_pair_width: np.ndarray | None = None
    self.line_pair_source = "none"
    self.width_reference: float | None = None
    self.last_continuity_center: float | None = None
    self.last_width: float | None = None
    self.last_lane_width = np.nan
    self.last_center_offset = np.nan
    self.last_policy_disagreement = np.nan
    self.last_policy_weight = 0.0
    self.last_path_weight = 0.0
    self.last_curvature_correction = 0.0
    self.filtered_center_y: np.ndarray | None = None
    self.last_lane_path_feasibility = np.nan
    self.last_requested_lateral_jerk = np.nan
    self.frame_dt = DT_MDL
    self.motion_dt = DT_MDL
    self.line_gate = "not_evaluated"
    self.edge_gate = "not_evaluated"
    self.entry_gate = "not_evaluated"
    self.policy_gate = "not_evaluated"
    self.candidate_options: dict[str, BoundaryCandidate | None] = {}
    self.challenger_source = "none"
    self.challenger_time = 0.0
    self.challenger_pair_center: np.ndarray | None = None
    self.challenger_pair_width: np.ndarray | None = None
    self.source_handoff_source = "none"
    self.source_handoff_remaining = 0.0
    self.source_handoff_previous_source = "none"
    self.source_handoff_previous_center: np.ndarray | None = None
    self.source_handoff_previous_width: np.ndarray | None = None
    self.candidate_gates: dict[str, tuple[str, str]] = {}

  @staticmethod
  def _lane_probs(model_output: dict[str, np.ndarray]) -> np.ndarray | None:
    raw = np.asarray(model_output.get("lane_lines_prob", []), dtype=np.float64).reshape(-1)
    if raw.size == 8:
      raw = raw[1::2]
    if raw.size != 4 or not np.all(np.isfinite(raw)):
      return None
    return raw

  @staticmethod
  def _std_valid(stds: np.ndarray, median_limit: float, max_limit: float) -> bool:
    near = np.interp(GEOMETRY_X, MODEL_X, stds)
    return bool(np.all(np.isfinite(near)) and np.median(near) <= median_limit and np.max(near) <= max_limit)

  @staticmethod
  def _road_edge_confidence(stds: np.ndarray) -> float:
    near = np.interp(GEOMETRY_X, MODEL_X, stds)
    if not np.all(np.isfinite(near)) or np.any(near < 0.0):
      return -np.inf
    return float(np.min(np.clip(1.0 - near, 0.0, 1.0)))

  @staticmethod
  def _extract_boundaries(model_output: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    try:
      lane_y = np.asarray(model_output["lane_lines"][0, :, :, 0], dtype=np.float64)
      lane_std = np.asarray(model_output["lane_lines_stds"][0, :, :, 0], dtype=np.float64)
      edge_y = np.asarray(model_output["road_edges"][0, :, :, 0], dtype=np.float64)
      edge_std = np.asarray(model_output["road_edges_stds"][0, :, :, 0], dtype=np.float64)
    except (KeyError, IndexError, TypeError, ValueError):
      return None

    if lane_y.shape != (4, ModelConstants.IDX_N) or lane_std.shape != lane_y.shape:
      return None
    if edge_y.shape != (2, ModelConstants.IDX_N) or edge_std.shape != edge_y.shape:
      return None
    return lane_y, lane_std, edge_y, edge_std

  @staticmethod
  def _lookahead(v_ego: float) -> float:
    return float(np.clip(0.9 * v_ego, 10.0, 30.0))

  @staticmethod
  def _fit_centerline(center_y: np.ndarray, lookahead: float) -> tuple[float, float, float, float]:
    center_sample = np.interp(GEOMETRY_X, MODEL_X, center_y)
    offset, slope, quadratic = CENTERLINE_FIT_OPERATOR @ center_sample

    heading_error = float(np.arctan(slope))
    curvature_x = 0.5 * lookahead
    curvature_slope = slope + 2.0 * quadratic * curvature_x
    path_curvature = float(2.0 * quadratic / (1.0 + curvature_slope * curvature_slope) ** 1.5)
    target_curvature = float(path_curvature + 2.0 * heading_error / lookahead +
                             2.0 * offset / (lookahead * lookahead))
    return float(offset), heading_error, path_curvature, target_curvature

  def _line_pair_available(self, source: str | None = None) -> bool:
    return bool(self.line_pair_recovery_armed and self.line_pair_center is not None and self.line_pair_width is not None and
                self.line_reference_age <= LINE_REFERENCE_MAX_AGE and
                (source is None or source == self.line_pair_source))

  @staticmethod
  def _pair_matches_reference(left_sample: np.ndarray, right_sample: np.ndarray,
                              reference_center: np.ndarray, reference_width: np.ndarray,
                              max_shift: float) -> tuple[bool, str]:
    if not (np.all(np.isfinite(left_sample)) and np.all(np.isfinite(right_sample))):
      return False, "pair_nonfinite"
    width = right_sample - left_sample
    if np.max(np.abs(width - reference_width)) > LINE_PAIR_WIDTH_TOLERANCE:
      return False, "pair_width"

    center = 0.5 * (left_sample + right_sample)
    pose_delta = center - reference_center
    lateral_shift, heading_shift = LINE_PAIR_POSE_OPERATOR @ pose_delta
    aligned_residual = pose_delta - (lateral_shift + heading_shift * GEOMETRY_X)
    if np.max(np.abs(aligned_residual)) > LINE_PAIR_SHAPE_TOLERANCE:
      return False, "pair_shape"

    if abs(lateral_shift) > max_shift:
      return False, "pair_shift"
    return True, "valid"

  def _line_pair_matches(self, left_sample: np.ndarray, right_sample: np.ndarray,
                         source: str) -> tuple[bool, str]:
    if not self._line_pair_available(source):
      return False, "pair_reference_unavailable"

    assert self.line_pair_center is not None
    assert self.line_pair_width is not None
    max_shift = LINE_PAIR_MAX_SHIFT_FRACTION * float(np.median(self.line_pair_width))
    return self._pair_matches_reference(
      left_sample, right_sample, self.line_pair_center, self.line_pair_width, max_shift,
    )

  def _clear_acquisition_tracking(self) -> None:
    self.acquire_time = 0.0
    self.acquire_wall_time = 0.0
    self.entry_confidence_invalid_time = 0.0
    self.recovery_acquiring = False
    self.acquisition_pair_center = None
    self.acquisition_pair_width = None
    self.acquisition_pair_source = "none"

  def _set_acquisition_pair(self, candidate: BoundaryCandidate) -> None:
    self.acquisition_pair_center = 0.5 * (
      np.interp(GEOMETRY_X, MODEL_X, candidate.geometry.left_y) +
      np.interp(GEOMETRY_X, MODEL_X, candidate.geometry.right_y)
    )
    self.acquisition_pair_width = (
      np.interp(GEOMETRY_X, MODEL_X, candidate.geometry.right_y) -
      np.interp(GEOMETRY_X, MODEL_X, candidate.geometry.left_y)
    )
    self.acquisition_pair_source = candidate.source

  def _candidate_matches_acquisition_pair(self, candidate: BoundaryCandidate) -> bool:
    if self.acquisition_pair_center is None or self.acquisition_pair_width is None:
      return False
    if candidate.source != self.acquisition_pair_source:
      return False
    left_sample = np.interp(GEOMETRY_X, MODEL_X, candidate.geometry.left_y)
    right_sample = np.interp(GEOMETRY_X, MODEL_X, candidate.geometry.right_y)
    matches, _ = self._pair_matches_reference(
      left_sample, right_sample,
      self.acquisition_pair_center, self.acquisition_pair_width,
      ACQUISITION_PAIR_MAX_SHIFT,
    )
    return matches

  def _raw_pair_matches_acquisition_pair(self, model_output: dict[str, np.ndarray]) -> bool:
    if self.acquisition_pair_center is None or self.acquisition_pair_width is None:
      return False
    extracted = self._extract_boundaries(model_output)
    if extracted is None:
      return False
    lane_y, _, edge_y, _ = extracted
    if self.acquisition_pair_source == "lane_lines":
      left_y, right_y = lane_y[1], lane_y[2]
    elif self.acquisition_pair_source == "left_line_right_edge":
      left_y, right_y = lane_y[1], edge_y[1]
    elif self.acquisition_pair_source == "left_edge_right_line":
      left_y, right_y = edge_y[0], lane_y[2]
    else:
      return False
    left_sample = np.interp(GEOMETRY_X, MODEL_X, left_y)
    right_sample = np.interp(GEOMETRY_X, MODEL_X, right_y)
    matches, _ = self._pair_matches_reference(
      left_sample, right_sample,
      self.acquisition_pair_center, self.acquisition_pair_width,
      ACQUISITION_PAIR_MAX_SHIFT,
    )
    return matches

  def _grace_acquisition_interruption(self, reason: str) -> bool:
    self.entry_confidence_invalid_time += self.frame_dt
    if self.entry_confidence_invalid_time <= ENTRY_CONFIDENCE_GRACE_TIME + TIME_EPSILON:
      self.reason = reason
      return True

    self.state = "inactive"
    self.source = "none"
    self._clear_acquisition_tracking()
    self._clear_challenger()
    self.reason = "entry_conditions"
    return False

  def _line_geometry_mode(self, source: str) -> str:
    if self.state == "active" and source == self.source:
      if self.source_handoff_source == source and self.source_handoff_remaining > TIME_EPSILON:
        return "handoff"
      return "hold"
    if (self.state == "active" and source == self.source_handoff_previous_source and
        self.source_handoff_remaining > TIME_EPSILON):
      return "rollback"
    if self.state == "exiting" and source == self.source:
      return "recovery"
    if (self.state == "acquiring" and self.recovery_acquiring and
        source == self.acquisition_pair_source):
      return "recovery"
    if self.state == "inactive" and self._line_pair_available(source):
      return "recovery"
    return "entry"

  def _geometry(self, left_y: np.ndarray, right_y: np.ndarray, lookahead: float,
                geometry_mode: str, source: str) -> tuple[BoundaryGeometry | None, str]:
    if not np.all(np.isfinite(left_y)) or not np.all(np.isfinite(right_y)):
      return None, "nonfinite"

    left_sample = np.interp(GEOMETRY_X, MODEL_X, left_y)
    right_sample = np.interp(GEOMETRY_X, MODEL_X, right_y)
    width = right_sample - left_sample

    min_width = MIN_LANE_WIDTH if geometry_mode == "entry" else HOLD_MIN_LANE_WIDTH
    if np.any(width < min_width) or np.any(width > MAX_LANE_WIDTH):
      return None, "width_range"
    if np.ptp(width) > MAX_LANE_WIDTH_SPAN:
      return None, "width_span"
    if np.any(left_sample >= right_sample):
      return None, "crossing"

    if geometry_mode == "entry":
      # The camera is 4 cm right of vehicle center. These are the same ego-envelope
      # conventions used by lane-departure warning.
      if np.any(left_sample[:2] > -(EGO_HALF_WIDTH + CAMERA_OFFSET)):
        return None, "ego_left"
      if np.any(right_sample[:2] < EGO_HALF_WIDTH - CAMERA_OFFSET):
        return None, "ego_right"
    else:
      vehicle_center_y = -CAMERA_OFFSET
      if np.any(left_sample[:2] >= vehicle_center_y):
        return None, "ego_center_left"
      if np.any(right_sample[:2] <= vehicle_center_y):
        return None, "ego_center_right"

    if geometry_mode == "recovery":
      pair_matches, pair_gate = self._line_pair_matches(left_sample, right_sample, source)
      if not pair_matches:
        return None, pair_gate
    elif geometry_mode == "rollback":
      if (self.source_handoff_previous_center is None or
          self.source_handoff_previous_width is None):
        return None, "rollback_reference_unavailable"
      pair_matches, pair_gate = self._pair_matches_reference(
        left_sample, right_sample,
        self.source_handoff_previous_center, self.source_handoff_previous_width,
        ACQUISITION_PAIR_MAX_SHIFT,
      )
      if not pair_matches:
        return None, f"rollback_{pair_gate}"

    center_y = 0.5 * (left_y + right_y) + CAMERA_OFFSET
    center_sample = np.interp(GEOMETRY_X, MODEL_X, center_y)
    center_at_lookahead = float(np.interp(lookahead, MODEL_X, center_y))
    continuity_center = float(np.interp(20.0, MODEL_X, center_y))
    median_width = float(np.median(width))

    if geometry_mode in ("hold", "handoff"):
      if self.last_continuity_center is not None and abs(continuity_center - self.last_continuity_center) > MAX_CENTER_STEP:
        return None, "center_step"
      if geometry_mode == "hold" and self.filtered_center_y is not None:
        predicted_center = np.interp(GEOMETRY_X, MODEL_X, self.filtered_center_y)
        if np.max(np.abs(center_sample - predicted_center)) > MAX_CENTER_SHAPE_INNOVATION:
          return None, "center_innovation"
      if self.last_width is not None and abs(median_width - self.last_width) > MAX_WIDTH_STEP:
        return None, "width_step"

    lateral_offset, heading_error, path_curvature, target_curvature = self._fit_centerline(center_y, lookahead)
    return BoundaryGeometry(left_y, right_y, center_y, median_width, center_at_lookahead, continuity_center,
                            lateral_offset, heading_error, path_curvature, target_curvature), "valid"

  def _line_candidate(self, model_output: dict[str, np.ndarray],
                      lookahead: float) -> tuple[BoundaryCandidate | None, tuple[bool, bool], str, str]:
    extracted = self._extract_boundaries(model_output)
    probs = self._lane_probs(model_output)
    if extracted is None:
      return None, (False, False), "boundary_data", "boundary_data"
    if probs is None:
      return None, (False, False), "probability_data", "probability_data"

    lane_y, lane_std, _, _ = extracted
    entry_stds = (
      self._std_valid(lane_std[1], ENTRY_STD_MEDIAN, ENTRY_STD_MAX),
      self._std_valid(lane_std[2], ENTRY_STD_MEDIAN, ENTRY_STD_MAX),
    )
    hold_stds = (
      self._std_valid(lane_std[1], HOLD_STD_MEDIAN, HOLD_STD_MAX),
      self._std_valid(lane_std[2], HOLD_STD_MEDIAN, HOLD_STD_MAX),
    )
    entry_sides = (
      probs[1] > ENTRY_LANE_PROB and entry_stds[0],
      probs[2] > ENTRY_LANE_PROB and entry_stds[1],
    )
    temporal_entry_sides = (
      probs[1] > ENTRY_LANE_PROB and hold_stds[0],
      probs[2] > ENTRY_LANE_PROB and hold_stds[1],
    )
    hold_sides = (
      probs[1] > HOLD_LANE_PROB and hold_stds[0],
      probs[2] > HOLD_LANE_PROB and hold_stds[1],
    )

    entry_failures = []
    for side, probability, std_valid in zip(("left", "right"), probs[1:3], entry_stds, strict=True):
      if probability <= ENTRY_LANE_PROB:
        entry_failures.append(f"{side}_probability")
      elif not std_valid:
        entry_failures.append(f"{side}_uncertainty")
    entry_gate = "+".join(entry_failures) if entry_failures else "valid"

    if not all(hold_sides):
      hold_failures = []
      for side, probability, std_valid in zip(("left", "right"), probs[1:3], hold_stds, strict=True):
        if probability <= HOLD_LANE_PROB:
          hold_failures.append(f"{side}_probability")
        elif not std_valid:
          hold_failures.append(f"{side}_uncertainty")
      return None, hold_sides, "+".join(hold_failures), entry_gate

    geometry_mode = self._line_geometry_mode("lane_lines")
    geometry, geometry_gate = self._geometry(lane_y[1], lane_y[2], lookahead, geometry_mode, "lane_lines")
    if geometry is None:
      return None, hold_sides, f"geometry_{geometry_gate}", entry_gate
    recovery_valid = geometry_mode == "recovery" and all(entry_sides)
    return BoundaryCandidate(
      geometry, "lane_lines", all(entry_sides), all(temporal_entry_sides), recovery_valid,
      float(np.clip(min(probs[1], probs[2]), 0.0, 1.0)),
    ), hold_sides, "valid", entry_gate

  def _mixed_candidate(self, model_output: dict[str, np.ndarray], lookahead: float,
                       source: str) -> tuple[BoundaryCandidate | None, str, str]:
    extracted = self._extract_boundaries(model_output)
    probs = self._lane_probs(model_output)
    if extracted is None:
      return None, "boundary_data", "boundary_data"
    if probs is None:
      return None, "probability_data", "probability_data"
    lane_y, lane_std, edge_y, edge_std = extracted

    if source == "left_line_right_edge":
      line_side = "left"
      edge_side = "right"
      line_index = 1
      edge_index = 1
      left_y, right_y = lane_y[1], edge_y[1]
    elif source == "left_edge_right_line":
      line_side = "right"
      edge_side = "left"
      line_index = 2
      edge_index = 0
      left_y, right_y = edge_y[0], lane_y[2]
    else:
      raise ValueError(f"unsupported mixed boundary source: {source!r}")

    entry_probability_valid = bool(probs[line_index] > ENTRY_LANE_PROB)
    hold_probability_valid = bool(probs[line_index] > HOLD_LANE_PROB)
    entry_std_valid = self._std_valid(lane_std[line_index], ENTRY_STD_MEDIAN, ENTRY_STD_MAX)
    hold_std_valid = self._std_valid(lane_std[line_index], HOLD_STD_MEDIAN, HOLD_STD_MAX)
    edge_confidence = self._road_edge_confidence(edge_std[edge_index])
    edge_valid = edge_confidence > ROAD_EDGE_CONFIDENCE
    line_entry_valid = entry_probability_valid and entry_std_valid
    line_temporal_valid = entry_probability_valid and hold_std_valid

    entry_failures = []
    if not entry_probability_valid:
      entry_failures.append(f"{line_side}_probability")
    elif not entry_std_valid:
      entry_failures.append(f"{line_side}_uncertainty")
    if not edge_valid:
      entry_failures.append(f"{edge_side}_confidence")
    entry_gate = "+".join(entry_failures) if entry_failures else "valid"

    hold_failures = []
    if not hold_probability_valid:
      hold_failures.append(f"{line_side}_probability")
    elif not hold_std_valid:
      hold_failures.append(f"{line_side}_uncertainty")
    if not edge_valid:
      hold_failures.append(f"{edge_side}_confidence")
    if hold_failures:
      return None, "+".join(hold_failures), entry_gate

    geometry_mode = self._line_geometry_mode(source)
    geometry, geometry_gate = self._geometry(left_y, right_y, lookahead, geometry_mode, source)
    if geometry is None:
      return None, f"geometry_{geometry_gate}", entry_gate
    recovery_valid = geometry_mode == "recovery" and line_entry_valid and edge_valid
    return BoundaryCandidate(
      geometry, source, line_entry_valid and edge_valid,
      line_temporal_valid and edge_valid, recovery_valid,
      float(np.clip(min(probs[line_index], edge_confidence), 0.0, 1.0)),
    ), "valid", entry_gate

  @staticmethod
  def _policy_agreement(model_output: dict[str, np.ndarray], center_y: np.ndarray) -> tuple[float, float, bool]:
    try:
      plan = np.asarray(model_output["plan"][0], dtype=np.float64)
      plan_x = plan[:, Plan.POSITION.start]
      plan_y = plan[:, Plan.POSITION.start + 1]
    except (KeyError, IndexError, TypeError, ValueError):
      return np.inf, 0.0, True

    if plan_x.size != ModelConstants.IDX_N or not np.all(np.isfinite(plan_x)) or not np.all(np.isfinite(plan_y)):
      return np.inf, 0.0, True
    if np.any(np.diff(plan_x) < 0.0) or plan_x[-1] < POLICY_X[-1]:
      return np.inf, 0.0, True

    plan_sample = np.interp(POLICY_X, plan_x, plan_y)
    center_sample = np.interp(POLICY_X, MODEL_X, center_y)
    residual = plan_sample - center_sample
    abs_residual = np.abs(residual)
    disagreement = float(np.max(abs_residual))

    if disagreement <= POLICY_FULL_AGREEMENT:
      weight = 1.0
    elif disagreement >= POLICY_ZERO_AGREEMENT:
      weight = 0.0
    else:
      weight = (POLICY_ZERO_AGREEMENT - disagreement) / (POLICY_ZERO_AGREEMENT - POLICY_FULL_AGREEMENT)

    avoidance_shape = bool(np.max(abs_residual[1:]) >= AVOIDANCE_DIVERGENCE and
                           np.max(abs_residual[1:]) - abs_residual[0] >= AVOIDANCE_GROWTH)
    return disagreement, float(weight), avoidance_shape

  def _evaluate_center_policy(self, model_output: dict[str, np.ndarray],
                              center_y: np.ndarray) -> CandidatePolicy:
    disagreement, weight, avoidance_shape = self._policy_agreement(model_output, center_y)
    if not np.isfinite(disagreement):
      gate = "plan_invalid"
    elif avoidance_shape:
      gate = "avoidance_shape"
    elif disagreement >= POLICY_ZERO_AGREEMENT:
      gate = "disagreement_veto"
    elif disagreement > POLICY_FULL_AGREEMENT:
      gate = "blended"
    else:
      gate = "full"
    entry_allowed = gate == "full" or (self.mode == "absolute" and gate == "blended")
    return CandidatePolicy(
      disagreement=disagreement,
      weight=weight,
      avoidance_shape=avoidance_shape,
      gate=gate,
      entry_allowed=entry_allowed,
      hard_veto=bool(disagreement >= POLICY_ZERO_AGREEMENT or avoidance_shape),
    )

  def _evaluate_policy(self, model_output: dict[str, np.ndarray],
                       candidate: BoundaryCandidate) -> CandidatePolicy:
    return self._evaluate_center_policy(model_output, candidate.geometry.center_y)

  def _update_line_reference(self, candidate: BoundaryCandidate | None) -> None:
    if candidate is not None and candidate.source in (
      "lane_lines", "left_line_right_edge", "left_edge_right_line",
    ) and candidate.entry_valid:
      if candidate.source != self.line_pair_source:
        self.lines_stable_time = 0.0
        self.line_reference_ready = False
        self.width_reference = None
      self.lines_stable_time += self.frame_dt
      if self.lines_stable_time + TIME_EPSILON >= LINE_REFERENCE_TIME:
        self.line_reference_ready = True
      self.line_reference_age = 0.0
      left_sample = np.interp(GEOMETRY_X, MODEL_X, candidate.geometry.left_y)
      right_sample = np.interp(GEOMETRY_X, MODEL_X, candidate.geometry.right_y)
      self.line_pair_center = 0.5 * (left_sample + right_sample)
      self.line_pair_width = right_sample - left_sample
      self.line_pair_source = candidate.source
      width = candidate.geometry.median_width
      alpha = self.frame_dt / (2.0 + self.frame_dt)
      self.width_reference = width if self.width_reference is None else (1.0 - alpha) * self.width_reference + alpha * width
    else:
      self.lines_stable_time = 0.0
      self.line_reference_age += self.frame_dt
      if self.line_reference_age > LINE_REFERENCE_MAX_AGE:
        self.line_reference_ready = False
        self.line_pair_recovery_armed = False
        self.line_pair_center = None
        self.line_pair_width = None
        self.line_pair_source = "none"
        self.width_reference = None

  def _clear_line_reference(self) -> None:
    self.lines_stable_time = 0.0
    self.line_reference_ready = False
    self.line_reference_age = np.inf
    self.line_pair_recovery_armed = False
    self.line_pair_center = None
    self.line_pair_width = None
    self.line_pair_source = "none"
    self.width_reference = None

  def _select_candidate(self, model_output: dict[str, np.ndarray], lookahead: float) -> BoundaryCandidate | None:
    line_candidate, _, self.line_gate, line_entry_gate = self._line_candidate(model_output, lookahead)
    left_mixed_candidate, left_mixed_gate, left_mixed_entry_gate = self._mixed_candidate(
      model_output, lookahead, "left_line_right_edge",
    )
    right_mixed_candidate, right_mixed_gate, right_mixed_entry_gate = self._mixed_candidate(
      model_output, lookahead, "left_edge_right_line",
    )
    self.edge_gate = "not_selected"
    self.entry_gate = line_entry_gate

    candidates = {
      "lane_lines": line_candidate,
      "left_line_right_edge": left_mixed_candidate,
      "left_edge_right_line": right_mixed_candidate,
    }
    gates = {
      "lane_lines": ("not_needed", line_entry_gate),
      "left_line_right_edge": (left_mixed_gate, left_mixed_entry_gate),
      "left_edge_right_line": (right_mixed_gate, right_mixed_entry_gate),
    }
    self.candidate_options = candidates
    self.candidate_gates = gates
    source_order = ("lane_lines", "left_line_right_edge", "left_edge_right_line")
    selected_candidate = None
    selected_source = None
    if self.state == "acquiring" and self.acquisition_pair_source in candidates:
      selected_source = self.acquisition_pair_source
      selected_candidate = candidates[selected_source]
    elif self.state in ("active", "exiting") and self.source in candidates:
      selected_source = self.source
      selected_candidate = candidates[selected_source]
    elif self.state == "inactive":
      eligible_sources = [
        source for source in source_order
        if candidates[source] is not None and candidates[source].entry_valid
      ]
      if not eligible_sources:
        eligible_sources = [source for source in source_order if candidates[source] is not None]
      selected_source = max(
        eligible_sources,
        key=lambda source: candidates[source].confidence,
        default=None,
      )
      selected_candidate = candidates[selected_source] if selected_source is not None else None
    else:
      selected_source = next((source for source in source_order if candidates[source] is not None), None)
      selected_candidate = candidates[selected_source] if selected_source is not None else None

    if selected_source is not None:
      self.edge_gate, self.entry_gate = gates[selected_source]

    if selected_candidate is not None:
      return selected_candidate

    return None

  def _use_candidate_gates(self, source: str) -> None:
    gates = self.candidate_gates.get(source)
    if gates is not None:
      self.edge_gate, self.entry_gate = gates

  def _clear_challenger(self) -> None:
    self.challenger_source = "none"
    self.challenger_time = 0.0
    self.challenger_pair_center = None
    self.challenger_pair_width = None

  def _clear_source_handoff(self) -> None:
    self.source_handoff_source = "none"
    self.source_handoff_remaining = 0.0
    self.source_handoff_previous_source = "none"
    self.source_handoff_previous_center = None
    self.source_handoff_previous_width = None

  @staticmethod
  def _candidate_pair_samples(candidate: BoundaryCandidate) -> tuple[np.ndarray, np.ndarray]:
    left_sample = np.interp(GEOMETRY_X, MODEL_X, candidate.geometry.left_y)
    right_sample = np.interp(GEOMETRY_X, MODEL_X, candidate.geometry.right_y)
    return 0.5 * (left_sample + right_sample), right_sample - left_sample

  def _select_policy_safe_cold_candidate(self,
                                         policy_options: dict[str, CandidatePolicy]) -> BoundaryCandidate | None:
    source_order = ("lane_lines", "left_line_right_edge", "left_edge_right_line")
    eligible = [
      source for source in source_order
      if (self.candidate_options.get(source) is not None and
          self.candidate_options[source].entry_valid and
          source in policy_options and policy_options[source].entry_allowed and
          not policy_options[source].hard_veto)
    ]
    selected_source = max(
      eligible,
      key=lambda source: self.candidate_options[source].confidence,
      default=None,
    )
    if selected_source is None:
      return None
    self._use_candidate_gates(selected_source)
    return self.candidate_options[selected_source]

  def _advance_source_challenger(self, policy_options: dict[str, CandidatePolicy]) -> BoundaryCandidate | None:
    source_order = ("lane_lines", "left_line_right_edge", "left_edge_right_line")
    if (self.state not in ("acquiring", "active", "exiting") or self.source not in source_order or
        self.source_handoff_remaining > TIME_EPSILON):
      self._clear_challenger()
      return None

    eligible = []
    for source in source_order:
      candidate = self.candidate_options.get(source)
      policy = policy_options.get(source)
      if (candidate is not None and candidate.entry_valid and policy is not None and
          policy.entry_allowed and not policy.hard_veto):
        eligible.append(source)

    lane_lines_clean = "lane_lines" in eligible
    challenger_source = None
    if self.source != "lane_lines" and lane_lines_clean:
      # A strict, policy-safe two-line pair is semantically stronger than a
      # line/edge fallback. It reclaims ownership after the same stable dwell,
      # even when the raw line/edge score is numerically higher.
      challenger_source = "lane_lines"
    elif self.source != "lane_lines" or not lane_lines_clean:
      highest_source = max(
        eligible,
        key=lambda source: self.candidate_options[source].confidence,
        default=None,
      )
      if highest_source is not None and highest_source != self.source:
        incumbent = self.candidate_options.get(self.source)
        challenger = self.candidate_options[highest_source]
        if (incumbent is None or
            challenger.confidence - incumbent.confidence + CONFIDENCE_EPSILON >=
            SOURCE_SWITCH_CONFIDENCE_ADVANTAGE):
          challenger_source = highest_source

    if challenger_source is None:
      self._clear_challenger()
      return None

    challenger = self.candidate_options[challenger_source]
    challenger_center, challenger_width = self._candidate_pair_samples(challenger)
    if challenger_source != self.challenger_source:
      self.challenger_source = challenger_source
      self.challenger_time = self.frame_dt
      self.challenger_pair_center = challenger_center
      self.challenger_pair_width = challenger_width
    else:
      assert self.challenger_pair_center is not None
      assert self.challenger_pair_width is not None
      left_sample = np.interp(GEOMETRY_X, MODEL_X, challenger.geometry.left_y)
      right_sample = np.interp(GEOMETRY_X, MODEL_X, challenger.geometry.right_y)
      pair_matches, _ = self._pair_matches_reference(
        left_sample, right_sample,
        self.challenger_pair_center, self.challenger_pair_width,
        ACQUISITION_PAIR_MAX_SHIFT,
      )
      if pair_matches:
        self.challenger_time += self.frame_dt
      else:
        self.challenger_time = self.frame_dt
        self.challenger_pair_center = challenger_center
        self.challenger_pair_width = challenger_width

    if self.challenger_time + TIME_EPSILON < SOURCE_SWITCH_DWELL_TIME:
      return None
    return challenger

  def _source_handoff_rollback_candidate(self,
                                         policy_options: dict[str, CandidatePolicy]) -> BoundaryCandidate | None:
    if (self.source_handoff_remaining <= TIME_EPSILON or
        self.source_handoff_previous_source == "none"):
      return None
    candidate = self.candidate_options.get(self.source_handoff_previous_source)
    policy = policy_options.get(self.source_handoff_previous_source)
    if (candidate is None or not candidate.entry_valid or policy is None or
        not policy.entry_allowed or policy.hard_veto):
      return None
    return candidate

  def _commit_source_handoff(self, candidate: BoundaryCandidate, *, rollback: bool = False) -> None:
    previous_source = self.source
    previous_candidate = self.candidate_options.get(previous_source)
    self.state = "active"
    self.source = candidate.source
    self.reason = "source_rollback" if rollback else "source_handoff"
    self.invalid_time = 0.0
    self._clear_acquisition_tracking()
    self.line_pair_recovery_armed = True
    self.source_handoff_source = candidate.source
    self.source_handoff_remaining = SOURCE_HANDOFF_BLEND_TIME
    if not rollback and previous_candidate is not None:
      previous_center, previous_width = self._candidate_pair_samples(previous_candidate)
      self.source_handoff_previous_source = previous_source
      self.source_handoff_previous_center = previous_center
      self.source_handoff_previous_width = previous_width
    else:
      self.source_handoff_previous_source = "none"
      self.source_handoff_previous_center = None
      self.source_handoff_previous_width = None
    self.last_continuity_center = candidate.geometry.continuity_center
    self.last_width = candidate.geometry.median_width
    self._clear_challenger()

  @staticmethod
  def _interp_with_extrapolation(x_new: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    result = np.interp(x_new, x, y)
    left_slope = (y[1] - y[0]) / max(x[1] - x[0], 1e-6)
    right_slope = (y[-1] - y[-2]) / max(x[-1] - x[-2], 1e-6)
    left = x_new < x[0]
    right = x_new > x[-1]
    result[left] = y[0] + left_slope * (x_new[left] - x[0])
    result[right] = y[-1] + right_slope * (x_new[right] - x[-1])
    return result

  def _propagate_centerline(self, v_ego: float, current_curvature: float) -> None:
    if self.filtered_center_y is None:
      return

    ds = max(float(v_ego), 0.0) * self.motion_dt
    curvature = float(np.clip(current_curvature, -MAX_INITIAL_CURVATURE, MAX_INITIAL_CURVATURE))
    dpsi = curvature * ds
    if abs(curvature) > 1e-6:
      dx = np.sin(dpsi) / curvature
      dy = (1.0 - np.cos(dpsi)) / curvature
    else:
      dx = ds
      dy = 0.0

    cosine = np.cos(dpsi)
    sine = np.sin(dpsi)
    delta_x = MODEL_X - dx
    delta_y = self.filtered_center_y - dy
    transformed_x = cosine * delta_x + sine * delta_y
    transformed_y = -sine * delta_x + cosine * delta_y
    order = np.argsort(transformed_x)
    transformed_x = transformed_x[order]
    transformed_y = transformed_y[order]
    if np.any(np.diff(transformed_x) <= 1e-6) or not np.all(np.isfinite(transformed_y)):
      self.filtered_center_y = None
      return
    self.filtered_center_y = self._interp_with_extrapolation(MODEL_X, transformed_x, transformed_y)

  def _observe_centerline(self, center_y: np.ndarray, source: str) -> None:
    measurement = np.asarray(center_y, dtype=np.float64)
    if measurement.shape != MODEL_X.shape or not np.all(np.isfinite(measurement)):
      return
    if self.filtered_center_y is None:
      self.filtered_center_y = measurement.copy()
      self._clear_source_handoff()
      return

    if self.source_handoff_source == source and self.source_handoff_remaining > TIME_EPSILON:
      alpha = min(1.0, self.frame_dt / self.source_handoff_remaining)
      self.source_handoff_remaining = max(0.0, self.source_handoff_remaining - self.frame_dt)
      if self.source_handoff_remaining <= TIME_EPSILON:
        self._clear_source_handoff()
    else:
      alpha = 1.0 - np.exp(-self.frame_dt / CENTERLINE_FILTER_TAU)
    self.filtered_center_y += alpha * (measurement - self.filtered_center_y)

  def _reference_coefficients(self) -> np.ndarray | None:
    if self.filtered_center_y is None:
      return None
    center_sample = np.interp(GEOMETRY_X, MODEL_X, self.filtered_center_y)
    coefficients = CENTERLINE_FIT_OPERATOR @ center_sample
    return coefficients if np.all(np.isfinite(coefficients)) else None

  @staticmethod
  def _quintic_coefficients(join_distance: float, reference: np.ndarray,
                            current_curvature: float) -> np.ndarray:
    offset, slope, quadratic = reference
    target_y = offset + slope * join_distance + quadratic * join_distance * join_distance
    target_slope = slope + 2.0 * quadratic * join_distance
    target_second = 2.0 * quadratic

    coefficients = np.zeros(6, dtype=np.float64)
    coefficients[2] = 0.5 * current_curvature
    matrix = np.array([
      [join_distance**3, join_distance**4, join_distance**5],
      [3.0 * join_distance**2, 4.0 * join_distance**3, 5.0 * join_distance**4],
      [6.0 * join_distance, 12.0 * join_distance**2, 20.0 * join_distance**3],
    ])
    residual = np.array([
      target_y - coefficients[2] * join_distance**2,
      target_slope - 2.0 * coefficients[2] * join_distance,
      target_second - 2.0 * coefficients[2],
    ])
    coefficients[3:] = np.linalg.solve(matrix, residual)
    return coefficients

  @staticmethod
  def _evaluate_spatial_path(x: np.ndarray, join_distance: float, reference: np.ndarray,
                             quintic: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    merge = x <= join_distance
    a0, a1, a2, a3, a4, a5 = quintic
    merge_y = (((((a5 * x + a4) * x + a3) * x + a2) * x + a1) * x + a0)
    merge_slope = ((((5.0 * a5 * x + 4.0 * a4) * x + 3.0 * a3) * x + 2.0 * a2) * x + a1)
    merge_second = (((20.0 * a5 * x + 12.0 * a4) * x + 6.0 * a3) * x + 2.0 * a2)

    offset, reference_slope, quadratic = reference
    reference_y = offset + reference_slope * x + quadratic * x * x
    reference_first = reference_slope + 2.0 * quadratic * x
    y = np.where(merge, merge_y, reference_y)
    slope = np.where(merge, merge_slope, reference_first)
    second = np.where(merge, merge_second, 2.0 * quadratic)
    return y, slope, second

  def _lane_plan_for_distance(self, base_plan: np.ndarray, reference: np.ndarray,
                              current_curvature: float, join_distance: float) -> np.ndarray:
    lane_plan = np.array(base_plan, dtype=np.float64, copy=True)
    speed = np.asarray(base_plan[0, :, Plan.VELOCITY.start], dtype=np.float64)
    speed = np.maximum(speed, 0.0)
    current_curvature = float(np.clip(current_curvature, -MAX_INITIAL_CURVATURE, MAX_INITIAL_CURVATURE))
    quintic = self._quintic_coefficients(join_distance, reference, current_curvature)

    time_indices = np.asarray(ModelConstants.T_IDXS, dtype=np.float64)
    time_steps = np.diff(time_indices)
    path_x = np.r_[0.0, np.cumsum(0.5 * (speed[:-1] + speed[1:]) * time_steps)]
    for _ in range(2):
      _, iteration_slope, _ = self._evaluate_spatial_path(path_x, join_distance, reference, quintic)
      forward_velocity = speed / np.sqrt(1.0 + iteration_slope * iteration_slope)
      path_x[1:] = np.cumsum(0.5 * (forward_velocity[:-1] + forward_velocity[1:]) * time_steps)

    path_y, slope, second = self._evaluate_spatial_path(path_x, join_distance, reference, quintic)
    curvature = second / np.power(1.0 + slope * slope, 1.5)
    yaw = np.arctan(slope)

    lane_plan[0, :, Plan.POSITION.start] = path_x
    lane_plan[0, :, Plan.POSITION.start + 1] = path_y
    lane_plan[0, :, Plan.VELOCITY.start + 1] = 0.0
    lane_plan[0, :, Plan.ACCELERATION.start + 1] = curvature * speed * speed
    lane_plan[0, :, Plan.T_FROM_CURRENT_EULER.start + 2] = yaw
    lane_plan[0, :, Plan.ORIENTATION_RATE.start + 2] = curvature * speed
    return lane_plan

  @staticmethod
  def _lateral_plan_feasible(plan: np.ndarray, margin: float = 1.0) -> bool:
    lateral_accel = plan[0, :, Plan.ACCELERATION.start + 1]
    time_indices = np.asarray(ModelConstants.T_IDXS, dtype=np.float64)
    lateral_jerk = np.diff(lateral_accel) / np.diff(time_indices)
    return bool(
      np.max(np.abs(lateral_accel)) <= margin * MAX_LANE_PATH_ACCEL + TIME_EPSILON and
      np.max(np.abs(lateral_jerk)) <= margin * MAX_LANE_PATH_JERK + TIME_EPSILON
    )

  def _build_lane_plan(self, base_plan: np.ndarray, v_ego: float,
                       current_curvature: float) -> tuple[np.ndarray | None, float]:
    if base_plan.shape != (1, ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH) or not np.all(np.isfinite(base_plan)):
      return None, np.nan
    reference = self._reference_coefficients()
    if reference is None:
      return None, np.nan

    nominal_join_distance = float(np.clip(
      NOMINAL_MERGE_TIME * max(v_ego, 1.0), MIN_MERGE_DISTANCE, MAX_MERGE_DISTANCE,
    ))
    lane_plan = self._lane_plan_for_distance(
      base_plan, reference, current_curvature, nominal_join_distance,
    )
    if self._lateral_plan_feasible(lane_plan, SPATIAL_DYNAMICS_MARGIN):
      self.last_lane_path_feasibility = 1.0
      return lane_plan, nominal_join_distance

    self.last_lane_path_feasibility = 0.0
    return None, nominal_join_distance

  @staticmethod
  def _blend_plans(base_plan: np.ndarray, lane_plan: np.ndarray, weight: float) -> np.ndarray:
    if weight <= 0.0:
      return base_plan
    if weight >= 1.0:
      return lane_plan

    selected = np.array(base_plan, dtype=np.float64, copy=True)
    inverse = 1.0 - weight
    selected[0, :, Plan.POSITION.start] = (
      inverse * base_plan[0, :, Plan.POSITION.start] + weight * lane_plan[0, :, Plan.POSITION.start]
    )
    selected[0, :, Plan.POSITION.start + 1] = (
      inverse * base_plan[0, :, Plan.POSITION.start + 1] + weight * lane_plan[0, :, Plan.POSITION.start + 1]
    )
    selected[0, :, Plan.VELOCITY.start + 1] = (
      inverse * base_plan[0, :, Plan.VELOCITY.start + 1] + weight * lane_plan[0, :, Plan.VELOCITY.start + 1]
    )
    selected[0, :, Plan.ACCELERATION.start + 1] = (
      inverse * base_plan[0, :, Plan.ACCELERATION.start + 1] + weight * lane_plan[0, :, Plan.ACCELERATION.start + 1]
    )

    base_yaw = base_plan[0, :, Plan.T_FROM_CURRENT_EULER.start + 2]
    lane_yaw = lane_plan[0, :, Plan.T_FROM_CURRENT_EULER.start + 2]
    yaw_delta = np.arctan2(np.sin(lane_yaw - base_yaw), np.cos(lane_yaw - base_yaw))
    selected[0, :, Plan.T_FROM_CURRENT_EULER.start + 2] = base_yaw + weight * yaw_delta
    selected[0, :, Plan.ORIENTATION_RATE.start + 2] = (
      inverse * base_plan[0, :, Plan.ORIENTATION_RATE.start + 2] +
      weight * lane_plan[0, :, Plan.ORIENTATION_RATE.start + 2]
    )
    return selected

  @staticmethod
  def _plan_curvature(plan: np.ndarray, v_ego: float, lat_action_t: float) -> float:
    return float(get_curvature_from_plan(
      plan[0, :, Plan.T_FROM_CURRENT_EULER.start + 2],
      plan[0, :, Plan.ORIENTATION_RATE.start + 2],
      ModelConstants.T_IDXS,
      v_ego,
      lat_action_t,
    ))

  def _select_output(self, model_output: dict[str, np.ndarray], v_ego: float,
                     current_curvature: float, lat_action_t: float, base_smoothed_curvature: float,
                     previous_selected_curvature: float) -> dict[str, np.ndarray]:
    previous_path_weight = self.last_path_weight
    self.last_path_weight = 0.0
    self.last_curvature_correction = 0.0
    self.last_requested_lateral_jerk = np.nan

    try:
      base_plan = np.asarray(model_output["plan"], dtype=np.float64)
    except (KeyError, IndexError, TypeError, ValueError):
      return model_output

    requested_weight = 0.0
    lane_plan = None
    if self.authority > 0.0 and self.filtered_center_y is not None:
      lane_plan, _ = self._build_lane_plan(base_plan, v_ego, current_curvature)
      if lane_plan is None:
        self.reset()
        self.reason = "path_infeasible"
        self.entry_gate = "path_infeasible"
        self.policy_gate = "path_infeasible"
        self.last_lane_path_feasibility = 0.0
    if lane_plan is None:
      return model_output

    if lane_plan is not None:
      requested_weight = self.authority if self.mode == "absolute" else self.authority * self.last_policy_weight
      requested_weight = float(np.clip(requested_weight, 0.0, 1.0))
    selected_plan = base_plan if lane_plan is None else self._blend_plans(base_plan, lane_plan, requested_weight)
    selected_curvature = self._plan_curvature(selected_plan, v_ego, lat_action_t)
    selected_smoothed_curvature = smooth_value(
      selected_curvature, previous_selected_curvature, ACTION_SMOOTH_SECONDS,
    )

    if self.mode == "capped" and lane_plan is not None and requested_weight > 0.0:
      correction_limit = min(
        MAX_CURVATURE_CORRECTION,
        MAX_LATERAL_ACCEL_CORRECTION / max(v_ego * v_ego, 1.0),
      )
      if abs(selected_smoothed_curvature - base_smoothed_curvature) > correction_limit:
        base_path_curvature = self._plan_curvature(base_plan, v_ego, lat_action_t)
        base_path_smoothed = smooth_value(
          base_path_curvature, previous_selected_curvature, ACTION_SMOOTH_SECONDS,
        )
        denominator = selected_smoothed_curvature - base_path_smoothed
        target_correction = float(np.clip(
          selected_smoothed_curvature - base_smoothed_curvature,
          -correction_limit,
          correction_limit,
        ))
        target_curvature = base_smoothed_curvature + target_correction
        if abs(denominator) <= TIME_EPSILON:
          weight_fraction = 0.0
        else:
          weight_fraction = float(np.clip(
            (target_curvature - base_path_smoothed) / denominator, 0.0, 1.0,
          ))
        requested_weight *= weight_fraction
        selected_plan = self._blend_plans(base_plan, lane_plan, requested_weight)
        selected_curvature = self._plan_curvature(selected_plan, v_ego, lat_action_t)
        selected_smoothed_curvature = smooth_value(
          selected_curvature, previous_selected_curvature, ACTION_SMOOTH_SECONDS,
        )

    if self.mode == "capped" and self.state == "active":
      maximum_weight_step = self.frame_dt / CAPPED_PATH_WEIGHT_SLEW_TIME
      slewed_weight = float(np.clip(
        requested_weight,
        max(0.0, previous_path_weight - maximum_weight_step),
        min(1.0, previous_path_weight + maximum_weight_step),
      ))
      if slewed_weight != requested_weight:
        requested_weight = slewed_weight
        selected_plan = self._blend_plans(base_plan, lane_plan, requested_weight)
        selected_curvature = self._plan_curvature(selected_plan, v_ego, lat_action_t)
        selected_smoothed_curvature = smooth_value(
          selected_curvature, previous_selected_curvature, ACTION_SMOOTH_SECONDS,
        )

    self.last_requested_lateral_jerk = (
      abs(selected_smoothed_curvature - previous_selected_curvature) *
      max(v_ego * v_ego, 1.0) / self.frame_dt
    )

    selected_output = model_output.copy()
    selected_output["plan"] = selected_plan.astype(model_output["plan"].dtype, copy=False)
    self.last_path_weight = requested_weight
    self.last_curvature_correction = float(selected_smoothed_curvature - base_smoothed_curvature)
    return selected_output

  def _start_exit(self, reason: str, hard: bool) -> None:
    if hard:
      self._clear_challenger()
      self._clear_source_handoff()
    if self.state != "exiting" or hard:
      self.exit_ramp_time = HARD_RAMP_OUT_TIME if hard else RAMP_OUT_TIME
    self.state = "exiting"
    self.reason = reason
    self._clear_acquisition_tracking()
    self.invalid_time = 0.0
    self.authority = max(0.0, self.authority - self.frame_dt / self.exit_ramp_time)
    if self.authority == 0.0:
      self.state = "inactive"
      self.source = "none"
      self._clear_challenger()
      self._clear_source_handoff()

  def _accept_candidate(self, candidate: BoundaryCandidate, policy_weight: float) -> None:
    self._observe_centerline(candidate.geometry.center_y, candidate.source)
    self.source = candidate.source
    self.last_continuity_center = candidate.geometry.continuity_center
    self.last_width = candidate.geometry.median_width
    self.last_policy_weight = policy_weight

  def _status(self) -> LaneCenteringStatus:
    return LaneCenteringStatus(
      state=self.state,
      source=self.source,
      reason=self.reason,
      authority=float(self.authority),
      lane_width=float(self.last_lane_width),
      center_offset=float(self.last_center_offset),
      policy_disagreement=float(self.last_policy_disagreement),
      curvature_correction=float(self.last_curvature_correction),
      lane_path_feasibility=float(self.last_lane_path_feasibility),
      path_weight=float(self.last_path_weight),
      requested_lateral_jerk=float(self.last_requested_lateral_jerk),
      line_gate=self.line_gate,
      edge_gate=self.edge_gate,
      entry_gate=self.entry_gate,
      policy_gate=self.policy_gate,
    )

  def _finish(self, model_output: dict[str, np.ndarray], v_ego: float,
              current_curvature: float, lat_action_t: float, base_smoothed_curvature: float,
              previous_selected_curvature: float) -> tuple[dict[str, np.ndarray], LaneCenteringStatus]:
    selected_output = self._select_output(
      model_output, v_ego, current_curvature, lat_action_t,
      base_smoothed_curvature, previous_selected_curvature,
    )
    return selected_output, self._status()

  def update(self, model_output: dict[str, np.ndarray], v_ego: float, current_curvature: float,
             lat_action_t: float, frame_dt: float, base_smoothed_curvature: float,
             previous_selected_curvature: float,
             lat_active: bool, model_valid: bool, left_blinker: bool, right_blinker: bool,
             lane_change_active: bool) -> tuple[dict[str, np.ndarray], LaneCenteringStatus]:
    self.line_gate = "not_evaluated"
    self.edge_gate = "not_evaluated"
    self.entry_gate = "not_evaluated"
    self.policy_gate = "not_evaluated"

    if self.mode == "off" or not lat_active:
      self.reset()
      self.reason = "mode_off" if self.mode == "off" else "lateral_inactive"
      self.entry_gate = self.reason
      self.policy_gate = self.reason
      return model_output, self._status()

    frame_gap = not np.isfinite(frame_dt) or frame_dt <= 0.0 or frame_dt > INVALID_GRACE_TIME
    if np.isfinite(frame_dt) and frame_dt > 0.0:
      # State-machine timers must follow real elapsed time. Only ego-motion
      # propagation is clipped, since an unusually delayed frame should not
      # extrapolate the tracked centerline an arbitrarily large distance.
      self.frame_dt = float(frame_dt)
      self.motion_dt = float(np.clip(frame_dt, 0.5 * DT_MDL, 5.0 * DT_MDL))
    else:
      self.frame_dt = DT_MDL
      self.motion_dt = DT_MDL
    motion_curvature = current_curvature if np.isfinite(current_curvature) else 0.0
    motion_speed = v_ego if np.isfinite(v_ego) else 0.0
    if frame_gap:
      self.filtered_center_y = None
    else:
      self._propagate_centerline(motion_speed, motion_curvature)

    hard_reason = None
    if frame_gap:
      hard_reason = "model_gap"
    elif (not model_valid or not np.isfinite(v_ego) or not np.isfinite(current_curvature) or
          not np.isfinite(lat_action_t) or not np.isfinite(base_smoothed_curvature) or
          not np.isfinite(previous_selected_curvature)):
      hard_reason = "invalid_model"
    elif left_blinker or right_blinker or lane_change_active:
      hard_reason = "lane_change"

    if hard_reason is not None:
      self.entry_gate = hard_reason
      self.policy_gate = hard_reason
      self._clear_line_reference()
      self._start_exit(hard_reason, hard=True)
      safe_action_t = lat_action_t if np.isfinite(lat_action_t) and lat_action_t > 0.0 else DT_MDL
      safe_base_curvature = base_smoothed_curvature if np.isfinite(base_smoothed_curvature) else 0.0
      safe_previous_curvature = previous_selected_curvature if np.isfinite(previous_selected_curvature) else 0.0
      return self._finish(
        model_output, motion_speed, motion_curvature, safe_action_t,
        safe_base_curvature, safe_previous_curvature,
      )

    lookahead = self._lookahead(v_ego)
    candidate = self._select_candidate(model_output, lookahead)
    candidate_valid = candidate is not None
    entry_valid = False
    temporal_entry_valid = False
    policy_weight = 0.0
    hard_policy_veto = False
    source_handoff_committed = False

    policy_options = {
      source: self._evaluate_policy(model_output, option)
      for source, option in self.candidate_options.items()
      if option is not None
    }
    if self.state == "inactive":
      policy_safe_candidate = self._select_policy_safe_cold_candidate(policy_options)
      if policy_safe_candidate is not None:
        candidate = policy_safe_candidate
        candidate_valid = True

    if candidate is not None:
      candidate_policy = policy_options[candidate.source]
      policy_weight = candidate_policy.weight
      self.last_lane_width = candidate.geometry.median_width
      self.last_center_offset = candidate.geometry.center_at_lookahead
      self.last_policy_disagreement = candidate_policy.disagreement
      hard_policy_veto = candidate_policy.hard_veto
      candidate_valid = not hard_policy_veto
      self.policy_gate = candidate_policy.gate
      entry_valid = candidate.entry_valid and candidate_policy.entry_allowed
      temporal_entry_valid = candidate.temporal_entry_valid and candidate_policy.entry_allowed
      if candidate.entry_valid and not candidate_policy.entry_allowed:
        self.entry_gate = f"policy_{self.policy_gate}"

    if self.authority > 0.0 and self.filtered_center_y is not None and self.source != "none":
      commanded_policy = self._evaluate_center_policy(model_output, self.filtered_center_y)
      if commanded_policy.hard_veto:
        hard_policy_veto = True
        candidate_valid = False
        self.last_policy_disagreement = commanded_policy.disagreement
        self.policy_gate = commanded_policy.gate

    if hard_policy_veto:
      self._update_line_reference(None)
      self._start_exit("policy_avoidance", hard=True)
      return self._finish(
        model_output, v_ego, current_curvature, lat_action_t,
        base_smoothed_curvature, previous_selected_curvature,
      )

    challenger = self._advance_source_challenger(policy_options)
    if challenger is not None:
      self._commit_source_handoff(challenger)
      self._use_candidate_gates(challenger.source)
      candidate = challenger
      candidate_policy = policy_options[challenger.source]
      candidate_valid = True
      entry_valid = True
      temporal_entry_valid = challenger.temporal_entry_valid and candidate_policy.entry_allowed
      policy_weight = candidate_policy.weight
      self.last_lane_width = challenger.geometry.median_width
      self.last_center_offset = challenger.geometry.center_at_lookahead
      self.last_policy_disagreement = candidate_policy.disagreement
      self.policy_gate = candidate_policy.gate
      source_handoff_committed = True

    acquisition_restart_reason = None
    if self.state == "acquiring":
      self.acquire_wall_time += self.frame_dt
      if self.acquire_wall_time > ENTRY_WINDOW_TIME + TIME_EPSILON:
        self.state = "inactive"
        self.source = "none"
        self._clear_acquisition_tracking()
        self._clear_challenger()
        acquisition_restart_reason = "entry_window_restart"

    if self.state == "acquiring":
      if candidate is None:
        if self._raw_pair_matches_acquisition_pair(model_output):
          acquisition_gate = self.line_gate if self.acquisition_pair_source == "lane_lines" else self.edge_gate
          grace_reason = "entry_geometry_grace" if acquisition_gate.startswith("geometry_") else "entry_confidence_grace"
          self._grace_acquisition_interruption(grace_reason)
          self._update_line_reference(None)
          return self._finish(
            model_output, v_ego, current_curvature, lat_action_t,
            base_smoothed_curvature, previous_selected_curvature,
          )
      elif not self._candidate_matches_acquisition_pair(candidate):
        self.state = "inactive"
        self.source = "none"
        self.filtered_center_y = None
        self._clear_acquisition_tracking()
        self._clear_challenger()
        self._clear_source_handoff()
        self._clear_line_reference()
        acquisition_restart_reason = "entry_pair_restart"

    reference_candidate = candidate if (
      candidate_valid and candidate is not None and
      (self.state == "active" or entry_valid or temporal_entry_valid)
    ) else None
    self._update_line_reference(reference_candidate)

    recovered = False
    if self.state == "exiting":
      if candidate_valid and candidate is not None and candidate.recovery_valid and entry_valid:
        self.state = "active"
        self._accept_candidate(candidate, policy_weight)
        self.reason = "recovered"
        self.invalid_time = 0.0
        self.line_pair_recovery_armed = True
        recovered = True
      else:
        self._start_exit(self.reason, hard=False)
        return self._finish(
          model_output, v_ego, current_curvature, lat_action_t,
          base_smoothed_curvature, previous_selected_curvature,
        )

    if not candidate_valid:
      self.reason = "boundary_invalid"
      if self.state == "acquiring":
        self.state = "inactive"
        self.source = "none"
        self._clear_acquisition_tracking()
        self._clear_challenger()
      elif self.state == "active":
        self.invalid_time += self.frame_dt
        if self.invalid_time > INVALID_GRACE_TIME:
          rollback_candidate = self._source_handoff_rollback_candidate(policy_options)
          if rollback_candidate is not None:
            rollback_policy = policy_options[rollback_candidate.source]
            self._commit_source_handoff(rollback_candidate, rollback=True)
            self._use_candidate_gates(rollback_candidate.source)
            self.last_lane_width = rollback_candidate.geometry.median_width
            self.last_center_offset = rollback_candidate.geometry.center_at_lookahead
            self.last_policy_disagreement = rollback_policy.disagreement
            self.policy_gate = rollback_policy.gate
            self._update_line_reference(rollback_candidate)
            self._accept_candidate(rollback_candidate, rollback_policy.weight)
            return self._finish(
              model_output, v_ego, current_curvature, lat_action_t,
              base_smoothed_curvature, previous_selected_curvature,
            )
          self._start_exit("boundary_lost", hard=False)
      return self._finish(
        model_output, v_ego, current_curvature, lat_action_t,
        base_smoothed_curvature, previous_selected_curvature,
      )

    self.invalid_time = 0.0

    if self.state == "inactive":
      if entry_valid:
        self.state = "acquiring"
        self.acquire_time = self.frame_dt
        self.acquire_wall_time = self.frame_dt
        self.entry_confidence_invalid_time = 0.0
        self.recovery_acquiring = candidate.recovery_valid
        self._set_acquisition_pair(candidate)
        self._accept_candidate(candidate, policy_weight)
        self.reason = acquisition_restart_reason or "tracking"
      else:
        self._clear_acquisition_tracking()
        self.reason = "entry_window_expired" if acquisition_restart_reason == "entry_window_restart" else "entry_conditions"
    elif self.state == "acquiring":
      if entry_valid or temporal_entry_valid:
        self.entry_confidence_invalid_time = 0.0
        self.acquire_time += self.frame_dt
        self._set_acquisition_pair(candidate)
        self._accept_candidate(candidate, policy_weight)
        self.reason = "tracking"
        if self.acquire_time + TIME_EPSILON >= ENTRY_TIME:
          self.state = "active"
          self.authority = 0.0
          self._clear_acquisition_tracking()
          self.line_pair_recovery_armed = True
      else:
        grace_reason = "entry_policy_grace" if self.policy_gate == "blended" else "entry_confidence_grace"
        self._grace_acquisition_interruption(grace_reason)
    elif self.state == "active":
      if not recovered:
        self._accept_candidate(candidate, policy_weight)
      handoff_active = source_handoff_committed or self.source_handoff_remaining > TIME_EPSILON
      self.reason = "recovered" if recovered else ("source_handoff" if handoff_active else "tracking")
      self.authority = min(1.0, self.authority + self.frame_dt / RAMP_IN_TIME)

    return self._finish(
      model_output, v_ego, current_curvature, lat_action_t,
      base_smoothed_curvature, previous_selected_curvature,
    )

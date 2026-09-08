"""Use the C3X selected lateral trajectory with C4 longitudinal actions."""
import math
import time
from collections import deque
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np

from openpilot.cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.controls.lib.drive_helpers import get_curvature_from_plan, smooth_value
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan
from openpilot.selfdrive.modeld.lane_centering_safety import LANE_CENTERING_STATUS_VERSION, MAX_MODEL_AGE_NS, model_clock_ns
from openpilot.selfdrive.modeld.lane_centering import (
  ACTION_SMOOTH_SECONDS, CAMERA_OFFSET, EGO_HALF_WIDTH, LaneCenteringController, get_lane_centering_input_status,
)


TELEMETRY_MIN_INTERVAL = 0.5
TELEMETRY_HEALTH_INTERVAL = 5.0
TELEMETRY_MAX_TRANSITIONS = 16
TELEMETRY_X = np.array([0.0, 5.0, 10.0, 20.0, 30.0])
FRAME_DELAY_TAU = 0.5


class ModelPublicationTiming:
  """Estimate complete camera-to-publication age for the next inference input."""
  def __init__(self):
    self.frame_delay = DT_MDL
    self.publication_interval = DT_MDL
    self.last_timestamp_eof = None
    self.last_publication_ns = None

  def observe(self, timestamp_eof, now_ns, valid=True):
    if timestamp_eof <= 0 or now_ns <= 0:
      return math.nan
    age = (now_ns - timestamp_eof) * 1e-9
    previous = self.last_timestamp_eof
    if previous is None or timestamp_eof > previous:
      self.last_timestamp_eof = timestamp_eof
      dt = DT_MDL if previous is None else (timestamp_eof - previous) * 1e-9
      previous_publication = self.last_publication_ns
      if (valid and 0.0 <= age <= MAX_MODEL_AGE_NS * 1e-9 and
          (previous_publication is None or now_ns > previous_publication)):
        self.last_publication_ns = now_ns
      else:
        return age
      if 0.0 < dt <= MAX_MODEL_AGE_NS * 1e-9:
        # Smooth camera phase/runtime jitter. This replaces the old fixed50ms;
        # actuator delay and selected-action smoothing remain separate terms.
        alpha = 1 - math.exp(-dt / FRAME_DELAY_TAU)
        self.frame_delay += alpha * (age - self.frame_delay)
        if previous_publication is not None:
          interval = (now_ns - previous_publication) * 1e-9
          if 0.0 < interval <= MAX_MODEL_AGE_NS * 1e-9:
            interval_alpha = 1 - math.exp(-interval / FRAME_DELAY_TAU)
            self.publication_interval += interval_alpha * (interval - self.publication_interval)
    return age


def _finite_number(value):
  value = float(value)
  return round(value, 6) if math.isfinite(value) else None


def _sample_geometry(x, y):
  x, y = np.asarray(x), np.asarray(y)
  if x.ndim != 1 or y.shape != x.shape or len(x) < 2 or not np.all(np.isfinite(x)) or not np.all(np.diff(x) > 0):
    return None
  # Do not imply knowledge beyond the available prediction horizon.
  return [_finite_number(value) for value in np.interp(TELEMETRY_X, x, y, left=np.nan, right=np.nan)]


def _geometry_evidence(original, selected):
  evidence = {'sample_x_m': TELEMETRY_X.tolist()}
  for name, output in (('original_path_y_m', original), ('selected_path_y_m', selected)):
    try:
      plan = output['plan'][0]
      evidence[name] = _sample_geometry(plan[:, Plan.POSITION.start], plan[:, Plan.POSITION.start + 1])
    except (KeyError, IndexError, TypeError, ValueError):
      evidence[name] = None
  for name, field, index in (('left_line_y_m', 'lane_lines', 1), ('right_line_y_m', 'lane_lines', 2),
                              ('left_edge_y_m', 'road_edges', 0), ('right_edge_y_m', 'road_edges', 1)):
    try:
      evidence[name] = _sample_geometry(ModelConstants.X_IDXS, original[field][0, index, :, 0] + CAMERA_OFFSET)
    except (KeyError, IndexError, TypeError, ValueError):
      evidence[name] = None
  for name in ('original', 'selected'):
    path = evidence[f'{name}_path_y_m']
    for boundary in ('line', 'edge'):
      left, right = evidence[f'left_{boundary}_y_m'], evidence[f'right_{boundary}_y_m']
      margins = [] if path is None or left is None or right is None else [
        min(y - l, r - y) - EGO_HALF_WIDTH for y, l, r in zip(path, left, right, strict=True)
        if y is not None and l is not None and r is not None
      ]
      evidence[f'{name}_sampled_min_{boundary}_axis_aligned_body_margin_m'] = _finite_number(min(margins)) if margins else None
  return evidence


class LaneCenteringTelemetry:
  """Record bounded transition batches and periodic geometry through logmessaged."""
  def __init__(self, clock=time.monotonic):
    self.clock = clock
    self.last_emit_time = None
    self.last_key = None
    self.transitions = deque(maxlen=TELEMETRY_MAX_TRANSITIONS)
    self.dropped_transitions = 0

  def update(self, mode, status, inputs, original, selected, timestamp_eof, frame_dt, v_ego, base_curvature, selected_curvature, timing=None):
    now = self.clock()
    key = (mode, status.state, status.source, status.reason, status.line_gate, status.edge_gate, status.entry_gate, status.policy_gate,
           getattr(status, 'containment', 'unavailable'), getattr(status, 'safety_blocked', False), status.collision_risk, status.policy_fallback,
           inputs.ready, inputs.calibration_seen, inputs.services_alive, inputs.services_valid, inputs.services_frequency_ok)
    geometry = None
    if key != self.last_key:
      geometry = _geometry_evidence(original, selected)
      if len(self.transitions) == self.transitions.maxlen:
        self.dropped_transitions += 1
      self.transitions.append({'timestamp_eof': int(timestamp_eof), 'mode': mode, 'state': status.state, 'source': status.source,
                               'reason': status.reason, 'line_gate': status.line_gate, 'edge_gate': status.edge_gate,
                               'entry_gate': status.entry_gate, 'policy_gate': status.policy_gate,
                               'authority': _finite_number(status.authority), 'path_weight': _finite_number(status.path_weight),
                               'containment': getattr(status, 'containment', 'unavailable'),
                               'safety_blocked': getattr(status, 'safety_blocked', False),
                               'collision_risk': status.collision_risk,
                               'policy_fallback': status.policy_fallback,
                               'min_clearance_m': _finite_number(status.min_clearance),
                               'checked_distance_m': _finite_number(status.checked_distance),
                               'response_time_s': _finite_number(status.response_time),
                               'inputs': asdict(inputs), 'geometry': geometry})
      self.last_key = key
    elapsed = math.inf if self.last_emit_time is None else now - self.last_emit_time
    if elapsed < TELEMETRY_MIN_INTERVAL or (not self.transitions and elapsed < TELEMETRY_HEALTH_INTERVAL):
      return
    # Status NaNs represent unavailable estimates. Encode them as JSON null so
    # route tools can consume this event without permissive NaN JSON parsing.
    status_fields = {key: _finite_number(value) if isinstance(value, float) else value for key, value in asdict(status).items()}
    cloudlog.event('lane_centering_status', schema_version=2, mode=mode, timestamp_eof=int(timestamp_eof),
                   frame_dt_s=_finite_number(frame_dt), speed_mps=_finite_number(v_ego), status=status_fields,
                   inputs=asdict(inputs), base_curvature=_finite_number(base_curvature), selected_curvature=_finite_number(selected_curvature),
                   geometry=geometry if geometry is not None else _geometry_evidence(original, selected),
                   timing=timing or {}, transitions=list(self.transitions),
                   dropped_transitions=self.dropped_transitions)
    self.transitions.clear()
    self.dropped_transitions = 0
    self.last_emit_time = now


def _get_lateral_curvature(model_output, previous_curvature, v_ego, lat_action_t, dt=DT_MDL):
  if not all(math.isfinite(value) for value in (previous_curvature, v_ego, lat_action_t, dt)) or dt <= 0.0:
    return math.nan
  if v_ego > 0.3:
    plan = model_output['plan'][0]
    curvature = get_curvature_from_plan(
      plan[:, Plan.T_FROM_CURRENT_EULER][:, 2], plan[:, Plan.ORIENTATION_RATE][:, 2],
      ModelConstants.T_IDXS, v_ego, lat_action_t,
    )
    return smooth_value(curvature, previous_curvature, ACTION_SMOOTH_SECONDS, dt=dt)
  return previous_curvature


def _valid_plan(model_output):
  try:
    plan = np.asarray(model_output['plan'])
    return bool(plan.shape == (1, ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH) and np.all(np.isfinite(plan)))
  except (KeyError, IndexError, TypeError, ValueError):
    return False


def update_lane_change_helpers(model_output, car_state, lat_active, v_ego, desire_helper, edge_controller, model_data_sp):
  # RELC consumes only these unchanged boundary fields. Evaluate it before
  # selecting the trajectory, without filling/advancing PublishState twice.
  boundary_model = SimpleNamespace(
    roadEdgeStds=model_output['road_edges_stds'][0, :, 0, 0].tolist(),
    laneLineProbs=model_output['lane_lines_prob'][0, 1::2].tolist(),
    roadEdges=[SimpleNamespace(y=edge[:, 0].tolist()) for edge in model_output['road_edges'][0]],
  )
  left_edge, right_edge = edge_controller.update_and_fill(boundary_model, model_data_sp, v_ego)
  desire = model_output['desire_state'][0].reshape(-1)
  lane_change_prob = desire[log.Desire.laneChangeLeft] + desire[log.Desire.laneChangeRight]
  desire_helper.update(car_state, lat_active, lane_change_prob, left_edge, right_edge)


class LaneCenteringModelAdapter:
  def __init__(self, mode="absolute"):
    self.controller = LaneCenteringController(mode)
    self.previous_base_action = log.ModelDataV2.Action()
    self.previous_selected_action = log.ModelDataV2.Action()
    self.last_timestamp_eof = None
    self.frame_valid = False
    self.plan_valid = False
    self.telemetry = LaneCenteringTelemetry()
    self.timing = ModelPublicationTiming()
    self.execution_time = 0.0
    self.planner_execution_time = 0.0

  @property
  def frame_delay(self):
    return self.timing.frame_delay

  @property
  def action_delay(self):
    return self.timing.publication_interval / 2

  def update(self, model_output, action_from_model, sm, calibration_seen, lane_change_state,
             timestamp_eof, v_ego, lat_action_t, long_action_t):
    execution_start = time.perf_counter()
    frame_dt = DT_MDL if self.last_timestamp_eof is None else (timestamp_eof - self.last_timestamp_eof) * 1e-9
    self.frame_valid = bool(timestamp_eof > 0 and 0.0 < frame_dt <= MAX_MODEL_AGE_NS * 1e-9)
    smoothing_dt = frame_dt if self.frame_valid else DT_MDL
    if self.last_timestamp_eof is None or timestamp_eof > self.last_timestamp_eof:
      self.last_timestamp_eof = timestamp_eof
    # Preserve native longitudinal action/history. Both lateral histories use
    # their corresponding plans, as on c3x, including when lane selection abstains.
    self.plan_valid = _valid_plan(model_output)
    action_valid = self.plan_valid and all(math.isfinite(value) for value in (v_ego, lat_action_t, long_action_t))
    base_action = self.previous_base_action
    if action_valid:
      try:
        raw_base_action = action_from_model(model_output, self.previous_base_action, lat_action_t, long_action_t, v_ego,
                                            lateral_smooth_seconds=0.0)
        base_curvature = _get_lateral_curvature(model_output, self.previous_base_action.desiredCurvature, v_ego, lat_action_t, smoothing_dt)
        action_valid = all(math.isfinite(value) for value in (base_curvature, raw_base_action.desiredCurvature,
                                                              raw_base_action.desiredAcceleration))
        if action_valid:
          base_action = log.ModelDataV2.Action(
            desiredCurvature=float(base_curvature), desiredAcceleration=raw_base_action.desiredAcceleration,
            shouldStop=raw_base_action.shouldStop,
          )
      except (KeyError, IndexError, TypeError, ValueError, FloatingPointError, OverflowError):
        action_valid = False
    inputs = get_lane_centering_input_status(sm, calibration_seen)
    # Both entrypoints include the captured 0.1 s smoothing compensation in
    # inference inputs and this lookahead, in addition to C4 frame/action delay.
    planner_start = time.perf_counter()
    selected_output, status = self.controller.update(
      model_output, v_ego, sm['carControl'].currentCurvature, lat_action_t, frame_dt,
      base_action.desiredCurvature, self.previous_selected_action.desiredCurvature,
      sm['carControl'].latActive, inputs.ready and action_valid, sm['carState'].leftBlinker, sm['carState'].rightBlinker,
      lane_change_state != log.LaneChangeState.off,
    )
    self.planner_execution_time = time.perf_counter() - planner_start

    # Match the action to the same trajectory published for NNLC preview. Keep
    # one selected filter history through acquisition, release and base-plan fallback.
    self.plan_valid = self.plan_valid and _valid_plan(selected_output)
    curvature = math.nan
    if action_valid and self.plan_valid:
      # A rejected lane proposal cannot continue steering through its retained
      # selected-filter history. The core returns the original policy plan.
      curvature = base_action.desiredCurvature if status.policy_fallback else _get_lateral_curvature(
        selected_output, self.previous_selected_action.desiredCurvature, v_ego, lat_action_t, smoothing_dt)
    action_valid = action_valid and math.isfinite(curvature)
    self.frame_valid = self.frame_valid and action_valid
    if self.frame_valid:
      action = log.ModelDataV2.Action(
        desiredCurvature=float(curvature), desiredAcceleration=base_action.desiredAcceleration,
        shouldStop=base_action.shouldStop,
      )
    else:
      # Held values are deliberately invalid commands. Do not advance either
      # history, poison the next frame, or claim a placeholder is a valid plan.
      base_action = self.previous_base_action
      action = self.previous_selected_action
      curvature = action.desiredCurvature
    if not action_valid:
      status = replace(status, reason='invalid_action', containment='blocked', safety_blocked=True)
    status = replace(status, curvature_correction=float(curvature - base_action.desiredCurvature),
                     requested_lateral_jerk=float(abs(curvature - self.previous_selected_action.desiredCurvature) *
                                                  max(v_ego * v_ego, 1.0) / self.controller.frame_dt))

    if self.frame_valid:
      self.previous_base_action = base_action
      self.previous_selected_action = action
    self.telemetry.update(self.controller.mode, status, inputs, model_output, selected_output, timestamp_eof, frame_dt, v_ego,
                          base_action.desiredCurvature, action.desiredCurvature, timing={
                            'planner_execution_s': _finite_number(self.planner_execution_time),
                            'adapter_execution_before_telemetry_s': _finite_number(time.perf_counter() - execution_start),
                            'adapter_output_age_s': _finite_number((model_clock_ns() - timestamp_eof) * 1e-9),
                            'frame_delay_s': _finite_number(self.frame_delay),
                            'lateral_action_delay_s': _finite_number(self.action_delay),
                          })
    self.execution_time = time.perf_counter() - execution_start
    return selected_output, action, status

  def fill_status(self, model, status):
    """Attach the decision only after fill_model_msg set the matching frame."""
    selected = model.laneCentering
    selected.version = LANE_CENTERING_STATUS_VERSION
    selected.valid = bool(self.frame_valid and model.timestampEof == self.last_timestamp_eof and math.isfinite(model.action.desiredCurvature))
    selected.frameId = model.frameId
    selected.timestampEof = model.timestampEof
    selected.state = status.state
    selected.source = status.source
    selected.reason = status.reason
    selected.authority = status.authority
    selected.pathWeight = status.path_weight
    selected.containment = status.containment
    selected.minClearance = status.min_clearance
    selected.responseTime = status.response_time
    selected.checkedDistance = status.checked_distance
    selected.safetyBlocked = status.safety_blocked
    selected.collisionRisk = status.collision_risk
    selected.policyFallback = status.policy_fallback
    selected.lineGate = status.line_gate
    selected.edgeGate = status.edge_gate
    selected.entryGate = status.entry_gate
    selected.policyGate = status.policy_gate
    selected.executionTime = self.execution_time
    selected.frameDelay = self.frame_delay
    selected.actionDelay = self.action_delay
    # Called after model/pose serialization, immediately before pm.send. Learn
    # the complete age once, keeping inference and extracted action horizons equal.
    selected.publishAge = self.timing.observe(model.timestampEof, model_clock_ns(), selected.valid)

  def fill_invalid_model(self, message, action, status, frame_id, timestamp_eof):
    """Publish explicit rejection when malformed geometry cannot be serialized."""
    message.valid = False
    model = message.modelV2
    model.frameId = frame_id
    model.timestampEof = timestamp_eof
    model.action = action
    self.fill_status(model, status)

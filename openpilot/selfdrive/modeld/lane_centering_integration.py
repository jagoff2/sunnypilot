"""Use the C3X selected lateral trajectory with C4 longitudinal actions."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from openpilot.cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.controls.lib.drive_helpers import MAX_LATERAL_JERK, get_curvature_from_plan, smooth_value
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan
from openpilot.selfdrive.modeld.lane_centering import (
  ACTION_SMOOTH_SECONDS, LaneCenteringController, get_lane_centering_input_status,
)


def _get_lateral_curvature(model_output, previous_curvature, v_ego, lat_action_t,
                           frame_dt=DT_MDL, smooth_seconds=ACTION_SMOOTH_SECONDS):
  if v_ego > 0.3:
    plan = model_output['plan'][0]
    if (not np.isfinite(lat_action_t) or lat_action_t <= 0.0 or
        not np.all(np.isfinite(plan[:, Plan.T_FROM_CURRENT_EULER.start + 2])) or
        not np.all(np.isfinite(plan[:, Plan.ORIENTATION_RATE.start + 2]))):
      return previous_curvature
    curvature = get_curvature_from_plan(
      plan[:, Plan.T_FROM_CURRENT_EULER][:, 2], plan[:, Plan.ORIENTATION_RATE][:, 2],
      ModelConstants.T_IDXS, v_ego, lat_action_t,
    )
    return smooth_value(curvature, previous_curvature, smooth_seconds, dt=frame_dt)
  return previous_curvature


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


def fill_lane_centering_status(model_message, model_output, status, mode):
  """Publish the original policy and exact selector state on every model frame."""
  diagnostic = model_message.laneCentering
  diagnostic.present = True
  diagnostic.mode = mode
  for field in ("state", "source", "reason", "authority"):
    setattr(diagnostic, field, getattr(status, field))
  for field, source in (
    ("pathWeight", "path_weight"), ("geometryHorizon", "geometry_horizon"),
    ("convergenceDistance", "convergence_distance"), ("convergenceTime", "convergence_time"),
    ("horizonLimited", "horizon_limited"), ("policyDisagreement", "policy_disagreement"),
    ("laneWidth", "lane_width"), ("centerOffset", "center_offset"), ("lineGate", "line_gate"),
    ("edgeGate", "edge_gate"), ("entryGate", "entry_gate"), ("policyGate", "policy_gate"),
    ("lanePathFeasibility", "lane_path_feasibility"), ("baseCurvature", "base_curvature"),
    ("selectedCurvatureRaw", "selected_curvature_raw"), ("selectedCurvature", "selected_curvature"),
    ("actionLimited", "action_limited"), ("requestedLateralJerk", "requested_lateral_jerk"),
    ("frameDt", "frame_dt"), ("modelTimestampEof", "model_timestamp_eof"),
    ("nativeCurvature", "native_curvature"), ("requestedCurvature", "requested_curvature"),
    ("actionTime", "action_time"),
  ):
    setattr(diagnostic, field, getattr(status, source))
  plan = np.asarray(model_output.get("plan", []))
  diagnostic.basePlanValid = bool(plan.shape == (1, ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH) and
                                  np.all(np.isfinite(plan)))
  if diagnostic.basePlanValid:
    for field, indices in (("basePosition", Plan.POSITION), ("baseOrientation", Plan.T_FROM_CURRENT_EULER),
                           ("baseOrientationRate", Plan.ORIENTATION_RATE), ("baseVelocity", Plan.VELOCITY),
                           ("baseAcceleration", Plan.ACCELERATION)):
      target = getattr(diagnostic, field)
      target.t = ModelConstants.T_IDXS
      target.x, target.y, target.z = plan[0, :, indices].T.tolist()
  for field, key in (("laneBoundaryStds", "lane_lines_stds"), ("roadBoundaryStds", "road_edges_stds")):
    values = np.asarray(model_output.get(key, []))
    if values.ndim == 4:
      setattr(diagnostic, field, values[0, :, :, 0].reshape(-1).tolist())


class LaneCenteringModelAdapter:
  def __init__(self, mode="absolute"):
    self.controller = LaneCenteringController(mode)
    self.previous_base_action = log.ModelDataV2.Action()
    self.previous_selected_action = log.ModelDataV2.Action()
    self.last_timestamp_eof = None
    self.last_log_key = None
    self.last_log_timestamp = None

  def update(self, model_output, action_from_model, sm, calibration_seen, lane_change_state,
             timestamp_eof, v_ego, lat_action_t, long_action_t):
    timestamp_valid = self.last_timestamp_eof is None or timestamp_eof > self.last_timestamp_eof
    if self.last_timestamp_eof is None or not timestamp_valid:
      frame_dt = DT_MDL
    else:
      frame_dt = (timestamp_eof - self.last_timestamp_eof) * 1e-9
    if timestamp_valid:
      self.last_timestamp_eof = timestamp_eof
    # Preserve native longitudinal action/history. Both lateral histories use
    # their corresponding plans, as on c3x, including when lane selection abstains.
    raw_base_action = action_from_model(model_output, self.previous_base_action, lat_action_t, long_action_t, v_ego,
                                        lateral_smooth_seconds=0.0)
    base_curvature = _get_lateral_curvature(model_output, self.previous_base_action.desiredCurvature, v_ego, lat_action_t, frame_dt)
    base_action = log.ModelDataV2.Action(
      desiredCurvature=float(base_curvature), desiredAcceleration=raw_base_action.desiredAcceleration,
      shouldStop=raw_base_action.shouldStop,
    )
    inputs = get_lane_centering_input_status(sm, calibration_seen)
    # Both entrypoints include the captured 0.1 s smoothing compensation in
    # inference inputs and this lookahead, in addition to C4 frame/action delay.
    selected_output, status = self.controller.update(
      model_output, v_ego, sm['carControl'].currentCurvature, lat_action_t, frame_dt if timestamp_valid else np.nan,
      base_action.desiredCurvature, self.previous_selected_action.desiredCurvature,
      sm['carControl'].latActive, inputs.ready, sm['carState'].leftBlinker, sm['carState'].rightBlinker,
      lane_change_state != log.LaneChangeState.off,
    )

    # Match the action to the same trajectory published for NNLC preview. Keep
    # one selected filter history through acquisition, release and base-plan fallback.
    previous_curvature = self.previous_selected_action.desiredCurvature
    raw_curvature = _get_lateral_curvature(selected_output, previous_curvature, v_ego, lat_action_t,
                                           frame_dt, smooth_seconds=0.0)
    requested_curvature = _get_lateral_curvature(selected_output, previous_curvature, v_ego, lat_action_t, frame_dt)
    # Candidate dynamics constrain samples within a trajectory. Independently
    # bound transitions between trajectories, including unsupported fallback.
    speed_squared = max(v_ego * v_ego, 1.0) if np.isfinite(v_ego) else 1.0
    max_step = MAX_LATERAL_JERK * min(frame_dt, 0.10) / speed_squared
    curvature = float(np.clip(requested_curvature, previous_curvature - max_step, previous_curvature + max_step))
    action = log.ModelDataV2.Action(
      desiredCurvature=float(curvature), desiredAcceleration=base_action.desiredAcceleration,
      shouldStop=base_action.shouldStop,
    )
    status = replace(status, curvature_correction=float(curvature - base_action.desiredCurvature),
                     requested_lateral_jerk=float(abs(curvature - self.previous_selected_action.desiredCurvature) *
                                                  speed_squared / frame_dt),
                     base_curvature=float(base_curvature), selected_curvature_raw=float(raw_curvature),
                     selected_curvature=curvature, action_limited=bool(curvature != requested_curvature),
                     frame_dt=float(frame_dt), model_timestamp_eof=int(timestamp_eof),
                     native_curvature=float(raw_base_action.desiredCurvature),
                     requested_curvature=float(requested_curvature), action_time=float(lat_action_t))

    self.previous_base_action = base_action
    self.previous_selected_action = action
    key = (status.state, status.source, status.reason, status.line_gate, status.edge_gate, status.policy_gate)
    log_due = self.last_log_timestamp is None or timestamp_eof < self.last_log_timestamp or timestamp_eof - self.last_log_timestamp >= 1_000_000_000
    if log_due and (key != self.last_log_key or status.authority > 0.0):
      cloudlog.info("lane centering mode=%s state=%s source=%s reason=%s authority=%.3f path_weight=%.3f correction=%.5f " +
                    "geometry_horizon=%.1f convergence_distance=%.1f convergence_time=%.2f horizon_limited=%s " +
                    "policy_disagreement=%.3f line_gate=%s edge_gate=%s policy_gate=%s",
                    self.controller.mode, status.state, status.source, status.reason,
                    status.authority, status.path_weight, status.curvature_correction,
                    status.geometry_horizon, status.convergence_distance, status.convergence_time, status.horizon_limited,
                    status.policy_disagreement, status.line_gate, status.edge_gate, status.policy_gate)
      self.last_log_key = key
      self.last_log_timestamp = timestamp_eof
    return selected_output, action, status

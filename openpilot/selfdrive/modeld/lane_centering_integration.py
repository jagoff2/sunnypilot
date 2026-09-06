"""Adapt the C3X lane trajectory selector to C4 model action heads."""
import logging
from dataclasses import replace
from types import SimpleNamespace

from openpilot.cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import get_curvature_from_plan, smooth_value
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan
from openpilot.selfdrive.modeld.lane_centering import (
  ACTION_SMOOTH_SECONDS, LaneCenteringController, get_lane_centering_input_status,
)


logger = logging.getLogger(__name__)


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
    self.last_log_key = None

  def update(self, model_output, action_from_model, sm, calibration_seen, lane_change_state,
             timestamp_eof, v_ego, lat_action_t, long_action_t):
    # Keep the native model's acceleration, stopping rules, and base action
    # history independent of the selected lateral trajectory.
    raw_base_action = action_from_model(model_output, self.previous_base_action, lat_action_t, long_action_t, v_ego,
                                        lateral_smooth_seconds=0.0)
    base_curvature = (smooth_value(raw_base_action.desiredCurvature, self.previous_base_action.desiredCurvature, ACTION_SMOOTH_SECONDS)
                      if v_ego > 0.3 else self.previous_base_action.desiredCurvature)
    base_action = log.ModelDataV2.Action(
      desiredCurvature=float(base_curvature), desiredAcceleration=raw_base_action.desiredAcceleration,
      shouldStop=raw_base_action.shouldStop,
    )
    if self.last_timestamp_eof is None or timestamp_eof <= self.last_timestamp_eof:
      frame_dt = DT_MDL
    else:
      frame_dt = (timestamp_eof - self.last_timestamp_eof) * 1e-9
    self.last_timestamp_eof = timestamp_eof

    inputs = get_lane_centering_input_status(sm, calibration_seen)
    # Both entrypoints include the captured 0.1 s smoothing compensation in
    # inference inputs and this lookahead, in addition to C4 frame/action delay.
    selected_output, status = self.controller.update(
      model_output, v_ego, sm['carControl'].currentCurvature, lat_action_t, frame_dt,
      base_action.desiredCurvature, self.previous_selected_action.desiredCurvature,
      sm['carControl'].latActive, inputs.ready, sm['carState'].leftBlinker, sm['carState'].rightBlinker,
      lane_change_state != log.LaneChangeState.off,
    )

    curvature = raw_base_action.desiredCurvature
    if selected_output is not model_output:
      plan = selected_output['plan'][0]
      curvature = get_curvature_from_plan(
        plan[:, Plan.T_FROM_CURRENT_EULER][:, 2], plan[:, Plan.ORIENTATION_RATE][:, 2],
        ModelConstants.T_IDXS, v_ego, lat_action_t,
      )
      base_plan = model_output['plan'][0]
      base_plan_curvature = get_curvature_from_plan(
        base_plan[:, Plan.T_FROM_CURRENT_EULER][:, 2], base_plan[:, Plan.ORIENTATION_RATE][:, 2],
        ModelConstants.T_IDXS, v_ego, lat_action_t,
      )
      # The captured selector blends against the base plan. Modern action heads
      # can disagree with that plan, so retain their contribution as authority
      # ramps to/from zero instead of switching baselines on the first tick.
      curvature += (1.0 - status.path_weight) * (raw_base_action.desiredCurvature - base_plan_curvature)
    if v_ego > 0.3:
      curvature = smooth_value(curvature, self.previous_selected_action.desiredCurvature, ACTION_SMOOTH_SECONDS)
    else:
      curvature = self.previous_selected_action.desiredCurvature
    # Filter the selected history on every tick, including the final return to
    # an action head. Returning base_action directly would switch filter states.
    action = log.ModelDataV2.Action(
      desiredCurvature=float(curvature), desiredAcceleration=base_action.desiredAcceleration,
      shouldStop=base_action.shouldStop,
    )
    status = replace(status, curvature_correction=float(curvature - base_action.desiredCurvature),
                     requested_lateral_jerk=float(abs(curvature - self.previous_selected_action.desiredCurvature) *
                                                  max(v_ego * v_ego, 1.0) / self.controller.frame_dt))

    self.previous_base_action = base_action
    self.previous_selected_action = action
    key = (status.state, status.source, status.reason, status.line_gate, status.edge_gate, status.policy_gate)
    if key != self.last_log_key:
      logger.info("lane centering mode=%s state=%s source=%s reason=%s authority=%.3f path_weight=%.3f correction=%.5f",
                  self.controller.mode, status.state, status.source, status.reason,
                  status.authority, status.path_weight, status.curvature_correction)
      self.last_log_key = key
    return selected_output, action, status

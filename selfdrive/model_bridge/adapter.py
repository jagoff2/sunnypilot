from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from cereal import log
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.fill_model_msg import fill_xyz_poly

from .ipc import FrameMetadata


@dataclass
class DiffusionDriveResponse:
  seq: int
  mono_time_ns: int
  horizon_s: float
  trajectory: np.ndarray
  confidence: float
  velocity_ref: float
  meta: dict[str, Any]
  model_execution_time: float | None
  status: str | None
  valid: bool
  lane_lines: list[np.ndarray] = field(default_factory=list)
  lane_line_probs: list[float] = field(default_factory=list)
  road_edges: list[np.ndarray] = field(default_factory=list)
  road_edge_stds: list[float] = field(default_factory=list)
  leads: list[dict[str, float]] = field(default_factory=list)

  @classmethod
  def from_dict(cls, payload: dict[str, Any]) -> "DiffusionDriveResponse":
    seq = int(payload.get("seq", 0))
    mono_time_ns = int(payload.get("mono_time_ns", 0))
    horizon_s = float(payload.get("horizon_s", payload.get("horizon", 0.0)))
    confidence = float(payload.get("confidence", 0.0))
    velocity_ref = float(payload.get("velocity", payload.get("v_ref", 0.0)))
    status = payload.get("status")

    traj_key = "trajectory"
    if traj_key not in payload and "trajectory_world" in payload:
      traj_key = "trajectory_world"
    raw_traj = payload.get(traj_key, [])

    if isinstance(raw_traj, dict):
      raw_traj = [raw_traj]

    points: list[tuple[float, float, float, float]] = []
    for entry in raw_traj:
      if not isinstance(entry, dict):
        continue
      try:
        t = float(entry.get("t", entry.get("time", 0.0)))
        x = float(entry.get("x", 0.0))
        y = float(entry.get("y", 0.0))
        z = float(entry.get("z", 0.0))
      except (TypeError, ValueError):
        continue
      points.append((t, x, y, z))

    if points:
      points.sort(key=lambda p: p[0])
      times = np.array([p[0] for p in points], dtype=np.float32)
      unique_times, unique_indices = np.unique(times, return_index=True)
      points_arr = np.array(points, dtype=np.float32)[unique_indices]
      trajectory = np.column_stack((unique_times, points_arr[:, 1:]))
    else:
      trajectory = np.zeros((0, 4), dtype=np.float32)

    meta = payload.get("meta")
    if not isinstance(meta, dict):
      meta = {}

    def _parse_path(data: Any) -> np.ndarray:
      if isinstance(data, dict):
        xs = data.get("x")
        ys = data.get("y")
        if isinstance(xs, Sequence) and isinstance(ys, Sequence) and len(xs) == len(ys) and len(xs) > 0:
          arr = np.column_stack((np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)))
        else:
          points_raw = data.get("points") or data.get("coords")
          arr = []
          if isinstance(points_raw, Sequence):
            for point in points_raw:
              if isinstance(point, dict):
                try:
                  px = float(point.get("x", 0.0))
                  py = float(point.get("y", 0.0))
                except (TypeError, ValueError):
                  continue
                arr.append((px, py))
          arr = np.asarray(arr, dtype=np.float32) if arr else np.zeros((0, 2), dtype=np.float32)
      elif isinstance(data, Sequence):
        arr = []
        for point in data:
          if isinstance(point, dict):
            try:
              px = float(point.get("x", 0.0))
              py = float(point.get("y", 0.0))
            except (TypeError, ValueError):
              continue
            arr.append((px, py))
        arr = np.asarray(arr, dtype=np.float32) if arr else np.zeros((0, 2), dtype=np.float32)
      else:
        arr = np.zeros((0, 2), dtype=np.float32)
      if arr.size:
        arr = arr[np.argsort(arr[:, 0])]
      return arr

    lane_lines_payload = payload.get("lane_lines")
    lane_lines: list[np.ndarray] = []
    if isinstance(lane_lines_payload, Sequence):
      for entry in lane_lines_payload:
        parsed = _parse_path(entry)
        if parsed.size:
          lane_lines.append(parsed)

    lane_probs_payload = payload.get("lane_line_probs")
    lane_line_probs: list[float] = []
    if isinstance(lane_probs_payload, Sequence):
      for value in lane_probs_payload:
        try:
          lane_line_probs.append(float(value))
        except (TypeError, ValueError):
          lane_line_probs.append(0.0)

    road_edges_payload = payload.get("road_edges")
    road_edges: list[np.ndarray] = []
    if isinstance(road_edges_payload, Sequence):
      for entry in road_edges_payload:
        parsed = _parse_path(entry)
        if parsed.size:
          road_edges.append(parsed)

    road_edge_stds_payload = payload.get("road_edge_stds")
    road_edge_stds: list[float] = []
    if isinstance(road_edge_stds_payload, Sequence):
      for value in road_edge_stds_payload:
        try:
          road_edge_stds.append(float(value))
        except (TypeError, ValueError):
          road_edge_stds.append(1.0)

    leads_payload = payload.get("leads")
    leads: list[dict[str, float]] = []
    if isinstance(leads_payload, Sequence):
      for lead in leads_payload:
        if not isinstance(lead, dict):
          continue
        try:
          x = float(lead.get("x", 0.0))
          y = float(lead.get("y", 0.0))
          prob = float(lead.get("prob", lead.get("score", 0.0)))
          heading = float(lead.get("heading", 0.0))
          length = float(lead.get("length", 4.5))
          width = float(lead.get("width", 1.8))
        except (TypeError, ValueError):
          continue
        vx = float(lead.get("vx", 0.0)) if isinstance(lead.get("vx"), (float, int)) else 0.0
        vy = float(lead.get("vy", 0.0)) if isinstance(lead.get("vy"), (float, int)) else 0.0
        ax = float(lead.get("ax", 0.0)) if isinstance(lead.get("ax"), (float, int)) else 0.0
        ay = float(lead.get("ay", 0.0)) if isinstance(lead.get("ay"), (float, int)) else 0.0
        leads.append({
          "x": x,
          "y": y,
          "prob": np.clip(prob, 0.0, 1.0),
          "heading": heading,
          "length": length,
          "width": width,
          "vx": vx,
          "vy": vy,
          "ax": ax,
          "ay": ay,
        })

    model_execution_time = payload.get("model_execution_time", payload.get("model_execution_time_s"))
    if model_execution_time is not None:
      try:
        model_execution_time = float(model_execution_time)
      except (TypeError, ValueError):
        model_execution_time = None

    valid = bool(len(trajectory) >= 2)

    return cls(
      seq=seq,
      mono_time_ns=mono_time_ns,
      horizon_s=horizon_s,
      trajectory=trajectory,
      confidence=confidence,
      velocity_ref=velocity_ref,
      meta=meta,
      model_execution_time=model_execution_time,
      status=status if isinstance(status, str) else None,
      valid=valid,
      lane_lines=lane_lines,
      lane_line_probs=lane_line_probs,
      road_edges=road_edges,
      road_edge_stds=road_edge_stds,
      leads=leads,
    )


@dataclass
class BridgeOutputs:
  model_msg: log.EventBuilder
  driving_msg: log.EventBuilder
  model_sp_msg: log.EventBuilder
  camera_odometry_msg: log.EventBuilder
  valid: bool


class DiffusionDriveAdapter:
  def __init__(self) -> None:
    self._t_samples = np.asarray(ModelConstants.T_IDXS, dtype=np.float32)

  def _resample(self, response: DiffusionDriveResponse) -> dict[str, np.ndarray]:
    if response.trajectory.size == 0:
      zero = np.zeros_like(self._t_samples)
      return {
        "t": self._t_samples,
        "x": zero,
        "y": zero,
        "z": zero,
      }

    times = response.trajectory[:, 0]
    x = response.trajectory[:, 1]
    y = response.trajectory[:, 2]
    if response.trajectory.shape[1] > 3:
      z = response.trajectory[:, 3]
    else:
      z = np.zeros_like(x)

    x_interp = np.interp(self._t_samples, times, x, left=x[0], right=x[-1])
    y_interp = np.interp(self._t_samples, times, y, left=y[0], right=y[-1])
    z_interp = np.interp(self._t_samples, times, z, left=z[0], right=z[-1])
    return {"t": self._t_samples, "x": x_interp, "y": y_interp, "z": z_interp}

  def build_messages(self, response: DiffusionDriveResponse, frame: FrameMetadata, frame_extra: FrameMetadata,
                     planner_frame_id: int, frame_drop_ratio: float, execution_time: float,
                     live_calib_seen: bool) -> BridgeOutputs:
    resampled = self._resample(response)
    t = resampled["t"]
    x = resampled["x"]
    y = resampled["y"]
    z = resampled["z"]

    # Derivatives with respect to time
    vx = np.gradient(x, t, edge_order=2)
    vy = np.gradient(y, t, edge_order=2)
    vz = np.gradient(z, t, edge_order=2)
    ax = np.gradient(vx, t, edge_order=2)
    ay = np.gradient(vy, t, edge_order=2)
    az = np.gradient(vz, t, edge_order=2)

    speed = np.clip(np.hypot(vx, vy), 0.0, None)
    speed0 = float(speed[0]) if speed.size else 0.0
    accel_along = np.gradient(speed, t, edge_order=2)

    curvature = np.zeros_like(t)
    denom = np.power(np.clip(vx * vx + vy * vy, 1e-6, None), 1.5)
    curvature = np.nan_to_num((vx * ay - vy * ax) / denom)

    heading = np.unwrap(np.arctan2(vy, vx))
    heading_rate = np.gradient(heading, t, edge_order=2)

    velocity_ref = response.velocity_ref if response.velocity_ref != 0.0 else speed0

    desired_accel = float(accel_along[0]) if accel_along.size else 0.0
    if not np.isfinite(desired_accel):
      desired_accel = 0.0
    if abs(desired_accel) < 1e-3 and velocity_ref:
      desired_accel = float(np.clip((velocity_ref - speed0) * 0.5, -5.0, 5.0))
    desired_curvature = float(curvature[0]) if curvature.size else 0.0

    meta_stop = response.meta.get('should_stop') if response.meta else None
    if meta_stop is None and response.meta:
      meta_stop = response.meta.get('shouldStop', response.meta.get('stop'))
    should_stop = bool(meta_stop) if meta_stop is not None else False

    valid = response.valid and live_calib_seen

    model_msg = log.Event.new_message('modelV2')
    driving_msg = log.Event.new_message('drivingModelData')
    model_sp_msg = log.Event.new_message('modelDataV2SP')
    camera_odometry_msg = log.Event.new_message('cameraOdometry')

    model_msg.valid = valid
    driving_msg.valid = valid
    model_sp_msg.valid = valid
    camera_odometry_msg.valid = False

    frame_age = planner_frame_id - frame.frame_id if planner_frame_id > frame.frame_id else 0
    frame_drop_perc = frame_drop_ratio * 100.0
    model_execution_time = response.model_execution_time if response.model_execution_time is not None else execution_time

    # Populate drivingModelData
    driving = driving_msg.drivingModelData
    driving.frameId = frame.frame_id
    driving.frameIdExtra = frame_extra.frame_id
    driving.frameDropPerc = frame_drop_perc
    driving.modelExecutionTime = model_execution_time
    driving.action.desiredCurvature = desired_curvature
    driving.action.desiredAcceleration = desired_accel
    driving.action.shouldStop = should_stop

    fill_xyz_poly(driving.path, ModelConstants.POLY_PATH_DEGREE, x, y, z)
    driving.meta.laneChangeState = log.LaneChangeState.off
    lane_direction = log.LaneChangeDirection.none
    if curvature.size:
      if curvature[-1] > 0.02:
        lane_direction = log.LaneChangeDirection.left
      elif curvature[-1] < -0.02:
        lane_direction = log.LaneChangeDirection.right
    driving.meta.laneChangeDirection = lane_direction
    def _interpolate_path(path: np.ndarray) -> np.ndarray:
      if path.shape[0] < 2:
        return np.zeros_like(x)
      base_x = path[:, 0]
      base_y = path[:, 1]
      return np.interp(x, base_x, base_y, left=base_y[0], right=base_y[-1])

    lane_arrays = [arr.copy() for arr in response.lane_lines]
    lane_probs = [float(np.clip(prob, 0.0, 1.0)) for prob in response.lane_line_probs]
    lane_stds = [float(np.clip(np.std(arr[:, 1]) if arr.shape[0] else 1.0, 0.1, 3.0)) for arr in lane_arrays]
    while len(lane_arrays) < 4:
      lane_arrays.append(np.zeros((0, 2), dtype=np.float32))
    while len(lane_probs) < 4:
      lane_probs.append(0.0)
    while len(lane_stds) < 4:
      lane_stds.append(1.0)

    model.init('laneLines', 4)
    for idx, lane_line in enumerate(model.laneLines):
      lane_line.t = t.tolist()
      lane_line.x = x.tolist()
      lane_values = _interpolate_path(lane_arrays[idx]) if lane_arrays[idx].shape[0] >= 2 else np.zeros_like(x)
      lane_line.y = lane_values.tolist()
      lane_line.z = np.zeros_like(x).tolist()
    model.laneLineProbs = lane_probs[:4]
    model.laneLineStds = lane_stds[:4]

    road_arrays = [arr.copy() for arr in response.road_edges]
    road_stds = [float(std) for std in response.road_edge_stds]
    while len(road_arrays) < 2:
      road_arrays.append(np.zeros((0, 2), dtype=np.float32))
    while len(road_stds) < 2:
      road_stds.append(1.0)

    model.init('roadEdges', 2)
    for idx, road_edge in enumerate(model.roadEdges):
      road_edge.t = t.tolist()
      road_edge.x = x.tolist()
      road_values = _interpolate_path(road_arrays[idx]) if road_arrays[idx].shape[0] >= 2 else np.zeros_like(x)
      road_edge.y = road_values.tolist()
      road_edge.z = np.zeros_like(x).tolist()
    model.roadEdgeStds = road_stds[:2]

    def _value_at_zero(path: np.ndarray) -> float:
      if path.shape[0] == 0:
        return 0.0
      return float(np.interp(0.0, path[:, 0], path[:, 1], left=path[0, 1], right=path[-1, 1]))

    left_offset = _value_at_zero(road_arrays[0]) if road_arrays else 0.0
    right_offset = _value_at_zero(road_arrays[1]) if len(road_arrays) > 1 else 0.0
    driving.laneLineMeta.leftY = left_offset
    driving.laneLineMeta.rightY = right_offset
    driving.laneLineMeta.leftProb = lane_probs[0] if lane_probs else 0.0
    driving.laneLineMeta.rightProb = lane_probs[-1] if lane_probs else 0.0

    # Populate modelV2
    model = model_msg.modelV2
    model.valid = valid
    model.frameId = frame.frame_id
    model.frameIdExtra = frame_extra.frame_id
    model.frameAge = frame_age
    model.frameDropPerc = frame_drop_perc
    model.timestampEof = frame.timestamp_eof
    model.modelExecutionTime = model_execution_time

    model.position.t = t.tolist()
    model.position.x = x.tolist()
    model.position.y = y.tolist()
    model.position.z = z.tolist()

    model.velocity.t = t.tolist()
    model.velocity.x = vx.tolist()
    model.velocity.y = vy.tolist()
    model.velocity.z = vz.tolist()

    model.acceleration.t = t.tolist()
    model.acceleration.x = ax.tolist()
    model.acceleration.y = ay.tolist()
    model.acceleration.z = az.tolist()

    model.orientation.t = t.tolist()
    model.orientation.x = np.zeros_like(t).tolist()
    model.orientation.y = np.zeros_like(t).tolist()
    model.orientation.z = heading.tolist()

    model.orientationRate.t = t.tolist()
    model.orientationRate.x = np.zeros_like(t).tolist()
    model.orientationRate.y = np.zeros_like(t).tolist()
    model.orientationRate.z = heading_rate.tolist()

    model.action.desiredCurvature = desired_curvature
    model.action.desiredAcceleration = desired_accel
    model.action.shouldStop = should_stop

    lead_ts = np.asarray(ModelConstants.LEAD_T_IDXS, dtype=np.float32)
    model.init('leadsV3', 3)
    for lead in model.leadsV3:
      lead.t = lead_ts.tolist()
      lead.x = [0.0] * len(lead_ts)
      lead.y = [0.0] * len(lead_ts)
      lead.v = [0.0] * len(lead_ts)
      lead.a = [0.0] * len(lead_ts)
      lead.xStd = [1.0] * len(lead_ts)
      lead.yStd = [1.0] * len(lead_ts)
      lead.vStd = [1.0] * len(lead_ts)
      lead.aStd = [1.0] * len(lead_ts)
      lead.prob = 0.0
      lead.probTime = 0.0

    for idx, lead_data in enumerate(response.leads[:3]):
      lead = model.leadsV3[idx]
      x0_lead = float(lead_data.get("x", 0.0))
      y0_lead = float(lead_data.get("y", 0.0))
      vx_lead = float(lead_data.get("vx", 0.0))
      vy_lead = float(lead_data.get("vy", 0.0))
      ax_lead = float(lead_data.get("ax", 0.0))
      ay_lead = float(lead_data.get("ay", 0.0))
      width_lead = float(abs(lead_data.get("width", 1.8)))
      length_lead = float(abs(lead_data.get("length", 4.5)))
      lead.x = (x0_lead + vx_lead * lead_ts + 0.5 * ax_lead * (lead_ts ** 2)).tolist()
      lead.y = (y0_lead + vy_lead * lead_ts + 0.5 * ay_lead * (lead_ts ** 2)).tolist()
      lead.v = (vx_lead + ax_lead * lead_ts).tolist()
      lead.a = [ax_lead] * len(lead_ts)
      lead.xStd = [max(0.3, width_lead * 0.25)] * len(lead_ts)
      lead.yStd = [max(0.3, length_lead * 0.25)] * len(lead_ts)
      lead.vStd = [1.0] * len(lead_ts)
      lead.aStd = [1.0] * len(lead_ts)
      lead.prob = float(np.clip(lead_data.get("prob", 0.0), 0.0, 1.0))
      lead.probTime = float(lead_ts[min(len(lead_ts) - 1, np.argmax(np.abs(lead.v)))]) if lead.v else 0.0

    meta = model.meta
    engaged_prob = None
    if response.meta:
      engaged_prob = response.meta.get('engaged_prob', response.meta.get('engagedProb'))
    if engaged_prob is None:
      engaged_prob = np.clip(response.confidence, 0.0, 1.0)
    meta.engagedProb = float(engaged_prob)
    meta.desireState = [0.0] * ModelConstants.DESIRE_LEN
    meta.desirePrediction = [0.0] * ModelConstants.DESIRE_LEN
    meta.hardBrakePredicted = False
    meta.laneChangeState = log.LaneChangeState.off
    meta.laneChangeDirection = lane_direction

    disengage = meta.init('disengagePredictions')
    disengage.t = list(ModelConstants.META_T_IDXS)
    disengage.brakeDisengageProbs = [0.0] * len(ModelConstants.META_T_IDXS)
    disengage.gasDisengageProbs = [0.0] * len(ModelConstants.META_T_IDXS)
    disengage.steerOverrideProbs = [0.0] * len(ModelConstants.META_T_IDXS)
    disengage.brake3MetersPerSecondSquaredProbs = [0.0] * len(ModelConstants.META_T_IDXS)
    disengage.brake4MetersPerSecondSquaredProbs = [0.0] * len(ModelConstants.META_T_IDXS)
    disengage.brake5MetersPerSecondSquaredProbs = [0.0] * len(ModelConstants.META_T_IDXS)
    disengage.gasPressProbs = [0.0] * len(ModelConstants.META_T_IDXS)
    disengage.brakePressProbs = [0.0] * len(ModelConstants.META_T_IDXS)

    if response.confidence >= 0.66:
      model.confidence = log.ModelDataV2.ConfidenceClass.green
    elif response.confidence >= 0.33:
      model.confidence = log.ModelDataV2.ConfidenceClass.yellow
    else:
      model.confidence = log.ModelDataV2.ConfidenceClass.red

    camera = camera_odometry_msg.cameraOdometry
    camera.frameId = frame.frame_id
    camera.timestampEof = frame.timestamp_eof
    if t.size >= 2:
      camera_odometry_msg.valid = valid
      trans_vec = [float(x[1] - x[0]), float(y[1] - y[0]), float(z[1] - z[0])]
      camera.trans = trans_vec
      camera.rot = [0.0, 0.0, float(heading[1] - heading[0])]
      speed_std = float(np.std(speed)) if speed.size else 0.1
      rot_std = float(np.clip(abs(heading_rate[0]), 0.01, 0.5)) if heading_rate.size else 0.1
      camera.transStd = [max(0.05, speed_std * 0.1)] * 3
      camera.rotStd = [0.1, 0.1, rot_std]
    else:
      camera_odometry_msg.valid = False
      camera.trans = [0.0, 0.0, 0.0]
      camera.rot = [0.0, 0.0, 0.0]
      camera.transStd = [1.0, 1.0, 1.0]
      camera.rotStd = [1.0, 1.0, 1.0]
    lane_center_offset = 0.5 * (left_offset + right_offset)
    camera.wideFromDeviceEuler = [0.0, 0.0, float(heading[0] if heading.size else 0.0)]
    camera.wideFromDeviceEulerStd = [0.1, 0.1, 0.1]
    camera.roadTransformTrans = [0.0, float(lane_center_offset), 0.0]
    camera.roadTransformTransStd = [0.2, 0.2, 0.2]

    lane_turn = log.ModelDataV2SP.TurnDirection.none
    if curvature.size:
      if curvature[-1] > 0.02:
        lane_turn = log.ModelDataV2SP.TurnDirection.left
      elif curvature[-1] < -0.02:
        lane_turn = log.ModelDataV2SP.TurnDirection.right
    model_sp_msg.modelDataV2SP.laneTurnDirection = lane_turn

    return BridgeOutputs(
      model_msg=model_msg,
      driving_msg=driving_msg,
      model_sp_msg=model_sp_msg,
      camera_odometry_msg=camera_odometry_msg,
      valid=valid,
    )


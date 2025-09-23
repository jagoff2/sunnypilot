from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    driving.meta.laneChangeDirection = log.LaneChangeDirection.none
    driving.laneLineMeta.leftY = 0.0
    driving.laneLineMeta.rightY = 0.0
    driving.laneLineMeta.leftProb = 0.0
    driving.laneLineMeta.rightProb = 0.0

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

    model.init('laneLines', 4)
    for lane_line in model.laneLines:
      lane_line.t = t.tolist()
      lane_line.x = np.zeros_like(t).tolist()
      lane_line.y = np.zeros_like(t).tolist()
      lane_line.z = np.zeros_like(t).tolist()
    model.laneLineProbs = [0.0] * 4
    model.laneLineStds = [1.0] * 4

    model.init('roadEdges', 2)
    for road_edge in model.roadEdges:
      road_edge.t = t.tolist()
      road_edge.x = np.zeros_like(t).tolist()
      road_edge.y = np.zeros_like(t).tolist()
      road_edge.z = np.zeros_like(t).tolist()
    model.roadEdgeStds = [1.0, 1.0]

    model.init('leadsV3', 3)
    for lead in model.leadsV3:
      lead.t = list(ModelConstants.LEAD_T_IDXS)
      lead.x = [0.0] * len(ModelConstants.LEAD_T_IDXS)
      lead.y = [0.0] * len(ModelConstants.LEAD_T_IDXS)
      lead.v = [0.0] * len(ModelConstants.LEAD_T_IDXS)
      lead.a = [0.0] * len(ModelConstants.LEAD_T_IDXS)
      lead.xStd = [1.0] * len(ModelConstants.LEAD_T_IDXS)
      lead.yStd = [1.0] * len(ModelConstants.LEAD_T_IDXS)
      lead.vStd = [1.0] * len(ModelConstants.LEAD_T_IDXS)
      lead.aStd = [1.0] * len(ModelConstants.LEAD_T_IDXS)
      lead.prob = 0.0
      lead.probTime = 0.0

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
    meta.laneChangeDirection = log.LaneChangeDirection.none

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
    camera.trans = [0.0, 0.0, 0.0]
    camera.rot = [0.0, 0.0, 0.0]
    camera.transStd = [1.0, 1.0, 1.0]
    camera.rotStd = [1.0, 1.0, 1.0]
    camera.wideFromDeviceEuler = [0.0, 0.0, 0.0]
    camera.wideFromDeviceEulerStd = [1.0, 1.0, 1.0]
    camera.roadTransformTrans = [0.0, 0.0, 0.0]
    camera.roadTransformTransStd = [1.0, 1.0, 1.0]

    model_sp_msg.modelDataV2SP.laneTurnDirection = log.ModelDataV2SP.TurnDirection.none

    return BridgeOutputs(
      model_msg=model_msg,
      driving_msg=driving_msg,
      model_sp_msg=model_sp_msg,
      camera_odometry_msg=camera_odometry_msg,
      valid=valid,
    )


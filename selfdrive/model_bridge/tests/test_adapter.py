from __future__ import annotations

import numpy as np

from openpilot.selfdrive.model_bridge.adapter import DiffusionDriveAdapter, DiffusionDriveResponse
from openpilot.selfdrive.model_bridge.ipc import FrameMetadata
from openpilot.selfdrive.modeld.constants import ModelConstants


def build_sample_response() -> DiffusionDriveResponse:
  horizon = 6.0
  times = np.linspace(0.0, horizon, num=6)
  trajectory = [{"t": float(t), "x": float(1.5 * t), "y": float(0.1 * t * t)} for t in times]
  payload = {
    "seq": 42,
    "mono_time_ns": 123456789,
    "horizon_s": horizon,
    "trajectory": trajectory,
    "confidence": 0.75,
    "velocity": 4.5,
    "lane_lines": [
      {"x": [0.0, 3.0, 6.0], "y": [3.5, 3.4, 3.3]},
      {"x": [0.0, 3.0, 6.0], "y": [1.0, 1.1, 1.2]},
      {"x": [0.0, 3.0, 6.0], "y": [-1.0, -1.1, -1.2]},
      {"x": [0.0, 3.0, 6.0], "y": [-3.5, -3.4, -3.3]},
    ],
    "lane_line_probs": [0.8, 0.9, 0.85, 0.8],
    "road_edges": [
      {"x": [0.0, 3.0, 6.0], "y": [3.8, 3.7, 3.6]},
      {"x": [0.0, 3.0, 6.0], "y": [-3.8, -3.7, -3.6]},
    ],
    "road_edge_stds": [0.35, 0.35],
    "leads": [
      {"x": 20.0, "y": 0.5, "prob": 0.7, "length": 4.5, "width": 1.8},
      {"x": 35.0, "y": -0.4, "prob": 0.5, "length": 4.7, "width": 2.0},
    ],
    "meta": {"should_stop": False},
  }
  return DiffusionDriveResponse.from_dict(payload)


def test_adapter_builds_messages() -> None:
  adapter = DiffusionDriveAdapter()
  response = build_sample_response()
  frame = FrameMetadata(frame_id=100, timestamp_sof=1_000_000, timestamp_eof=1_020_000)

  outputs = adapter.build_messages(response, frame, frame, planner_frame_id=102,
                                   frame_drop_ratio=0.1, execution_time=0.05,
                                   live_calib_seen=True)

  assert outputs.valid
  model_msg = outputs.model_msg.modelV2
  driving_msg = outputs.driving_msg.drivingModelData

  assert len(model_msg.position.x) == len(ModelConstants.T_IDXS)
  assert model_msg.frameId == frame.frame_id
  assert driving_msg.frameDropPerc == 0.1 * 100.0
  assert model_msg.modelExecutionTime == 0.05
  assert driving_msg.modelExecutionTime == 0.05

  # Lane line data propagated
  assert any(abs(y) > 0.0 for y in model_msg.laneLines[0].y)
  assert driving_msg.laneLineMeta.leftProb > 0.0
  assert driving_msg.laneLineMeta.rightProb > 0.0

  # Lead vehicles filled in
  assert model_msg.leadsV3[0].prob > 0.0
  assert model_msg.leadsV3[0].x[0] > 0.0

  # Camera odometry is populated when calibration is seen
  assert outputs.camera_odometry_msg.valid
  assert any(abs(v) > 0.0 for v in outputs.camera_odometry_msg.cameraOdometry.trans)

  # Trajectory should be monotonically increasing in x
  x_vals = np.array(model_msg.position.x)
  assert np.all(np.diff(x_vals) >= -1e-3)

  # Confidence should be green for high confidence
  assert model_msg.confidence == model_msg.ConfidenceClass.green


def test_adapter_marks_invalid_without_calibration() -> None:
  adapter = DiffusionDriveAdapter()
  response = build_sample_response()
  frame = FrameMetadata(frame_id=5, timestamp_sof=0, timestamp_eof=100)

  outputs = adapter.build_messages(response, frame, frame, planner_frame_id=5,
                                   frame_drop_ratio=0.0, execution_time=0.1,
                                   live_calib_seen=False)

  assert not outputs.valid
  assert not outputs.model_msg.valid
  assert outputs.driving_msg.valid is False


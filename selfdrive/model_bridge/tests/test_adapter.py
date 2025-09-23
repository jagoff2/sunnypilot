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


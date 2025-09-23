from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from msgq.visionipc import VisionStreamType
from openpilot.common.swaglog import cloudlog
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix


@dataclass
class CalibrationData:
  warp_matrix: np.ndarray
  intrinsics: np.ndarray
  rpy_calib: np.ndarray
  device_type: str
  sensor: str
  valid: bool = False

  def payload(self) -> dict[str, Any]:
    return {
      "device_type": self.device_type,
      "sensor": self.sensor,
      "warp_matrix": self.warp_matrix.tolist(),
      "intrinsics": self.intrinsics.tolist(),
      "rpy_calib": self.rpy_calib.tolist(),
      "valid": self.valid,
    }


class CalibrationManager:
  def __init__(self, stream_type: VisionStreamType):
    self.stream_type = stream_type
    self._data = CalibrationData(
      warp_matrix=np.eye(3, dtype=np.float32),
      intrinsics=np.zeros(9, dtype=np.float32),
      rpy_calib=np.zeros(3, dtype=np.float32),
      device_type="unknown",
      sensor="unknown",
      valid=False,
    )

  @property
  def data(self) -> CalibrationData:
    return self._data

  def update(self, device_type: str, sensor: str, rpy_calib: list[float]) -> None:
    key = (str(device_type), str(sensor))
    try:
      camera = DEVICE_CAMERAS[key]
    except KeyError:
      cloudlog.warning("calibration manager missing camera key %s", key)
      return

    calib_euler = np.asarray(rpy_calib, dtype=np.float32)
    if self.stream_type == VisionStreamType.VISION_STREAM_WIDE_ROAD:
      intrinsics = np.asarray(camera.ecam.intrinsics, dtype=np.float32)
      warp = get_warp_matrix(calib_euler, camera.ecam.intrinsics, True).astype(np.float32)
    else:
      intrinsics = np.asarray(camera.fcam.intrinsics, dtype=np.float32)
      warp = get_warp_matrix(calib_euler, camera.fcam.intrinsics, False).astype(np.float32)

    self._data = CalibrationData(
      warp_matrix=warp,
      intrinsics=intrinsics,
      rpy_calib=calib_euler,
      device_type=str(device_type),
      sensor=str(sensor),
      valid=True,
    )

  def payload(self) -> dict[str, Any]:
    return self._data.payload()


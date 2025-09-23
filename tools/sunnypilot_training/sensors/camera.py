"""Camera sensor utilities matching openpilot windshield mount."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

try:
  import carla  # type: ignore
except ImportError:  # pragma: no cover - imported at runtime on Windows
  carla = None  # type: ignore


AR0231_WIDTH = 1928
AR0231_HEIGHT = 1208
AR0231_FOCAL = 2648.0


@dataclass
class CameraSpecification:
  """Intrinsic/extrinsic specification for the simulated camera."""

  width: int = AR0231_WIDTH
  height: int = AR0231_HEIGHT
  focal: float = AR0231_FOCAL
  location: Tuple[float, float, float] = (1.37, 0.0, 1.22)
  rotation: Tuple[float, float, float] = (-1.0, 0.0, 0.0)
  fps: float = 20.0
  fstop: float = 1.8
  shutter_speed: float = 1.0 / 60.0
  exposure_compensation: float = 0.0
  enable_wide: bool = True
  wide_fov_degrees: float = 120.0

  def field_of_view(self) -> float:
    """Compute the horizontal FOV in degrees from the focal length."""
    return math.degrees(2.0 * math.atan(self.width / (2.0 * self.focal)))

  def intrinsics(self) -> Dict[str, float]:
    """Return the CARLA camera intrinsic parameters."""
    return {
      "image_size_x": str(self.width),
      "image_size_y": str(self.height),
      "fov": str(self.field_of_view()),
      "fstop": str(self.fstop),
      "shutter_speed": str(self.shutter_speed),
      "exposure_mode": "manual",
      "iso": "100",
      "gamma": "2.2",
    }


def _require_carla() -> None:
  if carla is None:
    raise ImportError(
      "carla module not available. Ensure CARLA 0.10.0 Python API is installed "
      "and PYTHONPATH is configured before running the collector."
    )


def create_camera_blueprint(blueprint_library: "carla.BlueprintLibrary", spec: CameraSpecification,
                             *, role_name: str = "front") -> "carla.ActorBlueprint":
  """Create a CARLA RGB camera blueprint with openpilot-aligned intrinsics."""
  _require_carla()
  camera_bp = blueprint_library.find("sensor.camera.rgb")
  for key, value in spec.intrinsics().items():
    camera_bp.set_attribute(key, value)
  camera_bp.set_attribute("sensor_tick", f"{1.0 / spec.fps:.6f}")
  camera_bp.set_attribute("motion_blur_intensity", "0.0")
  camera_bp.set_attribute("chromatic_aberration_intensity", "0.0")
  camera_bp.set_attribute("lens_flare_intensity", "0.0")
  camera_bp.set_attribute("shutter_speed", str(spec.shutter_speed))
  camera_bp.set_attribute("fstop", str(spec.fstop))
  camera_bp.set_attribute("enable_postprocess_effects", "false")
  camera_bp.set_attribute("role_name", role_name)
  return camera_bp


def create_wide_camera_blueprint(blueprint_library: "carla.BlueprintLibrary", spec: CameraSpecification,
                                 *, role_name: str = "front_wide") -> "carla.ActorBlueprint":
  """Create a wide FOV camera blueprint that shares the same mounting position."""
  _require_carla()
  camera_bp = blueprint_library.find("sensor.camera.rgb")
  camera_bp.set_attribute("image_size_x", str(spec.width))
  camera_bp.set_attribute("image_size_y", str(spec.height))
  camera_bp.set_attribute("fov", str(spec.wide_fov_degrees))
  camera_bp.set_attribute("sensor_tick", f"{1.0 / spec.fps:.6f}")
  camera_bp.set_attribute("fstop", str(spec.fstop))
  camera_bp.set_attribute("role_name", role_name)
  camera_bp.set_attribute("enable_postprocess_effects", "false")
  return camera_bp


def mount_transform(spec: CameraSpecification) -> "carla.Transform":
  """Return the transform matching the openpilot windshield mount."""
  _require_carla()
  location = carla.Location(*spec.location)
  rotation = carla.Rotation(pitch=spec.rotation[0], yaw=spec.rotation[1], roll=spec.rotation[2])
  return carla.Transform(location, rotation)


def camera_calibration_matrix(spec: CameraSpecification) -> Tuple[float, float, float, float]:
  """Return (fx, fy, cx, cy) calibration parameters."""
  fx = spec.focal
  fy = spec.focal
  cx = spec.width / 2.0
  cy = spec.height / 2.0
  return fx, fy, cx, cy


__all__ = [
  "CameraSpecification",
  "create_camera_blueprint",
  "create_wide_camera_blueprint",
  "mount_transform",
  "camera_calibration_matrix",
]

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
import zmq


@dataclass
class TrajectoryBundle:
  times: np.ndarray
  xyz: np.ndarray
  heading: np.ndarray


def _load_diffusiondrive(repo_root: Path) -> None:
  repo_path = repo_root.expanduser().resolve()
  if not repo_path.exists():
    raise FileNotFoundError(f"DiffusionDrive repository not found at {repo_root}")
  if str(repo_path) not in sys.path:
    sys.path.insert(0, str(repo_path))


def _nv12_to_rgb(buffer: memoryview | bytes, width: int, height: int, stride: int) -> np.ndarray:
  frame = np.frombuffer(buffer, dtype=np.uint8)
  expected = stride * height * 3 // 2
  if frame.size < expected:
    raise ValueError(f"Frame payload too small: {frame.size} < {expected}")

  y_plane = frame[:stride * height].reshape((height, stride))
  uv_plane = frame[stride * height:expected].reshape((height // 2, stride))

  yuv = np.vstack((y_plane[:, :width], uv_plane[:, :width]))
  yuv = np.ascontiguousarray(yuv)
  rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_NV12)
  return rgb


def _apply_warp(image: np.ndarray, warp_matrix: np.ndarray | None) -> np.ndarray:
  if warp_matrix is None:
    return image
  try:
    warped = cv2.warpPerspective(image, warp_matrix, (image.shape[1], image.shape[0]))
    return warped
  except cv2.error:
    return image


def _crop_and_resize(image: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
  target_w, target_h = output_size
  aspect = target_w / target_h
  height, width = image.shape[:2]
  if width <= 0 or height <= 0:
    raise ValueError("Invalid frame dimensions")

  frame_aspect = width / height
  if frame_aspect > aspect:
    new_width = int(height * aspect)
    offset_x = max((width - new_width) // 2, 0)
    cropped = image[:, offset_x:offset_x + new_width]
  else:
    new_height = int(width / aspect)
    offset_y = max((height - new_height) // 2, 0)
    cropped = image[offset_y:offset_y + new_height, :]
  resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)
  return resized


def _build_status_vector(car_state: dict[str, Any] | None) -> torch.Tensor:
  driving_command = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
  if not car_state:
    velocity = torch.zeros(2, dtype=torch.float32)
    acceleration = torch.zeros(2, dtype=torch.float32)
  else:
    v_ego = float(car_state.get("v_ego", 0.0))
    a_ego = float(car_state.get("a_ego", 0.0))
    lateral_v = float(car_state.get("v_ego_lat", 0.0))
    lateral_a = float(car_state.get("a_ego_lat", 0.0))
    velocity = torch.tensor([v_ego, lateral_v], dtype=torch.float32)
    acceleration = torch.tensor([a_ego, lateral_a], dtype=torch.float32)
  return torch.concatenate((driving_command, velocity, acceleration))
class DiffusionDriveModel:
  def __init__(self, repo: Path, weights: Path, device: str, plan_anchor: Path | None,
               dtype: str = "float32") -> None:
    _load_diffusiondrive(repo)

    from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig  # type: ignore
    from navsim.agents.diffusiondrive.transfuser_model_v2 import V2TransfuserModel  # type: ignore

    self.device = torch.device(device)
    self.dtype = torch.float16 if dtype == "float16" else torch.float32

    self.config = TransfuserConfig()
    if plan_anchor is not None:
      self.config.plan_anchor_path = str(plan_anchor)
    else:
      anchor = self._discover_plan_anchor(weights)
      if anchor is not None:
        self.config.plan_anchor_path = str(anchor)

    if not Path(self.config.plan_anchor_path).expanduser().exists():
      raise FileNotFoundError(f"Plan anchor not found: {self.config.plan_anchor_path}")

    self.model = V2TransfuserModel(self.config)
    self.model.eval()
    self.model.to(self.device)

    state_dict = self._load_state_dict(weights)
    missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
    if missing:
      print(f"Warning: missing keys when loading checkpoint: {sorted(missing)}")
    if unexpected:
      print(f"Warning: unexpected keys when loading checkpoint: {sorted(unexpected)}")

    self.interval_s = float(getattr(self.config.trajectory_sampling, "interval_length", 0.5))
    self.horizon_s = float(getattr(self.config.trajectory_sampling, "time_horizon", 4.0))

  def _load_state_dict(self, weights: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(str(weights), map_location=self.device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
      checkpoint = checkpoint["state_dict"]

    state_dict: dict[str, torch.Tensor] = {}
    if isinstance(checkpoint, dict):
      for key, value in checkpoint.items():
        new_key = key
        if new_key.startswith("agent."):
          new_key = new_key[len("agent."):]
        if new_key.startswith("_transfuser_model."):
          new_key = new_key[len("_transfuser_model."):]
        if new_key.startswith("model."):
          new_key = new_key[len("model."):]
        state_dict[new_key] = value.to(self.device)
    else:
      raise ValueError("Unsupported checkpoint format")
    return state_dict

  def _discover_plan_anchor(self, weights: Path) -> Path | None:
    weights_path = Path(weights).expanduser().resolve()
    search_root = weights_path.parent
    for candidate in search_root.rglob("*.npy"):
      if "kmeans" in candidate.name and "traj" in candidate.name:
        return candidate
    return None

  def infer(self, header: dict[str, Any], frame_bytes: memoryview | bytes) -> dict[str, Any]:
    width = int(header.get("width", 0))
    height = int(header.get("height", 0))
    stride = int(header.get("stride", width))
    calibration = header.get("calibration") or {}
    warp_matrix = None
    if calibration:
      warp = calibration.get("warp_matrix")
      if isinstance(warp, Sequence):
        warp_matrix = np.asarray(warp, dtype=np.float32).reshape(3, 3)

    rgb = _nv12_to_rgb(frame_bytes, width, height, stride)
    if warp_matrix is not None:
      rgb = _apply_warp(rgb, warp_matrix)
    rgb = _crop_and_resize(rgb, (1024, 256))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    camera_tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(torch.float32) / 255.0
    camera_tensor = camera_tensor.unsqueeze(0).to(self.device, dtype=self.dtype)

    lidar_shape = (1, 1, self.config.lidar_resolution_height, self.config.lidar_resolution_width)
    lidar_tensor = torch.zeros(lidar_shape, device=self.device, dtype=self.dtype)

    status_vector = _build_status_vector(header.get("car_state"))
    status_tensor = status_vector.unsqueeze(0).to(self.device, dtype=self.dtype)

    with torch.inference_mode():
      outputs = self.model({
        "camera_feature": camera_tensor,
        "lidar_feature": lidar_tensor,
        "status_feature": status_tensor,
      })

    trajectory = outputs["trajectory"].squeeze(0).to(torch.float32).cpu().numpy()
    traj_bundle = self._build_trajectory_bundle(trajectory)

    bev_map = outputs.get("bev_semantic_map")
    agents = outputs.get("agent_states"), outputs.get("agent_labels")
    lane_info = self._extract_lane_geometry(bev_map)
    agents_info = self._extract_agents(*agents)

    confidence = self._compute_confidence(traj_bundle)
    velocity_ref = self._estimate_velocity(traj_bundle)
    should_stop = bool(velocity_ref < 0.2 or confidence < 0.25)

    response: dict[str, Any] = {
      "seq": int(header.get("seq", 0)),
      "mono_time_ns": int(header.get("mono_time_ns", 0)),
      "horizon_s": float(self.horizon_s),
      "trajectory": [
        {"t": float(t), "x": float(x), "y": float(y), "z": float(z), "heading": float(h)}
        for t, (x, y, z), h in zip(traj_bundle.times, traj_bundle.xyz, traj_bundle.heading)
      ],
      "confidence": float(confidence),
      "velocity": float(velocity_ref),
      "status": "ok" if trajectory.shape[0] >= 2 else "invalid",
      "model_execution_time": None,
      "meta": {
        "should_stop": should_stop,
        "lane_width": lane_info.get("lane_width"),
        "lane_samples": lane_info.get("sampled_forward"),
      },
      "lane_lines": lane_info.get("lane_lines"),
      "lane_line_probs": lane_info.get("lane_line_probs"),
      "road_edges": lane_info.get("road_edges"),
      "road_edge_stds": lane_info.get("road_edge_stds"),
      "leads": agents_info,
    }
    return response

  def _build_trajectory_bundle(self, trajectory: np.ndarray) -> TrajectoryBundle:
    if trajectory.ndim != 2 or trajectory.shape[1] < 2:
      raise ValueError("Trajectory tensor has unexpected shape")

    num_points = trajectory.shape[0]
    times = np.linspace(0.0, self.horizon_s, num=num_points, dtype=np.float32)
    xyz = np.zeros((num_points, 3), dtype=np.float32)
    xyz[:, 0:2] = trajectory[..., :2]
    heading = trajectory[..., 2].astype(np.float32)
    return TrajectoryBundle(times=times, xyz=xyz, heading=heading)

  def _estimate_velocity(self, bundle: TrajectoryBundle) -> float:
    if bundle.times.size < 2:
      return 0.0
    dt = float(bundle.times[1] - bundle.times[0])
    if dt <= 0:
      dt = self.interval_s
    dx = bundle.xyz[1, 0] - bundle.xyz[0, 0]
    dy = bundle.xyz[1, 1] - bundle.xyz[0, 1]
    speed = math.hypot(dx, dy) / max(dt, 1e-3)
    return speed

  def _compute_confidence(self, bundle: TrajectoryBundle) -> float:
    if bundle.times.size < 3:
      return 0.0
    dt = np.diff(bundle.times)
    dt[dt <= 0] = self.interval_s
    vx = np.gradient(bundle.xyz[:, 0], bundle.times)
    vy = np.gradient(bundle.xyz[:, 1], bundle.times)
    ax = np.gradient(vx, bundle.times)
    ay = np.gradient(vy, bundle.times)
    speed = np.hypot(vx, vy)
    curvature_denom = np.clip(vx * vx + vy * vy, 1e-3, None) ** 1.5
    curvature = np.nan_to_num((vx * ay - vy * ax) / curvature_denom)
    speed_var = float(np.var(speed))
    curvature_mag = float(np.mean(np.abs(curvature)))
    confidence = math.exp(-0.05 * speed_var - 0.5 * curvature_mag)
    return float(np.clip(confidence, 0.0, 1.0))

  def _extract_lane_geometry(self, bev_map: torch.Tensor | None) -> dict[str, Any]:
    if bev_map is None:
      return {
        "lane_lines": [],
        "lane_line_probs": [],
        "road_edges": [],
        "road_edge_stds": [],
      }

    bev = torch.argmax(bev_map.squeeze(0), dim=0).cpu().numpy()
    height, width = bev.shape
    pixel = float(self.config.bev_pixel_size)
    center_col = width / 2.0

    left_samples: list[tuple[float, float]] = []
    right_samples: list[tuple[float, float]] = []
    center_samples: list[tuple[float, float]] = []
    forward_samples: list[float] = []

    for row in range(height):
      road_cols = np.where(bev[row] == 1)[0]
      if road_cols.size < 2:
        continue
      forward = (height - 1 - row) * pixel
      left_col = road_cols[0]
      right_col = road_cols[-1]
      left_y = (center_col - left_col) * pixel
      right_y = (center_col - right_col) * pixel
      centre_y = 0.5 * (left_y + right_y)
      left_samples.append((forward, left_y))
      right_samples.append((forward, right_y))
      center_samples.append((forward, centre_y))
      forward_samples.append(forward)

    lane_lines = []
    lane_probs = []
    road_edges = []
    road_edge_stds = []

    def _pack(points: Iterable[tuple[float, float]]) -> dict[str, Any]:
      arr = np.asarray(list(points), dtype=np.float32)
      return {"x": arr[:, 0].tolist(), "y": arr[:, 1].tolist()}

    if left_samples:
      lane_lines.append(_pack(left_samples))
      lane_probs.append(0.6)
      road_edges.append(_pack(left_samples))
      road_edge_stds.append(0.5)
    if center_samples:
      lane_lines.append(_pack(center_samples))
      lane_probs.append(0.8)
    if right_samples:
      lane_lines.append(_pack(right_samples))
      lane_probs.append(0.6)
      road_edges.append(_pack(right_samples))
      road_edge_stds.append(0.5)

    lane_width = None
    if left_samples and right_samples:
      widths = [abs(l - r) for (_, l), (_, r) in zip(left_samples, right_samples)]
      if widths:
        lane_width = float(np.mean(widths))

    return {
      "lane_lines": lane_lines,
      "lane_line_probs": lane_probs,
      "road_edges": road_edges,
      "road_edge_stds": road_edge_stds,
      "lane_width": lane_width,
      "sampled_forward": forward_samples,
    }

  def _extract_agents(self, states: torch.Tensor | None,
                      labels: torch.Tensor | None) -> list[dict[str, float]]:
    if states is None or labels is None:
      return []
    probs = torch.sigmoid(labels.squeeze(0)).cpu().numpy()
    boxes = states.squeeze(0).cpu().numpy()
    agents = []
    for prob, box in zip(probs, boxes):
      if prob < 0.2:
        continue
      x, y, heading, length, width = map(float, box[:5])
      if x < -1.0:
        continue
      distance = math.hypot(x, y)
      if distance > 150.0:
        continue
      agents.append({
        "x": x,
        "y": y,
        "heading": heading,
        "length": length,
        "width": width,
        "prob": float(prob),
      })

    agents.sort(key=lambda a: a["x"])
    return agents[:3]


def build_response(model: DiffusionDriveModel, header: dict[str, Any], frame: memoryview | bytes) -> dict[str, Any]:
  start = time.perf_counter()
  payload = model.infer(header, frame)
  payload["model_execution_time"] = time.perf_counter() - start
  payload["status"] = payload.get("status", "ok")
  return payload


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="DiffusionDrive RPC server")
  parser.add_argument("--bind", default="tcp://0.0.0.0:5555", help="Endpoint to bind the ZeroMQ REP socket")
  parser.add_argument("--diffusiondrive-dir", required=True, help="Path to the DiffusionDrive repository checkout")
  parser.add_argument("--weights", required=True, help="Path to the DiffusionDrive checkpoint (.ckpt)")
  parser.add_argument("--plan-anchor", help="Optional path to plan anchor .npy file")
  parser.add_argument("--device", default="cuda", help="Torch device (cuda or cpu)")
  parser.add_argument("--dtype", choices=["float16", "float32"], default="float16", help="Model precision")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  repo = Path(args.diffusiondrive_dir)
  weights = Path(args.weights)
  plan_anchor = Path(args.plan_anchor) if args.plan_anchor else None

  model = DiffusionDriveModel(repo=repo, weights=weights, device=args.device, plan_anchor=plan_anchor, dtype=args.dtype)

  context = zmq.Context.instance()
  socket = context.socket(zmq.REP)
  socket.bind(args.bind)
  print(f"DiffusionDrive server listening on {args.bind} using {args.device} ({args.dtype})")

  try:
    while True:
      parts = socket.recv_multipart()
      if not parts:
        continue
      header = json.loads(parts[0].decode("utf-8"))
      frame = parts[1] if len(parts) > 1 else memoryview(b"")
      try:
        response = build_response(model, header, frame)
      except Exception as err:  # pylint: disable=broad-except
        response = {
          "seq": int(header.get("seq", 0)),
          "mono_time_ns": int(header.get("mono_time_ns", 0)),
          "status": f"error: {err}",
          "confidence": 0.0,
          "trajectory": [],
        }
      socket.send_json(response)
  except KeyboardInterrupt:
    print("Stopping DiffusionDrive server")
  finally:
    socket.close(linger=0)
    context.term()


if __name__ == "__main__":
  main()


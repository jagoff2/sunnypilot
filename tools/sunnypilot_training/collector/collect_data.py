"""CARLA data collection pipeline producing Zarr shards for training."""
from __future__ import annotations

import argparse
import logging
import math
import queue
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np

from openpilot.selfdrive.modeld.constants import ModelConstants

from ..dataset import DEFAULT_SCHEMA, ZarrShardWriter
from ..sensors import CameraSpecification, create_camera_blueprint, create_wide_camera_blueprint, mount_transform
from ..scenarios.registry import BaseScenario, ScenarioRegistry, default_registry

try:
  import carla  # type: ignore
except ImportError:  # pragma: no cover - runtime dependency
  carla = None  # type: ignore


LOG = logging.getLogger(__name__)


@dataclass
class CollectorConfig:
  """Configuration for the CARLA data collector."""

  output_dir: Path
  scenario: str
  episodes: int = 10
  shard_size: int = 512
  seed: Optional[int] = None
  synchronous: bool = True
  fixed_delta_seconds: float = 1.0 / ModelConstants.MODEL_FREQ


class CarlaDataCollector:
  """Collect simulator rollouts and convert them into training samples."""

  def __init__(self, client: "carla.Client", registry: ScenarioRegistry, config: CollectorConfig,
               camera_spec: Optional[CameraSpecification] = None) -> None:
    if carla is None:
      raise ImportError("carla module is required for data collection")
    self.client = client
    self.registry = registry
    self.config = config
    self.camera_spec = camera_spec or CameraSpecification()
    self.schema = DEFAULT_SCHEMA

  def run(self) -> None:
    rng = np.random.default_rng(self.config.seed)
    for episode in range(self.config.episodes):
      seed = int(rng.integers(0, 2**31 - 1))
      scenario = self.registry.instantiate(self.config.scenario, self.client, seed)
      metadata = scenario.metadata
      LOG.info("Starting episode %d/%d: %s", episode + 1, self.config.episodes, metadata)
      shard_path = self.config.output_dir / f"{metadata.scenario}_{metadata.map_name}_{metadata.seed}.zarr"
      writer = ZarrShardWriter(shard_path, schema=self.schema, chunk_size=self.config.shard_size)
      try:
        self._collect_episode(scenario, writer)
      finally:
        writer.flush()
        scenario.teardown()

  def _collect_episode(self, scenario: BaseScenario, writer: ZarrShardWriter) -> None:
    world = scenario.world
    ego = scenario.ego_vehicle
    blueprint_library = world.get_blueprint_library()
    front_bp = create_camera_blueprint(blueprint_library, self.camera_spec, role_name="front")
    front_wide_bp = create_wide_camera_blueprint(blueprint_library, self.camera_spec, role_name="front_wide")
    transform = mount_transform(self.camera_spec)

    queues: Dict[str, "queue.Queue[carla.Image]"] = {
      "front": queue.Queue(),
      "front_wide": queue.Queue(),
    }

    def _make_callback(name: str):
      def _callback(image: "carla.Image") -> None:
        queues[name].put(image)
      return _callback

    front_cam = world.spawn_actor(front_bp, transform, attach_to=ego)
    front_cam.listen(_make_callback("front"))
    actors: List[carla.Actor] = [front_cam]

    wide_cam = None
    if self.camera_spec.enable_wide:
      wide_cam = world.spawn_actor(front_wide_bp, transform, attach_to=ego)
      wide_cam.listen(_make_callback("front_wide"))
      actors.append(wide_cam)

    image_buffer: Deque[np.ndarray] = deque(maxlen=self.schema.vision_inputs["img"][0])
    big_image_buffer: Deque[np.ndarray] = deque(maxlen=self.schema.vision_inputs["big_img"][0])
    feature_history: Deque[np.ndarray] = deque(maxlen=ModelConstants.INPUT_HISTORY_BUFFER_LEN)

    try:
      while scenario.tick():
        try:
          front_image = queues["front"].get(timeout=2.0)
          wide_image = queues["front_wide"].get(timeout=2.0) if wide_cam else front_image
        except queue.Empty:
          LOG.warning("Sensor data timeout; aborting episode")
          break
        img_stack, big_stack = self._process_images(front_image, wide_image, image_buffer, big_image_buffer)
        features, targets = self._build_targets(scenario, feature_history)
        sample = {
          "img": img_stack.astype(np.float32),
          "big_img": big_stack.astype(np.float32),
          "features_buffer": np.stack(list(feature_history), axis=0).astype(np.float32),
          "desire": targets.pop("desire_input"),
          "traffic_convention": targets.pop("traffic_convention"),
        }
        sample.update(targets)
        writer.append(sample)
    finally:
      for actor in actors:
        actor.stop()
        actor.destroy()

  def _process_images(self, front_image: "carla.Image", wide_image: "carla.Image",
                      narrow_buffer: Deque[np.ndarray], wide_buffer: Deque[np.ndarray]
                      ) -> Tuple[np.ndarray, np.ndarray]:
    front_np = self._to_numpy(front_image)
    wide_np = self._to_numpy(wide_image)
    narrow_frame = self._preprocess(front_np, output_shape=self.schema.vision_inputs["img"][1:])
    wide_frame = self._preprocess(wide_np, output_shape=self.schema.vision_inputs["big_img"][1:])
    if not narrow_buffer:
      for _ in range(narrow_buffer.maxlen):
        narrow_buffer.append(narrow_frame)
    else:
      narrow_buffer.append(narrow_frame)
    if not wide_buffer:
      for _ in range(wide_buffer.maxlen):
        wide_buffer.append(wide_frame)
    else:
      wide_buffer.append(wide_frame)
    return np.stack(list(narrow_buffer), axis=0), np.stack(list(wide_buffer), axis=0)

  def _to_numpy(self, image: "carla.Image") -> np.ndarray:
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    rgb = array[:, :, :3][:, :, ::-1]  # BGRA -> RGB
    return rgb

  def _preprocess(self, rgb: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
    resized = self._resize(rgb, output_shape)
    yuv = self._rgb_to_yuv(resized)
    y_channel = yuv[..., 0]
    return y_channel.astype(np.float32) / 255.0

  @staticmethod
  def _resize(img: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = output_shape
    y = np.linspace(0, img.shape[0] - 1, target_h)
    x = np.linspace(0, img.shape[1] - 1, target_w)
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    y0 = np.floor(grid_y).astype(np.int32)
    x0 = np.floor(grid_x).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, img.shape[0] - 1)
    x1 = np.clip(x0 + 1, 0, img.shape[1] - 1)
    wa = (x1 - grid_x) * (y1 - grid_y)
    wb = (grid_x - x0) * (y1 - grid_y)
    wc = (x1 - grid_x) * (grid_y - y0)
    wd = (grid_x - x0) * (grid_y - y0)
    resized = (
      wa[..., None] * img[y0, x0]
      + wb[..., None] * img[y0, x1]
      + wc[..., None] * img[y1, x0]
      + wd[..., None] * img[y1, x1]
    )
    return resized.astype(np.uint8)

  @staticmethod
  def _rgb_to_yuv(rgb: np.ndarray) -> np.ndarray:
    m = np.array([[0.299, 0.587, 0.114],
                  [-0.14713, -0.28886, 0.436],
                  [0.615, -0.51499, -0.10001]], dtype=np.float32)
    offset = np.array([0.0, 128.0, 128.0], dtype=np.float32)
    flat = rgb.reshape(-1, 3).astype(np.float32)
    yuv = flat @ m.T + offset
    return yuv.reshape(rgb.shape)

  def _build_targets(self, scenario: BaseScenario, feature_history: Deque[np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    ego = scenario.ego_vehicle
    world = scenario.world
    transform = ego.get_transform()
    velocity = ego.get_velocity()
    speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
    accel = ego.get_acceleration()
    yaw = math.radians(transform.rotation.yaw)

    plan_positions = self._sample_future_waypoints(world, transform)
    lane_lines = self._sample_lane_lines(world, transform)
    road_edges = self._sample_road_edges(world, transform)
    lead_state = self._find_lead_vehicle(world, ego, transform)

    features = self._encode_features(speed, accel, yaw, plan_positions, lead_state)
    feature_history.append(features)
    while len(feature_history) < ModelConstants.INPUT_HISTORY_BUFFER_LEN:
      feature_history.appendleft(features)

    desire_input = np.zeros((ModelConstants.DESIRE_LEN,), dtype=np.float32)
    desire_input[0] = 1.0  # keep lane
    traffic_convention = np.zeros((ModelConstants.TRAFFIC_CONVENTION_LEN,), dtype=np.float32)
    traffic_convention[0] = 1.0  # right-hand traffic

    targets: Dict[str, np.ndarray] = {
      "pose": self._encode_pose(transform, velocity, accel),
      "road_transform": self._encode_road_transform(transform),
      "lane_lines": lane_lines,
      "lane_lines_prob": np.ones((ModelConstants.NUM_LANE_LINES,), dtype=np.float32),
      "road_edges": road_edges,
      "lead": lead_state,
      "lead_prob": np.array([1.0], dtype=np.float32),
      "meta": self._encode_meta(speed),
      "desire_pred": desire_input.copy(),
      "hidden_state": features.copy(),
      "plan": plan_positions,
      "plan_prob": np.array([1.0], dtype=np.float32),
      "desire_state": desire_input.copy(),
      "desire_input": desire_input,
      "traffic_convention": traffic_convention,
    }
    return features, targets

  def _encode_features(self, speed: float, accel: "carla.Vector3D", yaw: float,
                        plan: np.ndarray, lead_state: np.ndarray) -> np.ndarray:
    base = np.concatenate([
      np.array([speed, accel.x, accel.y, accel.z, yaw], dtype=np.float32),
      plan.flatten(),
      lead_state.flatten(),
    ])
    if base.shape[0] > ModelConstants.FEATURE_LEN:
      base = base[:ModelConstants.FEATURE_LEN]
    return np.pad(base, (0, ModelConstants.FEATURE_LEN - base.shape[0]))

  @staticmethod
  def _encode_pose(transform: "carla.Transform", velocity: "carla.Vector3D", accel: "carla.Vector3D") -> np.ndarray:
    pitch = math.radians(transform.rotation.pitch)
    yaw = math.radians(transform.rotation.yaw)
    roll = math.radians(transform.rotation.roll)
    return np.array([velocity.x, velocity.y, velocity.z, pitch, yaw, roll], dtype=np.float32)

  @staticmethod
  def _encode_road_transform(transform: "carla.Transform") -> np.ndarray:
    pitch = math.radians(transform.rotation.pitch)
    yaw = math.radians(transform.rotation.yaw)
    roll = math.radians(transform.rotation.roll)
    return np.array([transform.location.x, transform.location.y, transform.location.z, pitch, yaw, roll], dtype=np.float32)

  @staticmethod
  def _encode_meta(speed: float) -> np.ndarray:
    meta = np.zeros((26,), dtype=np.float32)
    meta[0] = 1.0  # engaged
    meta[1] = speed
    meta[2] = max(0.0, speed - 5.0)
    meta[3] = max(0.0, 5.0 - speed)
    meta[4:] = speed / 30.0
    return meta

  def _sample_future_waypoints(self, world: "carla.World", transform: "carla.Transform") -> np.ndarray:
    waypoint = world.get_map().get_waypoint(transform.location)
    plan = np.zeros((ModelConstants.PLAN_MHP_SELECTION, ModelConstants.PLAN_WIDTH), dtype=np.float32)
    distances = ModelConstants.X_IDXS[:ModelConstants.PLAN_WIDTH]
    for idx, dist in enumerate(distances):
      next_wp_list = waypoint.next(float(dist) / 2.0 + 1e-3)
      if not next_wp_list:
        break
      next_wp = next_wp_list[0]
      rel_x, rel_y = self._to_vehicle_frame(transform, next_wp.transform.location)
      plan[0, idx] = math.sqrt(rel_x ** 2 + rel_y ** 2)
    return plan

  def _sample_lane_lines(self, world: "carla.World", transform: "carla.Transform") -> np.ndarray:
    waypoint = world.get_map().get_waypoint(transform.location)
    lane_lines = np.zeros((ModelConstants.NUM_LANE_LINES, ModelConstants.IDX_N, ModelConstants.LANE_LINES_WIDTH), dtype=np.float32)
    offsets = [-1.5, -0.5, 0.5, 1.5]
    for lane_idx, offset in enumerate(offsets):
      current = waypoint
      for point_idx, dist in enumerate(ModelConstants.X_IDXS):
        next_wps = current.next(float(dist) / 2.0 + 1e-3)
        if not next_wps:
          break
        current = next_wps[0]
        location = current.transform.location
        rel_x, rel_y = self._to_vehicle_frame(transform, location)
        lane_lines[lane_idx, point_idx, 0] = rel_x
        lane_lines[lane_idx, point_idx, 1] = rel_y + offset
    return lane_lines

  def _sample_road_edges(self, world: "carla.World", transform: "carla.Transform") -> np.ndarray:
    waypoint = world.get_map().get_waypoint(transform.location)
    edges = np.zeros((ModelConstants.NUM_ROAD_EDGES, ModelConstants.IDX_N, ModelConstants.ROAD_EDGES_WIDTH), dtype=np.float32)
    offsets = [-3.5, 3.5]
    for edge_idx, offset in enumerate(offsets):
      current = waypoint
      for point_idx, dist in enumerate(ModelConstants.X_IDXS):
        next_wps = current.next(float(dist) / 2.0 + 1e-3)
        if not next_wps:
          break
        current = next_wps[0]
        location = current.transform.location
        rel_x, rel_y = self._to_vehicle_frame(transform, location)
        edges[edge_idx, point_idx, 0] = rel_x
        edges[edge_idx, point_idx, 1] = rel_y + offset
    return edges

  @staticmethod
  def _to_vehicle_frame(transform: "carla.Transform", location: "carla.Location") -> Tuple[float, float]:
    dx = location.x - transform.location.x
    dy = location.y - transform.location.y
    yaw = math.radians(transform.rotation.yaw)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    forward = cos_yaw * dx + sin_yaw * dy
    lateral = -sin_yaw * dx + cos_yaw * dy
    return forward, lateral

  def _find_lead_vehicle(self, world: "carla.World", ego: "carla.Vehicle", transform: "carla.Transform") -> np.ndarray:
    actors = world.get_actors().filter("vehicle.*")
    ego_location = transform.location
    ego_yaw = math.radians(transform.rotation.yaw)
    cos_yaw = math.cos(ego_yaw)
    sin_yaw = math.sin(ego_yaw)
    best_distance = float("inf")
    best_velocity = 0.0
    for actor in actors:
      if actor.id == ego.id:
        continue
      loc = actor.get_location()
      dx = loc.x - ego_location.x
      dy = loc.y - ego_location.y
      forward = cos_yaw * dx + sin_yaw * dy
      lateral = -sin_yaw * dx + cos_yaw * dy
      if forward < 0 or abs(lateral) > 5.0:
        continue
      distance = math.hypot(dx, dy)
      if distance < best_distance:
        best_distance = distance
        vel = actor.get_velocity()
        best_velocity = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
    if best_distance == float("inf"):
      return np.zeros((ModelConstants.LEAD_MHP_SELECTION, ModelConstants.LEAD_WIDTH), dtype=np.float32)
    lead = np.zeros((ModelConstants.LEAD_MHP_SELECTION, ModelConstants.LEAD_WIDTH), dtype=np.float32)
    lead[0, 0] = best_distance
    lead[0, 1] = best_velocity
    lead[0, 2] = 0.0
    lead[0, 3] = 0.0
    return lead


def _parse_args(args: Optional[Iterable[str]] = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Collect CARLA data for sunnypilot training")
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=2000)
  parser.add_argument("--scenario", default="urban_intersection")
  parser.add_argument("--episodes", type=int, default=10)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--seed", type=int)
  parser.add_argument("--shard-size", type=int, default=512)
  return parser.parse_args(args)


def main(argv: Optional[Iterable[str]] = None) -> None:
  logging.basicConfig(level=logging.INFO)
  args = _parse_args(argv)
  if carla is None:
    raise ImportError("carla module is required to run the data collector")
  client = carla.Client(args.host, args.port)
  client.set_timeout(10.0)
  registry = default_registry()
  config = CollectorConfig(
    output_dir=args.output,
    scenario=args.scenario,
    episodes=args.episodes,
    shard_size=args.shard_size,
    seed=args.seed,
  )
  collector = CarlaDataCollector(client, registry, config)
  collector.run()


if __name__ == "__main__":
  main()

"""Scenario registry for CARLA data generation.

The scenarios are intentionally deterministic and reproducible to make
training runs turnkey. Each scenario encapsulates the spawning of the
EGO vehicle, traffic, route planning, and cleanup logic.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

try:
  import carla  # type: ignore
except ImportError:  # pragma: no cover - runtime dependency
  carla = None  # type: ignore


LOG = logging.getLogger(__name__)


@dataclass
class ScenarioMetadata:
  """Metadata describing a generated sample batch."""

  scenario: str
  map_name: str
  seed: int
  weather: str
  description: str = ""


@dataclass
class ScenarioConfig:
  """Configuration parameters for scenario instantiation."""

  name: str
  town: str
  description: str
  autopilot: bool = True
  max_actors: int = 40
  route_seeds: Sequence[int] = field(default_factory=lambda: [0])
  weather_presets: Sequence["carla.WeatherParameters"] = field(default_factory=list)
  ego_vehicle_bp: str = "vehicle.lincoln.mkz2017"
  max_steps: int = 1200
  synchronous: bool = True
  fixed_delta_seconds: float = 1.0 / 20.0


class BaseScenario:
  """Base scenario implementation used by all specific scenarios."""

  def __init__(self, world: "carla.World", client: "carla.Client", config: ScenarioConfig,
               seed: int, traffic_manager: Optional["carla.TrafficManager"] = None) -> None:
    if carla is None:
      raise ImportError("carla module is required to instantiate scenarios")
    self.world = world
    self.client = client
    self.config = config
    self.random = random.Random(seed)
    self.traffic_manager = traffic_manager or client.get_trafficmanager()
    self._actors: List[carla.Actor] = []
    self._ego_vehicle: Optional[carla.Vehicle] = None
    self._step = 0
    self.metadata = ScenarioMetadata(
      scenario=config.name,
      map_name=world.get_map().name,
      seed=seed,
      weather=str(world.get_weather()),
      description=config.description,
    )

  def setup(self) -> None:
    """Spawn the ego vehicle and register background traffic."""
    LOG.info("Setting up scenario %s", self.config.name)
    blueprint_library = self.world.get_blueprint_library()
    vehicle_bp = blueprint_library.find(self.config.ego_vehicle_bp)
    spawn_points = self.world.get_map().get_spawn_points()
    if not spawn_points:
      raise RuntimeError("No spawn points available in map")
    spawn_point = self.random.choice(spawn_points)
    vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(self.config.autopilot, self.traffic_manager.get_port())
    self._ego_vehicle = vehicle
    self._actors.append(vehicle)
    self._spawn_background_traffic()

  def _spawn_background_traffic(self) -> None:
    LOG.debug("Spawning background traffic up to %d actors", self.config.max_actors)
    blueprint_library = self.world.get_blueprint_library()
    spawn_points = list(self.world.get_map().get_spawn_points())
    self.random.shuffle(spawn_points)
    vehicles_to_spawn = min(self.config.max_actors, len(spawn_points))
    batch: List[carla.command.SpawnActor] = []
    for idx in range(vehicles_to_spawn):
      bp = random.choice(blueprint_library.filter("vehicle.*"))
      spawn_transform = spawn_points[idx]
      bp.set_attribute("role_name", f"autopilot_{idx}")
      command = carla.command.SpawnActor(bp, spawn_transform)
      command.then(carla.command.SetAutopilot(carla.command.FutureActor, True, self.traffic_manager.get_port()))
      batch.append(command)
    responses = self.client.apply_batch_sync(batch, self.config.synchronous)
    for response in responses:
      if response.error:
        LOG.warning("Failed to spawn background actor: %s", response.error)
        continue
      actor = self.world.get_actor(response.actor_id)
      if actor is not None:
        self._actors.append(actor)

  def tick(self) -> bool:
    """Advance the scenario one step. Returns False when complete."""
    self._step += 1
    if self._step >= self.config.max_steps:
      return False
    if self.config.synchronous:
      self.world.tick()
    else:
      self.world.wait_for_tick()
    return True

  def teardown(self) -> None:
    LOG.info("Tearing down scenario %s", self.config.name)
    for actor in self._actors:
      try:
        actor.destroy()
      except RuntimeError:
        LOG.exception("Failed to destroy actor %s", actor)
    self._actors.clear()
    self._ego_vehicle = None

  @property
  def ego_vehicle(self) -> carla.Vehicle:
    if self._ego_vehicle is None:
      raise RuntimeError("Ego vehicle not spawned; call setup() first")
    return self._ego_vehicle


class IntersectionScenario(BaseScenario):
  """Scenario focusing on intersection maneuvers with cross traffic."""

  def _spawn_background_traffic(self) -> None:
    super()._spawn_background_traffic()
    traffic_manager = self.traffic_manager
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_synchronous_mode(self.config.synchronous)


class HighwayFollowScenario(BaseScenario):
  """Scenario that keeps ego behind a lead vehicle on the highway."""

  def _spawn_background_traffic(self) -> None:
    super()._spawn_background_traffic()
    if self._ego_vehicle is None:
      return
    ego_location = self._ego_vehicle.get_location()
    waypoint = self.world.get_map().get_waypoint(ego_location)
    ahead_waypoint = waypoint.next(60.0)[0]
    blueprint_library = self.world.get_blueprint_library()
    lead_bp = blueprint_library.find("vehicle.tesla.model3")
    transform = carla.Transform(ahead_waypoint.transform.location, ahead_waypoint.transform.rotation)
    lead_vehicle = self.world.try_spawn_actor(lead_bp, transform)
    if lead_vehicle:
      lead_vehicle.set_autopilot(True, self.traffic_manager.get_port())
      self._actors.append(lead_vehicle)
      self.traffic_manager.vehicle_percentage_speed_difference(lead_vehicle, -20.0)


class ScenarioRegistry:
  """In-memory registry mapping scenario names to factories."""

  def __init__(self) -> None:
    self._registry: Dict[str, ScenarioConfig] = {}

  def register(self, config: ScenarioConfig) -> None:
    if config.name in self._registry:
      raise ValueError(f"Scenario {config.name} already registered")
    self._registry[config.name] = config

  def get(self, name: str) -> ScenarioConfig:
    return self._registry[name]

  def all(self) -> Iterable[ScenarioConfig]:
    return self._registry.values()

  def instantiate(self, name: str, client: "carla.Client", seed: Optional[int] = None,
                  traffic_manager: Optional["carla.TrafficManager"] = None) -> BaseScenario:
    config = self.get(name)
    if seed is None:
      seed = random.randint(0, 2**31 - 1)
    client.load_world(config.town)
    world = client.get_world()
    if config.synchronous:
      settings = world.get_settings()
      settings.fixed_delta_seconds = config.fixed_delta_seconds
      settings.synchronous_mode = True
      world.apply_settings(settings)
    if config.weather_presets:
      weather = random.choice(config.weather_presets)
      world.set_weather(weather)
    scenario_cls = IntersectionScenario if "intersection" in config.name else HighwayFollowScenario
    scenario = scenario_cls(world, client, config, seed, traffic_manager)
    scenario.setup()
    return scenario


def default_registry() -> ScenarioRegistry:
  """Return a registry populated with the default scenario set."""
  if carla is None:
    raise ImportError("carla module is required to construct the scenario registry")
  registry = ScenarioRegistry()
  registry.register(ScenarioConfig(
    name="urban_intersection",
    town="Town03",
    description="Signalised intersections with heavy cross traffic",
    max_actors=80,
    weather_presets=[carla.WeatherParameters.ClearNoon, carla.WeatherParameters.WetSunset],
  ))
  registry.register(ScenarioConfig(
    name="suburban_four_way_stop",
    town="Town05",
    description="Four-way stop with occluded traffic and pedestrians",
    max_actors=50,
    weather_presets=[carla.WeatherParameters.CloudySunset],
  ))
  registry.register(ScenarioConfig(
    name="highway_follow",
    town="Town04",
    description="Multi-lane highway with lead vehicle pacing",
    max_actors=60,
    weather_presets=[carla.WeatherParameters.ClearNoon, carla.WeatherParameters.WetCloudyNoon],
  ))
  return registry


__all__ = [
  "ScenarioMetadata",
  "ScenarioConfig",
  "BaseScenario",
  "IntersectionScenario",
  "HighwayFollowScenario",
  "ScenarioRegistry",
  "default_registry",
]

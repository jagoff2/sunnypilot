# Scenario registry

The scenario registry (`scenarios/registry.py`) defines deterministic templates used during
collection. Each scenario is configured through `ScenarioConfig`:

- `name`: unique identifier used when launching from CLI.
- `town`: CARLA town map.
- `description`: human readable summary.
- `max_actors`: total background vehicles to spawn.
- `weather_presets`: list of `carla.WeatherParameters` values.
- `max_steps`: number of synchronous ticks before terminating the episode.

## Available scenarios

| Scenario | Map    | Description                               | Notes |
|----------|--------|-------------------------------------------|-------|
| `urban_intersection` | Town03 | Signalized downtown intersection with heavy cross traffic | Uses reduced following distance for traffic manager |
| `suburban_four_way_stop` | Town05 | Four-way stop with occlusions and pedestrians | Encourages creep behavior |
| `highway_follow` | Town04 | Multi-lane highway follow with scripted lead | Lead vehicle slowed by 20% to create realistic following gaps |

## Extending the registry

Add new entries via `registry.register(ScenarioConfig(...))`. To customize behavior:

1. Inherit from `BaseScenario` and override `_spawn_background_traffic` or `tick`.
2. Register using the new scenario class by adjusting the factory logic in `ScenarioRegistry.instantiate`.
3. Provide appropriate metadata so generated shards capture the scenario details.

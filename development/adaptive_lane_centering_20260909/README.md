# Adaptive lane geometry with bounded computation

The active Python controller now estimates a reliable geometry prefix from existing model outputs and chooses centering response independently. This change builds on the reverted controller at `268e4ef777aa472d83cf5399443e5bfb001626c8`; it does not restore the archived experimental planner.

## Behavior

- Geometry support spans 15–60 m. Scan the native lane/edge grid outward and stop at the first nonfinite value, excessive uncertainty, implausible width, or disagreement with the ego-motion-propagated centerline. These are model support estimates, not calibrated physical visibility.
- Preserve native interpolation brackets, fit only supported samples, and compare acquisition/recovery/source references at common physical distances. Policy disagreement checks use the supported portion of their original 10–30 m interval. Their rejection thresholds and lane-change override remain in place.
- Support shrinks immediately and ages with distance travelled. Fresh observations replenish that distance; additional support grows by at most 10 m/s. Previously synthetic samples are replaced as real support grows.
- The response target is 2.0 s independently of support distance. The parallel-offset jerk estimate `60*abs(offset)/T^3` lengthens it when needed. Shortening the chosen response is limited to 0.5 s per second. Heading, curvature, speed variation, acceleration and jerk are validated on the complete generated trajectory.
- Construct one trajectory and, only if needed, one analytically lengthened retry. Both must pass the existing whole-plan 5 m/s² and 5 m/s³ checks. A rejected retry retains the existing fallback behavior.
- If convergence lies beyond reliable geometry, only partial recentering occurs inside the supported section. The complete quintic still finishes and unwinds the corrective maneuver against the local fitted curve. Its continuation is explicitly unobserved, never used to extend geometry/policy evidence, and never freezes mid-maneuver lateral acceleration indefinitely. `horizon_limited` identifies this condition.
- The action extraction point itself must remain inside supported geometry. If speed and action delay place it beyond the reliable horizon, selection yields with `geometry_support_short` rather than commanding curvature from the unobserved continuation.

## Cost and observability

No optimizer, model, worker, process, IPC channel or parameter polling was added. Each boundary scan considers at most 19 native samples. There are at most three candidate boundary pairs and two complete trajectory constructions. Prefix fit operators are calculated once when the module loads. The old per-frame 3×3 linear solve is replaced by its closed-form coefficients; repeated boundary extraction, support scans, fits and feasibility checks are avoided.

The existing adapter emits at most one ordinary diagnostic per model-timestamp second through `cloudlog`, which reaches the normal log pipeline. It includes geometry horizon, convergence distance/time, partial-support status, policy disagreement and gate names. No schema change or full per-distance uncertainty logging is introduced. This remains sparse telemetry and can miss subsecond transitions.

## Validation

`benchmark_core.py` compares identical inputs against the reverted controller. `benchmark-results.json` contains source hashes, CPU and wall percentiles, state verification and synthetic response results. Both modules were loaded into a separate process on the offroad C4, on CPU7 with a reported 1.6896 GHz maximum. The installed controller hash was unchanged and CPU7's previous offroad state was restored.

Each benchmark case uses 80 warmup and 150 timed frames. Representative p95 CPU milliseconds:

| Case | Reverted | Adaptive |
|---|---:|---:|
| Centered, lane pair | 2.35 | 2.36 |
| Offset, lane pair | 2.35 | 2.36 |
| Curved lane pair | 2.33 | 2.36 |
| All three boundary combinations available | 3.00 | 3.80 |
| Only 20 m supported | 0.97, inactive | 2.35, active |
| Infeasible trajectory | 2.32 | 2.90 |

This measures the pure controller, excluding inference, adapter logging and other onroad processes. It establishes isolated computation cost, not an end-to-end onroad timing guarantee.

Synthetic straight-road simulation at 16 m/s used perfect geometry, 0.475 s action preview, 0.1 s command smoothing and an assumed 0.20 s first-order actuator response, with no transport delay. Acquisition and perception filtering were bypassed to isolate path response. Time to remain within 0.1 m of center improved from 2.45 to 1.80 s for a 0.44 m initial offset and from 2.85 to 2.30 s for a 0.8 m offset. There were no rejected paths; overshoot remained below 1.3 cm. These results are not a replay or road validation of the bookmarks.

All 32 focused native C4 tests passed (`validation.txt`). Tests cover native stock/custom action integration, longitudinal preservation, invalid-input/lane-change release, support gaps and malformed tails, temporal support, minimum-support operation while moving, interpolation brackets, bounded partial maneuvers, dynamics, independent urgency and bounded construction count. Run:

```sh
python -m pytest openpilot/selfdrive/modeld/tests/test_lane_centering_integration.py openpilot/selfdrive/modeld/tests/test_adaptive_lane_centering.py
```

Local syntax, whitespace, and focused Ruff E4/E7/E9/F checks were used. The local Ruff executable cannot parse the repository's newer RUF103 rule selector, so a full repository-configured Ruff run was not claimed.

The code is implemented in the workspace. Vehicle installation and a moving road test were not performed by this task.

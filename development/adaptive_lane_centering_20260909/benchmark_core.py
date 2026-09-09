"""Bounded pure-planner benchmark. Caller injects modules `baseline` and `current`.

The wrapper owns offroad/device checks. This file never imports messaging or
inference, modifies installed code, or sends vehicle commands.
"""
import json
import math
import time
from copy import deepcopy
from collections import Counter

import numpy as np


FRAMES = 150
WARMUP = 80
DT = 0.05
SPEED = 16.0
ACTION_T = 0.475


def model_input(core, *, offset=0.12, curvature=0.0, support=None,
                three_sources=False, policy_shift=0.0):
  constants, plan_fields = core.ModelConstants, core.Plan
  native_x = np.asarray(constants.X_IDXS, dtype=np.float64)
  times = np.asarray(constants.T_IDXS, dtype=np.float64)
  n = constants.IDX_N
  center = offset + 0.5 * curvature * native_x ** 2
  lanes = np.zeros((1, 4, n, 2), dtype=np.float64)
  for i, shift in enumerate((-5.4, -1.8, 1.8, 5.4)):
    lanes[0, i, :, 0] = center + shift - core.CAMERA_OFFSET
  edges = np.zeros((1, 2, n, 2), dtype=np.float64)
  for i, shift in enumerate((-1.8, 1.8) if three_sources else (-5.4, 5.4)):
    edges[0, i, :, 0] = center + shift - core.CAMERA_OFFSET
  lane_std = np.full_like(lanes, 0.03)
  edge_std = np.full_like(edges, 0.05 if three_sources else 0.8)
  if support is not None:
    # Retain the observed upper interpolation bracket at the requested endpoint.
    bracket = native_x[np.searchsorted(native_x, support)]
    lane_std[:, :, native_x > bracket] = 1.0
    edge_std[:, :, native_x > bracket] = 1.0
  plan = np.zeros((1, n, constants.PLAN_WIDTH), dtype=np.float64)
  path_x = SPEED * times
  slope = curvature * path_x
  actual_curvature = curvature / (1.0 + slope * slope) ** 1.5
  plan[0, :, plan_fields.POSITION.start] = path_x
  plan[0, :, plan_fields.POSITION.start + 1] = offset + policy_shift + 0.5 * curvature * path_x ** 2
  plan[0, :, plan_fields.VELOCITY.start] = SPEED
  plan[0, :, plan_fields.ACCELERATION.start + 1] = actual_curvature * SPEED ** 2
  plan[0, :, plan_fields.T_FROM_CURRENT_EULER.start + 2] = np.arctan(slope)
  plan[0, :, plan_fields.ORIENTATION_RATE.start + 2] = actual_curvature * SPEED
  return {
    'plan': plan, 'lane_lines': lanes, 'lane_lines_stds': lane_std,
    'lane_lines_prob': np.repeat([0.97] * 4, 2)[None],
    'road_edges': edges, 'road_edges_stds': edge_std,
    'desire_state': np.zeros((1, 8)),
  }


def percentiles(values):
  return {label: float(np.percentile(values, percentile))
          for label, percentile in [('p50_ms', 50), ('p95_ms', 95), ('p99_ms', 99), ('max_ms', 100)]}


def timed_case(core, label, settings, require_active):
  controller = core.LaneCenteringController('absolute')
  model = model_input(core, **settings)
  curvature = settings.get('curvature', 0.0)
  previous_selected = curvature
  base_curvature = controller._plan_curvature(model['plan'], SPEED, ACTION_T)
  cpu_ms, wall_ms, states, reasons, sources = [], [], Counter(), Counter(), Counter()
  path_weights = []
  prepared_controller = None
  for frame in range(WARMUP + FRAMES):
    if label == 'path_infeasible' and frame >= WARMUP:
      if prepared_controller is None:
        # Reach active normally, then exercise one independently repeated bad
        # model-speed frame. Snapshot copying is outside the measured interval.
        prepared_controller = deepcopy(controller)
        model['plan'][0, :, core.Plan.VELOCITY.start] = 80.0
        curvature = 0.005
      controller = deepcopy(prepared_controller)
    wall_start = time.perf_counter_ns()
    cpu_start = time.process_time_ns()
    selected, status = controller.update(
      model, SPEED, curvature, ACTION_T, DT, base_curvature, previous_selected,
      True, True, False, False, False,
    )
    cpu_end = time.process_time_ns()
    wall_end = time.perf_counter_ns()
    previous_selected = core.smooth_value(
      controller._plan_curvature(selected['plan'], SPEED, ACTION_T), previous_selected,
      core.ACTION_SMOOTH_SECONDS,
    )
    if frame >= WARMUP:
      cpu_ms.append((cpu_end - cpu_start) / 1e6)
      wall_ms.append((wall_end - wall_start) / 1e6)
      states[status.state] += 1
      reasons[status.reason] += 1
      sources[status.source] += 1
      path_weights.append(status.path_weight)
  active_ok = states.get('active', 0) == FRAMES and min(path_weights) > 0.999
  result = {
    'case': label, 'frames': FRAMES, 'warmup_frames': WARMUP,
    'cpu': percentiles(cpu_ms), 'wall': percentiles(wall_ms),
    'states': dict(states), 'reasons': dict(reasons), 'sources': dict(sources),
    'minimum_path_weight': float(min(path_weights)), 'active_required': require_active,
    'active_verified': active_ok,
    'final_geometry_horizon_m': float(getattr(status, 'geometry_horizon', 30.0)),
    'final_convergence_distance_m': float(getattr(status, 'convergence_distance', 44.0)),
  }
  return result


def synthetic_response(core, initial_offset):
  """Exact straight-road observation and assumed first-order curvature actuator.

  Bypass acquisition, fallback gates, and perception filtering deliberately:
  compare the active path generators, not the complete driving system.
  """
  controller = core.LaneCenteringController('absolute')
  model = model_input(core)
  native_x = np.asarray(core.ModelConstants.X_IDXS, dtype=np.float64)
  plan_fields = core.Plan
  base_plan = model['plan']
  base_plan[0, :, plan_fields.POSITION.start + 1] = 0.0
  lateral_position, heading, curvature, previous_command = -initial_offset, 0.0, 0.0, 0.0
  actuator_tau = 0.20
  actuator_alpha = 1.0 - math.exp(-DT / actuator_tau)
  times, offsets, accelerations, commands, joins = [0.0], [initial_offset], [], [], []
  failed_builds = 0
  for frame in range(240):
    # Exact straight world centerline expressed in the current vehicle frame.
    controller.filtered_center_y = -lateral_position / math.cos(heading) - math.tan(heading) * native_x
    if hasattr(controller, 'geometry_horizon'):
      controller.geometry_horizon = 60.0
    lane_plan, join = controller._build_lane_plan(base_plan, SPEED, curvature)
    if lane_plan is None:
      failed_builds += 1
      requested = 0.0
    else:
      requested = controller._plan_curvature(lane_plan, SPEED, ACTION_T)
      joins.append(float(join))
    command = core.smooth_value(requested, previous_command, core.ACTION_SMOOTH_SECONDS)
    previous_command = command
    curvature += actuator_alpha * (command - curvature)
    heading += SPEED * curvature * DT
    lateral_position += SPEED * math.sin(heading) * DT
    times.append((frame + 1) * DT)
    offsets.append(-lateral_position)
    accelerations.append(curvature * SPEED ** 2)
    commands.append(command)
  absolute_offsets = np.abs(offsets)
  # Settled means remaining within 0.1 m for the rest of this finite run.
  unsettled = np.flatnonzero(absolute_offsets > 0.1)
  settled_index = int(unsettled[-1] + 1) if unsettled.size else 0
  settling = times[settled_index] if settled_index < len(times) else None
  snapshots = {str(second): float(offsets[int(round(second / DT))])
               for second in (0, 1, 2, 3, 4, 6, 8, 10, 12)}
  return {
    'initial_offset_m': initial_offset, 'duration_s': 12.0,
    'settling_time_within_0_1m_s': settling,
    'offset_at_seconds_m': snapshots,
    'maximum_overshoot_m': max(0.0, float(-min(offsets))),
    'peak_actual_lateral_acceleration_mps2': float(max(np.abs(accelerations))),
    'peak_actual_lateral_jerk_mps3': float(max(np.abs(np.diff(accelerations) / DT))),
    'failed_plan_builds': failed_builds,
    'convergence_distance_range_m': [min(joins), max(joins)] if joins else None,
  }


def run_benchmark(baseline, current):
  cases = [
    ('full_centered', {'offset': 0.0}, True, True),
    ('full_offset', {'offset': 0.44}, True, True),
    ('full_curve', {'offset': 0.12, 'curvature': 0.0015}, True, True),
    ('three_sources', {'three_sources': True}, True, True),
    ('short_support_20m', {'support': 20.0}, False, True),
    ('policy_veto', {'policy_shift': 1.0}, False, False),
    ('path_infeasible', {'offset': 0.12}, False, False),
  ]
  report = {
    'scope': 'Pure controller.update CPU/wall benchmark; no inference, messaging, live commands or installed-code changes.',
    'timing_inputs': 'Deterministic repeated model observations at 16 m/s; normal acquisition, no active-state seeding. Timing excludes input generation and action extraction.',
    'infeasible_case': 'After normal acquisition, repeat isolated snapshots with model speed 80 m/s and measured curvature 0.005/m at vehicle speed 16 m/s to force full-plan feasibility failure. Snapshot copying is untimed.',
    'simulation_assumptions': {
      'scope': 'Synthetic exact straight-road geometry, active path generator only; acquisition/fallback/perception filtering bypassed. This is not road validation.',
      'speed_mps': SPEED, 'step_s': DT, 'action_preview_s': ACTION_T,
      'command_smoothing_s': 0.1, 'first_order_actuator_tau_s': 0.20,
      'actuator_transport_delay_s': 0.0,
      'geometry_support_m': 60.0,
    },
    'timings': {}, 'synthetic_response': {},
  }
  for name, core in [('baseline', baseline), ('current', current)]:
    report['timings'][name] = [timed_case(core, label, settings, old_required if name == 'baseline' else new_required)
                               for label, settings, old_required, new_required in cases]
    report['synthetic_response'][name] = [synthetic_response(core, offset) for offset in (0.44, 0.8)]
  report['active_checks_passed'] = all(
    not row['active_required'] or row['active_verified']
    for rows in report['timings'].values() for row in rows
  )
  return report


if 'baseline' in globals() and 'current' in globals():
  BENCHMARK_REPORT = run_benchmark(baseline, current)
  print(json.dumps(BENCHMARK_REPORT, indent=2, allow_nan=False))

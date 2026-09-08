"""Independent spatial and closed-loop regressions for lane containment.

The footprint oracle and simple delayed vehicle below do not call the production
containment implementation. This is a deterministic regression plant, not a
validated vehicle model or a substitute for route replay and controlled testing.
"""
import copy
import json
from collections import deque
from pathlib import Path

import numpy as np
import pytest

from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import MAX_LATERAL_JERK, clip_curvature, get_curvature_from_plan, smooth_value
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan
from openpilot.selfdrive.modeld.lane_centering import ACTION_SMOOTH_SECONDS, CAMERA_OFFSET, LaneCenteringController
from openpilot.selfdrive.modeld import lane_centering as lane_core
from openpilot.selfdrive.modeld.lane_path import Corridor, LanePath, LaneReference, build_lane_candidates


HALF_WIDTH = 1.08
HALF_LENGTH = 2.5
TIMES = np.asarray(ModelConstants.T_IDXS)
DISTANCES = np.asarray(ModelConstants.X_IDXS)
FIXTURES = json.loads((Path(__file__).parent / 'fixtures/lane_boundary_regressions.json').read_text())['cases']
NATIVE_TURN = json.loads((Path(__file__).parent / 'fixtures/native_ninety_degree_turn.json').read_text())


def fixture_output(case):
  n = ModelConstants.IDX_N
  plan = np.zeros((1, n, ModelConstants.PLAN_WIDTH))
  for group, section in [('position', Plan.POSITION), ('velocity', Plan.VELOCITY), ('acceleration', Plan.ACCELERATION),
                          ('orientation', Plan.T_FROM_CURRENT_EULER), ('orientationRate', Plan.ORIENTATION_RATE)]:
    for axis, values in case['plan'][group].items():
      plan[0, :, section.start + 'xyz'.index(axis)] = values
  lines = np.zeros((1, 4, n, 2))
  lines[0, :, :, 0] = case['lane_lines_y']
  edges = np.zeros((1, 2, n, 2))
  edges[0, :, :, 0] = case['road_edges_y']
  lane_stds = np.broadcast_to(np.asarray(case['lane_std'])[None, :, None, None], lines.shape).copy()
  edge_stds = np.broadcast_to(np.asarray(case['road_edge_std'])[None, :, None, None], edges.shape).copy()
  # Logs expose scalar uncertainty, not the per-point raw confidence head.
  # Preserve the inspected near geometry. NaN explicitly means the far head is
  # unknown; a finite sentinel would invent a confidence-threshold crossing.
  confidence_end = DISTANCES[np.searchsorted(DISTANCES, 30.0)]
  lane_stds[:, :, DISTANCES > confidence_end] = np.nan
  edge_stds[:, :, DISTANCES > confidence_end] = np.nan
  return {'plan': plan, 'lane_lines': lines, 'lane_lines_stds': lane_stds,
          'lane_lines_prob': np.repeat(case['lane_probabilities'], 2)[None],
          'road_edges': edges,
          'road_edges_stds': edge_stds,
          'desire_state': np.zeros((1, 8))}


def mirror_output(output):
  mirrored = copy.deepcopy(output)
  mirrored['lane_lines'] = -mirrored['lane_lines'][:, ::-1]
  # Device-frame camera offset is fixed; mirror around vehicle center instead.
  mirrored['lane_lines'][0, :, :, 0] -= 2 * CAMERA_OFFSET
  mirrored['lane_lines_stds'] = mirrored['lane_lines_stds'][:, ::-1]
  mirrored['lane_lines_prob'] = mirrored['lane_lines_prob'].reshape(1, 4, 2)[:, ::-1].reshape(1, 8)
  mirrored['road_edges'] = -mirrored['road_edges'][:, ::-1]
  mirrored['road_edges'][0, :, :, 0] -= 2 * CAMERA_OFFSET
  mirrored['road_edges_stds'] = mirrored['road_edges_stds'][:, ::-1]
  for index in [Plan.POSITION.start + 1, Plan.VELOCITY.start + 1, Plan.ACCELERATION.start + 1,
                Plan.T_FROM_CURRENT_EULER.start + 2, Plan.ORIENTATION_RATE.start + 2]:
    mirrored['plan'][0, :, index] *= -1
  return mirrored


def boundary_y(x, values):
  """Observed forward boundary with a local tangent for the rear at origin."""
  result = np.interp(x, DISTANCES, values)
  tangent = (values[1] - values[0]) / (DISTANCES[1] - DISTANCES[0])
  return np.where(x < 0, values[0] + tangent * x, result)


def footprint_clearances(plan, output, horizon=2.0):
  dense_time = np.linspace(0, horizon, round(horizon / 0.02) + 1)
  px = np.interp(dense_time, TIMES, plan[0, :, Plan.POSITION.start])
  py = np.interp(dense_time, TIMES, plan[0, :, Plan.POSITION.start + 1])
  yaw = np.interp(dense_time, TIMES, plan[0, :, Plan.T_FROM_CURRENT_EULER.start + 2])
  left = output['lane_lines'][0, 1, :, 0] + CAMERA_OFFSET
  right = output['lane_lines'][0, 2, :, 0] + CAMERA_OFFSET
  clearances = []
  # Corners and side midpoints catch curved boundaries crossing body sides.
  for longitudinal in np.linspace(-HALF_LENGTH, HALF_LENGTH, 5):
    for lateral in (-HALF_WIDTH, HALF_WIDTH):
      x = px + longitudinal * np.cos(yaw) - lateral * np.sin(yaw)
      y = py + longitudinal * np.sin(yaw) + lateral * np.cos(yaw)
      clearances.extend([y - boundary_y(x, left), boundary_y(x, right) - y])
  return dense_time, np.min(clearances, axis=0)


def selected_step(controller, output, speed, curvature, previous_base=0.0, previous_selected=0.0, **conditions):
  action_time = 0.475
  frame_dt = conditions.get('frame_dt', DT_MDL)
  base = get_curvature_from_plan(output['plan'][0, :, Plan.T_FROM_CURRENT_EULER.start + 2],
                                 output['plan'][0, :, Plan.ORIENTATION_RATE.start + 2], TIMES, speed, action_time)
  base = smooth_value(base, previous_base, ACTION_SMOOTH_SECONDS, dt=frame_dt)
  arguments = {'lat_active': True, 'model_valid': True, 'left_blinker': False, 'right_blinker': False,
               'lane_change_active': False, 'frame_dt': DT_MDL}
  arguments.update(conditions)
  selected, status = controller.update(output, speed, curvature, action_time, arguments.pop('frame_dt'),
                                       base, previous_selected, **arguments)
  target = get_curvature_from_plan(selected['plan'][0, :, Plan.T_FROM_CURRENT_EULER.start + 2],
                                   selected['plan'][0, :, Plan.ORIENTATION_RATE.start + 2], TIMES, speed, action_time)
  action = base if getattr(status, 'policy_fallback', False) else smooth_value(
    target, previous_selected, ACTION_SMOOTH_SECONDS, dt=frame_dt)
  return selected, status, base, action


def road_output(speed, pose, road_curvature=0.0, width=3.6):
  x, y, heading = pose
  if abs(road_curvature) < 1e-8:
    station = x
  else:
    station = np.arctan2(road_curvature * x, 1 - road_curvature * y) / road_curvature
  s = station + np.linspace(-15, 300, 1261)
  theta = road_curvature * s
  if abs(road_curvature) < 1e-8:
    center_x, center_y = s, np.zeros_like(s)
  else:
    center_x = np.sin(theta) / road_curvature
    center_y = (1 - np.cos(theta)) / road_curvature

  def lane(offset):
    world_x = center_x - offset * np.sin(theta)
    world_y = center_y + offset * np.cos(theta)
    local_x = (world_x - x) * np.cos(heading) + (world_y - y) * np.sin(heading)
    local_y = -(world_x - x) * np.sin(heading) + (world_y - y) * np.cos(heading)
    return np.interp(DISTANCES, local_x, local_y)

  lines = np.zeros((1, 4, len(DISTANCES), 2))
  for i, offset in enumerate((-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width)):
    lines[0, i, :, 0] = lane(offset) - CAMERA_OFFSET
  edges = lines[:, [0, 3]].copy()
  plan = np.zeros((1, len(TIMES), ModelConstants.PLAN_WIDTH))
  center_local_x = (center_x - x) * np.cos(heading) + (center_y - y) * np.sin(heading)
  center_local_y = -(center_x - x) * np.sin(heading) + (center_y - y) * np.cos(heading)
  center_slope = np.tan(theta - heading)
  # A deliberately distant native-policy connector preserves road intent while
  # letting the lane controller improve its near-term convergence. Everything
  # starts at the actual ego origin with consistent yaw and curvature.
  join = 60.0
  target_y = np.interp(join, center_local_x, center_local_y)
  target_slope = np.interp(join, center_local_x, center_slope)
  target_second = road_curvature * (1 + target_slope**2)**1.5
  quadratic = 0.5 * road_curvature * join**2
  coefficients = np.r_[0., 0., quadratic, np.linalg.solve(
    [[1, 1, 1], [3, 4, 5], [6, 12, 20]],
    [target_y - quadratic, target_slope * join - 2 * quadratic, target_second * join**2 - 2 * quadratic])]

  def policy_path(plan_x):
    u = np.minimum(plan_x / join, 1)
    py = np.polynomial.polynomial.polyval(u, coefficients)
    slope = np.polynomial.polynomial.polyval(u, np.arange(1, 6) * coefficients[1:]) / join
    second = np.polynomial.polynomial.polyval(u, np.arange(1, 5) * np.arange(2, 6) * coefficients[2:]) / join**2
    beyond = plan_x > join
    py[beyond] = np.interp(plan_x[beyond], center_local_x, center_local_y)
    slope[beyond] = np.interp(plan_x[beyond], center_local_x, center_slope)
    second[beyond] = road_curvature * (1 + slope[beyond]**2)**1.5
    return py, slope, second

  plan_x = speed * TIMES
  for _ in range(3):
    _, slope, _ = policy_path(plan_x)
    vx = speed / np.sqrt(1 + slope**2)
    plan_x[1:] = np.cumsum(np.diff(TIMES) * (vx[:-1] + vx[1:]) / 2)
  plan_y, slope, second = policy_path(plan_x)
  plan_curvature = second / (1 + slope**2)**1.5
  plan[0, :, Plan.POSITION.start] = plan_x
  plan[0, :, Plan.POSITION.start + 1] = plan_y
  plan[0, :, Plan.VELOCITY.start] = speed
  plan[0, :, Plan.ACCELERATION.start + 1] = plan_curvature * speed**2
  plan[0, :, Plan.T_FROM_CURRENT_EULER.start + 2] = np.arctan(slope)
  plan[0, :, Plan.ORIENTATION_RATE.start + 2] = plan_curvature * speed
  return {'plan': plan, 'lane_lines': lines, 'lane_lines_stds': np.full_like(lines, 0.03),
          'lane_lines_prob': np.full((1, 8), 0.98), 'road_edges': edges,
          'road_edges_stds': np.full_like(edges, 0.8), 'desire_state': np.zeros((1, 8))}


def simulate_offset(offset, speed=22.0, delay=0.3, road_curvature=0.0, duration=7.0):
  controller = LaneCenteringController('absolute')
  pose = np.array([0.0, offset, 0.0])
  measured = road_curvature / (1 - offset * road_curvature)
  requested = target = previous_base = previous_selected = measured
  commands = deque([measured] * max(1, round(delay / DT_CTRL)))
  history = []
  for frame in range(round(duration / DT_CTRL)):
    if frame % round(DT_MDL / DT_CTRL) == 0:
      output = road_output(speed, pose, road_curvature)
      selected, status, previous_base, previous_selected = selected_step(
        controller, output, speed, measured, previous_base, previous_selected)
      target = previous_selected
      checked_distance = getattr(status, 'checked_distance', 26.0)
      horizon = np.interp(checked_distance, selected['plan'][0, :, Plan.POSITION.start], TIMES)
      _, footprint = footprint_clearances(selected['plan'], output, horizon=horizon)
      _, origin_footprint = footprint_clearances(np.zeros_like(selected['plan']), output, horizon=0.0)
      center = (output['lane_lines'][0, 1, 0, 0] + output['lane_lines'][0, 2, 0, 0]) / 2 + CAMERA_OFFSET
      history.append({'time': frame * DT_CTRL, 'offset': -center, 'authority': status.authority,
                      'clearance': float(footprint.min()), 'blocked': getattr(status, 'safety_blocked', False),
                      'origin_clearance': float(origin_footprint[0]),
                      'reason': status.reason, 'response_time': getattr(status, 'response_time', np.nan),
                      'target': target, 'measured': measured})
    controls_request, _ = clip_curvature(speed, requested, target, 0.0)
    # Final C3X command jerk cap, expressed as an equivalent acceleration slew
    # limit. This conservative surrogate does not model measured jerk or the
    # NN torque plant; both control stages use the same production limit.
    curvature_step = MAX_LATERAL_JERK * DT_CTRL / max(speed**2, 1.0)
    requested = np.clip(controls_request, requested - curvature_step, requested + curvature_step)
    commands.append(requested)
    delayed = commands.popleft()
    measured += (1 - np.exp(-DT_CTRL / 0.1)) * (delayed - measured)
    # Integrate a delayed curvature-controlled vehicle with midpoint heading.
    change = speed * measured * DT_CTRL
    pose[:2] += speed * DT_CTRL * np.array([np.cos(pose[2] + change / 2), np.sin(pose[2] + change / 2)])
    pose[2] += change
  return history


@pytest.mark.parametrize('offset', [-0.5, 0.5])
@pytest.mark.parametrize('speed,delay', [(5.0, 0.1), (22.0, 0.3), (30.0, 0.3)])
def test_closed_loop_offset_converges_without_penetration(offset, speed, delay):
  history = simulate_offset(offset, speed, delay)
  active = [row for row in history if row['authority'] > 0.95]
  assert active, 'clear straight-road geometry never acquired'
  assert not any(row['blocked'] for row in active), 'ordinary recoverable offset unexpectedly blocked'
  assert min(row['clearance'] for row in active) >= -0.03
  assert max(abs(row['offset']) for row in history) <= abs(offset) + 0.1
  assert abs(history[-1]['offset']) < 0.15, history[-1]
  # Realized origin footprint, not an optimistic controller clearance field.
  assert min(row['origin_clearance'] for row in history) > 0


@pytest.mark.parametrize('direction', [-1.0, 1.0])
def test_closed_loop_curved_lane_with_delay(direction):
  history = simulate_offset(direction * 0.4, speed=22.0, delay=0.3, road_curvature=direction * 0.004)
  assert not any(row['blocked'] for row in history if row['authority'] > 0.95)
  assert min(row['clearance'] for row in history if row['authority'] > 0.95) >= -0.03
  assert abs(history[-1]['offset']) < 0.18, history[-1]
  assert max(abs(row['offset']) for row in history) < 0.62
  assert min(row['origin_clearance'] for row in history) > 0


@pytest.mark.parametrize('case', FIXTURES, ids=lambda case: case['name'])
@pytest.mark.parametrize('mirror', [False, True], ids=['recorded_direction', 'mirrored_direction'])
def test_recorded_boundary_case_never_labels_intruding_path_contained(case, mirror):
  output = fixture_output(case)
  curvature = case['current_curvature']
  if mirror:
    output, curvature = mirror_output(output), -curvature
  controller = LaneCenteringController('absolute')
  previous_base = previous_selected = curvature
  for _ in range(45):
    selected, status, previous_base, previous_selected = selected_step(
      controller, output, case['speed'], curvature, previous_base, previous_selected)
    assert controller.corridor is not None
    assert controller.corridor.horizon <= DISTANCES[np.searchsorted(DISTANCES, 30.0)]
  checked_distance = getattr(status, 'checked_distance', 0.0)
  horizon = np.interp(checked_distance, selected['plan'][0, :, Plan.POSITION.start], TIMES)
  _, clearances = footprint_clearances(selected['plan'], output, horizon=horizon)
  if case['name'] != 'right_inside_recovery':
    # Safe fallback alone does not establish that the recorded turn can use
    # the lane tracker. These four snapshots have a feasible contained path.
    assert status.containment == 'contained', status
    assert status.lane_path_feasibility == 1.0, status
    assert status.path_weight > 0.95, status
  if getattr(status, 'containment', '') == 'contained':
    assert checked_distance >= 26.0
    assert clearances.min() >= -0.03, (case['name'], status, clearances.min())
  elif clearances.min() < -0.05:
    assert getattr(status, 'safety_blocked', False) or getattr(status, 'containment', '') == 'recovering', (case['name'], status)


def test_fixture_contains_only_anonymized_geometry():
  allowed = {'name', 'description', 'speed', 'current_curvature', 'lane_lines_y', 'lane_probabilities', 'lane_std',
             'road_edges_y', 'road_edge_std', 'plan'}
  for case in FIXTURES:
    assert set(case) == allowed
    assert np.asarray(case['lane_lines_y']).shape == (4, ModelConstants.IDX_N)
    assert np.asarray(case['road_edges_y']).shape == (2, ModelConstants.IDX_N)


def acquired_controller(output, speed=22.0):
  controller = LaneCenteringController('absolute')
  previous_base = previous_selected = 0.0
  for _ in range(40):
    _, status, previous_base, previous_selected = selected_step(controller, output, speed, 0.0, previous_base, previous_selected)
  assert status.authority > 0.95, status
  return controller, previous_base, previous_selected


@pytest.mark.parametrize('index,value', [(Plan.POSITION.start, 0.1), (Plan.POSITION.start + 1, 0.2),
                                        (Plan.T_FROM_CURRENT_EULER.start + 2, 0.04)])
def test_graph_incompatible_native_origin_releases_only_lane_override(index, value):
  output = road_output(22.0, np.array([0., 0., 0.]))
  controller, previous_base, previous_selected = acquired_controller(output)
  output['plan'][0, 0, index] = value
  selected, status, base_action, selected_action = selected_step(controller, output, 22.0, 0.0, previous_base, previous_selected)
  assert np.array_equal(selected['plan'], output['plan'])
  assert np.isfinite(selected_action) and selected_action == base_action
  assert status.path_weight == 0.0
  assert status.containment == 'unavailable'
  assert status.reason == 'native_path_geometry'
  assert not status.safety_blocked
  assert not status.collision_risk


@pytest.mark.parametrize('width', [2.0, 2.15])
def test_positive_but_narrow_corridor_blocks_unsafe_raw_fallback(width):
  output = road_output(22.0, np.zeros(3), width=width)
  _, status, _, _ = selected_step(LaneCenteringController('absolute'), output, 22.0, 0.0)
  assert status.safety_blocked
  assert status.containment == 'blocked'


def test_confidence_loss_retains_limits_briefly_then_blocks():
  output = road_output(22.0, np.zeros(3))
  controller, previous_base, previous_selected = acquired_controller(output)
  output['lane_lines_prob'][:] = 0.0
  for _ in range(3):
    _, status, previous_base, previous_selected = selected_step(controller, output, 22.0, 0.0, previous_base, previous_selected)
    assert not status.safety_blocked
  for _ in range(5):
    _, status, previous_base, previous_selected = selected_step(controller, output, 22.0, 0.0, previous_base, previous_selected)
  assert status.safety_blocked
  assert status.reason == 'corridor_lost'


@pytest.mark.parametrize('fault', ['nan_plan', 'stale_frame', 'invalid_model'])
def test_invalid_model_data_blocks_instead_of_certifying_fallback(fault):
  output = road_output(22.0, np.zeros(3))
  controller, previous_base, previous_selected = acquired_controller(output)
  conditions = {}
  if fault == 'nan_plan':
    output['plan'][0, 10, Plan.POSITION.start + 1] = np.nan
  elif fault == 'stale_frame':
    conditions['frame_dt'] = 0.5
  else:
    conditions['model_valid'] = False
  _, status, _, _ = selected_step(controller, output, 22.0, 0.0, previous_base, previous_selected, **conditions)
  assert status.safety_blocked
  assert status.containment == 'blocked'


@pytest.mark.parametrize('intent', ['left_blinker', 'right_blinker', 'lane_change_active'])
def test_explicit_lane_change_releases_selection_without_false_containment(intent):
  output = road_output(22.0, np.zeros(3))
  controller, previous_base, previous_selected = acquired_controller(output)
  selected, status, _, _ = selected_step(controller, output, 22.0, 0.0, previous_base, previous_selected, **{intent: True})
  assert status.containment == 'bypassed'
  assert not status.safety_blocked
  assert np.array_equal(selected['plan'], output['plan'])


def test_repeated_outside_origin_cannot_restart_recovery_deadline():
  output = road_output(22.0, np.array([0., 0.8, 0.]))
  controller = LaneCenteringController('absolute')
  previous_base = previous_selected = 0.0
  for _ in range(45):
    _, status, previous_base, previous_selected = selected_step(controller, output, 22.0, 0.0, previous_base, previous_selected)
  assert status.safety_blocked
  assert status.reason == 'recovery_timeout'


@pytest.mark.parametrize('mirror', [False, True])
def test_trusted_paint_does_not_move_reference_to_shoulder(mirror):
  case = next(case for case in FIXTURES if case['name'] == 'right_shoulder_handoff')
  output = fixture_output(case)
  # The shoulder is also credible. It must not displace two good painted lines.
  output['road_edges_stds'][:] = 0.05
  if mirror:
    output = mirror_output(output)
  controller = LaneCenteringController('absolute')
  previous_base = previous_selected = curvature = case['current_curvature'] * (-1 if mirror else 1)
  sources = []
  for _ in range(40):
    _, status, previous_base, previous_selected = selected_step(controller, output, case['speed'], curvature,
                                                               previous_base, previous_selected)
    sources.append(status.source)
  assert 'lane_lines' in sources
  assert not ({'left_line_right_edge', 'left_edge_right_line'} & set(sources))


@pytest.mark.parametrize('direction', [-1.0, 1.0])
def test_unsafe_native_avoidance_path_requires_takeover(direction):
  output = road_output(22.0, np.zeros(3))
  controller, previous_base, previous_selected = acquired_controller(output)
  plan = output['plan'][0]
  x = plan[:, Plan.POSITION.start]
  slope = direction * 0.004 * x
  curvature = direction * 0.004 / (1 + slope**2)**1.5
  plan[:, Plan.POSITION.start + 1] = direction * 0.002 * x**2
  plan[:, Plan.T_FROM_CURRENT_EULER.start + 2] = np.arctan(slope)
  plan[:, Plan.ORIENTATION_RATE.start + 2] = curvature * 22.0
  plan[:, Plan.ACCELERATION.start + 1] = curvature * 22.0**2
  for _ in range(5):
    selected, status, previous_base, previous_selected = selected_step(controller, output, 22.0, 0.0,
                                                                       previous_base, previous_selected)
    _, margin = footprint_clearances(selected['plan'], output, horizon=26.0 / 22.0)
    if margin.min() < -0.03:
      assert status.safety_blocked
  assert status.safety_blocked
  assert status.containment == 'blocked'
  assert status.reason == 'unsafe_policy_path'


@pytest.mark.parametrize('direction', [-1.0, 1.0])
def test_full_body_check_catches_boundary_notch_between_corners(direction):
  left, right = np.full(len(DISTANCES), -1.8), np.full(len(DISTANCES), 1.8)
  # All four body corners remain clear. An observed concave boundary knot lies
  # inside a body side, and the exact minimum is known without calling an oracle.
  if direction < 0:
    left[3] = -0.5
  else:
    right[3] = 0.5
  margin = Corridor(left, right).margins(np.array([DISTANCES[3] + 0.3]), np.zeros(1), np.zeros(1))[0]
  assert margin <= -0.58


def test_unmarked_intersection_does_not_claim_containment():
  output = road_output(10.0, np.zeros(3))
  output['lane_lines_prob'][:] = 0.0
  _, status, _, _ = selected_step(LaneCenteringController('absolute'), output, 10.0, 0.0)
  assert status.containment == 'unavailable'
  assert status.authority == 0


def test_standstill_releases_selection_without_false_containment():
  output = road_output(0.0, np.zeros(3))
  _, status, _, _ = selected_step(LaneCenteringController('absolute'), output, 0.0, 0.0)
  assert status.containment == 'bypassed'
  assert not status.safety_blocked


def retained_boundary_value(query, x, y):
  """Independent piecewise-linear oracle, including both terminal tangents."""
  result = np.interp(query, x, y)
  result = np.where(query < x[0], y[0] + (query - x[0]) * (y[1] - y[0]) / (x[1] - x[0]), result)
  return np.where(query > x[-1], y[-1] + (query - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2]), result)


@pytest.mark.parametrize('notch_index', [7, 13], ids=['inside_horizon', 'supported_endpoint'])
@pytest.mark.parametrize('side', [1, 2], ids=['left_boundary', 'right_boundary'])
@pytest.mark.parametrize('curvature', [-0.01, 0.0, 0.01], ids=['left_turn', 'straight', 'right_turn'])
def test_retained_corridor_never_widens_transformed_boundary(side, notch_index, curvature):
  output = road_output(22.0, np.zeros(3))
  sign = -1 if side == 1 else 1
  output['lane_lines'][0, side, notch_index, 0] = sign * 1.2 - CAMERA_OFFSET
  output['lane_lines_stds'][:, :, 14:] = 1.0
  controller = LaneCenteringController('absolute')
  controller.filtered_center_y = np.linspace(-0.2, 0.3, len(DISTANCES))
  saved_center = controller.filtered_center_y.copy()
  controller._refresh_corridor(output, 22.0, curvature, 0.475)
  original = controller.corridor
  assert original is not None

  output['lane_lines_prob'][:] = 0.0
  controller._refresh_corridor(output, 22.0, curvature, 0.475)
  retained = controller.corridor
  assert retained is not None
  assert np.array_equal(controller.filtered_center_y, saved_center)

  distance = 22.0 * DT_MDL
  angle = curvature * distance
  dx = np.sin(angle) / curvature if curvature else distance
  dy = (1 - np.cos(angle)) / curvature if curvature else 0.0
  transformed = []
  endpoints = []
  for values in (original.left, original.right):
    x = (DISTANCES - dx) * np.cos(angle) + (values - dy) * np.sin(angle)
    y = -(DISTANCES - dx) * np.sin(angle) + (values - dy) * np.cos(angle)
    transformed.append((x, y))
    endpoints.append((original.horizon - dx) * np.cos(angle) +
                     (np.interp(original.horizon, DISTANCES, values) - dy) * np.sin(angle))
  assert retained.horizon == pytest.approx(min(endpoints), abs=1e-10)

  # Include the rear tangent, both domain ends, and every original/retained
  # boundary knot as well as a dense independent set of positions.
  start = -HALF_LENGTH - HALF_WIDTH
  query = np.unique(np.r_[np.linspace(start, retained.horizon, 1001), DISTANCES,
                          transformed[0][0], transformed[1][0]])
  query = query[(query >= start) & (query <= retained.horizon)]
  for index, new_values in enumerate((retained.left, retained.right)):
    actual = retained_boundary_value(query, *transformed[index])
    reconstructed = retained_boundary_value(query, DISTANCES, new_values)
    assert np.min((reconstructed - actual) * (1 if index == 0 else -1)) >= -1e-10

  if notch_index == 7 and curvature == 0:
    # Previously this actual 10 cm intrusion was falsely given 13.5 cm clearance
    # because resampling smoothed the retained notch outward by 27.7 cm.
    position = DISTANCES[notch_index] - distance
    margin = retained.margins(np.array([position]), np.array([sign * 0.2]), np.zeros(1))[0]
    assert margin <= -0.1 + 1e-10


def test_retained_corridor_grace_reduces_horizon_and_expires():
  output = road_output(22.0, np.zeros(3))
  output['lane_lines'][0, 1, 7, 0] = -1.2 - CAMERA_OFFSET
  output['lane_lines_stds'][:, :, 14:] = 1.0
  controller = LaneCenteringController('absolute')
  controller._refresh_corridor(output, 22.0, 0.0, 0.475)
  original_horizon = controller.corridor.horizon
  output['lane_lines_prob'][:] = 0.0
  for frame in range(1, 6):
    controller._refresh_corridor(output, 22.0, 0.0, 0.475)
    assert controller.corridor is not None
    assert controller.corridor.horizon == pytest.approx(original_horizon - frame * 22.0 * DT_MDL, abs=1e-10)
  for _ in range(2):
    controller._refresh_corridor(output, 22.0, 0.0, 0.475)
  assert controller.corridor is None
  assert controller.safety_blocked
  assert controller.safety_reason == 'corridor_lost'


@pytest.mark.parametrize('elapsed', [0.01, 0.275, 0.3, 0.301])
def test_retained_corridor_uses_actual_accepted_frame_interval(elapsed):
  output = road_output(22.0, np.zeros(3))
  output['lane_lines_stds'][:, :, 14:] = 1.0
  controller = LaneCenteringController('absolute')
  _, _, previous_base, previous_selected = selected_step(controller, output, 22.0, 0.0)
  original_horizon = controller.corridor.horizon
  output['lane_lines_prob'][:] = 0.0
  _, status, _, _ = selected_step(controller, output, 22.0, 0.0, previous_base, previous_selected, frame_dt=elapsed)
  if elapsed > 0.3:
    assert status.safety_blocked
    assert status.reason == 'model_gap'
    assert controller.corridor is None
  else:
    assert not status.safety_blocked, status
    assert controller.motion_dt == elapsed
    assert controller.corridor.horizon == pytest.approx(original_horizon - 22.0 * elapsed, abs=1e-10)


def candidate_selection_case(name):
  for case in FIXTURES:
    if case['name'] == name:
      return fixture_output(case), case['speed'], case['current_curvature']
  speed, curvature = 22.0, 0.0
  offset = {'straight_centered': 0.0, 'straight_roundoff': 0.0, 'straight_offset': 0.5, 'curved_offset': 0.4,
            'curved_centered': 0.0, 'curved_centered_fast': 0.0,
            'noisy_markings': 0.15, 'narrow_no_path': 0.0, 'curvature_no_path': 0.0}[name]
  if name in ('curved_offset', 'curved_centered'):
    curvature = 0.004
  elif name == 'curved_centered_fast':
    speed, curvature = 30.0, 0.002
  output = road_output(speed, np.array([0., offset, 0.]), curvature, width=2.0 if name == 'narrow_no_path' else 3.6)
  if name == 'noisy_markings':
    perturbation = 0.25 * np.sin(DISTANCES / 25) + 0.05 * np.sin(DISTANCES / 7) * np.exp(-DISTANCES / 80)
    output['lane_lines'][0, :, :, 0] += perturbation
  if name == 'curvature_no_path':
    curvature = 0.05
  return output, speed, curvature


def exhaustive_candidate_minimum(controller, base, speed, curvature):
  """Enumerate every candidate; do not use the optimized selector or cache."""
  center = controller.filtered_center_y
  reference_y = center.copy()
  corridor = controller.corridor
  if corridor is not None:
    u = np.clip((DISTANCES - corridor.horizon) / max(speed, 10.0), 0, 1)
    weight = u**3 * (10 - 15*u + 6*u**2)
    native = retained_boundary_value(DISTANCES, base[0, :, Plan.POSITION.start], base[0, :, Plan.POSITION.start + 1])
    reference_y = (1 - weight) * center + weight * native
  reference = LaneReference(reference_y, speed)
  distances = np.maximum(1.0, speed * np.linspace(lane_core.MIN_RESPONSE_TIME, lane_core.MAX_RESPONSE_TIME, 28))
  end = min(float(base[0, -1, Plan.POSITION.start]), corridor.horizon if corridor is not None else 30.0)
  cost_x = np.linspace(0, max(end, 1.0), 101)
  accel_limit = lane_core.MAX_LANE_PATH_ACCEL * lane_core.SPATIAL_DYNAMICS_MARGIN
  jerk_limit = lane_core.MAX_LANE_PATH_JERK * lane_core.SPATIAL_DYNAMICS_MARGIN
  plans, acceleration, jerk, y = build_lane_candidates(base, reference, curvature, distances, cost_x,
                                                      accel_limit, jerk_limit, lane_core.TIME_EPSILON)
  cost = np.mean((y - np.interp(cost_x, DISTANCES, center))**2 * (0.2 + np.exp(-cost_x / max(speed, 1.0))), axis=1)
  cost += 0.00005 * jerk**2
  feasible = []
  for index in range(len(distances)):
    if acceleration[index] > accel_limit or jerk[index] > jerk_limit or not np.isfinite(cost[index]):
      continue
    published = plans[index].astype(np.float32)
    contained = corridor is None or corridor.check(published, controller.recovery_penetration, controller.recovery_elapsed,
                                                  controller.recovery_delay, LanePath(reference, curvature, distances[index]))[0]
    if contained:
      feasible.append((cost[index], index, published))
  if not feasible:
    return None, np.nan
  _, index, best = min(feasible, key=lambda row: (row[0], row[1]))
  return best, float(distances[index])


@pytest.mark.parametrize('name', [case['name'] for case in FIXTURES] + [
  'straight_centered', 'straight_roundoff', 'straight_offset', 'curved_offset', 'noisy_markings', 'narrow_no_path', 'curvature_no_path',
  'curved_centered', 'curved_centered_fast',
])
@pytest.mark.parametrize('mirror', [False, True], ids=['original', 'mirrored'])
def test_optimized_selection_matches_exhaustive_float32_minimum(name, mirror):
  output, speed, curvature = candidate_selection_case(name)
  if mirror:
    output, curvature = mirror_output(output), -curvature
  output['plan'] = output['plan'].astype(np.float32)
  controller = LaneCenteringController('absolute')
  controller.publish_dtype = np.dtype(np.float32)
  controller.filtered_center_y = (output['lane_lines'][0, 1, :, 0] + output['lane_lines'][0, 2, :, 0]) / 2 + CAMERA_OFFSET
  if name == 'straight_centered':
    # Exact zero gives an explicit equal-cost tie among all response scales.
    controller.filtered_center_y[:] = 0.0
  elif name == 'straight_roundoff':
    # Keep the real camera-offset cancellation residue. Previously its tiny
    # nonzero jerk cost forced exact selection to evaluate all 28 responses.
    assert 0 < np.max(abs(controller.filtered_center_y)) < 1e-12
  controller._refresh_corridor(output, speed, curvature, 0.475)
  if name in {case['name'] for case in FIXTURES}:
    assert controller.corridor is not None
    assert controller.corridor.horizon <= DISTANCES[np.searchsorted(DISTANCES, 30.0)]
  if name.startswith('curved_centered'):
    assert controller.corridor.horizon == 192.0
    assert controller.corridor.horizon - HALF_LENGTH - HALF_WIDTH == pytest.approx(188.42)
  base = output['plan'].astype(np.float64)
  expected, expected_distance = exhaustive_candidate_minimum(controller, base, speed, curvature)
  actual, actual_distance = controller._build_lane_plan(base, speed, curvature)
  if name in ('narrow_no_path', 'curvature_no_path'):
    assert expected is None
  else:
    assert expected is not None
  if expected is None:
    assert actual is None
  else:
    assert actual_distance == expected_distance
    assert actual.dtype == np.float32
    assert np.array_equal(actual, expected)
    if name in ('straight_centered', 'straight_roundoff'):
      assert actual_distance == max(1.0, speed * lane_core.MIN_RESPONSE_TIME)


@pytest.mark.parametrize('direction', [-1.0, 1.0])
def test_active_capped_transition_cannot_raise_weight_above_correction_limit(direction):
  speed = 30.0
  output = road_output(speed, np.zeros(3))
  controller = LaneCenteringController('capped')
  previous_base = previous_selected = 0.0
  for _ in range(70):
    _, status, previous_base, previous_selected = selected_step(
      controller, output, speed, 0.0, previous_base, previous_selected)
  assert status.state == 'active'
  assert status.path_weight == 1.0
  assert not status.safety_blocked

  # A 32 cm native lateral pulse returns to the same lane. Its first frame
  # remains policy-admissible and the mostly lane-selected blend is contained.
  # Previously the cap reduced its weight, but the active slew floor raised it
  # back to .95 and silently published a .919 m/s² correction above the .75 cap.
  plan = output['plan'][0]
  amplitude, length = direction * 0.24, 8.0
  u = plan[:, Plan.POSITION.start] / length
  slope = amplitude / length * u**2 * (3 - u) * np.exp(-u)
  second = amplitude / length**2 * u * (6 - 6*u + u**2) * np.exp(-u)
  curvature = second / (1 + slope**2)**1.5
  plan[:, Plan.POSITION.start + 1] = amplitude * u**3 * np.exp(-u)
  plan[:, Plan.T_FROM_CURRENT_EULER.start + 2] = np.arctan(slope)
  plan[:, Plan.ORIENTATION_RATE.start + 2] = speed * curvature
  plan[:, Plan.ACCELERATION.start + 1] = speed**2 * curvature
  _, status, base, selected = selected_step(controller, output, speed, 0.0, previous_base, previous_selected)
  correction = abs(selected - base)
  limit = min(lane_core.MAX_CURVATURE_CORRECTION, lane_core.MAX_LATERAL_ACCEL_CORRECTION / speed**2)
  assert status.safety_blocked or correction <= limit + 1e-10, (status, correction * speed**2)


def native_turn_output():
  speed = NATIVE_TURN['speed']
  output = road_output(speed, np.zeros(3))
  for group, section in [('position', Plan.POSITION), ('velocity', Plan.VELOCITY), ('acceleration', Plan.ACCELERATION),
                          ('orientation', Plan.T_FROM_CURRENT_EULER), ('orientationRate', Plan.ORIENTATION_RATE)]:
    for axis, values in NATIVE_TURN['plan'][group].items():
      output['plan'][0, :, section.start + 'xyz'.index(axis)] = values
  output['plan'] = output['plan'].astype(np.float32)
  output['lane_lines_prob'][:] = 0.0
  output['road_edges_stds'][:] = 1.0
  return output


@pytest.mark.parametrize('mirror', [False, True], ids=['right_turn', 'left_turn'])
def test_native_ninety_degree_turn_preserves_control_without_graph_certificate(mirror):
  output = native_turn_output()
  if mirror:
    output = mirror_output(output)
  original = output['plan'].copy()
  x = original[0, :, Plan.POSITION.start]
  yaw = original[0, :, Plan.T_FROM_CURRENT_EULER.start + 2]
  # Forward body motion through a perpendicular turn can reverse its coordinate
  # along the initial ego x axis. A graph y(x) is not a native-model contract.
  assert np.min(np.diff(x)) < -0.09
  assert np.max(abs(yaw)) > np.pi / 2
  assert np.all(original[0, :, Plan.VELOCITY.start] > 0)
  selected, status, base_action, selected_action = selected_step(
    LaneCenteringController('absolute'), output, NATIVE_TURN['speed'], 0.0)
  assert np.array_equal(selected['plan'], original)
  assert selected_action == base_action
  assert status.path_weight == 0.0
  assert status.containment == 'unavailable'
  assert status.reason == 'native_path_geometry'
  assert not status.safety_blocked
  assert not status.collision_risk


@pytest.mark.parametrize('mirror', [False, True], ids=['left_paint', 'right_paint'])
@pytest.mark.parametrize('next_knot_std', [0.49, 0.51])
def test_single_boundary_confidence_at_thirty_metres_does_not_require_next_knot(mirror, next_knot_std):
  output = road_output(9.6, np.zeros(3))
  output['lane_lines_prob'][:] = 0.0
  output['lane_lines_prob'][0, 2:4] = 0.66
  output['road_edges_stds'][:] = 1.0
  output['lane_lines_stds'][0, 1, :, :] = 0.24
  output['lane_lines_stds'][0, 1, 12, :] = 0.46
  output['lane_lines_stds'][0, 1, 13, :] = next_knot_std
  # This profile isolates the observed confidence-contract mismatch. Raw
  # per-point uncertainties were not logged, so it is not an exact head replay.
  assert DISTANCES[12] == 27.0 and DISTANCES[13] == 31.6875
  assert np.interp(30.0, DISTANCES, output['lane_lines_stds'][0, 1, :, 0]) < 0.5
  assert LaneCenteringController._std_valid(output['lane_lines_stds'][0, 1, :, 0], 0.3, 0.5)
  if mirror:
    output = mirror_output(output)
  controller = LaneCenteringController('absolute')
  selected, status, base_action, selected_action = selected_step(controller, output, 9.6, 0.0)
  assert controller.corridor is None
  assert np.array_equal(selected['plan'], output['plan'])
  assert selected_action == base_action
  assert status.path_weight == 0.0
  assert status.containment == 'unavailable'
  assert not status.safety_blocked
  assert not status.collision_risk


def test_native_turn_fixture_contains_only_relative_numeric_plan_and_speed():
  assert set(NATIVE_TURN) == {'description', 'speed', 'plan'}
  assert set(NATIVE_TURN['plan']) == {'position', 'velocity', 'acceleration', 'orientation', 'orientationRate'}
  for group in NATIVE_TURN['plan'].values():
    assert set(group) == {'x', 'y', 'z'}
    for values in group.values():
      assert len(values) == len(TIMES)
      assert all(isinstance(value, float) and np.isfinite(value) for value in values)

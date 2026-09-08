"""Analytic path and continuous vehicle-envelope regressions.

Expected derivatives and motion come from independent closed-form curves. The
dense footprint oracle below does not call the production corridor helpers.
"""
import math
import itertools
from dataclasses import dataclass

import numpy as np
import pytest

from openpilot.selfdrive.modeld.constants import ModelConstants, Plan
from openpilot.selfdrive.modeld.lane_path import (
  CHECK_STEP, CLEARANCE_BUFFER, HALF_LENGTH, HALF_WIDTH, MODEL_X, TIME, Corridor, LanePath, LaneReference,
  build_lane_candidates, build_sparse_lane_candidates, dynamics_grid, evaluate_lane_dynamics, lane_dynamics_lower_bounds, road_edge_collision,
)


@dataclass
class Quadratic:
  offset: float = 0.0
  slope: float = 0.0
  second: float = 0.0

  def evaluate(self, x):
    x = np.asarray(x)
    return (self.offset + self.slope * x + .5 * self.second * x**2,
            self.slope + self.second * x, np.full_like(x, self.second, dtype=float))


@dataclass
class LocalBump:
  center: float
  half_width: float
  amplitude: float

  def evaluate(self, x):
    # A compact C2 curve. Its derivatives all start at zero, so LanePath's
    # initial-error dynamics do not obscure the independent dynamics oracle.
    u = (np.asarray(x) - self.center) / self.half_width
    inside = abs(u) < 1.
    return (np.where(inside, self.amplitude * (1 - u*u)**3, 0.),
            np.where(inside, self.amplitude * (-6*u + 12*u**3 - 6*u**5) / self.half_width, 0.),
            np.where(inside, self.amplitude * (-6 + 36*u*u - 30*u**4) / self.half_width**2, 0.))


def straight_plan(speed=10.):
  plan = np.zeros((1, ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH))
  plan[0, :, Plan.POSITION.start] = speed * TIME
  plan[0, :, Plan.VELOCITY.start] = speed
  return plan


def flat_corridor(half_width=1.8, horizon=30.):
  return Corridor(np.full_like(MODEL_X, -half_width), np.full_like(MODEL_X, half_width), horizon)


def dense_rectangle_margin(corridor, x, y, yaw):
  """Sample each edge independently, including the tangent behind the camera."""
  along = np.linspace(-1., 1., 101)
  bx = np.r_[HALF_LENGTH * along, HALF_LENGTH * along, np.full_like(along, -HALF_LENGTH), np.full_like(along, HALF_LENGTH)]
  by = np.r_[np.full_like(along, -HALF_WIDTH), np.full_like(along, HALF_WIDTH), HALF_WIDTH * along, HALF_WIDTH * along]
  px = x[:, None] + np.cos(yaw)[:, None] * bx - np.sin(yaw)[:, None] * by
  py = y[:, None] + np.sin(yaw)[:, None] * bx + np.cos(yaw)[:, None] * by
  boundaries = []
  for values in (corridor.left, corridor.right):
    boundary = np.interp(px, MODEL_X, values)
    slope = (values[1] - values[0]) / (MODEL_X[1] - MODEL_X[0])
    boundaries.append(np.where(px < 0., values[0] + slope * px, boundary))
  return np.min(np.minimum(py - boundaries[0], boundaries[1] - py), axis=1) - CLEARANCE_BUFFER


@pytest.mark.parametrize("offset,slope,reference_second", [(.5, .04, -.003), (-.5, -.04, .003), (0., 0., 0.)])
@pytest.mark.parametrize("curvature", [-.08, 0., .08])
@pytest.mark.parametrize("distance", [2., 12., 60.])
def test_exact_initial_position_heading_and_actual_curvature(offset, slope, reference_second, curvature, distance):
  path = LanePath(Quadratic(offset, slope, reference_second), curvature, distance)
  y, first, second = path.evaluate(0.)
  assert abs(y) < 1e-12
  assert abs(first) < 1e-12
  assert abs(second - curvature) < 1e-12
  plan = path.plan(straight_plan())
  assert abs(plan[0, 0, Plan.T_FROM_CURRENT_EULER.start + 2]) < 1e-12
  assert abs(plan[0, 0, Plan.ORIENTATION_RATE.start + 2] - curvature * 10.) < 1e-12


@pytest.mark.parametrize("direction", [-1., 1.])
def test_already_centered_quadratic_is_unchanged_for_entire_horizon(direction):
  reference = Quadratic(second=direction * .006)
  path = LanePath(reference, direction * .006, 12.)
  x = np.linspace(0., 300., 1001)
  np.testing.assert_allclose(path.evaluate(x), reference.evaluate(x), atol=1e-12)


@pytest.mark.parametrize("speed", [1., 10., 22., 35.])
@pytest.mark.parametrize("second", [-.004, .004])
def test_reference_resampling_and_smoothing_preserve_quadratics(speed, second):
  expected = Quadratic(offset=.2, slope=-.01, second=second)
  reference = LaneReference(expected.evaluate(MODEL_X)[0], speed)
  x = np.linspace(0., MODEL_X[-1], 300)
  np.testing.assert_allclose(reference.evaluate(x), expected.evaluate(x), rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize('offset', [-9e-13, -3e-17, 0., 3e-17, 9e-13])
@pytest.mark.parametrize('speed', [22., 40.])
def test_centered_road_roundoff_produces_identical_zero_correction_candidates(offset, speed):
  reference = LaneReference(np.full_like(MODEL_X, offset), speed)
  np.testing.assert_array_equal(reference.evaluate(np.linspace(0., 400., 1001)), np.zeros((3, 1001)))
  distances = speed * np.linspace(.15, 1.5, 28)
  base = straight_plan(speed)
  plans, acceleration, jerk, y = build_lane_candidates(base, reference, 0., distances, np.linspace(0., 192., 101), 3., 5.)
  np.testing.assert_array_equal(plans, np.broadcast_to(base, plans.shape))
  np.testing.assert_array_equal(acceleration, np.zeros(28))
  np.testing.assert_array_equal(jerk, np.zeros(28))
  np.testing.assert_array_equal(y, np.zeros((28, 101)))


@pytest.mark.parametrize('offset', [-1e-6, -1e-8, -1e-12, 1e-12, 1e-8, 1e-6])
def test_reference_numeric_zero_handling_preserves_resolvable_offsets(offset):
  reference = LaneReference(np.full_like(MODEL_X, offset), 22.)
  y, _, _ = reference.evaluate(np.linspace(0., 192., 101))
  np.testing.assert_allclose(y, offset, rtol=1e-7, atol=0.)


@pytest.mark.parametrize("distance", [2., 12., 60.])
def test_straight_offset_recovers_continuously_without_remote_join(distance):
  reference = Quadratic(offset=-.5)
  path = LanePath(reference, 0., distance)
  x = np.linspace(0., 24. * distance, 1001)
  y, first, second = path.evaluate(x)
  assert y[0] == 0.
  assert y[1] < 0.  # correction begins before any putative terminal join
  assert np.all(np.diff(y) < 0.)
  assert np.all(first[1:] < 0.)
  assert abs(y[-1] + .5) < 1e-8
  assert abs(first[-1]) < 1e-8
  assert abs(second[-1]) < 1e-8


def test_error_derivatives_match_finite_differences_and_mirror():
  reference = Quadratic(offset=.4, slope=-.03, second=.003)
  path = LanePath(reference, -.006, 9.)
  mirrored = LanePath(Quadratic(-.4, .03, -.003), .006, 9.)
  x = np.linspace(.1, 100., 401)
  y, first, second = path.evaluate(x)
  h = .001
  before, after = path.evaluate(x - h)[0], path.evaluate(x + h)[0]
  np.testing.assert_allclose(first, (after - before) / (2*h), atol=1e-8)
  np.testing.assert_allclose(second, (after - 2*y + before) / h**2, atol=1e-8)
  np.testing.assert_allclose(mirrored.evaluate(x), -np.asarray(path.evaluate(x)), atol=1e-12)


@pytest.mark.parametrize("direction", [-1., 1.])
@pytest.mark.parametrize("speed", [5., 30.])
def test_reference_tail_preserves_c2_and_smoothly_unwinds_curvature(direction, speed):
  reference = LaneReference(direction * (.002 * MODEL_X**2 + .1 * np.sin(MODEL_X / 10.)), speed)
  end = float(MODEL_X[-1])
  y, slope, second = reference.evaluate(end)
  h = .001
  before, after = reference.evaluate(end - h), reference.evaluate(end + h)
  assert second * direction > .001
  assert abs(before[2] - second) < 1e-7
  assert abs(after[2] - second) < 1e-6  # no instantaneous switch to zero curvature
  assert (after[0] - before[0]) / (2*h) == pytest.approx(slope, abs=1e-7)
  assert (after[0] - 2*y + before[0]) / h**2 == pytest.approx(second, abs=2e-7)
  x = end + reference.tail_distance * np.array([0., 1., 5., 24., 25.])
  _, derivatives, seconds = reference.evaluate(x)
  assert np.all(direction * seconds > 0.)
  assert np.all(np.diff(abs(seconds)) < 0.)
  assert abs(seconds[-1]) < 1e-12
  assert abs(derivatives[-1] - derivatives[-2]) < 1e-10
  assert direction * (derivatives[-1] - slope) > 0.


def test_dense_dynamics_checks_past_the_spatial_boundary_grid():
  base = straight_plan(35.)
  center = .5 * (base[0, -6, Plan.POSITION.start] + base[0, -5, Plan.POSITION.start])
  assert center > MODEL_X[-1]
  path = LanePath(LocalBump(center, 2., .004), 0., 12.)
  candidate = path.plan(base)
  # Sparse published samples fall on either side of this short C2 bend.
  assert np.max(abs(candidate[0, :, Plan.ACCELERATION.start + 1])) == 0.
  acceleration, jerk = path.dynamics(candidate)
  assert acceleration > 5.
  assert jerk > 5.


@pytest.mark.parametrize("extent", [-1., math.nan, math.inf, 1001.])
def test_dense_dynamics_rejects_unbounded_or_invalid_extent(extent):
  base = straight_plan()
  base[0, -1, Plan.POSITION.start] = extent
  acceleration, jerk = LanePath(Quadratic(), 0., 12.).dynamics(base)
  assert math.isinf(acceleration) and math.isinf(jerk)


def test_candidate_dense_jerk_uses_its_own_time_and_distance_mapping(monkeypatch):
  from openpilot.selfdrive.modeld import lane_centering

  raw = straight_plan(1.)
  raw[0, :, Plan.VELOCITY.start] = 10. + 2. * TIME
  candidate_x = 10. * TIME + TIME**2
  center = .5 * (candidate_x[20] + candidate_x[21])
  reference = LocalBump(center, 2., .004)
  # Isolate the reference constructor at an exact analytic C2 curve. Every
  # remaining path construction, sparse/dense check, and selection is production.
  monkeypatch.setattr(lane_centering, "LaneReference", lambda values, speed: reference)
  controller = lane_centering.LaneCenteringController("absolute")
  controller.filtered_center_y = np.zeros_like(MODEL_X)
  candidate = LanePath(reference, 0., 12.).plan(raw)
  assert controller._lateral_plan_feasible(candidate)
  assert candidate[0, -1, Plan.POSITION.start] > 10 * raw[0, -1, Plan.POSITION.start]

  # Independent dense-time oracle sees a jerk peak between published knots.
  t = np.linspace(0., 10., 20001)
  _, slope, second = reference.evaluate(10. * t + t**2)
  acceleration = second / (1 + slope**2)**1.5 * (10. + 2.*t)**2
  assert np.max(abs(np.gradient(acceleration, t))) > 2 * lane_centering.MAX_LANE_PATH_JERK
  selected, _ = controller._build_lane_plan(raw, 10., 0.)
  assert selected is None


def test_rectangle_edge_detects_boundary_notch_between_clear_corners():
  corridor = flat_corridor(4.)
  corridor.right[5] = .8  # x=4.6875 m, between front/rear corners
  corners_x = np.array([2.5, 7.5])
  assert np.min(np.interp(corners_x, MODEL_X, corridor.right) - HALF_WIDTH) > 0.
  margin = corridor.margins(np.array([5.]), np.array([0.]), np.array([0.]))[0]
  assert margin == pytest.approx(.8 - HALF_WIDTH - CLEARANCE_BUFFER)


def test_swept_hull_detects_notch_between_clear_endpoint_poses():
  corridor = flat_corridor(4.)
  corridor.right[6] = .8
  x, y, yaw = np.array([0., 15.]), np.zeros(2), np.zeros(2)
  assert np.all(corridor.margins(x, y, yaw) > 0.)
  assert corridor.swept_margins(x, y, yaw)[0] < 0.


def test_swept_rotation_catches_between_pose_body_bulge():
  corridor = flat_corridor(2.64)
  x, y, yaw = np.array([10., 10.]), np.zeros(2), np.array([.8, 1.5])
  assert np.all(corridor.margins(x, y, yaw) > 0.)
  angle = math.atan2(HALF_LENGTH, HALF_WIDTH)
  actual = dense_rectangle_margin(corridor, np.array([10.]), np.array([0.]), np.array([angle]))[0]
  assert actual < 0.
  certified = corridor.swept_margins(x, y, yaw)[0]
  assert certified <= actual


@pytest.mark.parametrize("direction", [-1., 1.])
def test_curved_motion_is_checked_when_published_positions_stay_straight(direction):
  corridor = flat_corridor()
  plan = straight_plan()
  plan[0, :, Plan.T_FROM_CURRENT_EULER.start + 2] = direction * .05 * TIME
  # At2s, published centers remain at y=0 and the entire rotated body fits.
  assert np.all(corridor.margins(10. * TIME[TIME <= 2.], np.zeros(sum(TIME <= 2.)), direction * .05 * TIME[TIME <= 2.]) > 0.)
  _, _, swept = corridor.motion_margins(plan, 2.)
  assert np.min(swept) < 0.
  assert not corridor.check(plan)[0]


@pytest.mark.parametrize("direction", [-1., 1.])
def test_curved_motion_bound_contains_analytic_accelerating_circle(direction):
  corridor = flat_corridor(100., horizon=192.)
  plan = straight_plan()
  omega, initial_speed, accel = direction * .2, 10., .5
  plan[0, :, Plan.T_FROM_CURRENT_EULER.start + 2] = omega * TIME
  plan[0, :, Plan.VELOCITY.start] = initial_speed + accel * TIME
  times, _, certified = corridor.motion_margins(plan, 4.)
  for index, (start, stop) in enumerate(zip(times[:-1], times[1:], strict=True)):
    t = np.linspace(start, stop, 101)
    yaw = omega * t
    x = initial_speed * np.sin(yaw) / omega + accel * (t * np.sin(yaw) / omega + (np.cos(yaw) - 1) / omega**2)
    y = initial_speed * (1 - np.cos(yaw)) / omega + accel * (-t * np.cos(yaw) / omega + np.sin(yaw) / omega**2)
    exact_sampled_margin = np.min(dense_rectangle_margin(corridor, x, y, yaw))
    assert certified[index] <= exact_sampled_margin + 1e-9


def test_motion_rejects_forward_excursion_beyond_trusted_horizon():
  corridor = flat_corridor(100., horizon=30.)
  plan = straight_plan(speed=1.)
  # The motion head goes to x=50 m and returns to x=0 by10s, while published
  # positions remain inside10 m. Checking only final motion x misses the breach.
  plan[0, :, Plan.VELOCITY.start] = 20. - 4. * TIME
  _, _, swept = corridor.motion_margins(plan, 10.)
  assert np.any(np.isneginf(swept))
  assert not corridor.check(plan)[0]


def test_motion_rejects_between_pose_forward_excursion():
  corridor = flat_corridor(100., horizon=30.)
  plan = straight_plan(speed=1.)
  # At both last two motion endpoints center x is26.3m. Within that interval,
  # velocity transitions +9 to-9m/s, placing the front beyond the30m horizon.
  desired_x = 26.3
  transition_dt = TIME[-2] - TIME[-3]
  base_speed = (desired_x - .5 * 9. * transition_dt) / (TIME[-3] + .5 * transition_dt)
  plan[0, :, Plan.VELOCITY.start] = base_speed
  plan[0, -2:, Plan.VELOCITY.start] = [9., -9.]
  excursion = 9. * (TIME[-1] - TIME[-2]) / 4
  assert desired_x + excursion + HALF_LENGTH > corridor.horizon
  _, _, swept = corridor.motion_margins(plan, 10.)
  assert np.any(np.isneginf(swept))


def test_stationary_plan_cannot_translate_laterally_or_rotate():
  for index in (Plan.POSITION.start + 1, Plan.T_FROM_CURRENT_EULER.start + 2):
    plan = straight_plan(speed=0.)
    plan[0, :, index] = .1 * TIME
    assert not flat_corridor().check(plan)[0]


@pytest.mark.parametrize("initial_speed,accel", [(3., 2.), (35., -3.5), (22., .6), (0., 0.)])
@pytest.mark.parametrize("direction", [-1., 1.])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("seed", [None, 42])
def test_batched_candidates_match_scalar_plans_dynamics_and_cost_order(initial_speed, accel, direction, dtype, seed):
  base = straight_plan(initial_speed).astype(dtype)
  velocity = np.maximum(initial_speed + accel * TIME, 0.)
  center_y = .25 + .001 * MODEL_X**2 + .04 * np.sin(MODEL_X / 7.)
  if seed is not None:
    rng = np.random.default_rng(seed)
    velocity = np.maximum(velocity + rng.normal(0., .1, len(TIME)), 0.)
    center_y += rng.normal(0., .01, len(MODEL_X))
  base[0, :, Plan.VELOCITY.start] = velocity
  base[0, :, Plan.ACCELERATION.start] = accel
  base[0, :, Plan.POSITION.start] = np.r_[0., np.cumsum(.5 * (velocity[:-1] + velocity[1:]) * np.diff(TIME))]
  base[0, :, Plan.T_FROM_CURRENT_EULER.start] = .003 * np.sin(TIME)
  reference = LaneReference(direction * center_y, initial_speed)
  distances = np.maximum(1., initial_speed * np.linspace(.15, 1.5, 28))
  curvature = direction * .004
  cost_x = np.linspace(0., 30., 101)
  accel_limit, jerk_limit = 3., 5.
  batched, max_a, max_j, cost_y = build_lane_candidates(base, reference, curvature, distances, cost_x, accel_limit, jerk_limit)
  sparse_plans, sparse_ok, sparse_y = build_sparse_lane_candidates(base, reference, curvature, distances, cost_x, accel_limit, jerk_limit)
  lower_a, lower_j = lane_dynamics_lower_bounds(batched, reference, curvature, distances)
  assert np.all(lower_a <= max_a) and np.all(lower_j <= max_j)
  np.testing.assert_array_equal(sparse_plans, batched)
  np.testing.assert_array_equal(sparse_y, cost_y)
  scalar_plans, scalar_a, scalar_j, scalar_y = [], [], [], []
  for index, distance in enumerate(distances):
    path = LanePath(reference, curvature, distance)
    plan = path.plan(base)
    scalar_plans.append(plan)
    scalar_y.append(path.evaluate(cost_x)[0])
    acceleration = plan[0, :, Plan.ACCELERATION.start + 1]
    if (np.max(abs(acceleration)) <= accel_limit + 1e-9 and
        np.max(abs(np.diff(acceleration) / np.diff(TIME))) <= jerk_limit + 1e-9):
      assert sparse_ok[index]
      a, j = path.dynamics(plan)
    else:
      assert not sparse_ok[index]
      a, j = np.inf, np.inf
    scalar_a.append(a)
    scalar_j.append(j)
    assert lower_a[index] <= a and lower_j[index] <= j
  np.testing.assert_allclose(batched, scalar_plans, rtol=1e-13, atol=1e-13)
  np.testing.assert_allclose(max_a, scalar_a, rtol=1e-11, atol=1e-11)
  np.testing.assert_allclose(max_j, scalar_j, rtol=1e-10, atol=1e-10)
  np.testing.assert_allclose(cost_y, scalar_y, rtol=1e-13, atol=1e-13)
  weights = .2 + np.exp(-cost_x / max(initial_speed, 1.))
  center = reference.evaluate(cost_x)[0]
  cost = np.mean((cost_y - center)**2 * weights, axis=1) + .00005 * max_j**2
  scalar_cost = [np.mean((y - center)**2 * weights) + .00005 * j**2 for y, j in zip(scalar_y, scalar_j, strict=True)]
  np.testing.assert_array_equal(np.argsort(cost, kind="stable"), np.argsort(scalar_cost, kind="stable"))


def test_batched_candidates_preserve_invalid_row_positions_and_empty_input():
  reference = LaneReference(np.zeros_like(MODEL_X))
  distances = np.array([1., np.nan, 0., -1., np.inf, 10.])
  plans, acceleration, jerk, y = build_lane_candidates(straight_plan(), reference, 0., distances, np.array([0., 10.]), 3., 5.)
  assert plans.shape == (6, 1, 33, 15)
  np.testing.assert_array_equal(np.isfinite(acceleration), [True, False, False, False, False, True])
  np.testing.assert_array_equal(np.isfinite(jerk), np.isfinite(acceleration))
  assert np.all(np.isnan(plans[1:5])) and np.all(np.isnan(y[1:5]))
  for row in (0, 5):
    np.testing.assert_allclose(plans[row], LanePath(reference, 0., distances[row]).plan(straight_plan()), atol=1e-13)
  empty = build_lane_candidates(straight_plan(), reference, 0., np.array([]), np.array([0., 10.]), 3., 5.)
  assert [value.shape for value in empty] == [(0, 1, 33, 15), (0,), (0,), (0, 2)]
  sparse_plans, sparse_ok, sparse_y = build_sparse_lane_candidates(straight_plan(), reference, 0., distances, np.array([0., 10.]), 3., 5.)
  np.testing.assert_array_equal(sparse_plans, plans)
  np.testing.assert_array_equal(sparse_ok, np.isfinite(acceleration))
  np.testing.assert_array_equal(sparse_y, y)
  sparse_empty = build_sparse_lane_candidates(straight_plan(), reference, 0., np.array([]), np.array([0., 10.]), 3., 5.)
  assert [value.shape for value in sparse_empty] == [(0, 1, 33, 15), (0,), (0, 2)]


@pytest.mark.parametrize("invalid", ["base_nan", "curvature_nan", "cost_nan", "accel_nan", "jerk_nan", "too_long"])
def test_batched_candidate_invalid_bounds_cannot_create_finite_feasibility(invalid):
  reference = LaneReference(np.zeros_like(MODEL_X))
  base, curvature, cost_x, accel_limit, jerk_limit = straight_plan(), 0., np.array([0., 10.]), 3., 5.
  if invalid == "base_nan":
    base[0, 5, Plan.VELOCITY.start] = np.nan
  elif invalid == "curvature_nan":
    curvature = np.nan
  elif invalid == "cost_nan":
    cost_x[1] = np.nan
  elif invalid == "accel_nan":
    accel_limit = np.nan
  elif invalid == "jerk_nan":
    jerk_limit = np.nan
  else:
    base = straight_plan(110.)
  _, acceleration, jerk, _ = build_lane_candidates(base, reference, curvature, np.array([1., 10.]), cost_x, accel_limit, jerk_limit)
  assert np.all(np.isposinf(acceleration)) and np.all(np.isposinf(jerk))
  _, sparse_ok, _ = build_sparse_lane_candidates(base, reference, curvature, np.array([1., 10.]), cost_x, accel_limit, jerk_limit)
  assert not np.any(sparse_ok)


@pytest.mark.parametrize("outputs", [values for values in itertools.product([False, True], repeat=3) if any(values)])
@pytest.mark.parametrize("scalar", [False, True])
def test_selective_reference_evaluation_matches_full_tail_formula(outputs, scalar):
  reference = LaneReference(.3 + .002 * MODEL_X**2 + .2 * np.sin(MODEL_X / 5.), 22.)
  x = np.array(210.) if scalar else np.array([[-5., 0., 20., 191.9], [192., 192.1, 220., 300.]])
  idx = np.clip(np.searchsorted(reference.x, x, side="right") - 1, 0, len(reference.a) - 1)
  dx = np.minimum(x, reference.x[-1]) - reference.x[idx]
  y = reference.a[idx] + dx * (reference.b[idx] + dx * (reference.c[idx] + dx * reference.d[idx]))
  first = reference.b[idx] + dx * (2 * reference.c[idx] + 3 * dx * reference.d[idx])
  second = 2 * reference.c[idx] + 6 * dx * reference.d[idx]
  tail = np.maximum(x - reference.x[-1], 0.)
  change = -np.expm1(-tail / reference.tail_distance)
  expected = (y + first * tail + second * reference.tail_distance * (tail - reference.tail_distance * change),
              first + second * reference.tail_distance * change, second * np.exp(-tail / reference.tail_distance))
  for enabled, actual, value in zip(outputs, reference.evaluate_subset(x, outputs), expected, strict=True):
    if enabled:
      np.testing.assert_allclose(actual, value, rtol=1e-14, atol=1e-14)
    else:
      assert actual is None


def full_tensor_polygon_margin(corridor, px, py, hull):
  """Original exhaustive crossing enumeration, independent of sparse gathering."""
  boundary = []
  for values in (corridor.left, corridor.right):
    y = np.interp(px, MODEL_X, values)
    boundary.append(np.where(px < 0., values[0] + px * (values[1] - values[0]) / (MODEL_X[1] - MODEL_X[0]), y))
  margin = np.minimum(np.min(py - boundary[0], axis=1), np.min(boundary[1] - py, axis=1))
  knots = MODEL_X[MODEL_X <= corridor.horizon]
  if hull:
    first, second = np.triu_indices(px.shape[1], 1)
    nx, ny = px[:, second], py[:, second]
    px, py = px[:, first], py[:, first]
  else:
    nx, ny = np.roll(px, -1, axis=1), np.roll(py, -1, axis=1)
  dx = nx - px
  fraction = (knots[None, None, :] - px[:, :, None]) / np.where(abs(dx) > 1e-9, dx, 1.)[:, :, None]
  valid = (abs(dx)[:, :, None] > 1e-9) & (fraction >= 0.) & (fraction <= 1.)
  y = py[:, :, None] + fraction * (ny - py)[:, :, None]
  clearance = np.minimum(y - np.interp(knots, MODEL_X, corridor.left), np.interp(knots, MODEL_X, corridor.right) - y)
  return np.minimum(margin, np.min(np.where(valid, clearance, np.inf), axis=(1, 2))) - CLEARANCE_BUFFER


@pytest.mark.parametrize("hull", [False, True])
@pytest.mark.parametrize("horizon", [5., 30., 192.])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sparse_polygon_crossings_match_full_tensor_with_endpoint_roundoff(hull, horizon, dtype):
  rng = np.random.default_rng(42)
  corridor = Corridor(-1.8 + rng.normal(0., .1, len(MODEL_X)), 1.8 + rng.normal(0., .1, len(MODEL_X)), horizon)
  n = 8 if hull else 4
  x = rng.uniform(-10., 220., (100, 1)) + rng.uniform(-5., 5., (100, n))
  y = rng.uniform(-3., 3., (100, n))
  # Vertical and almost-vertical edges, and exact/adjacent machine numbers at
  # boundary knots exercise the gathering limits and original fraction test.
  x[0] = 5.
  x[1, :2] = [MODEL_X[5], np.nextafter(MODEL_X[5], np.inf)]
  x[2, :2] = [MODEL_X[6], np.nextafter(MODEL_X[6], -np.inf)]
  x[3, :2] = [10., 10. + 1e-9]
  x[4, :2] = [10., 10. + 2e-9]
  x, y = x.astype(dtype), y.astype(dtype)
  expected = full_tensor_polygon_margin(corridor, x, y, hull)
  np.testing.assert_allclose(corridor._polygon_margins(x, y, hull), expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("offset", [0., -.5, .5, -1., 1.])
@pytest.mark.parametrize("curve", [0., -.01, .01])
def test_fast_reject_preserves_acceptance_and_full_success_margin(offset, curve):
  corridor = flat_corridor()
  plan = straight_plan()
  plan[0, :, Plan.POSITION.start + 1] = offset * (1 - np.exp(-TIME))
  plan[0, :, Plan.T_FROM_CURRENT_EULER.start + 2] = curve * TIME
  full, fast = corridor.check(plan), corridor.check(plan, fast_reject=True)
  assert fast[0] == full[0]
  if full[0]:
    assert fast[1] == full[1]
  else:
    assert full[1] <= fast[1] < 0.


def test_fast_reject_still_checks_dense_generating_curve_after_safe_published_poses():
  corridor = flat_corridor()
  plan = straight_plan()
  assert corridor.check(plan, fast_reject=True)[0]
  assert not corridor.check(plan, path=LocalBump(10., 1., 1.), fast_reject=True)[0]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("recovery", [(0., 0., 0.), (.2, .3, .4)])
@pytest.mark.parametrize("speed,accel", [(10., 0.), (25., -3.)])
def test_candidate_corner_filter_only_rejects_full_check_failures(dtype, recovery, speed, accel):
  base = straight_plan(speed)
  base[0, :, Plan.VELOCITY.start] = np.maximum(speed + accel * TIME, 0.)
  distances = np.maximum(1., speed * np.linspace(.15, 1.5, 28))
  for direction in (-1., 1.):
    reference = LaneReference(direction * (.4 + .001 * MODEL_X**2), speed)
    plans, _, _ = build_sparse_lane_candidates(base, reference, direction * .004, distances, np.array([0., 10.]), 3., 5.)
    plans = plans.astype(dtype)
    for half_width in (1.05, 1.8, 3.):
      corridor = flat_corridor(half_width, horizon=100.)
      possible = corridor.candidate_pose_mask(plans, *recovery)
      full = np.array([corridor.check(plan, *recovery)[0] for plan in plans])
      assert not np.any(full & ~possible)
      if half_width == 1.05:
        assert not np.any(possible)
  invalid = np.full((1, 1, 33, 15), np.nan, dtype=dtype)
  assert not corridor.candidate_pose_mask(invalid)[0]
  assert corridor.candidate_pose_mask(plans[:0]).shape == (0,)


def test_corner_filter_does_not_replace_boundary_knot_proof():
  corridor = flat_corridor()
  corridor.right[1] = .8
  plan = straight_plan()
  # The concave notch intersects a body edge between corners. Passing the cheap
  # filter cannot publish this candidate without the stronger final check.
  possible = corridor.candidate_pose_mask(plan[None])
  assert possible[0]
  assert not corridor.check(plan)[0]


@pytest.mark.parametrize('end', [0., 1e-12, .05, .1, .2, .4, .6, np.nextafter(.6, np.inf), 1., 30., 192., 220.03, 1000.])
def test_common_dynamics_grid_keeps_exact_endpoint_and_no_fewer_checks(end):
  x = dynamics_grid(end)
  assert x[0] == 0. and x[-1] == end
  assert len(x) >= max(2, int(end / CHECK_STEP) + 2)
  assert np.max(np.diff(x)) <= CHECK_STEP + 1e-12
  if end > 0.:
    assert np.all(np.diff(x) > 0.)
  expected = np.r_[0., CHECK_STEP / 2, CHECK_STEP * np.arange(1, int(end / CHECK_STEP) + 1)]
  expected = np.r_[expected[expected < end], end] if end > 0. else [0., 0.]
  np.testing.assert_array_equal(x, expected)


@pytest.mark.parametrize('direction', [-1., 1.])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_existing_candidate_dynamics_batch_matches_scalar_after_reordering(direction, dtype):
  reference = LaneReference(direction * (.3 + .0003 * MODEL_X**2 + .01 * np.sin(MODEL_X / 5.)), 30.)
  base = straight_plan(30.).astype(dtype)
  base[0, :, Plan.VELOCITY.start] = 30. - 2. * TIME
  distances = 30. * np.linspace(.15, 1.5, 28)
  plans, _, _ = build_sparse_lane_candidates(base, reference, direction * .003, distances, np.array([0., 30.]), np.inf, np.inf)
  order = [27, 0, 14, 5, 21]
  plans, distances = plans[order].astype(dtype), distances[order]
  expected = np.array([LanePath(reference, direction * .003, distance).dynamics(plan)
                       for distance, plan in zip(distances, plans, strict=True)])
  acceleration, jerk = evaluate_lane_dynamics(plans, reference, direction * .003, distances)
  lower_a, lower_j = lane_dynamics_lower_bounds(plans, reference, direction * .003, distances)
  assert np.all(lower_a <= acceleration) and np.all(lower_j <= jerk)
  assert np.all(lower_a <= expected[:, 0]) and np.all(lower_j <= expected[:, 1])
  np.testing.assert_allclose(acceleration, expected[:, 0], rtol=1e-11, atol=1e-11)
  np.testing.assert_allclose(jerk, expected[:, 1], rtol=1e-10, atol=1e-10)


def test_shared_grid_dynamics_keeps_candidate_endpoint_peaks():
  ends = np.array([10.03, 10.08])
  reference = LocalBump(10.05, .04, .002)
  plans = np.array([straight_plan(end / TIME[-1]) for end in ends])
  distances = np.array([5., 20.])
  acceleration, jerk = evaluate_lane_dynamics(plans, reference, 0., distances)
  lower_a, lower_j = lane_dynamics_lower_bounds(plans, reference, 0., distances)
  np.testing.assert_array_equal(lower_a, acceleration)
  np.testing.assert_array_equal(lower_j, jerk)
  for index, end in enumerate(ends):
    x = dynamics_grid(end)
    _, previous_slope, previous_second = reference.evaluate(x[:-1])
    assert np.all(previous_slope == 0.) and np.all(previous_second == 0.)
    _, slope, second = reference.evaluate(end)
    speed = end / TIME[-1]
    expected_acceleration = abs(second / (1 + slope**2)**1.5 * speed**2)
    expected_jerk = expected_acceleration / (end - x[-2]) * speed
    assert expected_acceleration > 1.
    assert acceleration[index] == pytest.approx(expected_acceleration, rel=1e-12)
    assert jerk[index] == pytest.approx(expected_jerk, rel=1e-12)
    np.testing.assert_allclose((acceleration[index], jerk[index]), LanePath(reference, 0., distances[index]).dynamics(plans[index]), atol=1e-12)


def test_existing_candidate_dynamics_preserves_invalid_positions_and_empty_input():
  reference = LaneReference(np.zeros_like(MODEL_X))
  plans = np.array([straight_plan() for _ in range(6)])
  plans[1, 0, -1, Plan.POSITION.start] = np.nan
  plans[2, 0, -1, Plan.POSITION.start] = 1001.
  plans[3, 0, 8, Plan.POSITION.start] = -1.
  plans[4, 0, 5, Plan.VELOCITY.start] = np.nan
  distances = np.array([5., 5., 5., 5., 5., -1.])
  acceleration, jerk = evaluate_lane_dynamics(plans, reference, 0., distances)
  lower_a, lower_j = lane_dynamics_lower_bounds(plans, reference, 0., distances)
  np.testing.assert_array_equal(lower_a, acceleration)
  np.testing.assert_array_equal(lower_j, jerk)
  np.testing.assert_array_equal(acceleration, [0., np.inf, np.inf, np.inf, np.inf, np.inf])
  np.testing.assert_array_equal(jerk, acceleration)
  empty = evaluate_lane_dynamics(plans[:0], reference, 0., distances[:0])
  assert [value.shape for value in empty] == [(0,), (0,)]
  empty_lower = lane_dynamics_lower_bounds(plans[:0], reference, 0., distances[:0])
  assert [value.shape for value in empty_lower] == [(0,), (0,)]


@pytest.mark.parametrize('direction', [-1., 1.])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_dynamics_bounds_remain_lower_for_mixed_short_and_long_horizons(direction, dtype):
  ends = np.array([220.05, 0., .03, .1, .2, .21, .6, 31.69, 192., 1000.])
  plans = np.array([straight_plan(end / TIME[-1]) for end in ends]).astype(dtype)
  plans[:, 0, :, Plan.VELOCITY.start] *= 1. + .2 * np.sin(TIME)
  reference = LaneReference(direction * (.2 + .002 * MODEL_X**2 + .08 * np.sin(MODEL_X / 10.)), 22.)
  distances = np.linspace(1., 80., len(plans))
  actual_a, actual_j = evaluate_lane_dynamics(plans, reference, direction * .003, distances)
  lower_a, lower_j = lane_dynamics_lower_bounds(plans, reference, direction * .003, distances)
  assert np.all(lower_a <= actual_a) and np.all(lower_j <= actual_j)
  assert lower_a[1] == lower_j[1] == 0.


@pytest.mark.parametrize('speed', [22., 40.])
@pytest.mark.parametrize('curvature', [-.003, .003])
def test_reference_peak_bound_is_tight_on_aligned_curves(speed, curvature):
  reference = LaneReference(.5 * curvature * MODEL_X**2, speed)
  distances = speed * np.linspace(.15, 1.5, 28)
  plans, _, _ = build_sparse_lane_candidates(straight_plan(speed), reference, curvature, distances, np.array([0., 30.]), np.inf, np.inf)
  actual_a, actual_j = evaluate_lane_dynamics(plans, reference, curvature, distances)
  lower_a, lower_j = lane_dynamics_lower_bounds(plans, reference, curvature, distances)
  assert np.all(lower_a <= actual_a)
  np.testing.assert_array_equal(lower_j, actual_j)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_constant_and_variable_speed_rows_keep_scalar_dynamics_exact(dtype):
  speeds = [22., 0., -2., 22.]
  plans = np.array([straight_plan() for _ in speeds]).astype(dtype)
  for index, speed in enumerate(speeds):
    plans[index, 0, :, Plan.VELOCITY.start] = speed
  plans[-1, 0, :, Plan.VELOCITY.start] -= .5 * TIME
  reference = LaneReference(.3 + .001 * MODEL_X**2, 22.)
  distances = np.array([4., 10., 20., 30.])
  expected = np.array([LanePath(reference, .003, distance).dynamics(plan)
                       for distance, plan in zip(distances, plans, strict=True)])
  acceleration, jerk = evaluate_lane_dynamics(plans, reference, .003, distances)
  lower_a, lower_j = lane_dynamics_lower_bounds(plans, reference, .003, distances)
  np.testing.assert_array_equal(acceleration, expected[:, 0])
  np.testing.assert_array_equal(jerk, expected[:, 1])
  assert np.all(lower_a <= expected[:, 0]) and np.all(lower_j <= expected[:, 1])


def warning_edges(half_width=1.8, uncertainty=.03):
  return {'left_edge': np.full_like(MODEL_X, -half_width), 'right_edge': np.full_like(MODEL_X, half_width),
          'left_uncertainty': np.full_like(MODEL_X, uncertainty), 'right_uncertainty': np.full_like(MODEL_X, uncertainty)}


@pytest.mark.parametrize('direction', [-1., 1.])
@pytest.mark.parametrize('motion', [False, True])
def test_road_edge_warning_requires_explicit_predicted_body_contact(direction, motion):
  plan = straight_plan()
  if motion:
    plan[0, :, Plan.T_FROM_CURRENT_EULER.start + 2] = direction * .05 * TIME
    # Published centers stay straight, and even their rotated corners fit.
    assert np.min(flat_corridor().margins(10. * TIME[TIME <= 2.], np.zeros(sum(TIME <= 2.)), direction * .05 * TIME[TIME <= 2.])) > 0.
  else:
    plan[0, :, Plan.POSITION.start + 1] = direction * .5 * TIME
  risk, margin = road_edge_collision(plan, **warning_edges())
  assert risk and np.isfinite(margin) and margin < -.1


def test_narrow_painted_lane_does_not_imply_road_edge_collision():
  plan = straight_plan()
  assert not flat_corridor(.8).check(plan)[0]
  assert road_edge_collision(plan)[0] is False
  assert road_edge_collision(plan, **warning_edges(3.0))[0] is False


@pytest.mark.parametrize('uncertainty,expected', [(.03, True), (.4, False), (.41, False)])
def test_road_edge_warning_notch_requires_penetration_beyond_uncertainty(uncertainty, expected):
  edges = warning_edges(3., uncertainty)
  edges['right_edge'][5] = .7
  risk, margin = road_edge_collision(straight_plan(), **edges)
  assert risk is expected
  if expected:
    assert margin < 0.


def test_wide_noisy_road_and_failed_dynamics_do_not_trigger_collision_warning():
  rng = np.random.default_rng(42)
  edges = warning_edges(3., .3)
  edges['left_edge'] += rng.normal(0., .15, len(MODEL_X))
  edges['right_edge'] += rng.normal(0., .15, len(MODEL_X))
  plan = straight_plan()
  plan[0, :, Plan.ACCELERATION.start + 1] = 100.
  assert not road_edge_collision(plan, **edges)[0]


def test_warning_ignores_unsupported_geometry_and_corners_beyond_observed_horizon():
  plan = straight_plan()
  plan[0, :, Plan.POSITION.start + 1] = np.maximum(TIME - 1.5, 0.) * 3.
  edges = warning_edges()
  assert not road_edge_collision(plan, horizon=10., **edges)[0]
  edges['right_uncertainty'][7:] = np.nan
  assert not road_edge_collision(plan, **edges)[0]


@pytest.mark.parametrize('tail', ['reverse', 'nan'])
def test_warning_keeps_supported_prediction_when_far_tail_is_reversed_or_malformed(tail):
  plan = straight_plan()
  if tail == 'reverse':
    plan[0, 29:, Plan.POSITION.start] = -100.
    plan[0, 29:, Plan.POSITION.start + 1] = 100.
  else:
    plan[0, 29:, Plan.POSITION] = np.nan
  assert not flat_corridor().check(plan)[0]
  assert not road_edge_collision(plan, **warning_edges())[0]
  plan[0, :20, Plan.POSITION.start + 1] = .5 * TIME[:20]
  assert road_edge_collision(plan, **warning_edges())[0]


def test_warning_can_use_one_credible_physical_edge_and_preserves_near_evidence_before_far_nan():
  plan = straight_plan()
  plan[0, :, Plan.POSITION.start + 1] = .5 * TIME
  edges = warning_edges()
  edges['left_edge'] = edges['left_uncertainty'] = None
  edges['right_edge'][-1] = np.nan
  assert road_edge_collision(plan, **edges)[0]


@pytest.mark.parametrize('invalid', [None, np.zeros((3, 3)), 'malformed'])
def test_unavailable_plan_is_not_collision_evidence(invalid):
  risk, margin = road_edge_collision(invalid, **warning_edges())
  assert risk is False and np.isnan(margin)


def test_containment_buffer_failure_is_not_explicit_road_edge_contact():
  plan = straight_plan()
  edge_distance = HALF_WIDTH + CLEARANCE_BUFFER / 2
  assert not flat_corridor(edge_distance).check(plan)[0]
  risk, margin = road_edge_collision(plan, **warning_edges(edge_distance, 0.))
  assert risk is False and margin > 0.


def test_warning_does_not_reinterpret_a_returning_spatial_branch_as_forward_road():
  plan = straight_plan()
  plan[0, :, Plan.POSITION.start] = np.where(TIME <= .8, 10. * TIME, 8. - 2. * (TIME - .8))
  plan[0, :, Plan.POSITION.start + 1] = np.maximum(TIME - 1.2, 0.) * 20.
  plan[0, :, Plan.VELOCITY.start] = 0.
  assert not road_edge_collision(plan, **warning_edges())[0]


def test_warning_rejects_malformed_origin_instead_of_treating_it_as_body_contact():
  plan = straight_plan()
  plan[0, 0, Plan.POSITION.start + 1] = 10.
  risk, margin = road_edge_collision(plan, **warning_edges())
  assert risk is False and np.isnan(margin)

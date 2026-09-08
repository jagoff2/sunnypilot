"""Spatial lane paths and the vehicle envelope used to accept their near horizon.

Coordinates are vehicle centered, x forward and y right. Boundaries are measured
piecewise linear curves. These checks constrain the predicted body, not tire/curb
contact or the uncertainty of the model's estimate of the road.
"""
from dataclasses import dataclass

import numpy as np

from openpilot.selfdrive.modeld.constants import ModelConstants, Plan


HALF_WIDTH = 1.08
# Conservative 5 m passenger-car envelope, including fore/aft body overhang.
HALF_LENGTH = 2.5
CLEARANCE_BUFFER = 0.02
# Recovery starts on an already-overlapping body. Keep 0.1 mm of numerical
# slack inside the 20 mm geometric buffer for conservative interval inflation.
RECOVERY_NUMERICAL_MARGIN = 0.0001
CHECK_DISTANCE = 30.0
CHECK_STEP = 0.20
RECOVERY_TIME = 2.0
TIME = np.asarray(ModelConstants.T_IDXS)
MODEL_X = np.asarray(ModelConstants.X_IDXS)
_BODY_X = np.array([-HALF_LENGTH, HALF_LENGTH, HALF_LENGTH, -HALF_LENGTH])
_BODY_Y = np.array([-HALF_WIDTH, -HALF_WIDTH, HALF_WIDTH, HALF_WIDTH])
_HULL_PAIRS = np.triu_indices(8, 1)
REFERENCE_INDICES = np.unique(np.r_[np.argmin(abs(MODEL_X[:, None] - np.arange(0.0, 30.1, 5.0)), axis=0),
                                    np.flatnonzero(MODEL_X > 30.0)])
REFERENCE_X = MODEL_X[REFERENCE_INDICES]


def _reference_operators():
  # All geometry operators are constant. Factor the smoothing quadratic once,
  # leaving only matrix-vector products and a diagonal scaling per model frame.
  h = np.diff(REFERENCE_X)
  n = len(REFERENCE_X)
  matrix, rhs = np.zeros((n, n)), np.zeros((n, n))
  matrix[0, 0] = matrix[-1, -1] = 1.0
  slope = np.diff(np.eye(n), axis=0) / h[:, None]
  rhs[0] = 2 * (slope[1] - slope[0]) / (h[0] + h[1])
  rhs[-1] = 2 * (slope[-1] - slope[-2]) / (h[-1] + h[-2])
  idx = np.arange(1, n - 1)
  matrix[idx, idx - 1] = h[:-1]
  matrix[idx, idx] = 2 * (h[:-1] + h[1:])
  matrix[idx, idx + 1] = h[1:]
  rhs[1:-1] = 6 * np.diff(slope, axis=0)
  second = np.linalg.solve(matrix, rhs)
  third = np.diff(second, axis=0) / h[:, None]
  sqrt_weight = np.sqrt(np.where(REFERENCE_X <= CHECK_DISTANCE, 1.0, 0.1))
  scaled_third = third / sqrt_weight[None, :]
  eigenvalues, basis = np.linalg.eigh(scaled_third.T @ scaled_third)
  return second, sqrt_weight, np.maximum(eigenvalues, 0.0), basis


_SECOND_OPERATOR, _SQRT_WEIGHT, _SMOOTH_EIGENVALUES, _SMOOTH_BASIS = _reference_operators()


def interpolate(x, xp, yp):
  """Linear interpolation, with tangent extension for the body behind x=0."""
  x = np.asarray(x)
  y = np.interp(x, xp, yp)
  return np.where(x < xp[0], yp[0] + (x - xp[0]) * (yp[1] - yp[0]) / (xp[1] - xp[0]), y)


def valid_plan(plan):
  """Structural and kinematic prerequisites, independent of road confidence."""
  if plan.shape != (1, ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH) or not np.all(np.isfinite(plan)):
    return False
  x = plan[0, :, Plan.POSITION.start]
  y = plan[0, :, Plan.POSITION.start + 1]
  yaw = np.unwrap(plan[0, :, Plan.T_FROM_CURRENT_EULER.start + 2])
  if np.any(np.diff(x) < 0) or abs(x[0]) > 0.05 or abs(y[0]) > 0.05 or abs(yaw[0]) > 0.02:
    return False
  stationary = np.diff(x) <= 1e-8
  speed = np.hypot(plan[0, :, Plan.VELOCITY.start], plan[0, :, Plan.VELOCITY.start + 1])
  inconsistent = (abs(np.diff(y)) > 1e-5) | (abs(np.diff(yaw)) > 1e-5) | (np.maximum(speed[:-1], speed[1:]) > 0.3)
  return not np.any(stationary & inconsistent)


class LaneReference:
  """C2 cubic interpolation of measured lane shape, including S bends.

  Five-metre near-field knots avoid differentiating the model's very short first
  intervals. Endpoint second derivatives come from three points, preserving a
  quadratic road instead of forcing its curvature to zero at the ego position.
  """
  def __init__(self, center_y, speed=20.0):
    self.x = REFERENCE_X
    self.tail_distance = max(speed, 10.0)
    # Use original model knots, roughly five metres apart nearby. Resampling a
    # quadratic with linear interpolation would introduce false curvature/jerk.
    y = np.asarray(center_y)[REFERENCE_INDICES]
    # Camera-offset cancellation can leave a centered straight road a few
    # floating-point ulps from zero. Do not turn sub-picometre position noise
    # into distinct correction/jerk costs for otherwise identical candidates.
    y = np.where(abs(y) < 1e-12, 0., y)
    h = np.diff(self.x)
    # Perception knots are noisy: differentiating exact interpolation amplified
    # centimetre fluctuations into excessive jerk. Smooth the reference with a
    # third-derivative penalty (quadratic roads remain unchanged). At this speed,
    # a 10 cm fit error costs the same as roughly 2 m/s^3 of spatial jerk.
    regularization = (0.1 * max(speed, 1.0)**3 / 2.0)**2
    coordinates = _SMOOTH_BASIS.T @ (_SQRT_WEIGHT * y)
    y = (_SMOOTH_BASIS @ (coordinates / (1 + regularization * _SMOOTH_EIGENVALUES))) / _SQRT_WEIGHT
    slopes = np.diff(y) / h
    second = _SECOND_OPERATOR @ y
    self.a = y[:-1]
    self.b = slopes - h * (2 * second[:-1] + second[1:]) / 6
    self.c = second[:-1] / 2
    self.d = np.diff(second) / (6 * h)

  def evaluate(self, x):
    return self.evaluate_subset(x, (True, True, True))

  def evaluate_subset(self, x, outputs):
    """Evaluate only requested derivatives, without changing their formulas."""
    x = np.asarray(x)
    idx = np.clip(np.searchsorted(self.x, x, side="right") - 1, 0, len(self.a) - 1)
    dx = np.minimum(x, self.x[-1]) - self.x[idx]
    y = np.asarray(self.a[idx] + dx * (self.b[idx] + dx * (self.c[idx] + dx * self.d[idx]))) if outputs[0] else None
    first = np.asarray(self.b[idx] + dx * (2 * self.c[idx] + 3 * dx * self.d[idx])) if outputs[1] else None
    second = np.asarray(2 * self.c[idx] + 6 * dx * self.d[idx]) if outputs[2] else None
    # The full ten-second trajectory can extend beyond the model's spatial
    # boundary grid. Let terminal curvature decay smoothly there. Switching
    # straight to a tangent would introduce a second-derivative discontinuity.
    tail_mask = x > self.x[-1]
    if np.any(tail_mask):
      tail = x[tail_mask] - self.x[-1]
      tail_idx, tail_dx = idx[tail_mask], dx[tail_mask]
      tail_second = second[tail_mask] if second is not None else 2 * self.c[tail_idx] + 6 * tail_dx * self.d[tail_idx]
      if outputs[0] or outputs[1]:
        change = -np.expm1(-tail / self.tail_distance)
        if outputs[0]:
          tail_first = first[tail_mask] if first is not None else self.b[tail_idx] + tail_dx * (2 * self.c[tail_idx] + 3 * tail_dx * self.d[tail_idx])
          y[tail_mask] += tail_first * tail + tail_second * self.tail_distance * (tail - self.tail_distance * change)
        if outputs[1]:
          first[tail_mask] += tail_second * self.tail_distance * change
      if outputs[2]:
        second[tail_mask] *= np.exp(-tail / self.tail_distance)
    return y, first, second


def _evaluate_path(reference, x, coefficients, distance, outputs=(True, True, True), reference_values=None):
  if reference_values is None:
    evaluator = getattr(reference, "evaluate_subset", None)
    reference_values = reference.evaluate(x) if evaluator is None else evaluator(x, outputs)
  y, first, second = reference_values
  u = np.asarray(x) / distance
  decay = np.exp(-u)
  a, b, c = coefficients
  y = y + (a + b*u + c*u*u) * decay if outputs[0] else None
  first = first + (b - a + (2*c - b)*u - c*u*u) * decay / distance if outputs[1] else None
  second = second + (a - 2*b + 2*c + (b - 4*c)*u + c*u*u) * decay / distance**2 if outputs[2] else None
  return y, first, second


def _dynamics_base_grid(end):
  # The extra shared half-step keeps at least the former linspace sample count
  # even when end is an exact multiple of CHECK_STEP. It also resolves the
  # near-ego interval more finely. All other intervals are at most CHECK_STEP.
  x = np.r_[0., CHECK_STEP / 2, np.arange(1, int(end / CHECK_STEP) + 1) * CHECK_STEP]
  return x[x < end] if end > 0. else np.array([0.])


def dynamics_grid(end):
  """Shared 20 cm spatial grid, a 10 cm initial sample, and exact endpoint."""
  return np.r_[_dynamics_base_grid(end), end]


def _candidate_speeds(plans, rows, x):
  """Exact float64 interpolation, bypassing it for constant-speed rows."""
  velocity = plans[rows, 0, :, Plan.VELOCITY.start]
  constant = np.all(velocity == velocity[:, :1], axis=1)
  speed = np.broadcast_to(velocity[:, :1], (len(rows), len(x))).astype(np.float64, copy=True)
  for index in np.flatnonzero(~constant):
    speed[index] = np.interp(x, plans[rows[index], 0, :, Plan.POSITION.start], velocity[index])
  np.maximum(speed, 0., out=speed)
  return speed


class LanePath:
  """Track the whole lane reference with stable spatial error dynamics.

  e''' + 3 e''/L + 3 e'/L^2 + e/L^3 = 0 has three poles at -1/L.
  Its exact solution below matches the actual initial position, heading and
  curvature, and reduces their errors along the entire trajectory. L controls
  response rate, not a terminal point at which centering starts or completes.
  """
  def __init__(self, reference, curvature, response_distance):
    self.reference = reference
    self.response_distance = response_distance
    y, slope, second = reference.evaluate(0.0)
    a = -float(y)
    b = -float(slope) * response_distance + a
    c = 0.5 * ((curvature - float(second)) * response_distance**2 + 2*b - a)
    self.coefficients = np.array([a, b, c])

  def evaluate(self, x):
    return _evaluate_path(self.reference, x, self.coefficients, self.response_distance)

  def evaluate_subset(self, x, outputs):
    return _evaluate_path(self.reference, x, self.coefficients, self.response_distance, outputs)

  def plan(self, base):
    result = np.array(base, dtype=np.float64, copy=True)
    speed = np.maximum(base[0, :, Plan.VELOCITY.start], 0.0)
    dt = np.diff(TIME)
    x = np.r_[0., np.cumsum(0.5 * (speed[:-1] + speed[1:]) * dt)]
    for _ in range(3):
      _, slope, _ = self.evaluate(x)
      vx = speed / np.sqrt(1 + slope**2)
      x[1:] = np.cumsum(0.5 * (vx[:-1] + vx[1:]) * dt)
    y, slope, second = self.evaluate(x)
    curvature = second / (1 + slope**2)**1.5
    result[0, :, Plan.POSITION.start] = x
    result[0, :, Plan.POSITION.start + 1] = y
    result[0, :, Plan.VELOCITY.start + 1] = 0.0
    result[0, :, Plan.ACCELERATION.start + 1] = curvature * speed**2
    result[0, :, Plan.T_FROM_CURRENT_EULER.start + 2] = np.arctan(slope)
    result[0, :, Plan.ORIENTATION_RATE.start + 2] = curvature * speed
    return result

  def dynamics(self, base):
    # Check between model time knots too. A fast response can have a jerk peak
    # which the sparse ten-second published plan would otherwise skip.
    end = float(base[0, -1, Plan.POSITION.start])
    if not np.isfinite(end) or end < 0 or end > 1000.0:
      return np.inf, np.inf
    x = dynamics_grid(end)
    _, slope, second = self.evaluate_subset(x, (False, True, True))
    plan_x = base[0, :, Plan.POSITION.start]
    speed = np.maximum(np.interp(x, plan_x, base[0, :, Plan.VELOCITY.start]), 0.0)
    curvature = second / (1 + slope**2)**1.5
    acceleration = curvature * speed**2
    jerk = np.diff(acceleration) / np.maximum(np.diff(x), 1e-6) * 0.5 * (speed[:-1] + speed[1:]) / np.sqrt(1 + slope[:-1]**2)
    return float(np.max(np.abs(acceleration))), float(np.max(np.abs(jerk)))


def evaluate_lane_dynamics(plans, reference, current_curvature, distances):
  """Dense dynamics for existing candidates, sharing reference-grid work.

  Each candidate uses exactly dynamics_grid(candidate_end), its own speed/x
  mapping, and its exact endpoint. The reference is evaluated once on the
  common grid and once for the endpoint vector; no sparse plan is regenerated.
  Invalid rows retain infinite maxima in their original input positions.
  """
  distances = np.asarray(distances, dtype=np.float64)
  if distances.ndim != 1 or plans.shape != (len(distances), 1, ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH):
    raise ValueError("candidate plans and distances must have matching [K,1,33,15] and [K] shapes")
  max_accel, max_jerk = np.full(len(distances), np.inf), np.full(len(distances), np.inf)
  ends = plans[:, 0, -1, Plan.POSITION.start]
  valid = (np.all(np.isfinite(plans), axis=(1, 2, 3)) & np.isfinite(distances) & (distances > 0.) &
           (ends >= 0.) & (ends <= 1000.) & np.all(np.diff(plans[:, 0, :, Plan.POSITION.start], axis=1) >= 0., axis=1))
  rows = np.flatnonzero(valid)
  if not len(rows) or not np.isfinite(current_curvature):
    return max_accel, max_jerk
  ends = np.asarray(ends[rows], dtype=np.float64)
  distance = distances[rows, None]
  y0, slope0, second0 = reference.evaluate(0.)
  a = np.full_like(distance, -float(y0))
  b = -float(slope0) * distance + a
  c = .5 * ((current_curvature - float(second0)) * distance**2 + 2*b - a)
  coefficients = np.stack((a, b, c))
  common = _dynamics_base_grid(float(np.max(ends)))
  evaluator = getattr(reference, "evaluate_subset", None)
  common_reference = reference.evaluate(common) if evaluator is None else evaluator(common, (False, True, True))
  _, slope, second = _evaluate_path(reference, common, coefficients, distance, (False, True, True), common_reference)
  _, end_slope, end_second = _evaluate_path(reference, ends[:, None], coefficients, distance, (False, True, True))
  speed = _candidate_speeds(plans, rows, common)
  end_speed = np.maximum(np.asarray(plans[rows, 0, -1, Plan.VELOCITY.start], dtype=np.float64), 0.)
  acceleration = second / (1 + slope**2)**1.5 * speed**2
  end_acceleration = end_second[:, 0] / (1 + end_slope[:, 0]**2)**1.5 * end_speed**2
  jerk = (np.diff(acceleration, axis=1) / np.maximum(np.diff(common), 1e-6) *
          .5 * (speed[:, :-1] + speed[:, 1:]) / np.sqrt(1 + slope[:, :-1]**2))
  last = np.maximum(np.searchsorted(common, ends, side="left") - 1, 0)
  index = np.arange(len(rows))
  end_jerk = ((end_acceleration - acceleration[index, last]) / np.maximum(ends - common[last], 1e-6) *
              .5 * (end_speed + speed[index, last]) / np.sqrt(1 + slope[index, last]**2))
  max_accel[rows] = np.maximum(np.max(np.where(common < ends[:, None], abs(acceleration), 0.), axis=1), abs(end_acceleration))
  max_jerk[rows] = np.maximum(np.max(np.where(common[1:] < ends[:, None], abs(jerk), 0.), axis=1, initial=0.), abs(end_jerk))
  return np.where(np.isfinite(max_accel), max_accel, np.inf), np.where(np.isfinite(max_jerk), max_jerk, np.inf)


def lane_dynamics_lower_bounds(plans, reference, current_curvature, distances):
  """Necessary dynamics bounds from a subset of exact full-grid intervals.

  Reference jerk only chooses which intervals to inspect. Bound values use
  each candidate's actual curve and speed mapping at those exact adjacent
  grid positions, including its final interval. Their maxima cannot exceed
  the complete dense-grid maxima. Passing these bounds grants no acceptance.
  """
  distances = np.asarray(distances, dtype=np.float64)
  if distances.ndim != 1 or plans.shape != (len(distances), 1, ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH):
    raise ValueError("candidate plans and distances must have matching [K,1,33,15] and [K] shapes")
  lower_accel, lower_jerk = np.full(len(distances), np.inf), np.full(len(distances), np.inf)
  ends = plans[:, 0, -1, Plan.POSITION.start]
  valid = (np.all(np.isfinite(plans), axis=(1, 2, 3)) & np.isfinite(distances) & (distances > 0.) &
           (ends >= 0.) & (ends <= 1000.) & np.all(np.diff(plans[:, 0, :, Plan.POSITION.start], axis=1) >= 0., axis=1))
  rows = np.flatnonzero(valid)
  if not len(rows) or not np.isfinite(current_curvature):
    return lower_accel, lower_jerk
  ends = np.asarray(ends[rows], dtype=np.float64)
  distance = distances[rows, None]
  y0, slope0, second0 = reference.evaluate(0.)
  a = np.full_like(distance, -float(y0))
  b = -float(slope0) * distance + a
  c = .5 * ((current_curvature - float(second0)) * distance**2 + 2*b - a)
  coefficients = np.stack((a, b, c))
  common = _dynamics_base_grid(float(np.max(ends)))
  evaluator = getattr(reference, "evaluate_subset", None)
  _, ref_slope, ref_second = reference.evaluate(common) if evaluator is None else evaluator(common, (False, True, True))
  interval_count = len(common) - 1
  if interval_count:
    ref_speed = np.maximum(np.interp(common, plans[rows[0], 0, :, Plan.POSITION.start], plans[rows[0], 0, :, Plan.VELOCITY.start]), 0.)
    ref_accel = ref_second / (1 + ref_slope**2)**1.5 * ref_speed**2
    ref_jerk = (np.diff(ref_accel) / np.maximum(np.diff(common), 1e-6) *
                .5 * (ref_speed[:-1] + ref_speed[1:]) / np.sqrt(1 + ref_slope[:-1]**2))
    intervals = np.unique(np.r_[np.argsort(abs(ref_jerk), kind="stable")[-4:],
                                np.linspace(0, interval_count - 1, min(8, interval_count), dtype=int),
                                np.arange(min(3, interval_count))])
  else:
    intervals = np.empty(0, dtype=int)
  last = np.maximum(np.searchsorted(common, ends, side="left") - 1, 0)
  points = np.unique(np.r_[intervals, intervals + 1, last])
  x = common[points]
  values = (None, ref_slope[points], ref_second[points])
  _, slope, second = _evaluate_path(reference, x, coefficients, distance, (False, True, True), values)
  _, end_slope, end_second = _evaluate_path(reference, ends[:, None], coefficients, distance, (False, True, True))
  speed = _candidate_speeds(plans, rows, x)
  end_speed = np.maximum(np.asarray(plans[rows, 0, -1, Plan.VELOCITY.start], dtype=np.float64), 0.)
  acceleration = second / (1 + slope**2)**1.5 * speed**2
  end_acceleration = end_second[:, 0] / (1 + end_slope[:, 0]**2)**1.5 * end_speed**2
  first, following = np.searchsorted(points, intervals), np.searchsorted(points, intervals + 1)
  jerk = ((acceleration[:, following] - acceleration[:, first]) / np.maximum(x[following] - x[first], 1e-6) *
          .5 * (speed[:, first] + speed[:, following]) / np.sqrt(1 + slope[:, first]**2))
  last = np.searchsorted(points, last)
  index = np.arange(len(rows))
  end_jerk = ((end_acceleration - acceleration[index, last]) / np.maximum(ends - x[last], 1e-6) *
              .5 * (end_speed + speed[index, last]) / np.sqrt(1 + slope[index, last]**2))
  lower_accel[rows] = np.maximum(np.max(np.where(x < ends[:, None], abs(acceleration), 0.), axis=1), abs(end_acceleration))
  lower_jerk[rows] = np.maximum(np.max(np.where(x[following] < ends[:, None], abs(jerk), 0.), axis=1, initial=0.), abs(end_jerk))
  return np.where(np.isfinite(lower_accel), lower_accel, np.inf), np.where(np.isfinite(lower_jerk), lower_jerk, np.inf)


def build_lane_candidates(base, reference, current_curvature, distances, cost_x, accel_limit, jerk_limit, sparse_tolerance=1e-9):
  """Evaluate the unchanged candidate family together, leaving selection outside.

  Returns plans[K,1,33,15], dense acceleration/jerk maxima[K], and cost_y[K,M]
  in input-distance order. Invalid or sparse-infeasible rows have infinite
  maxima. Every retained dense row uses the scalar path's shared spatial grid,
  exact endpoint, and candidate-specific speed interpolation.
  No corridor checks or cost ordering are approximated here.
  """
  return _build_lane_candidates(base, reference, current_curvature, distances, cost_x, accel_limit, jerk_limit, sparse_tolerance, True)


def build_sparse_lane_candidates(base, reference, current_curvature, distances, cost_x, accel_limit, jerk_limit, sparse_tolerance=1e-9):
  """Return candidate plans, sparse eligibility, and tracking-cost positions.

  Sparse eligibility is only a preliminary filter. Every candidate considered
  for publication still requires LanePath.dynamics on its own generated plan
  and the complete corridor check. This stage allows exact lower-bound search
  to defer dense dynamics until that candidate could affect the optimum.
  """
  plans, acceleration, jerk, y = _build_lane_candidates(
    base, reference, current_curvature, distances, cost_x, accel_limit, jerk_limit, sparse_tolerance, False,
  )
  return plans, np.isfinite(acceleration) & np.isfinite(jerk), y


def _build_lane_candidates(base, reference, current_curvature, distances, cost_x, accel_limit, jerk_limit, sparse_tolerance, evaluate_dense):
  distances = np.asarray(distances, dtype=np.float64)
  cost_x = np.asarray(cost_x, dtype=np.float64)
  if distances.ndim != 1 or cost_x.ndim != 1:
    raise ValueError("candidate distances and cost positions must be one-dimensional")
  count = len(distances)
  plans = np.full((count, 1, ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH), np.nan)
  max_accel = np.full(count, np.inf)
  max_jerk = np.full(count, np.inf)
  cost_y = np.full((count, len(cost_x)), np.nan)
  if (base.shape != (1, ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH) or not np.all(np.isfinite(base)) or
      not np.isfinite(current_curvature) or not np.all(np.isfinite(cost_x)) or
      not (accel_limit >= 0 and jerk_limit >= 0 and sparse_tolerance >= 0)):
    return plans, max_accel, max_jerk, cost_y
  rows = np.flatnonzero(np.isfinite(distances) & (distances > 0))
  if not len(rows):
    return plans, max_accel, max_jerk, cost_y

  distance = distances[rows, None]
  y0, slope0, second0 = reference.evaluate(0.)
  a = np.full_like(distance, -float(y0))
  b = -float(slope0) * distance + a
  c = .5 * ((current_curvature - float(second0)) * distance**2 + 2*b - a)
  coefficients = np.stack((a, b, c))
  speed = np.maximum(base[0, :, Plan.VELOCITY.start], 0.)
  dt = np.diff(TIME)
  x0 = np.r_[0., np.cumsum(.5 * (speed[:-1] + speed[1:]) * dt)]
  x = np.broadcast_to(x0, (len(rows), len(TIME))).copy()
  for _ in range(3):
    _, slope, _ = _evaluate_path(reference, x, coefficients, distance, (False, True, False))
    vx = speed / np.sqrt(1 + slope**2)
    x[:, 1:] = np.cumsum(.5 * (vx[:, :-1] + vx[:, 1:]) * dt, axis=1)
  y, slope, second = _evaluate_path(reference, x, coefficients, distance)
  curvature = second / (1 + slope**2)**1.5
  generated = np.array(np.broadcast_to(base[0], (len(rows), ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH)), dtype=np.float64, copy=True)
  generated[:, :, Plan.POSITION.start] = x
  generated[:, :, Plan.POSITION.start + 1] = y
  generated[:, :, Plan.VELOCITY.start + 1] = 0.
  generated[:, :, Plan.ACCELERATION.start + 1] = curvature * speed**2
  generated[:, :, Plan.T_FROM_CURRENT_EULER.start + 2] = np.arctan(slope)
  generated[:, :, Plan.ORIENTATION_RATE.start + 2] = curvature * speed
  plans[rows, 0] = generated
  cost_y[rows] = _evaluate_path(reference, cost_x, coefficients, distance, (True, False, False))[0]

  acceleration = generated[:, :, Plan.ACCELERATION.start + 1]
  sparse_jerk = np.diff(acceleration, axis=1) / dt
  ends = x[:, -1]
  keep = (np.all(np.isfinite(generated), axis=(1, 2)) & (ends >= 0.) & (ends <= 1000.) &
          (np.max(abs(acceleration), axis=1) <= accel_limit + sparse_tolerance) &
          (np.max(abs(sparse_jerk), axis=1) <= jerk_limit + sparse_tolerance))
  dense_rows = np.flatnonzero(keep)
  if not len(dense_rows):
    return plans, max_accel, max_jerk, cost_y
  if not evaluate_dense:
    max_accel[rows[dense_rows]] = np.max(abs(acceleration[dense_rows]), axis=1)
    max_jerk[rows[dense_rows]] = np.max(abs(sparse_jerk[dense_rows]), axis=1)
    return plans, max_accel, max_jerk, cost_y

  output_rows = rows[dense_rows]
  max_accel[output_rows], max_jerk[output_rows] = evaluate_lane_dynamics(plans[output_rows], reference, current_curvature, distances[output_rows])
  return plans, max_accel, max_jerk, cost_y


@dataclass
class Corridor:
  left: np.ndarray
  right: np.ndarray

  horizon: float = CHECK_DISTANCE

  @staticmethod
  def _corners(x, y, yaw):
    cosine, sine = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    return x[:, None] + cosine * _BODY_X - sine * _BODY_Y, y[:, None] + sine * _BODY_X + cosine * _BODY_Y

  def _corner_margins(self, px, py):
    left = interpolate(px, MODEL_X, self.left)
    right = interpolate(px, MODEL_X, self.right)
    return np.minimum(np.min(py - left, axis=1), np.min(right - py, axis=1))

  def candidate_pose_mask(self, plans, recovery_penetration=0.0, recovery_elapsed=0.0, delay=0.0):
    """Reject candidates using only corners at already-published pose knots.

    This is a necessary-condition filter, never a containment certificate. The
    final check still includes every body edge, unsampled boundary/end crossing,
    curved motion and the dense generating path. Supply the publication dtype
    so trigonometry uses the same promoted values as final pose interpolation.
    """
    if plans.ndim != 4 or plans.shape[1:] != (1, ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH):
      raise ValueError("candidate plans must have shape [K,1,33,15]")
    count = len(plans)
    if count == 0:
      return np.empty(0, dtype=bool)
    px = plans[:, 0, :, Plan.POSITION.start].astype(np.float64)
    py = plans[:, 0, :, Plan.POSITION.start + 1].astype(np.float64)
    yaw = np.unwrap(plans[:, 0, :, Plan.T_FROM_CURRENT_EULER.start + 2], axis=1).astype(np.float64)
    # Final check uses np.interp at unique x positions, which selects the last
    # duplicate. Exclude earlier duplicates and the unsampled 0/end poses.
    last_duplicate = np.c_[np.diff(px, axis=1) != 0., np.ones(count, dtype=bool)]
    end = np.maximum(0., np.minimum(self.horizon - HALF_LENGTH - HALF_WIDTH, px[:, -1]))
    checked = (px > 0.) & (px < end[:, None]) & last_duplicate
    corners = self._corners(np.r_[px.ravel(), 0.], np.r_[py.ravel(), 0.], np.r_[yaw.ravel(), 0.])
    margin = self._corner_margins(*corners) - CLEARANCE_BUFFER
    progress = np.clip((recovery_elapsed + TIME - min(delay, 0.5)) / max(RECOVERY_TIME - min(delay, 0.5), 0.1), 0, 1)
    allowance = recovery_penetration * (1 - progress)
    failed = np.any(checked & (margin[:-1].reshape(count, -1) + allowance < -1e-6), axis=1)
    return np.all(np.isfinite(plans), axis=(1, 2, 3)) & ~failed & (margin[-1] + allowance[0] >= -1e-6)

  def _polygon_margins(self, px, py, hull=False):
    margin = self._corner_margins(px, py)
    knot_count = np.searchsorted(MODEL_X, self.horizon, side="right")
    knots = MODEL_X[:knot_count]
    if hull:
      # Every convex-hull edge is among these pairs. Checking all pairs avoids
      # an iterative hull algorithm and also catches concave road intrusions.
      first, second = _HULL_PAIRS if px.shape[1] == 8 else np.triu_indices(px.shape[1], 1)
      next_x, next_y = px[:, second], py[:, second]
      px, py = px[:, first], py[:, first]
    else:
      next_x, next_y = np.roll(px, -1, axis=1), np.roll(py, -1, axis=1)
    edges_per_polygon = px.shape[1]
    px, py, next_x, next_y = (value.ravel() for value in (px, py, next_x, next_y))
    dx = next_x - px
    # Enumerate every knot which can intersect each edge, plus both neighboring
    # knots so endpoint roundoff is still decided by the original fraction
    # predicate. This is the same proof as the full edge-by-knot tensor, with
    # its provably irrelevant entries omitted.
    start = np.maximum(np.searchsorted(knots, np.minimum(px, next_x), side="left") - 1, 0)
    stop = np.minimum(np.searchsorted(knots, np.maximum(px, next_x), side="right") + 1, knot_count)
    counts = np.where(abs(dx) > 1e-9, stop - start, 0)
    edges = np.repeat(np.arange(len(px)), counts)
    if len(edges):
      offset = np.arange(len(edges)) - np.repeat(np.cumsum(counts) - counts, counts)
      knot_index = start[edges] + offset
      fraction = (knots[knot_index] - px[edges]) / dx[edges]
      crossing_y = py[edges] + fraction * (next_y[edges] - py[edges])
      crossing_margin = np.minimum(crossing_y - self.left[knot_index], self.right[knot_index] - crossing_y)
      crossing_margin = np.where((fraction >= 0) & (fraction <= 1), crossing_margin, np.inf)
      np.minimum.at(margin, edges // edges_per_polygon, crossing_margin)
    return margin - CLEARANCE_BUFFER

  def margins(self, x, y, yaw):
    """Exact rectangle clearance against the piecewise linear boundaries.

    Along a body edge clearance is piecewise linear, so its minimum is at a
    corner or a boundary knot crossing the edge. Check both, including front
    and rear edges and concave boundaries intruding between corners.
    """
    return self._polygon_margins(*self._corners(x, y, yaw))

  def swept_margins(self, x, y, yaw, center_deviation=0.0):
    """Conservative continuous containment of interpolated published poses.

    The center interpolates linearly between plan samples. A rotating corner
    lies within the convex hull of its endpoint positions plus the rotational
    arc's sagitta. Inflate the hull by that bound, projected into the vertical
    clearance metric using the maximum observed boundary slope. This checks
    the interval between poses, not only a sampling of vehicle positions.
    """
    px, py = self._corners(x, y, yaw)
    hull_x = np.concatenate((px[:-1], px[1:]), axis=1)
    hull_y = np.concatenate((py[:-1], py[1:]), axis=1)
    margin = self._polygon_margins(hull_x, hull_y, hull=True)
    angle = abs(np.diff(yaw))
    # Linear interpolation error is bounded by max|r''(u)| / 8 on u in
    # [0,1]. A corner's second derivative has norm radius * angle^2.
    sagitta = np.hypot(HALF_LENGTH, HALF_WIDTH) * angle**2 / 8
    near = MODEL_X[:-1] < self.horizon
    slope = max(np.max(abs(np.diff(self.left)[near] / np.diff(MODEL_X)[near])),
                np.max(abs(np.diff(self.right)[near] / np.diff(MODEL_X)[near])))
    return np.where(angle < np.pi, margin - (sagitta + center_deviation) * (1 + slope), -np.inf)

  def motion_margins(self, plan, end_time):
    """Certify the curved motion implied by heading and body-frame velocity.

    Positions and headings are separate model heads. Check their implied motion
    as well as the published position interpolation. Linear velocity/heading
    interpolation has an explicit acceleration bound, which bounds center arc
    deviation between checked endpoints. Simpson integration has an explicit
    fourth-derivative error bound added to the hull inflation.
    """
    t = np.unique(np.r_[TIME[TIME < end_time], end_time])
    if len(t) < 2:
      return t, np.array([self.initial_margin()]), np.empty(0)
    dt = np.diff(t)
    yaw = np.interp(t, TIME, np.unwrap(plan[0, :, Plan.T_FROM_CURRENT_EULER.start + 2]))
    vx = np.interp(t, TIME, plan[0, :, Plan.VELOCITY.start])
    vy = np.interp(t, TIME, plan[0, :, Plan.VELOCITY.start + 1])
    vmx, vmy, ym = (vx[:-1] + vx[1:]) / 2, (vy[:-1] + vy[1:]) / 2, (yaw[:-1] + yaw[1:]) / 2
    dx = dt / 6 * (vx[:-1]*np.cos(yaw[:-1]) - vy[:-1]*np.sin(yaw[:-1]) +
                  4*(vmx*np.cos(ym) - vmy*np.sin(ym)) + vx[1:]*np.cos(yaw[1:]) - vy[1:]*np.sin(yaw[1:]))
    dy = dt / 6 * (vx[:-1]*np.sin(yaw[:-1]) + vy[:-1]*np.cos(yaw[:-1]) +
                  4*(vmx*np.sin(ym) + vmy*np.cos(ym)) + vx[1:]*np.sin(yaw[1:]) + vy[1:]*np.cos(yaw[1:]))
    x, y = np.r_[0., np.cumsum(dx)], np.r_[0., np.cumsum(dy)]
    omega = abs(np.diff(yaw) / dt)
    speed = np.maximum(np.hypot(vx[:-1], vy[:-1]), np.hypot(vx[1:], vy[1:]))
    body_accel = np.hypot(np.diff(vx), np.diff(vy)) / dt
    arc_bound = (body_accel + speed * omega) * dt**2 / 8
    integration_bound = np.cumsum(dt**5 / 2880 * (speed * omega**4 + 4 * body_accel * omega**3))
    swept = self.swept_margins(x, y, yaw, arc_bound + integration_bound)
    # Check every interval's possible extent. Looking only at the final point
    # would miss an excursion outside observed road followed by reverse motion.
    upper = np.maximum(x[:-1], x[1:]) + arc_bound + integration_bound
    lower = np.minimum(x[:-1], x[1:]) - arc_bound - integration_bound
    outside = (lower < -0.05) | (upper > self.horizon - np.hypot(HALF_LENGTH, HALF_WIDTH) + 1e-6)
    swept[outside] = -np.inf
    return t, self.margins(x, y, yaw), swept

  def initial_margin(self):
    return float(self.margins(np.array([0.]), np.array([0.]), np.array([0.]))[0])

  def check(self, plan, recovery_penetration=0.0, recovery_elapsed=0.0, delay=0.0, path=None, fast_reject=False):
    """Check all constraints; optional early rejection returns a failure witness.

    Successful checks always compute every margin. With fast_reject, a rejected
    plan reports the smallest margin already checked, rather than evaluating
    later constraints merely to obtain a global diagnostic minimum.
    """
    if not valid_plan(plan):
      return False, -np.inf
    px = plan[0, :, Plan.POSITION.start]
    py = plan[0, :, Plan.POSITION.start + 1]
    yaw = np.unwrap(plan[0, :, Plan.T_FROM_CURRENT_EULER.start + 2])
    end = max(0.0, min(self.horizon - HALF_LENGTH - HALF_WIDTH, float(px[-1])))
    x = np.unique(np.r_[0.0, px[(px > 0) & (px < end)], end])
    t = np.interp(x, px, TIME)
    t[0] = 0.0
    y, heading = np.interp(x, px, py), np.interp(x, px, yaw)
    margins = self.margins(x, y, heading)
    progress = np.clip((recovery_elapsed + t - min(delay, 0.5)) / max(RECOVERY_TIME - min(delay, 0.5), 0.1), 0, 1)
    allowance = recovery_penetration * (1 - progress)
    valid = np.all(margins + allowance >= -1e-6)
    if fast_reject and not valid:
      return False, float(np.min(margins))
    if len(x) > 1:
      swept = self.swept_margins(x, y, heading)
      valid = valid and np.all(swept + np.minimum(allowance[:-1], allowance[1:]) >= -1e-6)
      margins = np.r_[margins, swept]
      if fast_reject and not valid:
        return False, float(np.min(margins))
      motion_t, motion_margin, motion_swept = self.motion_margins(plan, float(t[-1]))
      motion_progress = np.clip((recovery_elapsed + motion_t - min(delay, 0.5)) /
                                max(RECOVERY_TIME - min(delay, 0.5), 0.1), 0, 1)
      motion_allowance = recovery_penetration * (1 - motion_progress)
      valid = valid and np.all(motion_margin + motion_allowance >= -1e-6)
      valid = valid and np.all(motion_swept + np.minimum(motion_allowance[:-1], motion_allowance[1:]) >= -1e-6)
      margins = np.r_[margins, motion_margin, motion_swept]
      if fast_reject and not valid:
        return False, float(np.min(margins))
    if path is not None:
      # Also check the generating curve, since sparse positions and headings
      # are consumed separately downstream. Acceptance always includes the
      # continuously checked interpolation of the final published poses above.
      dense_x = np.linspace(0, end, max(2, int(end / CHECK_STEP) + 2))
      evaluator = getattr(path, "evaluate_subset", None)
      dense_y, slope, _ = path.evaluate(dense_x) if evaluator is None else evaluator(dense_x, (True, True, False))
      dense_margin = self.margins(dense_x, dense_y, np.arctan(slope))
      dense_allowance = np.interp(dense_x, x, allowance)
      valid = valid and np.all(dense_margin + dense_allowance >= -1e-6)
      margins = np.r_[margins, dense_margin]
    # Always include the actual body at t=0, even for raw model paths whose first
    # position is slightly offset from the origin.
    initial = self.initial_margin()
    valid = valid and initial + allowance[0] >= -1e-6
    return bool(valid), float(min(np.min(margins), initial))


def road_edge_collision(plan, left_edge=None, right_edge=None, left_uncertainty=None, right_uncertainty=None,
                        horizon=192.0, prediction_seconds=2.0, half_length=HALF_LENGTH, half_width=HALF_WIDTH):
  """Warn only on explicit predicted footprint overlap with physical road edges.

  A negative returned margin is penetration beyond two positional standard
  deviations plus 5 cm. Unavailable geometry and failed containment proofs are
  not collision evidence. This warning samples the first forward time branch;
  it neither certifies clearance between samples nor controls steering.
  """
  try:
    plan = np.asarray(plan, dtype=float)
  except (TypeError, ValueError):
    return False, np.nan
  if (plan.shape != (1, ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH) or
      not all(np.isfinite(v) and v > 0. for v in (horizon, prediction_seconds, half_length, half_width))):
    return False, np.nan
  sides = []
  for side, (edge, uncertainty) in enumerate(((left_edge, left_uncertainty), (right_edge, right_uncertainty))):
    if edge is None or uncertainty is None:
      continue
    try:
      edge, uncertainty = np.asarray(edge, dtype=float), np.asarray(uncertainty, dtype=float)
    except (TypeError, ValueError):
      continue
    if edge.shape != MODEL_X.shape or uncertainty.shape != MODEL_X.shape:
      continue
    credible = np.isfinite(edge) & np.isfinite(uncertainty) & (uncertainty >= 0.) & (uncertainty <= .4)
    unsupported = np.flatnonzero(~credible)
    last = int(unsupported[0] - 1) if len(unsupported) else len(MODEL_X) - 1
    if last < 1:
      continue
    limit = min(float(horizon), float(MODEL_X[last]))
    knots = np.unique(np.r_[MODEL_X[MODEL_X < limit], limit])
    slope_bound = float(np.max(abs(np.diff(edge[:last + 1]) / np.diff(MODEL_X[:last + 1]))))
    sides.append((side, edge[:last + 1], uncertainty[:last + 1], MODEL_X[:last + 1], limit, knots, slope_bound))
  if not sides:
    return False, np.nan

  px, py = plan[0, :, Plan.POSITION.start], plan[0, :, Plan.POSITION.start + 1]
  heading = plan[0, :, Plan.T_FROM_CURRENT_EULER.start + 2]
  if not all(np.isfinite(v) for v in (px[0], py[0], heading[0])) or abs(px[0]) > .1 or abs(py[0]) > .1 or abs(heading[0]) > .05:
    return False, np.nan
  valid = np.isfinite(px) & np.isfinite(py) & np.isfinite(heading)
  invalid = np.flatnonzero(~valid)
  last = int(invalid[0] - 1) if len(invalid) else len(TIME) - 1
  if last < 1:
    return False, np.nan
  end = min(float(prediction_seconds), float(TIME[last]))
  t = np.unique(np.r_[TIME[TIME < end], np.arange(0., end, .05), end])
  yaw = np.interp(t, TIME[:last + 1], np.unwrap(heading[:last + 1]))
  x, y = np.interp(t, TIME[:last + 1], px[:last + 1]), np.interp(t, TIME[:last + 1], py[:last + 1])
  predictions = [(x, y, yaw, np.zeros_like(t))]

  # Independently inspect the motion implied by heading and body-frame velocity.
  # Native time knots are included, so every integration interval has linear
  # velocity/heading interpolation. Numerical uncertainty reduces the evidence
  # of contact; it never inflates the body into an obstacle.
  vx, vy = plan[0, :, Plan.VELOCITY.start], plan[0, :, Plan.VELOCITY.start + 1]
  velocity_valid = np.isfinite(vx) & np.isfinite(vy)
  invalid = np.flatnonzero(~velocity_valid)
  velocity_last = int(invalid[0] - 1) if len(invalid) else len(TIME) - 1
  if velocity_last >= 1:
    motion_t = t[t <= TIME[min(last, velocity_last)]]
    if len(motion_t) > 1:
      mx = np.interp(motion_t, TIME[:velocity_last + 1], vx[:velocity_last + 1])
      my = np.interp(motion_t, TIME[:velocity_last + 1], vy[:velocity_last + 1])
      mh = yaw[:len(motion_t)]
      dt = np.diff(motion_t)
      vmx, vmy, hm = (mx[:-1] + mx[1:]) / 2, (my[:-1] + my[1:]) / 2, (mh[:-1] + mh[1:]) / 2
      dx = dt / 6 * (mx[:-1]*np.cos(mh[:-1]) - my[:-1]*np.sin(mh[:-1]) +
                    4*(vmx*np.cos(hm) - vmy*np.sin(hm)) + mx[1:]*np.cos(mh[1:]) - my[1:]*np.sin(mh[1:]))
      dy = dt / 6 * (mx[:-1]*np.sin(mh[:-1]) + my[:-1]*np.cos(mh[:-1]) +
                    4*(vmx*np.sin(hm) + vmy*np.cos(hm)) + mx[1:]*np.sin(mh[1:]) + my[1:]*np.cos(mh[1:]))
      omega = abs(np.diff(mh) / dt)
      speed = np.maximum(np.hypot(mx[:-1], my[:-1]), np.hypot(mx[1:], my[1:]))
      body_accel = np.hypot(np.diff(mx), np.diff(my)) / dt
      error = np.r_[0., np.cumsum(dt**5 / 2880 * (speed * omega**4 + 4 * body_accel * omega**3))]
      predictions.append((np.r_[0., np.cumsum(dx)], np.r_[0., np.cumsum(dy)], mh, error))

  minimum = np.inf
  bx = np.array([-half_length, half_length, half_length, -half_length])
  by = np.array([-half_width, -half_width, half_width, half_width])
  for x, y, yaw, error in predictions:
    reversal = np.flatnonzero(np.diff(x) < -1e-6)
    count = int(reversal[0] + 1) if len(reversal) else len(x)
    x, y, yaw, error = x[:count], y[:count], yaw[:count], error[:count]
    cosine, sine = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    cx, cy = x[:, None] + cosine*bx - sine*by, y[:, None] + sine*bx + cosine*by
    nx, ny = np.roll(cx, -1, axis=1), np.roll(cy, -1, axis=1)
    dx = nx - cx
    for side, edge, uncertainty, edge_x, limit, knots, slope_bound in sides:
      numeric_margin = error * (1 + slope_bound)
      credible_corners = (cx - error[:, None] >= 0.) & (cx + error[:, None] <= limit)
      boundary = np.interp(cx, edge_x, edge)
      clearance = cy - boundary if side == 0 else boundary - cy
      margin = clearance + 2*np.interp(cx, edge_x, uncertainty) + .05 + numeric_margin[:, None]
      candidates = np.where(credible_corners & np.isfinite(margin), margin, np.inf)
      minimum = min(minimum, float(np.min(candidates)))
      # A boundary notch may intersect a body side between corners. These are
      # actual rectangle edges at one predicted pose, not a swept convex hull.
      fraction = (knots[None, None, :] - cx[:, :, None]) / np.where(abs(dx) > 1e-9, dx, 1.)[:, :, None]
      crosses = (abs(dx)[:, :, None] > 1e-9) & (fraction >= 0.) & (fraction <= 1.)
      crosses &= (knots[None, None, :] - error[:, None, None] >= 0.) & (knots[None, None, :] + error[:, None, None] <= limit)
      crossing_y = cy[:, :, None] + fraction * (ny - cy)[:, :, None]
      boundary = np.interp(knots, edge_x, edge)
      clearance = crossing_y - boundary if side == 0 else boundary - crossing_y
      margin = clearance + 2*np.interp(knots, edge_x, uncertainty) + .05 + numeric_margin[:, None, None]
      candidates = np.where(crosses & np.isfinite(margin), margin, np.inf)
      minimum = min(minimum, float(np.min(candidates)))
  return bool(minimum < 0.), float(minimum) if np.isfinite(minimum) else np.nan

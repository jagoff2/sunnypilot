import copy
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from openpilot.cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan
from openpilot.selfdrive.modeld.lane_centering import (
  CAMERA_OFFSET,
  ENTRY_TIME,
  MAX_GEOMETRY_HORIZON,
  MAX_LANE_PATH_ACCEL,
  MAX_LANE_PATH_JERK,
  MIN_GEOMETRY_HORIZON,
  MODEL_X,
  RAMP_IN_TIME,
  LaneCenteringController,
)
from openpilot.selfdrive.modeld.lane_centering_integration import LaneCenteringModelAdapter
from openpilot.selfdrive.modeld.tests.test_lane_centering_integration import Inputs, model_output


class TestAdaptiveLaneCentering(unittest.TestCase):
  def candidate(self, output, controller=None):
    controller = controller or LaneCenteringController("absolute")
    candidate, _, gate, _ = controller._line_candidate(output, 20.0)
    self.assertIsNotNone(candidate, gate)
    return candidate

  @staticmethod
  def straight_plan(speed):
    plan = model_output()['plan']
    plan[0, :, Plan.POSITION.start] = speed * np.asarray(ModelConstants.T_IDXS)
    plan[0, :, Plan.VELOCITY.start] = speed
    return plan

  def assert_feasible(self, plan):
    self.assertIsNotNone(plan)
    self.assertTrue(np.all(np.isfinite(plan)))
    acceleration = plan[0, :, Plan.ACCELERATION.start + 1]
    jerk = np.diff(acceleration) / np.diff(ModelConstants.T_IDXS)
    self.assertLessEqual(float(np.max(np.abs(acceleration))), MAX_LANE_PATH_ACCEL + 1e-8)
    self.assertLessEqual(float(np.max(np.abs(jerk))), MAX_LANE_PATH_JERK + 1e-8)

  def test_bad_distant_predictions_preserve_reliable_near_geometry(self):
    full = self.candidate(model_output())
    self.assertGreater(full.geometry.support_distance, MIN_GEOMETRY_HORIZON)
    self.assertLessEqual(full.geometry.support_distance, MAX_GEOMETRY_HORIZON)
    bad_tail = MODEL_X >= 30.0
    for kind in ("uncertainty", "nonfinite", "width"):
      with self.subTest(kind=kind):
        output = model_output()
        if kind == "uncertainty":
          output['lane_lines_stds'][0, 1:3, bad_tail, 0] = 5.0
        elif kind == "nonfinite":
          output['lane_lines'][0, 1:3, bad_tail, 0] = np.nan
        else:
          output['lane_lines'][0, 2, bad_tail, 0] += 3.0
        candidate = self.candidate(output)
        self.assertTrue(candidate.entry_valid)
        self.assertGreaterEqual(candidate.geometry.support_distance, MIN_GEOMETRY_HORIZON)
        self.assertLess(candidate.geometry.support_distance, float(MODEL_X[bad_tail][0]))
        self.assertLess(candidate.geometry.support_distance, full.geometry.support_distance)
        self.assertTrue(np.all(np.isfinite(candidate.geometry.center_y)))

  def test_isolated_confidence_gap_cannot_be_bridged_by_good_distant_points(self):
    output = model_output()
    bad_index = int(np.flatnonzero(MODEL_X >= 25.0)[0])
    output['lane_lines_stds'][0, 1:3, bad_index, 0] = 5.0
    candidate = self.candidate(output)
    self.assertGreaterEqual(candidate.geometry.support_distance, MIN_GEOMETRY_HORIZON)
    self.assertLess(candidate.geometry.support_distance, float(MODEL_X[bad_index]))

  def test_unreliable_near_geometry_still_rejects(self):
    for kind in ("uncertainty", "nonfinite", "width"):
      with self.subTest(kind=kind):
        output = model_output()
        near = MODEL_X < MIN_GEOMETRY_HORIZON
        if kind == "uncertainty":
          output['lane_lines_stds'][0, 1:3, near, 0] = 5.0
        elif kind == "nonfinite":
          output['lane_lines'][0, 1:3, near, 0] = np.nan
        else:
          output['lane_lines'][0, 2, near, 0] = output['lane_lines'][0, 1, near, 0] + 1.0
        controller = LaneCenteringController("absolute")
        self.assertIsNone(controller._line_candidate(output, 20.0)[0])

  def test_active_distant_shape_innovation_truncates_support(self):
    controller = LaneCenteringController("absolute")
    original = model_output()
    controller._accept_candidate(self.candidate(original), 1.0)
    controller.state = "active"
    controller.source = "lane_lines"
    controller.geometry_horizon = MAX_GEOMETRY_HORIZON
    output = copy.deepcopy(original)
    changed = MODEL_X >= 30.0
    output['lane_lines'][0, 1:3, changed, 0] += 0.7
    candidate = self.candidate(output, controller)
    self.assertGreaterEqual(candidate.geometry.support_distance, MIN_GEOMETRY_HORIZON)
    self.assertLess(candidate.geometry.support_distance, float(MODEL_X[changed][0]))

  def test_accepted_horizon_shrinks_immediately_and_grows_gradually(self):
    controller = LaneCenteringController("absolute")
    full = self.candidate(model_output())
    truncated = model_output()
    truncated['lane_lines_stds'][0, 1:3, MODEL_X >= 30.0, 0] = 5.0
    short = self.candidate(truncated)
    controller.geometry_horizon = full.geometry.support_distance
    controller._accept_candidate(short, 1.0)
    self.assertLessEqual(controller.geometry_horizon, short.geometry.support_distance)
    shorter_horizon = controller.geometry_horizon
    controller._accept_candidate(full, 1.0)
    self.assertGreater(controller.geometry_horizon, shorter_horizon)
    self.assertLess(controller.geometry_horizon, full.geometry.support_distance)

  def test_reference_fit_ignores_poisoned_unsupported_tail(self):
    controller = LaneCenteringController("absolute")
    controller.geometry_horizon = 20.0
    controller.filtered_center_y = 0.25 + 0.01 * MODEL_X + 0.0002 * MODEL_X**2
    expected = controller._reference_coefficients()
    self.assertIsNotNone(expected)
    for poison in (1000.0, np.nan):
      with self.subTest(poison=poison):
        controller.filtered_center_y[MODEL_X >= 30.0] = poison
        np.testing.assert_allclose(controller._reference_coefficients(), expected, atol=1e-12, rtol=0.0)

  def test_non_native_endpoint_preserves_observed_upper_interpolation_bracket(self):
    output = model_output()
    output['lane_lines_stds'][0, 1:3, MODEL_X >= 25.0, 0] = 5.0
    upper = int(np.searchsorted(MODEL_X, 20.0))
    # A local bump cannot be represented exactly by the quadratic fit. The
    # endpoint must still interpolate raw observations on both sides of 20 m.
    output['lane_lines'][0, 1:3, upper, 0] += 0.2
    raw_center = 0.5 * (output['lane_lines'][0, 1, :, 0].astype(np.float64) +
                        output['lane_lines'][0, 2, :, 0].astype(np.float64)) + CAMERA_OFFSET
    candidate = self.candidate(output)
    self.assertEqual(candidate.geometry.support_distance, 20.0)
    self.assertEqual(candidate.geometry.center_y[upper], raw_center[upper])
    self.assertAlmostEqual(np.interp(20.0, MODEL_X, candidate.geometry.center_y),
                           np.interp(20.0, MODEL_X, raw_center), places=12)

  def test_support_regrowth_refreshes_endpoint_bracket_without_synthetic_history(self):
    output = model_output()
    output['lane_lines_stds'][0, 1:3, MODEL_X >= 25.0, 0] = 5.0
    upper = int(np.searchsorted(MODEL_X, 20.0))
    output['lane_lines'][0, 1:3, upper, 0] += 0.2
    candidate = self.candidate(output)
    controller = LaneCenteringController("absolute")
    controller._accept_candidate(candidate, 1.0)
    controller._propagate_centerline(16.0, 0.0)
    self.assertAlmostEqual(controller.geometry_horizon, 19.2, places=8)
    self.assertGreater(abs(controller.filtered_center_y[upper] - candidate.geometry.center_y[upper]), 1e-4)
    controller._accept_candidate(candidate, 1.0)
    self.assertEqual(controller.geometry_horizon, 20.0)
    self.assertAlmostEqual(controller.filtered_center_y[upper], candidate.geometry.center_y[upper], places=12)

  def test_supported_policy_disagreement_still_vetoes_lane_control(self):
    output = model_output()
    output['lane_lines_stds'][0, 1:3, MODEL_X >= 25.0, 0] = 5.0
    controller = LaneCenteringController("absolute")
    candidate = self.candidate(output)
    self.assertTrue(controller._evaluate_policy(output, candidate).entry_allowed)
    output['plan'][0, :, Plan.POSITION.start + 1] = 1.5
    policy = controller._evaluate_policy(output, candidate)
    self.assertTrue(policy.hard_veto)
    self.assertFalse(policy.entry_allowed)

  def test_longer_geometry_horizon_does_not_weaken_near_centering(self):
    plans, joins = [], []
    for horizon in (40.0, MAX_GEOMETRY_HORIZON):
      controller = LaneCenteringController("absolute")
      controller.geometry_horizon = horizon
      controller.filtered_center_y = np.full_like(MODEL_X, 0.44)
      plan, join = controller._build_lane_plan(self.straight_plan(16.0), 16.0, 0.0)
      self.assert_feasible(plan)
      self.assertAlmostEqual(join / 16.0, 2.75)
      self.assertGreater(float(np.interp(1.0, ModelConstants.T_IDXS, plan[0, :, Plan.POSITION.start + 1])), 0.1)
      plans.append(plan)
      joins.append(join)
    self.assertAlmostEqual(joins[0], joins[1], places=6)
    near = np.asarray(ModelConstants.T_IDXS) <= 2.0
    np.testing.assert_allclose(plans[0][:, near], plans[1][:, near], atol=1e-8, rtol=0.0)

  def test_active_status_reports_geometry_and_convergence_separately(self):
    controller = LaneCenteringController("absolute")
    output = model_output()
    output['plan'] = self.straight_plan(16.0)
    for _ in range(round((ENTRY_TIME + RAMP_IN_TIME) / DT_MDL) + 5):
      _, status = controller.update(output, v_ego=16.0, current_curvature=0.0, lat_action_t=0.475,
                                    frame_dt=DT_MDL, base_smoothed_curvature=0.0, previous_selected_curvature=0.0,
                                    lat_active=True, model_valid=True, left_blinker=False, right_blinker=False,
                                    lane_change_active=False)
    self.assertEqual(status.state, "active")
    self.assertGreaterEqual(status.geometry_horizon, MIN_GEOMETRY_HORIZON)
    self.assertLessEqual(status.geometry_horizon, MAX_GEOMETRY_HORIZON)
    self.assertGreater(status.convergence_distance, 0.0)
    self.assertAlmostEqual(status.convergence_time, status.convergence_distance / 16.0, places=6)
    self.assertEqual(status.horizon_limited, status.convergence_distance > status.geometry_horizon)

  def test_minimum_supported_horizon_acquires_and_stays_active_while_moving(self):
    controller = LaneCenteringController("absolute")
    output = model_output()
    output['plan'] = self.straight_plan(16.0)
    output['lane_lines_stds'][0, 1:3, MODEL_X >= 20.0, 0] = 5.0
    self.assertEqual(self.candidate(output).geometry.support_distance, MIN_GEOMETRY_HORIZON)
    acquisition_frames = round((ENTRY_TIME + RAMP_IN_TIME) / DT_MDL) + 5
    for frame in range(acquisition_frames + 60):
      _, status = controller.update(output, v_ego=16.0, current_curvature=0.0, lat_action_t=0.475,
                                    frame_dt=DT_MDL, base_smoothed_curvature=0.0, previous_selected_curvature=0.0,
                                    lat_active=True, model_valid=True, left_blinker=False, right_blinker=False,
                                    lane_change_active=False)
      if frame >= acquisition_frames:
        with self.subTest(frame=frame):
          self.assertEqual(status.state, "active")
          self.assertEqual(status.authority, 1.0)
          self.assertEqual(status.geometry_horizon, MIN_GEOMETRY_HORIZON)

  def test_short_support_allows_feasible_partial_progress(self):
    for speed, offset, horizon in ((16.0, 0.8, 20.0), (16.0, -0.8, 20.0), (10.0, 0.44, MIN_GEOMETRY_HORIZON)):
      with self.subTest(speed=speed, offset=offset, horizon=horizon):
        controller = LaneCenteringController("absolute")
        controller.geometry_horizon = horizon
        controller.filtered_center_y = np.full_like(MODEL_X, offset)
        plan, join = controller._build_lane_plan(self.straight_plan(speed), speed, 0.0)
        self.assert_feasible(plan)
        at_support = float(np.interp(horizon, plan[0, :, Plan.POSITION.start], plan[0, :, Plan.POSITION.start + 1]))
        self.assertGreater(at_support * np.sign(offset), 0.0)
        self.assertLessEqual(abs(at_support), abs(offset) + 0.02)
        # A finite, dynamically feasible extrapolation must not continue the
        # partial correction's curvature until it crosses or leaves the lane.
        signed_position = plan[0, :, Plan.POSITION.start + 1] * np.sign(offset)
        self.assertGreaterEqual(float(np.min(signed_position)), -1e-7)
        self.assertLessEqual(float(np.max(signed_position)), abs(offset) + 1e-6)
        if speed == 16.0:
          self.assertGreater(join, horizon)

  def test_short_support_cannot_command_curvature_beyond_observed_geometry(self):
    controller = LaneCenteringController("absolute")
    controller.state, controller.source, controller.authority = "active", "lane_lines", 1.0
    controller.geometry_horizon = MIN_GEOMETRY_HORIZON
    controller.filtered_center_y = np.full_like(MODEL_X, 0.12)
    output = model_output()
    output['plan'] = self.straight_plan(40.0)
    selected = controller._select_output(output, 40.0, 0.0, 0.475, 0.0, 0.0)
    self.assertIs(selected, output)
    self.assertEqual(controller.state, "active")
    self.assertEqual(controller.last_path_weight, 0.0)
    self.assertEqual(controller.reason, "geometry_support_short")

  def test_typical_straight_and_curved_plans_obey_whole_plan_dynamics(self):
    cases = ((10.0, 0.2, 0.0), (16.0, 0.44, 0.0), (16.0, -0.8, 0.0),
             (25.0, 0.3, 0.0), (16.0, 0.2, 0.001), (16.0, -0.2, -0.001))
    for speed, offset, curvature in cases:
      with self.subTest(speed=speed, offset=offset, curvature=curvature):
        controller = LaneCenteringController("absolute")
        controller.geometry_horizon = MAX_GEOMETRY_HORIZON
        controller.filtered_center_y = offset + 0.5 * curvature * MODEL_X**2
        plan, _ = controller._build_lane_plan(self.straight_plan(speed), speed, curvature)
        self.assert_feasible(plan)

  def test_plan_construction_uses_at_most_two_candidates_and_no_runtime_matrix_solve(self):
    controller = LaneCenteringController("absolute")
    controller.geometry_horizon = MAX_GEOMETRY_HORIZON
    controller.filtered_center_y = np.full_like(MODEL_X, 0.8)
    with patch.object(controller, '_lane_plan_for_distance', wraps=controller._lane_plan_for_distance) as build, \
         patch('numpy.linalg.solve', side_effect=AssertionError("runtime matrix solve")), \
         patch('numpy.linalg.pinv', side_effect=AssertionError("runtime pseudoinverse")):
      plan, _ = controller._build_lane_plan(self.straight_plan(16.0), 16.0, 0.0)
    self.assert_feasible(plan)
    self.assertGreaterEqual(build.call_count, 1)
    self.assertLessEqual(build.call_count, 2)

  def test_heading_and_curvature_transition_uses_one_feasible_retry(self):
    controller = LaneCenteringController("absolute")
    controller.geometry_horizon = MAX_GEOMETRY_HORIZON
    controller.filtered_center_y = 0.44 + 0.03 * MODEL_X + 0.001 * MODEL_X**2
    with patch.object(controller, '_lane_plan_for_distance', wraps=controller._lane_plan_for_distance) as build:
      plan, _ = controller._build_lane_plan(self.straight_plan(16.0), 16.0, 0.0)
    self.assertEqual(build.call_count, 2)
    first_attempt = controller._lane_plan_for_distance(*build.call_args_list[0].args)
    self.assertFalse(controller._lateral_plan_feasible(first_attempt))
    self.assert_feasible(plan)

  def test_infeasible_initial_acceleration_rejects_after_bounded_retry(self):
    controller = LaneCenteringController("absolute")
    controller.geometry_horizon = MAX_GEOMETRY_HORIZON
    controller.filtered_center_y = 0.44 + 0.03 * MODEL_X + 0.005 * MODEL_X**2
    speed, initial_curvature = 16.0, 0.1
    self.assertGreater(initial_curvature * speed**2, MAX_LANE_PATH_ACCEL)
    with patch.object(controller, '_lane_plan_for_distance', wraps=controller._lane_plan_for_distance) as build:
      plan, _ = controller._build_lane_plan(self.straight_plan(speed), speed, initial_curvature)
    self.assertIsNone(plan)
    self.assertEqual(controller.last_lane_path_feasibility, 0.0)
    self.assertGreaterEqual(build.call_count, 1)
    self.assertLessEqual(build.call_count, 2)

  def test_rapid_status_changes_log_at_most_once_per_second_with_horizon_fields(self):
    adapter, inputs, output = LaneCenteringModelAdapter(), Inputs(), model_output()
    base_status = replace(adapter.controller._status(), state="active", source="lane_lines", authority=1.0,
                          geometry_horizon=20.0, convergence_distance=32.0, convergence_time=2.0, horizon_limited=True)
    log_timestamps = []

    def action_from_model(*args, **kwargs):
      return log.ModelDataV2.Action()

    with patch.object(adapter.controller, 'update') as update, \
         patch('openpilot.selfdrive.modeld.lane_centering_integration.cloudlog.info') as logger:
      for frame in range(81):
        timestamp = 1_000_000_000 + frame * 50_000_000
        update.return_value = output, replace(base_status, reason="alternating" if frame % 2 else "steady")
        previous_calls = logger.call_count
        adapter.update(output, action_from_model, inputs, True, log.LaneChangeState.off,
                       timestamp, 16.0, 0.475, 0.475)
        if logger.call_count > previous_calls:
          log_timestamps.append(timestamp)
      self.assertEqual(len(log_timestamps), 5)
      self.assertTrue(np.all(np.diff(log_timestamps) >= 1_000_000_000))
      for call in logger.call_args_list:
        rendered = call.args[0] % call.args[1:]
        for field in ("geometry_horizon=20.0", "convergence_distance=32.0", "convergence_time=2.00", "horizon_limited=True"):
          self.assertIn(field, rendered)

  def test_closed_form_merge_preserves_boundary_position_heading_and_curvature(self):
    for distance, reference, initial_curvature in ((20.0, (0.4, 0.01, 0.0005), 0.001),
                                                   (45.0, (-0.8, -0.02, -0.0004), -0.0003)):
      with self.subTest(distance=distance):
        reference = np.asarray(reference)
        coefficients = LaneCenteringController._quintic_coefficients(distance, reference, initial_curvature)
        first = np.polynomial.polynomial.polyder(coefficients)
        second = np.polynomial.polynomial.polyder(first)
        self.assertAlmostEqual(np.polynomial.polynomial.polyval(0.0, coefficients), 0.0, places=10)
        self.assertAlmostEqual(np.polynomial.polynomial.polyval(0.0, first), 0.0, places=10)
        self.assertAlmostEqual(np.polynomial.polynomial.polyval(0.0, second), initial_curvature, places=10)
        for derivative, reference_derivative in ((coefficients, reference),
                                                  (first, np.polynomial.polynomial.polyder(reference)),
                                                  (second, np.polynomial.polynomial.polyder(reference, 2))):
          self.assertAlmostEqual(np.polynomial.polynomial.polyval(distance, derivative),
                                 np.polynomial.polynomial.polyval(distance, reference_derivative), places=9)


if __name__ == '__main__':
  unittest.main()

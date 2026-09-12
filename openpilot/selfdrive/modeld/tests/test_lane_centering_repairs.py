"""Regressions for the September 12 curve-entry and repeated-correction audit."""
import copy
import unittest

import numpy as np

from openpilot.cereal import log
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan
from openpilot.selfdrive.modeld.lane_centering import CAMERA_OFFSET, MODEL_X, LaneCenteringController
from openpilot.selfdrive.modeld.lane_centering_integration import LaneCenteringModelAdapter, fill_lane_centering_status
from openpilot.selfdrive.modeld.tests.test_lane_centering_integration import Inputs, model_output


SPEED = 16.0
ACTION_TIME = 0.4594239175
TIMES = np.asarray(ModelConstants.T_IDXS)


def road_output(offset=0.44, curvature=0.0, onset=0.0):
  output = model_output()
  output = {key: value.astype(np.float64) for key, value in output.items()}
  center = offset + 0.5 * curvature * np.maximum(MODEL_X - onset, 0.0)**2
  for index, shift in enumerate((-5.4, -1.8, 1.8, 5.4)):
    output['lane_lines'][0, index, :, 0] = center + shift - CAMERA_OFFSET
  output['road_edges'] = output['lane_lines'][:, [0, 3]].copy()
  output['road_edges_stds'][:] = 0.9
  x = SPEED * TIMES
  slope = curvature * np.maximum(x - onset, 0.0)
  path_curvature = np.where(x >= onset, curvature, 0.0) / (1.0 + slope**2)**1.5
  output['plan'][:] = 0.0
  output['plan'][0, :, Plan.POSITION.start] = x
  output['plan'][0, :, Plan.POSITION.start + 1] = offset + 0.5 * curvature * np.maximum(x - onset, 0.0)**2
  output['plan'][0, :, Plan.VELOCITY.start] = SPEED
  output['plan'][0, :, Plan.T_FROM_CURRENT_EULER.start + 2] = np.arctan(slope)
  output['plan'][0, :, Plan.ORIENTATION_RATE.start + 2] = path_curvature * SPEED
  output['plan'][0, :, Plan.ACCELERATION.start + 1] = path_curvature * SPEED**2
  return output


def controller_for(output, horizon=60.0):
  controller = LaneCenteringController('absolute')
  controller.geometry_horizon = horizon
  controller.filtered_center_y = 0.5 * (output['lane_lines'][0, 1, :, 0] + output['lane_lines'][0, 2, :, 0]) + CAMERA_OFFSET
  return controller


class TestLaneCenteringRepairs(unittest.TestCase):
  def update(self, adapter, output, inputs=None, timestamp=None, dt=0.05):
    inputs = inputs or Inputs()
    if timestamp is None:
      timestamp = (adapter.last_timestamp_eof or 1_000_000_000) + round(dt * 1e9)
    return adapter.update(output, lambda *args, **kwargs: log.ModelDataV2.Action(), inputs, True,
                          log.LaneChangeState.off, timestamp, SPEED, ACTION_TIME, ACTION_TIME)

  def activate(self, output):
    adapter = LaneCenteringModelAdapter()
    for _ in range(45):
      _, _, status = self.update(adapter, output)
    self.assertEqual(status.state, 'active')
    self.assertEqual(status.path_weight, 1.0)
    return adapter

  def test_exact_straight_prefix_does_not_anticipate_a_curve_ten_metres_ahead(self):
    for sign in (-1, 1):
      with self.subTest(sign=sign):
        output = road_output(offset=0.0, curvature=sign * 0.003, onset=10.0)
        controller = controller_for(output)
        plan, _ = controller._build_lane_plan(output['plan'], SPEED, 0.0)
        self.assertIsNotNone(plan)
        # The audit's old global fit requested +/-0.322 m/s² before the bend.
        self.assertAlmostEqual(controller._plan_curvature(plan, SPEED, ACTION_TIME), 0.0, places=10)
        np.testing.assert_allclose(plan[0, TIMES <= 0.475, Plan.POSITION.start + 1], 0.0, atol=1e-10)
        self.assertGreater(sign * np.interp(1.5, TIMES, plan[0, :, Plan.POSITION.start + 1]), 0.0)

  def test_distant_shape_and_horizon_changes_cannot_change_near_road_shape(self):
    output = road_output(offset=0.3, curvature=0.001)
    near_paths = []
    for horizon in (30.0, 40.0, 49.9, 50.0, 60.0):
      controller = controller_for(output, horizon)
      controller.filtered_center_y[MODEL_X > 30.0] += 0.00002 * (MODEL_X[MODEL_X > 30.0] - 30.0)**3
      road = controller._reference_curve()
      near_paths.append(controller._evaluate_reference(np.linspace(0.0, 15.0, 101), road))
    for path in near_paths[1:]:
      np.testing.assert_allclose(path, near_paths[0], atol=1e-10, rtol=0.0)

  def test_local_reference_has_continuous_position_heading_and_curvature(self):
    controller = controller_for(road_output(offset=0.3, curvature=0.003, onset=10.0))
    road = controller._reference_curve()
    knots = road[0][1:-1]
    left = controller._evaluate_reference(knots - 1e-7, road)
    right = controller._evaluate_reference(knots + 1e-7, road)
    np.testing.assert_allclose(left, right, atol=1e-7, rtol=0.0)

  def test_support_endpoint_cannot_create_a_microscopic_reference_span(self):
    output = road_output(offset=0.0, curvature=0.001)
    for boundary in (15.0, 17.5, 20.0, 22.5, 30.0, 32.5, 60.0):
      for epsilon in (-1e-9, 0.0, 1e-9, 1e-6):
        horizon = max(15.0, boundary + epsilon)
        controller = controller_for(output, horizon)
        road = controller._reference_curve()
        self.assertGreaterEqual(float(np.min(np.diff(road[0]))), 2.5)
        samples = np.linspace(0.0, horizon, 300)
        values = controller._evaluate_reference(samples, road)
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertLess(float(np.max(np.abs(values[2]))), 0.005)

  def test_forward_jerk_projection_keeps_prefix_and_integrates_changed_path(self):
    output = road_output(offset=0.0)
    controller = controller_for(output)
    plan = output['plan'].copy()
    first_changed = 12
    plan[0, first_changed:, Plan.ACCELERATION.start + 1] = 2.0
    projected = controller._project_lane_dynamics(plan, controller._reference_curve())
    self.assertIsNotNone(projected)
    np.testing.assert_array_equal(projected[:, :first_changed], plan[:, :first_changed])
    self.assertTrue(controller._lateral_plan_feasible(projected, 0.999))
    acceleration = projected[0, :, Plan.ACCELERATION.start + 1]
    yaw_rate = projected[0, :, Plan.ORIENTATION_RATE.start + 2]
    np.testing.assert_allclose(yaw_rate * SPEED, acceleration, atol=1e-12)
    integrated_yaw = np.r_[0.0, np.cumsum(0.5 * (yaw_rate[:-1] + yaw_rate[1:]) * np.diff(TIMES))]
    np.testing.assert_allclose(projected[0, :, Plan.T_FROM_CURRENT_EULER.start + 2], integrated_yaw, atol=1e-12)
    self.assertGreater(projected[0, -1, Plan.POSITION.start + 1], 0.0)
    self.assertLess(projected[0, -1, Plan.POSITION.start], plan[0, -1, Plan.POSITION.start])

  def test_forward_projection_rejects_initial_violation_and_observed_corridor_crossing(self):
    output = road_output(offset=0.0)
    controller = controller_for(output)
    road = controller._reference_curve()
    plan = output['plan'].copy()
    plan[0, 0, Plan.ACCELERATION.start + 1] = 5.01
    self.assertIsNone(controller._project_lane_dynamics(plan, road))
    plan = output['plan'].copy()
    plan[0, 12:, Plan.ACCELERATION.start + 1] = 2.0
    controller.corridor_center_y = np.zeros_like(MODEL_X)
    controller.corridor_half_width = np.full_like(MODEL_X, 1.8)
    self.assertIsNone(controller._project_lane_dynamics(plan, road))
    plan[0, 12, Plan.ACCELERATION.start + 1] = np.nan
    self.assertIsNone(controller._project_lane_dynamics(plan, road))

  def test_ego_motion_resampling_preserves_a_curved_reference(self):
    for sign in (-1, 1):
      with self.subTest(sign=sign):
        output = road_output(offset=sign * 0.2, curvature=sign * 0.001)
        controller = LaneCenteringController('absolute')
        for frame in range(120):
          selected, status = controller.update(output, SPEED, sign * 0.001, ACTION_TIME, .05,
                                               0.0, 0.0, True, True, False, False, False)
          if frame >= 40:
            self.assertEqual(status.state, 'active')
            self.assertEqual(status.path_weight, 1.0)
            self.assertTrue(controller._lateral_plan_feasible(selected['plan']))

  def test_parallel_recentering_matches_restored_response(self):
    for horizon in (20.0, 40.0, 60.0):
      controller = controller_for(road_output(), horizon)
      plan, join = controller._build_lane_plan(road_output()['plan'], SPEED, 0.0)
      self.assertAlmostEqual(join / SPEED, 2.75)
      requested_acceleration = controller._plan_curvature(plan, SPEED, ACTION_TIME) * SPEED**2
      self.assertAlmostEqual(requested_acceleration, 0.4058000748, places=6)

  def test_single_support_dropout_withdraws_geometry_without_cold_reacquisition(self):
    output = road_output()
    adapter = self.activate(output)
    dropout = copy.deepcopy(output)
    dropout['lane_lines_stds'][:, :, MODEL_X >= 15.0] = 5.0
    previous = adapter.previous_selected_action.desiredCurvature
    selected, action, status = self.update(adapter, dropout)
    self.assertIs(selected, dropout)
    self.assertEqual(status.path_weight, 0.0)
    self.assertEqual(status.state, 'active')
    self.assertLessEqual(abs(action.desiredCurvature - previous) * SPEED**2 / 0.05, 5.0 + 1e-5)
    selected, _, status = self.update(adapter, output)
    self.assertEqual(status.state, 'active')
    self.assertGreater(status.path_weight, 0.0)
    self.assertLessEqual(status.path_weight, 0.05 + 1e-8)
    self.assertIsNot(selected, output)

  def test_path_replacement_action_is_bounded_even_when_both_paths_are_feasible(self):
    output = road_output(offset=0.0)
    adapter = LaneCenteringModelAdapter('off')
    for frame in range(40):
      curvature = 0.004 if frame < 20 else -0.004
      output['plan'][0, :, Plan.T_FROM_CURRENT_EULER.start + 2] = curvature * SPEED * TIMES
      output['plan'][0, :, Plan.ORIENTATION_RATE.start + 2] = curvature * SPEED
      previous = adapter.previous_selected_action.desiredCurvature
      _, action, status = self.update(adapter, output)
      self.assertLessEqual(abs(action.desiredCurvature - previous) * SPEED**2 / 0.05, 5.0 + 1e-5)
      self.assertAlmostEqual(status.selected_curvature, action.desiredCurvature, places=8)

  def test_actual_frame_interval_keeps_filter_time_constant(self):
    output = road_output(offset=0.0, curvature=0.0001)
    adapters = [LaneCenteringModelAdapter('off'), LaneCenteringModelAdapter('off')]
    for adapter in adapters:
      self.update(adapter, output)
    self.update(adapters[0], output, dt=0.10)
    self.update(adapters[1], output, dt=0.05)
    self.update(adapters[1], output, dt=0.05)
    self.assertAlmostEqual(adapters[0].previous_selected_action.desiredCurvature,
                           adapters[1].previous_selected_action.desiredCurvature, places=10)

  def test_policy_disagreement_blends_authority_before_veto(self):
    output = road_output(offset=0.3)
    adapter = self.activate(output)
    output['plan'][0, :, Plan.POSITION.start + 1] += 0.9
    for _ in range(25):
      _, _, status = self.update(adapter, output)
    self.assertEqual(status.policy_gate, 'blended')
    self.assertAlmostEqual(status.path_weight, (1.0 - 0.9) / 0.75, places=6)
    output['plan'][0, :, Plan.POSITION.start + 1] += 0.11
    _, _, status = self.update(adapter, output)
    self.assertEqual(status.state, 'exiting')
    output['plan'][0, :, Plan.POSITION.start + 1] -= 0.02
    for _ in range(10):
      _, _, status = self.update(adapter, output)
      self.assertNotEqual(status.state, 'active')

  def test_nonincreasing_timestamp_does_not_advance_observation_clock(self):
    output = road_output()
    adapter = self.activate(output)
    timestamp = adapter.last_timestamp_eof
    _, _, status = self.update(adapter, output, timestamp=timestamp - 10_000_000)
    self.assertEqual(adapter.last_timestamp_eof, timestamp)
    self.assertEqual(status.reason, 'model_gap')
    self.assertEqual(status.path_weight, 0.0)

  def test_corridor_validation_rejects_an_outward_path(self):
    output = road_output(offset=0.0)
    controller = controller_for(output)
    controller.corridor_half_width = np.full_like(MODEL_X, 1.8)
    plan = output['plan'].copy()
    plan[0, 4, Plan.POSITION.start + 1] = 0.9
    self.assertFalse(controller._corridor_feasible(plan, controller._reference_curve()))

  def test_fresh_painted_pair_recovers_small_existing_overlap_after_long_dropout(self):
    adapter = self.activate(road_output())
    missing = road_output()
    missing['lane_lines_prob'][:] = 0.0
    for _ in range(160):
      self.update(adapter, missing)
    self.assertFalse(adapter.controller._line_pair_available())
    recovered = road_output(offset=0.824)
    _, _, status = self.update(adapter, recovered)
    self.assertEqual(status.state, 'acquiring')
    self.assertEqual(status.entry_gate, 'fresh_recovery')
    self.assertEqual(status.path_weight, 0.0)
    for _ in range(40):
      selected, _, status = self.update(adapter, recovered)
    self.assertEqual(status.state, 'active')
    self.assertEqual(status.path_weight, 1.0)
    y = selected['plan'][0, :, Plan.POSITION.start + 1]
    self.assertGreater(np.interp(1.0, TIMES, y), 0.0)
    self.assertLessEqual(float(np.max(y)), 0.824 + 1e-6)

  def test_fresh_recovery_does_not_accept_low_confidence_or_large_overlap(self):
    for offset, probability in ((0.824, 0.59), (1.15, 0.95)):
      with self.subTest(offset=offset, probability=probability):
        output = road_output(offset=offset)
        output['lane_lines_prob'][:] = probability
        adapter = LaneCenteringModelAdapter()
        for _ in range(60):
          selected, _, status = self.update(adapter, output)
        self.assertEqual(status.path_weight, 0.0)
        self.assertIs(selected, output)

  def test_every_frame_diagnostics_serialize_original_policy_and_selection(self):
    output = road_output()
    adapter = self.activate(output)
    selected, action, status = self.update(adapter, output)
    message = log.ModelDataV2.new_message()
    self.assertFalse(message.laneCentering.present)
    fill_lane_centering_status(message, output, status, adapter.controller.mode)
    with log.ModelDataV2.from_bytes(message.to_bytes()) as parsed:
      diagnostic = parsed.laneCentering
      self.assertTrue(diagnostic.present)
      self.assertTrue(diagnostic.basePlanValid)
      self.assertEqual(diagnostic.modelTimestampEof, adapter.last_timestamp_eof)
      self.assertEqual(diagnostic.state, status.state)
      self.assertAlmostEqual(diagnostic.selectedCurvature, action.desiredCurvature, places=8)
      np.testing.assert_allclose(diagnostic.basePosition.y, output['plan'][0, :, Plan.POSITION.start + 1], atol=1e-6)
      self.assertEqual(len(diagnostic.laneBoundaryStds), 4 * ModelConstants.IDX_N)
      self.assertFalse(np.array_equal(selected['plan'], output['plan']))
    invalid = log.ModelDataV2.new_message()
    fill_lane_centering_status(invalid, {}, LaneCenteringController('off')._status(), 'off')
    with log.ModelDataV2.from_bytes(invalid.to_bytes()) as parsed:
      self.assertFalse(parsed.laneCentering.basePlanValid)
      self.assertEqual(len(parsed.laneCentering.basePosition.x), 0)


if __name__ == '__main__':
  unittest.main()

import tempfile
import unittest
import subprocess
import sys

from pathlib import Path

import numpy as np

from openpilot.sunnypilot.selfdrive.controls.lib.kona_residual_policy import (
  OBSERVATION_FEATURE_NAMES,
  ResidualPolicyState,
  ResidualTorquePolicy,
  build_observation,
)


class TestKonaResidualPolicy(unittest.TestCase):
  def test_observation_uses_current_and_previous_stock_openpilot_features(self):
    state = ResidualPolicyState()

    first = build_observation(state, 3.0, 1.0, 0.01, 0.2, 0.3, vehicle_bearing_deg=10.0)
    second = build_observation(state, 4.0, 2.0, 0.02, 0.4, 0.5, vehicle_bearing_deg=12.0)

    self.assertEqual(len(first), len(OBSERVATION_FEATURE_NAMES))
    np.testing.assert_allclose(first[:10], [3.0, 1.0, 0.01, 0.2, 0.3, 3.0, 1.0, 0.01, 0.2, 0.3])
    np.testing.assert_allclose(second[:10], [4.0, 2.0, 0.02, 0.4, 0.5, 3.0, 1.0, 0.01, 0.2, 0.3])
    self.assertAlmostEqual(first[OBSERVATION_FEATURE_NAMES.index("abs_steer_angle_ratio")], 1.0 / 89.6)
    self.assertAlmostEqual(first[OBSERVATION_FEATURE_NAMES.index("desired_lateral_accel_mps2")], 0.09)
    self.assertAlmostEqual(first[OBSERVATION_FEATURE_NAMES.index("requested_applied_torque_error")], -0.1)
    self.assertEqual(first[OBSERVATION_FEATURE_NAMES.index("stock_torque_sign_mismatch")], 0.0)
    self.assertEqual(first[OBSERVATION_FEATURE_NAMES.index("speed_delta_mps")], 0.0)
    self.assertEqual(first[OBSERVATION_FEATURE_NAMES.index("steer_angle_delta_deg")], 0.0)
    self.assertEqual(first[OBSERVATION_FEATURE_NAMES.index("desired_curvature_delta")], 0.0)
    self.assertEqual(first[OBSERVATION_FEATURE_NAMES.index("stock_requested_torque_delta")], 0.0)
    self.assertEqual(first[OBSERVATION_FEATURE_NAMES.index("stock_applied_torque_delta")], 0.0)
    self.assertEqual(first[OBSERVATION_FEATURE_NAMES.index("requested_applied_torque_error_delta")], 0.0)
    self.assertEqual(first[OBSERVATION_FEATURE_NAMES.index("abs_steer_angle_delta_deg")], 0.0)
    self.assertEqual(first[OBSERVATION_FEATURE_NAMES.index("abs_stock_applied_torque_delta")], 0.0)
    self.assertEqual(first[OBSERVATION_FEATURE_NAMES.index("steer_saturation_pressure")], 0.0)
    self.assertEqual(first[OBSERVATION_FEATURE_NAMES.index("near_steer_saturation")], 0.0)
    self.assertGreater(first[OBSERVATION_FEATURE_NAMES.index("model_desired_heading_integral_deg")], 0.0)
    self.assertAlmostEqual(first[OBSERVATION_FEATURE_NAMES.index("model_curvature_sign_run_s")], 0.01)
    self.assertEqual(second[OBSERVATION_FEATURE_NAMES.index("speed_delta_mps")], 1.0)
    self.assertEqual(second[OBSERVATION_FEATURE_NAMES.index("steer_angle_delta_deg")], 1.0)
    self.assertAlmostEqual(second[OBSERVATION_FEATURE_NAMES.index("desired_curvature_delta")], 0.01)
    self.assertAlmostEqual(second[OBSERVATION_FEATURE_NAMES.index("stock_requested_torque_delta")], 0.2)
    self.assertAlmostEqual(second[OBSERVATION_FEATURE_NAMES.index("stock_applied_torque_delta")], 0.2)
    self.assertAlmostEqual(second[OBSERVATION_FEATURE_NAMES.index("requested_applied_torque_error_delta")], 0.0)
    self.assertEqual(second[OBSERVATION_FEATURE_NAMES.index("abs_steer_angle_delta_deg")], 1.0)
    self.assertAlmostEqual(second[OBSERVATION_FEATURE_NAMES.index("abs_stock_applied_torque_delta")], 0.2)
    self.assertGreater(
      second[OBSERVATION_FEATURE_NAMES.index("model_desired_heading_integral_deg")],
      first[OBSERVATION_FEATURE_NAMES.index("model_desired_heading_integral_deg")],
    )
    self.assertEqual(first[OBSERVATION_FEATURE_NAMES.index("vehicle_heading_change_deg")], 0.0)
    self.assertEqual(second[OBSERVATION_FEATURE_NAMES.index("vehicle_heading_change_deg")], 2.0)

  def test_loaded_linear_policy_outputs_bounded_residual_delta(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      policy_path = Path(temp_dir) / "policy.npz"
      weight = np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32)
      weight[OBSERVATION_FEATURE_NAMES.index("stock_requested_torque")] = 10.0
      np.savez(
        policy_path,
        feature_mu=np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
        feature_sigma=np.ones(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
        weight=weight,
        bias=np.asarray([0.0], dtype=np.float32),
        max_delta=np.asarray([0.05], dtype=np.float32),
      )

      policy = ResidualTorquePolicy.from_path(str(policy_path), max_delta=0.1)
      observation = np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32)
      observation[OBSERVATION_FEATURE_NAMES.index("stock_requested_torque")] = 1.0

      self.assertAlmostEqual(policy.predict_delta(observation), 0.05, places=7)

  def test_old_linear_policy_artifact_uses_first_legacy_features(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      policy_path = Path(temp_dir) / "old_policy.npz"
      legacy_dim = 10
      weight = np.zeros(legacy_dim, dtype=np.float32)
      weight[3] = 10.0
      np.savez(
        policy_path,
        feature_mu=np.zeros(legacy_dim, dtype=np.float32),
        feature_sigma=np.ones(legacy_dim, dtype=np.float32),
        weight=weight,
        bias=np.asarray([0.0], dtype=np.float32),
        max_delta=np.asarray([0.05], dtype=np.float32),
      )

      policy = ResidualTorquePolicy.from_path(str(policy_path), max_delta=0.1)
      observation = np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32)
      observation[OBSERVATION_FEATURE_NAMES.index("stock_requested_torque")] = 1.0

      self.assertAlmostEqual(policy.predict_delta(observation), 0.05, places=7)

  def test_zero_policy_is_active_but_preserves_stock_delta(self):
    policy = ResidualTorquePolicy.from_path("zero", max_delta=0.2)

    self.assertTrue(policy.enabled)
    self.assertEqual(policy.predict_delta(np.ones(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32)), 0.0)

  def test_loaded_mlp_policy_outputs_bounded_residual_delta(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      policy_path = Path(temp_dir) / "mlp_policy.npz"
      input_dim = len(OBSERVATION_FEATURE_NAMES)
      np.savez_compressed(
        policy_path,
        feature_mu=np.zeros(input_dim, dtype=np.float32),
        feature_sigma=np.ones(input_dim, dtype=np.float32),
        weight=np.zeros(input_dim, dtype=np.float32),
        bias=np.asarray([0.0], dtype=np.float32),
        max_delta=np.asarray([0.05], dtype=np.float32),
        policy_type=np.asarray(["mlp_tanh"]),
        mlp_layer_count=np.asarray([2], dtype=np.int32),
        mlp_weight_0=np.ones((input_dim, 4), dtype=np.float32) * 0.1,
        mlp_bias_0=np.zeros(4, dtype=np.float32),
        mlp_weight_1=np.ones((4, 1), dtype=np.float32) * 0.2,
        mlp_bias_1=np.zeros(1, dtype=np.float32),
      )

      policy = ResidualTorquePolicy.from_path(str(policy_path), max_delta=0.05)
      delta = policy.predict_delta(np.ones(input_dim, dtype=np.float32))

      self.assertTrue(policy.enabled)
      self.assertEqual(policy.policy_type, "mlp_tanh")
      self.assertLessEqual(abs(delta), 0.05)
      self.assertGreater(delta, 0.0)

  def test_non_opposing_torque_guard_blocks_high_curvature_opposing_delta(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      policy_path = Path(temp_dir) / "guarded_policy.npz"
      input_dim = len(OBSERVATION_FEATURE_NAMES)
      weight = np.zeros(input_dim, dtype=np.float32)
      weight[OBSERVATION_FEATURE_NAMES.index("stock_requested_torque")] = -1.0
      np.savez(
        policy_path,
        feature_mu=np.zeros(input_dim, dtype=np.float32),
        feature_sigma=np.ones(input_dim, dtype=np.float32),
        weight=weight,
        bias=np.asarray([0.0], dtype=np.float32),
        max_delta=np.asarray([0.1], dtype=np.float32),
        observation_feature_names=np.asarray(OBSERVATION_FEATURE_NAMES),
        non_opposing_torque_guard=np.asarray([True]),
        guard_min_abs_desired_curvature=np.asarray([0.02], dtype=np.float32),
        guard_min_abs_steer_ratio=np.asarray([0.5], dtype=np.float32),
        guard_min_abs_stock_torque=np.asarray([0.2], dtype=np.float32),
      )

      policy = ResidualTorquePolicy.from_path(str(policy_path), max_delta=0.1)
      observation = np.zeros(input_dim, dtype=np.float32)
      observation[OBSERVATION_FEATURE_NAMES.index("desired_curvature")] = 0.05
      observation[OBSERVATION_FEATURE_NAMES.index("abs_steer_angle_ratio")] = 0.8
      observation[OBSERVATION_FEATURE_NAMES.index("stock_requested_torque")] = 0.5
      observation[OBSERVATION_FEATURE_NAMES.index("stock_applied_torque")] = 0.5

      self.assertEqual(policy.predict_delta(observation), 0.0)

  def test_non_opposing_torque_guard_allows_low_curvature_opposing_delta(self):
    policy = ResidualTorquePolicy(
      feature_mu=np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      feature_sigma=np.ones(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      weight=np.eye(1, len(OBSERVATION_FEATURE_NAMES), OBSERVATION_FEATURE_NAMES.index("stock_requested_torque")).reshape(-1) * -1.0,
      bias=0.0,
      max_delta=0.1,
      enabled=True,
      non_opposing_torque_guard=True,
      guard_min_abs_desired_curvature=0.02,
      guard_min_abs_steer_ratio=0.5,
      guard_min_abs_stock_torque=0.2,
    )
    observation = np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32)
    observation[OBSERVATION_FEATURE_NAMES.index("desired_curvature")] = 0.01
    observation[OBSERVATION_FEATURE_NAMES.index("abs_steer_angle_ratio")] = 0.8
    observation[OBSERVATION_FEATURE_NAMES.index("stock_requested_torque")] = 0.5
    observation[OBSERVATION_FEATURE_NAMES.index("stock_applied_torque")] = 0.5

    self.assertLess(policy.predict_delta(observation), 0.0)

  def test_non_opposing_torque_guard_can_cap_opposing_delta(self):
    policy = ResidualTorquePolicy(
      feature_mu=np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      feature_sigma=np.ones(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      weight=np.eye(1, len(OBSERVATION_FEATURE_NAMES), OBSERVATION_FEATURE_NAMES.index("stock_requested_torque")).reshape(-1) * -1.0,
      bias=0.0,
      max_delta=0.1,
      enabled=True,
      non_opposing_torque_guard=True,
      guard_min_abs_desired_curvature=0.02,
      guard_min_abs_steer_ratio=0.5,
      guard_min_abs_stock_torque=0.2,
      guard_max_opposing_delta=0.03,
    )
    observation = np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32)
    observation[OBSERVATION_FEATURE_NAMES.index("desired_curvature")] = 0.05
    observation[OBSERVATION_FEATURE_NAMES.index("abs_steer_angle_ratio")] = 0.8
    observation[OBSERVATION_FEATURE_NAMES.index("stock_requested_torque")] = 0.5
    observation[OBSERVATION_FEATURE_NAMES.index("stock_applied_torque")] = 0.5

    self.assertAlmostEqual(policy.predict_delta(observation), -0.03, places=7)

  def test_residual_delta_rate_limit_tracks_state(self):
    policy = ResidualTorquePolicy(max_delta=0.5, enabled=True, residual_delta_rate_limit=0.1)
    state = ResidualPolicyState()

    self.assertAlmostEqual(policy.rate_limit_delta(0.35, state), 0.1)
    self.assertAlmostEqual(policy.rate_limit_delta(0.35, state), 0.2)
    self.assertAlmostEqual(policy.rate_limit_delta(-0.35, state), 0.1)
    state.reset()
    self.assertAlmostEqual(policy.rate_limit_delta(-0.35, state), -0.1)

  def test_residual_activation_gate_suppresses_easy_low_turn_states(self):
    policy = ResidualTorquePolicy(
      feature_mu=np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      feature_sigma=np.ones(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      weight=np.eye(1, len(OBSERVATION_FEATURE_NAMES), OBSERVATION_FEATURE_NAMES.index("stock_requested_torque")).reshape(-1) * 10.0,
      bias=0.0,
      max_delta=0.3,
      enabled=True,
      residual_activation_min_abs_desired_curvature=0.006,
      residual_activation_min_abs_model_heading_deg=3.0,
      residual_activation_min_abs_vehicle_heading_deg=3.0,
    )
    observation = np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32)
    observation[OBSERVATION_FEATURE_NAMES.index("stock_requested_torque")] = 1.0
    observation[OBSERVATION_FEATURE_NAMES.index("desired_curvature")] = 0.002
    observation[OBSERVATION_FEATURE_NAMES.index("model_desired_heading_integral_deg")] = 1.0
    observation[OBSERVATION_FEATURE_NAMES.index("vehicle_heading_change_deg")] = 1.0

    self.assertEqual(policy.predict_delta(observation), 0.0)

    observation[OBSERVATION_FEATURE_NAMES.index("desired_curvature")] = 0.006
    self.assertAlmostEqual(policy.predict_delta(observation), 0.3, places=7)

  def test_residual_activation_gate_allows_model_or_vehicle_heading(self):
    policy = ResidualTorquePolicy(
      feature_mu=np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      feature_sigma=np.ones(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      weight=np.eye(1, len(OBSERVATION_FEATURE_NAMES), OBSERVATION_FEATURE_NAMES.index("stock_requested_torque")).reshape(-1) * 10.0,
      bias=0.0,
      max_delta=0.3,
      enabled=True,
      residual_activation_min_abs_desired_curvature=0.006,
      residual_activation_min_abs_model_heading_deg=3.0,
      residual_activation_min_abs_vehicle_heading_deg=3.0,
    )
    observation = np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32)
    observation[OBSERVATION_FEATURE_NAMES.index("stock_requested_torque")] = 1.0
    observation[OBSERVATION_FEATURE_NAMES.index("desired_curvature")] = 0.001
    observation[OBSERVATION_FEATURE_NAMES.index("model_desired_heading_integral_deg")] = 3.1
    self.assertAlmostEqual(policy.predict_delta(observation), 0.3, places=7)

    observation[OBSERVATION_FEATURE_NAMES.index("model_desired_heading_integral_deg")] = 0.0
    observation[OBSERVATION_FEATURE_NAMES.index("vehicle_heading_change_deg")] = -3.1
    self.assertAlmostEqual(policy.predict_delta(observation), 0.3, places=7)

  def test_residual_activation_can_require_speed_window(self):
    policy = ResidualTorquePolicy(
      feature_mu=np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      feature_sigma=np.ones(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      weight=np.eye(1, len(OBSERVATION_FEATURE_NAMES), OBSERVATION_FEATURE_NAMES.index("stock_requested_torque")).reshape(-1) * 10.0,
      bias=0.0,
      max_delta=0.5,
      enabled=True,
      residual_activation_min_speed_mps=5.0,
      residual_activation_max_speed_mps=9.0,
    )
    observation = np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32)
    observation[OBSERVATION_FEATURE_NAMES.index("stock_requested_torque")] = 1.0

    observation[OBSERVATION_FEATURE_NAMES.index("speed_mps")] = 4.9
    self.assertEqual(policy.predict_delta(observation), 0.0)

    observation[OBSERVATION_FEATURE_NAMES.index("speed_mps")] = 7.0
    self.assertAlmostEqual(policy.predict_delta(observation), 0.5, places=7)

    observation[OBSERVATION_FEATURE_NAMES.index("speed_mps")] = 9.1
    self.assertEqual(policy.predict_delta(observation), 0.0)

  def test_applied_torque_sign_hold_suppresses_brief_reversal(self):
    policy = ResidualTorquePolicy(
      max_delta=0.5,
      enabled=True,
      applied_torque_sign_hold_frames=2,
      applied_torque_sign_hold_min_abs_torque=0.02,
    )
    state = ResidualPolicyState()
    observation = np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32)
    observation[OBSERVATION_FEATURE_NAMES.index("stock_applied_torque")] = 0.1

    self.assertAlmostEqual(policy.stabilize_applied_delta(0.05, observation, state), 0.05)
    self.assertEqual(state.applied_torque_sign, 1.0)

    observation[OBSERVATION_FEATURE_NAMES.index("stock_applied_torque")] = -0.1
    self.assertAlmostEqual(policy.stabilize_applied_delta(-0.05, observation, state), 0.1)
    self.assertEqual(state.applied_torque_sign, 1.0)
    self.assertEqual(state.pending_applied_torque_sign_frames, 1)
    self.assertAlmostEqual(policy.stabilize_applied_delta(-0.05, observation, state), 0.1)
    self.assertEqual(state.pending_applied_torque_sign_frames, 2)
    self.assertAlmostEqual(policy.stabilize_applied_delta(-0.05, observation, state), -0.05)
    self.assertEqual(state.applied_torque_sign, -1.0)

  def test_applied_torque_sign_hold_can_be_gated_to_post_turn_low_curvature(self):
    policy = ResidualTorquePolicy(
      max_delta=0.5,
      enabled=True,
      applied_torque_sign_hold_frames=2,
      applied_torque_sign_hold_min_abs_torque=0.02,
      applied_torque_sign_hold_min_abs_vehicle_heading_deg=75.0,
      applied_torque_sign_hold_max_abs_desired_curvature=0.012,
    )
    state = ResidualPolicyState()
    observation = np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32)
    observation[OBSERVATION_FEATURE_NAMES.index("stock_applied_torque")] = 0.1
    observation[OBSERVATION_FEATURE_NAMES.index("vehicle_heading_change_deg")] = 80.0
    observation[OBSERVATION_FEATURE_NAMES.index("desired_curvature")] = 0.02

    self.assertAlmostEqual(policy.stabilize_applied_delta(0.05, observation, state), 0.05)
    observation[OBSERVATION_FEATURE_NAMES.index("stock_applied_torque")] = -0.1
    self.assertAlmostEqual(policy.stabilize_applied_delta(-0.05, observation, state), -0.05)
    self.assertEqual(state.applied_torque_sign, -1.0)

    observation[OBSERVATION_FEATURE_NAMES.index("stock_applied_torque")] = 0.1
    observation[OBSERVATION_FEATURE_NAMES.index("desired_curvature")] = 0.005
    self.assertAlmostEqual(policy.stabilize_applied_delta(0.05, observation, state), -0.1)
    self.assertEqual(state.applied_torque_sign, -1.0)

  def test_post_turn_output_cap_only_applies_after_vehicle_heading_gate(self):
    policy = ResidualTorquePolicy(
      feature_mu=np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      feature_sigma=np.ones(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      weight=np.eye(1, len(OBSERVATION_FEATURE_NAMES), OBSERVATION_FEATURE_NAMES.index("stock_requested_torque")).reshape(-1) * 10.0,
      bias=0.0,
      max_delta=1.0,
      enabled=True,
      policy_output_max_delta=0.45,
      post_turn_policy_output_max_delta=0.75,
      post_turn_policy_output_min_abs_vehicle_heading_deg=75.0,
      post_turn_policy_output_max_abs_vehicle_heading_deg=100.0,
      post_turn_policy_output_max_abs_desired_curvature=0.012,
    )
    observation = np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32)
    observation[OBSERVATION_FEATURE_NAMES.index("stock_requested_torque")] = 1.0
    observation[OBSERVATION_FEATURE_NAMES.index("desired_curvature")] = 0.005
    observation[OBSERVATION_FEATURE_NAMES.index("vehicle_heading_change_deg")] = 74.0
    self.assertAlmostEqual(policy.predict_delta(observation), 0.45, places=7)

    observation[OBSERVATION_FEATURE_NAMES.index("vehicle_heading_change_deg")] = 75.0
    self.assertAlmostEqual(policy.predict_delta(observation), 0.75, places=7)

    observation[OBSERVATION_FEATURE_NAMES.index("vehicle_heading_change_deg")] = 101.0
    self.assertAlmostEqual(policy.predict_delta(observation), 0.45, places=7)

    observation[OBSERVATION_FEATURE_NAMES.index("vehicle_heading_change_deg")] = 75.0
    observation[OBSERVATION_FEATURE_NAMES.index("desired_curvature")] = 0.02
    self.assertAlmostEqual(policy.predict_delta(observation), 0.45, places=7)

  def test_turn_assist_floor_is_gated_by_speed_heading_and_curvature(self):
    policy = ResidualTorquePolicy(
      feature_mu=np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      feature_sigma=np.ones(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      weight=np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      bias=0.0,
      max_delta=1.0,
      enabled=True,
      policy_output_max_delta=0.45,
      turn_assist_applied_torque_floor=0.8,
      turn_assist_max_speed_mps=5.0,
      turn_assist_min_abs_vehicle_heading_deg=55.0,
      turn_assist_max_abs_vehicle_heading_deg=88.0,
      turn_assist_min_abs_desired_curvature=0.015,
      turn_assist_max_abs_desired_curvature=0.07,
    )
    observation = np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32)
    observation[OBSERVATION_FEATURE_NAMES.index("speed_mps")] = 4.0
    observation[OBSERVATION_FEATURE_NAMES.index("desired_curvature")] = -0.04
    observation[OBSERVATION_FEATURE_NAMES.index("vehicle_heading_change_deg")] = 60.0
    observation[OBSERVATION_FEATURE_NAMES.index("stock_applied_torque")] = 0.3
    self.assertAlmostEqual(policy.predict_delta(observation), 0.5, places=7)

    observation[OBSERVATION_FEATURE_NAMES.index("speed_mps")] = 7.0
    self.assertEqual(policy.predict_delta(observation), 0.0)

    observation[OBSERVATION_FEATURE_NAMES.index("speed_mps")] = 4.0
    observation[OBSERVATION_FEATURE_NAMES.index("vehicle_heading_change_deg")] = 90.0
    self.assertEqual(policy.predict_delta(observation), 0.0)

  def test_turn_assist_floor_can_use_model_heading_preview_gate(self):
    policy = ResidualTorquePolicy(
      feature_mu=np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      feature_sigma=np.ones(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      weight=np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      bias=0.0,
      max_delta=1.0,
      enabled=True,
      policy_output_max_delta=0.45,
      turn_assist_applied_torque_floor=0.6,
      turn_assist_max_speed_mps=5.0,
      turn_assist_min_abs_model_heading_deg=8.0,
      turn_assist_max_abs_model_heading_deg=45.0,
      turn_assist_min_abs_vehicle_heading_deg=40.0,
      turn_assist_max_abs_vehicle_heading_deg=88.0,
      turn_assist_min_abs_desired_curvature=0.015,
      turn_assist_max_abs_desired_curvature=0.07,
    )
    observation = np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32)
    observation[OBSERVATION_FEATURE_NAMES.index("speed_mps")] = 4.0
    observation[OBSERVATION_FEATURE_NAMES.index("desired_curvature")] = -0.04
    observation[OBSERVATION_FEATURE_NAMES.index("vehicle_heading_change_deg")] = 20.0
    observation[OBSERVATION_FEATURE_NAMES.index("model_desired_heading_integral_deg")] = -8.0
    observation[OBSERVATION_FEATURE_NAMES.index("stock_applied_torque")] = 0.2
    self.assertAlmostEqual(policy.predict_delta(observation), 0.4, places=7)

    observation[OBSERVATION_FEATURE_NAMES.index("model_desired_heading_integral_deg")] = -7.9
    self.assertEqual(policy.predict_delta(observation), 0.0)

    observation[OBSERVATION_FEATURE_NAMES.index("model_desired_heading_integral_deg")] = -46.0
    self.assertEqual(policy.predict_delta(observation), 0.0)

  def test_late_exit_suppression_clamps_same_sign_model_curvature_after_turn(self):
    policy = ResidualTorquePolicy(
      feature_mu=np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      feature_sigma=np.ones(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      weight=np.eye(1, len(OBSERVATION_FEATURE_NAMES), OBSERVATION_FEATURE_NAMES.index("stock_requested_torque")).reshape(-1) * 10.0,
      bias=0.0,
      max_delta=0.8,
      enabled=True,
      policy_output_max_delta=0.45,
      late_exit_suppress_max_abs_delta=0.05,
      late_exit_suppress_max_speed_mps=5.0,
      late_exit_suppress_min_abs_vehicle_heading_deg=97.0,
      late_exit_suppress_min_abs_desired_curvature=0.001,
    )
    observation = np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32)
    observation[OBSERVATION_FEATURE_NAMES.index("speed_mps")] = 4.0
    observation[OBSERVATION_FEATURE_NAMES.index("stock_requested_torque")] = 1.0
    observation[OBSERVATION_FEATURE_NAMES.index("desired_curvature")] = 0.004
    observation[OBSERVATION_FEATURE_NAMES.index("vehicle_heading_change_deg")] = 98.0
    self.assertAlmostEqual(policy.predict_delta(observation), 0.05, places=7)

    observation[OBSERVATION_FEATURE_NAMES.index("desired_curvature")] = -0.004
    self.assertAlmostEqual(policy.predict_delta(observation), 0.45, places=7)

  def test_late_exit_suppression_can_use_lower_fast_heading_gate(self):
    policy = ResidualTorquePolicy(
      feature_mu=np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      feature_sigma=np.ones(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      weight=np.eye(1, len(OBSERVATION_FEATURE_NAMES), OBSERVATION_FEATURE_NAMES.index("stock_requested_torque")).reshape(-1) * 10.0,
      bias=0.0,
      max_delta=0.8,
      enabled=True,
      policy_output_max_delta=0.45,
      late_exit_suppress_max_abs_delta=0.05,
      late_exit_suppress_max_speed_mps=12.0,
      late_exit_suppress_min_abs_vehicle_heading_deg=90.0,
      late_exit_suppress_fast_min_speed_mps=5.0,
      late_exit_suppress_fast_min_abs_vehicle_heading_deg=70.0,
      late_exit_suppress_min_abs_desired_curvature=0.001,
    )
    observation = np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32)
    observation[OBSERVATION_FEATURE_NAMES.index("stock_requested_torque")] = 1.0
    observation[OBSERVATION_FEATURE_NAMES.index("desired_curvature")] = 0.004
    observation[OBSERVATION_FEATURE_NAMES.index("vehicle_heading_change_deg")] = 75.0

    observation[OBSERVATION_FEATURE_NAMES.index("speed_mps")] = 4.0
    self.assertAlmostEqual(policy.predict_delta(observation), 0.45, places=7)

    observation[OBSERVATION_FEATURE_NAMES.index("speed_mps")] = 7.0
    self.assertAlmostEqual(policy.predict_delta(observation), 0.05, places=7)

  def test_saturation_torque_limit_can_reduce_stock_torque_beyond_learned_output_cap(self):
    policy = ResidualTorquePolicy(
      feature_mu=np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      feature_sigma=np.ones(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      weight=np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      bias=0.0,
      max_delta=0.8,
      enabled=True,
      policy_output_max_delta=0.1,
      saturation_torque_limit=0.25,
      saturation_torque_limit_min_steer_ratio=0.98,
      saturation_torque_limit_min_abs_stock_torque=0.8,
      saturation_torque_limit_min_abs_desired_curvature=0.04,
    )
    observation = np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32)
    observation[OBSERVATION_FEATURE_NAMES.index("desired_curvature")] = -0.06
    observation[OBSERVATION_FEATURE_NAMES.index("abs_steer_angle_ratio")] = 1.0
    observation[OBSERVATION_FEATURE_NAMES.index("stock_requested_torque")] = 1.0
    observation[OBSERVATION_FEATURE_NAMES.index("stock_applied_torque")] = 1.0

    self.assertAlmostEqual(policy.predict_delta(observation), -0.75, places=7)

  def test_residual_saturation_fade_scales_residual_without_clamping_stock_torque(self):
    policy = ResidualTorquePolicy(
      feature_mu=np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      feature_sigma=np.ones(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      weight=np.eye(1, len(OBSERVATION_FEATURE_NAMES), OBSERVATION_FEATURE_NAMES.index("stock_requested_torque")).reshape(-1) * 10.0,
      bias=0.0,
      max_delta=0.8,
      enabled=True,
      policy_output_max_delta=0.4,
      residual_saturation_fade_start_steer_ratio=0.8,
      residual_saturation_fade_end_steer_ratio=1.0,
      residual_saturation_fade_min_scale=0.25,
      residual_saturation_fade_min_abs_stock_torque=0.5,
      residual_saturation_fade_min_abs_desired_curvature=0.04,
    )
    observation = np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32)
    observation[OBSERVATION_FEATURE_NAMES.index("desired_curvature")] = -0.06
    observation[OBSERVATION_FEATURE_NAMES.index("stock_requested_torque")] = 1.0
    observation[OBSERVATION_FEATURE_NAMES.index("stock_applied_torque")] = 0.8

    observation[OBSERVATION_FEATURE_NAMES.index("abs_steer_angle_ratio")] = 0.7
    self.assertAlmostEqual(policy.predict_delta(observation), 0.4, places=7)

    observation[OBSERVATION_FEATURE_NAMES.index("abs_steer_angle_ratio")] = 0.9
    self.assertAlmostEqual(policy.predict_delta(observation), 0.25, places=7)

    observation[OBSERVATION_FEATURE_NAMES.index("abs_steer_angle_ratio")] = 1.0
    self.assertAlmostEqual(policy.predict_delta(observation), 0.1, places=7)

  def test_saturation_torque_limit_does_not_change_unsaturated_states(self):
    policy = ResidualTorquePolicy(
      feature_mu=np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      feature_sigma=np.ones(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      weight=np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      bias=0.0,
      max_delta=0.8,
      enabled=True,
      policy_output_max_delta=0.1,
      saturation_torque_limit=0.25,
      saturation_torque_limit_min_steer_ratio=0.98,
      saturation_torque_limit_min_abs_stock_torque=0.8,
      saturation_torque_limit_min_abs_desired_curvature=0.04,
    )
    observation = np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32)
    observation[OBSERVATION_FEATURE_NAMES.index("desired_curvature")] = -0.06
    observation[OBSERVATION_FEATURE_NAMES.index("abs_steer_angle_ratio")] = 0.5
    observation[OBSERVATION_FEATURE_NAMES.index("stock_requested_torque")] = 1.0
    observation[OBSERVATION_FEATURE_NAMES.index("stock_applied_torque")] = 1.0

    self.assertEqual(policy.predict_delta(observation), 0.0)

  def test_saturation_torque_limit_can_be_speed_limited(self):
    policy = ResidualTorquePolicy(
      feature_mu=np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      feature_sigma=np.ones(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      weight=np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      bias=0.0,
      max_delta=0.8,
      enabled=True,
      policy_output_max_delta=0.1,
      saturation_torque_limit=0.25,
      saturation_torque_limit_max_speed_mps=5.0,
      saturation_torque_limit_min_steer_ratio=0.98,
      saturation_torque_limit_min_abs_stock_torque=0.8,
      saturation_torque_limit_min_abs_desired_curvature=0.04,
    )
    observation = np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32)
    observation[OBSERVATION_FEATURE_NAMES.index("speed_mps")] = 4.0
    observation[OBSERVATION_FEATURE_NAMES.index("desired_curvature")] = -0.06
    observation[OBSERVATION_FEATURE_NAMES.index("abs_steer_angle_ratio")] = 1.0
    observation[OBSERVATION_FEATURE_NAMES.index("stock_requested_torque")] = 1.0
    observation[OBSERVATION_FEATURE_NAMES.index("stock_applied_torque")] = 1.0
    self.assertAlmostEqual(policy.predict_delta(observation), -0.75, places=7)

    observation[OBSERVATION_FEATURE_NAMES.index("speed_mps")] = 7.0
    self.assertEqual(policy.predict_delta(observation), 0.0)

  def test_saturation_torque_limit_can_be_speed_window_limited(self):
    policy = ResidualTorquePolicy(
      feature_mu=np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      feature_sigma=np.ones(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      weight=np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      bias=0.0,
      max_delta=0.8,
      enabled=True,
      policy_output_max_delta=0.1,
      saturation_torque_limit=0.25,
      saturation_torque_limit_min_speed_mps=4.0,
      saturation_torque_limit_max_speed_mps=5.0,
      saturation_torque_limit_min_steer_ratio=0.98,
      saturation_torque_limit_min_abs_stock_torque=0.8,
      saturation_torque_limit_min_abs_desired_curvature=0.04,
    )
    observation = np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32)
    observation[OBSERVATION_FEATURE_NAMES.index("desired_curvature")] = -0.06
    observation[OBSERVATION_FEATURE_NAMES.index("abs_steer_angle_ratio")] = 1.0
    observation[OBSERVATION_FEATURE_NAMES.index("stock_requested_torque")] = 1.0
    observation[OBSERVATION_FEATURE_NAMES.index("stock_applied_torque")] = 1.0

    observation[OBSERVATION_FEATURE_NAMES.index("speed_mps")] = 3.9
    self.assertEqual(policy.predict_delta(observation), 0.0)

    observation[OBSERVATION_FEATURE_NAMES.index("speed_mps")] = 4.5
    self.assertAlmostEqual(policy.predict_delta(observation), -0.75, places=7)

    observation[OBSERVATION_FEATURE_NAMES.index("speed_mps")] = 5.1
    self.assertEqual(policy.predict_delta(observation), 0.0)

  def test_saturation_torque_limit_can_require_model_turn_progress(self):
    policy = ResidualTorquePolicy(
      feature_mu=np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      feature_sigma=np.ones(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      weight=np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      bias=0.0,
      max_delta=0.8,
      enabled=True,
      policy_output_max_delta=0.1,
      saturation_torque_limit=0.25,
      saturation_torque_limit_min_steer_ratio=0.0,
      saturation_torque_limit_min_abs_stock_torque=0.8,
      saturation_torque_limit_min_abs_desired_curvature=0.04,
      saturation_torque_limit_min_abs_model_heading_deg=75.0,
    )
    observation = np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32)
    observation[OBSERVATION_FEATURE_NAMES.index("desired_curvature")] = -0.06
    observation[OBSERVATION_FEATURE_NAMES.index("stock_requested_torque")] = 1.0
    observation[OBSERVATION_FEATURE_NAMES.index("stock_applied_torque")] = 1.0
    observation[OBSERVATION_FEATURE_NAMES.index("model_desired_heading_integral_deg")] = -74.0
    self.assertEqual(policy.predict_delta(observation), 0.0)

    observation[OBSERVATION_FEATURE_NAMES.index("model_desired_heading_integral_deg")] = -75.0
    self.assertAlmostEqual(policy.predict_delta(observation), -0.75, places=7)

  def test_saturation_torque_limit_can_require_vehicle_heading_change(self):
    policy = ResidualTorquePolicy(
      feature_mu=np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      feature_sigma=np.ones(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      weight=np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
      bias=0.0,
      max_delta=0.8,
      enabled=True,
      policy_output_max_delta=0.1,
      saturation_torque_limit=0.25,
      saturation_torque_limit_min_steer_ratio=0.0,
      saturation_torque_limit_min_abs_stock_torque=0.8,
      saturation_torque_limit_min_abs_desired_curvature=0.04,
      saturation_torque_limit_min_abs_vehicle_heading_deg=75.0,
    )
    observation = np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32)
    observation[OBSERVATION_FEATURE_NAMES.index("desired_curvature")] = -0.06
    observation[OBSERVATION_FEATURE_NAMES.index("stock_requested_torque")] = 1.0
    observation[OBSERVATION_FEATURE_NAMES.index("stock_applied_torque")] = 1.0
    observation[OBSERVATION_FEATURE_NAMES.index("vehicle_heading_change_deg")] = 74.0
    self.assertEqual(policy.predict_delta(observation), 0.0)

    observation[OBSERVATION_FEATURE_NAMES.index("vehicle_heading_change_deg")] = 75.0
    self.assertAlmostEqual(policy.predict_delta(observation), -0.75, places=7)



class TestRuntimePolicyValidation(unittest.TestCase):
  def test_bundled_artifact_loads_without_simulation_imports(self):
    artifact = Path(__file__).parents[1] / "models" / "kona_residual_policy.npz"
    policy = ResidualTorquePolicy.from_path(str(artifact))
    self.assertTrue(policy.enabled)
    self.assertEqual(policy.policy_type, "mlp_tanh")
    self.assertEqual(len(policy.policy_feature_names), 34)
    self.assertEqual([w.shape for w in policy.mlp_weights], [(34, 64), (64, 32), (32, 1)])
    self.assertAlmostEqual(policy.max_delta, 0.7, places=6)
    observation = build_observation(ResidualPolicyState(), 4.0, 20.0, 0.025, 0.2, 0.1)
    delta = policy.predict_delta(observation)
    self.assertTrue(np.isfinite(delta))
    self.assertLessEqual(abs(delta), policy.max_delta)
    subprocess.run([
      sys.executable, "-B", "-c",
      "import sys; from openpilot.sunnypilot.selfdrive.controls.lib import kona_residual_policy; "
      "assert not any(name.startswith('openpilot.tools.sim') for name in sys.modules)",
    ], cwd=Path(__file__).resolve().parents[6], check=True, capture_output=True, text=True)

  def test_disabled_and_zero_policy_preserve_delta(self):
    observation = np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32)
    for policy in (ResidualTorquePolicy.disabled(), ResidualTorquePolicy.zero()):
      self.assertEqual(policy.predict_delta(observation), 0.0)
    self.assertFalse(ResidualTorquePolicy.from_path("").enabled)
    self.assertTrue(ResidualTorquePolicy.from_path("zero").enabled)

  def test_reset_clears_observation_and_sign_history(self):
    state = ResidualPolicyState()
    build_observation(state, 5.0, 1.0, 0.02, 0.3, 0.2, vehicle_bearing_deg=80.0)
    state.residual_torque_delta = 0.2
    state.applied_torque_sign = 1.0
    state.pending_applied_torque_sign = -1.0
    state.pending_applied_torque_sign_frames = 2
    state.reset()
    self.assertEqual(state, ResidualPolicyState())

  def test_invalid_observations_are_rejected(self):
    policy = ResidualTorquePolicy.zero(0.1)
    for value in (float("nan"), float("inf"), -float("inf")):
      with self.subTest(value=value):
        observation = np.zeros(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32)
        observation[0] = value
        with self.assertRaises(ValueError):
          policy.predict_delta(observation)
        with self.assertRaises(ValueError):
          build_observation(ResidualPolicyState(), value, 0.0, 0.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
          policy.rate_limit_delta(value, ResidualPolicyState())
        with self.assertRaises(ValueError):
          policy.stabilize_applied_delta(0.0, observation, ResidualPolicyState())
    with self.assertRaises(ValueError):
      policy.predict_delta(np.zeros(2))
    with self.assertRaises(ValueError):
      build_observation(ResidualPolicyState(), 0., 0., 0., 0., 0., dt_s=0.)

  def test_nonfinite_policy_fields_are_rejected(self):
    for kwargs in ({"bias": float("nan")}, {"max_delta": float("inf")},
                   {"residual_delta_rate_limit": float("nan")},
                   {"feature_mu": np.full(len(OBSERVATION_FEATURE_NAMES), float("nan"))},
                   {"weight": np.full(len(OBSERVATION_FEATURE_NAMES), float("inf"))}):
      with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
        ResidualTorquePolicy(**kwargs)
    with self.assertRaises(ValueError):
      ResidualTorquePolicy.from_path("zero", float("nan"))

  def test_nonfinite_and_malformed_artifacts_are_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "invalid.npz"
      for fields in ({"bias": [float("nan")]}, {"guard_min_abs_stock_torque": [float("nan")]},
                     {"weight": np.zeros(2)}, {"feature_mu": np.zeros((2, 2))}):
        np.savez(path, **fields)
        with self.subTest(fields=list(fields)), self.assertRaises(ValueError):
          ResidualTorquePolicy.from_path(str(path))

  def test_feature_and_mlp_shapes_are_checked(self):
    for kwargs in ({"policy_feature_names": ["unknown"]},
                   {"policy_feature_names": ["speed_mps", "speed_mps"]},
                   {"policy_type": "mlp_tanh", "mlp_weights": [np.zeros((34, 2))], "mlp_biases": [np.zeros(2)]},
                   {"policy_type": "mlp_tanh", "mlp_weights": [np.zeros((33, 1))], "mlp_biases": [np.zeros(1)]}):
      with self.subTest(kwargs=list(kwargs)), self.assertRaises(ValueError):
        ResidualTorquePolicy(**kwargs)


if __name__ == "__main__":
  unittest.main()

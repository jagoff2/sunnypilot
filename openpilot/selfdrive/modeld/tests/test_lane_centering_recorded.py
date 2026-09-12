"""Native recorded geometry and analytic interpolation contracts.

The retained frames contain selected-plan speed and scalar boundary uncertainty.
This is a candidate-path regression, not an exact policy or vehicle replay.
"""
import json
from pathlib import Path
import unittest

import numpy as np

from openpilot.selfdrive.modeld.lane_centering import MODEL_X, LaneCenteringController
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan


FIXTURE = Path(__file__).with_name("fixtures") / "c4_20260912_native_lane_geometry.json"


class TestRecordedLaneReference(unittest.TestCase):
  def test_tiny_lane_noise_cannot_reintroduce_a_finite_early_curve_command(self):
    speed = 16.0
    base = np.zeros((1, ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH))
    base[0, :, Plan.VELOCITY.start] = speed
    for direction in (-1.0, 1.0):
      for seed in range(4):
        with self.subTest(direction=direction, seed=seed):
          controller = LaneCenteringController("absolute")
          controller.geometry_horizon = 60.0
          # First observed bend support is beyond 9 m; the action reaches only
          # about 7.35 m. A ten-micrometre perturbation must approach the same
          # zero-curvature result as exact straight geometry, not a finite jump.
          noise = np.random.default_rng(seed).normal(0.0, 0.00001, len(MODEL_X))
          controller.filtered_center_y = direction * 0.0015 * np.maximum(MODEL_X - 10.0, 0.0)**2 + noise
          plan, _ = controller._build_lane_plan(base, speed, 0.0)
          self.assertIsNotNone(plan)
          acceleration = controller._plan_curvature(plan, speed, 0.4594239175) * speed**2
          self.assertLess(abs(acceleration), 0.001)

  def test_smooth_cubic_road_does_not_gain_interpolation_curvature_ringing(self):
    knots = np.arange(0.0, 65.0, 5.0)
    x = np.linspace(10.0, 15.0, 1001)
    for direction in (-1.0, 1.0):
      coefficient = direction * 0.00005
      road = LaneCenteringController._interpolate_reference(knots, coefficient * knots**3)
      _, slope, second = LaneCenteringController._evaluate_reference(x, road)
      # Independent closed-form derivatives of the known smooth road. The
      # original repeated-gradient endpoint estimates produced a 9x false
      # curvature-derivative peak despite noiseless lane observations.
      np.testing.assert_allclose(slope, 3.0 * coefficient * x**2, atol=1e-10, rtol=0.0)
      np.testing.assert_allclose(second, 6.0 * coefficient * x, atol=1e-10, rtol=0.0)

  def test_native_camera_sample_density_does_not_destroy_credible_candidates(self):
    fixture = json.loads(FIXTURE.read_text())
    self.assertEqual(len(fixture["frames"]), 3)
    for frame in fixture["frames"]:
      with self.subTest(segment=frame["segment"], mono=frame["mono"]):
        lanes = np.zeros((1, 4, len(MODEL_X), 2))
        for index, (x, y) in enumerate(zip(frame["lane_x"], frame["lane_y"], strict=True)):
          np.testing.assert_allclose(x, MODEL_X, atol=0.0, rtol=0.0)
          lanes[0, index, :, 0] = y
        stds = np.broadcast_to(np.array(frame["lane_stds"])[None, :, None, None], lanes.shape).copy()
        edges = np.zeros((1, 2, len(MODEL_X), 2))
        for index, (x, y) in enumerate(zip(frame["edge_x"], frame["edge_y"], strict=True)):
          np.testing.assert_allclose(x, MODEL_X, atol=0.0, rtol=0.0)
          edges[0, index, :, 0] = y
        edge_stds = np.broadcast_to(np.array(frame["edge_stds"])[None, :, None, None], edges.shape).copy()
        output = {"plan": np.array(frame["selected_plan"])[None], "lane_lines": lanes, "lane_lines_stds": stds,
                  "lane_lines_prob": np.repeat(frame["lane_probs"], 2)[None], "road_edges": edges, "road_edges_stds": edge_stds}
        controller = LaneCenteringController("absolute")
        candidate, _, gate, _ = controller._line_candidate(output, controller._lookahead(frame["v_ego"]))
        self.assertIsNotNone(candidate, gate)
        self.assertTrue(candidate.entry_valid)
        controller._accept_candidate(candidate, 1.0)
        plan, _ = controller._build_lane_plan(output["plan"], frame["v_ego"], frame["current_curvature"])
        self.assertIsNotNone(plan)


if __name__ == "__main__":
  unittest.main()

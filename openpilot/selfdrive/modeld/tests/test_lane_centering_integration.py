import ast
import copy
import types
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from openpilot.cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan, get_curvature_from_plan, should_stop, smooth_value
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan
from openpilot.selfdrive.modeld.lane_centering import ACTION_SMOOTH_SECONDS, CAMERA_OFFSET, ENTRY_TIME, RAMP_IN_TIME
from openpilot.selfdrive.modeld.lane_centering_integration import LaneCenteringModelAdapter, update_lane_change_helpers


ROOT = Path(__file__).resolve().parents[4]


def load_function(relative_path, name, namespace, class_name=None):
  # Exercise each production action function without importing model runners,
  # native camera IPC, or loading inference hardware.
  tree = ast.parse((ROOT / relative_path).read_text())
  body = tree.body if class_name is None else next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name).body
  function = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == name)
  exec(compile(ast.Module(body=[function], type_ignores=[]), str(relative_path), "exec"), namespace)
  return namespace[name]


def native_actions():
  namespace = {"np": np, "log": log, "Plan": Plan, "ModelConstants": ModelConstants, "get_accel_from_plan": get_accel_from_plan,
               "get_curvature_from_plan": get_curvature_from_plan, "smooth_value": smooth_value, "should_stop": should_stop,
               "LAT_SMOOTH_SECONDS": 0.0, "LONG_SMOOTH_SECONDS": 0.3, "MIN_LAT_CONTROL_SPEED": 0.3}
  stock = load_function("openpilot/selfdrive/modeld/modeld.py", "get_action_from_model", namespace.copy())
  sp_namespace = namespace.copy()
  load_function("openpilot/sunnypilot/modeld_v2/fill_model_msg.py", "get_curvature_from_output", sp_namespace)
  sp_function = load_function("openpilot/sunnypilot/modeld_v2/modeld.py", "get_action_from_model", sp_namespace, "ModelState")
  model = SimpleNamespace(constants=ModelConstants, PLANPLUS_CONTROL=1.5, mlsim=False, generation=10,
                          LONG_SMOOTH_SECONDS=0.3, LAT_SMOOTH_SECONDS=0.0, MIN_LAT_CONTROL_SPEED=0.3)
  smoothed_model = copy.copy(model)
  smoothed_model.LAT_SMOOTH_SECONDS = 0.2
  return (("stock", stock), ("sp", types.MethodType(sp_function, model)),
          ("sp_model_smoothing", types.MethodType(sp_function, smoothed_model)))


class Inputs(dict):
  def __init__(self):
    super().__init__(carState=SimpleNamespace(leftBlinker=False, rightBlinker=False),
                     carControl=SimpleNamespace(currentCurvature=0.0, latActive=True))
    self.alive = True
    self.valid = True

  def all_alive(self, services):
    return self.alive

  def all_valid(self, services):
    return self.valid

  def all_freq_ok(self, services):
    return False  # Frequency is diagnostic in the captured controller.


def model_output():
  n = ModelConstants.IDX_N
  center = 0.12
  lines = np.zeros((1, 4, n, 2), dtype=np.float32)
  for i, offset in enumerate((-5.4, -1.8, 1.8, 5.4)):
    lines[0, i, :, 0] = center - CAMERA_OFFSET + offset
  edges = lines[:, [1, 2]].copy()
  plan = np.zeros((1, n, ModelConstants.PLAN_WIDTH), dtype=np.float32)
  plan[0, :, Plan.POSITION.start] = 20 * np.asarray(ModelConstants.T_IDXS)
  plan[0, :, Plan.POSITION.start + 1] = center
  plan[0, :, Plan.VELOCITY.start] = 20.0
  plan[0, :, Plan.ACCELERATION.start] = 0.3
  return {"plan": plan, "lane_lines": lines, "lane_lines_stds": np.full_like(lines, 0.05),
          "lane_lines_prob": np.full((1, 8), 0.95, dtype=np.float32), "road_edges": edges,
          "road_edges_stds": np.full_like(edges, 0.5), "desire_state": np.zeros((1, 8), dtype=np.float32)}


class TestLaneCenteringIntegration(unittest.TestCase):
  def run_frames(self, adapter, output, action_function, inputs, frames=1, calibration=True, state=log.LaneChangeState.off, gap=DT_MDL):
    result = None
    for _ in range(frames):
      timestamp = (adapter.last_timestamp_eof or 1_000_000_000) + round(gap * 1e9)
      result = adapter.update(output, action_function, inputs, calibration, state, timestamp, 20.0, 0.475, 0.475)
    return result

  def activate(self, output, action_function):
    adapter, inputs = LaneCenteringModelAdapter(), Inputs()
    selected, action, status = self.run_frames(adapter, output, action_function, inputs,
                                              round((ENTRY_TIME + RAMP_IN_TIME) / DT_MDL) + 4)
    self.assertEqual(status.state, "active")
    self.assertEqual(status.authority, 1.0)
    return adapter, inputs, selected, action

  def test_selected_path_overrides_modern_heads_and_preserves_longitudinal_action(self):
    for name, native in native_actions():
      for head in ("action", "desired_curvature", "planplus"):
        with self.subTest(model=name, head=head):
          output = model_output()
          if head == "action":
            output[head] = np.array([[2.4, -1.25]], dtype=np.float32)
          elif head == "desired_curvature":
            output[head] = np.array([[0.006]], dtype=np.float32)
          else:
            output[head] = np.zeros_like(output['plan'])
            output[head][0, :, Plan.ORIENTATION_RATE.start + 2] = 0.2
          original = copy.deepcopy(output)
          adapter, inputs, selected, _ = self.activate(output, native)
          previous_selected = adapter.previous_selected_action.desiredCurvature
          expected_base = native(output, adapter.previous_base_action, 0.475, 0.475, 20.0)
          selected, action, status = self.run_frames(adapter, output, native, inputs)
          plan = selected['plan'][0]
          raw_curvature = get_curvature_from_plan(plan[:, Plan.T_FROM_CURRENT_EULER][:, 2],
                                                 plan[:, Plan.ORIENTATION_RATE][:, 2], ModelConstants.T_IDXS, 20.0, 0.475)
          self.assertAlmostEqual(action.desiredCurvature, smooth_value(raw_curvature, previous_selected, ACTION_SMOOTH_SECONDS), places=8)
          self.assertEqual(action.desiredAcceleration, expected_base.desiredAcceleration)
          self.assertEqual(action.shouldStop, expected_base.shouldStop)
          self.assertEqual(status.path_weight, 1.0)
          for key in output:
            np.testing.assert_array_equal(output[key], original[key])
          np.testing.assert_array_equal(selected['plan'][:, :, Plan.VELOCITY.start], output['plan'][:, :, Plan.VELOCITY.start])
          np.testing.assert_array_equal(selected['plan'][:, :, Plan.ACCELERATION.start], output['plan'][:, :, Plan.ACCELERATION.start])

  def test_missing_calibration_and_unhealthy_inputs_cannot_acquire(self):
    native = native_actions()[0][1]
    for gate in ("calibration", "alive", "valid", "inactive"):
      with self.subTest(gate=gate):
        adapter, inputs, output = LaneCenteringModelAdapter(), Inputs(), model_output()
        if gate in ("alive", "valid"):
          setattr(inputs, gate, False)
        if gate == "inactive":
          inputs['carControl'].latActive = False
        selected, _, status = self.run_frames(adapter, output, native, inputs, 45, calibration=gate != "calibration")
        self.assertIs(selected, output)
        self.assertEqual(status.authority, 0.0)

  def test_blinker_lane_change_and_model_gap_exit(self):
    native = native_actions()[0][1]
    for gate in ("blinker", "lane_change", "gap"):
      with self.subTest(gate=gate):
        output = model_output()
        adapter, inputs, _, _ = self.activate(output, native)
        inputs['carState'].leftBlinker = gate == "blinker"
        _, _, status = self.run_frames(adapter, output, native, inputs, 5,
                                       state=log.LaneChangeState.laneChangeStarting if gate == "lane_change" else log.LaneChangeState.off,
                                       gap=0.4 if gate == "gap" else DT_MDL)
        self.assertEqual(status.authority, 0.0)

  def test_bad_geometry_and_policy_avoidance_abstain(self):
    native = native_actions()[0][1]
    for reason in ("width", "policy", "missing"):
      with self.subTest(reason=reason):
        output = model_output()
        if reason == "width":
          output['lane_lines'][0, 2, :, 0] = output['lane_lines'][0, 1, :, 0] + 1.0
        elif reason == "policy":
          output['plan'][0, :, Plan.POSITION.start + 1] = 1.5
        else:
          del output['lane_lines_stds']
        adapter, inputs = LaneCenteringModelAdapter(), Inputs()
        selected, _, status = self.run_frames(adapter, output, native, inputs, 45)
        self.assertIs(selected, output)
        self.assertEqual(status.authority, 0.0)

  def test_abstention_keeps_longitudinal_history_and_low_speed_curvature(self):
    for name, native in native_actions():
      with self.subTest(model=name):
        output = model_output()
        output['action'] = np.array([[1.0, -1.0]], dtype=np.float32)
        adapter, inputs = LaneCenteringModelAdapter(), Inputs()
        previous = log.ModelDataV2.Action()
        for frame in range(3):
          expected = native(output, previous, 0.475, 0.475, 0.0)
          selected, action, _ = adapter.update(output, native, inputs, False, log.LaneChangeState.off,
                                               (frame + 1) * 50_000_000, 0.0, 0.475, 0.475)
          self.assertIs(selected, output)
          self.assertEqual(action.desiredAcceleration, expected.desiredAcceleration)
          self.assertEqual(action.shouldStop, expected.shouldStop)
          self.assertEqual(action.desiredCurvature, expected.desiredCurvature)
          previous = expected

  def test_action_heads_cannot_change_lateral_targets_or_selection(self):
    for name, native in native_actions():
      for mode in ("absolute", "capped", "off"):
        with self.subTest(model=name, mode=mode):
          adapters = [LaneCenteringModelAdapter(mode), LaneCenteringModelAdapter(mode)]
          inputs = Inputs()
          phases = ((False, True, False), (True, True, False), (True, False, False),
                    (True, True, False), (True, True, True), (True, True, False))
          for active, boundaries_valid, blinker in phases:
            inputs['carControl'].latActive = active
            inputs['carState'].leftBlinker = blinker
            for _ in range(45):
              results = []
              for sign, adapter in zip((-1, 1), adapters, strict=True):
                output = model_output()
                output['action'] = np.array([[sign * 2.4, -1.0]], dtype=np.float32)
                if not boundaries_valid:
                  output['lane_lines_prob'][:] = 0.0
                results.append(self.run_frames(adapter, output, native, inputs))
              left, right = results
              np.testing.assert_array_equal(left[0]['plan'], right[0]['plan'])
              self.assertEqual(left[1].desiredCurvature, right[1].desiredCurvature)
              self.assertEqual(adapters[0].previous_base_action.desiredCurvature, adapters[1].previous_base_action.desiredCurvature)
              # Telemetry records the ignored native action head faithfully.
              self.assertEqual(replace(left[2], native_curvature=0.0), replace(right[2], native_curvature=0.0))

  def test_low_speed_holds_each_history_then_resumes_from_selected_plan(self):
    for name, native in native_actions():
      for mode, active in (("off", True), ("absolute", False)):
        with self.subTest(model=name, mode=mode, active=active):
          adapter, inputs, output = LaneCenteringModelAdapter(mode), Inputs(), model_output()
          inputs['carControl'].latActive = active
          output['action'] = np.array([[2.4, -1.0]], dtype=np.float32)
          adapter.previous_base_action = log.ModelDataV2.Action(desiredCurvature=-0.003)
          adapter.previous_selected_action = log.ModelDataV2.Action(desiredCurvature=0.004)
          for frame, speed in enumerate((0.0, 0.299, 0.3, float('nan'), 0.301), 1):
            selected, action, _ = adapter.update(output, native, inputs, True, log.LaneChangeState.off,
                                                 frame * 50_000_000, speed, 0.475, 0.475)
            self.assertIs(selected, output)
            if speed <= 0.3 or np.isnan(speed):
              self.assertAlmostEqual(action.desiredCurvature, 0.004, places=8)
              self.assertAlmostEqual(adapter.previous_base_action.desiredCurvature, -0.003, places=8)
            else:
              # This base plan is straight, irrespective of the nonzero action head.
              self.assertAlmostEqual(action.desiredCurvature, smooth_value(0.0, 0.004, ACTION_SMOOTH_SECONDS), places=8)
              self.assertAlmostEqual(adapter.previous_base_action.desiredCurvature, smooth_value(0.0, -0.003, ACTION_SMOOTH_SECONDS), places=8)

  def test_abstention_follows_curved_plan_despite_conflicting_heads(self):
    for name, native in native_actions():
      for head in ("action", "desired_curvature", "planplus"):
        for sign in (-1, 1):
          with self.subTest(model=name, head=head, sign=sign):
            adapter, inputs, output = LaneCenteringModelAdapter("off"), Inputs(), model_output()
            curvature = sign * 0.004
            output['plan'][0, :, Plan.T_FROM_CURRENT_EULER.start + 2] = curvature * 20.0 * np.asarray(ModelConstants.T_IDXS)
            output['plan'][0, :, Plan.ORIENTATION_RATE.start + 2] = curvature * 20.0
            output['plan'][0, :, Plan.ACCELERATION.start + 1] = curvature * 400.0
            if head == "action":
              output[head] = np.array([[-sign * 2.4, -1.0]], dtype=np.float32)
            elif head == "desired_curvature":
              output[head] = np.array([[-sign * 0.006]], dtype=np.float32)
            else:
              output[head] = copy.deepcopy(output['plan'])
              output[head][0, :, Plan.T_FROM_CURRENT_EULER.start + 2] *= -2.0
              output[head][0, :, Plan.ORIENTATION_RATE.start + 2] *= -2.0
            expected_long = native(output, adapter.previous_base_action, 0.475, 0.475, 20.0)
            selected, action, status = self.run_frames(adapter, output, native, inputs)
            self.assertIs(selected, output)
            self.assertAlmostEqual(action.desiredCurvature, sign * 5.0 * DT_MDL / 400.0, places=8)
            self.assertTrue(status.action_limited)
            self.assertAlmostEqual(adapter.previous_base_action.desiredCurvature,
                                   smooth_value(curvature, 0.0, ACTION_SMOOTH_SECONDS), places=8)
            self.assertEqual(action.desiredAcceleration, expected_long.desiredAcceleration)
            self.assertEqual(action.shouldStop, expected_long.shouldStop)

  def test_exit_and_reacquisition_keep_one_selected_filter_history(self):
    for name, native in native_actions():
      for exit_kind in ("ordinary", "hard"):
        with self.subTest(model=name, exit_kind=exit_kind):
          output = model_output()
          output['action'] = np.array([[2.4, -1.0]], dtype=np.float32)
          adapter, inputs, _, _ = self.activate(output, native)
          dropout = copy.deepcopy(output)
          dropout['lane_lines_prob'][:] = 0.0
          if exit_kind == "hard":
            inputs['carState'].leftBlinker = True
          saw_selected_exit = False
          saw_base_plan_fallback = False
          for _ in range(45):
            previous = adapter.previous_selected_action.desiredCurvature
            selected, action, status = self.run_frames(adapter, dropout, native, inputs)
            if selected is dropout:
              saw_base_plan_fallback = True
            else:
              saw_selected_exit = True
            plan = selected['plan'][0]
            raw = get_curvature_from_plan(plan[:, Plan.T_FROM_CURRENT_EULER][:, 2],
                                          plan[:, Plan.ORIENTATION_RATE][:, 2], ModelConstants.T_IDXS, 20.0, 0.475)
            self.assertAlmostEqual(action.desiredCurvature, smooth_value(raw, previous, ACTION_SMOOTH_SECONDS), places=8)
          self.assertTrue(saw_selected_exit)
          self.assertTrue(saw_base_plan_fallback)
          self.assertEqual(status.authority, 0.0)
          inputs['carState'].leftBlinker = False
          for _ in range(45):
            previous = adapter.previous_selected_action.desiredCurvature
            selected, action, status = self.run_frames(adapter, output, native, inputs)
            plan = selected['plan'][0]
            raw = get_curvature_from_plan(plan[:, Plan.T_FROM_CURRENT_EULER][:, 2],
                                          plan[:, Plan.ORIENTATION_RATE][:, 2], ModelConstants.T_IDXS, 20.0, 0.475)
            self.assertAlmostEqual(action.desiredCurvature, smooth_value(raw, previous, ACTION_SMOOTH_SECONDS), places=8)
          self.assertEqual(status.authority, 1.0)

  def test_action_head_disagreement_does_not_step_at_authority_endpoints(self):
    for name, native in native_actions():
      with self.subTest(model=name):
        output = model_output()
        output['action'] = np.array([[2.4, -1.0]], dtype=np.float32)
        adapter, inputs = LaneCenteringModelAdapter(), Inputs()
        first_step = None
        for _ in range(45):
          previous = adapter.previous_selected_action.desiredCurvature
          _, action, status = self.run_frames(adapter, output, native, inputs)
          step = abs(action.desiredCurvature - previous)
          if status.authority > 0.0 and first_step is None:
            first_step = step
            self.assertAlmostEqual(status.authority, 0.05)
            # Action curvature is serialized as Float32; diagnostics use Float64.
            self.assertAlmostEqual(status.requested_lateral_jerk, step * 400.0 / DT_MDL, delta=1e-5)
        self.assertIsNotNone(first_step)
        # The old baseline switch produced a 0.00235 curvature step here.
        self.assertLess(first_step, 0.0005)
        self.assertEqual(status.authority, 1.0)

        dropout = copy.deepcopy(output)
        dropout['lane_lines_prob'][:] = 0.0
        final_step = None
        previous_authority = status.authority
        for _ in range(45):
          previous = adapter.previous_selected_action.desiredCurvature
          _, action, status = self.run_frames(adapter, dropout, native, inputs)
          if previous_authority > 0.0 and status.authority == 0.0:
            final_step = abs(action.desiredCurvature - previous)
          previous_authority = status.authority
        self.assertIsNotNone(final_step)
        self.assertLess(final_step, 0.0005)

  def test_both_entrypoints_compensate_smoothing_before_inference(self):
    for relative in ("openpilot/selfdrive/modeld/modeld.py", "openpilot/sunnypilot/modeld_v2/modeld.py"):
      with self.subTest(module=relative):
        tree = ast.parse((ROOT / relative).read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        delays = [node for node in ast.walk(main) if isinstance(node, ast.Assign) and
                  any(isinstance(target, ast.Name) and target.id == "lat_delay" for target in node.targets)]
        self.assertEqual(len(delays), 1)
        self.assertEqual(ast.unparse(delays[0].value), "model.lat_delay + ACTION_SMOOTH_SECONDS")
        action_times = [node for node in ast.walk(main) if isinstance(node, ast.Assign) and
                        any(isinstance(target, ast.Name) and target.id == "lat_action_t" for target in node.targets)]
        self.assertEqual(ast.unparse(action_times[0].value), "lat_delay + frame_delay + action_delay")
        source = ast.unparse(main)
        self.assertLess(source.index('update_lane_change_helpers('), source.index('lane_centering.update('))
        self.assertIn('selected_model_output, action,', source)

  def test_relc_values_and_current_frame_desire_precede_selection(self):
    output, events = model_output(), []
    cs, sp = SimpleNamespace(), SimpleNamespace()
    class Edges:
      def update_and_fill(self, boundary, model_sp, speed):
        events.append("relc")
        np.testing.assert_array_equal(boundary.roadEdgeStds, output['road_edges_stds'][0, :, 0, 0])
        np.testing.assert_array_equal(boundary.laneLineProbs, output['lane_lines_prob'][0, 1::2])
        np.testing.assert_array_equal(boundary.roadEdges[0].y, output['road_edges'][0, 0, :, 0])
        model_sp.leftLaneChangeEdgeBlock = True
        return True, False
    class Desire:
      def update(self, state, active, probability, left, right):
        events.append("desire")
        assert left and not right
        self.lane_change_state = log.LaneChangeState.preLaneChange
    desire = Desire()
    update_lane_change_helpers(output, cs, True, 20.0, desire, Edges(), sp)
    self.assertEqual(events, ["relc", "desire"])
    self.assertTrue(sp.leftLaneChangeEdgeBlock)
    self.assertEqual(desire.lane_change_state, log.LaneChangeState.preLaneChange)


if __name__ == "__main__":
  unittest.main()

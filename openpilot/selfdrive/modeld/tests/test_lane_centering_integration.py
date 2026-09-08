import ast
import copy
import json
import types
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from openpilot.cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan, get_curvature_from_plan, should_stop, smooth_value
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan
from openpilot.selfdrive.modeld.lane_centering import (
  ACTION_SMOOTH_SECONDS, CAMERA_OFFSET, ENTRY_TIME, MAX_LATERAL_ACCEL_CORRECTION, RAMP_IN_TIME, get_lane_centering_input_status,
)
from openpilot.selfdrive.modeld.lane_centering_integration import (
  FRAME_DELAY_TAU, LaneCenteringModelAdapter, LaneCenteringTelemetry, ModelPublicationTiming,
  TELEMETRY_MAX_TRANSITIONS, update_lane_change_helpers,
)
from openpilot.selfdrive.modeld.lane_centering_safety import LaneCenteringSafetyLatch
from openpilot.selfdrive.modeld.lane_path import Corridor


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
  # A plan starts at the vehicle origin; the lane center can be offset.
  plan[0, :, Plan.POSITION.start + 1] = 0.0
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
              self.assertEqual(left[2], right[2])

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
            selected, action, _ = self.run_frames(adapter, output, native, inputs)
            self.assertIs(selected, output)
            self.assertAlmostEqual(action.desiredCurvature, smooth_value(curvature, 0.0, ACTION_SMOOTH_SECONDS), places=8)
            self.assertAlmostEqual(adapter.previous_base_action.desiredCurvature, action.desiredCurvature, places=8)
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
          # A blinker is immediate driver intent: stop selecting the lane path
          # while keeping the selected-action smoothing history continuous.
          self.assertEqual(saw_selected_exit, exit_kind == 'ordinary')
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
        frame_delays = [node for node in ast.walk(main) if isinstance(node, ast.Assign) and
                        any(isinstance(target, ast.Name) and target.id == 'frame_delay' for target in node.targets)]
        self.assertEqual(len(frame_delays), 1)
        self.assertEqual(ast.unparse(frame_delays[0].value), 'lane_centering.frame_delay')
        hold_delays = [node for node in ast.walk(main) if isinstance(node, ast.Assign) and
                       any(isinstance(target, ast.Name) and target.id == 'action_delay' for target in node.targets)]
        self.assertEqual(ast.unparse(hold_delays[0].value), 'lane_centering.action_delay')
        long_horizons = [node for node in ast.walk(main) if isinstance(node, ast.Assign) and
                         any(isinstance(target, ast.Name) and target.id == 'long_action_t' for target in node.targets)]
        self.assertEqual(ast.unparse(long_horizons[0].value), 'long_delay + frame_delay + DT_MDL / 2')
        source = ast.unparse(main)
        self.assertLess(source.index('update_lane_change_helpers('), source.index('lane_centering.update('))
        self.assertIn('selected_model_output, action,', source)
        self.assertLess(source.index('fill_model_msg('), source.index('lane_centering.fill_status('))
        self.assertLess(source.index('fill_pose_msg('), source.index('lane_centering.fill_status('))
        self.assertLess(source.index('lane_centering.fill_status('), source.rindex("pm.send('modelV2'"))
        validity = next(node for node in ast.walk(main) if isinstance(node, ast.Assign) and
                        any(ast.unparse(target) == 'modelv2_send.valid' for target in node.targets))
        # Execute the actual runner assignment: held actions cannot masquerade
        # as fresh native commands even when plan serialization succeeds.
        for native_valid in (False, True):
          for action_valid in (False, True):
            message = SimpleNamespace(valid=native_valid)
            namespace = {'modelv2_send': message, 'lane_centering': SimpleNamespace(frame_valid=action_valid)}
            exec(compile(ast.Module(body=[validity], type_ignores=[]), relative, 'exec'), namespace)
            self.assertEqual(message.valid, native_valid and action_valid)

  def test_nonmonotonic_camera_timestamps_reach_controller_as_invalid_elapsed_time(self):
    native = native_actions()[0][1]
    adapter, inputs, output = LaneCenteringModelAdapter(), Inputs(), model_output()
    for timestamp, expected_dt in ((1_000_000_000, DT_MDL), (1_000_000_000, 0.0), (950_000_000, -0.05), (1_050_000_000, 0.05)):
      with patch.object(adapter.controller, 'update', wraps=adapter.controller.update) as update:
        adapter.update(output, native, inputs, True, log.LaneChangeState.off, timestamp, 20.0, 0.475, 0.475)
      self.assertAlmostEqual(update.call_args.args[4], expected_dt)
      self.assertEqual(adapter.frame_valid, expected_dt > 0)

  def test_valid_frame_intervals_match_lateral_smoothing_and_capped_prediction(self):
    for name, native in native_actions():
      for mode in ('absolute', 'capped'):
        for dt in (0.05, 0.10, 0.275):
          with self.subTest(model=name, mode=mode, dt=dt):
            adapter, inputs, output = LaneCenteringModelAdapter(mode), Inputs(), model_output()
            self.run_frames(adapter, output, native, inputs, 40)
            adapter.previous_base_action = log.ModelDataV2.Action(desiredCurvature=-0.0004, desiredAcceleration=0.4)
            adapter.previous_selected_action = log.ModelDataV2.Action(desiredCurvature=0.001, desiredAcceleration=0.4)
            previous_base = adapter.previous_base_action.desiredCurvature
            previous_selected = adapter.previous_selected_action.desiredCurvature
            expected_native = native(output, adapter.previous_base_action, 0.475, 0.475, 20.0, lateral_smooth_seconds=0.0)
            core_status = []
            update = adapter.controller.update

            def capture(*args, update=update, core_status=core_status):
              result = update(*args)
              core_status.append(result[1])
              return result

            with patch.object(adapter.controller, 'update', side_effect=capture):
              selected, action, status = self.run_frames(adapter, output, native, inputs, gap=dt)
            plan = selected['plan'][0]
            raw = get_curvature_from_plan(plan[:, Plan.T_FROM_CURRENT_EULER][:, 2], plan[:, Plan.ORIENTATION_RATE][:, 2],
                                          ModelConstants.T_IDXS, 20.0, 0.475)
            self.assertTrue(adapter.frame_valid)
            self.assertFalse(status.safety_blocked)
            self.assertAlmostEqual(adapter.previous_base_action.desiredCurvature,
                                   smooth_value(0.0, previous_base, ACTION_SMOOTH_SECONDS, dt=dt), delta=1e-8)
            self.assertAlmostEqual(action.desiredCurvature, smooth_value(raw, previous_selected, ACTION_SMOOTH_SECONDS, dt=dt), delta=1e-8)
            self.assertAlmostEqual(core_status[0].curvature_correction,
                                   action.desiredCurvature - adapter.previous_base_action.desiredCurvature, delta=1e-8)
            self.assertAlmostEqual(core_status[0].requested_lateral_jerk,
                                   abs(action.desiredCurvature - previous_selected) * 400.0 / dt, delta=1e-5)
            self.assertEqual(action.desiredAcceleration, expected_native.desiredAcceleration)
            self.assertEqual(action.shouldStop, expected_native.shouldStop)

  def test_invalid_frame_intervals_do_not_advance_either_action_history(self):
    native, inputs, output = native_actions()[0][1], Inputs(), model_output()
    for dt in (0.0, -0.05, 0.35):
      with self.subTest(dt=dt):
        adapter = LaneCenteringModelAdapter('off')
        adapter.last_timestamp_eof = 1_000_000_000
        adapter.previous_base_action = log.ModelDataV2.Action(desiredCurvature=-0.004, desiredAcceleration=0.2)
        adapter.previous_selected_action = log.ModelDataV2.Action(desiredCurvature=0.003, desiredAcceleration=0.2)
        base, previous = adapter.previous_base_action, adapter.previous_selected_action
        _, action, _ = self.run_frames(adapter, output, native, inputs, gap=dt)
        self.assertFalse(adapter.frame_valid)
        self.assertIs(adapter.previous_base_action, base)
        self.assertIs(adapter.previous_selected_action, previous)
        self.assertIs(action, previous)

  def test_cold_engagement_preserves_usable_policy_command_when_geometry_cannot_be_certified(self):
    native = native_actions()[0][1]
    for narrow in (False, True):
      with self.subTest(narrow=narrow):
        adapter, inputs, output = LaneCenteringModelAdapter(), Inputs(), model_output()
        inputs['carControl'].latActive = False
        if narrow:
          output['lane_lines'][0, 2, :, 0] = output['lane_lines'][0, 1, :, 0] + 1.0
        _, action, status = self.run_frames(adapter, output, native, inputs)
        self.assertEqual(status.authority, 0.0)
        self.assertEqual(status.state, 'inactive')
        self.assertEqual(status.containment, 'blocked' if narrow else 'contained')
        model = log.ModelDataV2.new_message(frameId=1, timestampEof=adapter.last_timestamp_eof)
        model.action = action
        adapter.fill_status(model, status)
        self.assertTrue(model.laneCentering.valid)
        guard = LaneCenteringSafetyLatch()
        now = model.timestampEof + 150_000_000
        self.assertFalse(guard.update(model, True, True, now, False))
        # A control tick may receive the enable request before the next model
        # frame. Geometry uncertainty alone cannot prohibit engagement.
        self.assertFalse(guard.update(model, True, False, now, True))

  def test_capped_mode_blocks_unsafe_handoff_that_requires_instant_full_path_weight(self):
    native, output = native_actions()[0][1], model_output()
    # Near-field policy agrees with the lane, but farther along the raw plan
    # drifts a metre toward the right edge. The guard can replace it, but the
    # first-frame capped authority slew cannot admit that full replacement.
    plan = output['plan'][0]
    u = np.clip((plan[:, Plan.POSITION.start] - 40.0) / 40.0, 0.0, 1.0)
    slope = 30 * u**2 * (1 - u)**2 / 40.0
    second = 60 * u * (1 - u) * (1 - 2*u) / 40.0**2
    curvature = second / (1 + slope**2)**1.5
    plan[:, Plan.POSITION.start + 1] = 10*u**3 - 15*u**4 + 6*u**5
    plan[:, Plan.T_FROM_CURRENT_EULER.start + 2] = np.arctan(slope)
    plan[:, Plan.ORIENTATION_RATE.start + 2] = 20.0 * curvature
    plan[:, Plan.ACCELERATION.start + 1] = 400.0 * curvature
    statuses = {}
    for mode in ('absolute', 'capped'):
      adapter = LaneCenteringModelAdapter(mode)
      _, _, statuses[mode] = self.run_frames(adapter, output, native, Inputs())
    self.assertFalse(statuses['absolute'].safety_blocked)
    self.assertEqual(statuses['absolute'].containment, 'contained')
    self.assertEqual(statuses['absolute'].path_weight, 1.0)
    self.assertTrue(statuses['capped'].safety_blocked)
    self.assertEqual(statuses['capped'].containment, 'blocked')
    self.assertLess(statuses['capped'].path_weight, 1.0)

  def test_full_weight_certificate_checks_exact_published_float32_array(self):
    native, output = native_actions()[0][1], model_output()
    adapter, inputs, _, _ = self.activate(output, native)
    with patch.object(Corridor, 'check', autospec=True, side_effect=Corridor.check) as proofs:
      selected, _, status = self.run_frames(adapter, output, native, inputs)
    published = selected['plan']
    self.assertFalse(status.safety_blocked)
    self.assertEqual(status.path_weight, 1.0)
    self.assertEqual(published.dtype, np.dtype(np.float32))
    # The final array receives the full body proof after its publication cast.
    # Reusing that same certificate later in this frame avoids duplicate work.
    self.assertEqual(sum(call.args[1] is published for call in proofs.call_args_list), 1)
    self.assertIs(adapter.controller.certified_plan, published)

  def test_certificate_cannot_survive_into_next_frame_with_changed_boundaries(self):
    native, output = native_actions()[0][1], model_output()
    adapter, inputs, selected, _ = self.activate(output, native)
    previously_certified = selected['plan']
    self.assertIs(adapter.controller.certified_plan, previously_certified)
    changed = copy.deepcopy(output)
    # Reuse the identical array object, then change the road underneath it.
    # A certificate keyed only by identity across frames would admit this path.
    changed['plan'] = previously_certified
    changed['lane_lines'][0, 2, :, 0] = changed['lane_lines'][0, 1, :, 0] + 1.0
    with patch.object(Corridor, 'check', autospec=True, side_effect=Corridor.check) as proofs:
      _, _, status = self.run_frames(adapter, changed, native, inputs)
    self.assertTrue(any(call.args[1] is previously_certified for call in proofs.call_args_list))
    self.assertTrue(status.safety_blocked)
    self.assertEqual(status.containment, 'blocked')

  def test_partial_capped_blend_receives_its_own_float32_body_proof(self):
    native, output = native_actions()[0][1], model_output()
    adapter, inputs = LaneCenteringModelAdapter('capped'), Inputs()
    for _ in range(40):
      _, _, status = self.run_frames(adapter, output, native, inputs)
      if 0.0 < status.path_weight < 1.0:
        break
    else:
      self.fail('fixture never reached partial path authority')
    with patch.object(Corridor, 'check', autospec=True, side_effect=Corridor.check) as proofs:
      selected, _, status = self.run_frames(adapter, output, native, inputs)
    published, candidate = selected['plan'], adapter.controller.certified_plan
    self.assertFalse(status.safety_blocked)
    self.assertGreater(status.path_weight, 0.0)
    self.assertLess(status.path_weight, 1.0)
    self.assertEqual(published.dtype, np.dtype(np.float32))
    self.assertIsNot(published, candidate)
    self.assertTrue(any(call.args[1] is candidate for call in proofs.call_args_list))
    self.assertTrue(any(call.args[1] is published for call in proofs.call_args_list))

  def test_capped_weight_slew_cannot_restore_a_correction_above_hard_limit(self):
    adapter, output = LaneCenteringModelAdapter('capped'), model_output()
    controller = adapter.controller
    speed, curvature = 30.0, 0.004
    time = np.asarray(ModelConstants.T_IDXS)
    model_x = np.asarray(ModelConstants.X_IDXS)
    output['plan'][0, :, Plan.POSITION.start] = speed * time
    output['plan'][0, :, Plan.VELOCITY.start] = speed
    candidate = output['plan'].copy()
    angle = curvature * speed * time
    candidate[0, :, Plan.POSITION.start] = np.sin(angle) / curvature
    candidate[0, :, Plan.POSITION.start + 1] = (1 - np.cos(angle)) / curvature
    candidate[0, :, Plan.T_FROM_CURRENT_EULER.start + 2] = angle
    candidate[0, :, Plan.ORIENTATION_RATE.start + 2] = curvature * speed
    candidate[0, :, Plan.ACCELERATION.start + 1] = curvature * speed**2
    center = 1 / curvature - np.sqrt((1 / curvature)**2 - model_x**2)
    controller.corridor = Corridor(center - 1.8, center + 1.8)
    controller.filtered_center_y = center
    controller.state = 'active'
    controller.authority = controller.last_policy_weight = controller.last_path_weight = 1.0
    # Fix the feasible candidate to isolate selection, cap and slew ordering.
    # The actual final trajectory still receives the complete body proof.
    with patch.object(controller, '_build_lane_plan', return_value=(candidate, 1.0)):
      _, status = controller._finish(output, speed, curvature, 0.475, 0.0, 0.0)
    self.assertTrue(status.safety_blocked or abs(status.curvature_correction) <= MAX_LATERAL_ACCEL_CORRECTION / speed**2 + 1e-8)

  def test_status_serialization_matches_source_frame_and_preserves_block(self):
    adapter = LaneCenteringModelAdapter('off')
    adapter.frame_valid = True
    adapter.last_timestamp_eof = 1_000_000_000
    adapter.execution_time = 0.025
    fields = asdict(adapter.controller._status())
    fields.update(containment='blocked', min_clearance=-0.2, response_time=1.5, checked_distance=62.5, safety_blocked=True,
                  collision_risk=True, policy_fallback=True)
    status = SimpleNamespace(**fields)
    model = log.ModelDataV2.new_message(frameId=9, timestampEof=adapter.last_timestamp_eof)
    with patch('openpilot.selfdrive.modeld.lane_centering_integration.model_clock_ns', return_value=1_080_000_000):
      adapter.fill_status(model, status)
    self.assertEqual(model.laneCentering.version, 2)
    self.assertTrue(model.laneCentering.valid)
    self.assertEqual(model.laneCentering.frameId, model.frameId)
    self.assertEqual(model.laneCentering.timestampEof, model.timestampEof)
    self.assertTrue(model.laneCentering.safetyBlocked)
    self.assertTrue(model.laneCentering.collisionRisk)
    self.assertTrue(model.laneCentering.policyFallback)
    self.assertAlmostEqual(model.laneCentering.minClearance, -0.2)
    self.assertAlmostEqual(model.laneCentering.checkedDistance, 62.5)
    self.assertAlmostEqual(model.laneCentering.responseTime, 1.5)
    self.assertAlmostEqual(model.laneCentering.executionTime, 0.025)
    self.assertAlmostEqual(model.laneCentering.publishAge, 0.080)
    self.assertAlmostEqual(model.laneCentering.frameDelay, DT_MDL)
    self.assertAlmostEqual(model.laneCentering.actionDelay, DT_MDL / 2)
    self.assertGreater(adapter.frame_delay, DT_MDL)
    model.timestampEof += 1
    adapter.fill_status(model, status)
    self.assertFalse(model.laneCentering.valid)

  def test_status_serialization_accepts_numpy_scalars_without_changing_diagnostics(self):
    # This exact value/type crashed modeld on a retained road corridor. Other
    # scalar fields share the same native serialization boundary, including
    # intentional nonfinite values representing unavailable diagnostics.
    fields = {'authority': 'authority', 'path_weight': 'pathWeight', 'min_clearance': 'minClearance',
              'response_time': 'responseTime', 'checked_distance': 'checkedDistance'}
    for scalar in (np.float32, np.float64):
      for field, wire_field in fields.items():
        for value in (32.61423426478729, np.nan, np.inf, -np.inf):
          with self.subTest(scalar=scalar.__name__, field=field, value=value):
            adapter = LaneCenteringModelAdapter()
            adapter.frame_valid = np.bool_(True)
            adapter.last_timestamp_eof = 1_000_000_000
            adapter.execution_time = scalar(0.025)
            adapter.timing.frame_delay = scalar(0.05)
            adapter.timing.publication_interval = scalar(0.05)
            status = replace(adapter.controller._status(), **{field: scalar(value)})
            model = log.ModelDataV2.new_message(frameId=9, timestampEof=adapter.last_timestamp_eof)
            with patch.object(adapter.timing, 'observe', return_value=scalar(0.08)):
              adapter.fill_status(model, status)
            with log.ModelDataV2.from_bytes(model.to_bytes()) as restored:
              selected = restored.laneCentering
              actual, expected = getattr(selected, wire_field), float(np.float32(value))
              if np.isnan(expected):
                self.assertTrue(np.isnan(actual))
              else:
                self.assertEqual(actual, expected)
              self.assertTrue(selected.valid)
              self.assertEqual(selected.frameId, 9)
              self.assertEqual(selected.timestampEof, adapter.last_timestamp_eof)
              for timing_field, timing_value in (('executionTime', 0.025), ('frameDelay', 0.05),
                                                 ('actionDelay', 0.025), ('publishAge', 0.08)):
                self.assertEqual(getattr(selected, timing_field), float(np.float32(timing_value)))

    for value in (False, True):
      with self.subTest(boolean=value):
        adapter = LaneCenteringModelAdapter()
        status = replace(adapter.controller._status(), safety_blocked=np.bool_(value),
                         collision_risk=np.bool_(value), policy_fallback=np.bool_(value))
        model = log.ModelDataV2.new_message()
        adapter.fill_status(model, status)
        with log.ModelDataV2.from_bytes(model.to_bytes()) as restored:
          self.assertIs(restored.laneCentering.safetyBlocked, value)
          self.assertIs(restored.laneCentering.collisionRisk, value)
          self.assertIs(restored.laneCentering.policyFallback, value)

  def test_real_lane_activation_and_retained_corridor_serialize_every_publication(self):
    for name, native in native_actions():
      with self.subTest(model=name):
        adapter, inputs, output = LaneCenteringModelAdapter(), Inputs(), model_output()
        # Finite confidence support ends before the native plan. A subsequent
        # confidence dropout exercises ego-motion propagation of that horizon,
        # which originally leaked numpy.float64 into status.checked_distance.
        output['lane_lines_stds'][:, :, 15:, :] = 0.8
        phases = set()
        frame_count = round((ENTRY_TIME + RAMP_IN_TIME) / DT_MDL) + 4
        for frame in range(frame_count + 1):
          if frame == frame_count:
            output['lane_lines_prob'][:] = 0.0
          selected, action, status = self.run_frames(adapter, output, native, inputs)
          phases.add(status.state)
          model = log.ModelDataV2.new_message(frameId=frame, timestampEof=adapter.last_timestamp_eof)
          model.action = action
          adapter.fill_status(model, status)
          with log.ModelDataV2.from_bytes(model.to_bytes()) as restored:
            self.assertTrue(restored.laneCentering.valid)
            self.assertEqual(restored.laneCentering.state, status.state)
            self.assertEqual(restored.laneCentering.containment, status.containment)
            self.assertEqual(restored.laneCentering.checkedDistance, float(np.float32(status.checked_distance)))
            self.assertEqual(restored.action.desiredCurvature, action.desiredCurvature)
        self.assertEqual(phases, {'acquiring', 'active'})
        self.assertEqual(status.state, 'active')
        self.assertEqual(status.containment, 'contained')
        self.assertGreater(adapter.controller.corridor_age, 0.0)
        self.assertIsInstance(adapter.controller.corridor.horizon, np.floating)
        self.assertLess(status.checked_distance, selected['plan'][0, -1, Plan.POSITION.start])

  def test_rejected_lane_proposal_uses_native_plan_and_base_action_history(self):
    for name, native in native_actions():
      with self.subTest(model=name):
        adapter, inputs, output = LaneCenteringModelAdapter(), Inputs(), model_output()
        adapter.previous_base_action = log.ModelDataV2.Action(desiredCurvature=-0.003, desiredAcceleration=0.2)
        adapter.previous_selected_action = log.ModelDataV2.Action(desiredCurvature=0.009, desiredAcceleration=0.2)
        status = replace(adapter.controller._status(), reason='correction_limit', containment='blocked',
                         safety_blocked=True, policy_fallback=True)
        with patch.object(adapter.controller, 'update', return_value=(output, status)):
          selected, action, actual_status = self.run_frames(adapter, output, native, inputs)
        self.assertIs(selected, output)
        self.assertTrue(adapter.frame_valid)
        self.assertTrue(actual_status.policy_fallback)
        self.assertAlmostEqual(action.desiredCurvature, smooth_value(0.0, -0.003, ACTION_SMOOTH_SECONDS), places=8)
        self.assertEqual(action.desiredCurvature, adapter.previous_base_action.desiredCurvature)
        self.assertIs(action, adapter.previous_selected_action)
        self.assertEqual(actual_status.curvature_correction, 0.0)

  def test_optional_planner_exception_publishes_native_command_and_reacquires(self):
    for name, native in native_actions():
      with self.subTest(model=name):
        output = model_output()
        adapter, inputs, _, _ = self.activate(output, native)
        failed_controller = adapter.controller
        adapter.previous_base_action = log.ModelDataV2.Action(desiredCurvature=-0.003, desiredAcceleration=0.2)
        adapter.previous_selected_action = log.ModelDataV2.Action(desiredCurvature=0.009, desiredAcceleration=0.2)
        # Fail inside the real active controller, after state/geometry updates,
        # rather than replacing its entire result with a manufactured fallback.
        with patch.object(failed_controller, '_build_lane_plan', side_effect=RuntimeError('optional geometry failure')):
          selected, action, status = self.run_frames(adapter, output, native, inputs)
        self.assertIs(selected, output)
        self.assertTrue(adapter.frame_valid)
        self.assertTrue(adapter.plan_valid)
        self.assertFalse(adapter.auxiliary_valid)
        self.assertEqual(adapter.last_optional_error, 'RuntimeError')
        self.assertEqual(status.reason, 'planner_exception')
        self.assertTrue(status.policy_fallback)
        self.assertFalse(status.collision_risk)
        self.assertFalse(status.safety_blocked)
        self.assertEqual(status.path_weight, 0.0)
        self.assertAlmostEqual(action.desiredCurvature, smooth_value(0.0, -0.003, ACTION_SMOOTH_SECONDS), places=8)
        self.assertEqual(action.desiredCurvature, adapter.previous_base_action.desiredCurvature)
        self.assertIs(adapter.previous_selected_action, action)
        self.assertIsNot(adapter.controller, failed_controller)
        self.assertIsNone(adapter.controller.filtered_center_y)
        self.assertEqual(adapter.controller.authority, 0.0)
        model = log.ModelDataV2.new_message(frameId=4, timestampEof=adapter.last_timestamp_eof)
        model.action = action
        adapter.fill_status(model, status)
        self.assertFalse(model.laneCentering.valid)
        self.assertFalse(model.laneCentering.collisionRisk)
        self.assertFalse(LaneCenteringSafetyLatch().update(model, True, True, model.timestampEof + 100_000_000, True))

        _, action, status = self.run_frames(adapter, output, native, inputs)
        self.assertTrue(adapter.auxiliary_valid)
        self.assertEqual(action.desiredCurvature, adapter.previous_base_action.desiredCurvature)
        _, _, status = self.run_frames(adapter, output, native, inputs, frames=40)
        self.assertEqual(status.state, 'active')
        self.assertEqual(status.authority, 1.0)

  def test_optional_failure_cannot_validate_invalid_native_command_or_timestamp(self):
    native = native_actions()[0][1]
    for fault in ('plan', 'action', 'duplicate_timestamp'):
      with self.subTest(fault=fault):
        adapter, inputs, output = LaneCenteringModelAdapter(), Inputs(), model_output()
        self.run_frames(adapter, output, native, inputs)
        previous_base = adapter.previous_base_action
        adapter.previous_selected_action = log.ModelDataV2.Action(desiredCurvature=0.009)
        if fault == 'plan':
          output['plan'][0, 3, 1] = np.nan
        elif fault == 'action':
          output['action'] = np.array([[0.0, np.nan]], dtype=np.float32)
        with patch.object(adapter.controller, 'update', side_effect=RuntimeError('optional failure')):
          _, action, status = self.run_frames(adapter, output, native, inputs, gap=0.0 if fault == 'duplicate_timestamp' else DT_MDL)
        self.assertFalse(adapter.frame_valid)
        self.assertIs(adapter.previous_base_action, previous_base)
        self.assertIs(action, previous_base)
        self.assertIs(adapter.previous_selected_action, previous_base)
        self.assertFalse(status.collision_risk)
        self.assertFalse(adapter.auxiliary_valid)

  def test_malformed_optional_plan_falls_back_without_invalidating_valid_native(self):
    adapter, inputs, output = LaneCenteringModelAdapter(), Inputs(), model_output()
    malformed = {**output, 'plan': np.zeros((1, 2, 3), dtype=np.float32)}
    with patch.object(adapter.controller, 'update', return_value=(malformed, adapter.controller._status())):
      selected, action, status = self.run_frames(adapter, output, native_actions()[0][1], inputs)
    self.assertIs(selected, output)
    self.assertTrue(adapter.plan_valid)
    self.assertTrue(adapter.frame_valid)
    self.assertEqual(action.desiredCurvature, adapter.previous_base_action.desiredCurvature)
    self.assertEqual(status.reason, 'planner_exception')

  def test_telemetry_transport_exception_cannot_interrupt_selected_command(self):
    native, output = native_actions()[0][1], model_output()
    adapter, inputs, _, _ = self.activate(output, native)
    # Use the real telemetry call and fail only its transport. The handler must
    # not recursively attempt to log the logger failure.
    adapter.telemetry.last_emit_time = None
    with patch('openpilot.selfdrive.modeld.lane_centering_integration.cloudlog.event', side_effect=RuntimeError('transport')) as event:
      selected, action, status = self.run_frames(adapter, output, native, inputs)
    self.assertEqual(event.call_count, 1)
    self.assertEqual(adapter.telemetry_failures, 1)
    self.assertTrue(adapter.frame_valid)
    self.assertTrue(adapter.auxiliary_valid)
    self.assertEqual(status.path_weight, 1.0)
    self.assertIsNot(selected['plan'], output['plan'])
    self.assertEqual(action.desiredCurvature, adapter.previous_selected_action.desiredCurvature)

  def test_status_failure_is_transactional_and_keeps_driving_command(self):
    native, output = native_actions()[0][1], model_output()
    adapter, inputs, selected, action = self.activate(output, native)
    status = adapter.controller._status()
    model = log.ModelDataV2.new_message(frameId=9, timestampEof=adapter.last_timestamp_eof)
    model.action = action
    model.position.x = selected['plan'][0, :, Plan.POSITION.start].tolist()
    model.position.y = selected['plan'][0, :, Plan.POSITION.start + 1].tolist()
    original_action, original_x, original_y = model.action.to_dict(), list(model.position.x), list(model.position.y)
    model.laneCentering.valid = True
    model.laneCentering.collisionRisk = True
    # A late conversion error occurs after several staged fields are populated.
    broken = replace(status, checked_distance=object())
    adapter.fill_status(model, broken)
    with log.ModelDataV2.from_bytes(model.to_bytes()) as restored:
      self.assertEqual(restored.action.to_dict(), original_action)
      self.assertEqual(list(restored.position.x), original_x)
      self.assertEqual(list(restored.position.y), original_y)
      self.assertFalse(restored.laneCentering.valid)
      self.assertFalse(restored.laneCentering.collisionRisk)
      self.assertFalse(restored.laneCentering.safetyBlocked)
      self.assertEqual(restored.laneCentering.reason, 'status_serialization_error')
    self.assertEqual(adapter.status_failures, 1)
    self.assertTrue(adapter.frame_valid)
    self.assertFalse(LaneCenteringSafetyLatch().update(model, True, True, model.timestampEof + 100_000_000, True))
    adapter.fill_status(model, status)
    self.assertTrue(model.laneCentering.valid)

  def test_plan_sequences_have_one_array_contract_without_mutating_native_arrays(self):
    for name, native in native_actions():
      with self.subTest(model=name):
        adapter, inputs, original = LaneCenteringModelAdapter(), Inputs(), model_output()
        sequence = original['plan'].tolist()
        output = {**original, 'plan': sequence}
        selected, action, _ = self.run_frames(adapter, output, native, inputs)
        self.assertTrue(adapter.frame_valid)
        self.assertIsInstance(selected['plan'], np.ndarray)
        self.assertIs(output['plan'], sequence)
        np.testing.assert_array_equal(selected['plan'], original['plan'])
        for key in original.keys() - {'plan'}:
          self.assertIs(selected[key], original[key])
        self.assertTrue(np.isfinite(action.desiredCurvature))
        # The same normalization must hold on a direct warm core call too.
        self.run_frames(adapter, original, native, inputs, frames=40)
        selected, _ = adapter.controller.update(output, 20.0, 0.0, 0.475, DT_MDL,
                                               adapter.previous_base_action.desiredCurvature,
                                               adapter.previous_selected_action.desiredCurvature,
                                               True, True, False, False, False)
        self.assertTrue(adapter.frame_valid)
        self.assertIsInstance(selected['plan'], np.ndarray)
        self.assertIs(output['plan'], sequence)
        untouched = LaneCenteringModelAdapter('off')
        selected, _, _ = self.run_frames(untouched, original, native, inputs)
        self.assertIs(selected, original)
        self.assertIs(selected['plan'], original['plan'])

  def test_invalid_plan_sequence_is_rejected_before_native_action_or_writer(self):
    for value in (np.nan, 'invalid', 10**400):
      with self.subTest(value_type=type(value).__name__):
        adapter, inputs, output = LaneCenteringModelAdapter(), Inputs(), model_output()
        output['plan'] = output['plan'].tolist()
        output['plan'][0][3][1] = value
        native = Mock(side_effect=AssertionError('invalid plan reached native action'))
        with patch.object(adapter, '_optional_failure', wraps=adapter._optional_failure) as optional_failure:
          _, action, status = self.run_frames(adapter, output, native, inputs)
        native.assert_not_called()
        optional_failure.assert_not_called()
        self.assertFalse(adapter.plan_valid)
        self.assertFalse(adapter.frame_valid)
        message = log.Event.new_message()
        message.init('modelV2')
        adapter.fill_invalid_model(message, action, status, 1, adapter.last_timestamp_eof)
        self.assertFalse(message.valid)
        self.assertFalse(message.modelV2.laneCentering.valid)

  def test_invalid_plans_and_actions_preserve_finite_history_and_recover(self):
    for name, native in native_actions():
      for fault in ('missing_plan', 'plan_shape', 'nan_plan', 'nan_accel', 'nan_action', 'action_shape'):
        with self.subTest(model=name, fault=fault):
          adapter, inputs, output = LaneCenteringModelAdapter('off'), Inputs(), model_output()
          adapter.previous_base_action = log.ModelDataV2.Action(desiredCurvature=-0.003, desiredAcceleration=0.4)
          adapter.previous_selected_action = log.ModelDataV2.Action(desiredCurvature=0.004, desiredAcceleration=0.4)
          invalid = copy.deepcopy(output)
          if fault == 'missing_plan':
            del invalid['plan']
          elif fault == 'plan_shape':
            invalid['plan'] = np.zeros((1, 2, 3), dtype=np.float32)
          elif fault == 'nan_plan':
            invalid['plan'][0, 3, Plan.T_FROM_CURRENT_EULER.start + 2] = np.nan
          else:
            invalid['action'] = np.array([[0.0, 0.1]], dtype=np.float32)
            if fault == 'nan_accel':
              invalid['action'][0, 1] = np.nan
            elif fault == 'nan_action':
              invalid['action'][0, 0] = np.nan
            else:
              invalid['action'] = np.zeros((0,), dtype=np.float32)
          selected, action, status = self.run_frames(adapter, invalid, native, inputs)
          self.assertIs(selected, invalid)
          self.assertFalse(adapter.frame_valid)
          self.assertTrue(status.safety_blocked)
          self.assertEqual(status.reason, 'invalid_action')
          self.assertAlmostEqual(adapter.previous_base_action.desiredCurvature, -0.003)
          self.assertAlmostEqual(adapter.previous_selected_action.desiredCurvature, 0.004)
          self.assertAlmostEqual(action.desiredAcceleration, 0.4)
          message = log.Event.new_message()
          message.init('modelV2')
          adapter.fill_invalid_model(message, action, status, 4, adapter.last_timestamp_eof)
          self.assertFalse(message.valid)
          self.assertFalse(message.modelV2.laneCentering.valid)
          self.assertTrue(message.modelV2.laneCentering.safetyBlocked)
          self.assertEqual(len(message.modelV2.position.x), 0)
          self.assertEqual(message.modelV2.frameId, 4)
          _, action, status = self.run_frames(adapter, output, native, inputs)
          self.assertTrue(adapter.frame_valid)
          self.assertFalse(status.safety_blocked)
          self.assertAlmostEqual(action.desiredCurvature, smooth_value(0.0, 0.004, ACTION_SMOOTH_SECONDS), places=8)
          self.assertTrue(np.isfinite(adapter.previous_base_action.desiredAcceleration))


class TestModelPublicationTiming(unittest.TestCase):
  def test_lateral_hold_uses_valid_publication_cadence(self):
    adapter = LaneCenteringModelAdapter()
    timing = adapter.timing
    self.assertEqual(adapter.action_delay, DT_MDL / 2)
    for frame in range(21):
      timestamp = 1_000_000_000 + frame * 100_000_000
      timing.observe(timestamp, timestamp + 70_000_000)
    expected = 0.100 + (DT_MDL - 0.100) * np.exp(-20 * 0.100 / FRAME_DELAY_TAU)
    self.assertAlmostEqual(adapter.action_delay, expected / 2, places=12)
    previous = timing.publication_interval
    timing.observe(3_050_000_000, 3_120_000_000, valid=False)
    self.assertEqual(timing.publication_interval, previous)
    # The next accepted command follows a100ms gap, despite an invalid model
    # message arriving in between. Measure command cadence, not message count.
    timing.observe(3_100_000_000, 3_170_000_000)
    self.assertAlmostEqual(timing.publication_interval,
                           previous + (1 - np.exp(-0.100 / FRAME_DELAY_TAU)) * (0.100 - previous), places=12)

  def test_complete_age_replaces_fixed_delay_without_double_counting(self):
    timing = ModelPublicationTiming()
    self.assertEqual(timing.frame_delay, DT_MDL)
    for frame in range(31):
      timestamp = 1_000_000_000 + frame * 50_000_000
      self.assertAlmostEqual(timing.observe(timestamp, timestamp + 70_000_000), 0.070)
    expected = 0.070 + (DT_MDL - 0.070) * np.exp(-31 * DT_MDL / FRAME_DELAY_TAU)
    self.assertAlmostEqual(timing.frame_delay, expected, places=12)
    self.assertLess(abs(timing.frame_delay - 0.070), 0.001)

  def test_invalid_source_age_order_and_gaps_do_not_train_delay(self):
    timing = ModelPublicationTiming()
    timing.observe(1_000_000_000, 1_070_000_000)
    previous = timing.frame_delay
    for timestamp, age in ((0, 0), (1_000_000_000, 0.1), (950_000_000, 0.1),
                           (1_050_000_000, -0.001), (1_100_000_000, 0.301), (2_000_000_000, 0.08)):
      with self.subTest(timestamp=timestamp, age=age):
        timing.observe(timestamp, timestamp + round(age * 1e9))
        self.assertEqual(timing.frame_delay, previous)
    timing.observe(2_050_000_000, 2_130_000_000)
    self.assertGreater(timing.frame_delay, previous)
    self.assertLess(timing.frame_delay, 0.080)

  def test_camera_phase_jitter_is_smoothed(self):
    timing, estimates = ModelPublicationTiming(), []
    for frame in range(80):
      timestamp = 1_000_000_000 + frame * 50_000_000
      timing.observe(timestamp, timestamp + (40_000_000 if frame % 2 == 0 else 80_000_000))
      if frame >= 60:
        estimates.append(timing.frame_delay)
    self.assertLess(np.ptp(estimates), 0.003)
    self.assertAlmostEqual(float(np.mean(estimates)), 0.060, delta=0.0001)


class TestLaneCenteringTelemetry(unittest.TestCase):
  def test_structured_events_preserve_original_and_selected_geometry_and_valid_json(self):
    adapter, output, inputs = LaneCenteringModelAdapter('off'), model_output(), Inputs()
    selected = copy.deepcopy(output)
    selected['plan'][0, :, Plan.POSITION.start + 1] += 0.3
    telemetry = LaneCenteringTelemetry(clock=lambda: 0.0)
    with patch('openpilot.selfdrive.modeld.lane_centering_integration.cloudlog.event') as event:
      telemetry.update('off', adapter.controller._status(), get_lane_centering_input_status(inputs, True),
                       output, selected, 1_000_000_000, DT_MDL, 20.0, 0.001, 0.002)
    self.assertEqual(event.call_args.args, ('lane_centering_status',))
    payload = event.call_args.kwargs
    json.dumps(payload, allow_nan=False)
    self.assertNotIn('error', payload)
    self.assertIsNone(payload['status']['lane_width'])
    self.assertEqual(payload['timestamp_eof'], 1_000_000_000)
    self.assertEqual(payload['transitions'][0]['mode'], 'off')
    self.assertIsNone(payload['transitions'][0]['min_clearance_m'])
    self.assertEqual(payload['transitions'][0]['checked_distance_m'], 0.0)
    geometry = payload['geometry']
    self.assertEqual(len(geometry['sample_x_m']), 5)
    self.assertNotEqual(geometry['original_path_y_m'], geometry['selected_path_y_m'])
    self.assertGreater(geometry['original_sampled_min_line_axis_aligned_body_margin_m'],
                       geometry['selected_sampled_min_line_axis_aligned_body_margin_m'])

  def test_transition_churn_is_bounded_and_health_heartbeat_is_periodic(self):
    now = [0.0]
    telemetry, output = LaneCenteringTelemetry(clock=lambda: now[0]), model_output()
    status = LaneCenteringModelAdapter('off').controller._status()
    inputs = get_lane_centering_input_status(Inputs(), True)
    with patch('openpilot.selfdrive.modeld.lane_centering_integration.cloudlog.event') as event:
      for i in range(40):
        status = replace(status, entry_gate=str(i % 2))
        telemetry.update('off', status, inputs, output, output, i + 1, DT_MDL, 20.0, 0.0, 0.0)
      self.assertEqual(event.call_count, 1)
      self.assertEqual(len(telemetry.transitions), TELEMETRY_MAX_TRANSITIONS)
      now[0] = 0.5
      telemetry.update('off', status, inputs, output, output, 41, DT_MDL, 20.0, 0.0, 0.0)
      self.assertEqual(event.call_count, 2)
      self.assertEqual(len(event.call_args.kwargs['transitions']), TELEMETRY_MAX_TRANSITIONS)
      self.assertGreater(event.call_args.kwargs['dropped_transitions'], 0)
      now[0] = 5.4
      telemetry.update('off', status, inputs, output, output, 42, DT_MDL, 20.0, 0.0, 0.0)
      self.assertEqual(event.call_count, 2)
      now[0] = 5.5
      telemetry.update('off', status, inputs, output, output, 43, DT_MDL, 20.0, 0.0, 0.0)
      self.assertEqual(event.call_count, 3)
      self.assertEqual(event.call_args.kwargs['transitions'], [])

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

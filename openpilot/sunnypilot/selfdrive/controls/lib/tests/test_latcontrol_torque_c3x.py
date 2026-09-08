"""Controller contracts and recorded-event regression using production PID/NN code."""
import ast
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.drive_helpers import MAX_LATERAL_JERK
from openpilot.sunnypilot.selfdrive.controls.lib import latcontrol_torque_c3x
from openpilot.sunnypilot.selfdrive.controls.lib import latcontrol_torque_ext_override
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc import nnlc_c3x
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.sigmoid_map_tuner import SigmoidMapTuner

LIB = Path(__file__).parents[1]
MODEL = Path(__file__).parents[4] / "neural_network_data/neural_network_lateral_control/HYUNDAI_KONA_EV.json"
MAP = LIB / "nnlc/sigmoid_maps/HYUNDAI_KONA_EV_nnlc.json"
UNWIND_FIXTURE = Path(__file__).with_name("fixtures") / "c3x_unwind_event.json"


class FakeParams(dict):
  def get_bool(self, key):
    return bool(self.get(key, False))

  def get(self, key, default=None, return_default=False):
    return super().get(key, default)


class ObservationVehicleModel:
  @staticmethod
  def calc_curvature(angle, speed, roll):
    return -angle / 160.0


def build_controller(enabled=True, controller_class=latcontrol_torque_c3x.LatControlTorque, nnlc_module=nnlc_c3x):
  torque = SimpleNamespace(kp=1.0, ki=0.2, kf=0.7, friction=0.12359762054065548,
                           latAccelFactor=3.078814714619148, latAccelOffset=0.0, steeringAngleDeadzoneDeg=0.02)
  torque.as_builder = lambda: torque
  CP = SimpleNamespace(carFingerprint="HYUNDAI_KONA_EV", steerLimitTimer=0.8, steerActuatorDelay=0.1,
                        lateralTuning=SimpleNamespace(torque=torque))
  CP_SP = SimpleNamespace(neuralNetworkLateralControl=SimpleNamespace(model=SimpleNamespace(path=str(MODEL))))
  CI = SimpleNamespace(
    torque_from_lateral_accel=lambda: lambda accel, params: accel / params.latAccelFactor,
    lateral_accel_from_torque=lambda: lambda output, params: output * params.latAccelFactor,
    torque_from_lateral_accel_in_torque_space=lambda: lambda inputs, params, gravity_adjusted=False:
      inputs.lateral_acceleration / params.latAccelFactor,
  )
  params = FakeParams(NeuralNetworkLateralControl=enabled, EnforceTorqueControl=False, TorqueParamsOverrideEnabled=False)
  with patch.object(nnlc_module, "Params", return_value=params), \
       patch.object(latcontrol_torque_ext_override, "Params", return_value=params), \
       patch.object(nnlc_module.SigmoidMapTuner, "_candidate_storage_dirs", return_value=[str(MAP.parent)]):
    return controller_class(CP, CP_SP, CI, 0.01)


def run_trace(controller):
  cases = [
    ("inactive", 10., 0., 0., False, False, False, 4),
    ("left_entry", 10., .012, .004, True, False, False, 35),
    ("left_inside", 10., .012, .022, True, False, False, 35),
    ("right_entry", 10., -.012, -.004, True, False, False, 35),
    ("right_inside", 10., -.012, -.022, True, False, False, 35),
    ("preview_enabled", 10., .024, .029, True, False, False, 25),
    ("preview_disabled", 10., .026, .035, True, False, False, 25),
    ("fast_apex", 22., .008, .012, True, False, False, 35),
    ("tight_integrator", 18., .025, .03, True, False, False, 35),
    ("low_speed", 5., -.02, -.006, True, False, False, 25),
    ("driver_override", 12., -.01, -.009, True, True, False, 12),
    ("limited", 12., .01, .006, True, False, True, 12),
    ("disengage", 12., .01, .006, False, False, False, 4),
    ("reengage", 12., .01, .006, True, False, False, 25),
  ]
  samples = []
  params = SimpleNamespace(angleOffsetDeg=0.0, roll=0.01)
  pose = SimpleNamespace(orientation=SimpleNamespace(pitch=0.02, yaw=0.0))
  for name, speed, desired, actual, active, pressed, limited, frames in cases:
    model = SimpleNamespace(
      orientation=SimpleNamespace(x=[0.0] * 33, y=[0.0] * 33),
      acceleration=SimpleNamespace(y=[desired * speed ** 2 * (1.0 + 0.01 * t) for t in ModelConstants.T_IDXS]),
    )
    controller.extension.update_model_v2(model)
    controller.extension.update_lateral_lag(0.2)
    controller.extension.update_limits()
    CS = SimpleNamespace(vEgo=speed, aEgo=0.1, steeringAngleDeg=math.degrees(actual * 160.),
                         steeringRateDeg=0.05, steeringPressed=pressed)
    for frame in range(frames):
      output, _, state = controller.update(active, CS, ObservationVehicleModel(), params, limited, desired, pose, False, 0.2)
      samples.append({"case": name, "frame": frame, "speed": speed, "active": active,
                      "factor": controller.torque_params.latAccelFactor, "values": [float(value) for value in (
          output, state.desiredLateralAccel, state.actualLateralAccel, state.error, state.p, state.i, state.f,
          controller._last_output_lataccel, controller._last_error,
        )]})
  return samples


def model_for(accel=1.0, jerk=0.1):
  return SimpleNamespace(orientation=SimpleNamespace(x=[0.] * 33, y=[0.] * 33),
                         acceleration=SimpleNamespace(y=[accel + jerk * t for t in ModelConstants.T_IDXS]))


def step(controller, desired=.002, actual=.001, speed=20., rate=0., active=True, pressed=False, limited=False):
  cs = SimpleNamespace(vEgo=speed, aEgo=0., steeringAngleDeg=math.degrees(actual * 160.),
                       steeringRateDeg=rate, steeringPressed=pressed)
  return controller.update(active, cs, ObservationVehicleModel(), SimpleNamespace(angleOffsetDeg=0., roll=0.),
                           limited, desired, None, False, .2)


class TestC3XTorqueController(unittest.TestCase):
  def test_full_trace_has_finite_bounded_final_output_and_consistent_logging(self):
    # The old exact-output fixture included the double PID update, unbounded NN
    # override, and stale I after disengagement. Assert control contracts instead.
    for enabled in (False, True):
      with self.subTest(nnlc=enabled):
        samples = run_trace(build_controller(enabled))
        previous = 0.
        for sample in samples:
          values = sample["values"]
          self.assertTrue(np.isfinite(values).all())
          self.assertLessEqual(abs(values[0]), 1.)
          if sample["active"]:
            allowed = MAX_LATERAL_JERK * .01
            self.assertLessEqual(abs(values[0] - previous) * sample["factor"], allowed + 1e-9)
            self.assertAlmostEqual(values[3], values[4], places=6)  # kp = 1; actual selected error
          else:
            self.assertEqual(values[0], 0.)
          previous = values[0]

  def test_one_pid_update_and_one_history_sample_per_control_cycle(self):
    for enabled in (False, True):
      with self.subTest(nnlc=enabled):
        controller = build_controller(enabled)
        controller.extension.update_model_v2(model_for())
        with patch.object(controller.pid, "update", wraps=controller.pid.update) as update, \
             patch.object(controller.extension.sigmoid_map_tuner, "observe", wraps=controller.extension.sigmoid_map_tuner.observe) as observe:
          output, _, state = step(controller)
        self.assertEqual(update.call_count, 1)
        self.assertAlmostEqual(state.error, update.call_args.args[0], places=6)
        self.assertAlmostEqual(state.i, .2 * .01 * state.error, places=7)
        self.assertEqual(len(controller.extension.roll_deque), int(enabled))
        self.assertEqual(len(controller.extension.lateral_accel_desired_deque), int(enabled))
        self.assertEqual(observe.call_count, int(enabled))
        if enabled:
          self.assertEqual(observe.call_args.args[5], -output)
          self.assertTrue(observe.call_args.args[6])  # exclude slew-limited data from tuning

  def test_mode_changes_reset_integrator_units_and_history(self):
    controller = build_controller()
    for model, neural in ((model_for(), True), (None, False), (model_for(), True)):
      controller.pid.i = .4
      controller.extension.update_model_v2(model)
      # Establish a previous mode before testing the first transition.
      if controller._using_nnlc is None:
        controller._using_nnlc = False
      output, _, state = step(controller, pressed=True)
      self.assertTrue(math.isfinite(output))
      self.assertEqual(state.i, 0.)
      self.assertEqual(len(controller.extension.roll_deque), int(neural))
      if neural:
        self.assertEqual(controller.pid.pos_limit, 1.)
        self.assertEqual(controller.pid.neg_limit, -1.)
      else:
        self.assertEqual(controller.pid.pos_limit, controller.lateral_accel_from_torque(1., controller.torque_params))

  def test_scalar_fallback_conversion_uses_current_observation(self):
    controller = build_controller()
    controller.extension.update_model_v2(model_for())
    step(controller, speed=20.)
    controller.extension.update_model_v2(None)
    step(controller, speed=10.)
    self.assertEqual(controller.extension.sigmoid_map_tuner._current_speed, 10.)
    self.assertEqual(controller.pid.pos_limit, controller.lateral_accel_from_torque(1., controller.torque_params))

  def test_disengagement_clears_pid_history_and_startup_derivative(self):
    for enabled in (False, True):
      with self.subTest(nnlc=enabled):
        controller = build_controller(enabled)
        controller.extension.update_model_v2(model_for())
        step(controller)
        controller.pid.i = .4
        step(controller, active=False)
        self.assertEqual(controller.pid.i, 0.)
        self.assertEqual(len(controller.extension.roll_deque), 0)
        self.assertIsNone(controller._last_actual_lataccel)
        resumed = step(controller)[2]
        fresh = build_controller(enabled)
        fresh.extension.update_model_v2(model_for())
        expected = step(fresh)[2]
        np.testing.assert_allclose([resumed.error, resumed.i, resumed.f, resumed.output],
                                   [expected.error, expected.i, expected.f, expected.output], atol=1e-7)

  def test_integrator_freezes_for_driver_limits_and_low_speed(self):
    for enabled in (False, True):
      for constraints in ({"pressed": True}, {"limited": True}, {"speed": 4.}):
        with self.subTest(nnlc=enabled, constraints=constraints):
          controller = build_controller(enabled)
          controller.extension.update_model_v2(model_for())
          controller.pid.i = .04
          state = step(controller, **constraints)[2]
          self.assertAlmostEqual(state.i, .04, places=7)

  def test_live_calibration_change_preserves_output_slew_bound(self):
    for enabled in (False, True):
      with self.subTest(nnlc=enabled):
        controller = build_controller(enabled)
        controller.extension.update_model_v2(model_for())
        for _ in range(30):
          previous = step(controller)[0]
        controller.update_torque_parameters(1.5, 0., .12)
        output = step(controller)[0]
        self.assertLessEqual(abs(output - previous) * 1.5, MAX_LATERAL_JERK * .01 + 1e-9)

  def test_final_selected_request_uses_shared_jerk_envelope_at_all_speeds(self):
    for enabled in (False, True):
      for speed in (4., 10., 20., 30., 40.):
        with self.subTest(nnlc=enabled, speed=speed):
          controller = build_controller(enabled)
          controller.extension.update_model_v2(model_for(2., .1))
          previous = 0.
          for direction in (1., -1.):
            # Ample error makes the output limiter active, including on the
            # final neural request and when reversing/unwinding its command.
            output, _, _ = step(controller, desired=direction * 2. / speed**2, actual=0., speed=speed)
            self.assertTrue(controller._output_limited)
            calibrated_change = (output - previous) * controller.torque_params.latAccelFactor
            self.assertAlmostEqual(abs(calibrated_change), MAX_LATERAL_JERK * controller.dt, places=9)
            previous = output

  def test_slew_limit_stops_windup_but_permits_integrator_unwinding(self):
    for enabled in (False, True):
      with self.subTest(nnlc=enabled):
        controller = build_controller(enabled)
        controller.extension.update_model_v2(model_for())
        first = step(controller, actual=0.)[2]
        self.assertTrue(controller._output_limited)
        limited = step(controller, actual=0.)[2]
        self.assertEqual(first.i, limited.i)
        # Ramp the observation at finite jerk; an instantaneous angle jump also
        # hits the scalar PID's separate amplitude anti-windup limit.
        for actual in np.linspace(.0002, .004, 20):
          unwinding = step(controller, actual=actual)[2]
        self.assertLess(unwinding.i, limited.i)

  def test_opposite_turn_is_not_classified_as_cutting_inside(self):
    # A right request while still turning left must not have its rightward
    # correction weakened by a guard intended for excessive right curvature.
    for direction in (-1., 1.):
      controller = build_controller(False)
      state = step(controller, desired=direction * .02, actual=-direction * .03)[2]
      self.assertAlmostEqual(state.desiredLateralAccel, direction * 8., places=6)

  def test_invalid_model_arrays_fall_back_and_clear_jerk(self):
    controller = build_controller()
    ext = controller.extension
    for field in ("roll", "pitch", "accel"):
      for invalid in ([], [0.] * 17, [0.] * 32, [math.nan] * 33, [math.inf] * 33):
        with self.subTest(field=field, invalid_length=len(invalid)):
          model = model_for()
          if field == "roll":
            model.orientation.x = invalid
          elif field == "pitch":
            model.orientation.y = invalid
          else:
            model.acceleration.y = invalid
          ext.update_model_v2(model)
          self.assertFalse(ext.model_valid)
          self.assertTrue(math.isfinite(step(controller)[0]))
    ext.update_model_v2(model_for())
    self.assertTrue(ext.model_valid)

  def test_friction_jerk_gate_is_not_history_dependent(self):
    controller = build_controller()
    ext = controller.extension
    cs = SimpleNamespace(vEgo=20., steeringRateDeg=10.)
    ext.update_model_v2(model_for(1., 0.))
    ext.update_calculations(cs, ObservationVehicleModel(), 1.)
    self.assertEqual(ext.lookahead_lateral_jerk, 0.)
    self.assertEqual(ext.lat_accel_friction_factor, 1.)
    ext.update_model_v2(model_for(1., .1))
    ext.update_calculations(cs, ObservationVehicleModel(), 1.)
    self.assertGreater(ext.lookahead_lateral_jerk, 0.)
    self.assertEqual(ext.lat_accel_friction_factor, .7)
    for lag in (math.nan, math.inf, -math.inf, 0.):
      ext.update_lateral_lag(lag)
      self.assertTrue(math.isfinite(ext.desired_lat_jerk_time))
      self.assertGreater(ext.desired_lat_jerk_time, 0.)

  def test_recorded_unwind_feedback_corrects_acceleration_excess(self):
    event = json.loads(UNWIND_FIXTURE.read_text())
    ext = build_controller().extension
    v, sp, me, roll = (event[key] for key in ("speed", "setpoint", "measurement", "roll"))
    history = event["past_future_rolls"]
    # Reproduce the observed sign reversal in the old feedback on the exact
    # captured NN inputs. This is a frozen-input regression, not a vehicle sim.
    old = ext.model.evaluate([v, sp, event["jerk_setpoint"], roll] + [sp] * 7 + history) \
          - ext.model.evaluate([v, me, event["jerk_measurement"], roll] + [me] * 7 + history)
    self.assertAlmostEqual(old, event["p_reconstructed"], places=6)
    self.assertLess(abs(old - event["p_logged"]), 4e-5)
    self.assertGreater(old, 0.)
    for direction in (-1., 1.):
      feedback = ext.acceleration_error(v, direction * sp, direction * me, direction * event["desired_lateral_accel"],
                                        direction * roll, [direction * value for value in history])
      self.assertLess(direction * feedback, 0.)

  def test_nn_feedback_does_not_depend_on_wheel_jerk_or_friction_override(self):
    responses = []
    for rate in (-100., 0., 100.):
      for friction_override in (False, True):
        controller = build_controller()
        controller.extension.update_model_v2(model_for(1., -.1))
        controller.extension.model.friction_override = friction_override
        responses.append(step(controller, desired=.002, actual=.003, rate=rate)[2].error)
    np.testing.assert_allclose(responses, responses[0], atol=1e-7)

  def test_exact_model_and_frozen_map_are_present(self):
    self.assertEqual(hashlib.sha256(MODEL.read_bytes()).hexdigest(), "7084039a01e6c6628b21e6ca85bae3d646df8bd2fa9d4e86ae6664352852b769")
    self.assertEqual(hashlib.sha256(MAP.read_bytes()).hexdigest(), "83f5796acbb70a8a7d1c6f3fecf6709a903e9834f522663f4e2c5cc4b868b45d")

  def test_scalar_pid_and_source_extension_chain(self):
    controller = build_controller()
    for speed in (1., 5., 10., 20., 30.):
      controller.pid.speed = speed
      self.assertEqual(controller.pid.k_p, 1.0)
      self.assertEqual(controller.pid.k_i, 0.2)
    self.assertEqual(controller.feedforward_gain, 0.7)
    self.assertFalse(hasattr(controller.extension, "residual"))
    self.assertFalse(hasattr(controller.extension, "update_jerk_aware_torque_control"))
    self.assertTrue(controller.extension.sigmoid_map_tuner._frozen_from_disk)

  def test_inside_guards_reduce_shaped_acceleration_symmetrically(self):
    values = []
    for direction in (-1., 1.):
      controller = build_controller(False)
      CS = SimpleNamespace(vEgo=10., aEgo=0., steeringAngleDeg=math.degrees(direction * .022 * 160.),
                           steeringRateDeg=0., steeringPressed=False)
      _, _, state = controller.update(True, CS, ObservationVehicleModel(), SimpleNamespace(angleOffsetDeg=0., roll=0.),
                                        False, direction * .012, None, False, .2)
      values.append(state.desiredLateralAccel)
      self.assertLess(abs(state.desiredLateralAccel), 1.2)
    self.assertAlmostEqual(values[0], -values[1])

  def test_existing_device_map_precedes_bundled_fallback(self):
    controller = build_controller(False)
    with tempfile.TemporaryDirectory() as directory:
      local_map = Path(directory) / MAP.name
      local_map.write_text(json.dumps({"slices": [{"speed": 10., "slope": 2., "intercept": 0.1}]}))
      with patch.object(SigmoidMapTuner, "_candidate_storage_dirs", return_value=[directory, str(MAP.parent)]):
        tuner = SigmoidMapTuner(controller, controller.torque_params,
                                 controller.extension.torque_from_lateral_accel_in_torque_space, "HYUNDAI_KONA_EV")
      self.assertEqual(tuner._file_path, str(local_map))
      self.assertEqual(tuner._solution.slices[0].slope, 2.0)
      self.assertTrue(tuner._frozen_from_disk)

  def test_malformed_local_maps_use_packaged_fallback(self):
    invalid = [None, {"slices": []}, {"slices": [{}]}, {"slices": [1]},
               {"slices": [{"speed": 10., "slope": math.nan, "intercept": 0.}]},
               {"slices": [{"speed": 10., "slope": -1., "intercept": 0.}]}]
    for data in invalid:
      with self.subTest(data=data), tempfile.TemporaryDirectory() as directory:
        (Path(directory) / MAP.name).write_text(json.dumps(data))
        controller = build_controller(False)
        with patch.object(SigmoidMapTuner, "_candidate_storage_dirs", return_value=[directory, str(MAP.parent)]):
          tuner = SigmoidMapTuner(controller, controller.torque_params,
                                   controller.extension.torque_from_lateral_accel_in_torque_space, "HYUNDAI_KONA_EV")
        self.assertEqual(tuner._file_path, str(MAP))
        self.assertTrue(tuner._frozen_from_disk)

  def test_failed_local_write_never_writes_packaged_map(self):
    original = MAP.read_bytes()
    with tempfile.TemporaryDirectory() as directory:
      controller = build_controller(False)
      with patch.object(SigmoidMapTuner, "_candidate_storage_dirs", return_value=[directory, str(MAP.parent)]):
        tuner = SigmoidMapTuner(controller, controller.torque_params,
                                 controller.extension.torque_from_lateral_accel_in_torque_space, "HYUNDAI_KONA_EV")
      tuner._frozen_from_disk = False
      with patch("builtins.open", side_effect=OSError("unwritable local map")) as write, \
           patch("openpilot.sunnypilot.selfdrive.controls.lib.nnlc.sigmoid_map_tuner.cloudlog.event"):
        tuner._write_solution()
      write.assert_called_once_with(str(Path(directory) / MAP.name), "w", encoding="utf-8")
    self.assertEqual(MAP.read_bytes(), original)

  def test_torque_override_requires_explicit_enable(self):
    controller = build_controller(False)
    override = controller.extension
    override.params = FakeParams(TorqueParamsOverrideEnabled=True, TorqueParamsOverrideLatAccelFactor=4.,
                                  TorqueParamsOverrideFriction=0.2)
    self.assertFalse(override.update_override_torque_params(controller.torque_params))
    override.enforce_torque_control_toggle = True
    self.assertTrue(override.update_override_torque_params(controller.torque_params))
    self.assertEqual(controller.torque_params.latAccelFactor, 4.)
    self.assertEqual(controller.torque_params.friction, 0.2)

  def test_default_selection_and_explicit_alternatives(self):
    # Execute the production selector without starting messaging/native IPC.
    source = LIB.parent / "controlsd_ext.py"
    controls_class = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "ControlsExt")
    selector = next(node for node in controls_class.body if isinstance(node, ast.FunctionDef) and node.name == "initialize_lateral_control")
    for tuning, enforce, version, selected in (("torque", False, 0., "c3x"), ("pid", False, 0., "original"),
                                               ("torque", True, 0., "v0"), ("torque", True, 1., "original")):
      with self.subTest(tuning=tuning, enforce=enforce, version=version):
        controls = SimpleNamespace()
        controls.CP = SimpleNamespace(lateralTuning=SimpleNamespace(which=lambda tuning=tuning: tuning))
        controls.CP_SP = object()
        controls.params = FakeParams(EnforceTorqueControl=enforce, TorqueControlTune=version)
        original, c3x, v0, interface = object(), object(), object(), object()
        namespace = {"LatControlTorqueC3X": Mock(return_value=c3x), "LatControlTorqueV0": Mock(return_value=v0)}
        exec(compile(ast.Module(body=[selector], type_ignores=[]), str(source), "exec"), namespace)
        result = namespace["initialize_lateral_control"](controls, original, interface, 0.01)
        self.assertIs(result, {"c3x": c3x, "v0": v0, "original": original}[selected])


if __name__ == "__main__":
  unittest.main()

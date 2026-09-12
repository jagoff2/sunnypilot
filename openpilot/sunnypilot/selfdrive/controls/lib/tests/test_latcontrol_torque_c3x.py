"""C3X scenarios, model provenance and explicit controller selection.

The old frozen trace remains historical provenance. Current output is checked
against behavioral invariants instead of reproducing its defective control
updates. Only vehicle observations/conversion and Params storage are faked;
production PID, NNLC model, sigmoid map, and capnp logging execute in the tests.
"""
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
from openpilot.sunnypilot.selfdrive.controls.lib import latcontrol_torque_c3x
from openpilot.sunnypilot.selfdrive.controls.lib import latcontrol_torque_ext_override
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc import nnlc_c3x
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.sigmoid_map_tuner import SigmoidMapTuner

LIB = Path(__file__).parents[1]
MODEL = Path(__file__).parents[4] / "neural_network_data/neural_network_lateral_control/HYUNDAI_KONA_EV.json"
MAP = LIB / "nnlc/sigmoid_maps/HYUNDAI_KONA_EV_nnlc.json"
FIXTURE = Path(__file__).with_name("fixtures") / "c3x_torque_reference.json"


class FakeParams(dict):
  def get_bool(self, key):
    return bool(self.get(key, False))

  def get(self, key, default=None, return_default=False):
    return super().get(key, default)


class ObservationVehicleModel:
  @staticmethod
  def calc_curvature(angle, speed, roll):
    return -angle / 160.0


def build_controller(enabled=True, controller_class=latcontrol_torque_c3x.LatControlTorque, nnlc_module=nnlc_c3x, dt=0.01):
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
    return controller_class(CP, CP_SP, CI, dt)


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
      if frame in (0, 1, frames - 1):
        samples.append({"case": name, "frame": frame, "values": [float(value) for value in (
          output, state.desiredLateralAccel, state.actualLateralAccel, state.error, state.p, state.i, state.f,
          controller._last_output_lataccel, controller._last_error,
        )]})
  return samples


class TestC3XTorqueController(unittest.TestCase):
  def test_captured_scenarios_have_finite_commands_and_consistent_feedback_logs(self):
    # The historical trace preserves the defective double-PID behavior. Keep
    # its scenarios, not numerical equality to those obsolete control outputs.
    for enabled in (False, True):
      with self.subTest(nnlc=enabled):
        samples = run_trace(build_controller(enabled))
        self.assertTrue(np.isfinite([sample["values"] for sample in samples]).all())
        for sample in samples:
          self.assertLessEqual(abs(sample["values"][0]), 1.0)
          self.assertAlmostEqual(sample["values"][3], sample["values"][4])

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

  def test_curvature_overshoot_does_not_rewrite_planned_acceleration(self):
    values = []
    for direction in (-1., 1.):
      controller = build_controller(False)
      CS = SimpleNamespace(vEgo=10., aEgo=0., steeringAngleDeg=math.degrees(direction * .022 * 160.),
                           steeringRateDeg=0., steeringPressed=False)
      _, _, state = controller.update(True, CS, ObservationVehicleModel(), SimpleNamespace(angleOffsetDeg=0., roll=0.),
                                        False, direction * .012, None, False, .2)
      values.append(state.desiredLateralAccel)
      self.assertAlmostEqual(abs(state.desiredLateralAccel), 1.2)
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

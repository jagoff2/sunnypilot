import importlib
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from openpilot.sunnypilot.selfdrive.controls.lib.kona_residual_controller import DEFAULT_POLICY_PATH, KonaResidualController
from openpilot.sunnypilot.selfdrive.controls.lib.kona_residual_policy import ResidualTorquePolicy


class FakeParams(dict):
  def get(self, key, return_default=False):
    return super().get(key)


def make_adapter(fingerprint="HYUNDAI_KONA_EV", dt=0.01, **overrides):
  values = FakeParams(KonaResidualPolicyEnabled=True, KonaResidualPolicyPath="zero", KonaResidualPolicyMaxDelta=0.0)
  values.update(overrides)
  return KonaResidualController(SimpleNamespace(carFingerprint=fingerprint), values, dt)


def make_car_state(**overrides):
  values = {"vEgo": 7.0, "steeringAngleDeg": 10.0, "steeringPressed": False}
  values.update(overrides)
  return SimpleNamespace(**values)


def car_output(torque=0.3):
  return SimpleNamespace(actuatorsOutput=SimpleNamespace(torque=torque))


class TestKonaResidualAdapter(unittest.TestCase):
  def test_default_path_selects_packaged_asset(self):
    with patch.object(ResidualTorquePolicy, "from_path", return_value=ResidualTorquePolicy.zero()) as load:
      make_adapter(KonaResidualPolicyPath="")
    load.assert_called_once_with(str(DEFAULT_POLICY_PATH), 0.0)
    self.assertTrue(DEFAULT_POLICY_PATH.is_file())

  def test_unsupported_platform_or_rate_does_not_load_policy(self):
    for fingerprint, dt in (("HYUNDAI_KONA", 0.01), ("HONDA_CIVIC", 0.01), ("HYUNDAI_KONA_EV", 0.02)):
      with self.subTest(fingerprint=fingerprint, dt=dt), patch.object(ResidualTorquePolicy, "from_path") as load:
        adapter = make_adapter(fingerprint, dt)
        self.assertFalse(adapter.policy.enabled)
        load.assert_not_called()

  def test_explicit_disable_does_not_load_policy(self):
    with patch.object(ResidualTorquePolicy, "from_path") as load:
      adapter = make_adapter(KonaResidualPolicyEnabled=False)
      self.assertFalse(adapter.policy.enabled)
      load.assert_not_called()

  def test_zero_delta_preserves_base(self):
    adapter = make_adapter()
    self.assertEqual(adapter.update(make_car_state(), 0.02, -0.47, None, car_output()), -0.47)
    self.assertAlmostEqual(adapter.state.stock_requested_torque, 0.47)

  def test_known_delta_converts_sign_and_bounds_final_torque(self):
    adapter = make_adapter()
    adapter.policy.predict_delta = Mock(return_value=0.2)
    adapter.policy.rate_limit_delta = Mock(side_effect=lambda delta, state: delta)
    adapter.policy.stabilize_applied_delta = Mock(side_effect=lambda delta, observation, state: delta)
    self.assertAlmostEqual(adapter.update(make_car_state(), 0.02, -0.47, None, car_output()), -0.67)
    self.assertEqual(adapter.update(make_car_state(), 0.02, -0.95, None, car_output()), -1.0)

  def test_missing_or_nonfinite_feedback_resets_and_preserves_base(self):
    adapter = make_adapter()
    for feedback in (None, SimpleNamespace(), car_output(math.nan), car_output(math.inf)):
      with self.subTest(feedback=feedback):
        adapter.state.initialized = True
        self.assertEqual(adapter.update(make_car_state(), 0.02, -0.4, None, feedback), -0.4)
        self.assertFalse(adapter.state.initialized)

  def test_driver_override_resets_history(self):
    adapter = make_adapter()
    adapter.state.initialized = True
    self.assertEqual(adapter.update(make_car_state(steeringPressed=True), 0.02, -0.4, None, car_output()), -0.4)
    self.assertFalse(adapter.state.initialized)

  def test_nonfinite_observation_or_present_yaw_bypasses(self):
    adapter = make_adapter()
    cases = ((make_car_state(vEgo=math.nan), 0.02, None),
             (make_car_state(steeringAngleDeg=math.inf), 0.02, None),
             (make_car_state(), math.nan, None),
             (make_car_state(), 0.02, SimpleNamespace(orientation=SimpleNamespace(yaw=math.nan))))
    for state, curvature, pose in cases:
      with self.subTest(state=state, curvature=curvature, pose=pose):
        adapter.state.initialized = True
        self.assertEqual(adapter.update(state, curvature, -0.4, pose, car_output()), -0.4)
        self.assertFalse(adapter.state.initialized)

  def test_nonfinite_base_returns_zero(self):
    adapter = make_adapter()
    for value in (math.nan, math.inf, -math.inf):
      with self.subTest(value=value):
        adapter.state.initialized = True
        self.assertEqual(adapter.update(make_car_state(), 0.02, value, None, car_output()), 0.0)
        self.assertFalse(adapter.state.initialized)

  def test_prediction_failure_disables_policy_and_preserves_base(self):
    for failure in (math.nan, RuntimeError("invalid inference")):
      with self.subTest(failure=failure):
        adapter = make_adapter()
        adapter.policy.predict_delta = Mock(side_effect=failure) if isinstance(failure, Exception) else Mock(return_value=failure)
        with self.assertLogs("openpilot.sunnypilot.selfdrive.controls.lib.kona_residual_controller", level="WARNING"):
          self.assertEqual(adapter.update(make_car_state(), 0.02, -0.4, None, car_output()), -0.4)
        self.assertFalse(adapter.policy.enabled)
        self.assertFalse(adapter.state.initialized)
        self.assertEqual(adapter.update(make_car_state(), 0.02, -0.4, None, car_output()), -0.4)

  def test_control_loop_does_not_read_parameters_or_load_files(self):
    adapter = make_adapter()
    with patch.object(FakeParams, "get", side_effect=AssertionError("parameter access in update")), \
         patch.object(ResidualTorquePolicy, "from_path", side_effect=AssertionError("model load in update")):
      self.assertEqual(adapter.update(make_car_state(), 0.02, -0.4, None, car_output()), -0.4)


class TestKonaResidualControllerIntegration(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.extension_module = importlib.import_module("openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext")
    cls.controller_modules = [
      importlib.import_module("openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_v0"),
      importlib.import_module("openpilot.selfdrive.controls.lib.latcontrol_torque"),
    ]

  def test_residual_composes_with_final_extension_output(self):
    for delta in (0.0, 0.2):
      with self.subTest(delta=delta):
        extension = self.extension_module.LatControlTorqueExt.__new__(self.extension_module.LatControlTorqueExt)
        extension.residual = make_adapter()
        extension.residual.policy.predict_delta = Mock(return_value=delta)
        extension.residual.policy.rate_limit_delta = Mock(side_effect=lambda value, state: value)
        extension.residual.policy.stabilize_applied_delta = Mock(side_effect=lambda value, observation, state: value)
        extension.update_calculations = Mock()
        extension.update_jerk_aware_torque_control = Mock(side_effect=lambda *args, ext=extension: setattr(ext, "_output_torque", -0.35))
        extension.update_neural_network_feedforward = Mock(side_effect=lambda *args, ext=extension: setattr(ext, "_output_torque", -0.47))
        pid_log = SimpleNamespace()
        result_log, torque = extension.update(
          make_car_state(), None, None, None, 0.0, pid_log, 0.0, 0.0, None, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.02, 0.0, False, -0.1, car_output=car_output(),
        )
        self.assertIs(result_log, pid_log)
        self.assertAlmostEqual(torque, -0.47 - delta)
        self.assertAlmostEqual(extension.residual.state.stock_requested_torque, 0.47)

  @staticmethod
  def make_controller(module):
    torque_params = SimpleNamespace(steeringAngleDeadzoneDeg=0.0, latAccelOffset=0.0, friction=0.1, latAccelFactor=2.5)
    cp = SimpleNamespace(steerLimitTimer=0.8, lateralTuning=SimpleNamespace(torque=SimpleNamespace(as_builder=lambda: torque_params)))
    ci = SimpleNamespace(torque_from_lateral_accel=lambda: lambda value, params: value,
                         lateral_accel_from_torque=lambda: lambda value, params: value)
    extension = Mock()
    extension.update_override_torque_params.return_value = False
    extension.residual = make_adapter()
    with patch.object(module, "LatControlTorqueExt", return_value=extension):
      return module.LatControlTorque(cp, None, ci, 0.01)

  def test_both_controller_versions_clear_residual_on_reset(self):
    for module in self.controller_modules:
      with self.subTest(version=module.VERSION):
        controller = self.make_controller(module)
        controller.extension.residual.state.initialized = True
        controller.sat_time = 0.5
        controller.reset()
        self.assertFalse(controller.extension.residual.state.initialized)
        self.assertEqual(controller.sat_time, 0.0)

  def test_both_controller_versions_clear_residual_while_inactive(self):
    for module in self.controller_modules:
      with self.subTest(version=module.VERSION):
        controller = self.make_controller(module)
        controller.extension.residual.state.initialized = True
        vm = SimpleNamespace(calc_curvature=lambda *args: 0.0)
        params = SimpleNamespace(angleOffsetDeg=0.0, roll=0.0)
        torque, _, pid_log = controller.update(False, make_car_state(), vm, params, False, 0.02, None, False, 0.2,
                                               car_output=car_output())
        self.assertEqual(torque, 0.0)
        self.assertFalse(pid_log.active)
        self.assertFalse(controller.extension.residual.state.initialized)
        controller.extension.update.assert_not_called()


if __name__ == "__main__":
  unittest.main()

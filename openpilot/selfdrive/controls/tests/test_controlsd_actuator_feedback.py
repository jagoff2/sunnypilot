"""Execute the real publisher with capnp messages and isolated service boundaries.

Loading just the method avoids importing model accelerator and Params runtimes;
the applied-output checks and published controls/car-control messages are real.
"""
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

from opendbc.car.structs import car
from openpilot.cereal import log
from openpilot.common.constants import CV
from openpilot.selfdrive.controls.lib.latcontrol_angle import STEER_ANGLE_SATURATION_THRESHOLD


def new_message(service):
  message = log.Event.new_message()
  message.init(service)
  return message


class TestActuatorFeedback(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    source = Path(__file__).parents[1] / "controlsd.py"
    controls = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "Controls")
    publish = next(node for node in controls.body if isinstance(node, ast.FunctionDef) and node.name == "publish")
    scope = {"CV": CV, "car": car, "State": log.SelfdriveState.OpenpilotState,
             "STEER_ANGLE_SATURATION_THRESHOLD": STEER_ANGLE_SATURATION_THRESHOLD,
             "messaging": SimpleNamespace(new_message=new_message)}
    exec(compile(ast.Module(body=[publish], type_ignores=[]), str(source), "exec"), scope)
    cls.publish = staticmethod(scope["publish"])

  def run_publish(self, lat_active=True, selfdrive_active=False, output_valid=True, matched=False, angle=False):
    cp = car.CarParams.new_message()
    cp.lateralTuning.init("torque")
    cp.steerControlType = "angle" if angle else "torque"
    cs = car.CarState.new_message()
    cs.canValid = True
    output = car.CarOutput.new_message()
    command = car.CarControl.new_message()
    command.latActive = lat_active
    command.actuators.torque = 0.5
    command.actuators.steeringAngleDeg = 10.
    if matched:
      output.actuatorsOutput.torque = command.actuators.torque
      output.actuatorsOutput.steeringAngleDeg = command.actuators.steeringAngleDeg
    sm = ServiceMessages({
      "carState": cs, "carOutput": output, "selfdriveState": log.SelfdriveState.new_message(),
      "driverMonitoringState": log.DriverMonitoringState.new_message(),
      "longitudinalPlan": log.LongitudinalPlan.new_message(),
    }, output_valid)
    sm["selfdriveState"].active = selfdrive_active
    sm["selfdriveState"].enabled = selfdrive_active
    published = {}
    controller = SimpleNamespace(sm=sm, CP=cp, CP_SP=SimpleNamespace(pcmCruiseSpeed=True), calibrated_pose=None,
                                 curvature=0., desired_curvature=0., steer_limited_by_safety=True,
                                 LoC=SimpleNamespace(pid=SimpleNamespace(p=0., i=0., f=0.), long_control_state="off"),
                                 pm=SimpleNamespace(send=lambda service, message: published.update({service: message})))
    state_type = log.ControlsState.LateralAngleState if angle else log.ControlsState.LateralTorqueState
    self.publish(controller, command, state_type.new_message())
    self.assertIn("controlsState", published)
    self.assertEqual(published["carControl"].carControl.latActive, lat_active)
    return controller.steer_limited_by_safety

  def test_lateral_only_mads_tracks_applied_torque(self):
    self.assertTrue(self.run_publish())
    self.assertFalse(self.run_publish(matched=True))
    self.assertTrue(self.run_publish(selfdrive_active=True))

  def test_inactive_or_invalid_output_clears_stale_feedback(self):
    self.assertFalse(self.run_publish(lat_active=False))
    self.assertFalse(self.run_publish(output_valid=False))

  def test_angle_control_uses_angle_mismatch_in_mads(self):
    self.assertTrue(self.run_publish(angle=True))
    self.assertFalse(self.run_publish(angle=True, matched=True))

  def test_torque_diagnostics_serialize_with_unambiguous_units(self):
    message = new_message("controlsState")
    state = message.controlsState.lateralControlState.init("torqueState")
    state.error, state.p, state.i, state.f = -0.1, -0.1, 0.03, 0.2
    state.rawDesiredLateralAccel, state.feedbackLateralAccel = 1.2, 0.9
    state.actualLateralJerk, state.yawLateralAccel = -0.3, 1.1
    state.steeringRateDeg = -8.
    state.preLimitOutput, state.output, state.appliedOutput = -0.13, -0.10, -0.09
    state.outputLimited, state.appliedOutputValid = True, True
    state.feedbackDelay = 0.384424
    with log.Event.from_bytes(message.to_bytes()) as restored:
      output = restored.controlsState.lateralControlState.torqueState
      self.assertAlmostEqual(output.error, output.p)
      self.assertAlmostEqual(output.preLimitOutput, -(output.p + output.i + output.f))
      self.assertTrue(output.outputLimited and output.appliedOutputValid)
      self.assertLess(output.steeringRateDeg, 0.)


class ServiceMessages(dict):
  def __init__(self, services, output_valid):
    super().__init__(services)
    self.output_valid = output_valid
    self.valid = {"driverAssistance": False}
    self.logMonoTime = {"longitudinalPlan": 0, "modelV2": 0}

  def all_checks(self, services):
    return self.output_valid if services == ["carOutput"] else True


if __name__ == "__main__":
  unittest.main()

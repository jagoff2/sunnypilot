"""Native MPC/SCC publication regression for rejected model frames."""
import math

from openpilot.cereal import log, messaging
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.ldw import LaneDepartureWarning
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.plannerd import update_plans
from openpilot.selfdrive.modeld.lane_centering_integration import LaneCenteringModelAdapter
from openpilot.selfdrive.modeld.planning_model import PlanningModelCache
from openpilot.sunnypilot.selfdrive.controls.lib.dec.tests.test_dec_planner_gate import build_planner, build_sm


class TestPlannerdModelCache(OpenpilotTestCase):
  def setUp(self):
    super().setUp()
    self.planner = build_planner(False, 'acc')
    self.planner.CP.openpilotLongitudinalControl = True
    sm = self.sm = messaging.SubMaster.__new__(messaging.SubMaster)
    # Construct native messages and use actual health methods without sockets.
    sm.data = {name: value.as_reader() if hasattr(value, 'as_reader') else value for name, value in build_sm(False).items()}
    sm.data['selfdriveStateSP'] = messaging.new_message('selfdriveStateSP').selfdriveStateSP.as_reader()
    sm.services = list(sm.data)
    for field in ('seen', 'alive', 'valid', 'freq_ok', 'updated'):
      setattr(sm, field, dict.fromkeys(sm.services, True))
    sm.logMonoTime = dict.fromkeys(sm.services, 1_020_000_000)
    sm.recv_frame = dict.fromkeys(sm.services, 1)
    sm.frame = 200
    sm.ignore_alive, sm.ignore_valid, sm.ignore_average_freq = [], [], []
    self.replace_model(1_000_000_000)
    controls = sm['controlsState'].as_builder()
    controls.longControlState = LongCtrlState.pid
    sm.data['controlsState'] = controls.as_reader()
    self.cache, self.ldw = PlanningModelCache(), LaneDepartureWarning()
    self.sent = {}

  def replace_model(self, timestamp):
    model = self.sm['modelV2'].as_builder()
    for field in ('position', 'velocity', 'acceleration', 'orientation', 'orientationRate'):
      for axis in 'xyz':
        vector = getattr(getattr(model, field), axis)
        if len(vector) != 33:
          setattr(getattr(model, field), axis, [0.] * 33)
    model.timestampEof = timestamp
    self.sm.data['modelV2'] = model.as_reader()
    self.sm.updated['modelV2'] = self.sm.valid['modelV2'] = True

  def send(self, service, message):
    # Serialize every actual publication but never create a PubMaster.
    with log.Event.from_bytes(message.to_bytes()) as restored:
      self.sent[service] = restored.to_dict()

  def tick(self, now):
    return update_plans(self.sm, self, self.planner, self.ldw, self.cache, now)

  def reject(self):
    adapter = LaneCenteringModelAdapter()
    message = messaging.new_message('modelV2')
    adapter.fill_invalid_model(message, log.ModelDataV2.Action(), adapter.controller._status(), 2, 1_050_000_000)
    self.sm.data['modelV2'] = message.modelV2.as_reader()
    self.sm.valid['modelV2'] = False
    self.sm.logMonoTime['modelV2'] = 1_100_000_000

  def test_rejected_frame_preserves_complete_planning_and_truthful_derived_health(self):
    self.assertTrue(self.tick(1_150_000_000))
    self.reject()
    for now in (1_200_000_000, 1_250_000_000, 1_300_000_000):
      self.assertTrue(self.tick(now))
      self.assertTrue(self.sent['longitudinalPlan']['valid'])
      self.assertTrue(self.sent['driverAssistance']['valid'])
      plan = self.sent['longitudinalPlan']['longitudinalPlan']
      self.assertTrue(math.isfinite(plan['aTarget']))
      self.assertEqual(plan['modelMonoTime'], 1_020_000_000)
    self.sm.valid['radarState'] = False
    self.assertTrue(self.tick(1_300_000_000))
    self.assertFalse(self.sent['longitudinalPlan']['valid'])
    self.assertFalse(self.sent['driverAssistance']['valid'])
    self.sent.clear()
    self.assertFalse(self.tick(1_300_000_001))
    self.assertEqual(self.sent, {})

  def test_missing_startup_model_does_not_enter_native_consumers(self):
    self.reject()
    self.assertFalse(self.tick(1_200_000_000))
    self.assertEqual(self.sent, {})

  def test_recovery_resets_native_mpc_from_current_disengaged_vehicle_state(self):
    self.assertTrue(self.tick(1_150_000_000))
    self.reject()
    self.assertFalse(self.tick(1_300_000_001))
    self.replace_model(1_350_000_000)
    cs = self.sm['carState'].as_builder()
    cs.vEgo, cs.aEgo, cs.standstill = 0., -1., True
    self.sm.data['carState'] = cs.as_reader()
    controls = self.sm['controlsState'].as_builder()
    controls.longControlState = LongCtrlState.off
    self.sm.data['controlsState'] = controls.as_reader()
    self.assertTrue(self.tick(1_400_000_000))
    self.assertLess(abs(self.planner.v_desired_filter.x), .1)
    self.assertLess(self.planner.output_a_target, 0.)
    self.assertTrue(self.sent['longitudinalPlan']['valid'])

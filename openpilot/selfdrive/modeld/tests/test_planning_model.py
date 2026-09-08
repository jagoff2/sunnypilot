import ast
import copy
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from openpilot.cereal import log
from openpilot.selfdrive.modeld.lane_centering_integration import LaneCenteringModelAdapter
from openpilot.selfdrive.modeld.lane_centering_safety import MAX_MODEL_AGE_NS
from openpilot.selfdrive.modeld.planning_model import PlannerCadence, PlanningModelCache
from openpilot.selfdrive.modeld.tests.test_lane_centering_integration import ROOT, load_function


SERVICES = ('modelV2', 'carState', 'carControl', 'controlsState', 'radarState', 'selfdriveStateSP')
VECTORS = ('position', 'velocity', 'acceleration', 'orientation', 'orientationRate')


class Inputs:
  def __init__(self, model):
    self.services = SERVICES
    self.data = dict.fromkeys(SERVICES)
    self.data.update(modelV2=model, carState=SimpleNamespace(vEgo=15., standstill=False, vCruise=70., gasPressed=False),
                     carControl=SimpleNamespace(enabled=True), controlsState=SimpleNamespace(curvature=.001),
                     radarState=SimpleNamespace(leadOne=SimpleNamespace(present=False, dRel=10.)),
                     selfdriveStateSP=SimpleNamespace(buttonsReleaseToggle=False))
    self.seen = dict.fromkeys(SERVICES, True)
    self.alive = dict(self.seen)
    self.valid = dict(self.seen)
    self.freq_ok = dict(self.seen)
    self.updated = dict(self.seen)
    self.logMonoTime = dict.fromkeys(SERVICES, 1_020_000_000)
    self.ignore_alive = []
    self.ignore_valid = []
    self.ignore_average_freq = []
    self.frame = 1

  def __getitem__(self, service):
    return self.data[service]


# Use production health methods without opening native IPC sockets.
for _name in ('_check_avg_freq', 'all_alive', 'all_valid', 'all_freq_ok', 'all_checks'):
  setattr(Inputs, _name, load_function('openpilot/cereal/messaging/__init__.py', _name,
                                     {'SERVICE_LIST': {s: SimpleNamespace(frequency=20.) for s in SERVICES}}, 'SubMaster'))


def model_frame(timestamp=1_000_000_000):
  model = log.ModelDataV2.new_message(frameId=1, timestampEof=timestamp)
  for field in VECTORS:
    for axis in 'xyz':
      setattr(getattr(model, field), axis, [15. if field == 'velocity' and axis == 'x' else 0.] * 33)
  model.position.x = np.linspace(0., 120., 33).tolist()
  model.orientationRate.z = [.015] * 33
  model.action.desiredCurvature = .001
  return model.as_reader()


def new_message(service):
  msg = log.Event.new_message(logMonoTime=1_250_000_000)
  msg.init(service)
  return msg


class TestPlanningModelCache(unittest.TestCase):
  def test_bad_packets_use_original_complete_model_and_never_refresh_its_age(self):
    for fault in ('invalid', 'dead', 'unseen', 'empty', 'nonfinite', 'duplicate', 'reordered', 'future', 'action'):
      with self.subTest(fault=fault):
        good = model_frame()
        sm, cache = Inputs(good), PlanningModelCache()
        first = cache.update(sm, 1_150_000_000)
        self.assertTrue(first.updated['modelV2'])
        rejected = model_frame({'duplicate': 1_000_000_000, 'reordered': 950_000_000,
                                'future': 2_000_000_000}.get(fault, 1_050_000_000)).as_builder()
        if fault == 'invalid':
          sm.valid['modelV2'] = False
        elif fault == 'dead':
          sm.alive['modelV2'] = False
        elif fault == 'unseen':
          sm.seen['modelV2'] = False
        elif fault == 'empty':
          rejected.velocity.x = []
        elif fault == 'nonfinite':
          rejected.orientationRate.z = [math.nan] * 33
        elif fault == 'action':
          rejected.action.desiredAcceleration = math.nan
        sm.data['modelV2'] = rejected.as_reader()
        sm.logMonoTime['modelV2'] = 1_100_000_000
        sm.freq_ok['modelV2'] = False
        for age in (200_000_000, MAX_MODEL_AGE_NS):
          view = cache.update(sm, good.timestampEof + age)
          self.assertIs(view['modelV2'], good)
          self.assertTrue(view.all_checks())
          self.assertFalse(view.updated['modelV2'])
          self.assertEqual(view.logMonoTime['modelV2'], 1_020_000_000)
          self.assertEqual(sm['modelV2'].timestampEof, rejected.timestampEof)
          self.assertEqual(sm.logMonoTime['modelV2'], 1_100_000_000)
          self.assertFalse(sm.freq_ok['modelV2'])
        self.assertIsNone(cache.update(sm, good.timestampEof + MAX_MODEL_AGE_NS + 1))

  def test_startup_and_expiry_skip_consumers_then_recovery_uses_current_vehicle_state(self):
    sm, cache = Inputs(model_frame()), PlanningModelCache()
    sm.seen['modelV2'] = False
    self.assertIsNone(cache.update(sm, 1_150_000_000))
    sm.seen['modelV2'] = True
    cache.update(sm, 1_150_000_000)
    sm.updated['modelV2'] = False
    self.assertIsNone(cache.update(sm, 1_300_000_001))
    current = model_frame(1_350_000_000)
    sm.data['modelV2'] = current
    sm.updated['modelV2'] = True
    sm.data['carState'] = SimpleNamespace(vEgo=0., aEgo=-1., standstill=True)
    sm.data['controlsState'] = SimpleNamespace(longControlState='off')
    view = cache.update(sm, 1_400_000_000)
    self.assertIs(view['modelV2'], current)
    for service in SERVICES[1:]:
      self.assertIs(view[service], sm[service])
    self.assertTrue(view.updated['modelV2'])

  def test_only_selected_model_health_is_overridden_and_source_maps_are_unchanged(self):
    for health in ('valid', 'alive', 'freq_ok'):
      for service in SERVICES[1:]:
        with self.subTest(health=health, service=service):
          sm = Inputs(model_frame())
          sm.freq_ok['modelV2'] = False
          getattr(sm, health)[service] = False
          before = {name: copy.copy(getattr(sm, name)) for name in ('data', 'valid', 'alive', 'freq_ok', 'logMonoTime', 'updated')}
          view = PlanningModelCache().update(sm, 1_150_000_000)
          self.assertTrue(view.all_checks(['modelV2']))
          self.assertFalse(view.all_checks())
          for name, original in before.items():
            self.assertEqual(getattr(sm, name), original)
            self.assertIsNot(getattr(sm, name), getattr(view, name))

  def test_all_consumed_vectors_require_finite_complete_data(self):
    for field in VECTORS:
      for axis in 'xyz':
        for bad_values in ([], [0.] * 32, [0.] * 34, [math.nan] * 33, [math.inf] * 33):
          with self.subTest(field=field, axis=axis, length=len(bad_values)):
            bad = model_frame().as_builder()
            setattr(getattr(bad, field), axis, bad_values)
            self.assertIsNone(PlanningModelCache().update(Inputs(bad.as_reader()), 1_150_000_000))
    # Turning beyond ninety degrees can legitimately reverse longitudinal x.
    turn = model_frame().as_builder()
    turn.position.x = [float(i if i < 16 else 32-i) for i in range(33)]
    self.assertIsNotNone(PlanningModelCache().update(Inputs(turn.as_reader()), 1_150_000_000))

  def test_actual_rejection_packet_never_reaches_scc_and_derived_publications_remain_truthful(self):
    messaging = SimpleNamespace(new_message=new_message, SubMaster=Inputs)
    update = load_function('openpilot/selfdrive/controls/plannerd.py', 'update_plans', {'messaging': messaging})
    scc_calculate = load_function('openpilot/sunnypilot/selfdrive/controls/lib/smart_cruise_control/vision_controller.py',
                                  '_update_calculations', {'np': np, '_A_LAT_REG_MAX': 2., 'messaging': messaging},
                                  'SmartCruiseControlVision')
    publish = load_function('openpilot/selfdrive/controls/lib/longitudinal_planner.py', 'publish', {'messaging': messaging},
                            'LongitudinalPlanner')
    rejected = new_message('modelV2')
    adapter = LaneCenteringModelAdapter()
    adapter.fill_invalid_model(rejected, log.ModelDataV2.Action(), adapter.controller._status(), 2, 1_050_000_000)
    with log.Event.from_bytes(rejected.to_bytes()) as packet:
      self.assertFalse(packet.valid)
      self.assertEqual(len(packet.modelV2.velocity.x), 0)
      for enabled in (False, True):
        with self.subTest(scc_feature_enabled=enabled):
          sm, cache = Inputs(model_frame()), PlanningModelCache()
          sent, consumed = {}, []
          pm = SimpleNamespace(send=lambda service, msg, sent=sent: sent.__setitem__(service, msg))
          scc = SimpleNamespace(long_enabled=True, enabled=enabled, v_ego=15.)
          planner = SimpleNamespace(sla=SimpleNamespace(update_buttons=Mock()),
                                    mpc=SimpleNamespace(solve_time=0., source='cruise'),
                                    v_desired_trajectory=np.zeros(17), a_desired_trajectory=np.zeros(17), j_desired_trajectory=np.zeros(17),
                                    fcw=False, output_a_target=0., output_should_stop=False, allow_throttle=True,
                                    publish_longitudinal_plan_sp=Mock())
          def calculate(view, scc=scc, consumed=consumed):
            scc_calculate(scc, view)
            consumed.append(view)
          planner.update = calculate
          planner.publish = lambda view, pm, planner=planner: publish(planner, view, pm)
          ldw = SimpleNamespace(update=Mock(), left=False, right=False)
          self.assertTrue(update(sm, pm, planner, ldw, cache, 1_150_000_000))
          sm.data['modelV2'] = packet.modelV2
          sm.valid['modelV2'] = packet.valid
          self.assertTrue(update(sm, pm, planner, ldw, cache, 1_200_000_000))
          self.assertEqual(len(consumed[-1]['modelV2'].velocity.x), 33)
          self.assertTrue(sent['longitudinalPlan'].valid)
          self.assertTrue(sent['driverAssistance'].valid)
          self.assertEqual(sent['longitudinalPlan'].longitudinalPlan.modelMonoTime, 1_020_000_000)
          sm.valid['radarState'] = False
          self.assertTrue(update(sm, pm, planner, ldw, cache, 1_250_000_000))
          self.assertFalse(sent['longitudinalPlan'].valid)
          self.assertFalse(sent['driverAssistance'].valid)
          self.assertFalse(update(sm, pm, planner, ldw, cache, 1_300_000_001))
          self.assertEqual(len(consumed), 3)
          self.assertFalse(update(sm, pm, planner, ldw, PlanningModelCache(), 1_200_000_000))
          self.assertEqual(len(consumed), 3)

  def test_cached_frames_do_not_increment_perception_confirmation(self):
    messaging = SimpleNamespace(SubMaster=Inputs)
    calculate = load_function('openpilot/sunnypilot/selfdrive/controls/lib/dec/dec.py', '_update_calculations',
                               {'messaging': messaging, 'WMACConstants': SimpleNamespace(LEAD_PROB=.5)},
                               'DynamicExperimentalController')
    dec = SimpleNamespace(_standstill_count=0, _lead_filter=Mock(), _mpc_fcw_filter=Mock(), _has_slow_down=True,
                          _calculate_slow_down=Mock(), _mpc_fcw_crash_cnt=0)
    dec._lead_filter.get_value.return_value = 0.
    dec._mpc_fcw_filter.get_value.return_value = 0.
    sm = Inputs(model_frame())
    calculate(dec, sm)
    sm.updated['modelV2'] = False
    for _ in range(5):
      calculate(dec, sm)
    dec._calculate_slow_down.assert_called_once()
    self.assertEqual(dec._lead_filter.add_data.call_count, 6)
    trigger = load_function('openpilot/sunnypilot/selfdrive/controls/lib/e2e_alerts_helper.py', 'update_alert_trigger',
                             {'messaging': messaging, 'DT_MDL': .05, 'GREEN_LIGHT_X_THRESHOLD': 30,
                              'LEAD_DEPART_DIST_THRESHOLD': 1., 'TRIGGER_TIMER_THRESHOLD': .3,
                              'E2EStates': SimpleNamespace(INACTIVE=0, ARMED=1, CONSUMED=2)}, 'E2EAlertsHelper')
    helper = SimpleNamespace(frame=100, last_moving_frame=0, green_light_state=1, green_light_trigger_timer=5,
                             lead_depart_state=0, last_allowed=False, lead_depart_confirmed_lead=False)
    for _ in range(5):
      self.assertFalse(trigger(helper, sm)[0])
    self.assertEqual(helper.green_light_trigger_timer, 5)
    sm.updated['modelV2'] = True
    trigger(helper, sm)
    self.assertEqual(helper.green_light_trigger_timer, 6)

  def test_cadence_limits_cached_ticks_and_does_not_burst_after_a_stall(self):
    clock = [0.]
    sleeps = []
    def sleep(duration):
      sleeps.append(duration)
      clock[0] += duration
    cadence = PlannerCadence(.05, clock=lambda: clock[0], sleep=sleep)
    starts = []
    for _ in range(20):
      cadence.wait()
      starts.append(clock[0])
      clock[0] += .01
    np.testing.assert_allclose(np.diff(starts), .05)
    clock[0] += .5
    cadence.wait()
    after_stall = clock[0]
    cadence.wait()
    self.assertAlmostEqual(clock[0] - after_stall, .05)
    self.assertTrue(all(duration > 0. for duration in sleeps))
    tree = ast.parse((ROOT / 'openpilot/selfdrive/controls/plannerd.py').read_text())
    loop = next(node for node in ast.walk(tree) if isinstance(node, ast.While))
    self.assertEqual(ast.unparse(loop.body[0]), 'cadence.wait()')
    self.assertEqual(ast.unparse(loop.body[1]), 'sm.update(0)')


if __name__ == '__main__':
  unittest.main()

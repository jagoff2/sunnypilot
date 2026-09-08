import ast
import math
import unittest
from numbers import Number
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from openpilot.cereal import custom, log
from opendbc.car.structs import car
from openpilot.selfdrive.modeld.lane_centering_safety import (
  LANE_CENTERING_STATUS_VERSION, LaneCenteringSafetyLatch, MAX_MODEL_AGE_NS, PlannerSeverity, model_clock_ns,
)


ROOT = Path(__file__).resolve().parents[4]


def software_namespace(relative, namespace):
  # Exercise the production state machines without native Params/camera/msgq.
  tree = ast.parse((ROOT / relative).read_text())
  tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
  exec(compile(tree, relative, 'exec'), namespace)
  return namespace


def model_frame(timestamp=1_000_000_000, frame_id=1, containment='contained', reason=''):
  model = log.ModelDataV2.new_message(frameId=frame_id, timestampEof=timestamp)
  status = model.laneCentering
  status.version = LANE_CENTERING_STATUS_VERSION
  status.valid = True
  status.frameId = frame_id
  status.timestampEof = timestamp
  status.containment = containment
  status.reason = reason
  status.checkedDistance = 30.0
  return model


class TestLaneCenteringSafety(unittest.TestCase):
  def check(self, guard, model, requested=True, updated=True, now=None, healthy=True):
    return guard.update(model, healthy, updated, model.timestampEof + 150_000_000 if now is None else now, requested)

  def test_camera_age_uses_boottime_instead_of_python_message_monotonic_epoch(self):
    # Simulate a device that has suspended: these clocks must not be mixed.
    with patch('openpilot.selfdrive.modeld.lane_centering_safety.time.CLOCK_BOOTTIME', 7, create=True), \
         patch('openpilot.selfdrive.modeld.lane_centering_safety.time.clock_gettime_ns', return_value=5_000_000_000, create=True) as boot, \
         patch('openpilot.selfdrive.modeld.lane_centering_safety.time.monotonic_ns', return_value=1_000_000_000):
      self.assertEqual(model_clock_ns(), 5_000_000_000)
      boot.assert_called_once_with(7)

  def test_healthy_repeated_control_ticks_do_not_require_new_model_frame(self):
    guard, model = LaneCenteringSafetyLatch(), model_frame()
    self.assertFalse(self.check(guard, model))
    for _ in range(4):
      self.assertFalse(self.check(guard, model, updated=False))

  def test_auxiliary_status_cannot_disable_or_warn_about_a_usable_native_command(self):
    for fault, reason in (('legacy', 'missing_status'), ('version', 'missing_status'), ('invalid', 'invalid_status'),
                          ('frame', 'status_frame_mismatch'), ('timestamp', 'status_frame_mismatch')):
      with self.subTest(fault=fault):
        guard, model = LaneCenteringSafetyLatch(), model_frame()
        model.laneCentering.collisionRisk = True
        if fault == 'legacy':
          model = log.ModelDataV2.new_message(frameId=1, timestampEof=1_000_000_000)
        elif fault == 'version':
          model.laneCentering.version = 1
        elif fault == 'invalid':
          model.laneCentering.valid = False
        elif fault == 'frame':
          model.laneCentering.frameId += 1
        elif fault == 'timestamp':
          model.laneCentering.timestampEof += 1
        self.assertFalse(self.check(guard, model))
        self.assertEqual(guard.diagnostic_reason, reason)
        self.assertEqual(guard.severity, PlannerSeverity.NONE)
        self.assertIsNone(guard.latched_reason)

  def test_uncertifiable_geometry_is_silent_and_does_not_imply_collision(self):
    for reason in ('corridor_invalid', 'invalid_plan', 'boundary_invalid', 'corridor_lost', 'no_contained_path', 'correction_limit'):
      for containment in ('blocked', 'unavailable', 'unexpected', 'bypassed', 'contained', 'recovering'):
        with self.subTest(reason=reason, containment=containment):
          guard, model = LaneCenteringSafetyLatch(), model_frame(containment=containment, reason=reason)
          model.laneCentering.safetyBlocked = True
          model.laneCentering.minClearance = math.nan
          model.laneCentering.checkedDistance = 0.0
          self.assertFalse(self.check(guard, model))
          self.assertEqual(guard.severity, PlannerSeverity.NONE)
          self.assertIsNone(guard.latched_reason)

  def test_nonfinite_action_without_any_usable_command_is_fatal(self):
    for field in ('desiredCurvature', 'desiredAcceleration'):
      with self.subTest(field=field):
        guard, model = LaneCenteringSafetyLatch(), model_frame()
        setattr(model.action, field, math.nan)
        self.assertTrue(self.check(guard, model, requested=False))
        self.assertEqual(guard.reason, 'nonfinite_action')
        self.assertIsNone(guard.selected_model)

  def test_source_age_and_unhealthy_stream_block_before_engagement(self):
    for age in (MAX_MODEL_AGE_NS + 1, -50_000_001):
      with self.subTest(age=age):
        guard, model = LaneCenteringSafetyLatch(), model_frame()
        self.assertTrue(self.check(guard, model, requested=False, now=model.timestampEof + age))
        self.assertEqual(guard.reason, 'stale_model')
    guard = LaneCenteringSafetyLatch()
    self.assertTrue(self.check(guard, model_frame(), healthy=False))
    self.assertEqual(guard.reason, 'model_unhealthy')

  def test_transient_invalid_packets_use_whole_last_good_model_without_refreshing_its_age(self):
    for fault in ('duplicate', 'reordered', 'curvature', 'acceleration', 'invalid_event', 'future'):
      with self.subTest(fault=fault):
        guard, good = LaneCenteringSafetyLatch(), model_frame()
        good.action.desiredCurvature = 0.003
        good.orientationRate.z = [0.04] * 33
        self.assertFalse(self.check(guard, good))
        bad = model_frame({'duplicate': 1_000_000_000, 'reordered': 950_000_000, 'future': 2_000_000_000}.get(fault, 1_050_000_000), 2)
        bad.action.desiredCurvature = math.nan if fault == 'curvature' else -0.02
        if fault == 'acceleration':
          bad.action.desiredAcceleration = math.nan
        for age in (200_000_000, MAX_MODEL_AGE_NS):
          self.assertFalse(self.check(guard, bad, healthy=fault != 'invalid_event', now=good.timestampEof + age))
          self.assertIs(guard.selected_model, good)
          self.assertEqual(guard.last_timestamp_eof, good.timestampEof)
          self.assertIsNone(guard.latched_reason)
        self.assertTrue(self.check(guard, bad, healthy=fault != 'invalid_event', now=good.timestampEof + MAX_MODEL_AGE_NS + 1))
        self.assertEqual(guard.severity, PlannerSeverity.FATAL)
        self.assertIsNone(guard.selected_model)

  def test_valid_packet_after_transient_failure_recovers_without_reengagement(self):
    guard = LaneCenteringSafetyLatch()
    self.assertFalse(self.check(guard, model_frame()))
    bad = model_frame(1_050_000_000, 2)
    bad.action.desiredCurvature = math.nan
    self.assertFalse(self.check(guard, bad))
    good = model_frame(1_100_000_000, 3)
    self.assertFalse(self.check(guard, good))
    self.assertIs(guard.selected_model, good)
    self.assertIsNone(guard.latched_reason)

  def test_collision_warning_never_latches_and_clears_on_next_usable_model(self):
    guard, risky = LaneCenteringSafetyLatch(), model_frame(containment='blocked')
    risky.laneCentering.collisionRisk = True
    self.assertFalse(self.check(guard, risky))
    self.assertEqual(guard.severity, PlannerSeverity.COLLISION_WARNING)
    self.assertTrue(guard.collision_risk)
    self.assertIsNone(guard.latched_reason)
    self.assertFalse(self.check(guard, model_frame(1_050_000_000, 2)))
    self.assertEqual(guard.severity, PlannerSeverity.NONE)

  def test_controlsd_uses_cached_model_for_both_curvature_and_neural_preview_until_timeout(self):
    tree = ast.parse((ROOT / 'openpilot/selfdrive/controls/controlsd.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Controls')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'state_control')
    def noop(*args, **kwargs):
      pass

    for mads_only in (False, True):
      with self.subTest(mads_only=mads_only):
        clock = [1_150_000_000]
        namespace = {'math': math, 'Number': Number, 'car': car, 'model_clock_ns': lambda clock=clock: clock[0],
                     'LaneChangeState': log.LaneChangeState, 'LaneChangeDirection': log.LaneChangeDirection,
                     'CV': SimpleNamespace(KPH_TO_MS=1 / 3.6), 'LAT_SMOOTH_SECONDS': 0.1,
                     'ACTUATOR_FIELDS': tuple(car.CarControl.Actuators.schema.fields),
                     'clip_curvature': lambda speed, previous, desired, roll: (desired, False)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), 'controlsd.py', 'exec'), namespace)

        class Inputs(dict):
          def __init__(self):
            super().__init__()
            self.seen = {'modelV2': True, 'lateralManeuverPlan': False}
            self.updated = {'modelV2': True}
            self.healthy = True
            self.logMonoTime = {'modelV2': 1_150_000_000}

          def all_checks(self, services):
            return self.healthy if services == ['modelV2'] else True

          all_alive = all_valid = all_checks

        inputs, previews = Inputs(), []
        cp = car.CarParams.new_message(openpilotLongitudinalControl=True)
        cp.lateralTuning.init('torque')
        inputs.update(carState=SimpleNamespace(steeringAngleDeg=0.0, vEgo=20.0, vCruise=72.0, standstill=False,
                                               steerFaultTemporary=False, steerFaultPermanent=False),
                      vehicleParameters=SimpleNamespace(stiffnessFactor=1.0, steerRatio=15.0, angleOffsetDeg=0.0, roll=0.0),
                      selfdriveStateSP=SimpleNamespace(mads=SimpleNamespace(available=mads_only, enabled=True)),
                      selfdriveState=SimpleNamespace(enabled=not mads_only), onroadEvents=[],
                      lateralTorqueParameters=SimpleNamespace(useParams=False),
                      longitudinalPlan=SimpleNamespace(aTarget=0.0, shouldStop=False), carOutput=SimpleNamespace(),
                      lateralDelay=SimpleNamespace(lateralDelay=0.3))
        controls = SimpleNamespace(sm=inputs, CP=cp, CP_SP=SimpleNamespace(pcmCruiseSpeed=True),
                                   VM=SimpleNamespace(update_params=noop, calc_curvature=lambda *args: 0.0),
                                   lane_centering_safety=LaneCenteringSafetyLatch(), lat_delay=0.3,
                                   LaC=SimpleNamespace(reset=noop, update=lambda *args, **kwargs: (0.0, 0.0, None),
                                                       extension=SimpleNamespace(update_model_v2=previews.append, update_lateral_lag=noop)),
                                   LoC=SimpleNamespace(long_control_state='off', reset=noop, update=lambda *args: 0.0),
                                   CI=SimpleNamespace(get_pid_accel_limits=lambda *args: (-3.0, 2.0)),
                                   get_lat_active=lambda sm: True, desired_curvature=0.0,
                                   steer_limited_by_safety=False, calibrated_pose=None)
        good = model_frame()
        good.action.desiredCurvature = 0.003
        good.orientationRate.z = [0.04] * 33
        inputs['modelV2'] = good
        control, _ = namespace['state_control'](controls)
        self.assertTrue(control.latActive)
        self.assertIs(previews[-1], good)
        self.assertEqual(controls.selected_model_log_mono_time, 1_150_000_000)

        bad = model_frame(1_050_000_000, 2)
        bad.action.desiredCurvature = math.nan
        bad.orientationRate.z = [math.nan] * 33
        inputs['modelV2'] = bad
        inputs.logMonoTime['modelV2'] = 1_200_000_000
        clock[0] = 1_200_000_000
        control, _ = namespace['state_control'](controls)
        self.assertTrue(control.latActive)
        self.assertIs(previews[-1], good)
        self.assertEqual(controls.selected_model_log_mono_time, 1_150_000_000)
        self.assertAlmostEqual(control.actuators.curvature, good.action.desiredCurvature)
        clock[0] = good.timestampEof + MAX_MODEL_AGE_NS + 1
        control, _ = namespace['state_control'](controls)
        self.assertFalse(control.latActive)
        self.assertTrue(math.isfinite(control.actuators.curvature))

  def test_unavailable_and_bypassed_are_not_falsely_certified_or_blocked(self):
    for containment, reason in (('unavailable', ''), ('bypassed', 'mode_off'), ('bypassed', 'lane_change'), ('bypassed', 'low_speed')):
      with self.subTest(containment=containment, reason=reason):
        model = model_frame(containment=containment, reason=reason)
        model.laneCentering.minClearance = math.nan
        model.laneCentering.responseTime = math.nan
        model.laneCentering.checkedDistance = math.nan
        self.assertFalse(self.check(LaneCenteringSafetyLatch(), model))

  def test_latched_block_survives_controlsd_lat_active_feedback_until_engagement_cycle(self):
    control_guard, event_guard = LaneCenteringSafetyLatch(), LaneCenteringSafetyLatch()
    blocked = model_frame()
    blocked.action.desiredCurvature = math.nan
    for guard in (control_guard, event_guard):
      self.assertTrue(self.check(guard, blocked))
    # modeld observes carControl.latActive=False after controls stops steering
    # and publishes a bypassed decision. Requested engagement remains enabled.
    inactive_feedback = model_frame(1_050_000_000, 2, 'bypassed', 'lateral_inactive')
    for guard in (control_guard, event_guard):
      self.assertTrue(self.check(guard, inactive_feedback, requested=True))
      self.assertEqual(guard.reason, 'nonfinite_action')
    # Only the independent selfdrived/MADS engagement dropping clears the latch.
    for guard in (control_guard, event_guard):
      self.assertFalse(self.check(guard, inactive_feedback, requested=False, updated=False))
      self.assertIsNone(guard.latched_reason)
      self.assertFalse(self.check(guard, model_frame(1_100_000_000, 3), requested=True))

  def test_selfdrived_mads_only_engagement_uses_enabled_toggle_for_latch(self):
    tree = ast.parse((ROOT / 'openpilot/selfdrive/selfdrived/selfdrived.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'SelfdriveD')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'update_lane_centering_events')
    namespace = {'EventName': log.OnroadEvent.EventName}
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'selfdrived.py', 'exec'), namespace)

    class Inputs(dict):
      def __init__(self):
        super().__init__(controlsState=SimpleNamespace(lateralControlState=SimpleNamespace(which=lambda: 'torqueState')))
        self.seen = {'modelV2': True, 'lateralManeuverPlan': False}
        self.updated = {'modelV2': True}
        self.logMonoTime = {}

      def all_checks(self, services):
        return True

      all_alive = all_valid = all_checks

      def frame(self, model):
        self['modelV2'] = model
        self.logMonoTime['carState'] = model.timestampEof + 150_000_000

    inputs, events = Inputs(), set()
    namespace['model_clock_ns'] = lambda: inputs.logMonoTime['carState']
    selfdrive = SimpleNamespace(sm=inputs, enabled=False, events=events, lane_centering_safety=LaneCenteringSafetyLatch(),
                                mads=SimpleNamespace(enabled_toggle=True, available=False, enabled=True))
    update = namespace['update_lane_centering_events']
    bad = model_frame()
    bad.action.desiredCurvature = math.nan
    inputs.frame(bad)
    update(selfdrive)
    self.assertIn(log.OnroadEvent.EventName.laneCenteringUnavailable, events)
    self.assertEqual(selfdrive.lane_centering_safety.latched_reason, 'nonfinite_action')

    # Steering stops, then modeld reports lateral_inactive. ACC was already
    # disabled, so only the MADS requested-engagement state may clear this latch.
    events.clear()
    inputs.frame(model_frame(1_050_000_000, 2, 'bypassed', 'lateral_inactive'))
    update(selfdrive)
    self.assertIn(log.OnroadEvent.EventName.laneCenteringUnavailable, events)
    selfdrive.mads.enabled = False
    inputs.updated['modelV2'] = False
    events.clear()
    update(selfdrive)
    self.assertFalse(events)
    self.assertIsNone(selfdrive.lane_centering_safety.latched_reason)
    selfdrive.mads.enabled = True
    inputs.updated['modelV2'] = True
    inputs.frame(model_frame(1_100_000_000, 3))
    events.clear()
    update(selfdrive)
    self.assertFalse(events)

    # Both unavailable geometry and a predicted collision retain MADS engagement;
    # only the explicit collision flag produces any user-facing event.
    for frame_id, risk in ((4, False), (5, True), (6, False)):
      warning = model_frame(1_000_000_000 + frame_id * 50_000_000, frame_id, 'blocked', 'corridor_invalid')
      warning.laneCentering.safetyBlocked = True
      warning.laneCentering.collisionRisk = risk
      inputs.frame(warning)
      events.clear()
      update(selfdrive)
      self.assertEqual(events, {log.OnroadEvent.EventName.laneCenteringCollisionRisk} if risk else set())
      self.assertIsNone(selfdrive.lane_centering_safety.latched_reason)

  def test_takeover_event_disables_stock_and_mads_and_blocks_reentry(self):
    events_base = ast.parse((ROOT / 'openpilot/sunnypilot/selfdrive/selfdrived/events_base.py').read_text())
    et_class = next(node for node in events_base.body if isinstance(node, ast.ClassDef) and node.name == 'ET')
    namespace = {}
    exec(compile(ast.Module(body=[et_class], type_ignores=[]), 'events_base.py', 'exec'), namespace)
    et = namespace['ET']
    events_tree = ast.parse((ROOT / 'openpilot/selfdrive/selfdrived/events.py').read_text())
    mapping = next(node.value for node in events_tree.body if isinstance(node, ast.AnnAssign) and
                   isinstance(node.target, ast.Name) and node.target.id == 'EVENTS')
    takeover = next(value for key, value in zip(mapping.keys, mapping.values, strict=True)
                    if isinstance(key, ast.Attribute) and key.attr == 'laneCenteringUnavailable')
    event_types = {getattr(et, key.attr) for key in takeover.keys}
    self.assertEqual(event_types, {et.IMMEDIATE_DISABLE, et.NO_ENTRY})

    class EventSet:
      def __init__(self, kinds=()):
        self.kinds = set(kinds)

      def contains(self, kind):
        return kind in self.kinds

      def has(self, name):
        return False

      def contains_in_list(self, names):
        return False

    stock_ns = software_namespace('openpilot/selfdrive/selfdrived/state.py',
                                  {'log': log, 'ET': et, 'Events': EventSet, 'DT_CTRL': 0.01})
    mads_ns = software_namespace('openpilot/sunnypilot/mads/state.py',
                                 {'log': log, 'custom': custom, 'ET': et, 'DT_CTRL': 0.01, 'SOFT_DISABLE_TIME': 3})
    events = EventSet(event_types)
    stock = stock_ns['StateMachine']()
    stock.state = stock_ns['State'].enabled
    self.assertEqual(stock.update(events), (False, False))
    events.kinds.add(et.ENABLE)
    self.assertEqual(stock.update(events), (False, False))

    events.kinds = set(event_types)
    selfdrive = SimpleNamespace(enabled=False, state_machine=stock, events=events, events_sp=EventSet())
    mads = mads_ns['StateMachine'](SimpleNamespace(selfdrive=selfdrive))
    mads.state = mads_ns['State'].enabled
    self.assertEqual(mads.update(), (False, False))
    events.kinds.add(et.ENABLE)
    self.assertEqual(mads.update(), (False, False))
    events.kinds.clear()
    self.assertEqual(stock.update(events), (False, False))
    self.assertEqual(mads.update(), (False, False))
    events.kinds.add(et.ENABLE)
    self.assertEqual(stock.update(events), (True, True))
    self.assertEqual(mads.update(), (True, True))

    warning = next(value for key, value in zip(mapping.keys, mapping.values, strict=True)
                   if isinstance(key, ast.Attribute) and key.attr == 'laneCenteringCollisionRisk')
    events.kinds = {getattr(et, key.attr) for key in warning.keys}
    self.assertEqual(events.kinds, {et.WARNING})
    for _ in range(100):
      self.assertEqual(stock.update(events), (True, True))
      self.assertEqual(mads.update(), (True, True))
      self.assertIn(et.WARNING, stock.current_alert_types)

  def test_generic_model_health_checks_defer_to_usable_command_but_keep_other_service_faults(self):
    tree = ast.parse((ROOT / 'openpilot/selfdrive/selfdrived/selfdrived.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'SelfdriveD')
    guard_method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'update_lane_centering_events')
    update = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'update_events')
    # Execute the actual guard, auxiliary-event removal, generic checks and
    # frame-drop branch in their production order, without unrelated native IPC.
    relevant_names = ('model_command_usable', 'model_v2', 'num_events', 'has_disable_events', 'no_system_errors', 'health_services')
    selected = [node for node in update.body if
                (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in relevant_names for target in node.targets)) or
                (isinstance(node, ast.If) and (('model_command_usable' in ast.unparse(node.test) and
                                               'unavailable_processes' not in ast.unparse(node)) or
                                              'self.sm.all_checks(health_services)' in ast.unparse(node.test)))]
    update_subset = ast.FunctionDef(name='model_health_events', args=guard_method.args, body=selected, decorator_list=[])
    et = SimpleNamespace(NO_ENTRY='noEntry', SOFT_DISABLE='softDisable', IMMEDIATE_DISABLE='immediateDisable')
    event_name = log.OnroadEvent.EventName

    class Events(set):
      def remove(self, event):
        self.discard(event)

      def contains(self, event_type):
        disabling = {event_name.bigModelFailed, event_name.commIssue, event_name.commIssueAvgFreq, event_name.modeldLagging,
                     event_name.laneCenteringUnavailable}
        if event_type == et.NO_ENTRY:
          return bool(self & (disabling | {event_name.bigModelLoading}))
        if event_type == et.IMMEDIATE_DISABLE:
          return event_name.laneCenteringUnavailable in self
        return bool(self & (disabling - {event_name.laneCenteringUnavailable}))

    class Inputs(dict):
      def __init__(self):
        super().__init__(controlsState=SimpleNamespace(lateralControlState=SimpleNamespace(which=lambda: 'torqueState')))
        self.services = ['modelV2', 'carState', 'deviceMotion']
        self.seen = {'modelV2': True, 'lateralManeuverPlan': False}
        self.updated = {'modelV2': True}
        self.alive = dict.fromkeys(self.services, True)
        self.freq_ok = dict.fromkeys(self.services, True)
        self.valid = dict.fromkeys(self.services, True)

      def all_alive(self, services=None):
        return all(self.alive[s] for s in services or self.services)

      def all_freq_ok(self, services=None):
        return all(self.freq_ok[s] for s in services or self.services)

      def all_valid(self, services=None):
        return all(self.valid[s] for s in services or self.services)

      def all_checks(self, services=None):
        return self.all_alive(services) and self.all_freq_ok(services) and self.all_valid(services)

    for mads_only in (False, True):
      for fault in ('valid', 'alive', 'freq_ok', 'frame_drop'):
        for other_fault in (None, 'valid', 'alive', 'freq_ok'):
          with self.subTest(mads_only=mads_only, model_fault=fault, other_fault=other_fault):
            clock = [1_150_000_000]
            namespace = {'EventName': event_name, 'ET': et, 'SIMULATION': False, 'REPLAY': False,
                         'model_clock_ns': lambda clock=clock: clock[0], 'cloudlog': SimpleNamespace(event=lambda *args, **kwargs: None)}
            module = ast.fix_missing_locations(ast.Module(body=[guard_method, update_subset], type_ignores=[]))
            exec(compile(module, 'selfdrived.py', 'exec'), namespace)
            inputs, events = Inputs(), Events()
            selfdrive = SimpleNamespace(sm=inputs, enabled=not mads_only, events=events, logged_comm_issue=None,
                                        lane_centering_safety=LaneCenteringSafetyLatch(),
                                        mads=SimpleNamespace(enabled_toggle=mads_only, enabled=True))
            selfdrive.update_lane_centering_events = lambda selfdrive=selfdrive, namespace=namespace: namespace['update_lane_centering_events'](selfdrive)
            good = model_frame()
            inputs['modelV2'] = good
            namespace['model_health_events'](selfdrive)
            self.assertFalse(events)
            bad = model_frame(1_050_000_000, 2)
            bad.frameDropPerc = 50.0
            inputs['modelV2'] = bad
            if fault != 'frame_drop':
              getattr(inputs, fault)['modelV2'] = False
            if other_fault is not None:
              getattr(inputs, other_fault)['deviceMotion'] = False
            events.update((event_name.bigModelLoading, event_name.bigModelFailed))
            clock[0] = 1_200_000_000
            namespace['model_health_events'](selfdrive)
            expected = set() if other_fault is None else {event_name.commIssueAvgFreq if other_fault == 'freq_ok' else event_name.commIssue}
            self.assertEqual(events, expected)
            self.assertIsNone(selfdrive.lane_centering_safety.reason)
            self.assertIsNone(selfdrive.lane_centering_safety.latched_reason)

            # Once no fresh usable command remains, model-specific exclusions
            # stop, including the legacy frame-drop event and genuine fatal.
            clock[0] = 1_500_000_001
            events.clear()
            namespace['model_health_events'](selfdrive)
            self.assertIn(event_name.laneCenteringUnavailable, events)
            self.assertIn(event_name.modeldLagging, events)

    source = ast.unparse(update)
    self.assertLess(source.index('self.update_lane_centering_events()'), source.index('self.sm.all_checks(health_services)'))
    self.assertIn('desired_lateral_accel = model_v2.action.desiredCurvature', source)
    self.assertIn('model_fcw = model_v2.meta.hardBrakePredicted', source)

  def test_stopped_modeld_defers_only_while_command_usable_and_preserves_other_process_and_camera_faults(self):
    tree = ast.parse((ROOT / 'openpilot/selfdrive/selfdrived/selfdrived.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'SelfdriveD')
    guard_method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'update_lane_centering_events')
    update = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'update_events')
    # Preserve the entire production process-watchdog block, including the
    # camera-check else branch, and derive its usability from the real guard.
    start = next(i for i, node in enumerate(update.body) if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'not_running')
    end = next(i for i, node in enumerate(update.body[start:], start) if isinstance(node, ast.If) and
               'EventName.processNotRunning' in ast.unparse(node))
    decision = next(node for node in update.body if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'model_command_usable')
    process_method = ast.FunctionDef(name='process_events', args=guard_method.args, body=[decision, *update.body[start:end + 1]], decorator_list=[])
    event_name = log.OnroadEvent.EventName

    class Inputs(dict):
      def __init__(self, stopped, camera_fault):
        super().__init__(controlsState=SimpleNamespace(lateralControlState=SimpleNamespace(which=lambda: 'torqueState')),
                         modelV2=model_frame(), managerState=SimpleNamespace(processes=[
                           SimpleNamespace(name=name, running=False, shouldBeRunning=True) for name in stopped]))
        self.seen = {'modelV2': True, 'lateralManeuverPlan': False}
        self.updated = {'modelV2': True}
        self.recv_frame = {'managerState': 1}
        self.camera_fault = camera_fault

      def all_alive(self, services):
        return not (services == ['roadCameraState'] and self.camera_fault == 'alive')

      def all_freq_ok(self, services):
        return not (services == ['roadCameraState'] and self.camera_fault == 'frequency')

      def all_valid(self, services):
        return True

      def all_checks(self, services):
        return self.all_alive(services) and self.all_freq_ok(services)

    stopped_process_cases = tuple((backend, *others) for backend in ('modeld', 'modeld_tinygrad')
                                  for others in ((), ('controlsd',), ('mapd',)))
    for mads_only in (False, True):
      for stopped in (*stopped_process_cases, ('modeld', 'modeld_tinygrad')):
        for camera_fault in (None, 'alive', 'frequency'):
          with self.subTest(mads_only=mads_only, stopped=stopped, camera_fault=camera_fault):
            clock, logs = [1_150_000_000], []
            namespace = {'EventName': event_name, 'SIMULATION': False, 'model_clock_ns': lambda clock=clock: clock[0],
                         'cloudlog': SimpleNamespace(event=lambda name, logs=logs, **kwargs: logs.append((name, kwargs)))}
            module = ast.fix_missing_locations(ast.Module(body=[guard_method, process_method], type_ignores=[]))
            exec(compile(module, 'selfdrived.py', 'exec'), namespace)
            inputs, events = Inputs(stopped, camera_fault), set()
            selfdrive = SimpleNamespace(sm=inputs, enabled=not mads_only, events=events, ignored_processes={'mapd'},
                                        lane_centering_safety=LaneCenteringSafetyLatch(), not_running_prev=set(),
                                        mads=SimpleNamespace(enabled_toggle=mads_only, enabled=True),
                                        rk=SimpleNamespace(lagging=False), camera_packets=['roadCameraState'])
            selfdrive.update_lane_centering_events = lambda selfdrive=selfdrive, namespace=namespace: namespace['update_lane_centering_events'](selfdrive)
            namespace['process_events'](selfdrive)
            if 'controlsd' in stopped:
              expected = {event_name.processNotRunning}
            elif camera_fault is not None:
              expected = {event_name.cameraMalfunction if camera_fault == 'alive' else event_name.cameraFrameRate}
            else:
              expected = set()
            self.assertEqual(events, expected)
            self.assertEqual(logs[0][1]['not_running'], set(stopped))
            self.assertEqual(selfdrive.not_running_prev, set(stopped))
            self.assertEqual(selfdrive.ignored_processes, {'mapd'})

            inputs.updated['modelV2'] = False
            clock[0] = inputs['modelV2'].timestampEof + MAX_MODEL_AGE_NS
            events.clear()
            namespace['process_events'](selfdrive)
            self.assertEqual(events, expected)

            clock[0] = inputs['modelV2'].timestampEof + MAX_MODEL_AGE_NS + 1
            events.clear()
            namespace['process_events'](selfdrive)
            self.assertIn(event_name.laneCenteringUnavailable, events)
            self.assertIn(event_name.processNotRunning, events)

            # A real disengagement clears the fatal latch. Fresh model output
            # can recover even before the manager's process snapshot catches up.
            selfdrive.enabled = selfdrive.mads.enabled = False
            inputs['modelV2'] = model_frame(1_400_000_000, 2)
            inputs.updated['modelV2'] = True
            clock[0] = 1_450_000_000
            events.clear()
            namespace['process_events'](selfdrive)
            self.assertEqual(events, expected)
            self.assertIsNone(selfdrive.lane_centering_safety.reason)

  def test_model_ready_tracker_runs_without_emitting_noncollision_advisory(self):
    from openpilot.system.hardware.chestnut.availability import ModelAvailability

    tree = ast.parse((ROOT / 'openpilot/selfdrive/selfdrived/selfdrived.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'SelfdriveD')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'update_chestnut_events')
    clock = [0.0]
    namespace = {'EventName': log.OnroadEvent.EventName, 'custom': custom, 'time': SimpleNamespace(monotonic=lambda: clock[0])}
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'selfdrived.py', 'exec'), namespace)

    class Inputs(dict):
      seen = {'modelV2': True}
      healthy = True

      def all_checks(self, services):
        return self.healthy

    inputs = Inputs(modelV2=log.ModelDataV2.new_message(big=True))
    selfdrive = SimpleNamespace(sm=inputs, events=set(), events_sp=set(), model_availability=ModelAvailability(),
                                params=SimpleNamespace(get_bool=lambda name: False))
    for now in (0.0, 1.0, 2.0, 3.0):
      clock[0] = now
      namespace['update_chestnut_events'](selfdrive)
    self.assertTrue(selfdrive.model_availability.ready_announced)
    self.assertTrue(selfdrive.model_availability.was_big)
    self.assertFalse(selfdrive.events)
    self.assertFalse(selfdrive.events_sp)
    inputs.healthy = False
    namespace['update_chestnut_events'](selfdrive)
    self.assertIn(log.OnroadEvent.EventName.bigModelFailed, selfdrive.events)


if __name__ == '__main__':
  unittest.main()

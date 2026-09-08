import ast
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from openpilot.cereal import custom, log
from openpilot.selfdrive.modeld.lane_centering_safety import LaneCenteringSafetyLatch, MAX_MODEL_AGE_NS, model_clock_ns


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
  status.version = 1
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

  def test_missing_invalid_mismatched_and_nonfinite_status_block(self):
    for fault, reason in (('legacy', 'missing_status'), ('invalid', 'invalid_status'), ('frame', 'status_frame_mismatch'),
                          ('timestamp', 'status_frame_mismatch'), ('action', 'nonfinite_action'), ('acceleration', 'nonfinite_action'),
                          ('containment', 'invalid_containment_status'), ('blocked', 'containment_blocked'),
                          ('inactive_bypass', 'invalid_bypass_reason'), ('clearance', 'invalid_containment_geometry'),
                          ('distance_nan', 'invalid_containment_geometry'), ('distance_zero', 'invalid_containment_geometry'),
                          ('distance_negative', 'invalid_containment_geometry')):
      with self.subTest(fault=fault):
        guard, model = LaneCenteringSafetyLatch(), model_frame()
        if fault == 'legacy':
          model = log.ModelDataV2.new_message(frameId=1, timestampEof=1_000_000_000)
        elif fault == 'invalid':
          model.laneCentering.valid = False
        elif fault == 'frame':
          model.laneCentering.frameId += 1
        elif fault == 'timestamp':
          model.laneCentering.timestampEof += 1
        elif fault == 'action':
          model.action.desiredCurvature = math.nan
        elif fault == 'acceleration':
          model.action.desiredAcceleration = math.nan
        elif fault == 'containment':
          model.laneCentering.containment = 'unexpected'
        elif fault == 'inactive_bypass':
          model.laneCentering.containment = 'bypassed'
          model.laneCentering.reason = 'lateral_inactive'
        elif fault == 'clearance':
          model.laneCentering.minClearance = math.nan
        elif fault.startswith('distance_'):
          model.laneCentering.checkedDistance = {'distance_nan': math.nan, 'distance_zero': 0.0, 'distance_negative': -1.0}[fault]
        else:
          model.laneCentering.safetyBlocked = True
        self.assertTrue(self.check(guard, model, requested=False))
        self.assertEqual(guard.reason, reason)

  def test_source_age_and_unhealthy_stream_block_before_engagement(self):
    for age in (MAX_MODEL_AGE_NS + 1, -50_000_001):
      with self.subTest(age=age):
        guard, model = LaneCenteringSafetyLatch(), model_frame()
        self.assertTrue(self.check(guard, model, requested=False, now=model.timestampEof + age))
        self.assertEqual(guard.reason, 'stale_model')
    guard = LaneCenteringSafetyLatch()
    self.assertTrue(self.check(guard, model_frame(), healthy=False))
    self.assertEqual(guard.reason, 'model_unhealthy')

  def test_new_messages_with_duplicate_or_reordered_source_frames_block(self):
    for timestamp in (1_000_000_000, 950_000_000):
      with self.subTest(timestamp=timestamp):
        guard = LaneCenteringSafetyLatch()
        self.assertFalse(self.check(guard, model_frame()))
        self.assertTrue(self.check(guard, model_frame(timestamp, 2)))
        self.assertEqual(guard.reason, 'nonmonotonic_model')

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
    blocked = model_frame(containment='blocked')
    blocked.laneCentering.safetyBlocked = True
    for guard in (control_guard, event_guard):
      self.assertTrue(self.check(guard, blocked))
    # modeld observes carControl.latActive=False after controls stops steering
    # and publishes a bypassed decision. Requested engagement remains enabled.
    inactive_feedback = model_frame(1_050_000_000, 2, 'bypassed', 'lateral_inactive')
    for guard in (control_guard, event_guard):
      self.assertTrue(self.check(guard, inactive_feedback, requested=True))
      self.assertEqual(guard.reason, 'containment_blocked')
    # Only the independent selfdrived/MADS engagement dropping clears the latch.
    for guard in (control_guard, event_guard):
      self.assertTrue(self.check(guard, inactive_feedback, requested=False, updated=False))
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
    inputs.frame(model_frame(containment='blocked'))
    update(selfdrive)
    self.assertIn(log.OnroadEvent.EventName.laneCenteringUnavailable, events)
    self.assertEqual(selfdrive.lane_centering_safety.latched_reason, 'containment_blocked')

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
    self.assertIn(log.OnroadEvent.EventName.laneCenteringUnavailable, events)
    self.assertIsNone(selfdrive.lane_centering_safety.latched_reason)
    selfdrive.mads.enabled = True
    inputs.updated['modelV2'] = True
    inputs.frame(model_frame(1_100_000_000, 3))
    events.clear()
    update(selfdrive)
    self.assertFalse(events)

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


if __name__ == '__main__':
  unittest.main()

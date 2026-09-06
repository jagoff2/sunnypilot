import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from openpilot.system.hardware.chestnut.availability import ModelAvailability


ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize('loading', [False, True])
def test_small_model_remains_available_during_background_retry(loading):
  state = ModelAvailability()
  for now in range(200):
    alerts = state.update(now=now, healthy=True, big=False, loading=loading)
    assert not alerts.loading and not alerts.failed and not alerts.ready


def test_starting_without_a_dock_is_not_a_gpu_failure():
  state = ModelAvailability()
  alerts = state.update(now=0., healthy=False, big=False, loading=False)
  assert not alerts.loading and not alerts.failed and not alerts.ready


def test_loading_blocks_only_when_no_healthy_model_is_available():
  state = ModelAvailability()
  assert state.update(now=0., healthy=False, big=False, loading=True).loading
  assert not state.update(now=1., healthy=True, big=False, loading=True).loading


def test_cancelled_loading_does_not_announce_ready():
  state = ModelAvailability()
  state.update(now=0., healthy=True, big=False, loading=True)
  assert not state.update(now=1., healthy=True, big=False, loading=False).ready


def test_brief_fallbacks_do_not_generate_failure_or_reload_alerts():
  state = ModelAvailability()
  ready = 0
  for frame in range(1200):
    big = frame % 79 not in (0, 1, 2)
    alerts = state.update(now=frame / 20, healthy=True, big=big, loading=not big)
    assert not alerts.loading and not alerts.failed
    ready += alerts.ready
  assert ready == 1


def test_ready_requires_continuously_healthy_big_output():
  state = ModelAvailability()
  for now, big, healthy in [(0., True, True), (1.9, True, True), (2., False, True),
                            (3., True, True), (4., True, False), (5., True, True), (6.9, True, True)]:
    assert not state.update(now=now, healthy=healthy, big=big, loading=False).ready
  assert state.update(now=7., healthy=True, big=True, loading=False).ready


def test_disconnect_and_recovery_do_not_repeat_ready_announcement():
  state = ModelAvailability()
  state.update(now=0., healthy=True, big=True, loading=False)
  assert state.update(now=2., healthy=True, big=True, loading=False).ready
  state.update(now=3., healthy=True, big=False, loading=False)
  state.update(now=100., healthy=True, big=True, loading=False)
  assert not state.update(now=102., healthy=True, big=True, loading=False).ready


@pytest.mark.parametrize('loading', [False, True])
def test_loss_of_model_output_still_fails_until_healthy_output_returns(loading):
  state = ModelAvailability()
  state.update(now=0., healthy=True, big=True, loading=False)
  for now in (1., 2., 5.):
    assert state.update(now=now, healthy=False, big=True, loading=loading).failed
  assert not state.update(now=6., healthy=True, big=False, loading=loading).failed
  assert not state.update(now=7., healthy=True, big=True, loading=loading).failed


class SubMaster(dict):
  def __init__(self):
    super().__init__(modelV2=SimpleNamespace(big=False), deviceMotion=SimpleNamespace(posenetOK=True, inputsOK=True),
                     vehicleParameters=SimpleNamespace(valid=True))
    self.seen = {'modelV2': True}
    self.alive = {'modelV2': True}
    self.valid = {'modelV2': True}
    self.freq_ok = {'modelV2': True}

  def all_alive(self):
    return all(self.alive.values())

  def all_freq_ok(self):
    return all(self.freq_ok.values())

  def all_checks(self, services=None):
    assert services in (None, ['modelV2'])
    return self.all_alive() and self.all_freq_ok() and all(self.valid.values())


@pytest.fixture
def integration():
  # Exercise the actual selfdrived methods without importing native device
  # dependencies, opening driving sockets, or changing the host's Params.
  tree = ast.parse((ROOT / 'openpilot/selfdrive/selfdrived/selfdrived.py').read_text())
  cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SelfdriveD')
  methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
  event_names = ('bigModelLoading', 'bigModelFailed', 'bigModelReady', 'commIssue', 'commIssueAvgFreq',
                 'posenetInvalid', 'locationdTemporaryError', 'paramsdTemporaryError')
  enum = SimpleNamespace(**{k: k for k in event_names})
  clock = Mock(monotonic=Mock(return_value=0.))
  namespace = {'time': clock, 'EventName': enum, 'custom': SimpleNamespace(OnroadEventSP=SimpleNamespace(EventName=enum)),
               'ET': SimpleNamespace(NO_ENTRY='noEntry', SOFT_DISABLE='softDisable', IMMEDIATE_DISABLE='immediateDisable'),
               'cloudlog': Mock(), 'log': SimpleNamespace(ExtrinsicsCalibration=SimpleNamespace(Status=SimpleNamespace(calibrated=1))),
               'TESTING_CLOSET': False, 'SIMULATION': False, 'REPLAY': False}
  exec(compile(ast.Module(body=[methods['update_chestnut_events']], type_ignores=[]), '<selfdrived>', 'exec'), namespace)
  controller = SimpleNamespace(model_availability=ModelAvailability(), sm=SubMaster(), params=Mock(),
                               events=Mock(), events_sp=Mock(), CP=SimpleNamespace(notCar=False), logged_comm_issue=None)
  controller.params.get_bool.return_value = True
  return SimpleNamespace(controller=controller, update=namespace['update_chestnut_events'], methods=methods, namespace=namespace)


def test_selfdrived_ignores_stale_gpu_flags_while_fallback_is_healthy(integration):
  r = integration
  r.update(r.controller)
  r.controller.events.add.assert_not_called()
  r.controller.events_sp.add.assert_not_called()
  r.controller.params.get.assert_not_called()


@pytest.mark.parametrize('health_field', ['alive', 'valid', 'freq_ok'])
def test_selfdrived_protects_against_stale_invalid_or_slow_model_output(integration, health_field):
  r = integration
  r.controller.sm['modelV2'].big = True
  r.update(r.controller)
  getattr(r.controller.sm, health_field)['modelV2'] = False
  r.update(r.controller)
  r.controller.events.add.assert_any_call('bigModelFailed')
  r.controller.events.add.reset_mock()
  getattr(r.controller.sm, health_field)['modelV2'] = True
  r.controller.sm['modelV2'].big = False
  r.update(r.controller)
  r.controller.events.add.assert_not_called()


def test_unseen_model_cannot_announce_ready(integration):
  r = integration
  r.controller.sm.seen['modelV2'] = False
  r.controller.sm['modelV2'].big = True
  for now in range(5):
    r.namespace['time'].monotonic.return_value = now
    r.update(r.controller)
  r.controller.events_sp.add.assert_not_called()


@pytest.mark.parametrize('health_field,expected', [('alive', 'commIssue'), ('valid', 'commIssue'), ('freq_ok', 'commIssueAvgFreq')])
def test_background_loading_cannot_mask_comm_or_localization_failures(integration, health_field, expected):
  r = integration
  r.controller.big_model_loading = True
  r.controller.big_model_ready_t = 1e12
  r.controller.events.contains.return_value = False
  getattr(r.controller.sm, health_field)['modelV2'] = False
  r.controller.sm['deviceMotion'].posenetOK = False
  r.controller.sm['deviceMotion'].inputsOK = False
  r.controller.sm['vehicleParameters'].valid = False
  body = r.methods['update_events'].body
  start = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'has_disable_events')
  end = next(i for i, n in enumerate(body) if isinstance(n, ast.If) and 'EventName.sensorDataInvalid' in ast.unparse(n))
  namespace = dict(r.namespace, self=r.controller, num_events=0, cal_status=1)
  exec(compile(ast.Module(body=body[start:end], type_ignores=[]), '<selfdrived health checks>', 'exec'), namespace)
  for event in (expected, 'posenetInvalid', 'locationdTemporaryError', 'paramsdTemporaryError'):
    r.controller.events.add.assert_any_call(event)

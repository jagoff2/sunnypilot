import os
import signal
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from openpilot.system.hardware.chestnut import inference
from openpilot.system.hardware.chestnut.worker import GpuWorker, LOAD_TIMEOUT


class Fallback:
  vision_input_names = ['img', 'big_img']
  lat_delay = .1
  chestnut = False

  def __init__(self):
    self.history = []

  def run(self, bufs, transforms, inputs, *args):
    self.history.append(int(bufs['img'][0]))
    return {'frame': int(bufs['img'][0]), 'gpu': False}


class FakeWorker:
  def __init__(self, *args):
    self.metadata = {'host': {'vision_input_names': ['img', 'big_img']}, 'history_frames': 3}
    self.pending = None
    self.result = None
    self.closed = False
    self.fail = False
    self.stalled = False
    self.waits = []

  def submit(self, seq, frames, request):
    if self.metadata is None or self.pending is not None:
      return False
    self.pending = seq
    self.result = {'frame': int(frames['road'][0]), 'gpu': True}
    return True

  def poll(self, timeout=0.):
    self.waits.append(timeout)
    if self.fail:
      raise TimeoutError('GPU disappeared')
    if self.stalled or self.pending is None:
      return None
    result = self.pending, self.result
    self.pending = None
    return result

  def close(self):
    self.closed = True


@pytest.fixture
def runner(monkeypatch):
  clock = SimpleNamespace(now=0.)
  monkeypatch.setattr(inference.time, 'monotonic', lambda: clock.now)
  workers = []

  def factory(*args):
    worker = FakeWorker()
    workers.append(worker)
    return worker

  small = Fallback()
  params = Mock()
  model = inference.RecoveringModel(small, 'stock', 2, 2, 4, params, lambda: True, factory)

  def step(frame, device=(4, 2)):
    clock.now = float(frame)
    model.observe(device)
    bufs = {'img': np.full(4, frame, dtype=np.uint8), 'big_img': np.full(4, frame + 1, dtype=np.uint8)}
    return model.run(bufs, {k: np.eye(3) for k in bufs}, {'desire_pulse': np.zeros(8)})

  return SimpleNamespace(model=model, small=small, params=params, workers=workers, step=step, clock=clock)


def test_late_gpu_recovers_while_fallback_publishes_every_frame(runner):
  r = runner
  outputs = [r.step(i, None) for i in range(20)]
  outputs += [r.step(i) for i in range(20, 28)]
  assert all(out['frame'] == i for i, out in enumerate(outputs))
  assert not any(out['gpu'] for out in outputs[:25])
  assert outputs[-1]['gpu']
  assert r.small.history == list(range(28))
  assert len(r.workers) == 1
  r.params.put_bool.assert_any_call('ChestnutActive', True)
  # There is no vehicle speed, engagement, or ignition gate in the runtime path.
  assert r.model.chestnut


def test_disconnect_falls_back_on_same_frame_and_reconnects(runner):
  r = runner
  for i in range(7):
    r.step(i)
  assert r.model.chestnut
  assert r.step(7, None) == {'frame': 7, 'gpu': False}
  assert r.workers[0].closed
  for i in range(8, 15):
    out = r.step(i, (4, 3))
    assert out['frame'] == i
  assert out['gpu']
  assert len(r.workers) == 2
  assert r.small.history == list(range(15))


def test_hung_loading_never_delays_fallback(runner):
  r = runner
  for i in range(4):
    r.step(i)
  r.workers[0].metadata = None
  r.model.host = None
  for i in range(4, 10):
    assert r.step(i) == {'frame': i, 'gpu': False}
  assert r.workers[0].waits[-1] == 0.


def test_failed_worker_retries_without_restarting_fallback(runner):
  r = runner
  for i in range(7):
    r.step(i)
  r.workers[0].fail = True
  assert r.step(7) == {'frame': 7, 'gpu': False}
  assert r.workers[0].closed
  r.step(36)
  assert len(r.workers) == 1
  for i in range(37, 41):
    out = r.step(i)
  assert len(r.workers) == 2
  assert out['gpu']


def test_late_gpu_results_are_never_published_as_current_frame(runner):
  r = runner
  for i in range(4):
    r.step(i)
  r.workers[0].stalled = True
  assert r.step(4) == {'frame': 4, 'gpu': False}
  r.workers[0].stalled = False
  # Frame 4 arrives on frame 5. Only frame 5 may be selected, after warmup.
  for i in range(5, 9):
    assert r.step(i)['frame'] == i
  assert max(r.workers[0].waits) <= inference.FRAME_BUDGET


def test_reenumeration_rejects_old_device_results(runner):
  r = runner
  for i in range(4):
    r.step(i)
  old = r.workers[0]
  old.stalled = True
  r.step(4)
  assert r.step(5, (4, 3)) == {'frame': 5, 'gpu': False}
  assert old.closed


def test_brief_gpu_delay_requires_three_timely_results_before_handoff(runner):
  r = runner
  for i in range(7):
    r.step(i)
  r.workers[0].stalled = True
  assert not r.step(7)['gpu']
  assert not r.step(8)['gpu']
  r.workers[0].stalled = False
  for i in range(9, 11):
    assert not r.step(i)['gpu']
  assert r.step(11)['gpu']


def test_missing_compilation_keeps_fallback_live(runner):
  runner.model.available = lambda: False
  for i in range(10):
    assert not runner.step(i)['gpu']
  assert not runner.workers


def test_custom_desire_keys_and_input_copies():
  model = SimpleNamespace(vision_input_names=['road', 'big_road'], desire_key='desire')
  values = {'desire_pulse': np.ones(8), 'action_t': np.ones(2), 'lateral_control_params': np.ones(2)}
  frames = {'road': object(), 'wide': object()}
  bufs, _, mapped = inference.mapped_inputs(model, frames, {'road': np.eye(3), 'wide': np.eye(3)}, values)
  assert bufs['big_road'] is frames['wide']
  mapped['desire'][0] = 0
  assert values['desire_pulse'][0] == 1
  assert {'action_t', 'lateral_control_params'} <= mapped.keys()


def fake_session(config):
  # Imported in the actual isolated worker process for IPC/failure tests.
  class Session:
    metadata = {'test': True}

    def run(self, frames, request):
      if request.get('crash'):
        os._exit(17)
      if request.get('hang'):
        time.sleep(60)
      return {'road': frames['road'].copy(), 'wide': frames['wide'].copy(), 'tag': request['tag']}

  time.sleep(config.get('load_delay', 0.))
  return Session()


def await_ready(worker):
  deadline = time.monotonic() + 15
  while worker.metadata is None and time.monotonic() < deadline:
    worker.poll(.01)
  assert worker.metadata == {'test': True}


@pytest.fixture
def real_worker(tmp_path):
  worker = GpuWorker(16, {}, 'openpilot.system.hardware.tests.test_chestnut_inference:fake_session', tmp_path / 'worker.log')
  try:
    await_ready(worker)
    yield worker
  finally:
    worker.close()
    worker.process.wait(timeout=5)


def test_real_shared_frames_and_atomic_results(real_worker):
  w = real_worker
  frames = {'road': np.arange(16, dtype=np.uint8), 'wide': np.arange(16, dtype=np.uint8) + 10}
  assert w.submit(42, frames, {'tag': 'frame-42'})
  assert not w.submit(43, frames, {})
  result = None
  deadline = time.monotonic() + 5
  while result is None and time.monotonic() < deadline:
    result = w.poll(.01)
  seq, output = result
  assert seq == 42 and output['tag'] == 'frame-42'
  np.testing.assert_array_equal(output['road'], frames['road'])
  np.testing.assert_array_equal(output['wide'], frames['wide'])
  frames['road'][:] = 99
  w.submit(43, frames, {'tag': 'frame-43'})
  assert output['road'][0] == 0  # Previous output owns its memory.


@pytest.mark.parametrize('failure', ['crash', 'hang'])
def test_real_worker_failure_is_bounded(real_worker, failure):
  w = real_worker
  frames = dict.fromkeys(['road', 'wide'], np.zeros(16, dtype=np.uint8))
  w.submit(1, frames, {failure: True})
  deadline = time.monotonic() + 3
  with pytest.raises((RuntimeError, TimeoutError)):
    while time.monotonic() < deadline:
      w.poll(.01)
    pytest.fail('GPU failure did not time out')
  w.close()
  assert w.process.wait(timeout=5) in (17, -signal.SIGKILL)


def test_loading_timeout_and_close_are_bounded(monkeypatch, tmp_path):
  w = GpuWorker(16, {'load_delay': 60}, 'openpilot.system.hardware.tests.test_chestnut_inference:fake_session', tmp_path / 'worker.log')
  try:
    w.started -= LOAD_TIMEOUT + 1
    with pytest.raises(TimeoutError):
      w.poll()
  finally:
    w.close()
    w.close()
    assert w.process.wait(timeout=5) == -signal.SIGKILL

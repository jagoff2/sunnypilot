"""Same-frame lane computation with independent CPU, GIL and publication deadline."""
import math
import ctypes
import multiprocessing
import os
import pickle
import select
import signal
import socket
import time


MAX_PACKET_BYTES = 128 * 1024


class LanePlannerBusy(RuntimeError):
  pass


def configure_lane_planner_process():
  # CPU6 also runs camerad (SCHED_OTHER). Drop inherited realtime priority
  # BEFORE moving there, then use only CPU time left idle by camera work.
  # This process has its own GIL: starving it cannot hold modeld's Python lock.
  os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
  os.sched_setaffinity(0, {6})


def _encode(value):
  packet = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
  if len(packet) > MAX_PACKET_BYTES:
    raise ValueError('lane worker packet exceeds size limit')
  return packet


def _receive(channel):
  packet = channel.recv(MAX_PACKET_BYTES + 1)
  if not packet or len(packet) > MAX_PACKET_BYTES:
    raise RuntimeError('lane worker disconnected or oversized packet')
  return pickle.loads(packet)


def _work(channel, initialize, parent_pid):
  # Private socket only: no service publications or controls. Spawn avoids
  # inheriting a live GPU context.
  with channel:
    try:
      # Manager can SIGKILL modeld. Do not leave an orphan worker consuming CPU
      # after that restart, including the race before this child starts.
      if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'could not set lane worker parent-death signal')
      if os.getppid() != parent_pid:
        return
      if initialize is not None:
        initialize()
      # Readiness includes numerical imports, rather than charging first-job
      # module initialization against a single camera frame's compute budget.
      from openpilot.selfdrive.modeld import lane_centering  # noqa: F401
      channel.send(_encode(('ready',)))
    except Exception as error:
      channel.send(_encode(('error', type(error).__name__)))
      return
    while True:
      try:
        job = _receive(channel)
      except (EOFError, OSError, RuntimeError):
        return
      if job is None:
        return
      sequence, function, model_output, args, submitted = job
      started = time.monotonic()
      try:
        selected, status = function(model_output, *args)
        controller = getattr(function, '__self__', None)
        result = ('result', sequence, selected is model_output, selected, status,
                  vars(controller) if controller is not None else None,
                  time.monotonic() - started, started - submitted)
        packet = _encode(result)
      except Exception as error:
        packet = _encode(('error', sequence, type(error).__name__))
      try:
        channel.send(packet)
      except OSError:
        return


class LanePlannerWorker:
  def __init__(self, initialize=None, timeout=0.05):
    if not math.isfinite(timeout) or timeout <= 0:
      raise ValueError('lane worker timeout must be positive and finite')
    self.timeout = timeout
    self.last_compute_time = self.last_queue_time = None
    self.timeouts = 0
    self._sequence = 0
    self._pending = None
    self._ready = self._closed = False
    self._startup_error = None
    self._process = None
    self._channel = child = None
    try:
      self._channel, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
      self._channel.setblocking(False)
      self._process = multiprocessing.get_context('spawn').Process(
        target=_work, args=(child, initialize, os.getpid()), name='lane_planner', daemon=True)
      self._process.start()
    except Exception as error:
      self._startup_error = error
    finally:
      if child is not None:
        child.close()

  def _read(self, deadline):
    if not select.select([self._channel], [], [], max(0., deadline - time.monotonic()))[0]:
      raise TimeoutError('lane worker deadline exceeded')
    return _receive(self._channel)

  def wait_ready(self, timeout=0.):
    if self._closed:
      raise RuntimeError('lane worker closed')
    if self._startup_error is not None:
      raise RuntimeError('lane worker startup failed') from self._startup_error
    if not self._ready:
      ready = self._read(time.monotonic() + timeout)
      if ready != ('ready',):
        self._startup_error = RuntimeError(str(ready))
        raise self._startup_error
      self._ready = True

  def run(self, function, model_output, *args, deadline=None):
    self.last_compute_time = self.last_queue_time = None
    deadline = min(time.monotonic() + self.timeout, deadline) if deadline is not None else time.monotonic() + self.timeout
    # Startup runs independently; native publication never waits for imports.
    self.wait_ready()
    if self._pending is not None:
      if not select.select([self._channel], [], [], 0.)[0]:
        raise LanePlannerBusy('previous lane computation still running')
      self._read(time.monotonic())  # Discard the expired frame and its state.
      self._pending = None
    if time.monotonic() >= deadline:
      self.timeouts += 1
      raise TimeoutError('no lane computation budget remains')
    self._sequence += 1
    # Snapshot reused inference buffers and controller state. Only one bounded
    # atomic packet may be outstanding, with no feeder thread or queued backlog.
    packet = _encode((self._sequence, function, model_output, args, time.monotonic()))
    try:
      if not select.select([], [self._channel], [], max(0., deadline - time.monotonic()))[1]:
        raise TimeoutError('lane worker send deadline exceeded')
      self._channel.send(packet)
      self._pending = self._sequence
      reply = self._read(deadline)
      self._pending = None
      if time.monotonic() > deadline:
        raise TimeoutError('late lane result')
    except TimeoutError:
      self.timeouts += 1
      raise
    if reply[0] != 'result' or reply[1] != self._sequence:
      raise RuntimeError(f'lane worker failed: {reply[:3]}')
    _, _, native, selected, status, state, self.last_compute_time, self.last_queue_time = reply
    if state is not None:
      # Apply state only for this frame's timely result. The adapter resets its
      # controller on timeout; late results cannot resurrect its discarded state.
      controller = function.__self__
      controller.__dict__.clear()
      controller.__dict__.update(state)
    return (model_output if native else selected), status

  def close(self):
    if self._closed:
      return
    self._closed = True
    if self._channel is not None:
      self._channel.close()
    if self._process is not None and self._process.pid is not None:
      self._process.join(timeout=0.05)
      if self._process.is_alive():
        self._process.terminate()
        self._process.join(timeout=0.1)
      if self._process.is_alive():
        self._process.kill()
        self._process.join(timeout=0.1)


def create_lane_planner_worker():
  from openpilot.common.hardware import HARDWARE
  return LanePlannerWorker(configure_lane_planner_process) if HARDWARE.get_device_type() == 'mici' else None

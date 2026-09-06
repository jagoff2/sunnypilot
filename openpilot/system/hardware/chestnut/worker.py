"""Isolated GPU inference with shared camera buffers and atomic notifications."""
import atexit
import importlib
import json
import mmap
import os
import pickle
import select
import signal
import socket
import struct
import subprocess
import sys
import time


OUTPUT_BYTES = 8 * 1024 * 1024
HEADER = struct.Struct('!cQI')
LOAD_TIMEOUT = 180.
RUN_TIMEOUT = .5


class GpuWorker:
  def __init__(self, frame_bytes: int, config: dict, factory: str, log_path='/tmp/chestnut-inference.log'):
    self.frame_bytes = frame_bytes
    self.offset = frame_bytes * 2
    self.fd = os.memfd_create('chestnut-inference', os.MFD_CLOEXEC)
    os.ftruncate(self.fd, self.offset + OUTPUT_BYTES)
    self.memory = mmap.mmap(self.fd, self.offset + OUTPUT_BYTES)
    self.socket, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    self.socket.setblocking(False)
    self.process = None
    self.metadata = None
    self.pending = None
    self.started = time.monotonic()
    self.submitted = self.started
    self.error = None
    self.closed = False
    try:
      with open(log_path, 'w') as output:
        self.process = subprocess.Popen(
          [sys.executable, '-m', __name__, str(self.fd), str(child.fileno()), str(frame_bytes), str(os.getpid()), factory, json.dumps(config)],
          pass_fds=(self.fd, child.fileno()), stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
          start_new_session=True, env={**os.environ, 'PYTHONUNBUFFERED': '1', 'DEV': 'USB+AMD:LLVM', 'FRAME_DEV': 'CPU', 'GMMU': '0'})
    except BaseException:
      self.close()
      raise
    finally:
      child.close()
    atexit.register(self.close)

  def submit(self, sequence: int, frames: dict, request: dict) -> bool:
    if self.closed or self.metadata is None or self.pending is not None:
      return False
    for i, key in enumerate(('road', 'wide')):
      data = memoryview(frames[key].data).cast('B')
      if data.nbytes < self.frame_bytes:
        raise ValueError('Camera frame is shorter than the GPU shared buffer')
      self.memory[i*self.frame_bytes:(i+1)*self.frame_bytes] = data[:self.frame_bytes]
    payload = pickle.dumps((sequence, request), protocol=5)
    if len(payload) > 64 * 1024:
      raise ValueError('GPU request exceeds the control packet limit')
    self.socket.send(payload)
    self.pending = sequence
    self.submitted = time.monotonic()
    return True

  def poll(self, timeout: float = 0.):
    if self.closed:
      return None
    now = time.monotonic()
    if ((self.metadata is None and now - self.started > LOAD_TIMEOUT) or
        (self.pending is not None and now - self.submitted > RUN_TIMEOUT)):
      raise TimeoutError('GPU worker timed out')
    if not select.select([self.socket], [], [], max(0., timeout))[0]:
      if self.process.poll() is not None:
        raise RuntimeError(f'GPU worker exited {self.process.returncode}')
      return None
    packet = self.socket.recv(HEADER.size + 1)
    if len(packet) != HEADER.size:
      raise RuntimeError('GPU worker disconnected or sent an invalid notification')
    kind, sequence, size = HEADER.unpack(packet)
    if size > OUTPUT_BYTES:
      raise RuntimeError('GPU response exceeds shared buffer')
    result = pickle.loads(self.memory[self.offset:self.offset + size])
    if kind == b'E':
      raise RuntimeError(f'GPU worker: {result}')
    if kind == b'R' and self.metadata is None and self.pending is None:
      self.metadata = result
      return None
    if kind != b'O' or sequence != self.pending:
      raise RuntimeError('GPU response does not match the outstanding frame')
    self.pending = None
    return sequence, result

  def close(self):
    if self.closed:
      return
    self.closed = True
    if self.process is not None and self.process.poll() is None:
      try:
        os.killpg(self.process.pid, signal.SIGKILL)
      except ProcessLookupError:
        pass
    self.socket.close()
    self.memory.close()
    os.close(self.fd)
    atexit.unregister(self.close)
    # poll(), rather than wait(), keeps teardown off the frame deadline.
    if self.process is not None:
      self.process.poll()


def main():
  # The model publisher is realtime and pinned to CPU 7. Drop that scheduling
  # policy before importing model/GPU code; loading must not starve the publisher.
  os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
  os.sched_setaffinity(0, {min(cpu, (os.cpu_count() or 1) - 1) for cpu in (4, 5)})
  with open('/proc/self/oom_score_adj', 'w') as score:
    score.write('500')  # Prefer dropping background recovery over the live model on memory pressure.
  fd, sock_fd, frame_bytes, parent_pid = map(int, sys.argv[1:5])
  import ctypes
  libc = ctypes.CDLL(None, use_errno=True)
  if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
    raise OSError(ctypes.get_errno(), 'Unable to bind GPU worker lifetime to modeld')
  if os.getppid() != parent_pid:
    return

  offset = frame_bytes * 2
  memory = mmap.mmap(fd, offset + OUTPUT_BYTES)
  channel = socket.socket(fileno=sock_fd)

  def send(kind, sequence, value):
    data = pickle.dumps(value, protocol=5)
    if len(data) > OUTPUT_BYTES:
      raise ValueError('GPU result exceeds shared buffer')
    memory[offset:offset + len(data)] = data
    channel.send(HEADER.pack(kind, sequence, len(data)))

  try:
    import faulthandler
    faulthandler.enable()
    faulthandler.dump_traceback_later(120)
    print('Loading GPU inference session', flush=True)
    module, name = sys.argv[5].split(':')
    session = getattr(importlib.import_module(module), name)(json.loads(sys.argv[6]))
    faulthandler.cancel_dump_traceback_later()
    print('GPU inference session ready', flush=True)
    import numpy as np
    frames = {key: np.frombuffer(memory, dtype=np.uint8, count=frame_bytes, offset=i*frame_bytes)
              for i, key in enumerate(('road', 'wide'))}
    send(b'R', 0, session.metadata)
    while packet := channel.recv(64 * 1024):
      sequence, request = pickle.loads(packet)
      send(b'O', sequence, session.run(frames, request))
  except Exception as e:
    import traceback
    traceback.print_exc()
    try:
      send(b'E', 0, str(e))
    except OSError:
      pass


if __name__ == '__main__':
  main()

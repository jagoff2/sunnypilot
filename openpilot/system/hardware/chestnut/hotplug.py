import os
from pathlib import Path
import signal
import subprocess
import time


SETTLE_SECONDS = 3.
RETRY_SECONDS = 30.
MAX_RETRY_SECONDS = 300.
BUILD_TIMEOUT = 3600.
BUILD_TARGET = 'openpilot/selfdrive/modeld/models/big_driving_tinygrad.pkl.chunkmanifest'
USB_POWER_PATHS = (Path('/sys/bus/platform/devices/a600000.ssusb/power/control'), Path('/sys/bus/usb/devices/usb4/power/control'))
USB_WRAPPER_AUTOSUSPEND_MS = 1000


def keep_diy_usb_awake(paths=USB_POWER_PATHS):
  # Keep the SuperSpeed root hub awake while attached, but let the wrapper suspend
  # on detach so the next runtime resume reinitializes the QMP PHY.
  if os.getenv('CHESTNUT_DOCK') != 'asm2464':
    return
  # Wake the root hub before allowing its parent to autosuspend.
  for path in reversed(paths):
    try:
      resolved = path.resolve()
      if 'a600000.ssusb' not in resolved.parts:
        continue
      wrapper = resolved.parent.parent.name == 'a600000.ssusb'
      if wrapper:
        # The flasher can leave a negative delay, which blocks runtime suspend
        # even after control is restored to auto. Preserve valid custom delays.
        delay = path.with_name('autosuspend_delay_ms')
        if delay.exists() and int(delay.read_text().strip()) < 0:
          subprocess.run(['sudo', '-n', 'tee', str(delay)], input=f'{USB_WRAPPER_AUTOSUSPEND_MS}\n', text=True, check=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=2)
      policy = 'auto' if wrapper else 'on'
      if path.read_text().strip() == policy:
        continue
      subprocess.run(['sudo', '-n', 'tee', str(path)], input=f'{policy}\n', text=True, check=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=2)
    except (OSError, ValueError, subprocess.SubprocessError) as e:
      print(f'Unable to keep external USB awake: {e}', flush=True)


class DockRetry:
  def __init__(self):
    self.device: tuple[int, int] | None = None
    self.retry_at = 0.
    self.attempts = 0

  def observe(self, device: tuple[int, int] | None, now: float) -> None:
    if device != self.device:
      self.device = device
      self.attempts = 0
      self.retry_at = now + SETTLE_SECONDS

  def due(self, now: float) -> bool:
    return self.device is not None and now >= self.retry_at

  def attempted(self, now: float) -> None:
    self.retry_at = now + min(MAX_RETRY_SECONDS, RETRY_SECONDS * 2 ** min(self.attempts, 4))
    self.attempts += 1


def compilation_running(proc_root: Path = Path('/proc')) -> bool:
  # Avoid starting SCons alongside a manual build or the model downloader's compiler.
  for entry in proc_root.glob('[0-9]*'):
    try:
      args = (entry / 'cmdline').read_bytes().split(b'\0')
      if any(Path(os.fsdecode(arg)).name in ('scons', 'compile_modeld.py', 'compile3.py') for arg in args):
        return True
    except OSError:
      continue
  return False


class ChestnutModelBuilder:
  def __init__(self, root: Path, log_path: Path = Path('/tmp/chestnut-build.log')):
    self.root = root
    self.log_path = log_path
    self.retry = DockRetry()
    self.process: subprocess.Popen | None = None
    self.started_at = 0.
    self.stopping_at: float | None = None

  def _signal(self, sig: signal.Signals) -> None:
    if self.process is not None:
      try:
        os.killpg(self.process.pid, sig)
      except ProcessLookupError:
        pass

  def update(self, now: float, device: tuple[int, int] | None, *, offroad: bool, compiled: bool,
             source_available: bool, other_build_running: bool = False) -> None:
    changed = device != self.retry.device
    self.retry.observe(device, now)
    if self.process is not None:
      if self.process.poll() is not None:
        # Also reap any compiler descendants left behind by an interrupted SCons.
        self._signal(signal.SIGKILL)
        print(f'Chestnut build exited {self.process.returncode}; compiled={compiled}', flush=True)
        self.process = None
        self.stopping_at = None
        # Back off from completion, including SCons's "not ready, skipping" exit 0.
        self.retry.attempted(now)
      else:
        if self.stopping_at is None and (not offroad or device is None or changed or now - self.started_at >= BUILD_TIMEOUT):
          print('Stopping Chestnut build: drive started, dock changed, or build timed out', flush=True)
          self._signal(signal.SIGTERM)
          self.stopping_at = now
        if self.stopping_at is not None and now - self.stopping_at >= 2.:
          self._signal(signal.SIGKILL)
        return

    if compiled:
      self.retry.attempts = 0
      return
    if not offroad or not source_available or other_build_running or not self.retry.due(now):
      return

    env = {**os.environ, 'PYTHONPATH': str(self.root), 'PYTHONUNBUFFERED': '1'}
    try:
      with self.log_path.open('wb') as output:
        self.process = subprocess.Popen(['scons', '-j1', BUILD_TARGET], cwd=self.root, env=env,
                                        stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
      self.started_at = now
      print(f'Building missing Chestnut model; log: {self.log_path}', flush=True)
    except OSError as e:
      print(f'Unable to start Chestnut build: {e}', flush=True)
      self.retry.attempted(now)

  def close(self) -> None:
    if self.process is not None:
      self._signal(signal.SIGKILL)
      self.process.wait(timeout=5)
      self.process = None


def main() -> None:
  from openpilot.cereal import messaging
  from openpilot.common.basedir import BASEDIR
  from openpilot.common.hardware.usb import chestnut_usb_identity
  from openpilot.selfdrive.modeld.helpers import MODELS_DIR, chestnut_compiled

  def terminate(signum, frame):
    raise SystemExit

  signal.signal(signal.SIGTERM, terminate)
  sm = messaging.SubMaster(['deviceState'], poll='deviceState')
  builder = ChestnutModelBuilder(Path(BASEDIR))
  next_power_check = 0.
  try:
    while True:
      sm.update(1000)
      now = time.monotonic()
      if now >= next_power_check:
        keep_diy_usb_awake()
        next_power_check = now + 5.
      valid = sm.all_checks(['deviceState'])
      device = chestnut_usb_identity(sm['deviceState'].usbState.devices) if valid else None
      compiled = chestnut_compiled()
      source = MODELS_DIR / 'big_driving_supercombo.onnx'
      source_available = (source.is_file() or Path(f'{source}.chunkmanifest').is_file()) and not os.getenv('SKIP_TINYGRAD_COMPILE')
      busy = builder.process is None and not compiled and device is not None and compilation_running()
      builder.update(now, device, offroad=valid and not sm['deviceState'].started, compiled=compiled,
                     source_available=bool(source_available), other_build_running=busy)
  finally:
    builder.close()


if __name__ == '__main__':
  main()

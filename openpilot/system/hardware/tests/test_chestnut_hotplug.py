import ast
from enum import Enum
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from openpilot.system.hardware.chestnut import hotplug


ROOT = Path(__file__).resolve().parents[4]
PRODUCT = 'custom 6e1e151e-CLEAN'
DEVICE = (4, 2)


def load_definitions(relative, namespace, names):
  path = ROOT / relative
  body = [n for n in ast.parse(path.read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
  exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), namespace)
  return namespace


@pytest.fixture
def usb_identity():
  # usb.py is pure sysfs logic. Load it without importing the native hardware package.
  path = ROOT / 'openpilot/common/hardware/usb.py'
  namespace = {}
  exec(compile(path.read_text(), str(path), 'exec'), namespace)
  return namespace['chestnut_usb_identity']


def dock(**kwargs):
  return SimpleNamespace(**{'busnum': 4, 'devnum': 2, 'vendorId': 0x3801, 'productId': 1,
                            'product': PRODUCT, 'speedMbps': 5000, **kwargs})


@pytest.mark.parametrize('devices,expected', [
  ([], None), ([dock()], DEVICE), ([dock(vendorId=0xADD1)], DEVICE),
  ([dock(speedMbps=12)], None), ([dock(speedMbps=480)], None),
  ([dock(product='stale firmware')], None),
  ([dock(vendorId=0x174C, productId=0x2464)], None),
  ([dock(), dock(devnum=3)], None),
  ([dock(), dock(vendorId=0x174C, productId=0x2463)], None),
  ([dock(), dock(vendorId=0x2C7C, productId=0x6007)], DEVICE),
])
def test_identity_requires_one_current_superspeed_dock(usb_identity, devices, expected):
  assert usb_identity(devices) == expected


def test_diy_keeps_hub_awake_and_allows_parent_resume(monkeypatch, tmp_path):
  external = tmp_path / 'a600000.ssusb' / 'power' / 'control'
  hub = tmp_path / 'a600000.ssusb' / 'xhci-hcd.1.auto' / 'usb4' / 'power' / 'control'
  modem = tmp_path / 'a800000.ssusb' / 'power' / 'control'
  modem_hub = tmp_path / 'a800000.ssusb' / 'xhci-hcd.0.auto' / 'usb2' / 'power' / 'control'
  paths = [external, hub, modem, modem_hub]
  for path in paths:
    path.parent.mkdir(parents=True)
    path.write_text('auto\n')
  external.write_text('on\n')
  run = Mock(side_effect=lambda args, **kwargs: Path(args[-1]).write_text(kwargs['input']))
  monkeypatch.setattr(hotplug.subprocess, 'run', run)
  monkeypatch.setenv('CHESTNUT_DOCK', 'asm2464')
  hotplug.keep_diy_usb_awake(paths)
  assert [(call.args[0], call.kwargs['input']) for call in run.call_args_list] == [
    (['sudo', '-n', 'tee', str(hub)], 'on\n'),
    (['sudo', '-n', 'tee', str(external)], 'auto\n'),
  ]
  assert external.read_text() == 'auto\n'
  assert hub.read_text() == 'on\n'
  assert modem.read_text() == modem_hub.read_text() == 'auto\n'
  run.reset_mock()
  hotplug.keep_diy_usb_awake(paths)
  run.assert_not_called()


@pytest.mark.parametrize('parent_policy', ['auto', 'on'])
def test_diy_repairs_flasher_delay_before_skipping_or_allowing_parent(monkeypatch, tmp_path, parent_policy):
  external = tmp_path / 'a600000.ssusb' / 'power' / 'control'
  hub = tmp_path / 'a600000.ssusb' / 'xhci-hcd.1.auto' / 'usb4' / 'power' / 'control'
  modem = tmp_path / 'a800000.ssusb' / 'power' / 'control'
  for path in (external, hub, modem):
    path.parent.mkdir(parents=True)
    path.write_text('auto\n')
    path.with_name('autosuspend_delay_ms').write_text('-1\n')
  external.write_text(f'{parent_policy}\n')
  run = Mock(side_effect=lambda args, **kwargs: Path(args[-1]).write_text(kwargs['input']))
  monkeypatch.setattr(hotplug.subprocess, 'run', run)
  monkeypatch.setenv('CHESTNUT_DOCK', 'asm2464')

  hotplug.keep_diy_usb_awake([external, hub, modem])

  expected = [(str(hub), 'on\n'), (str(external.with_name('autosuspend_delay_ms')), '1000\n')]
  if parent_policy == 'on':
    expected.append((str(external), 'auto\n'))
  assert [(call.args[0][-1], call.kwargs['input']) for call in run.call_args_list] == expected
  assert external.read_text() == 'auto\n'
  assert external.with_name('autosuspend_delay_ms').read_text() == '1000\n'
  assert hub.with_name('autosuspend_delay_ms').read_text() == modem.with_name('autosuspend_delay_ms').read_text() == '-1\n'
  assert modem.read_text() == 'auto\n'
  run.reset_mock()
  hotplug.keep_diy_usb_awake([external, hub, modem])
  run.assert_not_called()


@pytest.mark.parametrize('delay_ms', [0, 1000, 5000])
def test_diy_preserves_nonnegative_parent_delay(monkeypatch, tmp_path, delay_ms):
  external = tmp_path / 'a600000.ssusb' / 'power' / 'control'
  external.parent.mkdir(parents=True)
  external.write_text('on\n')
  delay = external.with_name('autosuspend_delay_ms')
  delay.write_text(f'{delay_ms}\n')
  run = Mock(side_effect=lambda args, **kwargs: Path(args[-1]).write_text(kwargs['input']))
  monkeypatch.setattr(hotplug.subprocess, 'run', run)
  monkeypatch.setenv('CHESTNUT_DOCK', 'asm2464')

  hotplug.keep_diy_usb_awake([external])

  assert run.call_count == 1
  assert run.call_args.args[0] == ['sudo', '-n', 'tee', str(external)]
  assert run.call_args.kwargs['input'] == 'auto\n'
  assert delay.read_text() == f'{delay_ms}\n'


@pytest.mark.parametrize('mode', ['chestnut', 'unknown', None])
def test_non_diy_preserves_usb_power_policy(monkeypatch, tmp_path, mode):
  external = tmp_path / 'a600000.ssusb' / 'power' / 'control'
  hub = tmp_path / 'a600000.ssusb' / 'xhci-hcd.1.auto' / 'usb4' / 'power' / 'control'
  for path, policy in ((external, 'on\n'), (hub, 'auto\n')):
    path.parent.mkdir(parents=True)
    path.write_text(policy)
  run = Mock()
  monkeypatch.setattr(hotplug.subprocess, 'run', run)
  if mode is None:
    monkeypatch.delenv('CHESTNUT_DOCK', raising=False)
  else:
    monkeypatch.setenv('CHESTNUT_DOCK', mode)
  hotplug.keep_diy_usb_awake([external, hub])
  run.assert_not_called()


@pytest.fixture
def builder(monkeypatch, tmp_path):
  processes = []

  def popen(*args, **kwargs):
    p = Mock(pid=123 + len(processes), returncode=None)
    p.poll.side_effect = lambda: p.returncode
    processes.append(p)
    return p

  launch = Mock(side_effect=popen)
  signals = Mock()
  monkeypatch.setattr(hotplug.subprocess, 'Popen', launch)
  monkeypatch.setattr(hotplug.os, 'killpg', signals)
  value = hotplug.ChestnutModelBuilder(tmp_path, tmp_path / 'build.log')

  def update(now, device=DEVICE, **kwargs):
    value.update(now, device, **{'offroad': True, 'compiled': False, 'source_available': True, **kwargs})

  return SimpleNamespace(value=value, update=update, launch=launch, processes=processes, signals=signals)


def test_builds_after_arbitrarily_late_enumeration(builder):
  builder.update(0, None)
  builder.update(120, None)
  builder.update(121)
  builder.update(123)
  builder.launch.assert_not_called()
  builder.update(124)
  assert builder.launch.call_count == 1
  command = builder.launch.call_args.args[0]
  assert command == ['scons', '-j1', hotplug.BUILD_TARGET]
  assert builder.launch.call_args.kwargs['start_new_session']
  assert builder.launch.call_args.kwargs['env']['PYTHONPATH'] == str(builder.value.root)
  builder.update(200)
  assert builder.launch.call_count == 1


@pytest.mark.parametrize('kwargs', [{'offroad': False}, {'compiled': True}, {'source_available': False}, {'other_build_running': True}])
def test_does_not_start_unnecessary_or_conflicting_builds(builder, kwargs):
  builder.update(0, **kwargs)
  builder.update(100, **kwargs)
  builder.launch.assert_not_called()


@pytest.mark.parametrize('exit_code', [0, 1])
def test_failed_or_skipped_build_retries_with_backoff(builder, exit_code):
  builder.update(0)
  builder.update(3)
  builder.processes[-1].returncode = exit_code
  builder.update(10)
  builder.update(39)
  assert builder.launch.call_count == 1
  builder.update(40)
  assert builder.launch.call_count == 2
  builder.processes[-1].returncode = exit_code
  builder.update(50)
  builder.update(109)
  assert builder.launch.call_count == 2
  builder.update(110)
  assert builder.launch.call_count == 3


def test_successful_build_is_not_repeated(builder):
  builder.update(0)
  builder.update(3)
  builder.processes[-1].returncode = 0
  builder.update(10, compiled=True)
  builder.update(1000, compiled=True)
  assert builder.launch.call_count == 1
  assert builder.value.process is None


@pytest.mark.parametrize('device,offroad', [(None, True), ((4, 3), True), (DEVICE, False)])
def test_stops_entire_build_on_disconnect_reenumeration_or_drive(builder, device, offroad):
  builder.update(0)
  builder.update(3)
  builder.update(4, device, offroad=offroad)
  builder.signals.assert_called_with(123, signal.SIGTERM)
  builder.update(6, device, offroad=offroad)
  builder.signals.assert_called_with(123, signal.SIGKILL)
  assert builder.launch.call_count == 1
  builder.processes[-1].returncode = -9
  builder.update(7, device, offroad=offroad)
  builder.update(100, DEVICE)
  builder.update(103, DEVICE)
  assert builder.launch.call_count == 2


def test_stuck_build_is_killed_and_retried(builder):
  builder.update(0)
  builder.update(3)
  builder.update(3 + hotplug.BUILD_TIMEOUT)
  builder.signals.assert_called_with(123, signal.SIGTERM)
  builder.update(5 + hotplug.BUILD_TIMEOUT)
  builder.signals.assert_called_with(123, signal.SIGKILL)


def test_shutdown_cleans_up_compiler_children(builder):
  builder.update(0)
  builder.update(3)
  builder.value.close()
  builder.signals.assert_called_once_with(123, signal.SIGKILL)
  builder.processes[0].wait.assert_called_once_with(timeout=5)
  assert builder.value.process is None


def test_start_failure_backs_off(builder):
  builder.launch.side_effect = OSError('missing scons')
  builder.update(0)
  builder.update(3)
  builder.update(4)
  assert builder.launch.call_count == 1
  builder.update(33)
  assert builder.launch.call_count == 2


def test_recognizes_existing_compilers_without_matching_shell_text(tmp_path):
  proc = tmp_path / '123'
  proc.mkdir()
  cmdline = proc / 'cmdline'
  cmdline.write_bytes(b'bash\0-c\0scons -j1 target\0')
  assert not hotplug.compilation_running(tmp_path)
  cmdline.write_bytes(b'python3\0/usr/local/venv/bin/scons\0-j1\0')
  assert hotplug.compilation_running(tmp_path)
  cmdline.write_bytes(b'python3\0/data/openpilot/openpilot/selfdrive/modeld/compile_modeld.py\0')
  assert hotplug.compilation_running(tmp_path)








def test_retry_backoff_is_capped():
  r = hotplug.DockRetry()
  for _ in range(100):
    r.attempted(10)
  assert r.retry_at == 310










def test_ui_recovers_from_late_detection_and_latched_failure():
  path = ROOT / 'openpilot/selfdrive/ui/ui_state.py'
  namespace = {'Enum': Enum}
  load_definitions('openpilot/selfdrive/ui/ui_state.py', namespace, ['ChestnutState'])
  cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'UIState')
  method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_update_chestnut_state')
  exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
  state = namespace['ChestnutState']

  class Messages(dict):
    recv_frame = {'modelV2': 2}
    alive = {'modelV2': True}

  ui = SimpleNamespace(sm=Messages(deviceState=SimpleNamespace(chestnutPresent=False), modelV2=SimpleNamespace(big=False)),
                       started=True, started_frame=1, chestnut_present=False, chestnut_compiled=True,
                       chestnut_loading=False, chestnut_active=None, chestnut_state=state.DISCONNECTED)
  def update():
    namespace['_update_chestnut_state'](ui)

  update()
  assert ui.chestnut_state == state.DISCONNECTED
  ui.sm['deviceState'].chestnutPresent = True
  update()
  assert ui.chestnut_state == state.FAILED
  ui.chestnut_loading = True
  update()
  assert ui.chestnut_state == state.LOADING
  ui.chestnut_loading = False
  ui.chestnut_active = True
  ui.sm['modelV2'].big = True
  update()
  assert ui.chestnut_state == state.ACTIVE
  ui.sm['deviceState'].chestnutPresent = False
  update()
  assert ui.chestnut_state == state.FAILED
  ui.sm['deviceState'].chestnutPresent = True
  update()
  assert ui.chestnut_state == state.ACTIVE


def test_real_build_subprocess_group_is_cleaned_up(monkeypatch, tmp_path):
  # Simulate SCons spawning a compiler that ignores SIGTERM. Cancelling must
  # terminate both processes, otherwise the compiler keeps the USB GPU locked.
  script = tmp_path / 'scons'
  child = '''import os, signal, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path("child.pid").write_text(str(os.getpid()))
time.sleep(60)
'''
  script.write_text(f'''#!{sys.executable}
import signal, subprocess, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
subprocess.Popen([sys.executable, "-c", {child!r}])
time.sleep(60)
''')
  script.chmod(0o700)
  monkeypatch.setenv('PATH', str(tmp_path) + os.pathsep + os.environ['PATH'])
  builder = hotplug.ChestnutModelBuilder(tmp_path, tmp_path / 'build.log')
  try:
    for now in (0, 3):
      builder.update(now, DEVICE, offroad=True, compiled=False, source_available=True)
    deadline = time.monotonic() + 5
    while not (tmp_path / 'child.pid').exists() and time.monotonic() < deadline:
      time.sleep(.01)
    child_pid = int((tmp_path / 'child.pid').read_text())
    process = builder.process
    assert process is not None
    assert os.getpgid(child_pid) == process.pid
    builder.update(4, DEVICE, offroad=False, compiled=False, source_available=True)
    assert process.poll() is None
    builder.update(6, DEVICE, offroad=False, compiled=False, source_available=True)
    assert process.wait(timeout=5) == -signal.SIGKILL
    child_stat = Path(f'/proc/{child_pid}/stat')
    deadline = time.monotonic() + 5
    while child_stat.exists() and child_stat.read_text().split()[2] != 'Z' and time.monotonic() < deadline:
      time.sleep(.01)
    assert not child_stat.exists() or child_stat.read_text().split()[2] == 'Z'
  finally:
    builder.close()

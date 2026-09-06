import ast
import ctypes
import os
from pathlib import Path
import struct
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from openpilot.system.hardware.chestnut import flash, readiness


PRODUCT = "custom ed4e39b7-CLEAN"
ROOT = Path(__file__).resolve().parents[4]


class Clock:
  def __init__(self):
    self.now = 0.

  def monotonic(self):
    return self.now

  def sleep(self, duration):
    self.now += duration


@pytest.fixture
def clock(monkeypatch):
  value = Clock()
  monkeypatch.setattr(readiness, 'time', value)
  return value


@pytest.fixture
def bridge(monkeypatch):
  device = SimpleNamespace(ltssm=0x78, voltage=0, current=0, fault=False, short_request=None, error=False, transfers=[], fds=[])
  monkeypatch.setattr(flash, 'find_chestnut', lambda: ('/sys/bus/usb/devices/4-1', ('3801', '0001'), PRODUCT))

  def open_device(path):
    fd = os.open(os.devnull, os.O_RDWR)
    device.fds.append(fd)
    return fd

  def ioctl(fd, request, ctrl):
    assert request == flash.USBDEVFS_CONTROL
    assert 0 < ctrl.timeout <= 2000
    device.transfers.append((ctrl.request_type, ctrl.request, ctrl.value))
    if device.error:
      raise OSError("USB device disconnected")
    if (ctrl.request_type, ctrl.request, ctrl.value) == (0x40, 0xF3, 1):
      assert ctrl.length == 0
      device.ltssm = 0x78
      return 0
    if (ctrl.request_type, ctrl.request, ctrl.value) == (0xC0, 0xE4, 0xB450):
      payload = bytes([device.ltssm])
    else:
      assert (ctrl.request_type, ctrl.request, ctrl.value) == (0xC0, 0xC0, 0)
      payload = struct.pack('<Hh?', device.voltage, device.current, device.fault)
    assert ctrl.length == len(payload)
    if device.short_request == ctrl.request:
      payload = payload[:-1]
    ctypes.memmove(ctrl.data, payload, len(payload))
    return len(payload)

  monkeypatch.setattr(flash, 'open_device', open_device)
  monkeypatch.setattr(flash.fcntl, 'ioctl', ioctl)
  yield device
  for fd in device.fds:
    try:
      os.fstat(fd)
    except OSError:
      continue
    os.close(fd)
    pytest.fail("Readiness probe leaked a USB file descriptor")


@pytest.mark.parametrize('mode,voltage,current,fault,ltssm,ready,power_fault', [
  ('asm2464', 0, 0, False, 0x78, True, False),
  ('asm2464', 0, 0, False, 0, False, False),
  ('asm2464', 0, 0, True, 0x78, False, True),
  ('asm2464', 0, 1, False, 0x78, False, True),
  ('asm2464', 4900, 0, False, 0x78, False, True),
  ('asm2464', 5000, 0, False, 0x78, True, False),
  ('asm2464', 12000, 3000, False, 0, False, False),
  ('chestnut', 0, 0, False, 0x78, False, True),
  ('chestnut', 4999, 0, False, 0x78, False, True),
  ('chestnut', 5000, 0, False, 0x78, True, False),
  ('chestnut', 12000, 1000, True, 0x78, False, True),
  ('chestnut', 12000, 1000, False, 0, False, False),
  ('typo', 0, 0, False, 0x78, False, True),
  (None, 0, 0, False, 0x78, False, True),
])
def test_power_policy(monkeypatch, mode, voltage, current, fault, ltssm, ready, power_fault):
  if mode is None:
    monkeypatch.delenv('CHESTNUT_DOCK', raising=False)
  else:
    monkeypatch.setenv('CHESTNUT_DOCK', mode)
  state = readiness.ChestnutReadiness(voltage, current, fault, ltssm)
  assert readiness.chestnut_ready(state) == ready
  assert readiness.chestnut_power_fault(state) == power_fault


def test_probe_ready_diy_without_publisher(monkeypatch, bridge):
  monkeypatch.setenv('CHESTNUT_DOCK', 'asm2464')
  assert readiness.wait_for_chestnut_ready(PRODUCT)
  assert bridge.transfers == [(0xC0, 0xE4, 0xB450), (0xC0, 0xC0, 0)]


def test_probe_powers_up_disabled_pcie(monkeypatch, bridge):
  monkeypatch.setenv('CHESTNUT_DOCK', 'asm2464')
  bridge.ltssm = 0
  assert readiness.wait_for_chestnut_ready(PRODUCT)
  assert bridge.transfers == [(0xC0, 0xE4, 0xB450), (0x40, 0xF3, 1), (0xC0, 0xE4, 0xB450), (0xC0, 0xC0, 0)]


@pytest.mark.parametrize('product', ['old firmware', 'USB 3.2 PCIe TinyEnclosure', None])
def test_probe_rejects_wrong_firmware(monkeypatch, bridge, product):
  monkeypatch.setattr(flash, 'find_chestnut', lambda: ('/dock', ('3801', '0001'), product))
  assert readiness.read_chestnut_state(PRODUCT) is None
  assert not bridge.fds


def test_probe_rejects_missing_device(monkeypatch, bridge):
  monkeypatch.setattr(flash, 'find_chestnut', lambda: (None, None, None))
  assert readiness.read_chestnut_state(PRODUCT) is None
  assert not bridge.fds


def test_probe_rejects_multiple_devices(monkeypatch, bridge):
  monkeypatch.setattr(flash, 'find_chestnut', Mock(side_effect=RuntimeError('expected one chestnut, found 2')))
  assert readiness.read_chestnut_state(PRODUCT) is None
  assert not bridge.fds


@pytest.mark.parametrize('failure', ['link_short', 'supply_short', 'error'])
def test_probe_fails_closed_on_usb_errors(bridge, failure):
  if failure == 'error':
    bridge.error = True
  else:
    bridge.short_request = 0xE4 if failure == 'link_short' else 0xC0
  assert readiness.read_chestnut_state(PRODUCT) is None


def test_probe_timeout_closes_device(bridge, clock):
  assert readiness.read_chestnut_state(PRODUCT, timeout=0) is None
  assert not bridge.transfers


def test_wait_times_out_on_real_power_fault(monkeypatch, bridge, clock):
  monkeypatch.setenv('CHESTNUT_DOCK', 'asm2464')
  bridge.fault = True
  assert not readiness.wait_for_chestnut_ready(PRODUCT, timeout=.25)
  assert clock.now == .25


def test_wait_retries_until_pcie_ready(monkeypatch, clock):
  monkeypatch.setenv('CHESTNUT_DOCK', 'asm2464')
  states = [None, readiness.ChestnutReadiness(0, 0, False, 0), readiness.ChestnutReadiness(0, 0, False, 0x78)]
  probe = Mock(side_effect=states)
  monkeypatch.setattr(readiness, 'read_chestnut_state', probe)
  assert readiness.wait_for_chestnut_ready(PRODUCT, timeout=1.)
  assert probe.call_count == 3
  assert clock.now == .2


def startup_function(path, namespace):
  # Run the production startup decision without loading camera IPC or GPU runtimes.
  # The first top-level while is the camera connection loop, after dock selection.
  main = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
  camera_loop = next(i for i, n in enumerate(main.body) if isinstance(n, ast.While))
  main.body = main.body[:camera_loop] + [ast.Return(value=ast.Name(id='CHESTNUT', ctx=ast.Load()))]
  module = ast.fix_missing_locations(ast.Module(body=[main], type_ignores=[]))
  exec(compile(module, str(path), 'exec'), namespace)
  return namespace['main']


@pytest.mark.parametrize('path', ['openpilot/selfdrive/modeld/modeld.py', 'openpilot/sunnypilot/modeld_v2/modeld.py'])
@pytest.mark.parametrize('fault', [False, True])
def test_model_runners_select_dock_directly(monkeypatch, bridge, clock, path, fault):
  monkeypatch.setenv('CHESTNUT_DOCK', 'asm2464')
  monkeypatch.setenv('HCQDEV_WAIT_TIMEOUT_MS', '1000')
  bridge.fault = fault
  params = Mock()
  namespace = {'os': os, 'cloudlog': Mock(), 'sentry': Mock(), 'setproctitle': Mock(), 'PROCESS_NAME': 'modeld',
               'config_realtime_process': Mock(), 'chestnut_present': lambda: True, 'chestnut_compiled': lambda: True,
               'CHESTNUT_USB_PRODUCT': PRODUCT, 'wait_for_chestnut_ready': readiness.wait_for_chestnut_ready, 'Params': lambda: params}
  assert startup_function(ROOT / path, namespace)() == (not fault)
  params.put_bool.assert_any_call('ChestnutLoading', not fault)
  if fault:
    params.put_bool.assert_any_call('ChestnutActive', False)


def test_uncompiled_stock_model_does_not_probe_usb(monkeypatch, bridge):
  monkeypatch.setenv('CHESTNUT_DOCK', 'asm2464')
  namespace = {'os': os, 'cloudlog': Mock(), 'config_realtime_process': Mock(), 'chestnut_present': lambda: True,
               'chestnut_compiled': lambda: False, 'CHESTNUT_USB_PRODUCT': PRODUCT,
               'wait_for_chestnut_ready': readiness.wait_for_chestnut_ready, 'Params': Mock()}
  assert not startup_function(ROOT / 'openpilot/selfdrive/modeld/modeld.py', namespace)()
  assert not bridge.transfers


@pytest.fixture
def dock_status(tmp_path):
  # Exercise the real alert state machine without importing device hardware or
  # native cereal bindings. Firmware matching and the readiness policy stay real.
  usb_namespace = {}
  usb_path = ROOT / 'openpilot/common/hardware/usb.py'
  exec(compile(usb_path.read_text(), str(usb_path), 'exec'), usb_namespace)
  namespace = {'time': Clock(), 'CHESTNUT_USB_PRODUCT': PRODUCT,
               'is_chestnut_usb_id': usb_namespace['is_chestnut_usb_id'],
               'get_build_metadata': lambda: SimpleNamespace(channel='dev'), 'CHESTNUT_BRANCHES': {},
               'MODELS_DIR': tmp_path, 'chestnut_compiled': lambda: True,
               'CHESTNUT_PCIE_READY': readiness.CHESTNUT_PCIE_READY,
               'chestnut_power_fault': readiness.chestnut_power_fault, 'chestnut_powered': readiness.chestnut_powered}
  path = ROOT / 'openpilot/system/hardware/chestnut/status.py'
  body = [n for n in ast.parse(path.read_text()).body if isinstance(n, (ast.Assign, ast.ClassDef))]
  exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), namespace)
  status = namespace['ChestnutStatus']()
  state = SimpleNamespace(supplyVoltage=0, supplyCurrent=0, supplyFault=False, pcieLtssm=0x78, tempC=40., memoryTempC=40.)
  devices = [{'vendorId': 0x3801, 'productId': 1, 'product': PRODUCT, 'speedMbps': 5000}]
  alerts = {}

  def update(loading=False, active=True):
    def set_alert(name, enabled, message=None):
      alerts[name] = (enabled, message)
    status.update(False, 'GPU', devices, False, loading, active, state, set_alert)

  return SimpleNamespace(status=status, state=state, devices=devices, alerts=alerts, update=update)


def test_diy_status_accepts_missing_sensor_but_detects_link_loss(monkeypatch, dock_status):
  monkeypatch.setenv('CHESTNUT_DOCK', 'asm2464')
  dock_status.update(loading=True, active=None)
  dock_status.update()
  assert dock_status.status.power_seen
  assert not dock_status.alerts['Offroad_ChestnutPcieUnavailable'][0]
  dock_status.state.pcieLtssm = 0
  dock_status.update()
  assert not dock_status.alerts['Offroad_ChestnutPcieUnavailable'][0]
  dock_status.update()
  enabled, message = dock_status.alerts['Offroad_ChestnutPcieUnavailable']
  assert enabled and 'PCIe link is not up' in message
  assert not dock_status.status.power_lost


@pytest.mark.parametrize('mode,fault', [('chestnut', False), ('asm2464', True)])
def test_status_retains_supply_fault_alerts(monkeypatch, dock_status, mode, fault):
  monkeypatch.setenv('CHESTNUT_DOCK', mode)
  dock_status.state.supplyFault = fault
  dock_status.update(loading=True, active=None)
  dock_status.update(active=False)
  assert dock_status.status.power_lost
  assert dock_status.alerts['Offroad_ChestnutPcieUnavailable'][0]


def test_diy_status_retains_usb_disconnect_alert(monkeypatch, dock_status):
  monkeypatch.setenv('CHESTNUT_DOCK', 'asm2464')
  dock_status.update(loading=True, active=None)
  dock_status.update()
  dock_status.devices.clear()
  dock_status.update()
  assert dock_status.alerts['Offroad_ChestnutNotDetected'][0]

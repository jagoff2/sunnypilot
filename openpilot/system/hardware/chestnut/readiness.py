from dataclasses import dataclass
import math
import os
import struct
import time


CHESTNUT_POWERED_VOLTAGE = 5000
CHESTNUT_PCIE_READY = 0x78


@dataclass
class ChestnutReadiness:
  supplyVoltage: int
  supplyCurrent: int
  supplyFault: bool
  pcieLtssm: int


def chestnut_power_fault(state) -> bool:
  # DIY ASM2464 docks share Chestnut's USB identity, but have no INA supply sensor.
  # Only the explicitly selected DIY mode may accept the observed all-zero reply.
  no_sensor = (os.getenv("CHESTNUT_DOCK", "chestnut") == "asm2464" and
               state.supplyVoltage == 0 and state.supplyCurrent == 0 and not state.supplyFault)
  return state.supplyFault or (state.supplyVoltage < CHESTNUT_POWERED_VOLTAGE and not no_sensor)


def chestnut_powered(state) -> bool:
  return not chestnut_power_fault(state) and (state.supplyVoltage >= CHESTNUT_POWERED_VOLTAGE or state.pcieLtssm == CHESTNUT_PCIE_READY)


def chestnut_ready(state) -> bool:
  return chestnut_powered(state) and state.pcieLtssm == CHESTNUT_PCIE_READY


def read_chestnut_state(expected_product: str, timeout: float = 4.) -> ChestnutReadiness | None:
  # Reuse the USB control transport, without invoking any flash/reset operations.
  # Keep this Linux-only import out of the platform-independent power policy.
  from openpilot.system.hardware.chestnut import flash

  deadline = time.monotonic() + timeout
  fd = None
  try:
    path, _, product = flash.find_chestnut()
    if path is None or product != expected_product:
      return None
    fd = flash.open_device(path)

    def control(request_type: int, request: int, value: int, length: int) -> bytes:
      remaining = deadline - time.monotonic()
      if remaining <= 0:
        raise TimeoutError("Chestnut readiness probe timed out")
      buf = (flash.ctypes.c_ubyte * length)()
      transfer_timeout = max(1, min(2000, math.ceil(remaining * 1000)))
      transferred = flash.fcntl.ioctl(fd, flash.USBDEVFS_CONTROL,
                                     flash.Ctrl(request_type, request, value, 0, length, transfer_timeout,
                                                flash.ctypes.cast(buf, flash.ctypes.c_void_p)))
      if transferred != length:
        raise OSError("Short Chestnut readiness response")
      return bytes(buf)

    ltssm = control(0xC0, 0xE4, 0xB450, 1)[0]
    if ltssm != CHESTNUT_PCIE_READY:
      # The custom firmware can boot with PCIe off. Do not disturb a live link.
      control(0x40, 0xF3, 1, 0)
      ltssm = control(0xC0, 0xE4, 0xB450, 1)[0]
    voltage, current, fault = struct.unpack('<Hh?', control(0xC0, 0xC0, 0, 5))
    return ChestnutReadiness(voltage, current, fault, ltssm)
  except (OSError, RuntimeError):
    return None
  finally:
    if fd is not None:
      os.close(fd)


def wait_for_chestnut_ready(expected_product: str, timeout: float = 4.) -> bool:
  # modeld is the publisher of chestnutState, so startup must probe USB directly.
  deadline = time.monotonic() + timeout
  while (remaining := deadline - time.monotonic()) > 0:
    state = read_chestnut_state(expected_product, timeout=remaining)
    if state is not None and chestnut_ready(state):
      return True
    time.sleep(max(0., min(.1, deadline - time.monotonic())))
  return False

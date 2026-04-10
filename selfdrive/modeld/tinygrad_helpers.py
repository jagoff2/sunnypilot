import json
import os
import shlex
from pathlib import Path
from typing import Iterable

MODELS_DIR = Path(__file__).parent / 'models'
COMPILED_FLAGS_PATH = MODELS_DIR / 'tg_compiled_flags.json'
USB_GPU_DEV = 'USB+AMD:LLVM'
RUNTIME_ENV_KEYS = ('DEV',)
DEPRECATED_ENV_KEYS = ('AMD_IFACE',)


def _normalize_available_devices(available_devices: Iterable[str] | None) -> set[str]:
  return {device.split(':', 1)[0].upper() for device in available_devices or ()}


def _split_dev_target(dev_value: str) -> tuple[str, str]:
  target = dev_value.split(':', 1)[0].upper()
  if '+' in target:
    interface, device = target.split('+', 1)
    return interface, device
  return '', target


def _with_interface(dev_value: str, interface: str) -> str:
  target, *rest = dev_value.split(':', 1)
  if '+' in target or not target:
    return dev_value
  suffix = f":{rest[0]}" if rest else ""
  return f"{interface}+{target}{suffix}"


def _canonicalize_usb_gpu_dev(dev_value: str) -> str:
  interface, device = _split_dev_target(dev_value)
  if interface == 'USB' and device == 'AMD' and ':' not in dev_value:
    return USB_GPU_DEV
  return dev_value


def normalize_tinygrad_runtime_env(env_vars: dict[str, str] | None = None) -> dict[str, str]:
  normalized = {str(key): str(value) for key, value in (env_vars or {}).items()}
  dev_value = normalized.get('DEV', '')
  _, device = _split_dev_target(dev_value)

  if normalized.get('AMD_IFACE', '').upper() == 'USB' and device == 'AMD':
    normalized['DEV'] = _with_interface(dev_value, 'USB')

  if 'DEV' in normalized:
    normalized['DEV'] = _canonicalize_usb_gpu_dev(normalized['DEV'])

  for key in DEPRECATED_ENV_KEYS:
    normalized.pop(key, None)

  return normalized


def wants_usb_gpu(use_usb_gpu: bool | None = None) -> bool:
  if use_usb_gpu is not None:
    return use_usb_gpu
  interface, device = _split_dev_target(os.getenv('DEV', ''))
  return "USBGPU" in os.environ or os.getenv("AMD_IFACE", "").upper() == "USB" or (interface == 'USB' and device == 'AMD')


def get_tinygrad_runtime_env(arch: str | None = None, available_devices: Iterable[str] | None = None,
                             use_usb_gpu: bool | None = None, default_to_qcom: bool = False) -> dict[str, str]:
  if wants_usb_gpu(use_usb_gpu):
    return {'DEV': USB_GPU_DEV}

  available = _normalize_available_devices(available_devices)
  if 'CUDA' in available:
    return {'DEV': 'CUDA'}
  if 'QCOM' in available or default_to_qcom:
    return {'DEV': 'QCOM'}
  return {'DEV': 'CPU' if arch == 'Darwin' else 'CPU:LLVM'}


def get_tinygrad_compile_env(arch: str | None = None, available_devices: Iterable[str] | None = None,
                             use_usb_gpu: bool | None = None) -> dict[str, str]:
  compile_env = dict(get_tinygrad_runtime_env(arch, available_devices, use_usb_gpu))
  _, backend = _split_dev_target(compile_env['DEV'])

  if backend == 'QCOM':
    compile_env.update({'FLOAT16': '1', 'NOLOCALS': '1', 'JIT_BATCH_SIZE': '0'})
  elif backend == 'CPU':
    compile_env['THREADS'] = '0'

  return compile_env


def get_tinygrad_compile_image_flag(arch: str) -> str:
  return 'IMAGE=2' if arch == 'larch64' else 'IMAGE=0'


def get_tinygrad_available_devices(use_usb_gpu: bool | None = None) -> set[str]:
  if wants_usb_gpu(use_usb_gpu):
    return set()
  from tinygrad import Device
  return _normalize_available_devices(Device.get_available_devices())


def format_tinygrad_env(env_vars: dict[str, str], include_home: bool = False) -> str:
  command_env = normalize_tinygrad_runtime_env(env_vars)
  if include_home:
    command_env['HOME'] = os.path.expanduser("~")
  for key in DEPRECATED_ENV_KEYS:
    command_env.setdefault(key, '')
  return " ".join(f"{key}={shlex.quote(str(value))}" for key, value in command_env.items())


def get_runtime_flags_from_compile_env(env_vars: dict[str, str]) -> dict[str, str]:
  normalized = normalize_tinygrad_runtime_env(env_vars)
  return {key: str(value) for key, value in normalized.items() if key in RUNTIME_ENV_KEYS}


def write_tinygrad_compiled_flags(path: Path, env_vars: dict[str, str]) -> None:
  with open(path, "w") as f:
    json.dump(get_runtime_flags_from_compile_env(env_vars), f)
    f.write("\n")


def read_tinygrad_compiled_flags(path: Path = COMPILED_FLAGS_PATH) -> dict[str, str]:
  with open(path) as f:
    data = json.load(f)
  return normalize_tinygrad_runtime_env({str(key): str(value) for key, value in data.items()})


def set_tinygrad_backend_from_compiled_flags(path: Path = COMPILED_FLAGS_PATH,
                                             fallback_env: dict[str, str] | None = None) -> dict[str, str]:
  env_vars = read_tinygrad_compiled_flags(path) if path.exists() else normalize_tinygrad_runtime_env(dict(fallback_env or {}))
  for key in DEPRECATED_ENV_KEYS:
    os.environ.pop(key, None)
  for key, value in env_vars.items():
    os.environ[key] = value
  return env_vars


def is_usb_gpu_backend(env_vars: dict[str, str] | None = None) -> bool:
  backend_env = normalize_tinygrad_runtime_env(env_vars or os.environ)
  interface, device = _split_dev_target(backend_env.get('DEV', ''))
  return interface == 'USB' and device == 'AMD'

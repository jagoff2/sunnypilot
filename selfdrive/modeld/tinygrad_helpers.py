import json
import os
import shlex
from pathlib import Path
from typing import Iterable

MODELS_DIR = Path(__file__).parent / 'models'
COMPILED_FLAGS_PATH = MODELS_DIR / 'tg_compiled_flags.json'
RUNTIME_ENV_KEYS = ('DEV', 'AMD_IFACE')


def _normalize_available_devices(available_devices: Iterable[str] | None) -> set[str]:
  return {device.split(':', 1)[0].upper() for device in available_devices or ()}


def wants_usb_gpu(use_usb_gpu: bool | None = None) -> bool:
  if use_usb_gpu is not None:
    return use_usb_gpu
  return "USBGPU" in os.environ or os.getenv("AMD_IFACE", "").upper() == "USB"


def get_tinygrad_runtime_env(arch: str | None = None, available_devices: Iterable[str] | None = None,
                             use_usb_gpu: bool | None = None, default_to_qcom: bool = False) -> dict[str, str]:
  if wants_usb_gpu(use_usb_gpu):
    return {'DEV': 'AMD', 'AMD_IFACE': 'USB'}

  available = _normalize_available_devices(available_devices)
  if 'CUDA' in available:
    return {'DEV': 'CUDA'}
  if 'QCOM' in available or default_to_qcom:
    return {'DEV': 'QCOM'}
  return {'DEV': 'CPU' if arch == 'Darwin' else 'CPU:LLVM'}


def get_tinygrad_compile_env(arch: str | None = None, available_devices: Iterable[str] | None = None,
                             use_usb_gpu: bool | None = None) -> dict[str, str]:
  compile_env = dict(get_tinygrad_runtime_env(arch, available_devices, use_usb_gpu))
  backend = compile_env['DEV'].split(':', 1)[0]

  if backend == 'QCOM':
    compile_env.update({'FLOAT16': '1', 'NOLOCALS': '1', 'JIT_BATCH_SIZE': '0'})
  elif backend == 'CPU':
    compile_env['THREADS'] = '0'

  return compile_env


def get_tinygrad_compile_image_flag(arch: str) -> str:
  return 'IMAGE=2' if arch == 'larch64' else 'IMAGE=0'


def get_tinygrad_available_devices() -> set[str]:
  from tinygrad import Device
  return _normalize_available_devices(Device.get_available_devices())


def format_tinygrad_env(env_vars: dict[str, str], include_home: bool = False) -> str:
  command_env = dict(env_vars)
  if include_home:
    command_env['HOME'] = os.path.expanduser("~")
  return " ".join(f"{key}={shlex.quote(str(value))}" for key, value in command_env.items())


def get_runtime_flags_from_compile_env(env_vars: dict[str, str]) -> dict[str, str]:
  return {key: str(value) for key, value in env_vars.items() if key in RUNTIME_ENV_KEYS}


def write_tinygrad_compiled_flags(path: Path, env_vars: dict[str, str]) -> None:
  with open(path, "w") as f:
    json.dump(get_runtime_flags_from_compile_env(env_vars), f)
    f.write("\n")


def read_tinygrad_compiled_flags(path: Path = COMPILED_FLAGS_PATH) -> dict[str, str]:
  with open(path) as f:
    data = json.load(f)
  return {str(key): str(value) for key, value in data.items()}


def set_tinygrad_backend_from_compiled_flags(path: Path = COMPILED_FLAGS_PATH,
                                             fallback_env: dict[str, str] | None = None) -> dict[str, str]:
  env_vars = read_tinygrad_compiled_flags(path) if path.exists() else dict(fallback_env or {})
  for key, value in env_vars.items():
    os.environ[key] = value
  return env_vars


def is_usb_gpu_backend(env_vars: dict[str, str] | None = None) -> bool:
  backend_env = env_vars or os.environ
  return backend_env.get('DEV', '').split(':', 1)[0] == 'AMD' and backend_env.get('AMD_IFACE', '').upper() == 'USB'

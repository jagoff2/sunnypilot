"""Keep QCOM inference live while a replaceable process recovers the eGPU."""
import logging
import time

from openpilot.system.hardware.chestnut.hotplug import DockRetry
from openpilot.system.hardware.chestnut.worker import GpuWorker


logger = logging.getLogger(__name__)
FRAME_BUDGET = .045


def mapped_inputs(model, frames, transforms, inputs):
  bufs = {key: frames['wide' if 'big' in key else 'road'] for key in model.vision_input_names}
  tfms = {key: transforms['wide' if 'big' in key else 'road'] for key in model.vision_input_names}
  values = {k: v.copy() for k, v in inputs.items()}
  desire = next((values[k] for k in values if k.startswith('desire')), None)
  if desire is not None:
    values[getattr(model, 'desire_key', 'desire_pulse')] = desire
  return bufs, tfms, values


class RecoveringModel:
  def __init__(self, fallback, runner: str, cam_w: int, cam_h: int, frame_bytes: int, params,
               available, worker_factory=GpuWorker):
    self.fallback = fallback
    self.runner = runner
    self.config = {'runner': runner, 'cam_w': cam_w, 'cam_h': cam_h}
    self.frame_bytes = frame_bytes
    self.params = params
    self.available = available
    self.worker_factory = worker_factory
    self.worker = None
    self.host = None
    self.retry = DockRetry()
    self.chestnut = False
    self.live_frames = 0
    self.good_frames = 0
    self.required_frames = 100
    self.sequence = 0
    self._status = None
    self._lat_delay = fallback.lat_delay
    self._planplus = getattr(fallback, 'PLANPLUS_CONTROL', 1.)
    self._set_status(False, False)

  def __getattr__(self, key):
    return getattr(self.host if self.chestnut else self.fallback, key)

  @property
  def lat_delay(self):
    return self._lat_delay

  @lat_delay.setter
  def lat_delay(self, value):
    self._lat_delay = value
    self.fallback.lat_delay = value
    if self.host is not None:
      self.host.lat_delay = value

  @property
  def PLANPLUS_CONTROL(self):
    return self._planplus

  @PLANPLUS_CONTROL.setter
  def PLANPLUS_CONTROL(self, value):
    self._planplus = value
    self.fallback.PLANPLUS_CONTROL = value
    if self.host is not None:
      self.host.PLANPLUS_CONTROL = value

  def _set_status(self, loading, active):
    self.chestnut = active
    if self._status != (loading, active):
      self.params.put_bool('ChestnutLoading', loading)
      self.params.put_bool('ChestnutActive', active)
      if active:
        self.params.remove('ChestnutModelError')
      self._status = loading, active

  def _discard(self, error=None):
    if self.worker is not None:
      self.worker.close()
      self.worker = None
    self.host = None
    self.live_frames = 0
    self.good_frames = 0
    self._set_status(False, False)
    if error is not None:
      logger.warning('GPU recovery attempt failed: %s', error)
      self.params.put_bool('ChestnutModelError', True)
      self.retry.attempted(time.monotonic())

  def observe(self, device):
    now = time.monotonic()
    if device != self.retry.device:
      self._discard()
    self.retry.observe(device, now)
    if self.worker is None and self.retry.due(now):
      # Check compiled files only at retry intervals, never block on USB here.
      if self.available():
        try:
          self.worker = self.worker_factory(self.frame_bytes, self.config,
                                             'openpilot.system.hardware.chestnut.inference:create_model_session')
          self._set_status(True, False)
        except Exception as e:
          self._discard(e)
      else:
        self.retry.attempted(now)

  def run(self, bufs, transforms, inputs, prepare_only=False):
    deadline = time.monotonic() + FRAME_BUDGET
    self.sequence += 1
    sequence = self.sequence
    frames = {side: next(v for k, v in bufs.items() if ('big' in k) == (side == 'wide')) for side in ('road', 'wide')}
    tfms = {side: next(v for k, v in transforms.items() if ('big' in k) == (side == 'wide')) for side in ('road', 'wide')}
    submitted = False
    if self.worker is not None:
      try:
        # A late result belongs to a previous camera frame and is discarded.
        late = self.worker.poll()
        if late is not None:
          self.live_frames += 1
          self.good_frames = 0
        if self.worker.metadata is not None and self.host is None:
          self.host = object.__new__(type(self.fallback))
          self.host.__dict__.update(self.worker.metadata['host'])
          self.host.lat_delay = self.lat_delay
          self.host.PLANPLUS_CONTROL = self.PLANPLUS_CONTROL
          self.required_frames = self.worker.metadata['history_frames']
        request = {'transforms': tfms, 'inputs': inputs, 'prepare_only': prepare_only,
                   'lat_delay': self.lat_delay, 'planplus': self.PLANPLUS_CONTROL}
        submitted = self.worker.submit(sequence, frames, request)
      except Exception as e:
        self._discard(e)

    # Run every frame, even while the GPU is healthy, keeping fallback recurrent
    # history hot. The GPU gets a private input copy before either model mutates it.
    small_bufs, small_tfms, small_inputs = mapped_inputs(self.fallback, frames, tfms, inputs)
    if self.runner == 'stock':
      fallback_output = self.fallback.run(small_bufs, small_tfms, small_inputs)
    else:
      fallback_output = self.fallback.run(small_bufs, small_tfms, small_inputs, prepare_only)
    if self.worker is not None and submitted:
      try:
        result = self.worker.poll(max(0., deadline - time.monotonic()))
        if result is not None and result[0] == sequence and result[1] is not None:
          self.live_frames += 1
          self.good_frames += 1
          if self.live_frames >= self.required_frames and self.good_frames >= 3:
            self.retry.attempts = 0
            self._set_status(False, True)
            return result[1]
        else:
          self.good_frames = 0
      except Exception as e:
        self._discard(e)
    self._set_status(self.worker is not None, False)
    return fallback_output


def create_model_session(config):
  import os
  import numpy as np
  from openpilot.common.hardware.usb import CHESTNUT_USB_PRODUCT
  from openpilot.system.hardware.chestnut.readiness import wait_for_chestnut_ready
  from openpilot.cereal.messaging import PubMaster
  from openpilot.selfdrive.modeld.modeld import ChestnutState

  if not wait_for_chestnut_ready(CHESTNUT_USB_PRODUCT):
    raise RuntimeError('USB/PCIe is not ready')
  os.environ['HCQDEV_WAIT_TIMEOUT_MS'] = '3000'
  if config['runner'] == 'stock':
    from openpilot.selfdrive.modeld.modeld import ModelState
  else:
    from openpilot.sunnypilot.modeld_v2.modeld import ModelState
  model = ModelState(config['cam_w'], config['cam_h'], True)
  model.warmup()
  # Loading yields to controls on CPUs 4/5. Once ready, use the inference core
  # below the publisher's priority, so GPU submission cannot starve fallback.
  from openpilot.common.realtime import config_realtime_process
  config_realtime_process(7, 53)
  fields = ('vision_input_names', '_vision_input_names', '_desire_key', 'generation', 'constants',
            'LAT_SMOOTH_SECONDS', 'LONG_SMOOTH_SECONDS', 'MIN_LAT_CONTROL_SPEED', 'PLANPLUS_CONTROL')
  host = {k: model.__dict__[k] for k in fields if k in model.__dict__}
  host['chestnut'] = True
  if hasattr(model, 'numpy_inputs'):
    host['numpy_inputs'] = dict.fromkeys(model.numpy_inputs)
  history_frames = max((int(q.shape[0]) for k, q in model.input_queues.items() if k in ('feat_q', 'img_q', 'big_img_q', 'desire_q')), default=100)

  class Session:
    metadata = {'host': host, 'history_frames': max(3, history_frames)}

    def __init__(self):
      self.state = ChestnutState(PubMaster(['chestnutState']), True)
      self.frames = 0

    def run(self, frames, request):
      bufs, tfms, inputs = mapped_inputs(model, frames, request['transforms'], request['inputs'])
      model.lat_delay = request['lat_delay']
      model.PLANPLUS_CONTROL = request['planplus']
      self.frames += 1
      after_enqueue = self.state.send if self.frames % 2 == 0 else None
      if config['runner'] == 'stock':
        result = model.run(bufs, tfms, inputs, after_enqueue)
      else:
        result = model.run(bufs, tfms, inputs, request['prepare_only'], after_enqueue)
      if result is not None and any(not np.all(np.isfinite(value)) for value in result.values()):
        raise RuntimeError('GPU model returned non-finite output')
      return result

  return Session()

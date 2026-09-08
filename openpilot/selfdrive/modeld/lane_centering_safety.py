"""Keep command usability separate from advisory lane-containment evidence."""
import math
import time
from enum import IntEnum


LANE_CENTERING_STATUS_VERSION = 2
MAX_MODEL_AGE_NS = 300_000_000
MAX_FUTURE_MODEL_NS = 50_000_000

class PlannerSeverity(IntEnum):
  NONE = 0
  COLLISION_WARNING = 1
  FATAL = 2


def model_clock_ns():
  # Camera EOF uses nanos_since_boot/CLOCK_BOOTTIME, which includes suspend.
  # Python's monotonic clock and Python-produced logMonoTime do not on Linux.
  return time.clock_gettime_ns(time.CLOCK_BOOTTIME) if hasattr(time, 'CLOCK_BOOTTIME') else time.monotonic_ns()


class LaneCenteringSafetyLatch:
  def __init__(self):
    self.last_timestamp_eof = None
    self.latched_reason = None
    self.reason = None
    self.packet_reason = None
    self.diagnostic_reason = None
    self.severity = PlannerSeverity.NONE
    self.last_usable_model = None
    self.selected_model = None

  @property
  def collision_risk(self):
    return self.severity == PlannerSeverity.COLLISION_WARNING

  def update(self, model, model_healthy, model_updated, now_ns, engagement_requested):
    # Engagement comes from selfdrived/MADS enabled, not carControl.latActive:
    # removing steering authority must never clear its own safety latch.
    if not engagement_requested:
      self.latched_reason = None
    self.packet_reason = self._invalid_reason(model, model_healthy, model_updated, now_ns)
    if self.packet_reason is None and (self.last_timestamp_eof is None or model.timestampEof > self.last_timestamp_eof):
      # SubMaster replaces its immutable reader each update. Keeping this reader
      # retains the complete matching trajectory/action for controller preview.
      self.last_usable_model = model
      self.last_timestamp_eof = model.timestampEof
    cached = self.last_usable_model
    cache_fresh = cached is not None and -MAX_FUTURE_MODEL_NS <= now_ns - cached.timestampEof <= MAX_MODEL_AGE_NS
    self.selected_model = cached if cache_fresh else None
    reason = None if cache_fresh else (self.packet_reason or 'no_usable_model')
    if engagement_requested and reason is not None:
      self.latched_reason = self.latched_reason or reason
    self.reason = self.latched_reason or reason
    self.diagnostic_reason = self._status_reason(model)
    self.severity = PlannerSeverity.FATAL if self.reason is not None else PlannerSeverity.NONE
    if self.severity == PlannerSeverity.NONE and self._status_reason(cached) is None and cached.laneCentering.collisionRisk:
      self.severity = PlannerSeverity.COLLISION_WARNING
    return self.severity == PlannerSeverity.FATAL

  @staticmethod
  def _status_reason(model):
    # This extension cannot make a fresh, usable native steering command fatal.
    # Only a valid, matching producer may assert a collision warning.
    status = getattr(model, 'laneCentering', None)
    if status is None or status.version != LANE_CENTERING_STATUS_VERSION:
      return 'missing_status'
    if not status.valid:
      return 'invalid_status'
    if status.timestampEof != model.timestampEof or status.frameId != model.frameId:
      return 'status_frame_mismatch'
    return None

  def _invalid_reason(self, model, healthy, updated, now_ns):
    if not healthy:
      return 'model_unhealthy'
    if model.timestampEof <= 0 or now_ns <= 0:
      return 'invalid_timestamp'
    previous = self.last_timestamp_eof
    if previous is not None and (model.timestampEof < previous or (updated and model.timestampEof == previous)):
      return 'nonmonotonic_model'
    age_ns = now_ns - model.timestampEof
    if age_ns > MAX_MODEL_AGE_NS or age_ns < -MAX_FUTURE_MODEL_NS:
      return 'stale_model'
    if not all(math.isfinite(value) for value in (model.action.desiredCurvature, model.action.desiredAcceleration)):
      return 'nonfinite_action'
    return None

"""The selected model action and its containment decision are one contract."""
import math
import time


LANE_CENTERING_STATUS_VERSION = 1
MAX_MODEL_AGE_NS = 300_000_000
MAX_FUTURE_MODEL_NS = 50_000_000
CONTAINMENT_STATES = ('unavailable', 'contained', 'recovering', 'blocked', 'bypassed')
BYPASS_REASONS = ('mode_off', 'lane_change', 'low_speed')


def model_clock_ns():
  # Camera EOF uses nanos_since_boot/CLOCK_BOOTTIME, which includes suspend.
  # Python's monotonic clock and Python-produced logMonoTime do not on Linux.
  return time.clock_gettime_ns(time.CLOCK_BOOTTIME) if hasattr(time, 'CLOCK_BOOTTIME') else time.monotonic_ns()


class LaneCenteringSafetyLatch:
  def __init__(self):
    self.last_timestamp_eof = None
    self.latched_reason = None
    self.reason = None

  def update(self, model, model_healthy, model_updated, now_ns, engagement_requested):
    # Engagement comes from selfdrived/MADS enabled, not carControl.latActive:
    # removing steering authority must never clear its own safety latch.
    if not engagement_requested:
      self.latched_reason = None
    reason = self._invalid_reason(model, model_healthy, model_updated, now_ns)
    if engagement_requested and reason is not None:
      self.latched_reason = self.latched_reason or reason
    self.reason = self.latched_reason or reason
    return self.reason is not None

  def _invalid_reason(self, model, healthy, updated, now_ns):
    if not healthy:
      return 'model_unhealthy'
    status = model.laneCentering
    if status.version != LANE_CENTERING_STATUS_VERSION:
      return 'missing_status'
    if not status.valid:
      return 'invalid_status'
    if status.timestampEof != model.timestampEof or status.frameId != model.frameId:
      return 'status_frame_mismatch'
    if model.timestampEof <= 0 or now_ns <= 0:
      return 'invalid_timestamp'
    if updated:
      previous = self.last_timestamp_eof
      self.last_timestamp_eof = max(model.timestampEof, previous or 0)
      if previous is not None and model.timestampEof <= previous:
        return 'nonmonotonic_model'
    age_ns = now_ns - model.timestampEof
    if age_ns > MAX_MODEL_AGE_NS or age_ns < -MAX_FUTURE_MODEL_NS:
      return 'stale_model'
    if not all(math.isfinite(value) for value in (model.action.desiredCurvature, model.action.desiredAcceleration)):
      return 'nonfinite_action'
    if status.containment not in CONTAINMENT_STATES:
      return 'invalid_containment_status'
    if status.containment == 'bypassed' and status.reason not in BYPASS_REASONS:
      return 'invalid_bypass_reason'
    if status.safetyBlocked or status.containment == 'blocked':
      return 'containment_blocked'
    if status.containment in ('contained', 'recovering') and (
      not math.isfinite(status.minClearance) or not math.isfinite(status.checkedDistance) or status.checkedDistance <= 0.0
    ):
      return 'invalid_containment_geometry'
    return None

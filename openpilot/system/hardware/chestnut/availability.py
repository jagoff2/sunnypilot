"""Driving alerts follow usable model output, independent of GPU retry state."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelAlerts:
  loading: bool
  failed: bool
  ready: bool


class ModelAvailability:
  def __init__(self):
    self.was_big = False
    self.ready_since: float | None = None
    self.ready_announced = False

  def update(self, *, now: float, healthy: bool, big: bool, loading: bool) -> ModelAlerts:
    # Retain failure protection if the last usable GPU output stops being
    # healthy. Fresh small-model output clears it immediately, even onroad.
    failed = self.was_big and not healthy
    if healthy:
      self.was_big = big

    if healthy and big:
      if self.ready_since is None:
        self.ready_since = now
    else:
      self.ready_since = None

    # One announcement per drive, after two seconds of healthy big output.
    # A loading flag clearing can also mean a failed or cancelled attempt.
    ready = not self.ready_announced and self.ready_since is not None and now - self.ready_since >= 2.
    self.ready_announced |= ready
    return ModelAlerts(loading=loading and not healthy, failed=failed, ready=ready)

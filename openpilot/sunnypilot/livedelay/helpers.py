"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math

from openpilot.common.params import Params

# Shared with lagd: bounds for its learned estimate and its startup allowance.
MIN_LAT_DELAY = 0.15
MAX_LAT_DELAY = 0.65
DEFAULT_SOFTWARE_DELAY = 0.2
DEFAULT_LAT_DELAY = 0.3


def get_initial_lat_delay(steer_actuator_delay: float) -> float:
  try:
    delay = float(steer_actuator_delay)
  except (TypeError, ValueError, OverflowError):
    return DEFAULT_LAT_DELAY
  return delay + DEFAULT_SOFTWARE_DELAY if math.isfinite(delay) and delay >= 0.0 else DEFAULT_LAT_DELAY


def _validated_delay(value, upper_bound: float) -> float | None:
  try:
    delay = float(value)
  except (TypeError, ValueError, OverflowError):
    return None
  return delay if math.isfinite(delay) and MIN_LAT_DELAY <= delay <= upper_bound else None


def get_lat_delay(params: Params, stock_lat_delay: float, fallback_lat_delay: float = DEFAULT_LAT_DELAY) -> float:
  # Match the c3x delay source used by modeld and the torque-controller lookahead.
  # An unestimated lagd output is the vehicle's initial delay, which can be above
  # the learned range. Preserve that CP-derived value instead of clamping it.
  fallback = _validated_delay(fallback_lat_delay, math.inf)
  if fallback is None:
    fallback = DEFAULT_LAT_DELAY
  upper_bound = max(MAX_LAT_DELAY, fallback)
  if params.get_bool("LagdToggle"):
    try:
      cached_value = params.get("LagdValueCache")
    except (TypeError, ValueError, OverflowError):
      cached_value = None
    cached = _validated_delay(cached_value, upper_bound)
    if cached is not None:
      return cached

  published = _validated_delay(stock_lat_delay, upper_bound)
  return published if published is not None else fallback

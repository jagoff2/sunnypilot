import unittest

from openpilot.sunnypilot.livedelay.helpers import get_initial_lat_delay, get_lat_delay, MIN_LAT_DELAY, MAX_LAT_DELAY
from openpilot.sunnypilot.modeld_v2.modeld_base import ModelStateBase


class DelayParams:
  def __init__(self, enabled, cached):
    self.enabled = enabled
    self.cached = cached

  def get_bool(self, key):
    assert key == "LagdToggle"
    return self.enabled

  def get(self, key, return_default=False):
    assert key == "LagdValueCache" and not return_default
    return self.cached


class TestC3XDelaySelection(unittest.TestCase):
  def test_learning_uses_cached_delay_for_model_and_controller_lookahead(self):
    params = DelayParams(True, 0.3521042764186859)
    self.assertEqual(get_lat_delay(params, 0.4), params.cached)
    params.cached = 0.37
    self.assertEqual(get_lat_delay(params, 0.41), 0.37)

  def test_disabled_uses_published_delay_as_on_c3x(self):
    self.assertEqual(get_lat_delay(DelayParams(False, 0.2), 0.4), 0.4)

  def test_recorded_delay_is_preserved_exactly(self):
    delay = 0.28442391753196716
    for value in (delay, str(delay), str(delay).encode()):
      with self.subTest(value=value):
        self.assertEqual(get_lat_delay(DelayParams(True, value), 0.4), delay)

  def test_invalid_cache_uses_published_estimate(self):
    for cached in (None, "", b"", "invalid", float("nan"), float("inf"), -float("inf"), 0.0, -0.2, 0.149, 0.651, 100):
      with self.subTest(cached=cached):
        self.assertEqual(get_lat_delay(DelayParams(True, cached), 0.41), 0.41)

  def test_invalid_values_use_vehicle_initial_delay(self):
    for enabled in (False, True):
      for published in (None, b"", "bad", float("nan"), float("inf"), 0.0, -1.0, 10.0):
        with self.subTest(enabled=enabled, published=published):
          self.assertEqual(get_lat_delay(DelayParams(enabled, "bad"), published, 0.35), 0.35)
    self.assertEqual(get_lat_delay(DelayParams(True, "bad"), float("nan"), float("inf")), 0.3)

  def test_typed_params_decode_failure_uses_published_delay(self):
    class MalformedParams(DelayParams):
      def get(self, key, return_default=False):
        raise ValueError("malformed float parameter")

    self.assertEqual(get_lat_delay(MalformedParams(True, None), 0.42), 0.42)
    # A disabled learning toggle must not read the cached value.
    self.assertEqual(get_lat_delay(MalformedParams(False, None), 0.43), 0.43)

  def test_estimator_bounds_and_longer_vehicle_initial_delay(self):
    for delay in (MIN_LAT_DELAY, MAX_LAT_DELAY):
      self.assertEqual(get_lat_delay(DelayParams(True, delay), 0.4), delay)
    initial = get_initial_lat_delay(0.6)
    self.assertEqual(initial, 0.8)
    self.assertEqual(get_lat_delay(DelayParams(True, 0.8), 0.0, initial), 0.8)
    self.assertEqual(get_lat_delay(DelayParams(True, 20.0), 0.8, initial), 0.8)

  def test_model_base_initialization_respects_toggle_and_sanitizes_cache(self):
    self.assertEqual(ModelStateBase(DelayParams(True, 0.28442391753196716), 0.3).lat_delay, 0.28442391753196716)
    self.assertEqual(ModelStateBase(DelayParams(False, 0.2), 0.4).lat_delay, 0.4)
    self.assertEqual(ModelStateBase(DelayParams(True, "bad"), 0.35).lat_delay, 0.35)
    self.assertEqual(ModelStateBase(DelayParams(True, None), 0.35).lat_delay, 0.35)

  def test_invalid_actuator_delay_uses_estimator_default(self):
    for delay in (None, "bad", float("nan"), float("inf"), -0.1):
      with self.subTest(delay=delay):
        self.assertEqual(get_initial_lat_delay(delay), 0.3)


if __name__ == "__main__":
  unittest.main()

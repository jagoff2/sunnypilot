import unittest

from openpilot.sunnypilot.livedelay.helpers import get_lat_delay


class DelayParams:
  def __init__(self, enabled, cached):
    self.enabled = enabled
    self.cached = cached

  def get_bool(self, key):
    assert key == "LagdToggle"
    return self.enabled

  def get(self, key, return_default=False):
    assert key == "LagdValueCache" and return_default
    return self.cached


class TestC3XDelaySelection(unittest.TestCase):
  def test_learning_uses_cached_delay_for_model_and_controller_lookahead(self):
    params = DelayParams(True, 0.3521042764186859)
    self.assertEqual(get_lat_delay(params, 0.4), params.cached)
    params.cached = 0.37
    self.assertEqual(get_lat_delay(params, 0.41), 0.37)

  def test_disabled_uses_published_delay_as_on_c3x(self):
    self.assertEqual(get_lat_delay(DelayParams(False, 0.2), 0.4), 0.4)


if __name__ == "__main__":
  unittest.main()

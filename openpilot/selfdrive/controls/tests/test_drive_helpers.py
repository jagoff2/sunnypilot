import unittest

from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import (MAX_CURVATURE, MAX_LATERAL_ACCEL_NO_ROLL,
                                                          MAX_LATERAL_JERK, clip_curvature)


class TestCurvatureLimits(unittest.TestCase):
  def test_rate_limited_request_reports_limiting_in_both_directions(self):
    for speed in (1., 5., 20., 40.):
      for sign in (-1., 1.):
        with self.subTest(speed=speed, sign=sign):
          step = MAX_LATERAL_JERK * DT_CTRL / speed ** 2
          actual, limited = clip_curvature(speed, 0., sign * step * 2., 0.)
          self.assertAlmostEqual(actual, sign * step)
          self.assertTrue(limited)
          actual, limited = clip_curvature(speed, 0., sign * step * 0.5, 0.)
          self.assertAlmostEqual(actual, sign * step * 0.5)
          self.assertFalse(limited)

  def test_acceleration_and_absolute_limits_still_report(self):
    for sign in (-1., 1.):
      roll, speed = 0.03, 20.
      target = sign * 0.1
      actual, limited = clip_curvature(speed, target, target, roll)
      expected = (sign * MAX_LATERAL_ACCEL_NO_ROLL + roll * ACCELERATION_DUE_TO_GRAVITY) / speed ** 2
      self.assertAlmostEqual(actual, expected)
      self.assertTrue(limited)
      actual, limited = clip_curvature(0., sign, sign, 0.)
      self.assertEqual(actual, sign * MAX_CURVATURE)
      self.assertTrue(limited)


if __name__ == "__main__":
  unittest.main()

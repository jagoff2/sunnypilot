import math
import unittest

from opendbc.can import CANPacker
from opendbc.car import Bus
from opendbc.car.hyundai.carstate import CarState, SteeringRate
from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.values import CAR


class TestHyundaiSteeringRate(unittest.TestCase):
  def test_real_can_and_canfd_carstate(self):
    for car, message, angle_signal, speed_signal in (
      (CAR.HYUNDAI_KONA_EV, "SAS11", "SAS_Angle", "SAS_Speed"),
      (CAR.KIA_EV6, "STEERING_SENSORS", "STEERING_ANGLE", "STEERING_RATE"),
    ):
      with self.subTest(car=car):
        cp = CarInterface.get_non_essential_params(car)
        cp_sp = CarInterface.get_non_essential_params_sp(cp, car)
        state = CarState(cp, cp_sp)
        parsers = state.get_can_parsers(cp, cp_sp)
        pt = parsers[Bus.pt]
        packer = CANPacker(pt.dbc_name)
        state.update(parsers)  # subscribe to lazy CAN signals
        for frame, (angle, expected) in enumerate(((100., 0.), (99., -8.), (98., -8.), (99., 8.), (99., 0.), (-100., -8.), (-99., 8.))):
          pt.update([1_000_000_000 + frame * 10_000_000, [packer.make_can_msg(message, pt.bus, {angle_signal: angle, speed_signal: 8.})]])
          result, _ = state.update(parsers)
          self.assertEqual(pt.vl[message][speed_signal], 8.)  # raw CAN is unsigned
          self.assertEqual(result.steeringRateDeg, expected)
        pt.update([])
        self.assertEqual(state.update(parsers)[0].steeringRateDeg, 0.)

  def test_ordered_batches_reversals_and_unwrapped_angles(self):
    rate = SteeringRate()
    self.assertEqual(rate.update([359., 360., 361.], [8., 8., 8.]), 8.)
    self.assertEqual(rate.update([362., 361.], [8., 12.]), -12.)
    self.assertEqual(rate.update([-359., -360., -361.], [8., 8., 16.]), -16.)
    self.assertEqual(rate.update([-360.], [4.]), 4.)

  def test_unknown_direction_never_reuses_old_sign(self):
    rate = SteeringRate()
    self.assertEqual(rate.update([5.], [8.]), 0.)
    self.assertEqual(rate.update([6.], [8.]), 8.)
    self.assertEqual(rate.update([6.], [8.]), 0.)  # quantized, possibly reversing
    self.assertEqual(rate.update([], []), 0.)
    self.assertEqual(rate.update([7.], [8.]), 0.)  # no derivative across a gap
    self.assertEqual(rate.update([6.], [0.]), 0.)
    self.assertEqual(rate.update([5.], [8.]), -8.)

  def test_invalid_samples_reset_direction(self):
    for angles, speeds in (([math.nan], [8.]), ([math.inf], [8.]), ([0.], [math.nan]), ([0.], [-8.]), ([0., 1.], [8.])):
      with self.subTest(angles=angles, speeds=speeds):
        rate = SteeringRate()
        rate.update([1., 2.], [8., 8.])
        self.assertEqual(rate.update(angles, speeds), 0.)
        self.assertEqual(rate.update([3.], [8.]), 0.)


if __name__ == "__main__":
  unittest.main()

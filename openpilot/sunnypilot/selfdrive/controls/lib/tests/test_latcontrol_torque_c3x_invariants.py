"""Controller invariants and recorded unwind regression, using real PID and NN.

Params and vehicle observations are the test boundary. This is deterministic
controller execution, not an assertion about closed-loop onroad performance.
"""
import json
import math
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from opendbc.car.vehicle_model import VehicleModel
from openpilot.selfdrive.controls.lib.drive_helpers import MAX_LATERAL_JERK
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib.tests.test_latcontrol_torque_c3x import build_controller, ObservationVehicleModel

FIXTURE = Path(__file__).with_name("fixtures") / "c3x_20260912_unwind.json"


def model(accels, rolls=None, pitches=None):
  return SimpleNamespace(acceleration=SimpleNamespace(y=list(accels)),
                         orientation=SimpleNamespace(x=[0.0] * 33 if rolls is None else list(rolls),
                                                     y=[0.0] * 33 if pitches is None else list(pitches)))


def step(controller, desired=1.0, actual=0.0, speed=20.0, delay=0.0, active=True, rate=0.0, applied=None, valid_model=True,
         model_accels=None, pressed=False, limited=False, roll=0.0, a_ego=0.0):
  controller.extension.update_model_v2(model([desired] * 33 if model_accels is None else model_accels) if valid_model else None)
  cs = SimpleNamespace(vEgo=speed, aEgo=a_ego, steeringAngleDeg=math.degrees(actual / speed ** 2 * 160.0),
                       steeringRateDeg=rate, steeringPressed=pressed)
  output = None if applied is None else SimpleNamespace(actuatorsOutput=SimpleNamespace(torque=-applied))
  return controller.update(active, cs, ObservationVehicleModel(), SimpleNamespace(angleOffsetDeg=0.0, roll=roll),
                           limited, desired / speed ** 2, None, False, delay, car_output=output)


class TestC3XTorqueInvariants(unittest.TestCase):
  def test_one_pid_update_and_truthful_torque_domain_logs(self):
    for enabled in (False, True):
      controller = build_controller(enabled)
      with self.subTest(nnlc=enabled), patch.object(controller.pid, "update", wraps=controller.pid.update) as update:
        output, _, state = step(controller, desired=1.2, actual=0.3)
        self.assertEqual(update.call_count, 1)
        self.assertAlmostEqual(state.error, state.p)
        self.assertAlmostEqual(state.preLimitOutput, -(state.p + state.i + state.f), places=6)
        self.assertAlmostEqual(output, state.output)
        self.assertEqual((controller.pid.neg_limit, controller.pid.pos_limit), (-1.0, 1.0))

  def test_final_slew_and_amplitude_through_reversals_modes_and_live_calibration(self):
    for enabled in (False, True):
      controller = build_controller(enabled)
      previous = 0.0
      for frame in range(300):
        if frame == 120:
          controller.update_torque_parameters(4.2, -0.09, 0.12)
        desired = 8.0 if frame < 100 else -8.0 if frame < 200 else 0.0
        output, _, state = step(controller, desired=desired, valid_model=not 145 <= frame < 165)
        with self.subTest(nnlc=enabled, frame=frame):
          self.assertLessEqual(abs(output), 1.0)
          self.assertLessEqual(abs(output - previous) * controller.torque_params.latAccelFactor, MAX_LATERAL_JERK * controller.dt + 1e-9)
          self.assertTrue(np.isfinite([output, state.p, state.i, state.f]).all())
        previous = output

  def test_unwind_feedback_is_not_replaced_by_an_older_tighter_target(self):
    controller = build_controller(False)
    controller.extension.update_lateral_lag(0.284)
    for _ in range(40):
      step(controller, desired=2.6, actual=2.0, delay=0.384)
    _, _, state = step(controller, desired=1.58, actual=1.67, delay=0.384)
    self.assertAlmostEqual(state.feedbackLateralAccel, 1.58)
    self.assertLess(state.p, 0.0)
    self.assertAlmostEqual(state.feedbackDelay, 0.0)
    # Controlsd's physical NN lag is independent of the feedback API argument.
    self.assertAlmostEqual(controller.extension.desired_lat_jerk_time, 0.284)

  def test_entry_and_resume_have_no_measured_derivative_kick_or_stale_history(self):
    for enabled in (False, True):
      controller = build_controller(enabled)
      for _ in range(70):
        step(controller, desired=1.5, actual=0.7)
      controller.pid.i = 0.4
      controller.reset()
      _, _, inactive = step(controller, active=False)
      self.assertFalse(inactive.active)
      self.assertEqual(controller.pid.i, 0.0)
      self.assertEqual(len(controller.extension.lateral_accel_desired_deque), 0)
      resumed = step(controller, desired=0.9, actual=0.9, delay=0.3, rate=-100.0)
      fresh = step(build_controller(enabled), desired=0.9, actual=0.9, delay=0.3, rate=100.0)
      self.assertAlmostEqual(resumed[0], fresh[0])
      self.assertAlmostEqual(resumed[2].p, 0.0)
      self.assertAlmostEqual(resumed[2].actualLateralJerk, 0.0)

  def test_signed_monotonic_feedback_independent_of_measured_jerk(self):
    for direction in (-1.0, 1.0):
      for desired in (0.0, 0.5, 1.6, 3.0):
        previous = math.inf
        for actual in np.linspace(-4.0, 4.0, 41):
          values = []
          for rate in (-200.0, 0.0, 200.0):
            _, _, state = step(build_controller(), desired=direction * desired, actual=actual, rate=rate)
            values.append(state.p)
            self.assertGreaterEqual(state.p * (direction * desired - actual), -1e-8)
          self.assertAlmostEqual(min(values), max(values))
          self.assertLessEqual(values[0], previous + 1e-7)
          previous = values[0]

  def test_pathological_nn_slope_cannot_reverse_or_disable_feedback(self):
    controller = build_controller()
    for slope in (-100.0, 0.0, 100.0, math.nan):
      with self.subTest(slope=slope), patch.object(controller.extension.model, "evaluate", side_effect=lambda values, slope=slope: 3.0 + slope * values[1]):
        error = controller.extension.acceleration_error(20.0, 1.0, 1.2, 1.0, 0.0, [0.0] * 7)
        self.assertTrue(math.isfinite(error))
        self.assertLess(error, 0.0)
        self.assertGreaterEqual(abs(error), 0.5 * 0.2 / controller.torque_params.latAccelFactor - 1e-9)
        self.assertLessEqual(abs(error), 1.5 * 0.2 / controller.torque_params.latAccelFactor + 1e-9)

  def test_final_and_applied_limits_block_integral_growth_but_allow_unwind(self):
    for enabled in (False, True):
      controller = build_controller(enabled)
      for _ in range(60):
        step(controller, desired=2.0, actual=0.0, applied=0.0)
      self.assertAlmostEqual(controller.pid.i, 0.0)
      controller.pid.i = 0.3
      _, _, state = step(controller, desired=-1.0, actual=1.0, applied=0.0, limited=True)
      self.assertLess(controller.pid.i, 0.3)
      self.assertTrue(state.outputLimited)
      self.assertTrue(state.appliedOutputValid)
      self.assertEqual(state.appliedOutput, 0.0)
      # The generic PID's amplitude anti-windup must not retain an old positive
      # I when a large opposite demand already saturates the negative output.
      controller.pid.i = 0.3
      step(controller, desired=-8.0, actual=1.0, applied=0.0, limited=True)
      self.assertLess(controller.pid.i, 0.3)

  def test_driver_override_freezes_and_zero_deadzone_cleanup_remains_reachable(self):
    controller = build_controller(False)
    controller.steering_angle_deadzone_deg = 0.0
    controller.pid.i = 0.2
    step(controller, desired=1.0, pressed=True)
    self.assertEqual(controller.pid.i, 0.2)
    controller._last_error = 0.01
    step(controller, desired=0.0, actual=0.01)
    self.assertLess(controller.pid.i, 0.19)

  def test_no_cut_guard_cancels_an_s_bend(self):
    for direction in (-1.0, 1.0):
      _, _, state = step(build_controller(), desired=-direction * 0.5, actual=direction * 2.0)
      self.assertAlmostEqual(state.desiredLateralAccel, -direction * 0.5)
      self.assertLess(state.p * direction, 0.0)

  def test_continuous_planned_jerk_and_constant_friction_coefficient(self):
    controller = build_controller()
    for slope in (0.0, 1.0, -1.0, 0.0, 100.0, -100.0):
      previous = controller.extension.lookahead_lateral_jerk
      step(controller, model_accels=[slope * t for t in ModelConstants.T_IDXS])
      current = controller.extension.lookahead_lateral_jerk
      self.assertLessEqual(abs(current), MAX_LATERAL_JERK)
      self.assertLessEqual(abs(current - previous), 2 * MAX_LATERAL_JERK * controller.dt / (0.15 + controller.dt) + 1e-9)
      self.assertEqual(controller.extension.lat_accel_friction_factor, 0.7)

  def test_partial_and_nonfinite_model_fall_back_without_command_step(self):
    controller = build_controller()
    previous, _, _ = step(controller)
    for invalid in (model([0.0] * 17), model([math.nan] * 33), model([0.0] * 33, pitches=[])):
      controller.extension.update_model_v2(invalid)
      self.assertFalse(controller.extension._nnlc_enabled)
    with patch.object(controller.extension.model, "evaluate", return_value=math.nan):
      output, _, state = step(controller)
      self.assertFalse(controller._using_nnlc)
      self.assertTrue(np.isfinite([output, state.p, state.f]).all())
      self.assertLessEqual(abs(output - previous) * controller.torque_params.latAccelFactor, MAX_LATERAL_JERK * controller.dt + 1e-9)

  def test_bad_observation_clears_state_and_recovers_on_next_valid_sample(self):
    for field in ("vEgo", "aEgo", "steeringAngleDeg", "steeringRateDeg"):
      for invalid in (math.nan, math.inf):
        controller = build_controller()
        step(controller)
        controller.pid.i = 0.3
        cs = SimpleNamespace(vEgo=20.0, aEgo=0.0, steeringAngleDeg=0.0, steeringRateDeg=0.0, steeringPressed=False)
        setattr(cs, field, invalid)
        output, _, state = controller.update(True, cs, ObservationVehicleModel(), SimpleNamespace(angleOffsetDeg=0.0, roll=0.0),
                                              False, 0.0025, None, False, 0.384)
        self.assertEqual(output, 0.0)
        self.assertFalse(state.active)
        self.assertEqual(controller.pid.i, 0.0)
        fresh = step(build_controller())
        recovered = step(controller)
        self.assertAlmostEqual(fresh[0], recovered[0])
        self.assertTrue(recovered[2].active)

    controller = build_controller()
    output, _, state = step(controller, desired=math.nan)
    self.assertEqual(output, 0.0)
    self.assertFalse(state.active)

  def test_malformed_override_cannot_replace_valid_calibration(self):
    controller = build_controller()
    previous = (controller.torque_params.latAccelFactor, controller.torque_params.latAccelOffset, controller.torque_params.friction)
    def corrupt(torque_params):
      torque_params.latAccelFactor = math.nan
      torque_params.friction = -1.0
      return True
    with patch.object(controller.extension, "update_override_torque_params", side_effect=corrupt):
      output, _, state = step(controller)
    self.assertTrue(math.isfinite(output))
    self.assertTrue(state.active)
    self.assertEqual((controller.torque_params.latAccelFactor, controller.torque_params.latAccelOffset, controller.torque_params.friction), previous)

  def test_every_nn_preview_interval_has_bounded_jerk_and_acceleration(self):
    controller = build_controller()
    controller.extension.update_lateral_lag(0.284)
    raw = np.interp(ModelConstants.T_IDXS, [0.0] + controller.extension.nn_future_times + [10.0], [0.0, 9.0, -9.0, 9.0, -9.0, -9.0])
    with patch.object(controller.extension.model, "evaluate", wraps=controller.extension.model.evaluate) as evaluate:
      step(controller, desired=1.0, model_accels=raw)
    # The final model call is feedforward; preceding calls estimate local gain.
    values = evaluate.call_args.args[0]
    future = [1.0] + values[7:11]
    times = [0.0] + controller.extension.future_times
    self.assertTrue(np.all(np.abs(future) <= 5.0))
    self.assertTrue(np.all(np.abs(np.diff(future) / np.diff(times)) <= MAX_LATERAL_JERK + 1e-9))

  def test_time_parameterized_model_preview_is_not_warped_again_by_a_ego(self):
    inputs = []
    for a_ego in (-3.0, 0.0, 3.0):
      controller = build_controller()
      controller.extension.update_lateral_lag(0.284)
      with patch.object(controller.extension.model, "evaluate", wraps=controller.extension.model.evaluate) as evaluate:
        step(controller, a_ego=a_ego, model_accels=[0.5 * t for t in ModelConstants.T_IDXS])
      inputs.append(evaluate.call_args.args[0])
    np.testing.assert_array_equal(inputs[0], inputs[1])
    np.testing.assert_array_equal(inputs[1], inputs[2])

  def test_nn_history_uses_exact_declared_ages_at_each_control_rate(self):
    for dt in (0.01, 0.02):
      controller = build_controller(dt=dt)
      with patch.object(controller.extension.model, "evaluate", wraps=controller.extension.model.evaluate) as evaluate:
        for frame in range(40):
          step(controller, desired=frame * dt)
      past = evaluate.call_args.args[0][4:7]
      np.testing.assert_allclose(past, [39 * dt + age for age in controller.extension.past_times], atol=1e-9)

  def test_scalar_delayed_plant_settles_without_friction_error_limit_cycle(self):
    # Calibrated linear plant plus command delay and first-order rack response.
    # Dry-friction runs also require I to retain the steady load. This is a
    # robustness regression, not an identification of the actual Hyundai plant.
    for direction in (-1.0, 1.0):
      for tau, dry_friction in ((0.1, 0.0), (0.2, 0.0), (0.1, 0.1), (0.2, 0.1)):
        controller = build_controller(False)
        delay = deque([0.0] * round(0.284 / controller.dt))
        actual = applied = 0.0
        values = []
        held = []
        for frame in range(3000):
          t = frame * controller.dt
          magnitude = float(np.interp(t, [0.0, 1.0, 2.0, 12.0, 13.0, 30.0], [0.0, 0.0, 1.5, 1.5, 0.0, 0.0]))
          output, _, _ = step(controller, desired=direction * magnitude, actual=actual, applied=applied)
          applied = -output
          delay.append(applied)
          delayed = delay.popleft()
          effective = math.copysign(max(0.0, abs(delayed) - dry_friction), delayed)
          plant_target = effective * controller.torque_params.latAccelFactor
          actual += controller.dt / (tau + controller.dt) * (plant_target - actual)
          if 11.0 < t < 12.0:
            held.append(actual)
          if t > 28.0:
            values.append(actual)
        with self.subTest(direction=direction, tau=tau, dry_friction=dry_friction):
          self.assertLess(max(values) - min(values), 0.05)
          self.assertLess(abs(float(np.mean(values))), 0.1)
          self.assertGreater(direction * float(np.mean(held)), 1.0)

  def test_offset_is_applied_once_in_feedforward_and_not_as_tracking_error(self):
    for enabled in (False, True):
      controller = build_controller(enabled)
      controller.update_torque_parameters(3.0, 0.2, 0.0)
      if enabled:
        # A linear inverse model exposes acceleration/roll/offset units exactly.
        def evaluator(values):
          return (values[1] - values[3] * 9.81) / 3.0
        with patch.object(controller.extension.model, "evaluate", side_effect=evaluator):
          _, _, state = step(controller, desired=1.0, actual=1.0)
      else:
        _, _, state = step(controller, desired=1.0, actual=1.0)
      self.assertAlmostEqual(state.p, 0.0)
      self.assertAlmostEqual(state.f, 0.7 * (1.0 - 0.2) / 3.0)

  def test_recorded_unwind_and_mirror_have_correct_feedback_sign(self):
    fixture = json.loads(FIXTURE.read_text())
    models = {item["mono"]: item for item in fixture["models"]}
    for direction in (-1.0, 1.0):
      controller = build_controller()
      controller.steering_angle_deadzone_deg = 0.0
      vm = VehicleModel(SimpleNamespace(**fixture["vehicle_parameters"]))
      previous = 0.0
      for sample in fixture["samples"]:
        md = models[sample["model_mono"]]
        controller.extension.update_model_v2(model([direction * value for value in md["acceleration_y"]],
                                                   [direction * value for value in md["orientation_x"]], md["orientation_y"]))
        controller.extension.update_lateral_lag(sample["lat_delay"])
        vm.update_params(sample["stiffness_factor"], sample["steer_ratio"])
        cs = SimpleNamespace(vEgo=sample["v_ego"], aEgo=sample["a_ego"], steeringAngleDeg=direction * sample["steering_angle_deg"],
                             steeringRateDeg=direction * sample["steering_rate_deg"], steeringPressed=sample["steering_pressed"])
        params = SimpleNamespace(angleOffsetDeg=direction * sample["angle_offset_deg"], roll=direction * sample["roll"])
        pose = SimpleNamespace(orientation=SimpleNamespace(pitch=sample["pose_pitch"]))
        co = SimpleNamespace(actuatorsOutput=SimpleNamespace(torque=direction * sample["applied_torque"]))
        output, _, state = controller.update(True, cs, vm, params, False, direction * sample["desired_curvature"], pose,
                                              False, sample["lat_delay"], car_output=co)
        with self.subTest(direction=direction, mono=sample["mono"]):
          self.assertGreaterEqual(state.p * (state.feedbackLateralAccel - state.actualLateralAccel), -1e-7)
          self.assertAlmostEqual(state.error, state.p)
          self.assertLessEqual(abs(output - previous) * controller.torque_params.latAccelFactor, MAX_LATERAL_JERK * controller.dt + 1e-9)
        previous = output

      # Isolate the originally wrong-sign P input independently of reconstructed
      # startup history or of any actuator-delay estimate.
      pivot = min(fixture["samples"], key=lambda s: abs(s["mono"] - 3358019565331))
      error = controller.extension.acceleration_error(pivot["v_ego"], direction * pivot["setpoint"],
                                                      direction * pivot["measurement"], direction * pivot["desired_lateral_accel"],
                                                      direction * pivot["roll"], [direction * pivot["roll"]] * 7)
      self.assertGreater(error * direction * (pivot["setpoint"] - pivot["measurement"]), 0.0)


if __name__ == "__main__":
  unittest.main()

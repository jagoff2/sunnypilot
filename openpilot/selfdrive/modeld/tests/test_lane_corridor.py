"""Fresh containment bounds remain independent of lane-tracking gates."""
import copy

import numpy as np
import pytest

from openpilot.selfdrive.modeld import lane_centering as core
from openpilot.selfdrive.modeld.lane_path import MODEL_X
from openpilot.selfdrive.modeld.tests.test_lane_centering_regression import road_output, selected_step


def road():
  output = road_output(22., np.zeros(3))
  output['road_edges'][0, 0, :, 0] = -2.2 - core.CAMERA_OFFSET
  output['road_edges'][0, 1, :, 0] = 2.2 - core.CAMERA_OFFSET
  output['road_edges_stds'][:] = .03
  return output


def refresh(controller, output):
  controller._refresh_corridor(output, 22., 0., .475)


def pinch(output):
  output['lane_lines'][0, 2, :, 0] -= 1.1 * np.exp(-((MODEL_X - 15.) / 5.)**2)


@pytest.mark.parametrize('wide', [False, True])
def test_fresh_inward_bound_is_not_dropped_by_tracking_width_gates(wide):
  output = road()
  controller = core.LaneCenteringController('absolute')
  refresh(controller, output)
  if wide:
    output['lane_lines'][0, 1, :, 0] = -3.8 - core.CAMERA_OFFSET
    output['lane_lines'][0, 2, :, 0] = .8 - core.CAMERA_OFFSET
    output['road_edges'][0, 0, :, 0] = -5.4 - core.CAMERA_OFFSET
  else:
    pinch(output)
  refresh(controller, output)
  assert controller.corridor_age == 0.
  np.testing.assert_allclose(controller.corridor.right, output['lane_lines'][0, 2, :, 0] + core.CAMERA_OFFSET)
  assert not controller.corridor.check(output['plan'])[0]


@pytest.mark.parametrize('sides', [(1,), (2,), (1, 2)])
@pytest.mark.parametrize('head', ['lane_lines', 'lane_lines_stds'])
def test_far_malformed_paint_truncates_support_without_widening_to_edges(sides, head):
  output = road()
  controller = core.LaneCenteringController('absolute')
  for side in sides:
    output[head][0, side, -1, 0] = np.nan
  refresh(controller, output)
  assert not controller.safety_blocked
  assert controller.corridor.horizon == MODEL_X[-2]
  np.testing.assert_allclose(controller.corridor.left, -1.8)
  np.testing.assert_allclose(controller.corridor.right, 1.8)


@pytest.mark.parametrize('side', [1, 2])
@pytest.mark.parametrize('head', ['lane_lines', 'lane_lines_stds'])
@pytest.mark.parametrize('prior', [False, True])
def test_near_malformed_claimed_paint_blocks_instead_of_widening_or_holding(head, side, prior):
  output = road()
  controller = core.LaneCenteringController('absolute')
  if prior:
    refresh(controller, output)
  output[head][0, side, 7, 0] = np.nan
  refresh(controller, output)
  assert controller.safety_blocked
  assert controller.safety_reason == 'corridor_invalid'
  _, status, _, _ = selected_step(controller, output, 22., 0.)
  assert status.safety_blocked


def test_unsupported_interpolation_bracket_cannot_invent_thirty_metre_horizon():
  output = road()
  controller = core.LaneCenteringController('absolute')
  bracket = np.searchsorted(MODEL_X, 30.)
  output['lane_lines_stds'][0, 2, bracket, :] = .6
  refresh(controller, output)
  assert controller.safety_blocked
  assert controller.corridor is None
  assert controller.safety_reason == 'corridor_invalid'


@pytest.mark.parametrize('side', [0, 1])
def test_far_edge_nan_does_not_discard_credible_inward_edge_nearby(side):
  output = road()
  controller = core.LaneCenteringController('absolute')
  value = -.9 if side == 0 else .9
  output['road_edges'][0, side, :, 0] = value - core.CAMERA_OFFSET
  output['road_edges'][0, side, -1, 0] = np.nan
  refresh(controller, output)
  boundary = controller.corridor.left if side == 0 else controller.corridor.right
  np.testing.assert_allclose(boundary, value)
  assert controller.corridor.horizon == MODEL_X[-2]
  assert not controller.corridor.check(output['plan'])[0]


def test_partial_fresh_inward_boundary_tightens_held_limits_without_renewing_age():
  output = road()
  controller = core.LaneCenteringController('absolute')
  refresh(controller, output)
  pinch(output)
  output['lane_lines_prob'][0, 2:4] = 0.
  output['road_edges_stds'][0, 0] = 1.
  refresh(controller, output)
  assert controller.corridor_age == controller.frame_dt
  fresh = output['lane_lines'][0, 2, :, 0] + core.CAMERA_OFFSET
  assert np.all(controller.corridor.right <= fresh + 1e-12)
  assert not controller.corridor.check(output['plan'])[0]
  for _ in range(10):
    refresh(controller, output)
  assert controller.corridor is None
  assert controller.safety_blocked


def test_full_update_no_longer_labels_fresh_inward_pinch_contained():
  output = road()
  controller = core.LaneCenteringController('absolute')
  previous_base = previous_selected = 0.
  for _ in range(40):
    _, _, previous_base, previous_selected = selected_step(controller, output, 22., 0., previous_base, previous_selected)
  current = copy.deepcopy(output)
  pinch(current)
  _, status, _, _ = selected_step(controller, current, 22., 0., previous_base, previous_selected)
  assert status.safety_blocked
  assert status.containment == 'blocked'
  assert status.min_clearance < -.3

"""A rejected lane correction cannot replace the usable native control path."""
import copy

import numpy as np
import pytest

from openpilot.selfdrive.modeld.lane_centering import CAMERA_OFFSET, LaneCenteringController
from openpilot.selfdrive.modeld.lane_path import Corridor, MODEL_X, road_edge_collision
from openpilot.selfdrive.modeld.constants import Plan
from openpilot.selfdrive.modeld.tests.test_lane_centering_regression import road_output


@pytest.mark.parametrize('mirror', [-1., 1.])
def test_rejected_lane_blend_releases_native_plan_and_does_not_warn_for_discarded_path(monkeypatch, mirror):
  original = road_output(20., np.zeros(3))
  original['road_edges'][0, 0, :, 0] = -2.2 - CAMERA_OFFSET
  original['road_edges'][0, 1, :, 0] = 2.2 - CAMERA_OFFSET
  original['road_edges_stds'][:] = .02
  selected = copy.deepcopy(original)
  selected['plan'][0, :, Plan.POSITION.start + 1] = mirror * .05 * selected['plan'][0, :, Plan.POSITION.start]
  collision, _ = road_edge_collision(selected['plan'], np.full_like(MODEL_X, -2.2), np.full_like(MODEL_X, 2.2),
                                     np.full_like(MODEL_X, .02), np.full_like(MODEL_X, .02))
  assert collision

  controller = LaneCenteringController('absolute')
  controller.corridor = Corridor(np.full_like(MODEL_X, -1.8), np.full_like(MODEL_X, 1.8))
  controller.last_path_weight = .6
  monkeypatch.setattr(controller, '_select_output', lambda *args: selected)
  monkeypatch.setattr(controller, '_build_lane_plan', lambda *args: (None, np.nan))
  published, status = controller._finish(original, 20., 0., .4, 0., 0.)

  assert published is original
  assert status.path_weight == 0.
  assert status.policy_fallback
  assert not status.collision_risk
  assert status.containment == 'blocked'  # rejected correction, not unavailable controls
  assert np.isnan(status.min_clearance)  # do not label native output with the discarded blend's margin
  assert status.checked_distance == 0.

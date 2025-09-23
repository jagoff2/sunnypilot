"""Shared contract definitions mirroring sunnypilot modeld metadata."""
from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Tuple

import numpy as np

from openpilot.selfdrive.modeld.constants import ModelConstants


VISION_INPUT_SHAPES: "OrderedDict[str, Tuple[int, ...]]" = OrderedDict({
  "img": (12, 128, 256),
  "big_img": (12, 128, 256),
})

POLICY_INPUT_SHAPES: "OrderedDict[str, Tuple[int, ...]]" = OrderedDict({
  "features_buffer": (
    ModelConstants.INPUT_HISTORY_BUFFER_LEN,
    ModelConstants.FEATURE_LEN,
  ),
  "desire": (
    ModelConstants.INPUT_HISTORY_BUFFER_LEN,
    ModelConstants.DESIRE_LEN,
  ),
  "traffic_convention": (ModelConstants.TRAFFIC_CONVENTION_LEN,),
})

VISION_OUTPUT_SHAPES: "OrderedDict[str, Tuple[int, ...]]" = OrderedDict({
  "meta": (55,),
  "desire_pred": (
    ModelConstants.DESIRE_PRED_LEN,
    ModelConstants.DESIRE_PRED_WIDTH,
  ),
  "pose": (ModelConstants.POSE_WIDTH, 2),
  "wide_from_device_euler": (ModelConstants.WIDE_FROM_DEVICE_WIDTH, 2),
  "road_transform": (ModelConstants.POSE_WIDTH, 2),
  "lane_lines": (
    ModelConstants.NUM_LANE_LINES,
    ModelConstants.IDX_N,
    ModelConstants.LANE_LINES_WIDTH,
    2,
  ),
  "lane_lines_prob": (ModelConstants.NUM_LANE_LINES, 2),
  "road_edges": (
    ModelConstants.NUM_ROAD_EDGES,
    ModelConstants.IDX_N,
    ModelConstants.ROAD_EDGES_WIDTH,
    2,
  ),
  "lead": (
    ModelConstants.LEAD_MHP_SELECTION,
    ModelConstants.LEAD_TRAJ_LEN,
    ModelConstants.LEAD_WIDTH,
    2,
  ),
  "lead_prob": (ModelConstants.LEAD_MHP_SELECTION,),
  "hidden_state": (ModelConstants.FEATURE_LEN,),
})

POLICY_OUTPUT_SHAPES: "OrderedDict[str, Tuple[int, ...]]" = OrderedDict({
  "plan": (
    ModelConstants.IDX_N,
    ModelConstants.PLAN_WIDTH,
    2,
  ),
  "desire_state": (ModelConstants.DESIRE_PRED_WIDTH,),
})


def flatten_size(shape: Tuple[int, ...]) -> int:
  size = 1
  for dim in shape:
    size *= dim
  return size


def total_output_size(shapes: Iterable[Tuple[int, ...]]) -> int:
  return int(np.sum([flatten_size(shape) for shape in shapes]))


__all__ = [
  "VISION_INPUT_SHAPES",
  "POLICY_INPUT_SHAPES",
  "VISION_OUTPUT_SHAPES",
  "POLICY_OUTPUT_SHAPES",
  "flatten_size",
  "total_output_size",
]

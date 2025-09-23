"""Dataset schema definitions shared across training/export."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from openpilot.selfdrive.modeld.constants import ModelConstants


# Model input shapes expected by modeld
IMG_SHAPE = (12, 128, 256)
BIG_IMG_SHAPE = (12, 256, 512)
FEATURE_BUFFER_SHAPE = (ModelConstants.INPUT_HISTORY_BUFFER_LEN, ModelConstants.FEATURE_LEN)
DESIRE_SHAPE = (ModelConstants.DESIRE_LEN,)
TRAFFIC_CONVENTION_SHAPE = (ModelConstants.TRAFFIC_CONVENTION_LEN,)


VISION_TARGET_SHAPES: Dict[str, Tuple[int, ...]] = {
  "pose": (ModelConstants.POSE_WIDTH,),
  "road_transform": (ModelConstants.POSE_WIDTH,),
  "lane_lines": (
    ModelConstants.NUM_LANE_LINES,
    ModelConstants.IDX_N,
    ModelConstants.LANE_LINES_WIDTH,
  ),
  "lane_lines_prob": (ModelConstants.NUM_LANE_LINES,),
  "road_edges": (
    ModelConstants.NUM_ROAD_EDGES,
    ModelConstants.IDX_N,
    ModelConstants.ROAD_EDGES_WIDTH,
  ),
  "lead": (
    ModelConstants.LEAD_MHP_SELECTION,
    ModelConstants.LEAD_WIDTH,
  ),
  "lead_prob": (ModelConstants.LEAD_MHP_SELECTION,),
  "meta": (ModelConstants.META_T_IDXS.__len__() * 5 + 1,),
  "desire_pred": (ModelConstants.DESIRE_PRED_WIDTH,),
  "hidden_state": (ModelConstants.FEATURE_LEN,),
}

POLICY_TARGET_SHAPES: Dict[str, Tuple[int, ...]] = {
  "plan": (
    ModelConstants.PLAN_MHP_SELECTION,
    ModelConstants.PLAN_WIDTH,
  ),
  "plan_prob": (ModelConstants.PLAN_MHP_SELECTION,),
  "desire_state": (ModelConstants.DESIRE_PRED_WIDTH,),
}


@dataclass
class DatasetSchema:
  vision_inputs: Dict[str, Tuple[int, ...]]
  vision_targets: Dict[str, Tuple[int, ...]]
  policy_inputs: Dict[str, Tuple[int, ...]]
  policy_targets: Dict[str, Tuple[int, ...]]

  @classmethod
  def default(cls) -> "DatasetSchema":
    return cls(
      vision_inputs={
        "img": IMG_SHAPE,
        "big_img": BIG_IMG_SHAPE,
        "features_buffer": FEATURE_BUFFER_SHAPE,
        "desire": DESIRE_SHAPE,
        "traffic_convention": TRAFFIC_CONVENTION_SHAPE,
      },
      vision_targets=VISION_TARGET_SHAPES,
      policy_inputs={
        "features_buffer": FEATURE_BUFFER_SHAPE,
        "desire": DESIRE_SHAPE,
        "traffic_convention": TRAFFIC_CONVENTION_SHAPE,
      },
      policy_targets=POLICY_TARGET_SHAPES,
    )

  def target_dtype(self, name: str) -> np.dtype:
    if name.endswith("prob") or "desire" in name:
      return np.float32
    return np.float32


DEFAULT_SCHEMA = DatasetSchema.default()

__all__ = [
  "IMG_SHAPE",
  "BIG_IMG_SHAPE",
  "FEATURE_BUFFER_SHAPE",
  "DatasetSchema",
  "DEFAULT_SCHEMA",
]

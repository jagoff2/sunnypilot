"""Dataset schema definitions shared across training/export."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from ..contracts import (
  POLICY_INPUT_SHAPES,
  POLICY_OUTPUT_SHAPES,
  VISION_INPUT_SHAPES,
  VISION_OUTPUT_SHAPES,
)
from ..contracts import flatten_size


IMG_SHAPE = VISION_INPUT_SHAPES["img"]
BIG_IMG_SHAPE = VISION_INPUT_SHAPES["big_img"]
FEATURE_BUFFER_SHAPE = POLICY_INPUT_SHAPES["features_buffer"]
DESIRE_HISTORY_SHAPE = POLICY_INPUT_SHAPES["desire"]
TRAFFIC_CONVENTION_SHAPE = POLICY_INPUT_SHAPES["traffic_convention"]


@dataclass
class DatasetSchema:
  vision_inputs: Dict[str, Tuple[int, ...]]
  policy_inputs: Dict[str, Tuple[int, ...]]
  vision_targets: Dict[str, Tuple[int, ...]]
  policy_targets: Dict[str, Tuple[int, ...]]

  @classmethod
  def default(cls) -> "DatasetSchema":
    return cls(
      vision_inputs=dict(VISION_INPUT_SHAPES),
      policy_inputs=dict(POLICY_INPUT_SHAPES),
      vision_targets=dict(VISION_OUTPUT_SHAPES),
      policy_targets=dict(POLICY_OUTPUT_SHAPES),
    )

  def flattened_size(self, name: str) -> int:
    if name in self.vision_targets:
      return flatten_size(self.vision_targets[name])
    if name in self.policy_targets:
      return flatten_size(self.policy_targets[name])
    raise KeyError(name)


DEFAULT_SCHEMA = DatasetSchema.default()

__all__ = [
  "IMG_SHAPE",
  "BIG_IMG_SHAPE",
  "FEATURE_BUFFER_SHAPE",
  "DESIRE_HISTORY_SHAPE",
  "TRAFFIC_CONVENTION_SHAPE",
  "DatasetSchema",
  "DEFAULT_SCHEMA",
]

"""Utilities for generating sunnypilot model metadata pickles."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from openpilot.selfdrive.modeld.constants import ModelConstants

from ..dataset import DEFAULT_SCHEMA


@dataclass
class MetadataPackage:
  input_shapes: Dict[str, Tuple[int, ...]]
  output_shapes: Dict[str, Tuple[int, ...]]
  output_slices: Dict[str, Tuple[int, int]]

  def as_dict(self) -> Dict[str, Dict[str, Tuple[int, ...]]]:
    return {
      "input_shapes": self.input_shapes,
      "output_shapes": self.output_shapes,
      "output_slices": self.output_slices,
    }


def _flatten_shape(shape: Tuple[int, ...]) -> int:
  size = 1
  for dim in shape:
    size *= dim
  return size


def generate_metadata(output_shapes: Dict[str, Tuple[int, ...]], batch_dim: bool = True) -> MetadataPackage:
  input_shapes = {name: (1,) + shape for name, shape in DEFAULT_SCHEMA.vision_inputs.items()}
  output_slices: Dict[str, Tuple[int, int]] = {}
  offset = 0
  for name, shape in output_shapes.items():
    size = _flatten_shape(shape)
    output_slices[name] = (offset, offset + size)
    offset += size
  formatted_output_shapes = {name: ((1,) + shape if batch_dim else shape) for name, shape in output_shapes.items()}
  return MetadataPackage(input_shapes, formatted_output_shapes, output_slices)


def vision_output_shapes() -> Dict[str, Tuple[int, ...]]:
  return {
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
    "meta": (26,),
    "desire_pred": (ModelConstants.DESIRE_PRED_WIDTH,),
    "hidden_state": (ModelConstants.FEATURE_LEN,),
  }


def policy_output_shapes() -> Dict[str, Tuple[int, ...]]:
  return {
    "plan": (
      ModelConstants.PLAN_MHP_SELECTION,
      ModelConstants.PLAN_WIDTH,
    ),
    "plan_prob": (ModelConstants.PLAN_MHP_SELECTION,),
    "desire_state": (ModelConstants.DESIRE_PRED_WIDTH,),
  }


def generate_vision_metadata() -> MetadataPackage:
  return generate_metadata(vision_output_shapes())


def generate_policy_metadata() -> MetadataPackage:
  inputs = {
    "features_buffer": (1, ModelConstants.INPUT_HISTORY_BUFFER_LEN, ModelConstants.FEATURE_LEN),
    "desire": (1, ModelConstants.DESIRE_LEN),
    "traffic_convention": (1, ModelConstants.TRAFFIC_CONVENTION_LEN),
    "hidden_state": (1, ModelConstants.FEATURE_LEN),
  }
  outputs = policy_output_shapes()
  output_slices: Dict[str, Tuple[int, int]] = {}
  offset = 0
  for name, shape in outputs.items():
    size = _flatten_shape(shape)
    output_slices[name] = (offset, offset + size)
    offset += size
  return MetadataPackage(inputs, {name: (1,) + shape for name, shape in outputs.items()}, output_slices)


__all__ = [
  "MetadataPackage",
  "generate_vision_metadata",
  "generate_policy_metadata",
  "vision_output_shapes",
  "policy_output_shapes",
]

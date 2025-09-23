from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, OrderedDict

from ..contracts import (
  POLICY_INPUT_SHAPES,
  POLICY_OUTPUT_SHAPES,
  VISION_INPUT_SHAPES,
  VISION_OUTPUT_SHAPES,
  flatten_size,
  total_output_size,
)


@dataclass
class MetadataPackage:
  input_shapes: Dict[str, tuple[int, ...]]
  output_slices: Dict[str, slice]
  output_size: int
  model_checkpoint: str | None = None

  def as_dict(self) -> Dict[str, object]:
    data: Dict[str, object] = {
      "input_shapes": self.input_shapes,
      "output_slices": self.output_slices,
      "output_shapes": {"outputs": (1, self.output_size)},
    }
    if self.model_checkpoint:
      data["model_checkpoint"] = self.model_checkpoint
    return data


def _batchify(shapes: Dict[str, tuple[int, ...]]) -> Dict[str, tuple[int, ...]]:
  return {name: (1,) + shape for name, shape in shapes.items()}


def _compute_slices(outputs: "OrderedDict[str, tuple[int, ...]]") -> Dict[str, slice]:
  slices: Dict[str, slice] = {}
  offset = 0
  for name, shape in outputs.items():
    size = flatten_size(shape)
    slices[name] = slice(offset, offset + size)
    offset += size
  return slices


def generate_vision_metadata() -> MetadataPackage:
  input_shapes = _batchify(dict(VISION_INPUT_SHAPES))
  slices = _compute_slices(VISION_OUTPUT_SHAPES)
  output_size = total_output_size(VISION_OUTPUT_SHAPES.values())
  return MetadataPackage(input_shapes=input_shapes, output_slices=slices, output_size=output_size)


def generate_policy_metadata() -> MetadataPackage:
  input_shapes = _batchify(dict(POLICY_INPUT_SHAPES))
  slices = _compute_slices(POLICY_OUTPUT_SHAPES)
  output_size = total_output_size(POLICY_OUTPUT_SHAPES.values())
  return MetadataPackage(input_shapes=input_shapes, output_slices=slices, output_size=output_size)


def vision_output_shapes() -> Dict[str, tuple[int, ...]]:
  return dict(VISION_OUTPUT_SHAPES)


def policy_output_shapes() -> Dict[str, tuple[int, ...]]:
  return dict(POLICY_OUTPUT_SHAPES)


__all__ = [
  "MetadataPackage",
  "generate_vision_metadata",
  "generate_policy_metadata",
  "vision_output_shapes",
  "policy_output_shapes",
]

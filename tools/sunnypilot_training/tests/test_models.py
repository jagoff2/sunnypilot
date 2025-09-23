import pytest

torch = pytest.importorskip("torch")

from ..contracts import (
  POLICY_INPUT_SHAPES,
  POLICY_OUTPUT_SHAPES,
  VISION_INPUT_SHAPES,
  VISION_OUTPUT_SHAPES,
)
from ..models import DiffusionPolicyModel, DiffusionVisionModel


def test_vision_forward() -> None:
  model = DiffusionVisionModel()
  img = torch.randn(1, *VISION_INPUT_SHAPES["img"])
  big_img = torch.randn(1, *VISION_INPUT_SHAPES["big_img"])
  outputs = model(img, big_img)
  for name, shape in VISION_OUTPUT_SHAPES.items():
    assert name in outputs
    assert outputs[name].shape[1:] == shape


def test_policy_forward() -> None:
  model = DiffusionPolicyModel()
  features_buffer = torch.zeros(1, *POLICY_INPUT_SHAPES["features_buffer"])
  desire = torch.zeros(1, *POLICY_INPUT_SHAPES["desire"])
  traffic = torch.zeros(1, *POLICY_INPUT_SHAPES["traffic_convention"])
  outputs = model(features_buffer, desire, traffic)
  for name, shape in POLICY_OUTPUT_SHAPES.items():
    assert name in outputs
    assert outputs[name].shape[1:] == shape

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ..models import DiffusionPolicyModel, DiffusionVisionModel
from ..dataset.schema import IMG_SHAPE, BIG_IMG_SHAPE, FEATURE_BUFFER_SHAPE


def test_vision_forward() -> None:
  model = DiffusionVisionModel()
  img = torch.randn(1, *IMG_SHAPE)
  big_img = torch.randn(1, BIG_IMG_SHAPE[0], BIG_IMG_SHAPE[1], BIG_IMG_SHAPE[2])
  desire = torch.zeros(1, 8)
  traffic = torch.zeros(1, 2)
  features_buffer = torch.zeros(1, FEATURE_BUFFER_SHAPE[0], FEATURE_BUFFER_SHAPE[1])
  outputs = model(img, big_img, desire=desire, traffic_convention=traffic, features_buffer=features_buffer)
  assert outputs["hidden_state"].shape[-1] == 512


def test_policy_forward() -> None:
  model = DiffusionPolicyModel()
  features_buffer = torch.zeros(1, FEATURE_BUFFER_SHAPE[0], FEATURE_BUFFER_SHAPE[1])
  desire = torch.zeros(1, 8)
  traffic = torch.zeros(1, 2)
  hidden_state = torch.zeros(1, 512)
  outputs = model(features_buffer, desire, traffic, hidden_state)
  assert outputs["plan"].shape[-1] == model.plan_dim

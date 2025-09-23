from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ..export.common import load_checkpoint, save_metadata, save_tinygrad_weights
from ..export.export_policy import PolicyExportWrapper
from ..export.export_vision import VisionExportWrapper
from ..export.metadata import generate_vision_metadata
from ..models import DiffusionPolicyModel, DiffusionVisionModel


def test_export_wrappers(tmp_path: Path) -> None:
  vision = DiffusionVisionModel()
  policy = DiffusionPolicyModel()
  checkpoint = tmp_path / "ckpt.pt"
  torch.save({"vision": vision.state_dict(), "policy": policy.state_dict()}, checkpoint)
  device = torch.device("cpu")
  loaded_vision, loaded_policy = load_checkpoint(checkpoint, device)

  vision_wrapper = VisionExportWrapper(loaded_vision)
  policy_wrapper = PolicyExportWrapper(loaded_policy)

  vision_inputs = {
    "img": torch.randn(1, 12, 128, 256),
    "big_img": torch.randn(1, 12, 256, 512),
    "desire": torch.zeros(1, 8),
    "traffic_convention": torch.zeros(1, 2),
    "features_buffer": torch.zeros(1, 25, 512),
  }
  _ = vision_wrapper(**vision_inputs)

  policy_inputs = {
    "features_buffer": torch.zeros(1, 25, 512),
    "desire": torch.zeros(1, 8),
    "traffic_convention": torch.zeros(1, 2),
    "hidden_state": torch.zeros(1, 512),
  }
  _ = policy_wrapper(**policy_inputs)

  metadata = generate_vision_metadata()
  save_metadata(metadata, tmp_path / "vision_meta.pkl")
  save_tinygrad_weights(loaded_vision, tmp_path / "vision_weights.pkl")
  assert (tmp_path / "vision_meta.pkl").exists()
  assert (tmp_path / "vision_weights.pkl").exists()

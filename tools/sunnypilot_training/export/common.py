"""Shared utilities for exporting sunnypilot models."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Tuple

import torch

from ..models import DiffusionPolicyModel, DiffusionVisionModel
from .metadata import MetadataPackage


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[DiffusionVisionModel, DiffusionPolicyModel]:
  checkpoint = torch.load(checkpoint_path, map_location=device)
  default_policy = DiffusionPolicyModel()
  diffusion_steps = int(checkpoint.get("diffusion_steps", default_policy.schedule.timesteps))
  if diffusion_steps != default_policy.schedule.timesteps:
    policy = DiffusionPolicyModel(diffusion_steps=diffusion_steps)
  else:
    policy = default_policy
  vision = DiffusionVisionModel().to(device)
  policy = policy.to(device)
  vision.load_state_dict(checkpoint["vision"])
  policy.load_state_dict(checkpoint["policy"])
  vision.eval()
  policy.eval()
  return vision, policy


def export_onnx(model: torch.nn.Module, inputs: Dict[str, torch.Tensor], output_path: Path,
                dynamic_axes: Dict[str, Dict[int, str]] | None = None) -> None:
  output_path.parent.mkdir(parents=True, exist_ok=True)
  torch.onnx.export(
    model,
    tuple(inputs.values()),
    str(output_path),
    input_names=list(inputs.keys()),
    output_names=None,
    opset_version=16,
    dynamic_axes=dynamic_axes,
  )


def save_metadata(metadata: MetadataPackage, output_path: Path) -> None:
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with open(output_path, "wb") as f:
    pickle.dump(metadata.as_dict(), f)


def save_tinygrad_weights(model: torch.nn.Module, output_path: Path) -> None:
  """Persist the PyTorch state dict in a format tinygrad can ingest."""
  output_path.parent.mkdir(parents=True, exist_ok=True)
  state_dict = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
  with open(output_path, "wb") as f:
    pickle.dump(state_dict, f)


__all__ = [
  "load_checkpoint",
  "export_onnx",
  "save_metadata",
  "save_tinygrad_weights",
]

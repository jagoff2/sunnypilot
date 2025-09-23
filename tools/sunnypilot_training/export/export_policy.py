"""Export the diffusion policy model."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..models import DiffusionPolicyModel
from .common import export_onnx, load_checkpoint, save_metadata, save_tinygrad_weights
from .metadata import generate_policy_metadata, policy_output_shapes


class PolicyExportWrapper(torch.nn.Module):
  def __init__(self, model: DiffusionPolicyModel) -> None:
    super().__init__()
    self.model = model
    self.output_order = list(policy_output_shapes().keys())

  def forward(self, features_buffer: torch.Tensor, desire: torch.Tensor,
              traffic_convention: torch.Tensor, hidden_state: torch.Tensor) -> torch.Tensor:
    outputs = self.model(features_buffer, desire, traffic_convention, hidden_state)
    flat_outputs = [outputs[name].reshape(outputs[name].shape[0], -1) for name in self.output_order]
    return torch.cat(flat_outputs, dim=-1)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Export the diffusion policy model")
  parser.add_argument("checkpoint", type=Path)
  parser.add_argument("--onnx", type=Path, default=Path("driving_policy.onnx"))
  parser.add_argument("--tinygrad", type=Path, default=Path("driving_policy_tinygrad.pkl"))
  parser.add_argument("--metadata", type=Path, default=Path("driving_policy_metadata.pkl"))
  parser.add_argument("--device", default="cpu")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  device = torch.device(args.device)
  _, policy = load_checkpoint(args.checkpoint, device)
  metadata = generate_policy_metadata()

  wrapper = PolicyExportWrapper(policy)
  dummy_inputs = {
    "features_buffer": torch.zeros(metadata.input_shapes["features_buffer"], device=device),
    "desire": torch.zeros(metadata.input_shapes["desire"], device=device),
    "traffic_convention": torch.zeros(metadata.input_shapes["traffic_convention"], device=device),
    "hidden_state": torch.zeros(metadata.input_shapes["hidden_state"], device=device),
  }
  export_onnx(wrapper, dummy_inputs, args.onnx)
  save_tinygrad_weights(policy, args.tinygrad)
  save_metadata(metadata, args.metadata)


if __name__ == "__main__":
  main()

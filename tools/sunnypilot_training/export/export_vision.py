import argparse
from pathlib import Path

import torch

from ..models import DiffusionVisionModel
from .common import export_onnx, load_checkpoint, save_metadata, save_tinygrad_weights
from .metadata import generate_vision_metadata, vision_output_shapes


class VisionExportWrapper(torch.nn.Module):
  def __init__(self, model: DiffusionVisionModel) -> None:
    super().__init__()
    self.model = model
    self.output_order = list(vision_output_shapes().keys())

  def forward(self, img: torch.Tensor, big_img: torch.Tensor) -> torch.Tensor:
    outputs = self.model(img, big_img)
    flat_outputs = [outputs[name].reshape(outputs[name].shape[0], -1) for name in self.output_order]
    return torch.cat(flat_outputs, dim=-1)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Export the diffusion vision model")
  parser.add_argument("checkpoint", type=Path, help="Path to training checkpoint")
  parser.add_argument("--onnx", type=Path, default=Path("driving_vision.onnx"))
  parser.add_argument("--tinygrad", type=Path, default=Path("driving_vision_tinygrad.pkl"))
  parser.add_argument("--metadata", type=Path, default=Path("driving_vision_metadata.pkl"))
  parser.add_argument("--device", default="cpu")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  device = torch.device(args.device)
  vision, _ = load_checkpoint(args.checkpoint, device)
  metadata = generate_vision_metadata()
  metadata.model_checkpoint = str(args.checkpoint)

  wrapper = VisionExportWrapper(vision)
  dummy_inputs = {
    "img": torch.randn(1, *metadata.input_shapes["img"][1:], device=device),
    "big_img": torch.randn(1, *metadata.input_shapes["big_img"][1:], device=device),
  }
  export_onnx(wrapper, dummy_inputs, args.onnx)
  save_tinygrad_weights(vision, args.tinygrad)
  save_metadata(metadata, args.metadata)


if __name__ == "__main__":
  main()

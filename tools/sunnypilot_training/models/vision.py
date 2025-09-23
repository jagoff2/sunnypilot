from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from openpilot.selfdrive.modeld.constants import ModelConstants

from ..contracts import VISION_OUTPUT_SHAPES, flatten_size
from .modules import ConvStem, SpatialTransformer, TemporalAttention


class DiffusionVisionModel(nn.Module):
  """Vision backbone consuming temporal camera stacks."""

  def __init__(self, embed_dim: int = 192, temporal_heads: int = 4) -> None:
    super().__init__()
    self.narrow_stem = ConvStem(1, embed_dim)
    self.wide_stem = ConvStem(1, embed_dim)
    self.spatial = SpatialTransformer(embed_dim, depth=2, num_heads=4)
    fused_dim = embed_dim * 2
    self.temporal = TemporalAttention(fused_dim, num_heads=temporal_heads, depth=3)
    self.feature_proj = nn.Linear(fused_dim, ModelConstants.FEATURE_LEN)
    self.hidden_proj = nn.Linear(ModelConstants.FEATURE_LEN, ModelConstants.FEATURE_LEN)

    self.heads = nn.ModuleDict()
    for name, shape in VISION_OUTPUT_SHAPES.items():
      if name == "hidden_state":
        continue
      self.heads[name] = nn.Sequential(
        nn.Linear(ModelConstants.FEATURE_LEN, ModelConstants.FEATURE_LEN),
        nn.GELU(),
        nn.Linear(ModelConstants.FEATURE_LEN, flatten_size(shape)),
      )

  def forward(self, img: torch.Tensor, big_img: torch.Tensor) -> Dict[str, torch.Tensor]:
    narrow_tokens = self._encode_stack(img, self.narrow_stem)
    wide_tokens = self._encode_stack(big_img, self.wide_stem)
    fused = torch.cat([narrow_tokens, wide_tokens], dim=-1)
    fused = self.temporal(fused)
    pooled = fused.mean(dim=1)
    features = torch.tanh(self.feature_proj(pooled))

    outputs: Dict[str, torch.Tensor] = {}
    for name, head in self.heads.items():
      flat = head(features)
      outputs[name] = flat.view(img.shape[0], *VISION_OUTPUT_SHAPES[name])
    outputs["hidden_state"] = self.hidden_proj(features)
    return outputs

  def _encode_stack(self, stack: torch.Tensor, stem: nn.Module) -> torch.Tensor:
    b, t, h, w = stack.shape
    x = stack.view(b * t, 1, h, w)
    x = stem(x)
    _, c, h1, w1 = x.shape
    x = self.spatial(x, (h1, w1))
    x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(1, 1)).view(b, t, -1)
    return x


__all__ = [
  "DiffusionVisionModel",
]

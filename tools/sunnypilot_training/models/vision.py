"""Diffusion-inspired vision encoder producing parser-compatible outputs."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from openpilot.selfdrive.modeld.constants import ModelConstants

from .modules import ConvStem, MLPHead, SpatialTransformer, TemporalAttention


class DiffusionVisionModel(nn.Module):
  """Vision backbone consuming temporal camera stacks."""

  def __init__(self, embed_dim: int = 192, temporal_heads: int = 4) -> None:
    super().__init__()
    self.narrow_stem = ConvStem(1, embed_dim)
    self.wide_stem = ConvStem(1, embed_dim)
    self.spatial = SpatialTransformer(embed_dim, depth=2, num_heads=4)
    fused_dim = embed_dim * 2
    self.temporal = TemporalAttention(fused_dim, num_heads=temporal_heads, depth=3)
    self.fusion = nn.Linear(
      fused_dim + ModelConstants.DESIRE_LEN + ModelConstants.TRAFFIC_CONVENTION_LEN,
      ModelConstants.FEATURE_LEN,
    )
    self.hidden_proj = nn.Linear(ModelConstants.FEATURE_LEN, ModelConstants.FEATURE_LEN)

    output_dim = ModelConstants.FEATURE_LEN
    self.pose_head = MLPHead(output_dim, ModelConstants.POSE_WIDTH)
    self.road_transform_head = MLPHead(output_dim, ModelConstants.POSE_WIDTH)
    self.lane_head = MLPHead(
      output_dim,
      ModelConstants.NUM_LANE_LINES * ModelConstants.IDX_N * ModelConstants.LANE_LINES_WIDTH,
    )
    self.road_edge_head = MLPHead(
      output_dim,
      ModelConstants.NUM_ROAD_EDGES * ModelConstants.IDX_N * ModelConstants.ROAD_EDGES_WIDTH,
    )
    self.lane_prob_head = MLPHead(output_dim, ModelConstants.NUM_LANE_LINES)
    self.lead_head = MLPHead(output_dim, ModelConstants.LEAD_MHP_SELECTION * ModelConstants.LEAD_WIDTH)
    self.lead_prob_head = MLPHead(output_dim, ModelConstants.LEAD_MHP_SELECTION)
    self.meta_head = MLPHead(output_dim, 26)
    self.desire_head = MLPHead(output_dim, ModelConstants.DESIRE_PRED_WIDTH)

    self.dropout = nn.Dropout(0.1)

  def forward(self, img: torch.Tensor, big_img: torch.Tensor, *, desire: torch.Tensor,
              traffic_convention: torch.Tensor, features_buffer: torch.Tensor | None = None
              ) -> Dict[str, torch.Tensor]:
    # img / big_img: (B, T, H, W)
    narrow_tokens = self._encode_stack(img, self.narrow_stem)
    wide_tokens = self._encode_stack(big_img, self.wide_stem)
    fused = torch.cat([narrow_tokens, wide_tokens], dim=-1)
    fused = self.temporal(fused)
    pooled = fused.mean(dim=1)
    conditioning = torch.cat([pooled, desire, traffic_convention], dim=-1)
    features = self.fusion(conditioning)
    hidden = torch.tanh(self.hidden_proj(features))

    outputs: Dict[str, torch.Tensor] = {
      "pose": self.pose_head(hidden),
      "road_transform": self.road_transform_head(hidden),
      "lane_lines": self._reshape_lane(self.lane_head(hidden)),
      "road_edges": self._reshape_edges(self.road_edge_head(hidden)),
      "lane_lines_prob": torch.sigmoid(self.lane_prob_head(hidden)),
      "lead": self._reshape_lead(self.lead_head(hidden)),
      "lead_prob": torch.sigmoid(self.lead_prob_head(hidden)),
      "meta": self.meta_head(hidden),
      "desire_pred": torch.softmax(self.desire_head(hidden), dim=-1),
      "hidden_state": hidden,
    }
    return outputs

  def _encode_stack(self, stack: torch.Tensor, stem: nn.Module) -> torch.Tensor:
    b, t, h, w = stack.shape
    x = stack.view(b * t, 1, h, w)
    x = stem(x)
    _, c, h1, w1 = x.shape
    x = self.spatial(x, (h1, w1))
    x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(1, 1)).view(b, t, -1)
    return x

  def _reshape_lane(self, lane: torch.Tensor) -> torch.Tensor:
    shape = (lane.shape[0], ModelConstants.NUM_LANE_LINES, ModelConstants.IDX_N, ModelConstants.LANE_LINES_WIDTH)
    return lane.view(shape)

  def _reshape_edges(self, edge: torch.Tensor) -> torch.Tensor:
    shape = (edge.shape[0], ModelConstants.NUM_ROAD_EDGES, ModelConstants.IDX_N, ModelConstants.ROAD_EDGES_WIDTH)
    return edge.view(shape)

  def _reshape_lead(self, lead: torch.Tensor) -> torch.Tensor:
    shape = (lead.shape[0], ModelConstants.LEAD_MHP_SELECTION, ModelConstants.LEAD_WIDTH)
    return lead.view(shape)


__all__ = [
  "DiffusionVisionModel",
]

"""Reusable neural network building blocks."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class ConvStem(nn.Module):
  """Simple convolutional stem inspired by ConvNeXt."""

  def __init__(self, in_channels: int, embed_dim: int) -> None:
    super().__init__()
    self.stem = nn.Sequential(
      nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(embed_dim // 2),
      nn.GELU(),
      nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(embed_dim),
      nn.GELU(),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.stem(x)


class SpatialTransformer(nn.Module):
  """Transformer encoder applied to flattened spatial patches."""

  def __init__(self, embed_dim: int, depth: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
    super().__init__()
    encoder_layer = nn.TransformerEncoderLayer(
      d_model=embed_dim,
      nhead=num_heads,
      dim_feedforward=int(embed_dim * mlp_ratio),
      dropout=0.1,
      activation="gelu",
      batch_first=True,
    )
    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

  def forward(self, x: torch.Tensor, hw_shape: Tuple[int, int]) -> torch.Tensor:
    b, c, h, w = x.shape
    tokens = x.flatten(2).transpose(1, 2)
    encoded = self.encoder(tokens)
    return encoded.view(b, h, w, -1).permute(0, 3, 1, 2)


class TemporalAttention(nn.Module):
  """Temporal attention across stacked frame embeddings."""

  def __init__(self, embed_dim: int, num_heads: int, depth: int = 2) -> None:
    super().__init__()
    layers = []
    for _ in range(depth):
      layers.append(nn.MultiheadAttention(embed_dim, num_heads, batch_first=True))
      layers.append(nn.LayerNorm(embed_dim))
    self.layers = nn.ModuleList(layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: (B, T, C)
    out = x
    for attn, norm in zip(self.layers[0::2], self.layers[1::2]):
      residual = out
      out, _ = attn(out, out, out)
      out = norm(out + residual)
    return out


class MLPHead(nn.Module):
  def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512, dropout: float = 0.0) -> None:
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(input_dim, hidden_dim),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(hidden_dim, output_dim),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
  """Create sinusoidal embeddings as used in diffusion models."""
  device = timesteps.device
  half_dim = dim // 2
  exponent = torch.arange(half_dim, dtype=torch.float32, device=device)
  exponent = 1e-4 ** (exponent / half_dim)
  embeddings = timesteps.float()[:, None] * exponent[None, :]
  embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
  if dim % 2 == 1:
    embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
  return embeddings


__all__ = [
  "ConvStem",
  "SpatialTransformer",
  "TemporalAttention",
  "MLPHead",
  "sinusoidal_time_embedding",
]

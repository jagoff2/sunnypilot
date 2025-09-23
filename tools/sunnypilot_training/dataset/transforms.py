"""Data augmentation utilities for camera stacks."""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np


def photometric_jitter(frames: np.ndarray, brightness: float = 0.05, contrast: float = 0.05,
                       saturation: float = 0.05, hue: float = 0.02, rng: np.random.Generator | None = None
                       ) -> np.ndarray:
  """Apply simple photometric jitter to a stack of frames."""
  if rng is None:
    rng = np.random.default_rng()
  img = frames.astype(np.float32)
  img = img * (1.0 + rng.uniform(-contrast, contrast))
  img = img + rng.uniform(-brightness, brightness)
  img = np.clip(img, 0.0, 1.0)
  # approximate saturation by mixing towards the mean luminance
  mean = img.mean(axis=-3, keepdims=True)
  img = img * (1.0 + rng.uniform(-saturation, saturation)) + mean * rng.uniform(-saturation, saturation)
  # hue jitter is approximated by cyclic shift in YUV
  if img.shape[-3] == 3:
    u = img[..., 1, :, :]
    v = img[..., 2, :, :]
    angle = rng.uniform(-hue, hue)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    img[..., 1, :, :] = cos_a * u - sin_a * v
    img[..., 2, :, :] = sin_a * u + cos_a * v
  return np.clip(img, 0.0, 1.0)


def temporal_dropout(frames: np.ndarray, dropout_prob: float = 0.1,
                     rng: np.random.Generator | None = None) -> np.ndarray:
  """Randomly zero out temporal slices to encourage robustness."""
  if rng is None:
    rng = np.random.default_rng()
  if frames.ndim == 3:
    time_dim = frames.shape[0]
    mask = (rng.random(time_dim) > dropout_prob).astype(frames.dtype)
    return frames * mask[:, None, None]
  if frames.ndim == 4:
    time_dim = frames.shape[0]
    mask = (rng.random(time_dim) > dropout_prob).astype(frames.dtype)
    return frames * mask[:, None, None, None]
  return frames


AUGMENTATIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
  "photometric_jitter": photometric_jitter,
  "temporal_dropout": temporal_dropout,
}

__all__ = [
  "photometric_jitter",
  "temporal_dropout",
  "AUGMENTATIONS",
]

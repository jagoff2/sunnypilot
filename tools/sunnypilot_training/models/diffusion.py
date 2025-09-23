"""1D diffusion utilities used by the policy model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class DiffusionSchedule:
  timesteps: int
  beta_start: float = 1e-4
  beta_end: float = 0.02

  def betas(self, device: torch.device) -> torch.Tensor:
    return torch.linspace(self.beta_start, self.beta_end, self.timesteps, device=device)

  def alphas(self, device: torch.device) -> torch.Tensor:
    betas = self.betas(device)
    return 1.0 - betas

  def alpha_bars(self, device: torch.device) -> torch.Tensor:
    alphas = self.alphas(device)
    return torch.cumprod(alphas, dim=0)


class DiffusionProcess:
  def __init__(self, schedule: DiffusionSchedule) -> None:
    self.schedule = schedule

  def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None
               ) -> torch.Tensor:
    if noise is None:
      noise = torch.randn_like(x_start)
    alpha_bars = self.schedule.alpha_bars(x_start.device)
    sqrt_alpha = torch.sqrt(alpha_bars[timesteps])
    sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_bars[timesteps])
    while sqrt_alpha.ndim < x_start.ndim:
      sqrt_alpha = sqrt_alpha.unsqueeze(-1)
      sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
    return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

  def predict_start_from_noise(self, x_t: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    alpha_bars = self.schedule.alpha_bars(x_t.device)
    sqrt_alpha = torch.sqrt(alpha_bars[timesteps])
    sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_bars[timesteps])
    while sqrt_alpha.ndim < x_t.ndim:
      sqrt_alpha = sqrt_alpha.unsqueeze(-1)
      sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
    return (x_t - sqrt_one_minus_alpha * noise) / sqrt_alpha

  def p_losses(self, model, x_start: torch.Tensor, conditioning: torch.Tensor, timesteps: torch.Tensor,
               noise: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    if noise is None:
      noise = torch.randn_like(x_start)
    x_noisy = self.q_sample(x_start, timesteps, noise)
    predicted_noise = model(x_noisy, timesteps, conditioning)
    loss = torch.nn.functional.mse_loss(predicted_noise, noise)
    return loss, predicted_noise


__all__ = [
  "DiffusionSchedule",
  "DiffusionProcess",
]

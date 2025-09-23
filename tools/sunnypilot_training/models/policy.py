from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from openpilot.selfdrive.modeld.constants import ModelConstants

from .diffusion import DiffusionProcess, DiffusionSchedule
from .modules import sinusoidal_time_embedding


class PolicyConditionEncoder(nn.Module):
  def __init__(self, embed_dim: int) -> None:
    super().__init__()
    self.history_encoder = nn.GRU(ModelConstants.FEATURE_LEN, embed_dim, batch_first=True)
    self.desire_proj = nn.Linear(ModelConstants.DESIRE_LEN + ModelConstants.TRAFFIC_CONVENTION_LEN, embed_dim)
    self.final_proj = nn.Linear(embed_dim * 2, embed_dim)

  def forward(
    self,
    features_buffer: torch.Tensor,
    desire: torch.Tensor,
    traffic_convention: torch.Tensor,
  ) -> torch.Tensor:
    hist, _ = self.history_encoder(features_buffer)
    hist_vec = hist[:, -1]
    desire_context = torch.cat([desire[:, -1], traffic_convention], dim=-1)
    desire_vec = torch.relu(self.desire_proj(desire_context))
    cond = torch.cat([hist_vec, desire_vec], dim=-1)
    return torch.relu(self.final_proj(cond))


class PolicyDenoiser(nn.Module):
  def __init__(self, plan_dim: int, embed_dim: int, transformer_layers: int = 4, num_heads: int = 4) -> None:
    super().__init__()
    self.plan_dim = plan_dim
    self.embed_dim = embed_dim
    self.plan_proj = nn.Linear(plan_dim, embed_dim)
    encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
    self.condition_proj = nn.Linear(embed_dim, embed_dim)
    self.time_proj = nn.Sequential(
      nn.Linear(embed_dim, embed_dim),
      nn.GELU(),
      nn.Linear(embed_dim, embed_dim),
    )
    self.output_proj = nn.Linear(embed_dim, plan_dim)

  def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
    token = self.plan_proj(x_t).unsqueeze(1)
    time_emb = sinusoidal_time_embedding(timesteps, self.embed_dim)
    time_emb = self.time_proj(time_emb).unsqueeze(1)
    cond = self.condition_proj(conditioning).unsqueeze(1)
    tokens = token + time_emb
    tokens = torch.cat([cond, tokens], dim=1)
    encoded = self.encoder(tokens)
    return self.output_proj(encoded[:, 1, :])


class DiffusionPolicyModel(nn.Module):
  def __init__(self, embed_dim: int = 256, diffusion_steps: int = 20) -> None:
    super().__init__()
    self.plan_dim = ModelConstants.IDX_N * ModelConstants.PLAN_WIDTH
    self.condition_encoder = PolicyConditionEncoder(embed_dim)
    self.denoiser = PolicyDenoiser(self.plan_dim, embed_dim)
    self.schedule = DiffusionSchedule(timesteps=diffusion_steps)
    self.diffusion = DiffusionProcess(self.schedule)
    self.plan_mean_head = nn.Linear(embed_dim, self.plan_dim)
    self.plan_log_std_head = nn.Linear(embed_dim, self.plan_dim)
    self.desire_head = nn.Linear(embed_dim, ModelConstants.DESIRE_PRED_WIDTH)

  def forward(
    self,
    features_buffer: torch.Tensor,
    desire: torch.Tensor,
    traffic_convention: torch.Tensor,
  ) -> Dict[str, torch.Tensor]:
    conditioning = self.condition_encoder(features_buffer, desire, traffic_convention)
    plan_mu = self.plan_mean_head(conditioning).view(-1, ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH)
    plan_log_std = self.plan_log_std_head(conditioning).view_as(plan_mu)
    plan = torch.stack([plan_mu, plan_log_std], dim=-1)
    desire_state = self.desire_head(conditioning)
    return {
      "plan": plan,
      "desire_state": desire_state,
    }

  def sample(self, features_buffer: torch.Tensor, desire: torch.Tensor, traffic_convention: torch.Tensor) -> torch.Tensor:
    conditioning = self.condition_encoder(features_buffer, desire, traffic_convention)
    device = conditioning.device
    betas = self.schedule.betas(device)
    alphas = self.schedule.alphas(device)
    alpha_bars = self.schedule.alpha_bars(device)
    x_t = torch.randn(conditioning.shape[0], self.plan_dim, device=device)
    for t in reversed(range(self.schedule.timesteps)):
      timesteps = torch.full((conditioning.shape[0],), t, device=device, dtype=torch.long)
      noise_pred = self.denoiser(x_t, timesteps, conditioning)
      alpha = alphas[t]
      alpha_bar = alpha_bars[t]
      beta = betas[t]
      if t > 0:
        noise = torch.randn_like(x_t)
      else:
        noise = torch.zeros_like(x_t)
      x_t = (1.0 / torch.sqrt(alpha)) * (x_t - (beta / torch.sqrt(1 - alpha_bar)) * noise_pred) + torch.sqrt(beta) * noise
    return x_t.view(-1, ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH)

  def loss(
    self,
    target_plan: torch.Tensor,
    target_desire_state: torch.Tensor,
    features_buffer: torch.Tensor,
    desire: torch.Tensor,
    traffic_convention: torch.Tensor,
  ) -> torch.Tensor:
    conditioning = self.condition_encoder(features_buffer, desire, traffic_convention)
    plan_mu = self.plan_mean_head(conditioning).view(-1, ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH)
    plan_log_std = self.plan_log_std_head(conditioning).view_as(plan_mu)
    target_mu = target_plan[..., 0]
    target_log_std = target_plan[..., 1]
    supervised = F.mse_loss(plan_mu, target_mu) + F.mse_loss(plan_log_std, target_log_std)

    desire_logits = self.desire_head(conditioning)
    target_idx = torch.argmax(target_desire_state, dim=-1)
    desire_loss = F.cross_entropy(desire_logits, target_idx)

    batch_size = target_plan.shape[0]
    timesteps = torch.randint(0, self.schedule.timesteps, (batch_size,), device=target_plan.device)
    diff_loss, _ = self.diffusion.p_losses(self.denoiser, target_mu.view(batch_size, -1), conditioning, timesteps)
    return supervised + desire_loss + diff_loss


__all__ = [
  "DiffusionPolicyModel",
]

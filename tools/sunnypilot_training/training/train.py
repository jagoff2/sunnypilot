from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..dataset import CarlaZarrDataset
from ..models import DiffusionPolicyModel, DiffusionVisionModel

try:
  import hydra
  from hydra.utils import to_absolute_path
  from omegaconf import DictConfig, OmegaConf
except ImportError as exc:  # pragma: no cover
  raise ImportError("Hydra and OmegaConf are required for training. Install hydra-core on Windows.") from exc


LOG = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)


def build_dataloaders(cfg: DictConfig) -> Dict[str, DataLoader]:
  train_dataset = CarlaZarrDataset(Path(to_absolute_path(cfg.dataset.train_path)), augmentations=cfg.dataset.augmentations)
  val_dataset = CarlaZarrDataset(Path(to_absolute_path(cfg.dataset.val_path)), augmentations=None)
  train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    pin_memory=True,
  )
  val_loader = DataLoader(
    val_dataset,
    batch_size=cfg.val_batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=True,
  )
  return {"train": train_loader, "val": val_loader}


def compute_vision_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
  losses = []
  losses.append(F.mse_loss(outputs["pose"], batch["pose"]))
  losses.append(F.mse_loss(outputs["road_transform"], batch["road_transform"]))
  losses.append(F.mse_loss(outputs["lane_lines"], batch["lane_lines"]))
  losses.append(F.mse_loss(outputs["road_edges"], batch["road_edges"]))
  losses.append(F.mse_loss(outputs["lead"], batch["lead"]))
  losses.append(F.binary_cross_entropy_with_logits(outputs["lane_lines_prob"], batch["lane_lines_prob"]))
  losses.append(F.binary_cross_entropy_with_logits(outputs["lead_prob"], batch["lead_prob"]))
  losses.append(F.binary_cross_entropy_with_logits(outputs["meta"], batch["meta"]))
  logits = outputs["desire_pred"].view(-1, outputs["desire_pred"].shape[-1])
  targets = torch.argmax(batch["desire_pred"], dim=-1).view(-1)
  losses.append(F.cross_entropy(logits, targets))
  losses.append(F.mse_loss(outputs["hidden_state"], batch["hidden_state"]))
  return sum(losses)


def save_checkpoint(
  output_dir: Path,
  epoch: int,
  vision: DiffusionVisionModel,
  policy: DiffusionPolicyModel,
  optimizer: torch.optim.Optimizer,
) -> None:
  ckpt_dir = output_dir / "checkpoints"
  ckpt_dir.mkdir(parents=True, exist_ok=True)
  ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
  torch.save(
    {
      "epoch": epoch,
      "vision": vision.state_dict(),
      "policy": policy.state_dict(),
      "optimizer": optimizer.state_dict(),
    },
    ckpt_path,
  )
  LOG.info("Saved checkpoint to %s", ckpt_path)


def run_epoch(
  loader: DataLoader,
  vision: DiffusionVisionModel,
  policy: DiffusionPolicyModel,
  optimizer: torch.optim.Optimizer,
  scaler: torch.cuda.amp.GradScaler,
  device: torch.device,
  train: bool,
  cfg: DictConfig,
) -> Dict[str, float]:
  if train:
    vision.train()
    policy.train()
  else:
    vision.eval()
    policy.eval()

  total_loss = 0.0
  vision_loss_total = 0.0
  policy_loss_total = 0.0
  count = 0
  for step, batch in enumerate(loader):
    count += 1
    for key, value in batch.items():
      batch[key] = value.to(device)
    optimizer.zero_grad(set_to_none=True)

    with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and cfg.mixed_precision)):
      vision_outputs = vision(batch["img"], batch["big_img"])
      v_loss = compute_vision_loss(vision_outputs, batch)

      features_buffer = batch["features_buffer"].clone()
      features_buffer[:, -1, :] = vision_outputs["hidden_state"].detach()
      target_plan = batch["plan"]
      target_desire_state = batch["desire_state"]

      p_loss = policy.loss(
        target_plan,
        target_desire_state,
        features_buffer,
        batch["desire"],
        batch["traffic_convention"],
      )
      sampled_plan = policy.sample(features_buffer, batch["desire"], batch["traffic_convention"])
      p_loss = p_loss + F.mse_loss(sampled_plan, target_plan[..., 0])
      loss = v_loss + p_loss

    if train:
      scaler.scale(loss).backward()
      torch.nn.utils.clip_grad_norm_(list(vision.parameters()) + list(policy.parameters()), 1.0)
      scaler.step(optimizer)
      scaler.update()

    total_loss += loss.item()
    vision_loss_total += v_loss.item()
    policy_loss_total += p_loss.item()

    if train and (step + 1) % cfg.log_interval == 0:
      LOG.info("step %d vision_loss=%.4f policy_loss=%.4f", step + 1, v_loss.item(), p_loss.item())

  return {
    "loss": total_loss / count,
    "vision_loss": vision_loss_total / count,
    "policy_loss": policy_loss_total / count,
  }


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
  logging.basicConfig(level=logging.INFO)
  LOG.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
  set_seed(cfg.seed)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  dataloaders = build_dataloaders(cfg)
  vision = DiffusionVisionModel().to(device)
  policy = DiffusionPolicyModel(diffusion_steps=cfg.diffusion.steps).to(device)
  optimizer = torch.optim.AdamW(
    list(vision.parameters()) + list(policy.parameters()),
    lr=cfg.learning_rate,
    weight_decay=cfg.weight_decay,
  )
  scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and cfg.mixed_precision))

  output_dir = Path(os.getcwd())
  output_dir.mkdir(parents=True, exist_ok=True)

  for epoch in range(1, cfg.max_epochs + 1):
    train_metrics = run_epoch(dataloaders["train"], vision, policy, optimizer, scaler, device, True, cfg)
    with torch.no_grad():
      val_metrics = run_epoch(dataloaders["val"], vision, policy, optimizer, scaler, device, False, cfg)
    LOG.info(
      "epoch %d train_loss=%.4f val_loss=%.4f vision=%.4f policy=%.4f",
      epoch,
      train_metrics["loss"],
      val_metrics["loss"],
      val_metrics["vision_loss"],
      val_metrics["policy_loss"],
    )
    if epoch % cfg.checkpoint_interval == 0:
      save_checkpoint(output_dir, epoch, vision, policy, optimizer)


if __name__ == "__main__":
  main()

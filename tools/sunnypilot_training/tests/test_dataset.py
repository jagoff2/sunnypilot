from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from ..dataset import DEFAULT_SCHEMA, CarlaZarrDataset, ZarrShardWriter


def create_sample() -> dict[str, np.ndarray]:
  sample = {}
  for name, shape in DEFAULT_SCHEMA.vision_inputs.items():
    sample[name] = np.zeros(shape, dtype=np.float32)
  for name, shape in DEFAULT_SCHEMA.vision_targets.items():
    sample[name] = np.zeros(shape, dtype=np.float32)
  for name, shape in DEFAULT_SCHEMA.policy_targets.items():
    sample[name] = np.zeros(shape, dtype=np.float32)
  return sample


def test_zarr_roundtrip(tmp_path: Path) -> None:
  writer = ZarrShardWriter(tmp_path / "sample.zarr")
  writer.append(create_sample())
  writer.flush()
  dataset = CarlaZarrDataset(tmp_path / "sample.zarr")
  item = dataset[0]
  assert set(DEFAULT_SCHEMA.vision_inputs).issubset(item.keys())
  assert set(DEFAULT_SCHEMA.vision_targets).issubset(item.keys())
  assert set(DEFAULT_SCHEMA.policy_targets).issubset(item.keys())

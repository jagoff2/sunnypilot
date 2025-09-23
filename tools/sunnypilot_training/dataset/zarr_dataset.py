"""Zarr-backed dataset and writer for CARLA-generated samples."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, MutableMapping, Optional

import numpy as np
import torch
import zarr

from .schema import DEFAULT_SCHEMA, DatasetSchema
from .transforms import AUGMENTATIONS


class ZarrShardWriter:
  """Incrementally append samples to a Zarr store."""

  def __init__(self, path: Path, schema: DatasetSchema = DEFAULT_SCHEMA, chunk_size: int = 128) -> None:
    self.path = path
    self.schema = schema
    self.chunk_size = chunk_size
    self._store = zarr.DirectoryStore(str(path))
    self._root = zarr.group(store=self._store, overwrite=False)
    self._arrays: Dict[str, zarr.Array] = {}
    self._init_arrays()

  def _init_arrays(self) -> None:
    specs: Dict[str, tuple[int, ...]] = {}
    specs.update(self.schema.vision_inputs)
    specs.update(self.schema.policy_inputs)
    specs.update(self.schema.vision_targets)
    specs.update(self.schema.policy_targets)
    for name, shape in specs.items():
      array_shape = (0,) + shape
      chunks = (min(self.chunk_size, 64),) + shape
      dtype = np.float16 if name in {"img", "big_img"} else np.float32
      if name in self._root:
        self._arrays[name] = self._root[name]
        continue
      compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.SHUFFLE)
      self._arrays[name] = self._root.create(
        name,
        shape=array_shape,
        chunks=chunks,
        dtype=dtype,
        compressor=compressor,
      )
    self._root.attrs["schema"] = json.dumps({
      "vision_inputs": self.schema.vision_inputs,
      "policy_inputs": self.schema.policy_inputs,
      "vision_targets": self.schema.vision_targets,
      "policy_targets": self.schema.policy_targets,
    })

  def append(self, sample: MutableMapping[str, np.ndarray]) -> None:
    for key, array in self._arrays.items():
      if key not in sample:
        raise KeyError(f"Missing required key {key} in sample")
      array.append(np.asarray(sample[key])[None, ...])

  @property
  def size(self) -> int:
    first_key = next(iter(self._arrays))
    return int(self._arrays[first_key].shape[0])

  def flush(self) -> None:
    for array in self._arrays.values():
      array.flush()


class CarlaZarrDataset(torch.utils.data.Dataset):
  """PyTorch dataset streaming from a Zarr store."""

  def __init__(
    self,
    path: Path,
    schema: DatasetSchema = DEFAULT_SCHEMA,
    augmentations: Optional[Iterable[str]] = None,
    rng: Optional[np.random.Generator] = None,
  ) -> None:
    self.path = path
    self.schema = schema
    self._store = zarr.DirectoryStore(str(path))
    self._root = zarr.open_group(self._store, mode="r")
    self._arrays = {key: self._root[key] for key in self._root.array_keys()}
    self._size = self._arrays["img"].shape[0]
    self.augmentations = [AUGMENTATIONS[name] for name in augmentations or []]
    self.rng = rng or np.random.default_rng()

  def __len__(self) -> int:
    return self._size

  def _apply_augmentations(self, frames: np.ndarray) -> np.ndarray:
    augmented = frames
    for aug in self.augmentations:
      augmented = aug(augmented, rng=self.rng)
    return augmented

  def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array.astype(np.float32))

  def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
    sample: Dict[str, torch.Tensor] = {}
    for name in self.schema.vision_inputs:
      arr = np.asarray(self._arrays[name][index])
      if name in {"img", "big_img"} and self.augmentations:
        arr = self._apply_augmentations(arr)
      sample[name] = self._to_tensor(arr)
    for name in self.schema.policy_inputs:
      arr = np.asarray(self._arrays[name][index])
      sample[name] = self._to_tensor(arr)
    for name in self.schema.vision_targets:
      arr = np.asarray(self._arrays[name][index])
      sample[name] = self._to_tensor(arr)
    for name in self.schema.policy_targets:
      arr = np.asarray(self._arrays[name][index])
      sample[name] = self._to_tensor(arr)
    return sample


def iter_zarr(path: Path, batch_size: int, keys: Iterable[str]) -> Iterator[Dict[str, np.ndarray]]:
  """Utility generator to iterate over Zarr arrays in batches."""
  store = zarr.DirectoryStore(str(path))
  root = zarr.open_group(store, mode="r")
  key_list = list(keys)
  if not key_list:
    raise ValueError("keys must not be empty")
  size = root[key_list[0]].shape[0]
  for start in range(0, size, batch_size):
    end = min(start + batch_size, size)
    yield {key: np.asarray(root[key][start:end]) for key in key_list}


__all__ = [
  "ZarrShardWriter",
  "CarlaZarrDataset",
  "iter_zarr",
]

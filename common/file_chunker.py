import math
import os
import pickle
import shutil
from pathlib import Path

CHUNK_SIZE = 45 * 1024 * 1024  # 45MB, under GitHub's 50MB limit

def get_chunk_name(name, idx, num_chunks):
  return f"{name}.chunk{idx+1:02d}of{num_chunks:02d}"

def get_manifest_path(name):
  return f"{name}.chunkmanifest"

def get_chunk_paths(path, file_size):
  num_chunks = math.ceil(file_size / CHUNK_SIZE)
  return [get_manifest_path(path)] + [get_chunk_name(path, i, num_chunks) for i in range(num_chunks)]

def chunk_file(path, targets):
  manifest_path, *chunk_paths = targets
  with open(path, 'rb') as f:
    data = f.read()
  actual_num_chunks = max(1, math.ceil(len(data) / CHUNK_SIZE))
  assert len(chunk_paths) >= actual_num_chunks, f"Allowed {len(chunk_paths)} chunks but needs at least {actual_num_chunks}, for path {path}"
  for i, chunk_path in enumerate(chunk_paths):
    with open(chunk_path, 'wb') as f:
      f.write(data[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE])
  Path(manifest_path).write_text(str(len(chunk_paths)))


def materialize_file_chunked(path):
  path = Path(path)
  manifest_path = Path(get_manifest_path(path))
  if not manifest_path.is_file():
    if path.is_file():
      return path
    raise FileNotFoundError(path)

  num_chunks = int(manifest_path.read_text().strip())
  chunk_paths = [Path(get_chunk_name(path, i, num_chunks)) for i in range(num_chunks)]

  if path.is_file():
    path_mtime = path.stat().st_mtime_ns
    newest_chunk_mtime = max([manifest_path.stat().st_mtime_ns, *[chunk.stat().st_mtime_ns for chunk in chunk_paths]])
    if path_mtime >= newest_chunk_mtime:
      return path

  tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
  try:
    with open(tmp_path, 'wb') as out:
      for chunk_path in chunk_paths:
        with open(chunk_path, 'rb') as chunk:
          shutil.copyfileobj(chunk, out, length=1024 * 1024)
    os.replace(tmp_path, path)
  except Exception:
    tmp_path.unlink(missing_ok=True)
    raise
  return path


def read_file_chunked(path):
  path = materialize_file_chunked(path) if not os.path.isfile(path) else Path(path)
  return path.read_bytes()


def load_pickle_chunked(path):
  with open(materialize_file_chunked(path), 'rb') as f:
    return pickle.load(f)

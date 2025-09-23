#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from huggingface_hub import HfApi, snapshot_download


DEFAULT_PATTERNS: tuple[str, ...] = (
  "*.ckpt",
  "*.pt",
  "*.pth",
  "*.onnx",
  "*.bin",
  "*.yaml",
  "*.yml",
)
DEFAULT_IGNORE: tuple[str, ...] = (
  "*.md5",
  "*.sha*",
  "*.json",
  "*.jsonl",
  "*.zip",
)


def _print_summary(local_dir: Path, repo_id: str, revision: str | None) -> None:
  files = sorted(p for p in local_dir.rglob('*') if p.is_file())
  total_size = sum(p.stat().st_size for p in files)
  print(f"Fetched {len(files)} files ({total_size / (1024 ** 2):.2f} MiB) into {local_dir}")
  print(f"Repo: {repo_id} Revision: {revision or 'default'}")
  for path in files:
    rel = path.relative_to(local_dir)
    print(f"  - {rel} ({path.stat().st_size / (1024 ** 2):.2f} MiB)")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Download DiffusionDrive weights from Hugging Face")
  parser.add_argument("--repo", default="hustvl/DiffusionDrive", help="Model repository ID")
  parser.add_argument("--revision", help="Optional git revision/tag")
  parser.add_argument("--local-dir", required=True, help="Destination directory for the snapshot")
  parser.add_argument("--cache-dir", help="Override Hugging Face cache directory")
  parser.add_argument("--pattern", dest="patterns", action="append", help="Allow pattern (glob). Can be repeated.")
  parser.add_argument("--ignore", dest="ignore_patterns", action="append", help="Ignore pattern (glob). Can be repeated.")
  parser.add_argument("--token", help="Hugging Face token if authentication is required")
  parser.add_argument("--max-workers", type=int, help="Parallel workers for downloads")
  parser.add_argument("--list-only", action="store_true", help="List available files without downloading")
  return parser.parse_args()


def list_repo_contents(repo_id: str, revision: str | None, token: str | None) -> None:
  api = HfApi(token=token)
  siblings = api.list_repo_files(repo_id, revision=revision, repo_type="model")
  print(f"Files available in {repo_id} ({revision or 'default'}):")
  for entry in siblings:
    print(f"  - {entry}")


def main() -> None:
  args = parse_args()
  token = args.token or os.getenv("HUGGINGFACEHUB_API_TOKEN")

  if args.list_only:
    list_repo_contents(args.repo, args.revision, token)
    return

  allow_patterns: Sequence[str] = args.patterns if args.patterns else DEFAULT_PATTERNS
  ignore_patterns: Sequence[str] = args.ignore_patterns if args.ignore_patterns else DEFAULT_IGNORE

  local_dir = Path(args.local_dir).expanduser().resolve()
  local_dir.mkdir(parents=True, exist_ok=True)

  if token:
    os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", token)

  snapshot_download(
    repo_id=args.repo,
    revision=args.revision,
    allow_patterns=list(allow_patterns),
    ignore_patterns=list(ignore_patterns),
    local_dir=str(local_dir),
    local_dir_use_symlinks=False,
    cache_dir=args.cache_dir,
    max_workers=args.max_workers,
    repo_type="model",
  )

  _print_summary(local_dir, args.repo, args.revision)


if __name__ == "__main__":
  main()

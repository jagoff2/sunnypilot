"""sunnypilot training package bootstrap."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_on_path() -> None:
  """Add the repository root to ``sys.path`` if it isn't already present.

  The training utilities import modules from the main ``openpilot`` package.
  When the tools are executed as a module (``python -m``) Python only adds the
  directory containing ``tools`` to ``sys.path``.  On Windows this meant the
  repository root was not available for imports, causing
  ``ModuleNotFoundError: openpilot`` during the training pipeline.  Doing the
  adjustment here guarantees all entry points see the correct module search
  path without relying on shell specific environment configuration.
  """

  repo_root = Path(__file__).resolve().parents[2]
  repo_root_str = str(repo_root)
  if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)


_ensure_repo_on_path()

__all__ = []
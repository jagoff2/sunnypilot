#!/usr/bin/env python3
"""Summarize NNLC sigmoid map JSON files.

This helper inspects sigmoid map slices (speed, slope, intercept) and reports
coverage against an expected speed grid along with basic slope/intercept stats.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

MPH_TO_MS = 0.44704
MS_TO_MPH = 1.0 / MPH_TO_MS


def _load_slices(path: Path) -> List[dict]:
  with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
  slices = data.get("slices", [])
  if not isinstance(slices, list):
    raise ValueError("Expected 'slices' list in map JSON")
  return slices


def _expected_centers(min_speed_mph: float, max_speed_mph: float, step_mph: float) -> List[float]:
  centers: List[float] = []
  current = min_speed_mph
  while current <= max_speed_mph + 1e-6:
    centers.append(current)
    current += step_mph
  return centers


def _missing_speeds(actual_mph: Iterable[float], expected_mph: Iterable[float], tolerance_mph: float) -> List[float]:
  actual_sorted = sorted(actual_mph)
  missing: List[float] = []
  for target in expected_mph:
    found = any(abs(target - value) <= tolerance_mph for value in actual_sorted)
    if not found:
      missing.append(target)
  return missing


def _describe(values: List[float]) -> str:
  if not values:
    return "n/a"
  minimum = min(values)
  maximum = max(values)
  mean = sum(values) / len(values)
  return f"min={minimum:.3f}, max={maximum:.3f}, mean={mean:.3f}"


def summarize_map(path: Path, min_speed_mph: float, max_speed_mph: float, step_mph: float, tolerance_mph: float) -> str:
  slices = _load_slices(path)
  if not slices:
    return "No slices found in map JSON."

  speeds_mph = [slice_["speed"] * MS_TO_MPH for slice_ in slices]
  slopes = [slice_["slope"] for slice_ in slices]
  intercepts = [slice_["intercept"] for slice_ in slices]

  expected = _expected_centers(min_speed_mph, max_speed_mph, step_mph)
  missing = _missing_speeds(speeds_mph, expected, tolerance_mph)
  coverage = 100.0 * (1.0 - len(missing) / len(expected)) if expected else 0.0

  lines = [
    f"File: {path}",
    f"Slices: {len(slices)} present, coverage {coverage:.1f}% ({len(expected) - len(missing)}/{len(expected)})",
    f"Speed centers (mph): { _describe(speeds_mph) }",
    f"Slope stats: { _describe(slopes) }",
    f"Intercept stats: { _describe(intercepts) }",
  ]

  if missing:
    missing_str = ", ".join(f"{m:.1f}" for m in missing)
    lines.append(f"Missing expected speed centers (mph): {missing_str}")

  return "\n".join(lines)


def main() -> None:
  parser = argparse.ArgumentParser(description="Summarize an NNLC sigmoid map JSON file")
  parser.add_argument("map_json", type=Path, help="Path to the sigmoid map JSON")
  parser.add_argument("--min-speed", type=float, default=5.0, help="Minimum expected speed center in mph (default: 5)")
  parser.add_argument("--max-speed", type=float, default=70.0, help="Maximum expected speed center in mph (default: 70)")
  parser.add_argument("--step", type=float, default=5.0, help="Expected speed center step in mph (default: 5)")
  parser.add_argument("--tolerance", type=float, default=2.5, help="Tolerance when matching expected centers in mph (default: half a bin)")
  args = parser.parse_args()

  summary = summarize_map(args.map_json, args.min_speed, args.max_speed, args.step, args.tolerance)
  print(summary)


if __name__ == "__main__":
  main()

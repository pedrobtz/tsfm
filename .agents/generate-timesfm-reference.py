#!/usr/bin/env python3
"""Generate compact TimesFM 2.5 forecast fixtures from pinned official code.

Required environment variables:
  TIMESFM_SOURCE       checkout of google-research/timesfm at SOURCE_COMMIT
  TIMESFM_CHECKPOINT   directory containing model.safetensors, or that file

Usage:
  python .agents/generate-timesfm-reference.py tests/testthat/fixtures/timesfm
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

SOURCE_COMMIT = "3dae50b20d7a724981e8ea36cda75578f80dd2dc"
CHECKPOINT_ID = "google/timesfm-2.5-200m-pytorch"
CHECKPOINT_REVISION = "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def assert_source_pin(source: Path) -> None:
  actual = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=source, text=True
  ).strip()
  if actual != SOURCE_COMMIT:
    raise RuntimeError(f"TimesFM source is {actual}; expected {SOURCE_COMMIT}")


def generated_context(spec: dict) -> np.ndarray:
  kind = spec["kind"]
  length = int(spec["length"])
  x = np.arange(1, length + 1, dtype=np.float32)
  if kind == "trend_seasonal":
    return (100.0 + 0.15 * x + 4.0 * np.sin(x * 2.0 * np.pi / 12.0)).astype(
      np.float32
    )
  if kind == "long_trend_seasonal":
    return (50.0 + 0.002 * x + 2.0 * np.sin(x * 2.0 * np.pi / 24.0)).astype(
      np.float32
    )
  if kind == "mixed_sign":
    # Crosses zero repeatedly and ends below it. The official decoder only
    # clamps forecasts to be non-negative when every observed value is
    # non-negative, so a series like this is the one that exercises the other
    # branch --- and every other fixture here is strictly positive.
    return (3.0 * np.sin(x * 2.0 * np.pi / 16.0) - 0.05 * x).astype(np.float32)
  raise ValueError(f"Unknown context generator: {kind}")


def round_up(value: int, multiple: int) -> int:
  return ((value + multiple - 1) // multiple) * multiple


def fixture_cases() -> list[dict]:
  return [
    {
      "name": "typical",
      "contexts": [
        {"kind": "trend_seasonal", "length": 96},
      ],
      "horizon": 12,
      "max_context": 96,
    },
    {
      "name": "short_context",
      "contexts": [
        {"values": [3.0, 4.5, 4.0]},
      ],
      "horizon": 8,
      "max_context": 32,
    },
    {
      "name": "context_truncation",
      "contexts": [
        {"kind": "long_trend_seasonal", "length": 16270},
      ],
      "horizon": 4,
      # The official decoder rounds any positive horizon to a 128-value output
      # patch. 16,256 is therefore the largest context with context+horizon
      # still inside the 16,384 token limit.
      "max_context": 16256,
    },
    {
      "name": "mixed_sign",
      "contexts": [
        {"kind": "mixed_sign", "length": 96},
      ],
      "horizon": 12,
      "max_context": 96,
    },
    {
      "name": "batch_agreement",
      "contexts": [
        {"kind": "trend_seasonal", "length": 64},
        {"values": [20.0 + 0.25 * i for i in range(1, 49)]},
      ],
      "horizon": 6,
      "max_context": 64,
    },
  ]


def materialize_context(spec: dict) -> np.ndarray:
  if "values" in spec:
    return np.asarray(spec["values"], dtype=np.float32)
  return generated_context(spec)


def main() -> None:
  if len(sys.argv) != 2:
    raise SystemExit("usage: generate-timesfm-reference.py OUTPUT_DIR")
  source = Path(os.environ["TIMESFM_SOURCE"]).resolve()
  checkpoint = Path(os.environ["TIMESFM_CHECKPOINT"]).resolve()
  output_dir = Path(sys.argv[1]).resolve()
  assert_source_pin(source)
  sys.path.insert(0, str(source / "src"))

  from timesfm import configs  # pylint: disable=import-error,import-outside-toplevel
  from timesfm.timesfm_2p5.timesfm_2p5_torch import (  # pylint: disable=import-error,import-outside-toplevel
    TimesFM_2p5_200M_torch,
  )

  model = TimesFM_2p5_200M_torch(torch_compile=False)
  model.load_checkpoint(str(checkpoint), torch_compile=False)
  output_dir.mkdir(parents=True, exist_ok=True)

  for case in fixture_cases():
    contexts = [materialize_context(spec) for spec in case["contexts"]]
    context_files = []
    for index, context in enumerate(contexts, start=1):
      context_file = f"{case['name']}-context-{index}.f32"
      np.asarray(context, dtype="<f4").tofile(output_dir / context_file)
      context_files.append(context_file)
    forecast_config = configs.ForecastConfig(
        max_context=case["max_context"],
        max_horizon=round_up(case["horizon"], 128),
        per_core_batch_size=len(contexts),
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
      )
    model.compile(forecast_config)
    point, full = model.forecast(case["horizon"], list(contexts))
    payload = {
      "schema_version": 1,
      "name": case["name"],
      "source_commit": SOURCE_COMMIT,
      "model_id": CHECKPOINT_ID,
      "revision": CHECKPOINT_REVISION,
      "reference": {
        "numpy": np.__version__,
        "torch": __import__("torch").__version__,
        "use_continuous_quantile_head": True,
        "force_flip_invariance": True,
        "infer_is_positive": True,
        "fix_quantile_crossing": True,
      },
      "context_specs": case["contexts"],
      "context_files": context_files,
      "max_context": case["max_context"],
      "horizon": case["horizon"],
      "quantile_levels": QUANTILE_LEVELS,
      "expected_point": point.tolist(),
      # Channel zero is the mean and channels 1:10 are the trained quantiles.
      "expected_quantiles": full[..., 1:].tolist(),
      # Compared as |actual - expected| <= atol + rtol * |expected|, the
      # criterion torch.testing.assert_close() and the upstream TimesFM tests
      # use. A single absolute number is the wrong shape here: these forecasts
      # sit near 116, where 1e-4 absolute is 8.6e-7 relative --- tighter than
      # PyTorch's own float32 default --- while the same number would be very
      # loose on a series near zero.
      "atol": 1e-4,
      "rtol": 1e-5,
    }
    if case["name"] == "batch_agreement":
      model.compile(
        configs.ForecastConfig(
          **{
            **forecast_config.__dict__,
            "per_core_batch_size": 1,
          }
        )
      )
      loop_point = []
      loop_full = []
      for context in contexts:
        each_point, each_full = model.forecast(case["horizon"], [context])
        loop_point.append(each_point[0])
        loop_full.append(each_full[0])
      loop_point = np.asarray(loop_point)
      loop_full = np.asarray(loop_full)
      max_difference = max(
        float(np.max(np.abs(point - loop_point))),
        float(np.max(np.abs(full - loop_full))),
      )
      if max_difference > 1e-4:
        raise RuntimeError(f"batch/loop reference mismatch: {max_difference}")
      payload["batch_loop_max_abs_difference"] = max_difference
    path = output_dir / f"{case['name']}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
  main()

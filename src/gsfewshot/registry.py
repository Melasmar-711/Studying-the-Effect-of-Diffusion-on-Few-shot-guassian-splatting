"""Experiment registry — the append-only record of every trained run.

One JSON object per line in ``results/registry.jsonl``. This is the single
source of truth the comparison tooling reads. Re-running an experiment appends
a new row; the latest row per ``exp_id`` wins when de-duplicating.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import REGISTRY_PATH

# Columns that define the "shape" of a run record. Extra keys are allowed.
CORE_FIELDS = [
    "exp_id", "scene", "kind", "n_real", "ratio_pct", "strategies",
    "s_synth", "total_images", "seed",
    "psnr", "ssim", "lpips",
    "train_time_s", "peak_vram_mb", "num_test_views",
    "downscale_factor", "max_num_iterations",
    "status", "timestamp", "config_path", "output_dir",
]


def record_run(record: dict[str, Any], path: Path | None = None) -> None:
    """Append a run record to the registry (adds a UTC timestamp)."""
    path = path or REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_runs(path: Path | None = None) -> list[dict[str, Any]]:
    """All run records, oldest first."""
    path = path or REGISTRY_PATH
    if not path.exists():
        return []
    runs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    return runs


def latest_runs(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Latest record per exp_id (later lines override earlier ones)."""
    out: dict[str, dict[str, Any]] = {}
    for r in load_runs(path):
        out[r["exp_id"]] = r
    return out


def to_dataframe(path: Path | None = None, latest_only: bool = True):
    """Return a pandas DataFrame of runs (import pandas lazily)."""
    import pandas as pd

    runs = list(latest_runs(path).values()) if latest_only else load_runs(path)
    df = pd.DataFrame(runs)
    if not df.empty and "strategies" in df:
        df["strategies"] = df["strategies"].apply(
            lambda s: "+".join(s) if isinstance(s, list) else (s or "")
        )
    return df

"""Project configuration + canonical paths.

All paths are derived from the repository root so scripts can be run from
anywhere. Reads ``configs/project.yaml`` once and exposes typed accessors.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# repo root = two levels up from this file (src/gsfewshot/config.py -> repo)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Canonical workspace locations -------------------------------------------------
CONFIG_DIR = PROJECT_ROOT / "configs"
EXPERIMENT_CONFIG_DIR = CONFIG_DIR / "experiments"
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = PROJECT_ROOT / "results"
REGISTRY_PATH = RESULTS_DIR / "registry.jsonl"
COMPARISONS_DIR = RESULTS_DIR / "comparisons"


def round_half_up(x: float) -> int:
    """Deterministic round-half-up (matches the S values in the plan table)."""
    return int(math.floor(x + 0.5))


def compute_s(n: int, ratio_pct: int, round_mode: str = "half_up") -> int:
    """Number of synthetic images S for a given real count N and ratio (percent).

    ratio 0 -> 0 synthetic. Otherwise at least 1. Uses round-half-up so the
    ambiguous rows in the plan (e.g. N=5, 50% -> 3) resolve deterministically.
    """
    if ratio_pct == 0:
        return 0
    raw = n * ratio_pct / 100.0
    s = round_half_up(raw) if round_mode == "half_up" else round(raw)
    return max(1, s)


@dataclass
class Config:
    raw: dict[str, Any]

    # -- convenience accessors --------------------------------------------------
    @property
    def project_name(self) -> str:
        return self.raw["project_name"]

    @property
    def scenes(self) -> dict[str, Any]:
        return self.raw["scenes"]

    def scene(self, name: str) -> dict[str, Any]:
        if name not in self.scenes:
            raise KeyError(f"unknown scene '{name}'. Known: {list(self.scenes)}")
        return self.scenes[name]

    def scene_source(self, name: str) -> Path:
        return (PROJECT_ROOT / self.scene(name)["source"]).resolve()

    def scene_transforms(self, name: str) -> Path:
        return (PROJECT_ROOT / self.scene(name)["transforms"]).resolve()

    @property
    def splits(self) -> dict[str, Any]:
        return self.raw["splits"]

    @property
    def grid(self) -> dict[str, Any]:
        return self.raw["grid"]

    @property
    def diffusion(self) -> dict[str, Any]:
        return self.raw["diffusion"]

    @property
    def training(self) -> dict[str, Any]:
        return self.raw["training"]

    @property
    def eval(self) -> dict[str, Any]:
        return self.raw["eval"]

    def compute_s(self, n: int, ratio_pct: int) -> int:
        return compute_s(n, ratio_pct, self.grid.get("round_mode", "half_up"))


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else CONFIG_DIR / "project.yaml"
    with open(path) as f:
        return Config(yaml.safe_load(f))

"""Experiment grid — encodes the Phase-3 matrix from the Experimental Plan.

Enumerates every planned run as an :class:`Experiment`, gives each a stable
``exp_id``, and can materialise them as per-experiment YAML configs plus an
index CSV. This is the machine-readable form of the plan's combination table.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml

from .config import Config, EXPERIMENT_CONFIG_DIR


def _purpose(n: int, ratio: int) -> str:
    if ratio == 0:
        return "Few-shot baseline (0% synthetic control)"
    if ratio == 25:
        return "Extremely low synthetic impact"
    if ratio == 50:
        return "Minority synthetic"
    if ratio == 100:
        return "1:1 real-to-synthetic"
    if ratio == 200:
        return "Synthetic outweighs real"
    return ""


@dataclass
class Experiment:
    exp_id: str
    scene: str
    kind: str            # "full" | "fewshot_baseline" | "augmented"
    n: int               # real images
    ratio: int           # synthetic:real percent
    strategies: list[str]
    s: int               # synthetic images
    total: int           # total training images
    purpose: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _strategy_tag(strategies: list[str]) -> str:
    return "+".join(strategies)


def enumerate_experiments(cfg: Config, scene: str) -> list[Experiment]:
    """Full experiment list for one scene, in a sensible run order."""
    g = cfg.grid
    sizes = cfg.splits["few_shot_sizes"]
    ratios = g["ratios"]
    strategies = g["strategies"]
    combos: list[list[str]] = [list(c) for c in g.get("strategy_combos", [])]

    exps: list[Experiment] = []

    # Phase 1a: full (N>=100) upper-bound baseline
    if g.get("include_full_baseline", True):
        exps.append(Experiment(
            exp_id=f"{scene}__full", scene=scene, kind="full",
            n=-1, ratio=0, strategies=[], s=0, total=-1,
            purpose="Full-data upper bound (N>=100 real images)",
        ))

    # Phase 1b: few-shot baselines (0% synthetic)
    for n in sizes:
        exps.append(Experiment(
            exp_id=f"{scene}__n{n}_r0", scene=scene, kind="fewshot_baseline",
            n=n, ratio=0, strategies=[], s=0, total=n, purpose=_purpose(n, 0),
        ))

    # Phase 2/3: augmented runs — every (N, ratio>0, strategy) cell + combos
    for n in sizes:
        for ratio in ratios:
            if ratio == 0:
                continue
            s = cfg.compute_s(n, ratio)
            for strat in [[st] for st in strategies] + combos:
                tag = _strategy_tag(strat)
                exps.append(Experiment(
                    exp_id=f"{scene}__n{n}_r{ratio}_{tag}", scene=scene,
                    kind="augmented", n=n, ratio=ratio, strategies=strat,
                    s=s, total=n + s, purpose=_purpose(n, ratio),
                ))
    return exps


def write_configs(cfg: Config, scene: str, out_dir: Path | None = None) -> list[Path]:
    """Write one YAML per experiment + an index.csv. Returns written paths."""
    out_dir = out_dir or EXPERIMENT_CONFIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    exps = enumerate_experiments(cfg, scene)

    written: list[Path] = []
    for e in exps:
        p = out_dir / f"{e.exp_id}.yaml"
        with open(p, "w") as f:
            yaml.safe_dump(e.to_dict(), f, sort_keys=False)
        written.append(p)

    # index.csv — human overview of the whole grid
    index = out_dir / "index.csv"
    with open(index, "w") as f:
        f.write("exp_id,scene,kind,n_real,ratio_pct,strategies,s_synth,total,purpose\n")
        for e in exps:
            f.write(
                f"{e.exp_id},{e.scene},{e.kind},{e.n},{e.ratio},"
                f"{_strategy_tag(e.strategies)},{e.s},{e.total},\"{e.purpose}\"\n"
            )
    written.append(index)
    return written


def load_experiment(exp_id: str, cfg: Config | None = None) -> Experiment:
    """Load a single experiment definition from its YAML config."""
    p = EXPERIMENT_CONFIG_DIR / f"{exp_id}.yaml"
    if not p.exists():
        raise FileNotFoundError(
            f"no config for '{exp_id}' at {p}. Run scripts/gen_grid.py first."
        )
    with open(p) as f:
        return Experiment(**yaml.safe_load(f))

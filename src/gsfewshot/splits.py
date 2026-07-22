"""Deterministic dataset splitting.

Produces, per scene:
  * a FIXED held-out test set of real views (every k-th frame) used to score
    *every* experiment identically, so PSNR/SSIM are comparable across runs;
  * a `full` train pool (the remaining real frames, N>=100 -> upper bound);
  * nested few-shot subsets (n5 subset of n10 subset of n20) drawn from the pool.

Stdlib only (json/math/shutil) — importable without torch/nerfstudio.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .config import Config, SPLITS_DIR, PROJECT_ROOT

# top-level camera-intrinsic / meta keys copied verbatim into each split file
_META_KEYS = (
    "w", "h", "fl_x", "fl_y", "cx", "cy", "k1", "k2", "k3", "k4",
    "p1", "p2", "camera_model", "applied_transform", "ply_file_path",
)


def linspace_indices(n_total: int, k: int) -> list[int]:
    """k distinct indices evenly spread across [0, n_total)."""
    if k >= n_total:
        return list(range(n_total))
    if k == 1:
        return [n_total // 2]
    step = (n_total - 1) / (k - 1)
    idx = sorted({int(round(i * step)) for i in range(k)})
    # pad if rounding collapsed any duplicates
    i = 0
    while len(idx) < k and i < n_total:
        if i not in idx:
            idx.append(i)
        i += 1
    return sorted(idx)[:k]


def _abs_image_path(scene_source: Path, file_path: str) -> str:
    """Resolve a transforms.json file_path to an absolute path on disk."""
    p = Path(file_path)
    if p.is_absolute():
        return str(p)
    return str((scene_source / file_path).resolve())


def _write_split(dst_dir: Path, meta: dict[str, Any], frames: list[dict],
                 scene_source: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    out = {k: meta[k] for k in _META_KEYS if k in meta}
    out_frames = []
    for fr in frames:
        fr2 = dict(fr)
        fr2["file_path"] = _abs_image_path(scene_source, fr["file_path"])
        out_frames.append(fr2)
    out["frames"] = out_frames
    with open(dst_dir / "transforms.json", "w") as f:
        json.dump(out, f, indent=2)


def make_splits(cfg: Config, scene: str, overwrite: bool = False) -> dict[str, Any]:
    """Create all splits for a scene. Returns the manifest dict."""
    transforms_path = cfg.scene_transforms(scene)
    scene_source = cfg.scene_source(scene)
    with open(transforms_path) as f:
        meta = json.load(f)

    frames = sorted(meta["frames"], key=lambda fr: fr["file_path"])
    n_frames = len(frames)

    k = int(cfg.splits["holdout_every"])
    sizes = sorted(cfg.splits["few_shot_sizes"], reverse=True)
    nested = bool(cfg.splits.get("nested", True))

    # 1) fixed held-out test set: every k-th frame
    test_pos = [i for i in range(n_frames) if i % k == 0]
    pool_pos = [i for i in range(n_frames) if i % k != 0]
    if len(pool_pos) < 100:
        raise ValueError(
            f"train pool has only {len(pool_pos)} frames (<100). "
            f"Lower splits.holdout_every (currently {k})."
        )

    # 2) few-shot subsets from the pool
    subset_pos: dict[int, list[int]] = {}
    if nested:
        largest = sizes[0]
        sel = [pool_pos[j] for j in linspace_indices(len(pool_pos), largest)]
        subset_pos[largest] = sel
        for n in sizes[1:]:
            sub = [sel[j] for j in linspace_indices(len(sel), n)]
            subset_pos[n] = sub
            sel = sub
    else:
        for n in sizes:
            subset_pos[n] = [pool_pos[j] for j in linspace_indices(len(pool_pos), n)]

    # 3) write everything
    scene_dir = SPLITS_DIR / scene
    if scene_dir.exists() and overwrite:
        shutil.rmtree(scene_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)

    _write_split(scene_dir / "test", meta, [frames[i] for i in test_pos], scene_source)
    _write_split(scene_dir / "full", meta, [frames[i] for i in pool_pos], scene_source)
    for n, pos in subset_pos.items():
        _write_split(scene_dir / f"n{n}", meta, [frames[i] for i in pos], scene_source)

    manifest = {
        "scene": scene,
        "source_transforms": str(transforms_path),
        "n_frames": n_frames,
        "holdout_every": k,
        "nested": nested,
        "resolution": {"w": meta.get("w"), "h": meta.get("h")},
        "test": {"count": len(test_pos),
                 "files": [frames[i]["file_path"] for i in test_pos]},
        "full": {"count": len(pool_pos)},
        "subsets": {f"n{n}": {"count": len(pos),
                              "files": [frames[i]["file_path"] for i in pos]}
                    for n, pos in subset_pos.items()},
    }
    with open(scene_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def split_dir(scene: str, name: str) -> Path:
    """Path to a split's transforms dir. name in {test, full, n5, n10, ...}."""
    return SPLITS_DIR / scene / name


def load_split_frames(scene: str, name: str) -> tuple[dict, list[dict]]:
    """Load (meta, frames) for a split."""
    p = split_dir(scene, name) / "transforms.json"
    with open(p) as f:
        d = json.load(f)
    return d, d["frames"]

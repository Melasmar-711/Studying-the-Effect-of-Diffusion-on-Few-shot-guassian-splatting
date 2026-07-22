"""Assemble a self-contained Nerfstudio data dir for one experiment.

Given an :class:`Experiment`, gather the real few-shot frames + the first S
synthetic frames and emit a clean, local directory::

    <out>/images/*.png        (symlinks: real + synthetic)
    <out>/transforms.json     (local file_paths; per-frame intrinsics for synth)

Real frames use the scene's global intrinsics; each synthetic frame carries its
own intrinsics/pose from its sidecar JSON (poses for synthetic views are
approximate by construction — see synthetic.py). Held-out test frames are never
included here; evaluation renders them separately.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import Config, SYNTHETIC_DIR
from .grid import Experiment
from .splits import load_split_frames

_META_KEYS = (
    "w", "h", "fl_x", "fl_y", "cx", "cy", "k1", "k2", "k3", "k4",
    "p1", "p2", "camera_model", "applied_transform",
)
# nerfstudio uses one global image size for all cameras, but honours per-frame
# focal length / principal point — so we override those, not w/h.
_PERFRAME_INTRINSICS = ("fl_x", "fl_y", "cx", "cy")


def synthetic_pool_dir(scene: str, n: int, strategy: str) -> Path:
    return SYNTHETIC_DIR / scene / f"n{n}" / strategy


def _load_synth_pool(scene: str, n: int, strategy: str) -> list[dict[str, Any]]:
    """Ordered list of synthetic sidecar records for (scene, n, strategy)."""
    d = synthetic_pool_dir(scene, n, strategy)
    manifest = d / "manifest.json"
    if not manifest.exists():
        return []
    with open(manifest) as f:
        entries = json.load(f)
    for e in entries:
        e["_abs_png"] = str((d / e["file"]).resolve())
    return entries


def gather_synthetic(scene: str, n: int, strategies: list[str], s: int) -> list[dict]:
    """Pick S synthetic frames, round-robin across the requested strategies."""
    if s <= 0 or not strategies:
        return []
    pools = {st: _load_synth_pool(scene, n, st) for st in strategies}
    missing = [st for st, p in pools.items() if len(p) < _needed_per(st, strategies, s)]
    if missing:
        have = {st: len(p) for st, p in pools.items()}
        raise FileNotFoundError(
            f"not enough synthetic images for scene={scene} n={n} "
            f"strategies={strategies} (need total {s}, have {have}). "
            f"Run scripts/generate_synthetic.py first."
        )
    picked: list[dict] = []
    cursors = {st: 0 for st in strategies}
    i = 0
    while len(picked) < s:
        st = strategies[i % len(strategies)]
        picked.append(pools[st][cursors[st]])
        cursors[st] += 1
        i += 1
    return picked


def _needed_per(strategy: str, strategies: list[str], s: int) -> int:
    """How many images this strategy contributes to a round-robin of size s."""
    idx = strategies.index(strategy)
    return sum(1 for k in range(s) if k % len(strategies) == idx)


def assemble_experiment_data(cfg: Config, exp: Experiment, out_dir: Path,
                             downscale: int | None = None) -> dict:
    """Build <out_dir>/{images, images_<ds>, transforms.json}.

    Nerfstudio, given an explicit ``--downscale-factor ds``, reads pre-downscaled
    images from ``images_<ds>/`` and does NOT generate them. So we materialise
    that folder here: real frames reuse the scene's existing ``images_<ds>``
    downsamples (symlink); synthetic frames get a resized copy. Intrinsics in
    transforms.json stay at full resolution — nerfstudio divides them by ``ds``.
    """
    from PIL import Image

    ds = int(downscale if downscale is not None else cfg.training["downscale_factor"])
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    ds_dir = out_dir / f"images_{ds}" if ds > 1 else images_dir
    ds_dir.mkdir(parents=True, exist_ok=True)

    real_split = "full" if exp.kind == "full" else f"n{exp.n}"
    meta, real_frames = load_split_frames(exp.scene, real_split)

    out: dict[str, Any] = {k: meta[k] for k in _META_KEYS if k in meta}
    frames: list[dict] = []

    # --- real frames: full-res symlink + downscaled symlink/copy ---------------
    for fr in real_frames:
        src = Path(fr["file_path"])          # absolute (written by splits.py)
        _symlink(src, images_dir / src.name)
        if ds > 1:
            pre = src.parent.parent / f"images_{ds}" / src.name   # scene's own downsample
            if pre.exists():
                _symlink(pre, ds_dir / src.name)
            else:                                                  # fall back: resize
                Image.open(src).reduce(ds).save(ds_dir / src.name)
        f2 = {k: v for k, v in fr.items() if k != "file_path"}
        f2["file_path"] = f"images/{src.name}"
        frames.append(f2)

    # --- synthetic frames: resized to the REAL resolution (nerfstudio needs a
    #     single global image size); FoV is carried by per-frame focal length ---
    Wg, Hg = int(meta["w"]), int(meta["h"])
    synth = gather_synthetic(exp.scene, exp.n, exp.strategies, exp.s) if exp.s else []
    for i, e in enumerate(synth):
        src = Path(e["_abs_png"])
        name = f"syn_{e['strategy']}_{i:04d}{src.suffix}"
        gen = Image.open(src).convert("RGB")
        gen.resize((Wg, Hg)).save(images_dir / name)
        if ds > 1:
            gen.resize((max(1, Wg // ds), max(1, Hg // ds))).save(ds_dir / name)
        f2: dict[str, Any] = {"file_path": f"images/{name}",
                              "transform_matrix": e["transform_matrix"]}
        for k in _PERFRAME_INTRINSICS:
            if k in e:
                f2[k] = e[k]
        f2["synthetic"] = True
        f2["strategy"] = e["strategy"]
        f2["source_file"] = e.get("source_file")
        frames.append(f2)

    # --- point-cloud init: copy the scene's (object-cropped) ply into the run.
    #     Without this splatfacto random-inits and few-shot fogs on the black bg. ---
    import json as _json
    import shutil
    src_meta = _json.load(open(cfg.scene_transforms(exp.scene)))
    ply_name = src_meta.get("ply_file_path")
    used_ply = None
    if ply_name:
        src_ply = cfg.scene_source(exp.scene) / ply_name
        if src_ply.exists():
            shutil.copy(src_ply, out_dir / Path(ply_name).name)
            out["ply_file_path"] = Path(ply_name).name
            used_ply = Path(ply_name).name

    out["frames"] = frames
    with open(out_dir / "transforms.json", "w") as f:
        json.dump(out, f, indent=2)

    return {"data_dir": str(out_dir), "n_real": len(real_frames),
            "n_synth": len(synth), "total": len(frames), "downscale_factor": ds,
            "ply_init": used_ply}


def _symlink(src: Path, link: Path) -> None:
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(src)

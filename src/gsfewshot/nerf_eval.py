"""Render the fixed held-out real test views from a trained splatfacto model
and compute PSNR/SSIM/LPIPS.

We render the *same* real held-out views for every experiment (poses come from
the scene's test split, not from Nerfstudio's internal eval split), so scores
are directly comparable across the whole grid. To place those poses in the
trained model's coordinate frame we re-apply the dataparser's stored
transform + scale. A ``--sanity`` mode renders a training frame instead, which
should score very high — a quick check that the transform is correct.

Imports nerfstudio/torch lazily.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .metrics import all_metrics
from .splits import split_dir


def _load_gt(path: str, w: int, h: int) -> np.ndarray:
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def _fit_similarity(raw_list: list[np.ndarray], act_list: list[np.ndarray]):
    """Global similarity (R_g, scale, b) mapping raw camera-to-world poses to the
    trained model's coordinate frame, fit from training correspondences::

        act_R = R_g @ raw_R ;  act_t = scale * R_g @ raw_t + b

    This reproduces nerfstudio's orientation + centering + scaling exactly
    (including any applied_transform) without depending on their conventions.
    """
    raw_R = [p[:3, :3] for p in raw_list]; raw_t = [p[:3, 3] for p in raw_list]
    act_R = [a[:3, :3] for a in act_list]; act_t = [a[:3, 3] for a in act_list]

    M = sum(aR @ rR.T for aR, rR in zip(act_R, raw_R))
    U, _, Vt = np.linalg.svd(M)
    Rg = U @ Vt
    if np.linalg.det(Rg) < 0:
        U[:, -1] *= -1
        Rg = U @ Vt

    scales = []
    for i in range(len(raw_t)):
        for j in range(i + 1, len(raw_t)):
            dr = np.linalg.norm(raw_t[i] - raw_t[j])
            if dr > 1e-8:
                scales.append(np.linalg.norm(act_t[i] - act_t[j]) / dr)
    scale = float(np.median(scales)) if scales else 1.0
    b = np.mean([act_t[i] - scale * (Rg @ raw_t[i]) for i in range(len(raw_t))], axis=0)

    resid = float(np.mean([np.linalg.norm(act_t[i] - (scale * Rg @ raw_t[i] + b))
                           for i in range(len(raw_t))]))
    return Rg, scale, b, resid


def evaluate_model(cfg: Config, scene: str, config_yml: Path, out_dir: Path,
                   downscale: int | None = None, max_views: int | None = None,
                   sanity: bool = False) -> dict[str, Any]:
    import torch
    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.cameras.cameras import Cameras, CameraType

    downscale = downscale or int(cfg.training["downscale_factor"])
    want = tuple(cfg.eval["metrics"])
    # object-centric scenes carry per-frame masks -> metrics measure the object
    scene_cfg = cfg.scene(scene)
    masks_dir = scene_cfg.get("masks_dir")
    mask_root = (cfg.scene_source(scene) / masks_dir) if masks_dir else None
    out_dir.mkdir(parents=True, exist_ok=True)
    renders_dir = out_dir / "renders"
    if cfg.eval.get("save_renders", True):
        renders_dir.mkdir(exist_ok=True)

    config, pipeline, _ckpt, _step = eval_setup(Path(config_yml), test_mode="inference")
    model = pipeline.model
    device = model.device

    # --- fit raw->model transform from the training cameras -------------------
    data_dir = Path(pipeline.datamanager.dataparser.config.data)
    with open(data_dir / "transforms.json") as f:
        train_meta = json.load(f)
    # fit only on REAL frames (synthetic poses are approximate by construction)
    raw_by_name = {Path(fr["file_path"]).name: np.array(fr["transform_matrix"], float)
                   for fr in train_meta["frames"] if not fr.get("synthetic")}
    train_ds = pipeline.datamanager.train_dataset
    tc = train_ds.cameras
    names = [Path(f).name for f in train_ds.image_filenames]
    raw_list, act_list = [], []
    for k, nm in enumerate(names):
        if nm in raw_by_name:
            raw_list.append(raw_by_name[nm])
            act_list.append(tc.camera_to_worlds[k].cpu().numpy())
    Rg, sim_scale, b, resid = _fit_similarity(raw_list, act_list)

    # intrinsics: reuse the trained (undistorted) camera — single shared camera,
    # already at the training downscale, so held-out views match exactly.
    fx = float(tc.fx[0]); fy = float(tc.fy[0]); cx = float(tc.cx[0]); cy = float(tc.cy[0])
    W = int(tc.width[0]); H = int(tc.height[0])

    def to_model_c2w(c2w_raw: np.ndarray) -> np.ndarray:
        R = Rg @ c2w_raw[:3, :3]
        t = sim_scale * (Rg @ c2w_raw[:3, 3]) + b
        out = np.zeros((3, 4), dtype=np.float32)
        out[:3, :3] = R; out[:3, 3] = t
        return out

    # choose which frames to render
    if sanity:
        frames = train_meta["frames"][:3]
        gt_base = data_dir
    else:
        with open(split_dir(scene, "test") / "transforms.json") as f:
            meta = json.load(f)
        frames = meta["frames"]
        gt_base = None
    if max_views:
        frames = frames[:max_views]

    def gt_path(fr):
        p = Path(fr["file_path"])
        return str(p if p.is_absolute() else (gt_base / p) if gt_base else p)

    def load_mask(fr, w, h):
        if mask_root is None or sanity:
            return None
        mp = mask_root / Path(fr["file_path"]).name
        if not mp.exists():
            return None
        from PIL import Image
        return np.asarray(Image.open(mp).convert("L").resize((w, h)),
                          dtype=np.float32) / 255.0

    per_view = []
    for i, fr in enumerate(frames):
        c2w_raw = np.array(fr["transform_matrix"], dtype=np.float64)
        new = torch.tensor(to_model_c2w(c2w_raw))                   # (3,4)

        cam = Cameras(
            camera_to_worlds=new[None],
            fx=torch.tensor([[fx]]), fy=torch.tensor([[fy]]),
            cx=torch.tensor([[cx]]), cy=torch.tensor([[cy]]),
            width=torch.tensor([[W]]), height=torch.tensor([[H]]),
            camera_type=CameraType.PERSPECTIVE,
        ).to(device)

        with torch.no_grad():
            outputs = model.get_outputs_for_camera(cam)
        rgb = outputs["rgb"].detach().cpu().numpy()                 # (H,W,3) in [0,1]

        gt = _load_gt(gt_path(fr), rgb.shape[1], rgb.shape[0])
        mask = load_mask(fr, rgb.shape[1], rgb.shape[0])
        m = all_metrics(rgb, gt, want, mask=mask)
        m["view"] = Path(fr["file_path"]).name
        per_view.append(m)

        if cfg.eval.get("save_renders", True):
            from PIL import Image
            stem = Path(fr["file_path"]).stem
            Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(
                renders_dir / f"{stem}_render.png")
            Image.fromarray((np.clip(gt, 0, 1) * 255).astype(np.uint8)).save(
                renders_dir / f"{stem}_gt.png")

    def _mean(key):
        vals = [v[key] for v in per_view if v.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    summary = {
        "scene": scene,
        "num_test_views": len(per_view),
        "psnr": _mean("psnr"),
        "ssim": _mean("ssim"),
        "lpips": _mean("lpips"),
        "pose_fit_residual": round(resid, 6),   # ~0 means the raw->model fit is exact
        "masked_metrics": bool(mask_root) and not sanity,
        "per_view": per_view,
        "sanity": sanity,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary

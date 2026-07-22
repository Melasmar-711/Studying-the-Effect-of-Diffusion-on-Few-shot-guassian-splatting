#!/usr/bin/env python
"""Definitive check: is each SVD training frame's IMAGE consistent with the POSE
+ intrinsics it's assigned? Render the good FULL model at the SVD frame's pose and
per-frame intrinsics; if it matches the SVD image, the pose is right (so a flat
result is genuine, not a pose bug). If they disagree, the poses are wrong.
"""
import json
from pathlib import Path
import numpy as np
import _bootstrap  # noqa: F401
from gsfewshot.nerf_eval import _fit_similarity

FULL_CFG = Path("experiments/scene02__n20_r100_inpaint/train/scene02__n20_r100_inpaint/splatfacto/run/config.yml")
POOL = Path("data/synthetic/scene02/n20/svd")
OUT = Path("results/previews/svd_multiview_clincher.png")


def main():
    import torch
    from PIL import Image, ImageDraw
    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.cameras.cameras import Cameras, CameraType

    _, pipeline, _, _ = eval_setup(FULL_CFG, test_mode="inference")
    model = pipeline.model; device = model.device

    # similarity: raw transforms.json c2w (real train frames) -> model cams
    data_dir = Path(pipeline.datamanager.dataparser.config.data)
    tm = json.load(open(data_dir / "transforms.json"))
    raw_by_name = {Path(f["file_path"]).name: np.array(f["transform_matrix"], float)
                   for f in tm["frames"] if not f.get("synthetic")}
    tc = pipeline.datamanager.train_dataset.cameras
    names = [Path(f).name for f in pipeline.datamanager.train_dataset.image_filenames]
    raw_list, act_list = [], []
    for k, nm in enumerate(names):
        if nm in raw_by_name:
            raw_list.append(raw_by_name[nm]); act_list.append(tc.camera_to_worlds[k].cpu().numpy())
    Rg, s, b, resid = _fit_similarity(raw_list, act_list)
    print(f"[check] similarity residual {resid:.4f}")

    def to_model(c2w):
        out = np.zeros((3, 4), np.float32)
        out[:3, :3] = Rg @ c2w[:3, :3]; out[:3, 3] = s * (Rg @ c2w[:3, 3]) + b
        return out

    man = json.load(open(POOL / "manifest.json"))
    picks = [man[0], man[3], man[6], man[9]]      # a few different seeds
    W, H = 480, 270                                # render size
    rows, sims = [], []
    for e in picks:
        img = Image.open(POOL / e["file"]).convert("RGB").resize((W, H))
        c2w = np.array(e["transform_matrix"], float)
        sx, sy = W / 1920.0, H / 1080.0            # per-frame intrinsics are at 1920x1080
        cam = Cameras(camera_to_worlds=torch.tensor(to_model(c2w))[None],
                      fx=e["fl_x"]*sx, fy=e["fl_y"]*sy, cx=e["cx"]*sx, cy=e["cy"]*sy,
                      width=W, height=H, camera_type=CameraType.PERSPECTIVE).to(device)
        with torch.no_grad():
            out = model.get_outputs_for_camera(cam)
        rgb = (out["rgb"].clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        ren = Image.fromarray(rgb)
        # silhouette IoU (non-black) as a quick orientation-match number
        a = (np.asarray(img).sum(2) > 30); c = (np.asarray(ren).sum(2) > 30)
        iou = (a & c).sum() / max(1, (a | c).sum())
        sims.append(iou); rows.append((e["file"], img, ren, iou))

    pad = 4
    panel = Image.new("RGB", (W*2+pad*3, (H+22)*len(rows)+pad), (30, 30, 30)); y = pad
    for name, img, ren, iou in rows:
        ImageDraw.Draw(panel).text((pad, y), f"{name}  LEFT=SVD image  RIGHT=full model @ that pose  silhouette IoU={iou:.2f}", fill=(240,240,240))
        panel.paste(img, (pad, y+18)); panel.paste(ren, (W+pad*2, y+18)); y += H+22
    OUT.parent.mkdir(parents=True, exist_ok=True); panel.save(OUT)
    print(f"[check] mean silhouette IoU {np.mean(sims):.3f}  -> {OUT}")
    print("[check] high IoU + matching orientation => poses correct (flat result is genuine)")


if __name__ == "__main__":
    main()

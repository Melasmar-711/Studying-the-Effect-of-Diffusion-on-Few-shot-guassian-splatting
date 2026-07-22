#!/usr/bin/env python
"""Is each zero123 synthetic frame CONSISTENT with its claimed pose+intrinsics?

Render the GOOD full model (PSNR 25) at each zero123 synthetic pose, using the
SCENE intrinsics — that is exactly what a scene camera at that pose should see.
Compare to the zero123 image. If they disagree (orientation/perspective), the
zero123 frame lies about its own projection -> that's why GS diverges.
"""
import json
from pathlib import Path
import numpy as np

import _bootstrap  # noqa: F401
from gsfewshot.nerf_eval import _fit_similarity

FULL_CFG = Path("experiments/scene02__full/train/scene02__full/splatfacto/run/config.yml")
POOL = Path("data/synthetic/scene02/n5/zero123")
OUT = Path("results/previews/zero123_consistency.png")


def main():
    import torch
    from PIL import Image, ImageDraw
    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.cameras.cameras import Cameras, CameraType

    _, pipeline, _, _ = eval_setup(FULL_CFG, test_mode="inference")
    model = pipeline.model; device = model.device
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
    print(f"[diag] similarity residual {resid:.4f} (fit on {len(raw_list)} real cams)")
    fx = float(tc.fx[0]); fy = float(tc.fy[0]); cx = float(tc.cx[0]); cy = float(tc.cy[0])
    W = int(tc.width[0]); H = int(tc.height[0])

    def to_model(c2w):
        out = np.zeros((3, 4), np.float32)
        out[:3, :3] = Rg @ c2w[:3, :3]; out[:3, 3] = s * (Rg @ c2w[:3, 3]) + b
        return out

    man = json.load(open(POOL / "manifest.json"))[:4]
    rows = []
    for e in man:
        z = Image.open(POOL / e["file"]).convert("RGB")
        c2w = np.array(e["transform_matrix"], float)
        cam = Cameras(camera_to_worlds=torch.tensor(to_model(c2w))[None],
                      fx=fx, fy=fy, cx=cx, cy=cy, width=W, height=H,
                      camera_type=CameraType.PERSPECTIVE).to(device)
        with torch.no_grad():
            out = model.get_outputs_for_camera(cam)
        rgb = (out["rgb"].clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        rows.append((e["file"], z, Image.fromarray(rgb)))

    tw, th = 384, 216
    panel = Image.new("RGB", (tw * 2 + 12, (th + 24) * len(rows)), (40, 40, 40)); y = 0
    for name, z, r in rows:
        ImageDraw.Draw(panel).text((4, y + 4),
            f"{name}: LEFT=zero123 image   RIGHT=full model rendered at that pose", fill=(255, 255, 255))
        panel.paste(z.resize((tw, th)), (0, y + 20)); panel.paste(r.resize((tw, th)), (tw + 12, y + 20)); y += th + 24
    OUT.parent.mkdir(parents=True, exist_ok=True); panel.save(OUT)
    print(f"[diag] panel -> {OUT}  (if LEFT != RIGHT, zero123 frame is inconsistent with pose+intrinsics)")


if __name__ == "__main__":
    main()

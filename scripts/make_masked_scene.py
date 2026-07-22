#!/usr/bin/env python
"""Create an object-on-fixed-background copy of a scene (object-centric study).

Segments the object in every frame (rembg), composites it over a fixed
background at FULL resolution, and writes a parallel scene dir:

    <out>/images/<name>     object on fixed bg (same size/name as original)
    <out>/masks/<name>      object alpha mask (for masked metrics)
    <out>/transforms.json   copied from the source (SAME camera poses)

Camera poses are unchanged — COLMAP is NOT re-run (it needs the background to
solve poses). We only swap pixel content. The `ply_file_path` is dropped so GS
uses random init (the COLMAP point cloud includes background points).

    python scripts/make_masked_scene.py --scene scene01 --out output_masked --bg black
"""
import argparse
import json
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401
from gsfewshot import load_config
from gsfewshot.config import PROJECT_ROOT

BG = {"black": (0, 0, 0), "white": (255, 255, 255), "grey": (128, 128, 128)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene01")
    ap.add_argument("--out", default="output_masked")
    ap.add_argument("--bg", default="black", choices=list(BG))
    ap.add_argument("--segmenter", default="sam", choices=["sam", "rembg"],
                    help="sam = isnet+SAM (robust, default); rembg = plain rembg")
    ap.add_argument("--model", default="u2net", help="rembg model (rembg segmenter only)")
    ap.add_argument("--limit", type=int, default=None, help="only N frames (debug)")
    args = ap.parse_args()

    from PIL import Image

    if args.segmenter == "sam":
        from gsfewshot.segment import Segmenter
        seg = Segmenter()

        def get_alpha(img_rgb):
            return seg.segment(img_rgb)
    else:
        from rembg import remove, new_session
        session = new_session(args.model)

        def get_alpha(img_rgb):
            return remove(img_rgb.convert("RGBA"), session=session,
                          post_process_mask=True).split()[-1]

    cfg = load_config()
    src_root = cfg.scene_source(args.scene)
    with open(cfg.scene_transforms(args.scene)) as f:
        meta = json.load(f)

    out_root = PROJECT_ROOT / args.out
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "masks").mkdir(parents=True, exist_ok=True)

    bg = BG[args.bg]
    bg_img = None
    frames = meta["frames"][:args.limit] if args.limit else meta["frames"]
    n = len(frames)
    print(f"Masking {n} frames of '{args.scene}' -> {out_root} "
          f"(bg={args.bg}, segmenter={args.segmenter})")

    for i, fr in enumerate(frames):
        rel = fr["file_path"]                      # e.g. images/frame_00001.png
        src = (src_root / rel)
        if not src.exists():
            src = src_root / Path(rel).name
        img = Image.open(src).convert("RGB")
        alpha = get_alpha(img)                      # PIL 'L' object mask
        if bg_img is None or bg_img.size != img.size:
            bg_img = Image.new("RGB", img.size, bg)
        comp = Image.composite(img, bg_img, alpha)
        name = Path(rel).name
        comp.save(out_root / "images" / name)
        alpha.save(out_root / "masks" / name)
        if (i + 1) % 25 == 0 or i + 1 == n:
            print(f"  {i+1}/{n}")

    # copy transforms.json with poses intact; KEEP the COLMAP point cloud — it
    # anchors the object's geometry so the reconstruction generalises to held-out
    # views (random init overfits on a textureless black background). Background
    # points render to black and are ignored by the masked metrics.
    out_meta = dict(meta)
    src_ply = src_root / meta.get("ply_file_path", "sparse_pc.ply")
    if src_ply.exists():
        shutil.copy(src_ply, out_root / src_ply.name)
        out_meta["ply_file_path"] = src_ply.name
    else:
        out_meta.pop("ply_file_path", None)
    out_meta["frames"] = [{**fr, "file_path": f"images/{Path(fr['file_path']).name}"}
                          for fr in meta["frames"]]
    with open(out_root / "transforms.json", "w") as f:
        json.dump(out_meta, f, indent=2)
    print(f"Done. Wrote {n} masked images + masks + transforms.json to {out_root}")
    print(f"Next: add scene under configs/project.yaml and run make_splits.")


if __name__ == "__main__":
    main()

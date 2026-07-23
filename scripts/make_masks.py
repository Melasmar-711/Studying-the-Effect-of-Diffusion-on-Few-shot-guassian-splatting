#!/usr/bin/env python
"""Generate object masks aligned to a folder of images (for object-masked eval).

Runs the isnet->SAM segmenter on every image and writes `<out>/<same-name>.png`
(white = object). Because the mask filenames match the image filenames exactly,
this sidesteps the frame-reindexing offset you hit if you segment *before* COLMAP.

Typical use when adding a NEW natural-background scene:

    ns-process-data images --data my_frames --output-dir scenes/myscene/nerf
    python scripts/make_masks.py --images scenes/myscene/nerf/images \
                                 --out    scenes/myscene/nerf/masks
    # then add a `myscene:` block to configs/project.yaml (masks_dir: masks)
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import _bootstrap  # noqa: F401  (sets TORCH_CUDA_ARCH_LIST etc.)
from gsfewshot.segment import Segmenter

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder of images to segment")
    ap.add_argument("--out", required=True, help="folder to write masks into")
    ap.add_argument("--min-frac", type=float, default=0.005,
                    help="warn if the object covers less than this fraction (bad segmentation)")
    args = ap.parse_args()

    imgs = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMG_EXT)
    if not imgs:
        raise SystemExit(f"no images found in {args.images}")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    seg = Segmenter()
    low = 0
    for i, p in enumerate(imgs, 1):
        mask = seg.segment(Image.open(p).convert("RGB"))     # PIL 'L', 0/255
        frac = float((np.asarray(mask) > 127).mean())
        mask.save(out / f"{p.stem}.png")
        if frac < args.min_frac:
            low += 1
            print(f"  [warn] {p.name}: object covers only {frac:.2%} — check this mask")
        if i % 25 == 0:
            print(f"  [{i}/{len(imgs)}]")
    print(f"[masks] wrote {len(imgs)} masks -> {out}  ({low} low-coverage)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Segment-FIRST scene prep: sample frames from a video, segment the object onto
black, and save masks — BEFORE any COLMAP. Feeding the object-on-black frames to
COLMAP makes its output inherently object-only (no post-hoc point cropping).

    python scripts/extract_and_segment.py --video video/2.mp4 \
        --out scenes/scene02 --num-frames 320

Writes:
    <out>/frames_seg/frame_XXXXX.png   object-on-black (COLMAP + training input)
    <out>/masks/frame_XXXXX.png        object mask (white on black)
    <out>/frames_raw/frame_XXXXX.jpg   original crop (reference / re-segmentation)
"""
import argparse
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

import _bootstrap  # noqa: F401
from gsfewshot.segment import Segmenter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-frames", type=int, default=320)
    ap.add_argument("--min-object-frac", type=float, default=0.01,
                    help="skip frames whose mask covers < this fraction (bad/empty)")
    args = ap.parse_args()

    out = Path(args.out)
    seg_dir = out / "frames_seg"; msk_dir = out / "masks"; raw_dir = out / "frames_raw"
    for d in (seg_dir, msk_dir, raw_dir):
        d.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // args.num_frames)
    idxs = list(range(0, total, step))[: args.num_frames]
    print(f"[extract] {args.video}: {total} frames -> sampling {len(idxs)} (every {step})")

    seg = Segmenter()
    kept = skipped = 0
    for out_i, fidx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, bgr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        mask = seg.segment(pil)                      # PIL 'L' 0/255
        m = np.asarray(mask, dtype=np.float32) / 255.0
        frac = float((m > 0.5).mean())
        if frac < args.min_object_frac:
            skipped += 1
            print(f"  [skip] frame {fidx}: object frac {frac:.3%} too small")
            continue
        obj = (rgb.astype(np.float32) * m[..., None]).astype(np.uint8)  # object on black
        name = f"frame_{out_i:05d}"
        Image.fromarray(obj).save(seg_dir / f"{name}.png")
        mask.save(msk_dir / f"{name}.png")
        Image.fromarray(rgb).save(raw_dir / f"{name}.jpg", quality=90)
        kept += 1
        if kept % 25 == 0:
            print(f"  [{kept}/{len(idxs)}] frame {fidx} object frac {frac:.1%}")
    cap.release()
    print(f"[done] kept {kept}, skipped {skipped} -> {seg_dir}")


if __name__ == "__main__":
    main()

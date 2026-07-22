#!/usr/bin/env python
"""Stage 2: segment the SVD-generated frames onto black (+ masks), matching the
object-centric training format. COLMAP uses the natural-bg frames (gen_raw);
TRAINING uses these object-on-black frames (gen_seg) + masks (gen_mask).
"""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image
import _bootstrap  # noqa: F401
from gsfewshot.segment import Segmenter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/svd/scene02/gen_raw")
    ap.add_argument("--seg", default="data/svd/scene02/gen_seg")
    ap.add_argument("--mask", default="data/svd/scene02/gen_mask")
    ap.add_argument("--min-frac", type=float, default=0.01)
    args = ap.parse_args()

    raw = Path(args.raw); seg = Path(args.seg); msk = Path(args.mask)
    clips = sorted([d for d in raw.iterdir() if d.is_dir()])
    segm = Segmenter()
    recs = []
    for cd in clips:
        (seg / cd.name).mkdir(parents=True, exist_ok=True)
        (msk / cd.name).mkdir(parents=True, exist_ok=True)
        for fp in sorted(cd.glob("gen_*.png")):
            rgb = np.asarray(Image.open(fp).convert("RGB"))
            mask = segm.segment(Image.fromarray(rgb))         # PIL 'L'
            m = np.asarray(mask, np.float32) / 255.0
            frac = float((m > 0.5).mean())
            obj = (rgb.astype(np.float32) * m[..., None]).astype(np.uint8)
            Image.fromarray(obj).save(seg / cd.name / fp.name)
            mask.save(msk / cd.name / fp.name)
            recs.append({"seed": cd.name, "frame": fp.name, "object_frac": frac})
        print(f"[seg] {cd.name}: {len(list(cd.glob('gen_*.png')))} frames")
    json.dump(recs, open(seg / "seg_manifest.json", "w"), indent=2)
    good = sum(r["object_frac"] >= args.min_frac for r in recs)
    print(f"[seg] DONE {len(recs)} frames, {good} with object_frac>={args.min_frac}")


if __name__ == "__main__":
    main()

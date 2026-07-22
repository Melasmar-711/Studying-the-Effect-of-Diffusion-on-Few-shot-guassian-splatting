#!/usr/bin/env python
"""Test SAM (facebook/sam-vit-base) segmentation on specific frames, prompted
with a box derived from a coarse rembg mask (unioned with a central box so it
still works when the coarse mask is a fragment). Tiles results for inspection.

    python scripts/seg_sam_test.py --frames 00090 00101 00233 00256 00289 00001
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
from gsfewshot import load_config
from gsfewshot.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "results" / "previews" / "seg_compare"


def prompt_points(coarse, W, H, n_pos=6):
    """Positive points sampled ON the object (from the coarse mask) + negative
    points at the corners (background). Anchors SAM to the object, away from the
    table."""
    ys, xs = np.where(coarse > 0.5)
    pos = []
    if len(xs):
        sel = np.linspace(0, len(xs) - 1, min(n_pos, len(xs))).astype(int)
        pos = [[int(xs[i]), int(ys[i])] for i in sel]
    else:
        pos = [[W // 2, H // 2]]
    neg = [[8, 8], [W - 8, 8], [8, H - 8], [W - 8, H - 8],
           [W // 2, 8], [W // 2, H - 8]]
    pts = pos + neg
    labels = [1] * len(pos) + [0] * len(neg)
    return pts, labels


def _iou(a, b):
    a, b = a > 0.5, b > 0.5
    inter = np.logical_and(a, b).sum()
    return inter / (np.logical_or(a, b).sum() + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene01")
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--coarse-model", default="isnet-general-use")
    args = ap.parse_args()

    import torch
    from PIL import Image, ImageDraw
    from rembg import remove, new_session
    from transformers import SamModel, SamProcessor

    cfg = load_config()
    src_dir = cfg.scene_source(args.scene) / cfg.scene(args.scene)["images_dir"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sam = SamModel.from_pretrained("facebook/sam-vit-base").to(dev)
    proc = SamProcessor.from_pretrained("facebook/sam-vit-base")
    coarse_sess = new_session(args.coarse_model)

    T = 300
    tiles = []
    for fr in args.frames:
        fp = src_dir / f"frame_{fr}.png"
        img = Image.open(fp).convert("RGB")
        W, H = img.size
        cut = remove(img.convert("RGBA"), session=coarse_sess)
        coarse = np.asarray(cut.split()[-1], dtype=np.float32) / 255.0
        pts, labels = prompt_points(coarse, W, H)

        inp = proc(img, input_points=[[pts]], input_labels=[[labels]],
                   return_tensors="pt").to(dev)
        with torch.no_grad():
            out = sam(**inp)
        masks = proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), inp["original_sizes"].cpu(),
            inp["reshaped_input_sizes"].cpu())[0][0]        # (3, H, W)
        # pick the mask that best overlaps the coarse object mask (avoids table)
        cand = [masks[k].numpy().astype(np.float32) for k in range(masks.shape[0])]
        best = max(cand, key=lambda m: _iou(m, coarse))

        comp = Image.composite(img, Image.new("RGB", img.size, (0, 0, 0)),
                               Image.fromarray((best * 255).astype(np.uint8)))
        comp.thumbnail((T, T))
        # label tile
        tile = Image.new("RGB", (T, comp.height + 18), (30, 30, 30))
        tile.paste(comp, ((T - comp.width) // 2, 18))
        ImageDraw.Draw(tile).text((3, 2), f"SAM {fr}", fill=(210, 210, 210))
        tiles.append(tile)

    W = sum(t.width for t in tiles) + 6 * (len(tiles) - 1)
    Hs = max(t.height for t in tiles)
    sheet = Image.new("RGB", (W, Hs), (30, 30, 30))
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0)); x += t.width + 6
    OUT.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT / "sheet_sam.png")
    print(f"-> {OUT/'sheet_sam.png'}")


if __name__ == "__main__":
    main()

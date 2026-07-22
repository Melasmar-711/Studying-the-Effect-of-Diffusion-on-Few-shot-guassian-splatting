#!/usr/bin/env python
"""Tile many segmented frames into one grid image to spot segmentation failures
(object parts cut off) across the whole trajectory, and compare rembg models.

    python scripts/mask_contact_sheet.py --model u2net --k 30
    python scripts/mask_contact_sheet.py --model isnet-general-use --k 30
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from gsfewshot import load_config
from gsfewshot.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "results" / "previews" / "seg_compare"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene01")   # segment the ORIGINAL frames
    ap.add_argument("--model", default="u2net")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--thumb", type=int, default=220)
    ap.add_argument("--alpha-matting", action="store_true")
    ap.add_argument("--no-post", action="store_true", help="disable post_process_mask")
    args = ap.parse_args()

    from PIL import Image, ImageDraw
    from rembg import remove, new_session

    cfg = load_config()
    src_dir = cfg.scene_source(args.scene) / cfg.scene(args.scene)["images_dir"]
    imgs = sorted(src_dir.glob("*.png")) + sorted(src_dir.glob("*.jpg"))
    idx = [int(round(i * (len(imgs) - 1) / (args.k - 1))) for i in range(args.k)]
    frames = [imgs[i] for i in idx]

    session = new_session(args.model)
    T = args.thumb
    rows = (len(frames) + args.cols - 1) // args.cols
    sheet = Image.new("RGB", (args.cols * T, rows * (T + 16)), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    for j, fp in enumerate(frames):
        img = Image.open(fp).convert("RGBA")
        cut = remove(img, session=session, alpha_matting=args.alpha_matting,
                     post_process_mask=not args.no_post)
        comp = Image.alpha_composite(Image.new("RGBA", cut.size, (0, 0, 0, 255)),
                                     cut).convert("RGB")
        # letterbox thumb
        comp.thumbnail((T, T))
        r, c = divmod(j, args.cols)
        x, y = c * T, r * (T + 16)
        draw.text((x + 2, y + 1), fp.stem.replace("frame_", ""), fill=(200, 200, 200))
        sheet.paste(comp, (x + (T - comp.width) // 2, y + 16))
    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.model + ("_matte" if args.alpha_matting else "") + ("_nopost" if args.no_post else "")
    out = OUT / f"sheet_{tag}.png"
    sheet.save(out)
    print(f"-> {out}")


if __name__ == "__main__":
    main()

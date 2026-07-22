#!/usr/bin/env python
"""Validate object segmentation across the trajectory BEFORE masking the dataset.

For a spread of real frames it renders [ original | white-bg | black-bg ] at
full resolution (downscaled only for the contact sheet) so we can judge whether
rembg segments the object consistently (esp. the thin gun barrel). Also lets us
compare rembg models (u2net default vs isnet-general-use).

    python scripts/preview_masking.py --model isnet-general-use
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from gsfewshot import load_config
from gsfewshot.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "results" / "previews" / "masking"


def composite(cut, rgb):
    from PIL import Image
    bg = Image.new("RGBA", cut.size, rgb + (255,))
    return Image.alpha_composite(bg, cut).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene01")
    ap.add_argument("--model", default="u2net",
                    help="rembg model: u2net | isnet-general-use")
    ap.add_argument("--n-frames", type=int, default=6)
    ap.add_argument("--alpha-matting", action="store_true")
    args = ap.parse_args()

    from PIL import Image, ImageDraw
    from rembg import remove, new_session

    cfg = load_config()
    src_dir = cfg.scene_source(args.scene) / cfg.scene(args.scene)["images_dir"]
    all_imgs = sorted(src_dir.glob("*.png")) + sorted(src_dir.glob("*.jpg"))
    if not all_imgs:
        raise SystemExit(f"no images in {src_dir}")
    # even spread across the whole trajectory (varied angles)
    idx = [int(round(i * (len(all_imgs) - 1) / (args.n_frames - 1)))
           for i in range(args.n_frames)]
    frames = [all_imgs[i] for i in idx]

    session = new_session(args.model)
    OUT.mkdir(parents=True, exist_ok=True)
    H = 300
    panels = []
    for fp in frames:
        img = Image.open(fp).convert("RGBA")
        cut = remove(img, session=session,
                     alpha_matting=args.alpha_matting,
                     post_process_mask=True)
        tiles = [("original", img.convert("RGB")),
                 ("white", composite(cut, (255, 255, 255))),
                 ("black", composite(cut, (0, 0, 0)))]
        row_w = sum(int(t[1].width * H / t[1].height) for t in tiles) + 12
        row = Image.new("RGB", (row_w, H + 22), (40, 40, 40))
        x = 0
        for label, im in tiles:
            w = int(im.width * H / im.height)
            row.paste(im.resize((w, H)), (x, 22))
            ImageDraw.Draw(row).text((x + 4, 4), f"{fp.name} — {label}", fill=(230, 230, 230))
            x += w + 6
        p = OUT / f"mask_{fp.stem}.png"
        row.save(p)
        panels.append(p)
        print(f"  {fp.name} -> {p.name}")

    # gallery
    import base64
    html = ["<style>body{background:#111;margin:1.5rem}img{max-width:100%;"
            "display:block;margin:8px 0;border:1px solid #333}</style>",
            f"<h2 style='color:#eee;font-family:sans-serif'>Segmentation check "
            f"(model={args.model}, alpha_matting={args.alpha_matting})</h2>"]
    for p in panels:
        b = base64.b64encode(p.read_bytes()).decode()
        html.append(f"<img src='data:image/png;base64,{b}'/>")
    (OUT / "index.html").write_text("\n".join(html))
    print(f"\nGallery: {OUT/'index.html'}")


if __name__ == "__main__":
    main()

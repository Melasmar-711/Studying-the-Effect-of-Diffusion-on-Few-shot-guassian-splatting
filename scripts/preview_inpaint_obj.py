#!/usr/bin/env python
"""Preview object-centric inpaint-blur on the masked (object-on-black) scene.

For a few object views it degrades a region ON THE OBJECT (blur, placed inside
the object's mask bbox) then restores it, and shows a triptych
    [ OBJECT | DEGRADED | RESTORED ]
so we can confirm "recover degraded object detail" looks sensible before training.

    python scripts/preview_inpaint_obj.py --scene scene01_obj --n 5 --sources 3
"""
import argparse
import base64
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
from gsfewshot import load_config
from gsfewshot.config import PROJECT_ROOT
from gsfewshot.splits import load_split_frames
from gsfewshot.synthetic import SyntheticGenerator

OUT = PROJECT_ROOT / "results" / "previews" / "inpaint_obj"


def label_tile(img, text, H=432):
    from PIL import Image, ImageDraw
    img = img.convert("RGB").resize((max(1, int(img.width * H / img.height)), H))
    bar = 26
    tile = Image.new("RGB", (img.width, H + bar), (24, 24, 24))
    tile.paste(img, (0, bar))
    ImageDraw.Draw(tile).text((6, 6), text, fill=(240, 240, 240))
    return tile


def hcat(tiles, pad=6):
    from PIL import Image
    H = max(t.height for t in tiles)
    W = sum(t.width for t in tiles) + pad * (len(tiles) - 1)
    out = Image.new("RGB", (W, H), (255, 255, 255))
    x = 0
    for t in tiles:
        out.paste(t, (x, 0)); x += t.width + pad
    return out


def mask_bbox(mask_arr, pad=0.12):
    ys, xs = np.where(mask_arr > 0.5)
    if len(xs) == 0:
        h, w = mask_arr.shape
        return (0, 0, w, h)
    h, w = mask_arr.shape
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    px, py = int((x1 - x0) * pad), int((y1 - y0) * pad)
    return (max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene01_obj")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--sources", type=int, default=3)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    from PIL import Image
    cfg = load_config(args.config)
    scene_cfg = cfg.scene(args.scene)
    mask_root = cfg.scene_source(args.scene) / scene_cfg.get("masks_dir", "masks")
    _, frames = load_split_frames(args.scene, f"n{args.n}")
    frames = frames[:args.sources]
    prompt = scene_cfg.get("prompt", "the object")
    gw, gh = int(cfg.diffusion["gen_width"]), int(cfg.diffusion["gen_height"])

    gen = SyntheticGenerator(cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    panels = []
    for i, fr in enumerate(frames):
        name = Path(fr["file_path"]).name
        src = Image.open(fr["file_path"]).convert("RGB").resize((gw, gh))
        mpath = mask_root / name
        marr = (np.asarray(Image.open(mpath).convert("L").resize((gw, gh)),
                           dtype=np.float32) / 255.0) if mpath.exists() else np.ones((gh, gw))
        box = mask_bbox(marr)
        degraded, _ = gen.build_inpaint_input(src.copy(), i, mode="blur", focus_box=box)
        out = gen.gen_inpaint(src, prompt, i, mode="blur", focus_box=box)
        panel = hcat([label_tile(src, "OBJECT"),
                      label_tile(degraded, "DEGRADED (blur on object)"),
                      label_tile(out, "RESTORED (inpaint)")])
        p = OUT / f"inpaint_obj__{name}"
        panel.save(p)
        panels.append((name, p))
        print(f"  {name} -> {p.name}")

    html = ["<style>body{background:#111;margin:1.5rem}img{max-width:100%;"
            "display:block;margin:8px 0;border:1px solid #333}"
            "div{color:#ddd;font-family:sans-serif}</style>",
            "<h2 style='color:#eee;font-family:sans-serif'>Object-centric inpaint-blur"
            " (recover degraded object detail)</h2>"]
    for name, p in panels:
        html.append(f"<div>{name}</div>"
                    f"<img src='data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}'/>")
    (OUT / "index.html").write_text("\n".join(html))
    print(f"\nGallery: {OUT/'index.html'}")


if __name__ == "__main__":
    main()

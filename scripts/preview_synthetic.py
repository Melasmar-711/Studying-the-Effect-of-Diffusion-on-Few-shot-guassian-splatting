#!/usr/bin/env python
"""Generate a few SAMPLE synthetic images for visual inspection BEFORE training.

For a handful of real source views it renders, per strategy/variant, a triptych
    [ SOURCE | INPUT-TO-MODEL | OUTPUT ]
and collects them into results/previews/index.html so you can eyeball whether the
augmentation is doing something useful.

    python scripts/preview_synthetic.py --scene scene01 --n 5 --sources 3

Variants shown:
  inpaint-blur   region blurred then restored
  inpaint-spots  black holes scattered then restored (strength 1.0)
  outpaint       blurred wide-canvas context then border repainted (feathered)
  zero123        novel viewpoints of the SEGMENTED object (Stable Zero123)
"""
import argparse
import base64
from pathlib import Path

import _bootstrap  # noqa: F401
from gsfewshot import load_config
from gsfewshot.splits import load_split_frames
from gsfewshot.synthetic import SyntheticGenerator

PREVIEW_DIR = Path(__file__).resolve().parents[1] / "results" / "previews"


def label_tile(img, text, H=432):
    from PIL import Image, ImageDraw
    img = img.convert("RGB")
    img = img.resize((max(1, int(img.width * H / img.height)), H))
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


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene01")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--sources", type=int, default=3, help="how many real views to sample")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    meta, frames = load_split_frames(args.scene, f"n{args.n}")
    frames = frames[:args.sources]
    prompt = cfg.scene(args.scene).get("prompt", "a photo of the scene")
    gw, gh = int(cfg.diffusion["gen_width"]), int(cfg.diffusion["gen_height"])

    gen = SyntheticGenerator(cfg)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    # variant -> list of (source_name, panel_path)
    results: dict[str, list] = {}

    def load(fr):
        from PIL import Image
        return Image.open(fr["file_path"]).convert("RGB").resize((gw, gh))

    def save_panel(variant, idx, name, tiles):
        panel = hcat(tiles)
        p = PREVIEW_DIR / f"{variant}__src{idx}.png"
        panel.save(p)
        results.setdefault(variant, []).append((name, p))

    # ---- Group A: inpaint pipeline (inpaint-blur, inpaint-spots, outpaint) ----
    print("[preview] group A: inpaint/outpaint ...")
    for i, fr in enumerate(frames):
        name = Path(fr["file_path"]).name
        src = load(fr)
        seed = i
        # inpaint blur
        manip, _ = gen.build_inpaint_input(src.copy(), seed, mode="blur")
        out = gen.gen_inpaint(src, prompt, seed, mode="blur")
        save_panel("inpaint-blur", i, name,
                   [label_tile(src, "SOURCE"), label_tile(manip, "INPUT (blurred region)"),
                    label_tile(out, "OUTPUT (restored)")])
        # inpaint spots
        manip, _ = gen.build_inpaint_input(src.copy(), seed, mode="spots")
        out = gen.gen_inpaint(src, prompt, seed, mode="spots")
        save_panel("inpaint-spots", i, name,
                   [label_tile(src, "SOURCE"), label_tile(manip, "INPUT (black spots)"),
                    label_tile(out, "OUTPUT (restored)")])
        # outpaint (reconstruct the blurred wide canvas for display)
        from PIL import ImageFilter
        blur = float(cfg.diffusion.get("manip_blur", 18))
        inner = 1.0 - float(cfg.diffusion["outpaint_expand"])
        iw, ih = int(gw * inner), int(gh * inner)
        canvas = src.resize((gw, gh)).filter(ImageFilter.GaussianBlur(blur))
        canvas.paste(src.resize((iw, ih)), ((gw - iw) // 2, (gh - ih) // 2))
        out, _ = gen.gen_outpaint(src, prompt, seed)
        save_panel("outpaint", i, name,
                   [label_tile(src, "SOURCE"), label_tile(canvas, "INPUT (blurred wide canvas)"),
                    label_tile(out, "OUTPUT (widened FoV)")])
    gen.free_pipes()

    gen.free_pipes()

    # ---- Group B: zero123 novel views of the object ----
    print("[preview] group B: zero123 novel views ...")
    from gsfewshot.zero123 import Zero123Generator, isolate_object
    from PIL import Image
    d = cfg.diffusion
    azims = d.get("zero123_azimuths", [-30, 30, -15, 15])
    elevs = d.get("zero123_elevations", [0, 0, 10, -10])
    steps = int(d.get("zero123_steps", 50))
    z = Zero123Generator(d.get("zero123_model", "kxic/zero123-xl"))
    for i, fr in enumerate(frames):
        name = Path(fr["file_path"]).name
        full = Image.open(fr["file_path"]).convert("RGB")
        cond, seg = isolate_object(full, 256)
        tiles = [label_tile(full, "SOURCE"), label_tile(cond, "SEGMENTED (input)")]
        for az, el in zip(azims, elevs):
            view = z.novel_view(cond, el, az, steps=steps, seed=i)
            tiles.append(label_tile(view, f"NEW VIEW az{az} el{el}"))
        save_panel("zero123", i, name, tiles)
    z.free()

    # ---- HTML gallery ----
    order = ["inpaint-blur", "inpaint-spots", "outpaint", "zero123"]
    style = ("<style>body{font-family:system-ui,sans-serif;margin:2rem;max-width:1200px;"
             "background:#111;color:#eee}img{max-width:100%;border:1px solid #333;"
             "border-radius:6px;margin:4px 0}h2{margin-top:2rem;border-bottom:1px solid #333}"
             "code{background:#222;padding:1px 5px;border-radius:3px}</style>")
    html = [style, "<h1>Synthetic augmentation preview</h1>",
            "<p>Each row is <code>SOURCE | INPUT-TO-MODEL | OUTPUT</code> for one real view.</p>"]
    for v in order:
        if v not in results:
            continue
        html.append(f"<h2>{v}</h2>")
        for name, p in results[v]:
            html.append(f"<div>{name}</div><img src='data:image/png;base64,{b64(p)}'/>")
    (PREVIEW_DIR / "index.html").write_text("\n".join(html))
    print(f"\nPreview gallery: {PREVIEW_DIR/'index.html'}")
    print(f"Panels: {PREVIEW_DIR}/*.png")


if __name__ == "__main__":
    main()

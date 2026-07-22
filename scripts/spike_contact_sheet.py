#!/usr/bin/env python
"""Tile a generated clip into one contact sheet so the morph is visible at a glance."""
import sys
from pathlib import Path
from PIL import Image, ImageDraw

clip = Path(sys.argv[1] if len(sys.argv) > 1 else "results/spikes/svd/frame_00319_mb127")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "results/spikes/contact_mb127.png")
frames = sorted(clip.glob("gen_*.png"))
cols, tw, th, pad = 5, 256, 146, 4
rows = (len(frames) + cols - 1) // cols
sheet = Image.new("RGB", (cols * (tw + pad) + pad, rows * (th + 18 + pad) + pad), (30, 30, 30))
for i, fp in enumerate(frames):
    r, c = divmod(i, cols)
    x, y = pad + c * (tw + pad), pad + r * (th + 18 + pad)
    sheet.paste(Image.open(fp).convert("RGB").resize((tw, th)), (x, y + 16))
    ImageDraw.Draw(sheet).text((x + 2, y + 2), f"frame {i:02d}", fill=(240, 240, 240))
out.parent.mkdir(parents=True, exist_ok=True); sheet.save(out)
print(f"-> {out}")

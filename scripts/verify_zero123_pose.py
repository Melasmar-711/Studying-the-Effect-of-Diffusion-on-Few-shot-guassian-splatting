#!/usr/bin/env python
"""Empirically pin down Zero123's (elevation, azimuth) convention against ground
truth, so generated novel views can be placed at correct camera poses.

Idea: pick two REAL frames A,B that orbit the same object center. Compute the
relative (Delta_elev, Delta_azim) from A to B in a self-consistent spherical
frame. Feed A to Zero123 with that offset under each sign convention; the output
that matches real B tells us the correct convention. Saves a panel.

    python scripts/verify_zero123_pose.py --scene scene02
"""
import argparse
from pathlib import Path
import numpy as np

import _bootstrap  # noqa: F401
from gsfewshot.splits import load_split_frames
from gsfewshot.zero123 import Zero123Generator

OUT = Path(__file__).resolve().parents[1] / "results" / "previews"


def cond_from_mask(img_path, size=256, pad=1.3):
    """Object-on-white 256px from the frame + its mask (clean, no rembg)."""
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    mp = Path(str(img_path).replace("/images/", "/masks/"))
    m = np.asarray(Image.open(mp).convert("L")) > 127
    a = np.asarray(img).astype(np.uint8)
    white = np.full_like(a, 255)
    comp = np.where(m[..., None], a, white).astype(np.uint8)
    ys, xs = np.where(m)
    if len(xs) == 0:
        return Image.fromarray(comp).resize((size, size))
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) * pad / 2
    x0, y0 = int(cx - half), int(cy - half)
    canvas = Image.new("RGB", (int(2 * half), int(2 * half)), (255, 255, 255))
    crop = Image.fromarray(comp).crop((x0, y0, x0 + int(2 * half), y0 + int(2 * half)))
    return crop.resize((size, size))


def cam_center(frames):
    """Least-squares intersection of camera view rays (-z) = object center."""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for f in frames:
        T = np.array(f["transform_matrix"]); o = T[:3, 3]; d = -T[:3, 2]
        d = d / np.linalg.norm(d)
        P = np.eye(3) - np.outer(d, d); A += P; b += P @ o
    return np.linalg.solve(A, b)


def spherical(p, C, up, e1, e2):
    d = (p - C); r = np.linalg.norm(d); d = d / r
    elev = np.degrees(np.arcsin(np.clip(d @ up, -1, 1)))
    dh = d - (d @ up) * up
    azim = np.degrees(np.arctan2(d @ e2, d @ e1))
    return elev, azim, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene02")
    ap.add_argument("--src", type=int, default=40)
    args = ap.parse_args()
    from PIL import Image, ImageDraw

    _, frames = load_split_frames(args.scene, "full")
    C = cam_center(frames)
    pos = np.array([np.array(f["transform_matrix"])[:3, 3] for f in frames])
    # orbit axis = smallest-spread PCA direction of camera positions (the turntable
    # axis). Mean camera-up is NOT this. Orient it so cameras sit at +elevation.
    v = pos - C
    _u, _s, _vt = np.linalg.svd(v - v.mean(0))
    up = _vt[2] / np.linalg.norm(_vt[2])
    if (v @ up).mean() < 0:
        up = -up
    # build an azimuth reference frame in the plane perpendicular to up
    seed = np.array([1.0, 0, 0]);
    if abs(seed @ up) > 0.9: seed = np.array([0, 1.0, 0])
    e1 = seed - (seed @ up) * up; e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)

    sph = np.array([spherical(p, C, up, e1, e2) for p in pos])   # (N,3) elev,azim,r
    from skimage.metrics import structural_similarity as ssim
    gen = Zero123Generator()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[verify] center={C.round(3)} up={up.round(3)}")

    def run_pair(kind):
        """kind='azim' -> azimuth-dominant pair; 'elev' -> elevation-dominant."""
        emid = np.median(sph[:, 0])
        A = int(np.argmin(np.abs(sph[:, 0] - emid)))
        ea, aa, _ = sph[A]
        daz = (sph[:, 1] - aa + 180) % 360 - 180
        dele = sph[:, 0] - ea
        if kind == "azim":
            cand = [i for i in range(len(frames)) if 15 <= abs(daz[i]) <= 30 and abs(dele[i]) <= 6 and i != A]
            key = lambda i: abs(abs(daz[i]) - 20)
        else:
            cand = [i for i in range(len(frames)) if 15 <= abs(dele[i]) <= 30 and abs(daz[i]) <= 8 and i != A]
            key = lambda i: abs(abs(dele[i]) - 20)
        if not cand:
            return None
        B = min(cand, key=key)
        d_el = float(sph[B, 0] - ea); d_az = float((sph[B, 1] - aa + 180) % 360 - 180)
        condA = cond_from_mask(frames[A]["file_path"]); realB = cond_from_mask(frames[B]["file_path"])
        gb = np.asarray(realB.convert("L"))
        tiles = [(f"A elev{ea:.0f}", condA), ("B real", realB)]
        scores = []
        for sel, saz in [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]:
            img = gen.novel_view(condA, sel * d_el, saz * d_az, 0.0, steps=50, seed=0)
            sc = ssim(np.asarray(img.resize(realB.size).convert("L")), gb)
            scores.append(((sel, saz), sc)); tiles.append((f"e{sel:+d}a{saz:+d}:{sc:.2f}", img))
        print(f"[verify] {kind}-pair A={Path(frames[A]['file_path']).name} B={Path(frames[B]['file_path']).name} "
              f"dElev={d_el:.1f} dAzim={d_az:.1f}")
        for (sel, saz), sc in scores:
            print(f"    elev*{sel:+d} azim*{saz:+d} -> SSIM {sc:.3f}")
        # save panel
        H = 256; W = sum(t[1].width for t in tiles) + 6 * (len(tiles) - 1)
        panel = Image.new("RGB", (W, H + 24), (255, 255, 255)); x = 0
        for label, im in tiles:
            panel.paste(im.resize((H, H)), (x, 24)); ImageDraw.Draw(panel).text((x + 4, 6), label, fill=(0, 0, 0)); x += H + 6
        panel.save(OUT / f"zero123_verify_{kind}.png")
        return max(scores, key=lambda x: x[1])[0], scores

    az_best, _ = run_pair("azim")
    el_res = run_pair("elev")
    gen.free()
    print(f"\n[verify] azimuth sign winner: {az_best}")
    if el_res:
        print(f"[verify] elevation sign winner: {el_res[0]}")
    print(f"[verify] panels -> {OUT}/zero123_verify_azim.png, zero123_verify_elev.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Honest few-shot init: build a point cloud from ONLY the N real few-shot views
(baseline) or the N real views + their SVD neighbours (svd), then map it into the
existing transforms.json frame so the current poses/eval keep working.

This tests the mechanism behind "use the ply from the new COLMAP": with few real
views COLMAP triangulates a sparse cloud; the generated views may densify it.

CPU COLMAP (use_gpu 0) so it doesn't fight the GPU training jobs.
"""
import argparse, json, subprocess, shutil, os
from pathlib import Path
import numpy as np

ROOT = Path("/home/asmar/GS_VR")
IMAGES = ROOT / "scenes/scene02/nerf/images"           # segmented (object-on-black) real
GENSEG = ROOT / "data/svd/scene02/gen_seg"
JSON = ROOT / "scenes/scene02/nerf/transforms.json"
CONDA = "/home/asmar/miniconda3/envs/nerfstudio"
WORKROOT = Path("/tmp/claude-1000/-home-asmar-GS-VR/f9750b5d-50b0-4def-974a-72ced49c4ec7/scratchpad/honest")


def qvec2rot(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def fit_similarity(raw, act):
    rR = [p[:3, :3] for p in raw]; rt = [p[:3, 3] for p in raw]
    aR = [a[:3, :3] for a in act]; at = [a[:3, 3] for a in act]
    M = sum(A @ R.T for A, R in zip(aR, rR))
    U, _, Vt = np.linalg.svd(M); Rg = U @ Vt
    if np.linalg.det(Rg) < 0:
        U[:, -1] *= -1; Rg = U @ Vt
    scales = []
    for i in range(len(rt)):
        for j in range(i+1, len(rt)):
            dr = np.linalg.norm(rt[i]-rt[j])
            if dr > 1e-8: scales.append(np.linalg.norm(at[i]-at[j])/dr)
    s = float(np.median(scales)) if scales else 1.0
    b = np.mean([at[i]-s*(Rg@rt[i]) for i in range(len(rt))], axis=0)
    resid = float(np.mean([np.linalg.norm(at[i]-(s*Rg@rt[i]+b)) for i in range(len(rt))]))
    return Rg, s, b, resid


RAW = ROOT / "scenes/scene02/frames_raw"
GENRAW = ROOT / "data/svd/scene02/gen_raw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--split-path", default=None,
                    help="seed-frame split json (overrides --n)")
    ap.add_argument("--gen-dir", default=None,
                    help="gen frames dir (overrides default gen_raw/gen_seg)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--with-gen", action="store_true")
    ap.add_argument("--natural", action="store_true",
                    help="use natural-bg frames (more features) instead of segmented")
    ap.add_argument("--crop-ref", default=None,
                    help="object .ply whose bbox crops out background points")
    ap.add_argument("--gpu", action="store_true", help="use GPU SIFT (faster; GPU must be free)")
    ap.add_argument("--out", required=True, help="output .ply path (in original frame)")
    args = ap.parse_args()
    G = "1" if args.gpu else "0"

    env = dict(os.environ)
    env["PATH"] = f"{CONDA}/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = f"{CONDA}/lib:" + env.get("LD_LIBRARY_PATH", "")

    tag = args.tag or f"n{args.n}_{'svd' if args.with_gen else 'real'}"
    WORK = WORKROOT / tag
    if WORK.exists(): shutil.rmtree(WORK)
    (WORK/"images/real").mkdir(parents=True)
    if args.with_gen: (WORK/"images/gen").mkdir(parents=True)
    (WORK/"sparse").mkdir(parents=True)

    split_path = ROOT/args.split_path if args.split_path else ROOT/f"data/splits/scene02/n{args.n}/transforms.json"
    split = json.load(open(split_path))
    stems = [Path(f["file_path"]).stem for f in split["frames"]]        # frame_000K (1-idx)
    real_src, gen_src = (RAW, GENRAW) if args.natural else (IMAGES, GENSEG)
    if args.gen_dir:
        gen_src = ROOT/args.gen_dir
    real_ext = ".jpg" if args.natural else ".png"
    for st in stems:
        k = int(st.split("_")[1])
        rn = f"frame_{k-1:05d}" if args.natural else st                 # raw is 0-indexed
        src = real_src/f"{rn}{real_ext}"
        if src.exists(): shutil.copy(src, WORK/"images/real"/f"{rn}{real_ext}")
    nr = len(list((WORK/"images/real").glob("*")))
    ng = 0
    if args.with_gen:
        for st in stems:
            cd = gen_src/st
            if cd.is_dir():
                for fp in sorted(cd.glob("gen_*.png")):
                    shutil.copy(fp, WORK/"images/gen"/f"{st}__{fp.name}"); ng += 1
    print(f"[{tag}] real {nr}, gen {ng}  ({'natural-bg' if args.natural else 'segmented'})")

    def col(*a): subprocess.run(["colmap", *a], env=env, check=True,
                                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    col("feature_extractor", "--database_path", str(WORK/"db.db"),
        "--image_path", str(WORK/"images"),
        "--ImageReader.single_camera_per_folder", "1",
        "--ImageReader.camera_model", "OPENCV", "--SiftExtraction.use_gpu", G)
    col("exhaustive_matcher", "--database_path", str(WORK/"db.db"),
        "--SiftMatching.use_gpu", G)
    col("mapper", "--database_path", str(WORK/"db.db"),
        "--image_path", str(WORK/"images"), "--output_path", str(WORK/"sparse"))

    models = [d for d in (WORK/"sparse").iterdir() if (d/"images.bin").exists()]
    if not models:
        print(f"[{tag}] NO MODEL RECONSTRUCTED"); return
    def nreg(d):
        col("model_converter", "--input_path", str(d), "--output_path", str(d), "--output_type", "TXT")
        return sum(1 for l in (d/"images.txt").read_text().splitlines()
                   if l and not l.startswith("#")) // 2
    M = max(models, key=nreg)

    # real poses in this solve -> fit similarity to original transforms.json
    lines = [l for l in (M/"images.txt").read_text().splitlines() if l and not l.startswith("#")]
    solve = {}
    for pl in lines[0::2]:
        f = pl.split(); q = list(map(float, f[1:5])); t = np.array(list(map(float, f[5:8]))); nm = f[9]
        Rwc = qvec2rot(q); c2w = np.eye(4); c2w[:3, :3] = Rwc.T; c2w[:3, 3] = -Rwc.T@t
        solve[nm] = c2w
    tj = json.load(open(JSON))
    orig = {Path(fr["file_path"]).stem: np.array(fr["transform_matrix"], float) for fr in tj["frames"]}
    F = np.diag([1, -1, -1.])                                   # OpenCV -> OpenGL (validated)
    raw, act = [], []
    for nm, c2w in solve.items():
        if nm.startswith("real/"):
            st = Path(nm).stem
            if args.natural:                                   # raw frame_000J -> orig frame_000(J+1)
                st = f"frame_{int(st.split('_')[1])+1:05d}"
            if st in orig:
                g = c2w.copy(); g[:3, :3] = c2w[:3, :3] @ F
                raw.append(g); act.append(orig[st])
    n_real_reg = len(raw)
    if n_real_reg < 3:
        print(f"[{tag}] only {n_real_reg} real registered — cannot fit frame"); return
    Rg, s, b, resid = fit_similarity(raw, act)
    print(f"[{tag}] real registered {n_real_reg}/{nr}, gen registered "
          f"{sum(1 for k in solve if k.startswith('gen/'))}/{ng}; sim residual {resid:.4f}")

    # points3D -> map to original frame, write ply
    pts = []
    for l in (M/"points3D.txt").read_text().splitlines():
        if l.startswith("#") or not l: continue
        p = l.split(); xyz = np.array(list(map(float, p[1:4]))); rgb = list(map(int, p[4:7]))
        xyz2 = s*(Rg@xyz)+b
        pts.append((xyz2, rgb))
    n_all = len(pts)

    # crop background: keep only points inside the object's bbox (from a ref object ply)
    if args.crop_ref:
        ref = ROOT/args.crop_ref if not Path(args.crop_ref).is_absolute() else Path(args.crop_ref)
        xyz_ref = []
        started = False
        for line in ref.read_text().splitlines():
            if started and line.strip():
                v = line.split()
                if len(v) >= 3:
                    try: xyz_ref.append([float(v[0]), float(v[1]), float(v[2])])
                    except ValueError: pass
            if line.startswith("end_header"): started = True
        xyz_ref = np.array(xyz_ref)
        lo = np.percentile(xyz_ref, 1, axis=0); hi = np.percentile(xyz_ref, 99, axis=0)
        mrg = 0.10 * (hi - lo); lo -= mrg; hi += mrg
        pts = [(p, c) for p, c in pts if np.all(p >= lo) and np.all(p <= hi)]
        print(f"[{tag}] crop to object bbox: {n_all} -> {len(pts)} points")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uint8 red\nproperty uint8 green\nproperty uint8 blue\nend_header\n")
        for xyz, rgb in pts:
            f.write(f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} {rgb[0]} {rgb[1]} {rgb[2]}\n")
    print(f"[{tag}] {len(pts)} points -> {out}")


if __name__ == "__main__":
    main()

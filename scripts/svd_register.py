#!/usr/bin/env python
"""Stage 3a: co-register SVD frames with the real orbit via COLMAP, then map their
poses into the EXISTING scene02 transforms.json frame (so results stay comparable).

- COLMAP co-solve: natural-bg real anchors (every 3rd + the 20 seeds) + all gen
  frames, single_camera_per_folder so the gen group gets its OWN intrinsics.
- Convert COLMAP (OpenCV) c2w -> OpenGL, auto-picking the camera-axis flip that
  minimises the similarity residual to transforms.json (robust sanity gate).
- Emit data/svd/scene02/gen_poses.json: per gen frame, transform_matrix in the
  transforms.json frame + per-frame intrinsics scaled to the real (1920x1080) size.
"""
import argparse, json, subprocess, shutil, os
from pathlib import Path
import numpy as np

ROOT = Path("/home/asmar/GS_VR")
RAW = ROOT / "scenes/scene02/frames_raw"
JSON = ROOT / "scenes/scene02/nerf/transforms.json"
WORK = Path("/tmp/claude-1000/-home-asmar-GS-VR/f9750b5d-50b0-4def-974a-72ced49c4ec7/scratchpad/svd_reg")
CONDA = "/home/asmar/miniconda3/envs/nerfstudio"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="data/splits/scene02/n20/transforms.json")
    ap.add_argument("--gen", default="data/svd/scene02/gen_raw")
    ap.add_argument("--out", default="data/svd/scene02/gen_poses.json")
    ap.add_argument("--every", type=int, default=3)
    args = ap.parse_args()

    env = dict(os.environ)
    env["PATH"] = f"{CONDA}/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = f"{CONDA}/lib:" + env.get("LD_LIBRARY_PATH", "")

    if WORK.exists(): shutil.rmtree(WORK)
    (WORK/"images/real").mkdir(parents=True); (WORK/"images/gen").mkdir(parents=True)
    (WORK/"sparse").mkdir()

    # seeds (raw idx = json idx - 1)
    n20 = json.load(open(ROOT/args.split))["frames"]
    seed_raw = {int(Path(f["file_path"]).stem.split("_")[1]) - 1 for f in n20}
    anchors = sorted(set(range(0, 320, args.every)) | seed_raw)
    nr = 0
    for i in anchors:
        f = RAW / f"frame_{i:05d}.jpg"
        if f.exists(): shutil.copy(f, WORK/"images/real"/f.name); nr += 1
    ng = 0
    for cd in sorted((ROOT/args.gen).iterdir()):
        if not cd.is_dir(): continue
        for fp in sorted(cd.glob("gen_*.png")):
            shutil.copy(fp, WORK/"images/gen"/f"{cd.name}__{fp.name}"); ng += 1
    print(f"[reg] real anchors {nr}, gen frames {ng}")

    def col(*a): subprocess.run(["colmap", *a], env=env, check=True,
                                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print("[reg] feature_extractor"); col("feature_extractor",
        "--database_path", str(WORK/"db.db"), "--image_path", str(WORK/"images"),
        "--ImageReader.single_camera_per_folder", "1",
        "--ImageReader.camera_model", "OPENCV", "--SiftExtraction.use_gpu", "1")
    print("[reg] exhaustive_matcher"); col("exhaustive_matcher",
        "--database_path", str(WORK/"db.db"), "--SiftMatching.use_gpu", "1")
    print("[reg] mapper"); col("mapper", "--database_path", str(WORK/"db.db"),
        "--image_path", str(WORK/"images"), "--output_path", str(WORK/"sparse"))

    # largest sub-model
    models = [d for d in (WORK/"sparse").iterdir() if (d/"images.bin").exists()]
    best = max(models, key=lambda d: len(list((d).glob("*")))) if models else None
    def nreg(d):
        col("model_converter", "--input_path", str(d), "--output_path", str(d), "--output_type", "TXT")
        return sum(1 for l in (d/"images.txt").read_text().splitlines()
                   if l and not l.startswith("#")) // 2
    best = max(models, key=nreg)
    M = best
    print(f"[reg] using model {M.name}")

    # cameras
    cams = {}
    for l in (M/"cameras.txt").read_text().splitlines():
        if l.startswith("#") or not l: continue
        p = l.split(); cams[int(p[0])] = {"model": p[1], "w": int(p[2]), "h": int(p[3]),
                                          "params": list(map(float, p[4:]))}
    # images
    lines = [l for l in (M/"images.txt").read_text().splitlines() if l and not l.startswith("#")]
    poses = lines[0::2]; pts = lines[1::2]
    real_gl, real_name, gen = {}, {}, {}
    for pl, ptl in zip(poses, pts):
        f = pl.split(); q = list(map(float, f[1:5])); t = np.array(list(map(float, f[5:8])))
        camid = int(f[8]); name = f[9]
        Rwc = qvec2rot(q); C = -Rwc.T @ t; Rc2w = Rwc.T
        nvalid = sum(1 for k, v in enumerate(ptl.split()) if (k % 3 == 2) and v != "-1")
        c2w = np.eye(4); c2w[:3, :3] = Rc2w; c2w[:3, 3] = C
        if name.startswith("gen/"):
            gen[name] = {"c2w_cv": c2w, "camid": camid, "npts": nvalid}
        else:
            real_gl[name] = c2w; real_name[name] = Path(name).name

    # transforms.json real poses, keyed by raw name (json frame_000K.png -> raw frame_000(K-1).jpg)
    tj = json.load(open(JSON))
    json_by_raw = {}
    for fr in tj["frames"]:
        k = int(Path(fr["file_path"]).stem.split("_")[1])
        json_by_raw[f"frame_{k-1:05d}.jpg"] = np.array(fr["transform_matrix"], float)

    # shared reals; try camera-axis flips, pick min residual
    flips = {"I": np.diag([1,1,1.]), "yz": np.diag([1,-1,-1.]),
             "xz": np.diag([-1,1,-1.]), "xy": np.diag([-1,-1,1.])}
    shared = [(nm, real_gl[nm]) for nm in real_gl if Path(nm).name in json_by_raw]
    best_flip, best_res, best_fit = None, 1e9, None
    for fname, F in flips.items():
        raw_list, act_list = [], []
        for nm, c2w in shared:
            g = c2w.copy(); g[:3, :3] = c2w[:3, :3] @ F
            raw_list.append(g); act_list.append(json_by_raw[Path(nm).name])
        Rg, s, b, res = fit_similarity(raw_list, act_list)
        print(f"[reg] flip {fname}: resid {res:.4f} (n={len(raw_list)})")
        if res < best_res:
            best_flip, best_res, best_fit = fname, res, (Rg, s, b, F)
    Rg, s, b, F = best_fit
    print(f"[reg] chosen flip={best_flip} residual={best_res:.4f}  scale={s:.4f}")

    # map gen poses + intrinsics
    def to_json(c2w_cv):
        g = c2w_cv.copy(); g[:3, :3] = c2w_cv[:3, :3] @ F
        out = np.eye(4)
        out[:3, :3] = Rg @ g[:3, :3]; out[:3, 3] = s * (Rg @ g[:3, 3]) + b
        return out
    Wg, Hg = int(tj["w"]), int(tj["h"])           # real 1920x1080
    recs = []
    for name, d in sorted(gen.items()):
        cam = cams[d["camid"]]; px = cam["params"]
        fx, fy, cx, cy = px[0], px[1], px[2], px[3]
        sx, sy = Wg / cam["w"], Hg / cam["h"]
        seed, frame = name[len("gen/"):].split("__")
        tm = to_json(d["c2w_cv"])
        recs.append({"seed": seed, "frame": frame,
                     "transform_matrix": tm.tolist(),
                     "fl_x": fx*sx, "fl_y": fy*sy, "cx": cx*sx, "cy": cy*sy,
                     "npts": d["npts"], "registered": True})
    reg_names = {(r["seed"], r["frame"]) for r in recs}
    total = sum(len(list((ROOT/args.gen/cd.name).glob("gen_*.png")))
                for cd in (ROOT/args.gen).iterdir() if cd.is_dir())
    out = {"residual": best_res, "flip": best_flip, "scale": s,
           "n_registered": len(recs), "n_total_gen": total, "frames": recs}
    json.dump(out, open(ROOT/args.out, "w"), indent=2)
    print(f"[reg] DONE registered {len(recs)}/{total} gen frames -> {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Crop the COLMAP point cloud to the OBJECT region for object-centric GS.

The full `sparse_pc.ply` spans the whole room; with few training views splatfacto
can't prune the far background points and they bloom into gray fog on held-out
views. We find the object centre (least-squares camera look-at convergence) and
keep only points within `radius` of it, then point transforms.json at the cropped
cloud. Full-view runs are unaffected (object points are all they need).

    python scripts/make_object_pointcloud.py --scene-dir output_masked --radius 3.0
"""
import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def lookat_center(frames):
    O = np.array([np.array(f["transform_matrix"])[:3, 3] for f in frames])
    D = np.array([-np.array(f["transform_matrix"])[:3, 2] for f in frames])  # OpenGL -z
    D = D / np.linalg.norm(D, axis=1, keepdims=True)
    A = np.zeros((3, 3)); b = np.zeros(3)
    for o, d in zip(O, D):
        P = np.eye(3) - np.outer(d, d)
        A += P; b += P @ o
    return np.linalg.solve(A, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", default="output_masked")
    ap.add_argument("--radius", type=float, default=3.0,
                    help="keep points within this distance of the object centre")
    ap.add_argument("--in-ply", default="sparse_pc.ply")
    ap.add_argument("--out-ply", default="sparse_pc_object.ply")
    args = ap.parse_args()

    sd = Path(args.scene_dir)
    meta = json.load(open(sd / "transforms.json"))
    center = lookat_center(meta["frames"])

    ply = PlyData.read(str(sd / args.in_ply))
    v = ply["vertex"]
    P = np.stack([v["x"], v["y"], v["z"]], 1)
    dist = np.linalg.norm(P - center, axis=1)
    keep = dist < args.radius

    kept = v.data[keep]
    el = PlyElement.describe(kept, "vertex")
    PlyData([el], text=False).write(str(sd / args.out_ply))
    meta["ply_file_path"] = args.out_ply
    json.dump(meta, open(sd / "transforms.json", "w"), indent=2)

    print(f"object centre: {np.round(center,3)}")
    print(f"kept {keep.sum()}/{len(P)} points (r<{args.radius}) -> {sd/args.out_ply}")
    print(f"transforms.json ply_file_path -> {args.out_ply}")


if __name__ == "__main__":
    main()

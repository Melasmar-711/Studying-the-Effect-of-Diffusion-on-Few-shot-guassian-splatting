#!/usr/bin/env python
"""Stage 4: assemble the SVD synthetic pools the experiment grid consumes.

Merges registered poses (gen_poses.json) with segmentation quality
(seg_manifest.json), then writes data/synthetic/scene02/n{10,20}/svd/ in the same
format as inpaint/zero123. Manifest order is round-robin across seeds (uniform
orbit coverage) with early/rigid frames first, so the r25..r200 slices stay clean.
"""
import argparse, json, shutil
from pathlib import Path

ROOT = Path("/home/asmar/GS_VR")
PROMPT = "a scale model tank on a plain background, sharp focus, high detail"


def frame_idx(fname):  # gen_007.png -> 7
    return int(Path(fname).stem.split("_")[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", default="data/svd/scene02/gen_poses.json")
    ap.add_argument("--seg", default="data/svd/scene02/gen_seg")
    ap.add_argument("--min-frac", type=float, default=0.02)
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 20])
    args = ap.parse_args()

    poses = json.load(open(ROOT/args.poses))["frames"]
    seg_manifest = ROOT/args.seg/"seg_manifest.json"
    frac = ({(r["seed"], r["frame"]): r["object_frac"] for r in json.load(open(seg_manifest))}
            if seg_manifest.exists() else None)   # natural-bg frames have no object_frac -> keep all

    # keep registered + segmented-well; group by seed
    by_seed = {}
    for r in poses:
        key = (r["seed"], r["frame"])
        if frac is not None and frac.get(key, 0.0) < args.min_frac:
            continue
        by_seed.setdefault(r["seed"], []).append(r)
    for s in by_seed:
        by_seed[s].sort(key=lambda r: frame_idx(r["frame"]))   # early frames first
    print(f"[pool] {sum(len(v) for v in by_seed.values())} usable frames "
          f"across {len(by_seed)} seeds")

    for n in args.sizes:
        split = json.load(open(ROOT/f"data/splits/scene02/n{n}/transforms.json"))
        keep = {Path(f["file_path"]).stem for f in split["frames"]}     # seed dir names
        seeds = [s for s in sorted(by_seed) if s in keep]
        # round-robin: rank-major across seeds
        cols = [by_seed[s] for s in seeds]
        order = []
        for rank in range(max(len(c) for c in cols)):
            for c in cols:
                if rank < len(c):
                    order.append(c[rank])
        outdir = ROOT/f"data/synthetic/scene02/n{n}/svd"
        if outdir.exists(): shutil.rmtree(outdir)
        outdir.mkdir(parents=True)
        man = []
        for i, r in enumerate(order):
            src = ROOT/args.seg/r["seed"]/r["frame"]
            name = f"syn_{i:04d}.png"
            shutil.copy(src, outdir/name)
            man.append({"file": name, "strategy": "svd",
                        "source_file": str(ROOT/"scenes/scene02/nerf/images"/f"{r['seed']}.png"),
                        "transform_matrix": r["transform_matrix"], "prompt": PROMPT,
                        "seed": 0, "w": 1920, "h": 1080,
                        "fl_x": r["fl_x"], "fl_y": r["fl_y"], "cx": r["cx"], "cy": r["cy"]})
        json.dump(man, open(outdir/"manifest.json", "w"), indent=2)
        print(f"[pool] n{n}: {len(man)} frames from {len(seeds)} seeds -> {outdir}")
    print("[pool] DONE")


if __name__ == "__main__":
    main()

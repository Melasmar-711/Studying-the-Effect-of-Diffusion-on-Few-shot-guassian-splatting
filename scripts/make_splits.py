#!/usr/bin/env python
"""Create the fixed held-out test set + few-shot subsets for a scene.

    python scripts/make_splits.py --scene scene01 [--overwrite]
"""
import argparse
import _bootstrap  # noqa: F401
from gsfewshot import load_config
from gsfewshot.splits import make_splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene01")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    m = make_splits(cfg, args.scene, overwrite=args.overwrite)
    print(f"scene            : {m['scene']}")
    print(f"total real frames: {m['n_frames']}")
    print(f"held-out test    : {m['test']['count']} views (every "
          f"{m['holdout_every']}th frame)")
    print(f"full train pool  : {m['full']['count']} views")
    for name, info in m["subsets"].items():
        print(f"few-shot {name:<4}    : {info['count']} views")
    print("\nWrote data/splits/%s/{test,full,n5,n10,n20}/transforms.json" % m["scene"])


if __name__ == "__main__":
    main()

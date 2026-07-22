#!/usr/bin/env python
"""Generate the synthetic image pool for a scene with diffusion models.

Pre-generates enough images per (N, strategy) to cover the largest ratio
(200% -> S = 2N), so every ratio experiment just draws the first S it needs.

    # everything needed for the whole grid on scene01:
    python scripts/generate_synthetic.py --scene scene01

    # a single (subset, strategy), custom count (good for a smoke test):
    python scripts/generate_synthetic.py --scene scene01 --n 5 \
        --strategies guided --count 1
"""
import argparse
import _bootstrap  # noqa: F401
from gsfewshot import load_config
from gsfewshot.synthetic import SyntheticGenerator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene01")
    ap.add_argument("--n", type=int, default=None,
                    help="few-shot size; default: all sizes in config")
    ap.add_argument("--strategies", nargs="+", default=None,
                    help="default: all strategies in config")
    ap.add_argument("--count", type=int, default=None,
                    help="images per (N, strategy); default: 2*N (covers 200%%)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    sizes = [args.n] if args.n else cfg.splits["few_shot_sizes"]
    strategies = args.strategies or cfg.grid["strategies"]

    gen = SyntheticGenerator(cfg)
    for n in sizes:
        for strat in strategies:
            count = args.count if args.count is not None else 2 * n
            print(f"[{args.scene}] n={n} {strat}: generating {count} images ...")
            out = gen.generate_for_subset(args.scene, n, strat, count,
                                          overwrite=args.overwrite)
            print(f"    -> {out}")
    print("Done.")


if __name__ == "__main__":
    main()

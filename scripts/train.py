#!/usr/bin/env python
"""Train a single experiment with splatfacto.

    python scripts/train.py --exp scene01__n5_r0
    python scripts/train.py --exp scene01__n5_r100_guided --iterations 2000
"""
import argparse
import json
import _bootstrap  # noqa: F401
from gsfewshot import load_config
from gsfewshot.grid import load_experiment
from gsfewshot.trainer import train_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="experiment id (see configs/experiments/)")
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    exp = load_experiment(args.exp, cfg)
    print(f"Training {exp.exp_id}  (N={exp.n} ratio={exp.ratio}% S={exp.s} "
          f"strategies={exp.strategies or '-'})")
    result = train_experiment(cfg, exp, max_iterations=args.iterations)
    print(json.dumps({k: v for k, v in result.items() if k != "cmd"}, indent=2))
    if result["status"] != "trained":
        raise SystemExit(f"training failed (rc={result['returncode']})")


if __name__ == "__main__":
    main()

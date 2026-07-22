#!/usr/bin/env python
"""Generate the experiment grid (one YAML per run + index.csv) from the plan.

    python scripts/gen_grid.py --scene scene01
"""
import argparse
import _bootstrap  # noqa: F401
from gsfewshot import load_config
from gsfewshot.grid import write_configs, enumerate_experiments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene01")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    exps = enumerate_experiments(cfg, args.scene)
    paths = write_configs(cfg, args.scene)
    print(f"Wrote {len(paths)} files to configs/experiments/")
    print(f"Total experiments for '{args.scene}': {len(exps)}")
    for e in exps:
        strat = "+".join(e.strategies) if e.strategies else "-"
        print(f"  {e.exp_id:<34} N={e.n:<4} ratio={e.ratio:<4} "
              f"S={e.s:<3} total={e.total:<4} [{strat}]  {e.purpose}")


if __name__ == "__main__":
    main()

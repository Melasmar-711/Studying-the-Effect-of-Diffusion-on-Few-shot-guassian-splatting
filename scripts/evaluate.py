#!/usr/bin/env python
"""Render the held-out real test views for a trained experiment and score them.

    python scripts/evaluate.py --exp scene01__n5_r0
    python scripts/evaluate.py --exp scene01__n5_r0 --sanity   # renders a train view
"""
import argparse
import json
from pathlib import Path
import _bootstrap  # noqa: F401
from gsfewshot import load_config
from gsfewshot.config import EXPERIMENTS_DIR
from gsfewshot.grid import load_experiment
from gsfewshot.trainer import train_config_path
from gsfewshot.nerf_eval import evaluate_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--config-yml", default=None,
                    help="override path to trained nerfstudio config.yml")
    ap.add_argument("--max-views", type=int, default=None)
    ap.add_argument("--sanity", action="store_true",
                    help="render training views (expect high PSNR) to check poses")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    exp = load_experiment(args.exp, cfg)
    out_root = EXPERIMENTS_DIR / exp.exp_id
    config_yml = Path(args.config_yml) if args.config_yml else train_config_path(exp, out_root)
    if not config_yml.exists():
        raise SystemExit(f"trained model not found: {config_yml}\nTrain first.")

    summary = evaluate_model(cfg, exp.scene, config_yml, out_root / "eval",
                             max_views=args.max_views, sanity=args.sanity)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_view"}, indent=2))


if __name__ == "__main__":
    main()

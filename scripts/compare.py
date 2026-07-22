#!/usr/bin/env python
"""Pull up a comparison across all recorded experiments.

Regenerates results/comparisons/{index.html, results.csv, results.md, *.png}
from results/registry.jsonl.

    python scripts/compare.py            # rebuild dashboard + print table
    python scripts/compare.py --open     # also print the file:// URL
"""
import argparse
import _bootstrap  # noqa: F401
from gsfewshot.registry import to_dataframe
from gsfewshot.viz import build_dashboard
from gsfewshot.config import COMPARISONS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    df = to_dataframe()
    if df.empty:
        print("No runs recorded yet. Train something with run_experiment.py first.")
    else:
        cols = [c for c in ["exp_id", "n_real", "ratio_pct", "strategies",
                            "s_synth", "total_images", "psnr", "ssim", "lpips",
                            "train_time_s", "peak_vram_mb"] if c in df.columns]
        print(df[cols].sort_values(
            [c for c in ["n_real", "ratio_pct"] if c in df]).to_string(index=False))

    html = build_dashboard()
    print(f"\nDashboard: {html}")
    if args.open:
        print(f"Open: file://{html.resolve()}")


if __name__ == "__main__":
    main()

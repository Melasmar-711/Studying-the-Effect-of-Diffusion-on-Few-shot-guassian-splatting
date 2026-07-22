#!/usr/bin/env python
"""End-to-end runner for one experiment: splits -> synthetic -> train -> eval
-> register. This is the single command that "records" an experiment.

    python scripts/run_experiment.py --exp scene01__n5_r100_guided
    python scripts/run_experiment.py --exp scene01__n5_r0 --iterations 2000

Idempotent-ish: reuses existing splits and synthetic pools; pass --regen-synth
to force regeneration.
"""
import argparse
import _bootstrap  # noqa: F401
from gsfewshot import load_config
from gsfewshot.config import EXPERIMENTS_DIR, SPLITS_DIR
from gsfewshot.grid import load_experiment
from gsfewshot.splits import make_splits
from gsfewshot.assemble import synthetic_pool_dir, _load_synth_pool, _needed_per
from gsfewshot.trainer import train_experiment, train_config_path
from gsfewshot.nerf_eval import evaluate_model
from gsfewshot.registry import record_run


def ensure_splits(cfg, scene):
    if not (SPLITS_DIR / scene / "manifest.json").exists():
        print(f"[splits] creating splits for {scene} ...")
        make_splits(cfg, scene)


def ensure_synthetic(cfg, exp, regen=False):
    if exp.s <= 0:
        return
    from gsfewshot.synthetic import SyntheticGenerator
    gen = None
    for strat in exp.strategies:
        need = _needed_per(strat, exp.strategies, exp.s)
        have = len(_load_synth_pool(exp.scene, exp.n, strat))
        if regen or have < need:
            target = max(2 * exp.n, need)
            print(f"[synth] n={exp.n} {strat}: have {have}, need {need} "
                  f"-> generating {target}")
            gen = gen or SyntheticGenerator(cfg)
            gen.generate_for_subset(exp.scene, exp.n, strat, target, overwrite=regen)
        else:
            print(f"[synth] n={exp.n} {strat}: reusing pool ({have} images)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--regen-synth", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--max-eval-views", type=int, default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    exp = load_experiment(args.exp, cfg)
    print(f"=== {exp.exp_id} :: {exp.purpose} ===")

    ensure_splits(cfg, exp.scene)
    ensure_synthetic(cfg, exp, regen=args.regen_synth)

    tr = train_experiment(cfg, exp, max_iterations=args.iterations)
    print(f"[train] status={tr['status']} time={tr['train_time_s']}s "
          f"peak_vram={tr['peak_vram_mb']}MB")

    record = {
        "exp_id": exp.exp_id, "scene": exp.scene, "kind": exp.kind,
        "n_real": exp.n, "ratio_pct": exp.ratio, "strategies": exp.strategies,
        "s_synth": exp.s, "total_images": tr.get("total", exp.total),
        "seed": cfg.diffusion.get("seed", 0),
        "train_time_s": tr["train_time_s"], "peak_vram_mb": tr["peak_vram_mb"],
        "downscale_factor": tr["downscale_factor"],
        "max_num_iterations": tr["max_num_iterations"],
        "config_path": tr["config_yml"], "output_dir": str(EXPERIMENTS_DIR / exp.exp_id),
        "status": tr["status"],
    }

    if tr["status"] == "trained" and not args.no_eval:
        cfg_yml = train_config_path(exp, EXPERIMENTS_DIR / exp.exp_id)
        ev = evaluate_model(cfg, exp.scene, cfg_yml,
                            EXPERIMENTS_DIR / exp.exp_id / "eval",
                            max_views=args.max_eval_views)
        record.update({"psnr": ev["psnr"], "ssim": ev["ssim"],
                       "lpips": ev["lpips"], "num_test_views": ev["num_test_views"],
                       "status": "done"})
        print(f"[eval] PSNR={ev['psnr']:.3f} SSIM={ev['ssim']:.4f} "
              f"LPIPS={ev['lpips']}  ({ev['num_test_views']} views)")

    record_run(record)
    print(f"[registry] appended -> results/registry.jsonl")


if __name__ == "__main__":
    main()

# Getting Started — how to run the study & understand the pipeline

The runbook. **Part 1** is the mental model. **Part 2** is the exact command sequence.
**Part 3** is reference material (exp-id decoder, init control, config knobs, troubleshooting).

> **Setup at a glance.** The active scene is **`scene02`** (a scale-model tank). Training
> happens on the **natural background** and metrics are **object-masked** — we measure the
> object, not the backdrop. (An earlier object-on-black "segment-first" pipeline was found to
> *handicap* few-shot reconstruction and was archived; see [`REPORT.md`](REPORT.md) §3.)

> Quickstart:
> ```bash
> cd ~/GS_VR && source .venv/bin/activate
> python scripts/gen_grid.py    --scene scene02        # enumerate the experiment grid
> python scripts/make_splits.py --scene scene02        # fixed test set + nested few-shot subsets
> # one experiment end-to-end (random init = honest few-shot):
> python scripts/run_experiment.py --exp scene02__n20_r100_inpaint --config configs/project.yaml
> python scripts/compare.py                            # rebuild the comparison dashboard
> ```

---

## Part 1 — The pipeline (mental model)

The study is a data-flow from one video to a table of comparable numbers. Each stage has a
script, reads well-defined inputs, and writes well-defined outputs.

```
  video/2.mp4
      │  extract_and_segment.py  (sample 320 frames + object masks)
      │  run_colmap_scene02.sh   (COLMAP → poses + point cloud)
      ▼
  scenes/scene02/nerf/          natural-bg images + masks + transforms.json + sparse_pc.ply
      │
      │  make_splits.py
      ▼
  data/splits/scene02/
      ├── test/    40 held-out real views  ── FIXED, scores every run identically
      ├── full/    the all-views pool (the ceiling run)
      └── n5 n10 n20/            few-shot subsets (nested: n5 ⊂ n10 ⊂ n20)
      │
      │  synthetic generation  (see Part 1: strategies)
      ▼
  data/synthetic/scene02/nN/{inpaint,svd}/*.png (+ per-frame poses & intrinsics)
      │
      │  run_experiment.py  ── orchestrates one experiment ──┐
      │   (a) assemble:  real subset + first S synthetic  →  experiments/<id>/data/
      │   (b) train:     ns-train splatfacto              →  experiments/<id>/train/…  (+ time, VRAM)
      │   (c) evaluate:  render the 40 held-out views     →  experiments/<id>/eval/  (masked PSNR/SSIM/LPIPS)
      │   (d) register:  append one row                   →  results/registry.jsonl
      ▼
  results/registry.jsonl        the append-only record of every run
      │  compare.py
      ▼
  results/comparisons/index.html   +   results/report/  (report + dashboard)
```

### The augmentation strategies

- **inpaint** — degrade a region **on the object** (masked to the object bbox), then let
  SD-inpainting restore it. A frame at the *exact real pose* → same-pose regularization.
  Generated on the fly by `run_experiment.py`, or via `generate_synthetic.py`.
- **svd** — genuine novel viewpoints via **Stable Video Diffusion**, with poses solved by
  running **COLMAP on the real + generated frames together** (no hand-computed geometry).
  Built by the chain: `gen_svd_neighbors.py` → `svd_segment.py` → `svd_register.py` →
  `svd_build_pool.py` (produces `data/synthetic/scene02/nN/svd/`).
- *(zero123, outpaint, guided were evaluated and dropped — see `REPORT.md`.)*

### Init regimes (this is what the study turns on)

How the Gaussians are initialized decides half the outcome — the full-scene point cloud
leaks geometry into "few-shot", so the honest baseline uses **random init**:

| Regime | How | Meaning |
|---|---|---|
| **random** | strip `ply_file_path` from the scene `transforms.json` (trap-restore) | honest few-shot floor — no geometry prior |
| **ply** | keep the ply; use `--config configs/project_natbg.yaml` (the cap prevents a many-view densification runaway) | leaky reference (`*_plyinit`) and the `full` ceiling |
| **own-ply** | `build_honest_ply.py` on an experiment's own views | realistic init (feasible only at n20 — fewer views don't register) |

**The key idea:** you never edit code to run an experiment. Pick an `exp_id` (all
pre-generated in `configs/experiments/`) and run it; everything is parameterised by
`configs/project.yaml`.

---

## Part 2 — Steps to run

### Step 0 — activate (every session)
```bash
cd ~/GS_VR && source .venv/bin/activate
```
*(Only if rebuilding the machine: `bash setup.sh` reinstalls everything.)*

### Step 1 — grid & splits
```bash
python scripts/gen_grid.py    --scene scene02     # configs/experiments/*.yaml + index.csv
python scripts/make_splits.py --scene scene02     # data/splits/scene02/…
```

### Step 2 — the honest baselines first (random init)
Baselines are the reference points. Random init strips the ply so they don't inherit
full-scene geometry:
```bash
# run the whole honest grid (baselines + inpaint + svd) with the trap-restore runner:
bash scripts/run_natbg_random_grid.sh
# …or a single cell manually:
python scripts/run_experiment.py --exp scene02__n20_r0 --config configs/project.yaml
```

### Step 3 — the reference / ceiling runs
```bash
bash scripts/run_natbg_full.sh          # `full` upper bound (ply init)
bash scripts/run_ownply_grid.sh         # own-ply arm (n20 only)
```

### Step 4 — pull up the comparison
```bash
python scripts/compare.py               # rebuilds results/comparisons/index.html
```
The finished report + dashboard live in **`results/report/`** — start from
[`REPORT.md`](REPORT.md) (renders on GitHub) or the interactive `fewshot_report.html`.

---

## Part 3 — Reference

### Decode an `exp_id`
```
scene02 __ n20 _ r100 _ inpaint            [ _plyinit | _ownply ]
   │        │     │       └── strategy: inpaint | svd (combos joined with +)
   │        │     └────────── synthetic:real ratio (percent)
   │        └──────────────── few-shot size N (or "full" for the ceiling)
   └───────────────────────── scene
```
`scene02__n5_r0` = honest few-shot baseline · `scene02__n20_r0_plyinit` = same but with the
leaky full ply · `scene02__full` = all-views ceiling.

### Init control (random vs ply)
Random init = no `ply_file_path` in `scenes/scene02/nerf/transforms.json`. The grid runners
do the swap with an `EXIT`-trap that always restores the ply, so an interrupted run never
leaves the scene in a half-modified state. Ply init uses the capped config
`configs/project_natbg.yaml` (`stop-split-at 3000`) — without the cap, ply + many views
OOMs at ~step 3800.

### Tuning knobs — all in `configs/project.yaml`
| Want to change… | Field |
|---|---|
| few-shot sizes / held-out cadence | `splits.few_shot_sizes`, `splits.holdout_every` |
| ratios / strategies to sweep | `grid.ratios`, `grid.strategies`, `grid.strategy_combos` |
| training length / resolution | `training.max_num_iterations`, `training.downscale_factor`, `training.extra_args` |
| diffusion model / steps | `diffusion.base_model`, `diffusion.inpaint_model`, `diffusion.steps` |
| inpaint behaviour | `diffusion.inpaint_manip`, `inpaint_region`, `inpaint_strength`, `manip_blur` |
| SVD generation | `diffusion.zero123_*` / SVD knobs in `scripts/gen_svd_neighbors.py` |
After editing, re-run `gen_grid.py` (and `make_splits.py` if you touched `splits`).

### Add another scene
```bash
ns-process-data video --data my.mp4 --output-dir scenes/myscene/nerf   # poses + ply + images
python scripts/make_masks.py --images scenes/myscene/nerf/images --out scenes/myscene/nerf/masks
# copy the `myscene:` template in configs/project.yaml, then:
python scripts/gen_grid.py --scene myscene && python scripts/make_splits.py --scene myscene
python scripts/run_experiment.py --exp myscene__n20_r0 --config configs/project.yaml
```
`object_centric: false` in the scene block scores the whole image (no masks). To design a
different sweep, edit `grid.*` / `splits.*` in `configs/project.yaml` and re-run `gen_grid.py`.
Full recipe + knob table in [README.md](README.md#adding-your-own-scene).

### Troubleshooting
- **`ns-train` / gsplat CUDA error** — use the prebuilt gsplat wheel; the system nvcc 11.5
  can't compile for the 4060. `setup.sh` handles it; don't `pip install gsplat` from PyPI.
- **CUDA out of memory** — ply-init many-view runs need the cap (`configs/project_natbg.yaml`);
  otherwise raise `training.downscale_factor` to 4.
- **HF download times out** — re-run; downloads resume.
- **A run failed mid-grid** — it's logged with `status: train_failed` (grid runners mark it
  `FAIL` and continue); just re-run that one `exp_id`.
- **Sanity-check pose handling** — `pose_fit_residual` in each `metrics.json` should be ≈ 0.

---

Architecture details in [README.md](README.md); the full narrative in [REPORT.md](REPORT.md);
every number in [results/registry.jsonl](results/registry.jsonl).

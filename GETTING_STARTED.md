# Getting Started — how to run the study & understand the pipeline

This is your runbook. **Part 1** explains how the pipeline works (the mental model).
**Part 2** is the exact sequence of commands to run from here. **Part 3** is reference
material (exp-id decoder, config knobs, troubleshooting).

> ⚠️ **The study is now OBJECT-CENTRIC.** We segment the tank onto a fixed black
> background for every image and measure masked metrics. Use scene **`scene01_obj`**
> (not `scene01`, which is the deprecated full-background version). See
> [Object-centric mode](README.md#object-centric-mode) in the README.

> Quickstart (object-centric):
> ```bash
> cd ~/GS_VR && source .venv/bin/activate
> # one-time: build the object-on-black dataset (segmentation + poses), then grid/splits
> python scripts/make_masked_scene.py --scene scene01 --out output_masked --bg black --segmenter sam
> python scripts/gen_grid.py    --scene scene01_obj
> python scripts/make_splits.py --scene scene01_obj
> # baselines, then compare
> python scripts/run_experiment.py --exp scene01_obj__n5_r0
> python scripts/run_experiment.py --exp scene01_obj__full
> python scripts/compare.py
> ```
> Preview the augmentations before training on them:
> `scripts/preview_inpaint_obj.py --scene scene01_obj` and `scripts/spike_zero123.py --src 0`.

---

## Part 1 — The pipeline (mental model)

The whole study is a data-flow from one video to a table of comparable numbers.
Each stage has a script, reads well-defined inputs, and writes well-defined outputs.

```
  video/1.mp4
      │  (already done for you with ns-process-data)
      ▼
  output/                         real images + COLMAP poses + transforms.json   ← the "scene"
      │
      │  make_splits.py
      ▼
  data/splits/scene01/
      ├── test/    41 held-out real views  ── FIXED, scores every run identically
      ├── full/    281 real views (the N≥100 upper-bound pool)
      └── n5 n10 n20/   few-shot subsets (nested: n5 ⊂ n10 ⊂ n20)
      │
      │  generate_synthetic.py   (Stable Diffusion 1.5)
      ▼
  data/synthetic/scene01/nN/{outpaint,inpaint,guided}/*.png (+ sidecar poses)
      │
      │  run_experiment.py  ── orchestrates one experiment ──┐
      │                                                      │
      │   (a) assemble:  real subset + first S synthetic  →  experiments/<id>/data/
      │   (b) train:     ns-train splatfacto              →  experiments/<id>/train/…/config.yml  (+ time, VRAM)
      │   (c) evaluate:  render the 41 held-out views     →  experiments/<id>/eval/  (PSNR/SSIM/LPIPS + renders)
      │   (d) register:  append one row                   →  results/registry.jsonl
      ▼
  results/registry.jsonl          the append-only record of every run
      │
      │  compare.py
      ▼
  results/comparisons/index.html  table + PSNR/SSIM/LPIPS-vs-ratio plots + GT|render panels
```

### Why each stage exists

- **Splits** — the experiment only means something if every run is scored on the
  *same* held-out real views. `make_splits.py` reserves those (every 8th frame) and
  never lets them into training or augmentation. The few-shot subsets are drawn from
  what's left, nested so `n5 ⊂ n10 ⊂ n20` (clean, interpretable).

- **Synthetic generation** — creates extra training views *only from the real subset*
  using three strategies:
  - **outpaint** widens the field of view (blurred real context in the border, then repaint);
  - **inpaint** degrades a region of the real image (blur/cut) then restores it — stays scene-consistent;
  - **guided** conditions ControlNet on the source's edges at an interpolated nearby pose.

  It pre-builds `2·N` images per (N, strategy) — enough to cover the 200% ratio — and
  each experiment just uses the first `S` it needs.

- **assemble** (inside `run_experiment`) — builds a self-contained nerfstudio data dir
  for the experiment: `images/` (full res) + `images_2/` (½ res, for 8 GB) + a
  `transforms.json` mixing the real subset and the S synthetic frames.

- **train** — `ns-train splatfacto` on that data dir, capturing wall-clock time and
  peak VRAM. Random Gaussian init (no COLMAP point cloud) so few-shot runs don't get a
  hidden advantage from the full-scene reconstruction.

- **evaluate** — loads the trained model, renders the 41 held-out poses, and compares
  to ground truth (PSNR/SSIM/LPIPS). It fits the exact camera alignment from the
  training cameras, so `pose_fit_residual ≈ 0` (a health check printed in `metrics.json`).

- **register + compare** — every run appends a row to `registry.jsonl`; `compare.py`
  turns that into the dashboard. This is the "pull up a comparison" surface.

**The key idea:** you never edit code to run an experiment. You pick an `exp_id`
(they're all pre-generated in `configs/experiments/`) and run it. Everything is
parameterised by `configs/project.yaml`.

---

## Part 2 — Steps to take from here

### Step 0 — activate the environment (every session)

```bash
cd ~/GS_VR
source .venv/bin/activate
```

*(Only if you ever rebuild the machine: `bash setup.sh` re-installs everything.)*

### Step 1 — grid & splits  ✅ already done

Already generated for `scene01` (40 experiments, splits with 41 test / 281 pool).
Re-run any time if you change `configs/project.yaml`:

```bash
python scripts/gen_grid.py    --scene scene01     # writes configs/experiments/*.yaml + index.csv
python scripts/make_splits.py --scene scene01     # writes data/splits/scene01/…
```

### Step 2 — build the synthetic pools (one-time, the slow part)

```bash
python scripts/generate_synthetic.py --scene scene01
```

Generates ~210 images (`2·N` per N per strategy). First run downloads the SD models
(a few GB — if the network hiccups, just re-run; it resumes). **Look at a few outputs**
in `data/synthetic/scene01/…` to sanity-check them before training on them.

> You can skip this — `run_experiment.py` will generate whatever a run needs on the
> fly — but doing it up front gets all the diffusion/download work out of the way once.

### Step 3 — run the baselines first

Baselines are your reference points; run them before the augmented grid.

```bash
python scripts/run_experiment.py --exp scene01__full     # N≥100 upper bound (longest run)
python scripts/run_experiment.py --exp scene01__n5_r0    # few-shot floor
python scripts/run_experiment.py --exp scene01__n10_r0
python scripts/run_experiment.py --exp scene01__n20_r0
```

Each prints `[train] … [eval] PSNR=… SSIM=… [registry] appended`. Runs use the full
`max_num_iterations` (30000) from the config. **Time one run first** to gauge how long
the whole grid will take on your 4060 (few-shot runs are quick; `full` is the longest).

### Step 4 — run the augmented grid

Run individual cells:

```bash
python scripts/run_experiment.py --exp scene01__n5_r100_inpaint
python scripts/run_experiment.py --exp scene01__n10_r50_guided
```

…or run **everything** in `index.csv` sequentially (long; safe to stop/resume — done
runs are simply appended again if you re-run them):

```bash
tail -n +2 configs/experiments/index.csv | cut -d, -f1 | while read exp; do
  echo "=== $exp ==="
  python scripts/run_experiment.py --exp "$exp" || echo "FAILED: $exp"
done
```

Tip: to smoke-test the loop fast, add `--iterations 2000` to each run first, then do
the real 30k pass.

### Step 5 — pull up the comparison

```bash
python scripts/compare.py            # rebuilds results/comparisons/index.html
python scripts/compare.py --open     # also prints the file:// link
```

Open `results/comparisons/index.html`. You get: the results table (also `results.csv`
/ `results.md`), PSNR/SSIM/LPIPS-vs-ratio plots faceted by N, and GT|render panels.
Re-run `compare.py` any time to fold in new results.

### Step 6 — write the report

Everything the ~5-page report needs is now generated:
- **Tables** → `results/comparisons/results.md` / `.csv`
- **Plots** → `results/comparisons/*_vs_ratio.png`
- **Visual examples** → `results/comparisons/panels/`
Drop your prose + these assets into `results/report/`.

---

## Part 3 — Reference

### Decode an `exp_id`

```
scene01 __ n5 _ r100 _ guided
  │        │     │       └── strategy: outpaint | inpaint | guided (combos joined with +)
  │        │     └────────── synthetic:real ratio (percent)
  │        └──────────────── few-shot size N (or "full" for the N≥100 baseline)
  └───────────────────────── scene
```
`scene01__n5_r0` = few-shot baseline (0% synthetic). `scene01__full` = upper bound.

### Tuning knobs — all in `configs/project.yaml`

| Want to change… | Field |
|---|---|
| few-shot sizes / held-out cadence | `splits.few_shot_sizes`, `splits.holdout_every` |
| which ratios / strategies to sweep | `grid.ratios`, `grid.strategies`, `grid.strategy_combos` |
| training length / resolution | `training.max_num_iterations`, `training.downscale_factor` |
| diffusion model / steps | `diffusion.base_model`, `diffusion.inpaint_model`, `diffusion.steps` |
| **inpaint/outpaint behaviour** | `diffusion.inpaint_manip`, `inpaint_region`, `inpaint_strength`, `manip_blur`, `outpaint_expand`, `negative_prompt` |
After editing, re-run `gen_grid.py` (and `make_splits.py` if you touched `splits`).

### Useful one-offs

```bash
# sanity-check pose handling for a trained run (renders TRAIN views → PSNR should be high)
python scripts/evaluate.py --exp scene01__n5_r0 --sanity

# re-evaluate an already-trained run
python scripts/evaluate.py --exp scene01__n5_r0

# regenerate a specific synthetic pool
python scripts/generate_synthetic.py --scene scene01 --n 5 --strategies inpaint --count 10 --overwrite
```

### Add another scene

1. `ns-process-data video --data your.mp4 --output-dir output_scene2`
2. Add it under `scenes:` in `configs/project.yaml` (copy the `scene01` block).
3. `python scripts/gen_grid.py --scene scene2 && python scripts/make_splits.py --scene scene2`
4. Run experiments with `scene2__…` ids. More scenes = stronger conclusions.

### Troubleshooting

- **`ns-train` / gsplat CUDA error** — you must use the prebuilt gsplat wheel (the
  system nvcc 11.5 can't compile for the 4060). `setup.sh` handles it; don't
  `pip install gsplat` from PyPI. See the env notes in [README.md](README.md).
- **HF download times out** — re-run the command; downloads resume. `prefetch_models.py`
  wraps this in a retry loop.
- **CUDA out of memory** — raise `training.downscale_factor` to 4 in the config.
- **Synthetic images look wrong** — tune `inpaint_strength` (lower = closer to the real
  image) and `inpaint_region`; view outputs before training on them.
- **A run failed mid-grid** — it's recorded with `status: train_failed`; just re-run
  that one `exp_id`.

---

Full architecture details are in [README.md](README.md); the experiment matrix is
[configs/experiments/index.csv](configs/experiments/index.csv).

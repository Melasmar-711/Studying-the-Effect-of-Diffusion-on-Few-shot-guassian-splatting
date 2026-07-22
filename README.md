# Few-Shot Gaussian Splatting with Diffusion-Based Data Augmentation

An organized workspace for the study described in the project spec + experimental
plan: train Gaussian Splatting from a **few** real views, augment the training set
with **diffusion-generated** synthetic views, and measure how the synthetic:real
ratio and the augmentation strategy affect held-out render quality.

- **GS backend:** Nerfstudio `splatfacto` (gsplat)
- **Study mode:** **object-centric** — the object is segmented onto a fixed black
  background for every image (real + synthetic); see [Object-centric mode](#object-centric-mode).
- **Augmentation:** SD 1.5 **inpaint** (recover degraded object detail) + **Zero123**
  novel object views. (Outpaint & ControlNet-guided were evaluated and dropped — see below.)
- **Metrics:** PSNR, SSIM, LPIPS **restricted to the object mask** + train time + peak VRAM.
- **Target box:** RTX 4060 (8 GB), driver 580, nvcc 11.5 — see [Environment notes](#environment-notes)

> **Two scenes live in the config:** `scene01` (original, full background — deprecated)
> and **`scene01_obj`** (object-on-black — the active study). Work on `scene01_obj`.

---

## TL;DR

```bash
bash setup.sh                                  # one-time install (venv + all deps)
source .venv/bin/activate

python scripts/gen_grid.py    --scene scene01  # enumerate the experiment matrix
python scripts/make_splits.py --scene scene01  # fixed test set + few-shot subsets

# run one experiment end-to-end (splits -> synth -> train -> eval -> registry)
python scripts/run_experiment.py --exp scene01__n5_r0
python scripts/run_experiment.py --exp scene01__n5_r100_guided

python scripts/compare.py                      # rebuild the comparison dashboard
```

`scripts/compare.py` writes `results/comparisons/index.html` — open it to see the
results table, metric-vs-ratio plots, and GT-vs-render panels for every run so far.

---

## The experiment grid (Phase 3 of the plan)

For each scene:

| Base real N | Synthetic ratio | S (synthetic) | Total | Purpose |
|---|---|---|---|---|
| full (≥100) | — | 0 | all | Upper-bound baseline |
| 5 / 10 / 20 | 0% | 0 | N | Few-shot baseline (control) |
| 5 / 10 / 20 | 25% | round(0.25·N) | N+S | Extremely low synthetic impact |
| 5 / 10 / 20 | 50% | round(0.50·N) | N+S | Minority synthetic |
| 5 / 10 / 20 | 100% | N | 2N | 1:1 real-to-synthetic |
| 5 / 10 / 20 | 200% | 2N | 3N | Synthetic outweighs real |

…each augmented cell run for every diffusion **strategy**
(`outpaint`, `inpaint`, `guided`) and any `strategy_combos` you add.
`python scripts/gen_grid.py` materializes this as one YAML per run in
`configs/experiments/` plus `index.csv`. Edit `configs/project.yaml` to change
sizes/ratios/strategies — everything downstream reads from it.

**S is computed with round-half-up** (`N=5, 50% → 3`) so the ambiguous rows in the
plan resolve deterministically.

---

## Layout

```
configs/
  project.yaml              # SINGLE SOURCE OF TRUTH (scenes, grid, diffusion, training)
  experiments/              # generated: one <exp_id>.yaml per run + index.csv
data/
  scenes/scene01/           # symlink -> output/ (the ns-process-data result)
  splits/scene01/           # generated: test/ full/ n5/ n10/ n20/ + manifest.json
  synthetic/scene01/        # generated: n{N}/{outpaint,inpaint,guided}/*.png (+ sidecars)
src/gsfewshot/              # importable package (all logic lives here)
scripts/                    # thin CLIs (gen_grid, make_splits, generate_synthetic,
                            #            train, evaluate, run_experiment, compare)
experiments/<exp_id>/       # per-run record: data/ train/ eval/ (renders + metrics.json)
results/
  registry.jsonl            # append-only record of every run  <-- the source of truth
  comparisons/              # generated dashboard: index.html, results.csv/.md, plots, panels
  report/                   # your ~5-page report assets
```

### `exp_id` naming

- `scene01__full` — full-data upper bound
- `scene01__n5_r0` — few-shot baseline (N=5, 0% synthetic)
- `scene01__n10_r100_guided` — N=10, 100% synthetic, guided strategy
- `scene01__n20_r50_outpaint+inpaint` — combo strategies joined by `+`

---

## How evaluation stays fair

`make_splits.py` reserves a **fixed held-out test set** of real views (every _k_-th
frame, default `k=8`) that is **identical for every experiment** and never used for
training or augmentation. Few-shot subsets are drawn (nested: `n5 ⊂ n10 ⊂ n20`) from
the remaining pool. `evaluate.py` renders those exact test poses from each trained
model and computes PSNR/SSIM/LPIPS against ground truth — so numbers are directly
comparable across the whole grid.

> Sanity check the pose handling anytime with
> `python scripts/evaluate.py --exp <id> --sanity` (renders *training* views; PSNR
> should be high).

---

## Object-centric mode

The object is segmented onto a **fixed black background** for every image so the study
measures *object* reconstruction, and so real and Zero123 (white-bg) views share one
visual domain. Build the masked scene (poses unchanged — COLMAP is **not** re-run):

```bash
python scripts/make_masked_scene.py --scene scene01 --out output_masked --bg black --segmenter sam
python scripts/gen_grid.py    --scene scene01_obj
python scripts/make_splits.py --scene scene01_obj
```

- **Segmentation** (`src/gsfewshot/segment.py`): plain rembg (u2net/isnet) cuts the
  object on hard top-down angles, so we use **isnet → SAM**: rembg locates the object,
  SAM (prompted with object points + negative corner points, mask chosen by
  coarse-overlap) recovers the *complete* object without drifting to the table.
- **Masked metrics**: PSNR over object pixels, SSIM/LPIPS over the object bbox.
- **Point-cloud init is kept** in `output_masked/transforms.json` — dropping it makes
  splatfacto overfit on the textureless black background (train views perfect, held-out
  views misaligned).

## Augmentation strategies (object-centric)

- **inpaint** ✅ — degrade a region **on the object** (blur, placed inside the mask bbox)
  then restore it → recover degraded object detail. `gen_inpaint(..., focus_box=bbox)`.
- **zero123** ✅ (preview-only for now) — genuine **novel object viewpoints** via
  `kxic/zero123-xl` (`src/gsfewshot/zero123.py`). To train with it: re-key the white
  output onto black + set the camera pose to orbit the object centre.
- **outpaint** ❌ dropped — extends blank background.
- **guided** ❌ dropped — text2img hallucinated; img2img just copied the input.

---

## Adding a scene

1. Run `ns-process-data video --data your.mp4 --output-dir output_scene2` (or images).
2. Add it under `scenes:` in `configs/project.yaml`.
3. `gen_grid.py --scene scene2 && make_splits.py --scene scene2` and run experiments.

---

## Environment notes

- **venv** at `.venv` (Python 3.10). `torch==2.1.2+cu118`, `nerfstudio 1.1.5`,
  `gsplat 1.4.0+pt21cu118`, `diffusers 0.27.2`.
- **CUDA / gsplat:** the system `nvcc` is **11.5**, which cannot compile for the
  4060's Ada `sm_89`. So we install gsplat's **prebuilt** `+pt21cu118` wheel (built
  with CUDA 11.8) — it ships `gsplat/csrc.so` and needs no local compilation. Do
  **not** `pip install gsplat` from PyPI (that pulls a source wheel that JIT-fails).
  `setup.sh` handles this; `scripts/_bootstrap.py` also sets
  `TORCH_CUDA_ARCH_LIST=8.6+PTX` as a harmless fallback in case any extension ever
  does compile.
- **setuptools<81:** required so `pkg_resources` still exists for torch 2.1.2's
  `cpp_extension`.
- **8 GB VRAM:** training runs at `downscale_factor=2` (960×540) by default; diffusion
  uses attention/VAE slicing. Tune both in `configs/project.yaml`.

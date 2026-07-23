# Studying the Effect of Diffusion Augmentation on Few-Shot Gaussian Splatting

Train Gaussian Splatting (`splatfacto`) from a **handful** of real views, generate the
missing supervision with a **diffusion model**, and measure whether it recovers the
quality of a full capture. Object-centric study of a scale-model Abrams tank; every
score is **object-masked PSNR** (the object is measured, never the background).

> **The short version:** diffusion augmentation *does* help few-shot object GS — but as
> clean **same-pose regularization** (inpaint), **not** as novel-view synthesis
> (Zero123 / SVD, which soften and *hurt*). And getting to that answer meant discovering
> that **two of our own setup choices** were quietly deciding the outcome.

📄 **Full report (the whole story + figures):** [`results/report/fewshot_report.html`](results/report/fewshot_report.html)
📊 **All comparisons at a glance:** [`results/report/fewshot_dashboard.html`](results/report/fewshot_dashboard.html)
🧭 **Working log / train-of-thought:** [`results/report/TRAIN_OF_THOUGHT.md`](results/report/TRAIN_OF_THOUGHT.md)

---

## Headline findings

- **Two confounds each moved the numbers more than any augmentation did:**
  1. **Object-on-black segmentation crippled few-shot.** Same 20 views, natural background
     instead of black: baseline PSNR **6.3 → 27.4**, with *no* augmentation. Few-shot GS
     needs background texture to anchor a sparse optimization; black destabilizes it.
  2. **The full-320-view ply init leaked geometry** into every "few-shot" run — worth
     **~+7 PSNR** (n5: `20.95` with the ply vs `14.39` random init). *20 honest views ≈ 5
     views + the full ply.*
- **Once both are removed** (natural background + random init), the honest picture:

  | masked PSNR | baseline | + inpaint (best) | + SVD |
  |---|---|---|---|
  | n5  | 14.4 | 16.1 (r200, **+1.7**) | ≈ flat |
  | n10 | 14.8 | 16.1 (r100, **+1.3**) | ≈ flat |
  | n20 | 20.9 | 23.8 (r100, **+2.9**) | **−2 to −3** at high dose |
  | full (all views) | **30.3** | | |

- **Inpaint helps, peaks near a 1:1 ratio.** **SVD is dose-dependent** — a little helps at
  n20, a lot dilutes (SVD frames are soft diffusion output; piling them on drags the
  reconstruction toward that softer average). Verified *not* a pose/intrinsics bug.
- **The unifying insight:** augmentation's value **inverts with init quality** — synthetic
  frames *help a poor init* (they add missing supervision) and *dilute a good one*.

---

## What's in this repo (and what isn't)

Tracked here: all **code** (`src/`, `scripts/`), the **experiment configs**, the
**results registry**, and the **report + dashboard**. Deliberately *not* tracked (huge,
regenerable — see `.gitignore`): the 39 GB `archive/`, trained models under
`experiments/`, COLMAP frames under `scenes/`, generated frames under `data/`, `video/`,
and `.venv/`. Every run's numbers live in **`results/registry.jsonl`**.

---

## The experiment grid

Few-shot size **N ∈ {5, 10, 20}** × synthetic:real ratio **{0, 25, 50, 100, 200}%** ×
augmentation **strategy**, plus a `full` (all-views) upper bound.

| Base real N | Synthetic ratio | S = round₊(ratio·N) | Total | Purpose |
|---|---|---|---|---|
| full | — | 0 | all | Upper-bound ceiling |
| 5 / 10 / 20 | 0% | 0 | N | Few-shot baseline (control) |
| 5 / 10 / 20 | 25 / 50 / 100 / 200% | 1 … 2N | N+S | Rising synthetic dose |

`python scripts/gen_grid.py --scene scene02` materializes one YAML per run in
`configs/experiments/`. `configs/project.yaml` is the single source of truth — edit it to
change sizes / ratios / strategies. **Init regimes** the study compares:

- **random init** — no geometry prior; the *honest* few-shot floor.
- **ply init** — the full-scene point cloud (leaks geometry; kept as a `*_plyinit` reference and for `full`).
- **own-ply** — COLMAP on each experiment's *own* views (feasible only at n20 — fewer views don't register).

---

## Augmentation strategies

- **inpaint** ✅ — degrade a region **on the object** (masked to the object bbox) then let
  SD-inpainting **restore** it. A frame at the *exact real pose*, photometrically
  consistent — same-pose regularization. `src/gsfewshot/synthetic.py`.
- **svd** — genuine novel viewpoints via **Stable Video Diffusion**, with poses solved by
  running **COLMAP on real + generated together** (no hand-computed geometry).
  `gen_svd_neighbors.py` → `svd_segment.py` → `svd_register.py` → `svd_build_pool.py`.
  Poses/intrinsics verified correct; still *dilutes* because the frames are soft.
- **zero123** ❌ — earlier arm; hand-computed orbit poses diverged (~6 PSNR). Superseded by
  SVD's COLMAP-solved poses.
- **outpaint / guided** ❌ — dropped early (extend blank background / hallucinate).

---

## Reproduce a run

```bash
bash setup.sh && source .venv/bin/activate            # one-time install (venv + deps)

python scripts/gen_grid.py    --scene scene02          # enumerate the grid
python scripts/make_splits.py --scene scene02          # fixed test set + nested few-shot subsets

# one experiment end-to-end: assemble (real + synthetic) -> ns-train -> masked eval -> registry
python scripts/run_experiment.py --exp scene02__n20_r100_inpaint --config configs/project.yaml

python scripts/compare.py                              # rebuild the comparison dashboard
```

**Init control:** *random* = strip `ply_file_path` from the scene `transforms.json`
(then restore); *ply* = keep it and use the capped config `configs/project_natbg.yaml`
(the cap prevents a many-view densification runaway). The grid runners
(`run_natbg_random_grid.sh`, `run_ownply_grid.sh`, …) automate the swaps with a
trap-restore.

**Fair evaluation:** `make_splits.py` reserves a fixed held-out set of real views (every
8th frame), identical for every experiment. Each trained model is rendered at those exact
poses and scored with object-masked PSNR/SSIM/LPIPS; a global similarity is fit from the
training cameras so numbers are comparable across the grid (`pose_fit_residual ≈ 0`).

---

## Scenes & the pivot

- **`scene01`, `scene01_obj`** — first captures, archived to `archive/scene01/`.
- **`scene02`** — the main study. Originally built **segment-first** (object-on-black
  before COLMAP); after discovering segmentation was the handicap, it was **re-run on the
  natural background** (train on the full image, still score only the object). The
  segmented pipeline is preserved in `archive/scene02_segmented/` (restorable).

---

## Layout

```
configs/
  project.yaml            # single source of truth (scenes, grid, diffusion, training)
  project_natbg.yaml      # natural-bg + densification cap (ply-init runs)
  experiments/            # generated: one <exp_id>.yaml per run + index.csv
src/gsfewshot/            # importable package (all logic)
scripts/                  # CLIs + pipeline: run_experiment, compare, gen_svd_*,
                          #   svd_register, build_honest_ply, grid runners, diagnostics
results/
  registry.jsonl          # append-only record of every run  <-- source of truth
  report/                 # the report + dashboard + TRAIN_OF_THOUGHT.md
experiments/<exp_id>/     # per-run: assembled data / trained model / eval  (git-ignored)
scenes/, data/, archive/  # frames, generated views, archived pipelines     (git-ignored)
```

`exp_id` examples: `scene02__full` · `scene02__n5_r0` (baseline) ·
`scene02__n20_r100_inpaint` · `scene02__n10_r200_svd` · `..._plyinit` / `..._ownply`
(init-regime variants).

---

## Environment notes

- **venv** (`Python 3.10`): `torch==2.1.2+cu118`, `nerfstudio 1.1.5`,
  `gsplat 1.4.0+pt21cu118`, `diffusers 0.27.2`.
- **gsplat:** the system `nvcc` is **11.5**, which can't compile for the 4060's Ada
  `sm_89`, so we install gsplat's **prebuilt** `+pt21cu118` wheel (no local compilation).
  Do **not** `pip install gsplat` from PyPI. `setup.sh` handles this.
- **setuptools<81** so `pkg_resources` still exists for torch 2.1.2's `cpp_extension`.
- **8 GB VRAM:** training at `downscale_factor=2` (960×540); diffusion uses attention /
  VAE slicing; SVD uses model-offload + forward-chunking. Segmentation is `isnet → SAM`
  (`src/gsfewshot/segment.py`).
- Target box: RTX 4060 (8 GB), no sudo.

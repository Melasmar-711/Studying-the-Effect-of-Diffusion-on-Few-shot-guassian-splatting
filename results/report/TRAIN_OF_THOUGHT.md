# Few-Shot Gaussian Splatting + Diffusion Augmentation — Train of Thought & Report

*Working log of the whole investigation, in the order we reasoned through it.
Written before archiving the segmented-pipeline experiments so nothing is lost.
Culminates in a discovery that reframes the study: **object-on-black segmentation
was itself crippling the few-shot baselines**, so everything is being re-run on
natural backgrounds.*

---

## 0. The goal

Train `splatfacto` (nerfstudio/gsplat) from **few real views** (N = 5, 10, 20) of an
object, **augment** the training set with diffusion-generated synthetic views, and
measure how the **synthetic : real ratio** (0 / 25 / 50 / 100 / 200 %) affects
held-out reconstruction quality (PSNR / SSIM / LPIPS), plus train time / VRAM.
Object of study: a scale-model Abrams tank. A `full` (all-views) run is the upper bound.

Design: installable package `gsfewshot` + thin CLIs; `configs/project.yaml` is the
single source of truth; `results/registry.jsonl` is the append-only record;
held-out eval fits a global similarity from training-camera correspondences so PSNR
is comparable across runs (`pose_fit_residual ≈ 0`).

---

## 1. scene01 — first capture, and the pivot to object-centric

The first video had a busy background. We pivoted to an **object-centric** setup:
segment the tank onto a **fixed black background** (`isnet` → `SAM`), and compute
**mask-restricted** metrics so the score measures the *object*, not the background.

Findings (scene01):
- **Inpaint augmentation helped**: masked PSNR n20 ~4 → ~15 across ratio 0 → 200,
  visually confirmed (r0 rendered black; high ratio rendered a recognizable tank).
- `full` upper bound = **19.75**, but only with **random init** — point-cloud init
  with many views triggered a **densification runaway → OOM** (a recurring theme).
- Few-shot behaviour was "good near training views, black on novel views."

We archived scene01 and treated its main lesson as: *inpaint helps, but it's a
regularizer at the real poses, not true novel-view synthesis.*

## 2. scene02 — segment-FIRST pipeline (cleaner capture)

New video on a cleaner (white fabric) background. Key idea: **segment the whole
video BEFORE COLMAP**, so COLMAP's output is inherently object-only (no point
cropping, consistent scale). Pipeline: `extract_and_segment.py` (320 frames →
object-on-black + masks) → `run_colmap_scene02.sh` (COLMAP 3.11) → **320/320
registered** → object-only `sparse_pc.ply`.

**Segmented results (masked PSNR):**

| few-shot | r0 (baseline) | inpaint r100 | inpaint r200 | full |
|----------|---------------|--------------|--------------|------|
| n5       | 6.74          | 13.90        | 17.49        |      |
| n10      | 6.92          | 18.63        | 19.64        |      |
| n20      | 6.34\*        | **23.49**    | 22.00        | **24.98** |

\* n20_r0 needed a densification cap (ply-init runaway). Verdict at the time:
inpaint reaches **94 % of the full ceiling** — a large, monotonic, visually-confirmed
effect — but **the same honest caveat**: inpaint frames sit at the **real training
poses** (verified), so they add clean supervision / regularization, **not** new
viewpoints. The baselines looked uniformly poor (~6.3–6.9). *We took this at face
value as "few-shot collapses; augmentation rescues it." That interpretation was
wrong — see §6.*

## 3. Zero123 — the first attempt at *genuine* novel views

To test whether diffusion can add real new viewpoints, we tried Zero123
(`kxic/zero123-xl`): generate the object at novel azimuths, **compute** the camera
pose ourselves (orbit axis via PCA, azimuth rotation, project the object centroid),
re-key onto black, and feed it in.

**Result: FAILED.** The augmentation diverged into "needle-fog," PSNR ~6–7 (below
baseline, far below inpaint); the combo dragged inpaint down. Diagnosis (not a
placement bug): silhouettes matched the true render to ~8°, but the frames were
**fundamentally inconsistent** — Zero123 invents per-frame surface detail and its
render **FOV ≠ the scene focal length**. Few-shot GS can't reconcile a handful of
exact real frames with contradictory hallucinated ones.

The open worry: *was it the poses we hand-computed, or the images themselves?*

## 4. SVD video-extension — fixing the pose problem

Insight: **stop computing poses ourselves.** Extend the real video with a video
diffusion model, then **run COLMAP on real + generated together so COLMAP assigns
the poses.** Registration also acts as a free consistency filter.

Pipeline (all reusable): `gen_svd_neighbors.py` (Stable-Video-Diffusion img2vid;
8 GB-friendly: 896×512, fp16 + model-offload + attention-slicing + forward-chunking;
seed a 14-frame clip from each few-shot view) → `svd_segment.py` (isnet+SAM) →
`svd_register.py` (COLMAP co-solve, `single_camera_per_folder` so the generated
group gets its OWN intrinsics; OpenCV→OpenGL flip auto-picked by min residual;
similarity-fit into the dataset frame) → `svd_build_pool.py`.

**Mechanism validated — both Zero123 failure modes removed:**
- **271/280** generated frames co-registered onto the orbit.
- Similarity fit residual **0.044** (across 121 shared frames, two independent solves).
- Recovered intrinsics **fl_x 1790 vs real 1775 (~1 %)** — FOV is correct.
- Pose-consistency check: rendering the full model at each SVD frame's pose gave
  **silhouette IoU 0.92** — the frames sit at the right orientation.

**But the augmentation was NEUTRAL:**

| masked PSNR | baseline | inpaint | SVD (full ply) | SVD (short+SVD ply) |
|-------------|----------|---------|----------------|---------------------|
| n10 r100    | 6.92     | 18.63   | 6.95           | 6.70                |
| n10 r200    | 6.92     | 19.64   | 6.51           | 6.48                |
| n20 r100    | 6.34\*   | 23.49   | 6.28\*         | 6.25\*              |
| n20 r200    | 6.34\*   | 22.00   | 6.24\*         | 6.25\*              |

We then chased every remaining explanation and ruled each out:
- **Init geometry:** built a ply from a *shortened* 40-frame capture **completed by
  SVD** (COLMAP registered 40/40 real + 550/560 SVD → 15 k-pt object cloud). No
  change (a hair lower — sparser/noisier). → not the init.
- **Random init** (n10 r200): 6.38 — no hidden geometric contribution being masked.
- **Pose bug?** No — IoU 0.92, residual 0.0. The SVD *images* are just soft and
  slightly wrong (e.g. a truncated gun barrel), so they mildly **conflict** with the
  faithful real frames (dose-response is slightly *negative*: r100 6.95 → r200 6.51).

Interim conclusion (correct as far as it went): *diffusion helps few-shot object GS
as clean **same-pose** regularization (inpaint), not as **novel-view** synthesis —
even geometrically-perfect novel views carry detail that isn't multiview-consistent
enough to add signal.* Zero123's failure was **never really about poses.**

## 5. The nagging doubt → the control that broke it open

Something didn't add up: adding *decent-looking* extra views did **nothing** — flat
at baseline. That smelled like a plumbing problem, not physics. We audited the SVD
training (init ply loaded ✓, poses/intrinsics/images present ✓, frames genuinely
used ✓) and it was all correct. But one thing was flagged for a **control test**:
the **object-on-black segmentation** itself.

**Test:** train the **n20 baseline on the natural-background (unsegmented) frames**,
same 20 poses, same masked eval.

## 6. THE DISCOVERY — segmentation was crippling few-shot

| n20 baseline, same 20 views, masked | PSNR | SSIM |
|-------------------------------------|------|------|
| **Segmented (object-on-black)**     | 6.34 | —    |
| **Natural background**              | **27.36** | **0.84** |

A **21-point** jump from a background change alone — and the natural-bg few-shot
model **beats inpaint-on-segmented (23.49) and the full 320-view segmented model
(24.98)**. Verified by eye (sharp, correct novel-view renders) and by re-scoring
against the raw photos: **PSNR is identical vs either GT** (27.39 seg / 27.36 raw),
and the low SSIM (0.48) was itself a **segmented-GT artifact** — against the raw
photo SSIM is **0.84** (the black-vs-fabric mask boundary was corrupting SSIM's
local windows; PSNR was immune).

**Why:** few-shot Gaussian Splatting leans on background **texture** for a stable,
well-constrained optimization — the fabric anchors camera geometry and Gaussian
placement. Object-on-black removes all of it, leaving a uniform region with no
photometric gradient → the few-view optimization collapses into a foggy/degenerate
solution. `full` (320 views) had enough constraints to partly survive it; few-shot
did not. **The segmentation, meant to clean up the metric, was destabilizing the
reconstruction.**

### What this means for everything above
- The few-shot "collapse" (baselines ~6.3–6.9) was **largely a segmentation
  artifact**, not an inherent few-shot limitation.
- **Inpaint's gains were mostly recovery from that self-inflicted handicap** — dense
  supervision that re-stabilized black-background training. On natural bg, the bare
  baseline already surpasses it.
- Zero123/SVD being "neutral or slightly harmful" is consistent: on an already-broken
  black-bg baseline there was little to add, and their soft content only conflicted.
- The correct setup is **train on natural background, eval masked** (object region) —
  clean object metrics *and* a stable reconstruction.

---

## 7. Numbers appendix (segmented pipeline, masked PSNR)

- `full` = 24.98
- baselines r0: n5 6.74 · n10 6.92 · n20 6.34
- inpaint: n5 13.90/17.49 · n10 18.63/19.64 · n20 23.49/22.00 (r100/r200)
- zero123 n5: 6.25/6.78 (diverged)
- SVD full-ply: n10 6.95/6.51 · n20 6.28/6.24
- SVD short+SVD-ply: n10 6.70/6.48 · n20 6.25/6.25
- SVD random-init n10 r200: 6.38
- **natural-bg n20 baseline: 27.36 PSNR / 0.84 SSIM**

## 8. Reusable pipeline / key scripts
`extract_and_segment.py` · `run_colmap_scene02.sh` · `run_experiment.py` (assemble→train→
masked eval→registry) · `gen_svd_neighbors.py` · `svd_segment.py` · `svd_register.py` ·
`svd_build_pool.py` · `build_honest_ply.py` · `svd_pose_check.py` · `diag_zero123_consistency.py`
· `render_ply.py` · `compare.py`. Env: RTX 4060 8 GB, gsplat `+pt21cu118`, diffusers 0.27.2.

## 9. Next: re-run everything on natural background
Re-baseline the entire grid — **all baselines, all inpaint, all SVD** — with
**natural-background training + masked eval**. The one thing kept **unchanged**:
**inpaint still manipulates/restores only the pixels ON the object** (masked
inpainting), exactly as before. Open question this finally lets us answer cleanly:
*with the segmentation handicap gone (baseline already ~27), does any augmentation
still help?*

---

## 10. The natural-bg re-run — and a SECOND confound: the ply leak

We archived the segmented pipeline (`archive/scene02_segmented/`, 28 GB, restorable),
swapped the scene images to natural background (same poses/masks/ply), reset the
registry, and re-ran. **Inpaint kept unchanged (object-focused).** But before trusting
the baselines, a control (n5 random init vs ply init) exposed a second confound:

- **n5: with full-320 ply = 20.95, random init (no ply) = 14.39.** The point cloud came
  from COLMAP on **all 320 views**, so every "few-shot" run was handed the *complete
  object geometry* for free — a **~7-point leak, roughly constant across N** (n20:
  28.10 ply → 20.91 random). A neat calibration: **20 honest views ≈ 5 views + full ply.**

Decision: run the honest grid with **random init** for all few-shot + augmentation
(no geometry prior), keep the leaky ply baselines as `*_plyinit` reference, and use **ply
init only for `full`** (to showcase the ply + many-views ceiling).

### Honest random-init natural-bg grid (masked PSNR)
```
   N   base |  inp25  inp50 inp100 inp200 |  svd25  svd50 svd100 svd200
n5    14.39 |  15.07  14.51  15.33  16.06 |    —      —      —      —
n10   14.77 |  14.78  15.39  16.07  15.52 |  13.93  14.71  14.33  15.48
n20   20.91 |  23.59 (18.54) 23.76  21.97 |  22.61 (15.50) 18.83  17.51
full (natural-bg, ply init) = 30.32   |   plyinit ref: n5 20.95 · n10 24.33 · n20 28.10
```
(parenthesised = random-init instability outliers — see caveat.)

**Findings (leak-free):**
- **Inpaint helps at every N**, peaking ~r100 (n5 +1.7 @r200, n10 +1.3 @r100, n20 +2.85
  @r100). Robust — it's sharp same-pose supervision.
- **SVD is dose-dependent, not simply harmful:** at n20 a *small* dose **helps**
  (svd25 = 22.61, **+1.7**) but larger doses **dilute and hurt** (svd100 −2.1, svd200 −3.4).
  A little novel-view coverage is useful; too much soft supervision drags it down.
- **Ceiling context:** honest few-shot + augmentation tops out ~22–24 vs the 30.3 full,
  so there is genuine room and augmentation closes part of it.
- **Caveat: random init is noisy.** The n20 r50 cells (inpaint 18.54, svd 15.50) are clear
  instability outliers. A publishable table needs **multiple seeds** / outlier re-runs.

## 11. SVD deep-dive — "good images that hurt" is NOT a bug
Why do good-looking SVD frames hurt? We verified, twice, that it is **not** a
pose/intrinsics/resolution bug: the object centroid projects **onto the tank** in every
frame (`svd_align_check.png`), and rendering a good model at each SVD pose matches 3/4
frames (`svd_multiview_clincher.png`). The frames really are fine. The cause is
**dilution**: SVD frames are diffusion output — slightly soft/approximate vs a real photo
— so mixing many of them pulls the reconstruction toward that softer average. Inpaint
avoids this because its frames *are* the real image with detail restored (sharp, faithful).
So SVD is **lower-quality supervision**, which costs you where real signal is already good
(n20, high ratio). Key evidence: the monotonic n20 svd100→svd200 decline.

## 12. Own-ply arm — per-experiment init
A separate arm (does not touch the above): every experiment inits from a ply built by COLMAP on
**its own natural-bg data** — baseline/inpaint from the N real views, SVD from N real + SVD
frames (`run_ownply_grid.sh`, `_ownply` exp-ids). **Result: n20-ONLY.** COLMAP could NOT build a
ply from few natural-bg views (n10 2/10, n5 2/5, n10_svd 0/10) — so **realistic few-shot has no
usable COLMAP ply below ~20 views here** (itself a finding; matches the earlier segmented failure).
n20 own-ply (masked PSNR): **base 27.1** — nearly the leaky plyinit 28.1 and far above random 20.9
(20 own views ≈ the full ply), inpaint 27.7/25.5/25.3/24.4 (r25/50/100/200), svd r100 = 20.6 (svd
hurts even here). n5 svd random-init + n20 svd own-ply r25/50/200 filled via `run_grid_fill.sh`.

## 13. Future directions
- Multi-seed averaging to tighten the noisy random-init cells (esp. n20 r50).
- A "realistic sparse-ply-per-N" grid — build the init from each N's own views where they
  register, extending the own-ply arm below n20.
- Multi-view-consistent generators (e.g. SV3D or camera-controlled video diffusion) as the
  next test of whether *any* novel-view synthesis can reinforce rather than dilute.

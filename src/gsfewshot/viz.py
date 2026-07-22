"""Comparison + reporting: tables, metric plots, GT-vs-render panels, and a
single self-contained HTML dashboard built from the experiment registry.

This is the "pull up a comparison" surface: run ``scripts/compare.py`` any time
and it regenerates ``results/comparisons/index.html`` from whatever runs are
recorded so far.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

from .config import COMPARISONS_DIR, EXPERIMENTS_DIR
from .registry import to_dataframe

_METRICS = ["psnr", "ssim", "lpips"]


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def results_table(df, out_dir: Path) -> Path:
    """Write a tidy results table as CSV + Markdown. Returns the .md path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ["exp_id", "kind", "n_real", "ratio_pct", "strategies",
                        "s_synth", "total_images", "psnr", "ssim", "lpips",
                        "train_time_s", "peak_vram_mb", "num_test_views"]
            if c in df.columns]
    t = df[cols].copy()
    for m in _METRICS:
        if m in t:
            t[m] = t[m].round(4)
    t = t.sort_values([c for c in ["n_real", "ratio_pct", "strategies"] if c in t])
    t.to_csv(out_dir / "results.csv", index=False)
    md = t.to_markdown(index=False)
    (out_dir / "results.md").write_text(md)
    return out_dir / "results.md"


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()


def metric_vs_ratio_plots(df, out_dir: Path, tag: str = "") -> dict[str, str]:
    """One figure per metric: metric vs synthetic ratio, a line per strategy,
    faceted by few-shot N. Full-data baseline drawn as a dashed reference.

    ``df`` MUST already be scoped to a single scene — the full-data reference and
    the augmented curves are taken from ``df`` as-is, so mixing scenes here would
    plot one scene's full baseline against another's ratio curves. ``tag`` (the
    scene name) prefixes the output PNG filenames. Returns {metric: base64_png}.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    pfx = f"{tag}_" if tag else ""
    aug = df[df["kind"].isin(["augmented", "fewshot_baseline"])].copy()
    ns = sorted(aug["n_real"].dropna().unique()) if not aug.empty else []
    full = df[df["kind"] == "full"]

    imgs: dict[str, str] = {}
    for metric in _METRICS:
        if metric not in df.columns or aug.empty:
            continue
        fig, axes = plt.subplots(1, max(1, len(ns)), figsize=(5 * max(1, len(ns)), 4),
                                 squeeze=False)
        for ax, n in zip(axes[0], ns):
            sub = aug[aug["n_real"] == n]
            for strat in sorted(x for x in sub["strategies"].unique() if x):
                s = sub[sub["strategies"] == strat].sort_values("ratio_pct")
                if not s[metric].notna().any():
                    continue
                ax.plot(s["ratio_pct"], s[metric], marker="o", label=strat)
            # baseline (ratio 0) point may live under empty strategy
            b = sub[(sub["ratio_pct"] == 0)]
            if not b.empty and b[metric].notna().any():
                ax.scatter([0], [b[metric].iloc[0]], color="black", zorder=5,
                           label="baseline (0%)")
            if not full.empty and full[metric].notna().any():
                ax.axhline(full[metric].dropna().iloc[0], ls="--", color="gray",
                           label="full-data")
            ax.set_title(f"N = {int(n)} real")
            ax.set_xlabel("synthetic ratio (%)")
            ax.set_ylabel(metric.upper())
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(f"{tag + ' — ' if tag else ''}{metric.upper()} vs synthetic ratio")
        (out_dir / f"{pfx}{metric}_vs_ratio.png").write_bytes(
            base64.b64decode(_b := _fig_to_b64(fig)))
        imgs[metric] = _b
    return imgs


# --------------------------------------------------------------------------- #
# Qualitative panels
# --------------------------------------------------------------------------- #
def make_panels(exp_ids: list[str], out_dir: Path, max_views: int = 3) -> list[Path]:
    """Stack GT | render side-by-side for a few held-out views per experiment."""
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []
    for exp_id in exp_ids:
        rdir = EXPERIMENTS_DIR / exp_id / "eval" / "renders"
        if not rdir.exists():
            continue
        gts = sorted(rdir.glob("*_gt.png"))[:max_views]
        rows = []
        for gt in gts:
            rp = gt.with_name(gt.name.replace("_gt.png", "_render.png"))
            if not rp.exists():
                continue
            g, r = Image.open(gt), Image.open(rp)
            h = min(g.height, r.height)
            g = g.resize((int(g.width * h / g.height), h))
            r = r.resize((int(r.width * h / r.height), h))
            row = Image.new("RGB", (g.width + r.width + 8, h), (0, 0, 0))
            row.paste(g, (0, 0)); row.paste(r, (g.width + 8, 0))
            rows.append(row)
        if not rows:
            continue
        W = max(x.width for x in rows)
        panel = Image.new("RGB", (W, sum(x.height for x in rows) + 8 * (len(rows) - 1)),
                          (255, 255, 255))
        y = 0
        for x in rows:
            panel.paste(x, (0, y)); y += x.height + 8
        p = out_dir / f"panel_{exp_id}.png"
        panel.save(p)
        made.append(p)
    return made


# --------------------------------------------------------------------------- #
# HTML dashboard
# --------------------------------------------------------------------------- #
def _img_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def build_dashboard(out_dir: Path | None = None) -> Path:
    out_dir = out_dir or COMPARISONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    df = to_dataframe()
    if df.empty:
        (out_dir / "index.html").write_text(
            "<h1>No runs recorded yet</h1><p>Train an experiment, then re-run "
            "<code>scripts/compare.py</code>.</p>")
        return out_dir / "index.html"

    results_table(df, out_dir)          # combined table (all scenes) on disk
    show = [c for c in ["exp_id", "n_real", "ratio_pct", "strategies", "s_synth",
                       "total_images", "psnr", "ssim", "lpips", "train_time_s",
                       "peak_vram_mb"] if c in df.columns]

    scenes = sorted(df["scene"].dropna().unique()) if "scene" in df else [None]
    parts = ["<h1>Few-Shot GS + Diffusion — comparison</h1>",
             f"<p>{len(df)} experiment(s) across {len(scenes)} scene(s): "
             f"{', '.join(str(s) for s in scenes)}.</p>"]

    # One self-contained section per scene: its own table, plots (with that
    # scene's OWN full-data baseline), and qualitative panels. Never mix scenes.
    for scene in scenes:
        d = df[df["scene"] == scene] if scene is not None else df
        parts.append(f"<hr><h2>Scene: {scene}</h2>")
        parts.append(d[show].sort_values(
            [c for c in ["n_real", "ratio_pct"] if c in d]).to_html(index=False))
        plots = metric_vs_ratio_plots(d, out_dir, tag=str(scene))
        if plots:
            parts.append("<h3>Metrics vs synthetic ratio</h3>")
            for m, b in plots.items():
                parts.append(f"<h4>{m.upper()}</h4>"
                             f"<img src='data:image/png;base64,{b}'/>")
        panel_ids = d.sort_values("total_images")["exp_id"].tolist()[:6]
        panels = make_panels(panel_ids, out_dir / "panels")
        if panels:
            parts.append("<h3>Qualitative (GT | render)</h3>")
            for p in panels:
                parts.append(f"<h5>{p.stem}</h5>"
                             f"<img src='data:image/png;base64,{_img_b64(p)}'/>")

    style = ("<style>body{font-family:system-ui,sans-serif;margin:2rem;max-width:"
             "1100px}img{max-width:100%;border:1px solid #ddd;border-radius:6px}"
             "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:"
             "4px 8px;font-size:13px}th{background:#f4f4f4}</style>")
    (out_dir / "index.html").write_text(style + "\n".join(parts))
    return out_dir / "index.html"

#!/usr/bin/env python
"""Quick multi-view render of one or more colored PLY point clouds -> PNG."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def load(p):
    xyz, rgb = [], []
    started = False
    for l in Path(p).read_text().splitlines():
        if started and l.strip():
            v = l.split()
            if len(v) >= 6:
                try:
                    xyz.append([float(v[0]), float(v[1]), float(v[2])])
                    rgb.append([int(v[3]), int(v[4]), int(v[5])])
                except ValueError:
                    pass
        if l.startswith("end_header"):
            started = True
    return np.array(xyz), np.array(rgb) / 255.0


plys = sys.argv[1:-1]
out = sys.argv[-1]
views = [(90, -90, "top-down"), (20, -60, "oblique"), (5, 0, "side")]
fig = plt.figure(figsize=(5 * len(views), 5 * len(plys)))
for r, p in enumerate(plys):
    xyz, rgb = load(p)
    # keep to the object core (robust bbox) so background outliers don't shrink it
    lo, hi = np.percentile(xyz, 2, 0), np.percentile(xyz, 98, 0)
    m = np.all((xyz >= lo) & (xyz <= hi), axis=1)
    xyz, rgb = xyz[m], rgb[m]
    for c, (el, az, name) in enumerate(views):
        ax = fig.add_subplot(len(plys), len(views), r * len(views) + c + 1, projection="3d")
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=1.5, depthshade=False)
        ax.view_init(elev=el, azim=az)
        ax.set_title(f"{Path(p).stem}  ({len(xyz)} pts)  {name}", fontsize=9)
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
plt.tight_layout()
Path(out).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=110, bbox_inches="tight")
print(f"-> {out}")

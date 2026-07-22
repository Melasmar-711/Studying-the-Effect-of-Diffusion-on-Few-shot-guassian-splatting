#!/usr/bin/env python
"""Top-down plot of the orbit: real cameras + COLMAP-registered SVD-generated cams."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

M = Path("/tmp/claude-1000/-home-asmar-GS-VR/f9750b5d-50b0-4def-974a-72ced49c4ec7/scratchpad/gate2/sparse/1")
lines = [l for l in (M/"images.txt").read_text().splitlines() if l and not l.startswith("#")]
poses = lines[0::2]

def center(q, t):
    w, x, y, z = q
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    return -R.T @ t

realC, genC, seed = [], [], None
for p in poses:
    f = p.split(); q = list(map(float, f[1:5])); t = np.array(list(map(float, f[5:8]))); nm = f[9]
    C = center(q, t)
    if "gen" in nm: genC.append(C)
    else:
        realC.append(C)
        if "00319" in nm: seed = C
realC = np.array(realC); genC = np.array(genC)
ctr = realC.mean(0)
_, _, Vt = np.linalg.svd(realC - ctr)
e1, e2 = Vt[0], Vt[1]
proj = lambda P: np.c_[(P-ctr)@e1, (P-ctr)@e2]
rp, gp, sp = proj(realC), proj(genC), proj(seed[None])[0]

plt.figure(figsize=(7, 7))
plt.scatter(rp[:, 0], rp[:, 1], s=18, c="#4C78A8", label=f"real cameras ({len(realC)})")
plt.scatter(gp[:, 0], gp[:, 1], s=45, c="#E45756", marker="^",
            label=f"SVD-generated, COLMAP-registered ({len(genC)})", zorder=3)
plt.scatter([sp[0]], [sp[1]], s=160, c="#F58518", marker="*",
            edgecolor="k", label="seed frame_00319", zorder=4)
plt.scatter([0], [0], s=80, c="k", marker="+", label="orbit center")
plt.gca().set_aspect("equal"); plt.grid(alpha=.3)
plt.title("Orbit (top-down): SVD frames land on the ring, clustered ~3-9° from the seed")
plt.xlabel("orbit-plane x"); plt.ylabel("orbit-plane y"); plt.legend(loc="upper right", fontsize=8)
out = Path("results/spikes/orbit_registration.png")
plt.savefig(out, dpi=110, bbox_inches="tight")
print(f"-> {out}")

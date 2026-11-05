"""Generate the figures shown in the README.

    python demo.py
"""
import time

import matplotlib.pyplot as plt
import numpy as np

from geometry import convex_hull, delaunay, voronoi_edges

rng = np.random.default_rng(0)
pts = rng.random((60, 2))


def save(fig, name):
    fig.savefig(f"images/{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# 1. convex hull
hull = convex_hull(pts)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(*pts.T, "o", color="#1f4e79", ms=4)
ax.fill(*hull.T, alpha=0.15, color="#1f4e79")
ax.plot(*np.vstack([hull, hull[:1]]).T, "-", color="#1f4e79", lw=2)
ax.set_title("Convex hull (monotone chain)")
ax.set_aspect("equal"); ax.axis("off")
save(fig, "convex_hull")

# 2. Delaunay triangulation
tris = delaunay(pts)
fig, ax = plt.subplots(figsize=(5, 5))
ax.triplot(pts[:, 0], pts[:, 1], tris, color="#1f4e79", lw=0.8)
ax.plot(*pts.T, "o", color="#c0392b", ms=3)
ax.set_title("Delaunay triangulation (Bowyer-Watson)")
ax.set_aspect("equal"); ax.axis("off")
save(fig, "delaunay")

# 3. Voronoi diagram on top of the triangulation
edges = voronoi_edges(pts, tris)
fig, ax = plt.subplots(figsize=(5, 5))
ax.triplot(pts[:, 0], pts[:, 1], tris, color="#bbbbbb", lw=0.5)
for p, q in edges:
    ax.plot([p[0], q[0]], [p[1], q[1]], color="#1f4e79", lw=1.2)
ax.plot(*pts.T, "o", color="#c0392b", ms=3)
ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
ax.set_title("Voronoi diagram (dual of Delaunay)")
ax.set_aspect("equal"); ax.axis("off")
save(fig, "voronoi")

# 4. running time
sizes = [100, 200, 400, 800]
times = []
for n in sizes:
    p = rng.random((n, 2))
    t0 = time.perf_counter(); delaunay(p); times.append(time.perf_counter() - t0)
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.loglog(sizes, times, "o-", color="#1f4e79")
ax.set_xlabel("number of points"); ax.set_ylabel("seconds")
ax.set_title("Bowyer-Watson running time")
ax.grid(True, which="both", alpha=0.3)
save(fig, "timing")
print("Figures saved in images/")
for n, t in zip(sizes, times):
    print(f"{n:5d} points: {t:.2f} s")

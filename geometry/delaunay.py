"""Delaunay triangulation with the Bowyer-Watson algorithm.

A triangulation is Delaunay when no point lies inside the circumcircle of any
triangle. Among all triangulations it maximises the smallest angle, which is
why it is the standard choice for meshes.

Bowyer-Watson inserts the points one by one:
1. start with a "super triangle" that contains every point;
2. for each new point, find the triangles whose circumcircle contains it
   (the "bad" triangles) and remove them, leaving a star-shaped hole;
3. connect the new point to every edge on the boundary of the hole;
4. at the end, remove the triangles that use a vertex of the super triangle.

This simple version runs in O(n^2) in the worst case, which is fine for a few
thousand points. Production libraries (Qhull, CGAL) are faster and use exact
arithmetic to stay robust.
"""
import numpy as np

from .predicates import in_circumcircle, orientation


def delaunay(points):
    """Return an array of triangles, each given as 3 indices into `points`,
    with vertices in counter-clockwise order."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)

    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centre, size = (lo + hi) / 2, (hi - lo).max() + 1.0
    # a super triangle that is too small can wrongly remove triangles on the
    # convex hull, so it is made very large compared with the input
    big = 1e4 * size
    super_pts = np.array([
        centre + [-2 * big, -big],
        centre + [2 * big, -big],
        centre + [0, 2 * big],
    ])
    allpts = [tuple(p) for p in np.vstack([pts, super_pts])]
    triangles = [(n, n + 1, n + 2)]

    for i in range(n):
        p = allpts[i]
        bad = [t for t in triangles
               if in_circumcircle(allpts[t[0]], allpts[t[1]], allpts[t[2]], p) > 0]

        # boundary of the hole = edges that belong to exactly one bad triangle
        edge_count = {}
        for a, b, c in bad:
            for e in ((a, b), (b, c), (c, a)):
                key = tuple(sorted(e))
                edge_count[key] = edge_count.get(key, 0) + 1
        boundary = [e for e, k in edge_count.items() if k == 1]

        triangles = [t for t in triangles if t not in bad]
        for a, b in boundary:
            # keep every triangle counter-clockwise
            if orientation(allpts[a], allpts[b], p) > 0:
                triangles.append((a, b, i))
            else:
                triangles.append((b, a, i))

    result = [t for t in triangles if max(t) < n]
    return np.array(result, dtype=int)


def is_delaunay(points, triangles):
    """Check the empty-circumcircle property (used by the tests)."""
    pts = np.asarray(points, dtype=float)
    for t in triangles:
        a, b, c = pts[t]
        for j, d in enumerate(pts):
            if j in t:
                continue
            if in_circumcircle(a, b, c, d) > 1e-9:
                return False
    return True

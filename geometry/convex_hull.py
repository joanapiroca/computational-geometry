"""Convex hull of a set of points in the plane.

Andrew's monotone chain algorithm, O(n log n):
1. sort the points by x (then y);
2. build the lower hull from left to right, removing the last point while
   the last three points do not make a left turn;
3. build the upper hull from right to left in the same way;
4. join both chains.
"""
import numpy as np

from .predicates import orientation


def convex_hull(points):
    """Return the hull vertices in counter-clockwise order.

    points: array-like of shape (n, 2). Collinear points on the boundary are
    discarded, so only the corners of the hull are returned.
    """
    pts = sorted(set(map(tuple, np.asarray(points, dtype=float))))
    if len(pts) <= 2:
        return np.array(pts)

    def half_hull(sequence):
        chain = []
        for p in sequence:
            while len(chain) >= 2 and orientation(chain[-2], chain[-1], p) <= 0:
                chain.pop()
            chain.append(p)
        return chain

    lower = half_hull(pts)
    upper = half_hull(reversed(pts))
    # the last point of each chain is the first point of the other one
    return np.array(lower[:-1] + upper[:-1])


def polygon_area(polygon):
    """Area of a simple polygon given in order (shoelace formula)."""
    x, y = np.asarray(polygon).T
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

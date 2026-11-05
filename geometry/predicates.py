"""Basic geometric predicates in the plane.

Every algorithm in this repository is built on two questions:

* orientation(a, b, c): does the path a -> b -> c turn left, turn right or go straight?
* in_circumcircle(a, b, c, d): is d inside the circle through a, b and c?

Both reduce to the sign of a determinant, so the algorithms never need angles,
square roots or trigonometry.
"""
import numpy as np


def orientation(a, b, c):
    """Twice the signed area of triangle abc.

    > 0  counter-clockwise (left turn)
    < 0  clockwise (right turn)
    = 0  collinear
    """
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def in_circumcircle(a, b, c, d):
    """Positive if d lies strictly inside the circumcircle of the
    counter-clockwise triangle abc, negative if outside, zero if on it.

    Uses the classic 3x3 "lifting" determinant: points are lifted to the
    paraboloid z = x^2 + y^2, and the circle test becomes a plane test.
    """
    adx, ady = a[0] - d[0], a[1] - d[1]
    bdx, bdy = b[0] - d[0], b[1] - d[1]
    cdx, cdy = c[0] - d[0], c[1] - d[1]
    ad, bd, cd = adx * adx + ady * ady, bdx * bdx + bdy * bdy, cdx * cdx + cdy * cdy
    return (adx * (bdy * cd - bd * cdy)
            - ady * (bdx * cd - bd * cdx)
            + ad * (bdx * cdy - bdy * cdx))


def circumcenter(a, b, c):
    """Centre of the circle through a, b and c (they must not be collinear)."""
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
    return np.array([ux, uy])

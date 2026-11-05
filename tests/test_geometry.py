"""Tests: compare every algorithm with SciPy (Qhull) on random point sets."""
import numpy as np
from scipy.spatial import ConvexHull, Delaunay

from geometry import convex_hull, delaunay, is_delaunay, polygon_area, voronoi_edges


def random_points(n, seed):
    return np.random.default_rng(seed).random((n, 2))


def test_convex_hull_matches_scipy():
    for seed in range(10):
        pts = random_points(300, seed)
        ours = convex_hull(pts)
        ref = pts[ConvexHull(pts).vertices]
        assert {tuple(p) for p in ours} == {tuple(p) for p in ref}


def test_convex_hull_square():
    square = [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5), (0.5, 0)]
    hull = convex_hull(square)
    assert len(hull) == 4
    assert np.isclose(polygon_area(hull), 1.0)


def test_delaunay_matches_scipy():
    for seed in range(5):
        pts = random_points(150, seed)
        ours = {tuple(sorted(t)) for t in delaunay(pts)}
        ref = {tuple(sorted(t)) for t in Delaunay(pts).simplices}
        assert ours == ref


def test_delaunay_empty_circumcircle():
    pts = random_points(80, 42)
    assert is_delaunay(pts, delaunay(pts))


def test_euler_formula():
    # for a triangulation of n points with h on the hull: T = 2n - h - 2
    pts = random_points(200, 7)
    n, h = len(pts), len(convex_hull(pts))
    assert len(delaunay(pts)) == 2 * n - h - 2


def test_voronoi_edges_are_equidistant():
    pts = random_points(60, 3)
    tris = delaunay(pts)
    for p, q in voronoi_edges(pts, tris):
        mid = (p + q) / 2
        d = np.sort(np.linalg.norm(pts - mid, axis=1))
        assert np.isclose(d[0], d[1], rtol=1e-6)

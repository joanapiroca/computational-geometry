"""Voronoi diagram as the dual of the Delaunay triangulation.

The Voronoi cell of a site is the set of points closer to it than to any other
site. Its vertices are exactly the circumcentres of the Delaunay triangles
around that site, so once we have the triangulation the diagram comes almost
for free:

* every Delaunay triangle gives one Voronoi vertex (its circumcentre);
* two triangles that share an edge give one Voronoi edge (joining their
  circumcentres).

Edges of the triangulation on the convex hull belong to only one triangle;
their Voronoi edges are infinite rays, which are skipped here.
"""
import numpy as np

from .predicates import circumcenter


def voronoi_edges(points, triangles):
    """Return the finite Voronoi edges as an array of shape (m, 2, 2)."""
    pts = np.asarray(points, dtype=float)
    centres = np.array([circumcenter(*pts[t]) for t in triangles])

    # map each Delaunay edge to the triangles that contain it
    edge_to_tris = {}
    for k, (a, b, c) in enumerate(triangles):
        for e in ((a, b), (b, c), (c, a)):
            edge_to_tris.setdefault(tuple(sorted(e)), []).append(k)

    edges = [(centres[t1], centres[t2])
             for tris in edge_to_tris.values() if len(tris) == 2
             for t1, t2 in [tris]]
    return np.array(edges)


def voronoi_vertices(points, triangles):
    """Circumcentres of the Delaunay triangles (one per triangle)."""
    pts = np.asarray(points, dtype=float)
    return np.array([circumcenter(*pts[t]) for t in triangles])

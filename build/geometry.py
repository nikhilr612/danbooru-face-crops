"""Landmark groups and convex hulls for the face / eye / mouth region masks."""

import numpy as np

# Landmark groups of hysts/anime-face-detector's 28 points, as used by tinyflow's layouts.
# Their meaning is inferred, not documented upstream: check them with the review tool's index layer.
FACE = slice(0, 11)  # chin contour (0-4) + brows (5-10)
EYE_A = slice(11, 17)
EYE_B = slice(17, 23)
MOUTH = slice(24, 28)


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Convex hull of ``(k, 2)`` points, counter-clockwise (monotone chain)."""
    pts = sorted(map(tuple, points))
    if len(pts) <= 2:
        return np.asarray(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1])

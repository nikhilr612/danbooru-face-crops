"""Density and coverage (Naeem et al. 2020): k-NN balls around real features; robust to outliers where precision and recall are not."""
import numpy as np


def _sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.clip((a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T, 0, None)


def knn_radii(x: np.ndarray, k: int = 5, chunk: int = 2048) -> np.ndarray:
    """Distance from each row to its k-th nearest other row."""
    x = np.asarray(x, np.float32)
    out = []
    for i in range(0, len(x), chunk):
        out.append(np.sqrt(np.partition(_sqdist(x[i:i + chunk], x), k, axis=1)[:, k]))  # column 0 is the point itself
    return np.concatenate(out)


def density_coverage(real: np.ndarray, radii: np.ndarray, fake: np.ndarray, k: int = 5, chunk: int = 2048) -> dict[str, float]:
    """`radii` = knn_radii(real, k). density: mean number of real balls containing a generated point, over k (about 1 for real data).
    coverage: share of real points whose ball contains at least one generated point (1 = every real region has a nearby sample)."""
    r, f = np.asarray(real, np.float32), np.asarray(fake, np.float32)
    r2 = np.asarray(radii, np.float32) ** 2
    inside = np.zeros(len(f))
    covered = np.zeros(len(r), bool)
    for i in range(0, len(f), chunk):
        d = _sqdist(f[i:i + chunk], r)
        inside[i:i + chunk] = (d <= r2[None]).sum(1)
        covered |= (d <= r2[None]).any(0)
    return {"density": float(inside.mean() / k), "coverage": float(covered.mean())}

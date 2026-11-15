"""Kernel Inception Distance: unbiased MMD^2 with the cubic polynomial kernel (Binkowski et al. 2018)."""
import numpy as np


def kid(a: np.ndarray, b: np.ndarray, subsets: int = 100, subset_size: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Mean and standard deviation over random subsets. Unlike FD it has no bias in N, so it is the one to read at a few thousand samples."""
    rng = np.random.default_rng(seed)
    m = min(subset_size, len(a), len(b))
    d = a.shape[1]
    vals = []
    for _ in range(subsets):
        x = np.asarray(a[rng.choice(len(a), m, replace=False)], np.float64)
        y = np.asarray(b[rng.choice(len(b), m, replace=False)], np.float64)
        kxx, kyy, kxy = ((u @ v.T / d + 1) ** 3 for u, v in ((x, x), (y, y), (x, y)))
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))

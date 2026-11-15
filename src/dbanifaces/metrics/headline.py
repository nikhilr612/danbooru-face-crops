"""One number to minimise: FD excess over the real-data floor, plus a penalty for copying the training set.

    headline = log10(1 + max(0, FD - floor(N))) + u / tau
    u        = max(0, memorised_frac - (Q + 3 SE)) / (1 - Q)        Q = 0.01 (the calibrated rate), SE = sqrt(Q (1 - Q) / N)

The floor is the FD of real held-out images at the same N (`floor(N) = a + b / N`, fitted on the test split, see scripts/make_headline.py). `tau` is solved so that a generator that
copies the training set (FD excess about 0, u about 1) scores the same as one that outputs noise (u = 0, FD excess of the noise): equally worthless. A perfect generator scores
about 0. The penalty is additive because copying leaves FD at its floor, so a product would stay 0.
"""
import math

import numpy as np

from dbanifaces.metrics.reference import QUANTILE

TOLERANCE_SE = 3.0


def floor_fd(n: int, a: float, b: float) -> float:
    return a + b / n


def copy_excess(memorised_frac: float, n: int) -> float:
    """Share of samples copied beyond the calibrated rate and its sampling noise, rescaled so that every sample copied is 1."""
    q = QUANTILE
    return max(0.0, memorised_frac - (q + TOLERANCE_SE * math.sqrt(q * (1 - q) / n))) / (1 - q)


def solve_tau(noise_fd: float, n: int, a: float, b: float) -> float:
    """tau for which full copying (u = 1) and noise score the same."""
    return copy_excess(1.0, n) / math.log10(1 + max(0.0, noise_fd - floor_fd(n, a, b)))


def headline(fd: float, memorised_frac: float, n: int, params: dict) -> float:
    """`params`: {"a", "b", "tau"} from the reference (`Reference.meta["headline"]`)."""
    excess = max(0.0, fd - floor_fd(n, params["a"], params["b"]))
    return math.log10(1 + excess) + copy_excess(memorised_frac, n) / params["tau"]


def fit_floor(ns, fds) -> tuple[float, float]:
    """Least squares a + b / N through the measured real-vs-real FD at each N."""
    b, a = np.polyfit(1 / np.asarray(ns, float), np.asarray(fds, float), 1)
    return float(a), float(b)

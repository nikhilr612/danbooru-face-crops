"""One number to minimise: FD excess over the real-data floor, plus a penalty for copying the training set.

    headline = log10(1 + max(0, FD - floor(N))) + u / tau
    u        = max(0, memorised_frac - (Q + 3 SE)) / (1 - Q)        Q = 0.01 (the calibrated rate), SE = sqrt(Q (1 - Q) / N)

The floor is the FD of real held-out images at the same N: measured at N = 1k, 2k, 5k, 10k and 20k for each split's reference (scripts/make_headline.py, on real images that are neither in the
reference nor from the test split) and interpolated linearly in log-log between them; outside that range the nearest segment is extended. `tau` is solved so that a generator that
copies the training set (FD excess about 0, u about 1) scores the same as one that outputs noise (u = 0, FD excess of the noise): equally worthless. A perfect generator scores
about 0. The penalty is additive because copying leaves FD at its floor, so a product would stay 0.
"""
import math

import numpy as np

from dbanifaces.metrics.reference import QUANTILE

TOLERANCE_SE = 3.0


def floor_fd(n: int, ns, fds) -> float:
    """Real-data FD at N = n: log-log interpolation through the measured (ns, fds); the end segments are extended."""
    x, y = np.log(np.asarray(ns, float)), np.log(np.asarray(fds, float))
    t = math.log(n)
    i = int(np.clip(np.searchsorted(x, t) - 1, 0, len(x) - 2))
    return float(math.exp(y[i] + (y[i + 1] - y[i]) * (t - x[i]) / (x[i + 1] - x[i])))


def copy_excess(memorised_frac: float, n: int) -> float:
    """Share of samples copied beyond the calibrated rate and its sampling noise, rescaled so that every sample copied is 1."""
    q = QUANTILE
    return max(0.0, memorised_frac - (q + TOLERANCE_SE * math.sqrt(q * (1 - q) / n))) / (1 - q)


def solve_tau(noise_fd: float, n: int, ns, fds) -> float:
    """tau for which full copying (u = 1) and noise score the same."""
    return copy_excess(1.0, n) / math.log10(1 + max(0.0, noise_fd - floor_fd(n, ns, fds)))


def headline(fd: float, memorised_frac: float, n: int, params: dict) -> float:
    """`params`: {"floor_n", "floor_fd", "tau"}, from `Reference.headline_params`."""
    excess = max(0.0, fd - floor_fd(n, params["floor_n"], params["floor_fd"]))
    return math.log10(1 + excess) + copy_excess(memorised_frac, n) / params["tau"]

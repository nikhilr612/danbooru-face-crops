"""Frechet distance on feature arrays, in float64."""
import numpy as np


class FD:
    """Reference Gaussian (mean, covariance) with its matrix square root cached: each evaluation is one symmetric eigensolve of root @ sigma @ root."""

    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu, self.sigma = np.asarray(mu, np.float64), np.asarray(sigma, np.float64)
        w, v = np.linalg.eigh(self.sigma)
        self.root = (v * np.sqrt(np.clip(w, 0, None))) @ v.T

    @classmethod
    def from_features(cls, feats: np.ndarray) -> "FD":
        f = np.asarray(feats, np.float64)
        return cls(f.mean(0), np.cov(f, rowvar=False))

    def __call__(self, feats: np.ndarray) -> float:
        f = np.asarray(feats, np.float64)
        mu, sigma = f.mean(0), np.cov(f, rowvar=False)
        tr = np.sqrt(np.clip(np.linalg.eigvalsh(self.root @ sigma @ self.root), 0, None)).sum()
        d = self.mu - mu
        return float(d @ d + np.trace(self.sigma) + np.trace(sigma) - 2 * tr)


def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    return FD.from_features(a)(b)

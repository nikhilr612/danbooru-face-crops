"""Memorisation fraction: share of generated samples closer to the training set than almost any real held-out image is.

FD, KID and density/coverage cannot see this (a copy of a training image is a real image), and this alone cannot see quality; report it next to them.
"""
import numpy as np


def _unit(x) -> np.ndarray:
    a = np.asarray(x, np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)


def max_cosine(query: np.ndarray, bank: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Cosine similarity of each query to its most similar bank row (numpy; the torch and JAX Evaluators swap in a GPU version)."""
    q, b = _unit(query), _unit(bank)
    return np.concatenate([(q[i:i + chunk] @ b.T).max(1) for i in range(0, len(q), chunk)])


def threshold(heldout: np.ndarray, bank: np.ndarray, quantile: float = 0.01, search=max_cosine) -> float:
    """Similarity above which a sample counts as a copy: the (1 - quantile) quantile of held-out images' best similarity to the bank."""
    return float(np.quantile(search(heldout, bank), 1 - quantile))


def memorised_fraction(gen: np.ndarray, bank: np.ndarray, thr: float, search=max_cosine) -> float:
    """A generator that generalises like real data scores about the quantile used for `thr` (0.01); copying scores far above."""
    return float((search(gen, bank) > thr).mean())

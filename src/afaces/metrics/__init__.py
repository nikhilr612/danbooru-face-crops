"""Metrics on DINOv2 CLS features (numpy; works with features from `afaces.torch` or `afaces.jax`).

Frechet distance (FD-DINOv2), KID, density and coverage, memorisation fraction; `Evaluator` ties them to a reference and a feature extractor.
"""
from afaces.metrics.evaluator import Evaluator
from afaces.metrics.fd import FD, frechet_distance
from afaces.metrics.kid import kid
from afaces.metrics.memorisation import max_cosine, memorised_fraction, threshold
from afaces.metrics.prdc import density_coverage, knn_radii
from afaces.metrics.reference import Reference

__all__ = ["FD", "Evaluator", "Reference", "density_coverage", "frechet_distance", "kid", "knn_radii", "max_cosine", "memorised_fraction", "threshold"]

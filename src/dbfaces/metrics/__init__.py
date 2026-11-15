"""Metrics on DINOv2 CLS features (numpy; works with features from `dbfaces.torch` or `dbfaces.jax`).

Frechet distance (FD-DINOv2), KID, density and coverage, memorisation fraction; `Evaluator` ties them to a reference and a feature extractor.
"""
from dbfaces.metrics.evaluator import Evaluator
from dbfaces.metrics.fd import FD, frechet_distance
from dbfaces.metrics.kid import kid
from dbfaces.metrics.memorisation import max_cosine, memorised_fraction, threshold
from dbfaces.metrics.prdc import density_coverage, knn_radii
from dbfaces.metrics.reference import Reference

__all__ = ["FD", "Evaluator", "Reference", "density_coverage", "frechet_distance", "kid", "knn_radii", "max_cosine", "memorised_fraction", "threshold"]

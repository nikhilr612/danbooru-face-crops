"""Metrics on DINOv2 CLS features (numpy; works with features from `dbanifaces.torch` or `dbanifaces.jax`).

Frechet distance (FD-DINOv2), KID, density and coverage, memorisation fraction; `Evaluator` ties them to a reference and a feature extractor.
"""
from dbanifaces.metrics.evaluator import Evaluator
from dbanifaces.metrics.fd import FD, frechet_distance
from dbanifaces.metrics.kid import kid
from dbanifaces.metrics.memorisation import max_cosine, memorised_fraction, threshold
from dbanifaces.metrics.prdc import density_coverage, knn_radii
from dbanifaces.metrics.reference import Reference

__all__ = ["FD", "Evaluator", "Reference", "density_coverage", "frechet_distance", "kid", "knn_radii", "max_cosine", "memorised_fraction", "threshold"]

"""dbfaces: loader and metrics for the Danbooru Face Crops 128 dataset.

The top level is backend-free (numpy only). `dbfaces.torch` and `dbfaces.jax` are the optional parts and import their framework only when
you import them; `dbfaces.metrics` is numpy and works with features from either backend.
"""
from dbfaces.data import SPLITS, Split, load, reliable_tags, resolve, tag_info, unpack_flags

__all__ = ["SPLITS", "Split", "load", "reliable_tags", "resolve", "tag_info", "unpack_flags"]
__version__ = "0.2.0"

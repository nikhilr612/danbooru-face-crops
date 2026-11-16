"""dbanifaces: loader and metrics for the Danbooru Face Crops 128 dataset.

The top level is backend-free (numpy only). `dbanifaces.torch` and `dbanifaces.jax` are the optional parts and import their framework only when
you import them; `dbanifaces.metrics` is numpy and works with features from either backend.
"""
from dbanifaces.data import DEFAULT_REPO, DEFAULT_REVISION, SPLITS, Split, default_revision, default_root, load, reliable_tags, resolve, tag_info, unpack_flags

__all__ = ["DEFAULT_REPO", "DEFAULT_REVISION", "SPLITS", "Split", "default_revision", "default_root", "load", "reliable_tags", "resolve", "tag_info", "unpack_flags"]
__version__ = "0.2.0"

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # JAX would otherwise reserve 75% of the GPU for itself
import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def root():
    """The npy release: a local copy or a Hub repo id (set DBANIFACES_ROOT); tests that need the data are skipped without it."""
    p = os.environ.get("DBANIFACES_ROOT")
    if not p or not (Path(p).is_dir() or p.count("/") == 1):
        pytest.skip("DBANIFACES_ROOT not set")
    return p

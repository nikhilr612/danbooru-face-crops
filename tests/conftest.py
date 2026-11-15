import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # JAX would otherwise reserve 75% of the GPU for itself
import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def root():
    """A local copy of the npy release (set DBFACES_ROOT); tests that need the data are skipped without it."""
    p = os.environ.get("DBFACES_ROOT")
    if not p or not Path(p).is_dir():
        pytest.skip("DBFACES_ROOT not set")
    return p

"""Evaluator with the JAX DINOv2 extractor and a GPU memorisation search."""
import jax
import jax.numpy as jnp
import numpy as np

from afaces.jax.dino import DinoV2
from afaces.metrics import Evaluator as _Evaluator
from afaces.metrics import Reference


@jax.jit
def _best(q, b):
    return jnp.matmul(q, b.T, precision="highest").max(1)


def _unit(x):
    a = jnp.asarray(np.asarray(x, np.float32))
    return a / jnp.maximum(jnp.linalg.norm(a, axis=1, keepdims=True), 1e-8)


def max_cosine(query, bank, chunk: int = 4096) -> np.ndarray:
    unit = _unit
    b = unit(bank)  # float32 throughout: float16 similarities near 1 are spaced 5e-4 apart, which would make the memorisation threshold a coarse grid
    return np.concatenate([np.asarray(_best(unit(query[i:i + chunk]), b), np.float32) for i in range(0, len(query), chunk)])


class Evaluator(_Evaluator):
    """`Evaluator(Reference.load())`."""

    def __init__(self, reference: Reference | None = None, size: int = 128, batch: int = 128):
        super().__init__(DinoV2(size, batch).features, reference or Reference.load(), max_cosine)

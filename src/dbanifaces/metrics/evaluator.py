"""Streaming evaluator for a training loop: feed generated batches, read the scores."""
import numpy as np

from dbanifaces.metrics.kid import kid
from dbanifaces.metrics.memorisation import max_cosine, memorised_fraction
from dbanifaces.metrics.prdc import density_coverage
from dbanifaces.metrics.reference import Reference
from dbanifaces.weights import to_uint8

ALL = ("fd", "kid", "density_coverage", "memorisation")


class Evaluator:
    """`dbanifaces.torch.Evaluator` and `dbanifaces.jax.Evaluator` bind the DINOv2 extractor and GPU search for you; this class takes any `featurize(uint8 NHWC) -> (B, 1024)`.

        ev = Evaluator(featurize, Reference.load())
        for batch in samples: ev.update(batch)   # uint8 NHWC, or float [-1, 1]
        ev.compute()   # {"fd_dinov2": ..., "kid_dinov2": ..., "kid_dinov2_std": ..., "density": ..., "coverage": ..., "memorised_frac": ..., "n": ...}

    Use the same number of samples every time you compare runs: FD is biased upwards at small N (KID is not). Memorisation needs the 300 MB feature bank (downloaded on first use).
    """

    def __init__(self, featurize, reference: Reference, search=max_cosine):
        self.featurize, self.ref, self.search = featurize, reference, search
        self.feats: list[np.ndarray] = []

    def reset(self) -> None:
        self.feats = []

    def update(self, images) -> None:
        self.update_features(self.featurize(to_uint8(images)))

    def update_features(self, feats: np.ndarray) -> None:
        self.feats.append(np.asarray(feats, np.float32))

    def compute(self, metrics=ALL) -> dict:
        f = np.concatenate(self.feats)
        out = {"n": len(f)}
        if "fd" in metrics:
            out["fd_dinov2"] = self.ref.fd(f)
        if "kid" in metrics:
            out["kid_dinov2"], out["kid_dinov2_std"] = kid(self.ref.real, f)
        if "density_coverage" in metrics:
            out |= density_coverage(self.ref.real, self.ref.radii, f, self.ref.meta["k"])
        if "memorisation" in metrics:
            out["memorised_frac"] = memorised_fraction(f, self.ref.bank, self.ref.mem_thr, self.search)
        return out

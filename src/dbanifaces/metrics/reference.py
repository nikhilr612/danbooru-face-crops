"""Reference statistics of the dataset in DINOv2-L CLS space, computed once from the train split (see scripts/make_reference.py).

`reference/dinov2l_stats.npz`: Gaussian (mu, sigma) of all train features, a fixed `real` subset (the first 10,000 train rows: a stratified sample) with its k-NN radii
for density/coverage and KID, and the memorisation threshold. `reference/dinov2l_{train,val,test}.npy`: the CLS features of every row of each split as float16, row for row the
same as the split's arrays (train is about 300 MB and is the memorisation bank; each of val and test about 60 MB), fetched only when asked for.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dbanifaces.metrics.fd import FD
from dbanifaces.metrics.prdc import knn_radii

STATS, FEATURES = "reference/dinov2l_stats.npz", "reference/dinov2l_{}.npy"
REAL_N, K, QUANTILE = 10_000, 5, 0.01


@dataclass
class Reference:
    fd: FD
    real: np.ndarray
    radii: np.ndarray
    mem_thr: float
    meta: dict
    root: Path | str = field(default="", repr=False)
    _bank: np.ndarray | None = field(default=None, repr=False)

    @classmethod
    def build(cls, train: np.ndarray, heldout: np.ndarray, meta: dict | None = None, search=None) -> "Reference":
        """`train`, `heldout` (validation): (N, 1024) CLS features. `search` is a (query, bank) -> max cosine function (default numpy; pass a GPU one for 150k rows)."""
        from dbanifaces.metrics.memorisation import max_cosine, threshold

        real = np.asarray(train[:REAL_N], np.float32)
        return cls(FD.from_features(train), real, knn_radii(real, K), threshold(heldout, train, QUANTILE, search or max_cosine),
                   {"k": K, "quantile": QUANTILE, "real_n": REAL_N, "n_train": len(train), "n_heldout": len(heldout), **(meta or {})})

    def save(self, directory: str | Path, features: dict[str, np.ndarray]) -> None:
        """`features`: split name -> (N, 1024) CLS features of that split, e.g. {"train": ..., "val": ..., "test": ...}."""
        d = Path(directory)
        (d / "reference").mkdir(parents=True, exist_ok=True)
        np.savez(d / STATS, mu=self.fd.mu, sigma=self.fd.sigma, real=self.real, radii=self.radii, mem_thr=self.mem_thr, meta=json.dumps(self.meta))
        for split, f in features.items():
            np.save(d / FEATURES.format(split), np.asarray(f, np.float16))

    @classmethod
    def load(cls, root: str | Path | None = None) -> "Reference":
        """`root`: a local release directory or the Hub repo id of the npy release (downloads the small stats file only); default as in `dbanifaces.default_root`."""
        from dbanifaces.data import default_root

        root = default_root(root)
        d = _fetch(root, [STATS])
        z = np.load(d / STATS)
        return cls(FD(z["mu"], z["sigma"]), z["real"], z["radii"], float(z["mem_thr"]), json.loads(str(z["meta"])), root)

    def features(self, split: str) -> np.ndarray:
        """float16 CLS features of every row of `split`, memory-mapped (downloaded on first use)."""
        return np.load(_fetch(self.root, [FEATURES.format(split)]) / FEATURES.format(split), mmap_mode="r")

    @property
    def bank(self) -> np.ndarray:
        """The train features, the memorisation bank."""
        if self._bank is None:
            self._bank = self.features("train")
        return self._bank


def _fetch(root: str | Path, patterns: list[str]) -> Path:
    if Path(root).is_dir():
        return Path(root)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(root), repo_type="dataset", allow_patterns=patterns))

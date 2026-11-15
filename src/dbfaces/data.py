"""Loader for the Danbooru Face Crops 128 dataset (memmapped .npy files, one per field, same row order in every file of a split).

    import dbfaces
    ds = dbfaces.load("path/to/danbooru-face-crops-128", "train", fields=("image", "tags"))
    ds = dbfaces.load("nikhilr01/danbooru-face-crops-128-npy", "train", fields=("image", "tags"))   # downloads only those files, then memory-maps them
    ds[0]                    # {"image": (128, 128, 3) uint8, "tags": (65,) int8}
    ds.batch(np.arange(64))  # fancy index (sorted internally), dict of arrays
    ds.subset(10_000)        # the first 10k rows: a nested stratified quick-eval subset
    ds.torch()               # torch.utils.data.Dataset      (extra `torch`: dbfaces.torch)
    ds.grain()               # random-access source for Grain (extra `jax`:   dbfaces.jax)
"""
import json
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")
IMAGE_FIELDS = ("image", "hull", "parse", "mouth_fill")


class Split:
    def __init__(self, root: Path, split: str, fields, length: int | None = None):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.root, self.split, self.fields = Path(root), split, tuple(fields)
        d = self.root / split
        self.arrays = {f: np.load(d / f"{f}.npy", mmap_mode="r") for f in self.fields}
        n = {len(a) for a in self.arrays.values()}
        if len(n) != 1:
            raise ValueError(f"fields have different lengths: {n}")
        self.n = min(n.pop(), length) if length else n.pop()

    def __getstate__(self):
        return {"root": self.root, "split": self.split, "fields": self.fields, "length": self.n}

    def __setstate__(self, st):
        self.__init__(st["root"], st["split"], st["fields"], st["length"])

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i):
        if isinstance(i, (int, np.integer)):
            if not -self.n <= i < self.n:
                raise IndexError(i)
            return {f: np.asarray(a[i % self.n]) for f, a in self.arrays.items()}
        return self.batch(np.arange(self.n)[i])

    def batch(self, idx) -> dict:
        idx = np.asarray(idx)
        order = np.argsort(idx, kind="stable")
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        return {f: np.asarray(a[idx[order]])[inv] for f, a in self.arrays.items()}

    def subset(self, k: int) -> "Split":
        """First k rows. Rows within a split are ordered so that every prefix is a stratified sample (rating, hair and eye colour class)."""
        return Split(self.root, self.split, self.fields, min(k, self.n))

    def torch(self, **kw):
        """torch Dataset over this split (needs the `torch` extra)."""
        from dbfaces.torch import Dataset

        return Dataset(self, **kw)

    def grain(self, **kw):
        """Grain random-access source over this split (needs the `jax` extra)."""
        from dbfaces.jax import Source

        return Source(self, **kw)


def resolve(root, split: str | None = None, fields=()) -> Path:
    """A local directory is used as is. Otherwise `root` is a Hugging Face dataset repo id ("user/name"): the files needed for `split` and `fields` (plus tags.json) are downloaded
    into the Hub cache and the snapshot directory is returned. Needs `huggingface_hub`; the arrays are memory-mapped from the cache."""
    if Path(root).is_dir():
        return Path(root)
    if str(root).count("/") != 1 or str(root).startswith((".", "/")):
        raise FileNotFoundError(f"{root} is neither a directory nor a Hugging Face repo id")
    from huggingface_hub import snapshot_download
    patterns = ["tags.json"] + ([f"{split}/{f}.npy" for f in fields] if split else [])
    return Path(snapshot_download(str(root), repo_type="dataset", allow_patterns=patterns))


def load(root, split: str = "train", fields=("image",)) -> Split:
    """`root` is a local directory or a Hub repo id such as "nikhilr01/danbooru-face-crops-128-npy"."""
    return Split(resolve(root, split, fields), split, fields)


def tag_info(root) -> dict:
    """tags.json: names (column order = reliability order), group, exclusive, colour, auc, precision, recall, tier, positive_only."""
    return json.load(open(resolve(root) / "tags.json"))


def reliable_tags(root, tiers=("high",)) -> list[int]:
    """Column indices of the tags in the given reliability tiers (high: AUC >= 0.97, medium: >= 0.93, low: below), measured against Danbooru's own tags."""
    t = tag_info(root)
    return [i for i, x in enumerate(t["tier"]) if x in tiers]


def numeric(row: dict) -> dict:
    """The row for a framework that has no bytes dtype: `rating` (b"g" / b"s") becomes uint8 0 (g) / 1 (s). Everything else is unchanged."""
    if "rating" in row:
        row = row | {"rating": (np.asarray(row["rating"]) == b"s").astype(np.uint8)}
    return row


def unpack_flags(flags: np.ndarray) -> dict:
    """flags bits: 0 profile proxy (tagger from_side >= 0.5), 1 mouth parse missing (landmark hull used), 2 frame shifted (window clamped at the image edge)."""
    return {"profile": flags & 1 > 0, "mouth_parse_missing": flags >> 1 & 1 > 0, "frame_shifted": flags >> 2 & 1 > 0}

"""Loader for the anime-faces 128 px dataset (memmapped .npy files, one per field, same row order in every file of a split).

    import afaces
    ds = afaces.load("path/to/anime-faces-128", "train", fields=("image", "tags"))
    ds = afaces.load("nikhilr01/danbooru-anime-faces-128-npy", "train", fields=("image", "tags"))   # downloads only those files, then memory-maps them
    ds[0]                    # {"image": (128, 128, 3) uint8, "tags": (65,) int8}
    ds.batch(np.arange(64))  # fancy index (sorted internally), dict of arrays
    ds.subset(10_000)        # the first 10k rows: a nested stratified quick-eval subset
    ds.torch()               # torch.utils.data.Dataset
    ds.grain()               # random-access data source for Grain (anything with __len__ and __getitem__ works)
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

    def torch(self):
        import torch

        outer = self

        class _DS(torch.utils.data.Dataset):
            def __len__(self):
                return len(outer)

            def __getitem__(self, i):
                return {k: torch.from_numpy(v.copy()) for k, v in outer[i].items()}

        return _DS()

    def grain(self):
        return self


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
    """`root` is a local directory or a Hub repo id such as "nikhilr01/danbooru-anime-faces-128-npy"."""
    return Split(resolve(root, split, fields), split, fields)


def tag_info(root) -> dict:
    """tags.json: names (column order = reliability order), group, exclusive, colour, auc, precision, recall, tier, positive_only."""
    return json.load(open(resolve(root) / "tags.json"))


def reliable_tags(root, tiers=("high",)) -> list[int]:
    """Column indices of the tags in the given reliability tiers (high: AUC >= 0.97, medium: >= 0.93, low: below), measured against Danbooru's own tags."""
    t = tag_info(root)
    return [i for i, x in enumerate(t["tier"]) if x in tiers]


def unpack_flags(flags: np.ndarray) -> dict:
    """flags bits: 0 profile proxy (tagger from_side >= 0.5), 1 mouth parse missing (landmark hull used), 2 frame shifted (window clamped at the image edge)."""
    return {"profile": flags & 1 > 0, "mouth_parse_missing": flags >> 1 & 1 > 0, "frame_shifted": flags >> 2 & 1 > 0}

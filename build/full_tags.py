"""Rewrite tags.npy and tags.json of an assembled release directory from the cached tagger probabilities (reliability-ordered columns, see build/tagspec.py).
Also sets flags bit 0 (profile proxy): the tagger's from_side probability >= 0.5.
Usage: .envs/metrics/bin/python -I build/full_tags.py RELEASE_DIR"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "build")
import tagspec  # noqa: E402

D = Path("build_data")
out = Path(sys.argv[1])
ts = tagspec.spec()
voc = json.load(open(D / "feats/wd_general.json"))
cols = [voc.index(t) for t in ts["names"]]
sel_all = pd.read_parquet(D / "selection.parquet")
for split in ("train", "val", "test"):
    sel = sel_all[sel_all.split == split].sort_values("idx").reset_index(drop=True)
    assert (np.load(out / split / "post_id.npy") == sel.pid.to_numpy()).all(), "release rows differ from the selection"
    T = np.zeros((len(sel), len(cols)), np.int8)
    prof = np.zeros(len(sel), bool)
    fs = ts["names"].index("from_side")
    for tag, g in sel.groupby("tag"):
        rows = np.load(D / f"feats/{tag}_rows.npy")
        W = np.load(D / f"feats/{tag}_wd.npy", mmap_mode="r")[np.searchsorted(rows, g.row.to_numpy())][:, cols].astype(np.float32)
        T[g.index.to_numpy()] = tagspec.values(W, ts)
        prof[g.index.to_numpy()] = W[:, fs] >= tagspec.PRESENT
    np.save(out / split / "tags.npy", T)
    fl = np.load(out / split / "flags.npy")
    np.save(out / split / "flags.npy", ((fl & ~np.uint8(1)) | prof.astype(np.uint8)).astype(np.uint8))
    print(split, "profile flag share", float(prof.mean()))
    print(split, T.shape, "present", float((T == 1).mean()), "unknown", float((T == -1).mean()), flush=True)
json.dump(ts, open(out / "tags.json", "w"), indent=1)

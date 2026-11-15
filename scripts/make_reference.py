"""Compute the reference statistics (dbanifaces.metrics.Reference) from a local release directory.

    python scripts/make_reference.py RELEASE_DIR [--backend jax|torch] [--features-cache DIR]

Writes RELEASE_DIR/reference/dinov2l_stats.npz and dinov2l_{train,val,test}.npy. Features of every split are cached as float16 in --features-cache when given.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import dbanifaces
from dbanifaces import weights as W
from dbanifaces.metrics import Reference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("release")
    ap.add_argument("--backend", choices=("jax", "torch"), default="jax")
    ap.add_argument("--features-cache")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if a.backend == "jax":
        from dbanifaces.jax import DinoV2
        from dbanifaces.jax.metrics import max_cosine
    else:
        from dbanifaces.torch import DinoV2
        from dbanifaces.torch.metrics import max_cosine
    net = DinoV2()
    feats = {}
    for split in ("train", "val", "test"):
        ds = dbanifaces.load(a.release, split, fields=("image",))
        n = min(len(ds), a.limit) if a.limit else len(ds)
        cache = Path(a.features_cache) / f"{split}.npy" if a.features_cache else None
        if cache and cache.exists() and len(np.load(cache, mmap_mode="r")) == n:
            feats[split] = np.load(cache).astype(np.float32)
            continue
        out, t = [], time.time()
        for i in range(0, n, 2048):
            out.append(net.features(ds.batch(np.arange(i, min(i + 2048, n)))["image"]))
            if (i // 2048) % 10 == 0:
                print(split, i, n, f"{time.time() - t:.0f}s", flush=True)
        feats[split] = np.concatenate(out)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, feats[split].astype(np.float16))
    meta = {"model": W.REPO, "revision": W.REVISION, "sha256": W.SHA256, "backend": a.backend, "package": f"dbanifaces {dbanifaces.__version__}",
            "dataset": json.loads((Path(a.release) / "manifest.json").read_text()).get("version", "unknown")}
    ref = Reference.build(feats["train"], feats["val"], meta, search=max_cosine)
    ref.save(a.release, feats)
    print("mem_thr", ref.mem_thr, "meta", ref.meta)


if __name__ == "__main__":
    main()

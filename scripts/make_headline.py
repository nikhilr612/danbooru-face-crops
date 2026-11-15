"""Measure the headline parameters for each split's reference and write them into the stats file (`Reference.meta["headline"][split]`).

    python scripts/make_headline.py RELEASE_DIR [--backend jax|torch]

For each reference split (train, val, test) the real-data FD floor at each N is measured on real images that are not part of that reference and not from the test split unless the test split
is itself the reference: the validation rows for the train reference, the train rows for the val and test references (stratified prefixes of N = 1k, 2k, 5k, 10k, 20k rows). `tau` is solved so
that full copying of the training set and 10,000 Gaussian-noise images get the same headline (`dbanifaces.metrics.headline`). Rewrites RELEASE_DIR/reference/dinov2l_stats.npz in place.
"""
import argparse
import json

import numpy as np

from dbanifaces.metrics import Reference
from dbanifaces.metrics.headline import solve_tau

NS = (1000, 2000, 5000, 10000, 20000)
REF_N = 10_000
INDEPENDENT = {"train": "val", "val": "train", "test": "train"}


def noise(n, rng):
    return np.clip(rng.normal(128, 64, (n, 128, 128, 3)), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("release")
    ap.add_argument("--backend", choices=("jax", "torch"), default="jax")
    a = ap.parse_args()
    if a.backend == "jax":
        from dbanifaces.jax import Evaluator
    else:
        from dbanifaces.torch import Evaluator
    rng = np.random.default_rng(0)
    ev = Evaluator(Reference.load(a.release))
    noise_feats = np.concatenate([ev.featurize(noise(1000, rng)) for _ in range(REF_N // 1000)]).astype(np.float32)
    out = {}
    for split, other in INDEPENDENT.items():
        ref = Reference.load(a.release, split=split)
        real = np.asarray(ref.features(other), np.float32)
        fds = [ref.fd(real[:n]) for n in NS]
        noise_fd = ref.fd(noise_feats)
        out[split] = {"tau": solve_tau(noise_fd, REF_N, NS, fds), "floor_n": list(NS), "floor_fd": fds, "floor_real": other, "noise": "gaussian", "noise_fd_10000": noise_fd}
        print(split, json.dumps(out[split]), flush=True)
    z = dict(np.load(f"{a.release}/reference/dinov2l_stats.npz"))
    z["meta"] = json.dumps(json.loads(str(z["meta"])) | {"headline": out})
    np.savez(f"{a.release}/reference/dinov2l_stats.npz", **z)


if __name__ == "__main__":
    main()

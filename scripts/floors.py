"""Reference numbers for reading scores: real-vs-real floors at several N, and sanity responses to known problems, on the TEST split (not used to build the reference).

    python scripts/floors.py RELEASE_DIR [--backend jax|torch] > floors.json
"""
import argparse
import json

import numpy as np

import dbanifaces
from dbanifaces.metrics import Reference
from dbanifaces.weights import to_uint8


def blur(x, s):
    from scipy.ndimage import gaussian_filter

    return np.stack([gaussian_filter(i.astype(np.float32), (s, s, 0)) for i in x]).round().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("release")
    ap.add_argument("--backend", choices=("jax", "torch"), default="jax")
    ap.add_argument("--n", type=int, nargs="+", default=[1000, 2000, 5000, 10000])
    a = ap.parse_args()
    if a.backend == "jax":
        from dbanifaces.jax import Evaluator
    else:
        from dbanifaces.torch import Evaluator
    ref = Reference.load(a.release)
    ev = Evaluator(ref)
    test, train = dbanifaces.load("test", fields=("image",), root=a.release), dbanifaces.load("train", fields=("image",), root=a.release)
    out = {"real_vs_real_test": {}, "problems_N2000": {}}
    for n in a.n:
        ev.reset()
        ev.update(test[:n]["image"])  # the first n rows of a split are a stratified sample
        out["real_vs_real_test"][n] = ev.compute()
        print(n, out["real_vs_real_test"][n], flush=True)
    x = test[:2000]["image"]
    probs = {"blur_0.5": blur(x, 0.5), "blur_1": blur(x, 1.0), "blur_2": blur(x, 2.0),
             "grey_20pct": np.concatenate([x[:1600], np.repeat(x[1600:].mean(-1, keepdims=True), 3, -1).astype(np.uint8)]),
             "flip_vertical": x[:, ::-1],
             "half_dropped_dark": x[x.reshape(len(x), -1).mean(1) > np.median(x.reshape(len(x), -1).mean(1))],
             "train_copies": train[:2000]["image"],
             "20pct_train_copies": np.concatenate([x[:1600], train[:400]["image"]])}
    for k, v in probs.items():
        ev.reset()
        ev.update(to_uint8(np.ascontiguousarray(v)))
        out["problems_N2000"][k] = ev.compute()
        print(k, out["problems_N2000"][k], flush=True)
    print(json.dumps(out))


if __name__ == "__main__":
    main()

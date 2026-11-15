"""Fit the headline parameters and write them into the reference (`Reference.meta["headline"]`): the real-data FD floor a + b / N and tau.

    python scripts/make_headline.py RELEASE_DIR [--backend jax|torch] [--noise gaussian|uniform|pink]

The floor is fitted on the test split's first N rows (a stratified sample) against the train statistics; tau is solved so that full copying of the training set and noise of the
given type, both at N = 10,000, get the same headline (`dbanifaces.metrics.headline`). Rewrites RELEASE_DIR/reference/dinov2l_stats.npz in place.
"""
import argparse
import json

import numpy as np

from dbanifaces.metrics import Reference
from dbanifaces.metrics.headline import fit_floor, solve_tau

NS = (1000, 2000, 5000, 10000)
REF_N = 10_000


def noise(kind, n, rng):
    if kind == "gaussian":
        return np.clip(rng.normal(128, 64, (n, 128, 128, 3)), 0, 255).astype(np.uint8)
    if kind == "uniform":
        return rng.integers(0, 256, (n, 128, 128, 3), dtype=np.uint8)
    f = np.fft.fftfreq(128)
    r = np.sqrt(f[:, None] ** 2 + f[None] ** 2)
    r[0, 0] = 1
    x = np.fft.ifft2(np.fft.fft2(rng.normal(size=(n, 128, 128, 3)), axes=(1, 2)) / r[None, :, :, None], axes=(1, 2)).real
    x = (x - x.mean((1, 2, 3), keepdims=True)) / x.std((1, 2, 3), keepdims=True)
    return np.clip(128 + 50 * x, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("release")
    ap.add_argument("--backend", choices=("jax", "torch"), default="jax")
    ap.add_argument("--noise", choices=("gaussian", "uniform", "pink"), default="gaussian")
    a = ap.parse_args()
    if a.backend == "jax":
        from dbanifaces.jax import Evaluator
    else:
        from dbanifaces.torch import Evaluator
    ref = Reference.load(a.release)
    ev = Evaluator(ref)
    test = np.asarray(ref.features("test"), np.float32)
    fds = [ref.fd(test[:n]) for n in NS]
    fa, fb = fit_floor(NS, fds)
    rng = np.random.default_rng(0)
    for i in range(0, REF_N, 1000):
        ev.update(noise(a.noise, 1000, rng))
    noise_fd = ev.compute(("fd",))["fd_dinov2"]
    tau = solve_tau(noise_fd, REF_N, fa, fb)
    params = {"a": fa, "b": fb, "tau": tau, "floor_n": list(NS), "floor_fd": fds, "noise": a.noise, "noise_fd_10000": noise_fd}
    print(json.dumps(params, indent=1))
    z = dict(np.load(f"{a.release}/reference/dinov2l_stats.npz"))
    z["meta"] = json.dumps(json.loads(str(z["meta"])) | {"headline": params})
    np.savez(f"{a.release}/reference/dinov2l_stats.npz", **z)


if __name__ == "__main__":
    main()

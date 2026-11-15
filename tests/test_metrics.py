import numpy as np
import scipy.linalg

from dbanifaces.metrics import FD, Reference, density_coverage, frechet_distance, kid, knn_radii, memorised_fraction, threshold


def gauss(n, d=32, shift=0.0, scale=1.0, seed=0):
    return np.random.default_rng(seed).normal(shift, scale, (n, d)).astype(np.float32)


def test_fd_matches_sqrtm_formula():
    a, b = gauss(3000, seed=1), gauss(3000, shift=0.3, scale=1.2, seed=2)
    ma, mb, sa, sb = a.mean(0), b.mean(0), np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    ref = float(((ma - mb) ** 2).sum() + np.trace(sa + sb - 2 * scipy.linalg.sqrtm(sa @ sb).real))
    assert abs(frechet_distance(a, b) - ref) < 1e-6 * ref
    assert abs(frechet_distance(a, a)) < 1e-6
    assert abs(frechet_distance(a, b) - frechet_distance(b, a)) < 1e-6 * ref


def test_fd_reference_object_and_ordering():
    ref = FD.from_features(gauss(5000, seed=3))
    same, shifted, wide = ref(gauss(5000, seed=4)), ref(gauss(5000, shift=0.5, seed=5)), ref(gauss(5000, scale=2, seed=6))
    assert same < 0.5 < shifted and same < wide


def test_kid_zero_for_same_distribution_and_positive_for_shift():
    m, _ = kid(gauss(2000, seed=7), gauss(2000, seed=8), subsets=20)
    s, _ = kid(gauss(2000, seed=7), gauss(2000, shift=0.5, seed=9), subsets=20)
    assert abs(m) < 0.01 < s


def test_density_coverage():
    real = gauss(2000, seed=10)
    r = knn_radii(real, 5)
    same = density_coverage(real, r, gauss(2000, seed=11), 5)
    assert 0.8 < same["density"] < 1.2 and same["coverage"] > 0.9
    far = density_coverage(real, r, gauss(2000, shift=2.0, seed=12), 5)  # a different region: no density, and no real point is covered
    assert far["density"] < 0.1 and far["coverage"] < same["coverage"]
    collapsed = density_coverage(real, r, np.repeat(real[:1], 2000, 0), 5)
    assert collapsed["coverage"] < 0.1


def test_memorisation_flags_copies_only():
    bank, held = gauss(3000, 64, seed=13), gauss(1000, 64, seed=14)
    thr = threshold(held, bank, 0.01)
    fresh = gauss(1000, 64, seed=15)
    assert memorised_fraction(fresh, bank, thr) < 0.05
    mixed = np.concatenate([fresh[:500], bank[:500] + 0.01 * gauss(500, 64, seed=16)])
    assert 0.4 < memorised_fraction(mixed, bank, thr) < 0.6


def test_reference_roundtrip(tmp_path):
    train, held = gauss(3000, 16, seed=17), gauss(1000, 16, seed=18)
    ref = Reference.build(train, held, {"x": 1})
    ref.save(tmp_path, {"train": train, "val": held})
    back = Reference.load(tmp_path)
    assert back.meta["x"] == 1 and np.allclose(back.fd.mu, ref.fd.mu) and back.mem_thr == ref.mem_thr
    assert back.bank.shape == (3000, 16) and back.bank.dtype == np.float16 and back.features("val").shape == (1000, 16)


def test_evaluators_agree_across_backends(root):
    """The metrics are numpy on the features, so the two backends differ only through the features (5e-6 relative)."""
    import pytest

    pytest.importorskip("torch")
    pytest.importorskip("jax")
    import dbanifaces
    from dbanifaces.jax import Evaluator as J
    from dbanifaces.torch import Evaluator as T

    ref = Reference.load(root)
    x = dbanifaces.load(root, "test", fields=("image",))[:1000]["image"]
    out = []
    for E in (J, T):
        ev = E(ref)
        ev.update(x)
        out.append(ev.compute())
    for k in out[0]:
        assert abs(out[0][k] - out[1][k]) <= 1e-3 * max(abs(out[0][k]), 1e-2), (k, out[0][k], out[1][k])

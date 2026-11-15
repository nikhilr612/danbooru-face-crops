import numpy as np

import dbanifaces
from dbanifaces import palette as P


def test_extract_reproduces_the_shipped_field(root):
    r = dbanifaces.load("val", fields=("image", "parse", "palette", "palette_valid"), root=root)[:100]
    p, v = P.extract(r["image"], r["parse"])
    assert np.allclose(p.reshape(100, 32), r["palette"], atol=1e-5) and (v == r["palette_valid"]).all()


def test_own_palette_beats_other_palettes(root):
    r = dbanifaces.load("val", fields=("image", "parse", "palette", "palette_valid"), root=root)[:300]
    pal, val = r["palette"].reshape(300, 4, 8), r["palette_valid"]
    own = P.adherence(r["image"], r["parse"], pal, val, np.array([P.CHANCE[k] for k in P.REGIONS]))
    perm = np.random.default_rng(0).permutation(300)
    other = P.adherence(r["image"], r["parse"], pal[perm], val[perm], np.array([P.CHANCE[k] for k in P.REGIONS]))
    for k in P.REGIONS:
        assert own[k]["dE"] < 0.7 * other[k]["dE"], (k, own[k]["dE"], other[k]["dE"])
        assert own[k]["dE"] < P.CHANCE[k]


def test_oklab_roundtrip():
    u8 = np.random.default_rng(0).integers(0, 256, (50, 3), dtype=np.uint8)
    assert np.abs(P.oklab_to_srgb(P.srgb_to_oklab(u8)).astype(int) - u8).max() <= 1

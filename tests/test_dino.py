import numpy as np
import pytest

from dbfaces import weights as W

ROOT_FIELDS = ("image",)


def test_bicubic_matches_torch():
    torch = pytest.importorskip("torch")
    for n_in, n_out in ((128, 224), (37, 16)):
        x = torch.randn(1, 1, n_in, n_in)
        ref = torch.nn.functional.interpolate(x, size=(n_out, n_out), mode="bicubic", align_corners=False)[0, 0].numpy()
        m = W.bicubic_matrix(n_in, n_out)
        np.testing.assert_allclose(m @ x[0, 0].numpy() @ m.T, ref, atol=1e-4)
        np.testing.assert_allclose(m.sum(1), 1, atol=1e-6)


@pytest.fixture(scope="module")
def images(root):
    import dbfaces

    return dbfaces.load(root, "val", fields=ROOT_FIELDS)[:48]["image"]


def rel(a, b):
    return np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)


def test_matches_transformers_float64(images):
    """Both backends against the HF model run in float64 (HF float32 on a GPU is itself about 4e-3 off that reference; ours are about 1e-4)."""
    torch = pytest.importorskip("torch")
    tr = pytest.importorskip("transformers")
    pytest.importorskip("jax")
    from dbfaces.jax.dino import DinoV2 as J
    from dbfaces.torch.dino import DinoV2 as T

    ref_model = tr.Dinov2Model.from_pretrained(W.REPO, revision=W.REVISION, dtype=torch.float64).eval()
    x = torch.from_numpy(images).permute(0, 3, 1, 2).double() / 255
    x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1)
    x = (x - torch.tensor(W.MEAN, dtype=torch.float64).view(1, 3, 1, 1)) / torch.tensor(W.STD, dtype=torch.float64).view(1, 3, 1, 1)
    with torch.no_grad():
        ref = ref_model(pixel_values=x).pooler_output.numpy()
    for name, feats in (("torch", T().features(images)), ("jax", J().features(images))):
        assert feats.shape == (48, 1024)
        assert rel(feats, ref).max() < 5e-4, (name, rel(feats, ref).max())


def test_jax_matches_torch(images):
    pytest.importorskip("jax")
    pytest.importorskip("torch")
    from dbfaces.jax.dino import DinoV2 as J
    from dbfaces.torch.dino import DinoV2 as T

    a, b = J().features(images), T().features(images)
    assert rel(a, b).max() < 1e-4, rel(a, b).max()
    assert J(batch=32).features(images[:5]).shape == (5, 1024)

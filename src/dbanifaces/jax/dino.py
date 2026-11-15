"""DINOv2 ViT-L/14 CLS features in JAX (the architecture of facebook/dinov2-large, 224 px input): one `lax.scan` over the stacked layers, so it compiles quickly."""
import jax
import jax.numpy as jnp
import numpy as np

from dbanifaces import weights as W


def _ln(x, w, b):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) * jax.lax.rsqrt(var + W.EPS) * w + b


def _layer(x, l):
    B, N, D = x.shape
    H = W.HEADS
    q, k, v = jnp.moveaxis((_ln(x, l["n1w"], l["n1b"]) @ l["qkv_w"].T + l["qkv_b"]).reshape(B, N, 3, H, D // H), 2, 0)
    a = jax.nn.dot_product_attention(q, k, v).reshape(B, N, D)
    x = x + l["ls1"] * (a @ l["o_w"].T + l["o_b"])
    h = jax.nn.gelu(_ln(x, l["n2w"], l["n2b"]) @ l["fc1_w"].T + l["fc1_b"], approximate=False)
    return x + l["ls2"] * (h @ l["fc2_w"].T + l["fc2_b"]), None


LAYER_KEYS = ("n1w", "n1b", "n2w", "n2b", "qkv_w", "qkv_b", "o_w", "o_b", "ls1", "ls2", "fc1_w", "fc1_b", "fc2_w", "fc2_b")


def forward(p, m, u8):
    """p: params from `dbanifaces.weights.load_params` as jax arrays, m: resize matrix, u8: (B, H, W, 3) uint8 -> (B, 1024) CLS features."""
    x = u8.astype(jnp.float32) / 255
    x = jnp.einsum("oh,pw,bhwc->bopc", m, m, x, precision="highest").clip(0, 1)
    x = (x - W.MEAN) / W.STD
    B, g = x.shape[0], W.SIZE // W.PATCH
    x = x.reshape(B, g, W.PATCH, g, W.PATCH, 3).transpose(0, 1, 3, 5, 2, 4).reshape(B, g * g, 3 * W.PATCH * W.PATCH)
    x = x @ p["patch_w"].reshape(W.DIM, -1).T + p["patch_b"]
    x = jnp.concatenate([jnp.broadcast_to(p["cls"], (B, 1, W.DIM)), x], 1) + p["pos"]
    x, _ = jax.lax.scan(_layer, x, {k: p[k] for k in LAYER_KEYS})
    return _ln(x[:, 0], p["ln_w"], p["ln_b"])


class DinoV2:
    """`features(images)`: uint8 NHWC (numpy or jax array) -> (B, 1024) float32 numpy CLS features. Batches are padded to a fixed size so JIT compiles once."""

    def __init__(self, size: int = 128, batch: int = 128):
        self.p = {k: jnp.asarray(v) for k, v in W.load_params().items()}
        self.m, self.batch = jnp.asarray(W.bicubic_matrix(size, W.SIZE)), batch
        self._f = jax.jit(forward)

    def features(self, images) -> np.ndarray:
        u8, out = np.asarray(images), []
        with jax.default_matmul_precision("highest"):
            for i in range(0, len(u8), self.batch):
                c = u8[i:i + self.batch]
                pad = np.concatenate([c, np.zeros((self.batch - len(c), *c.shape[1:]), np.uint8)]) if len(c) < self.batch else c
                out.append(np.asarray(self._f(self.p, self.m, jnp.asarray(pad)))[:len(c)])
        return np.concatenate(out)

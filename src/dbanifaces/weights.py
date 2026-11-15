"""The one DINOv2 ViT-L/14 weight set both backends use, and the resize that feeds it. Needs numpy (and huggingface_hub + safetensors to read the weights).

Weights are the Hub copy of facebook/dinov2-large (Apache-2.0), pinned by revision and checked against a SHA-256, so torch and JAX score with the same numbers.
Resizing is a fixed separable bicubic matrix (the kernel torch uses: a = -0.75, half-pixel centres, edge-clamped taps), applied as a matrix product in either backend.
"""
import hashlib
import os
from pathlib import Path

import numpy as np

REPO, REVISION = "facebook/dinov2-large", "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"
SHA256 = "399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23"
DIM, DEPTH, HEADS, PATCH, GRID0, SIZE = 1024, 24, 16, 14, 37, 224  # width, layers, heads, patch, native position grid, input side
EPS = 1e-6
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def bicubic_matrix(n_in: int, n_out: int, a: float = -0.75) -> np.ndarray:
    """(n_out, n_in) float32 matrix of torch's bicubic interpolation (align_corners=False, no antialiasing), edge taps clamped."""
    def k(x):
        x = np.abs(x)
        return np.where(x <= 1, ((a + 2) * x - (a + 3)) * x * x + 1, np.where(x < 2, ((a * x - 5 * a) * x + 8 * a) * x - 4 * a, 0.0))

    scale = n_in / n_out
    src = (np.arange(n_out) + 0.5) * scale - 0.5
    base = np.floor(src).astype(int)
    m = np.zeros((n_out, n_in), np.float64)
    for t in range(-1, 3):
        np.add.at(m, (np.arange(n_out), np.clip(base + t, 0, n_in - 1)), k(src - (base + t)))
    return m.astype(np.float32)


def _path() -> Path:
    from huggingface_hub import hf_hub_download

    p = Path(os.environ.get("DBANIFACES_DINOV2") or hf_hub_download(REPO, "model.safetensors", revision=REVISION))
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    if h.hexdigest() != SHA256:
        raise ValueError(f"{p} is not the pinned DINOv2-L weights (sha256 {h.hexdigest()})")
    return p


def load_params() -> dict:
    """float32 numpy arrays: embeddings (position embedding already resized from 37x37 to 16x16 at 224 px) and per-layer weights stacked on axis 0, (out, in) layout."""
    from safetensors.numpy import load_file

    w = load_file(str(_path()))
    pos = w["embeddings.position_embeddings"][0]
    g = SIZE // PATCH
    m = bicubic_matrix(GRID0, g)
    grid = np.einsum("oh,pw,hwc->opc", m, m, pos[1:].reshape(GRID0, GRID0, DIM)).reshape(g * g, DIM)
    stack = lambda f: np.stack([w[f"encoder.layer.{i}.{f}"] for i in range(DEPTH)])
    a = "attention.attention."
    return {
        "cls": w["embeddings.cls_token"][0, 0], "pos": np.concatenate([pos[:1], grid]).astype(np.float32),
        "patch_w": w["embeddings.patch_embeddings.projection.weight"], "patch_b": w["embeddings.patch_embeddings.projection.bias"],
        "ln_w": w["layernorm.weight"], "ln_b": w["layernorm.bias"],
        "n1w": stack("norm1.weight"), "n1b": stack("norm1.bias"), "n2w": stack("norm2.weight"), "n2b": stack("norm2.bias"),
        "qkv_w": np.stack([np.concatenate([w[f"encoder.layer.{i}.{a}{n}.weight"] for n in ("query", "key", "value")]) for i in range(DEPTH)]),
        "qkv_b": np.stack([np.concatenate([w[f"encoder.layer.{i}.{a}{n}.bias"] for n in ("query", "key", "value")]) for i in range(DEPTH)]),
        "o_w": stack("attention.output.dense.weight"), "o_b": stack("attention.output.dense.bias"),
        "ls1": stack("layer_scale1.lambda1"), "ls2": stack("layer_scale2.lambda1"),
        "fc1_w": stack("mlp.fc1.weight"), "fc1_b": stack("mlp.fc1.bias"), "fc2_w": stack("mlp.fc2.weight"), "fc2_b": stack("mlp.fc2.bias"),
    }


def to_uint8(x) -> np.ndarray:
    """Float images in [-1, 1] are rounded to uint8 first (scores never depend on float precision); uint8 passes through. NHWC numpy."""
    x = np.asarray(x)
    return x if x.dtype == np.uint8 else (np.clip((x.astype(np.float32) + 1) * 0.5, 0, 1) * 255).round().astype(np.uint8)

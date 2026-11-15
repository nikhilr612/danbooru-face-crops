"""DINOv2 ViT-L/14 CLS features in PyTorch (the architecture of facebook/dinov2-large, 224 px input), written out so no `transformers` pin is needed."""
import numpy as np
import torch
import torch.nn.functional as F

from dbanifaces import weights as W


class DinoV2:
    """`features(images)`: uint8 NHWC array or tensor (any size the resize matrix is built for, 128 px by default) -> (B, 1024) float32 numpy CLS features."""

    def __init__(self, device=None, size: int = 128, batch: int = 128):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.p = {k: torch.from_numpy(v).to(self.device) for k, v in W.load_params().items()}
        self.batch = batch
        self.m = torch.from_numpy(W.bicubic_matrix(size, W.SIZE)).to(self.device)
        self.mean, self.std = (torch.from_numpy(a).to(self.device).view(1, 3, 1, 1) for a in (W.MEAN, W.STD))

    @torch.no_grad()
    def _forward(self, u8: torch.Tensor) -> torch.Tensor:
        p, D, H = self.p, W.DIM, W.HEADS
        x = u8.permute(0, 3, 1, 2).float() / 255
        x = torch.einsum("oh,pw,bchw->bcop", self.m, self.m, x).clamp(0, 1)
        x = F.conv2d((x - self.mean) / self.std, p["patch_w"], p["patch_b"], stride=W.PATCH).flatten(2).transpose(1, 2)
        x = torch.cat([p["cls"].expand(len(x), 1, D), x], 1) + p["pos"]
        B, N, _ = x.shape
        for i in range(W.DEPTH):
            h = F.layer_norm(x, (D,), p["n1w"][i], p["n1b"][i], W.EPS)
            q, k, v = F.linear(h, p["qkv_w"][i], p["qkv_b"][i]).view(B, N, 3, H, D // H).permute(2, 0, 3, 1, 4)
            a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D)
            x = x + p["ls1"][i] * F.linear(a, p["o_w"][i], p["o_b"][i])
            h = F.layer_norm(x, (D,), p["n2w"][i], p["n2b"][i], W.EPS)
            x = x + p["ls2"][i] * F.linear(F.gelu(F.linear(h, p["fc1_w"][i], p["fc1_b"][i])), p["fc2_w"][i], p["fc2_b"][i])
        return F.layer_norm(x[:, 0], (D,), p["ln_w"], p["ln_b"], W.EPS)

    def features(self, images) -> np.ndarray:
        u8 = torch.as_tensor(np.asarray(images)) if not isinstance(images, torch.Tensor) else images
        out = [self._forward(u8[i:i + self.batch].to(self.device)).float().cpu() for i in range(0, len(u8), self.batch)]
        return torch.cat(out).numpy()

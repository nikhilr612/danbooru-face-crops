"""Evaluator with the PyTorch DINOv2 extractor and a GPU memorisation search."""
import numpy as np
import torch

from dbfaces.metrics import Evaluator as _Evaluator
from dbfaces.metrics import Reference
from dbfaces.torch.dino import DinoV2


def max_cosine(query, bank, chunk: int = 4096, device=None) -> np.ndarray:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    unit = lambda x: torch.nn.functional.normalize(torch.as_tensor(np.asarray(x, np.float32)).to(dev), dim=1)
    b = unit(bank)  # float32 throughout: float16 similarities near 1 are spaced 5e-4 apart, which would make the memorisation threshold a coarse grid
    with torch.no_grad():
        return torch.cat([(unit(query[i:i + chunk]) @ b.T).max(1).values.cpu() for i in range(0, len(query), chunk)]).numpy()


class Evaluator(_Evaluator):
    """`Evaluator(Reference.load())`; `device` for the network (default cuda if available)."""

    def __init__(self, reference: Reference | None = None, device=None, size: int = 128, batch: int = 128):
        super().__init__(DinoV2(device, size, batch).features, reference or Reference.load(), lambda q, b: max_cosine(q, b, device=device))

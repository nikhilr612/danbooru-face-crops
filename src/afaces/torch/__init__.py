"""PyTorch backend (extra `torch`): DataLoader and the DINOv2 Evaluator for the metrics."""
from afaces.torch.dino import DinoV2
from afaces.torch.loader import Dataset, dataloader
from afaces.torch.metrics import Evaluator

__all__ = ["Dataset", "DinoV2", "Evaluator", "dataloader"]

"""PyTorch backend (extra `torch`): DataLoader and the DINOv2 Evaluator for the metrics."""
from dbfaces.torch.dino import DinoV2
from dbfaces.torch.loader import Dataset, dataloader
from dbfaces.torch.metrics import Evaluator

__all__ = ["Dataset", "DinoV2", "Evaluator", "dataloader"]

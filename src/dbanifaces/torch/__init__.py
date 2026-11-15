"""PyTorch backend (extra `torch`): DataLoader and the DINOv2 Evaluator for the metrics."""
from dbanifaces.torch.dino import DinoV2
from dbanifaces.torch.loader import Dataset, dataloader
from dbanifaces.torch.metrics import Evaluator

__all__ = ["Dataset", "DinoV2", "Evaluator", "dataloader"]

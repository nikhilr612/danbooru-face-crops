"""torch Dataset and DataLoader over a Split."""
import numpy as np
import torch

from dbanifaces.data import Split, numeric


class Dataset(torch.utils.data.Dataset):
    """Rows as dicts of tensors (`rating` as uint8 0 = g, 1 = s). Batches are read with one sorted fancy index per batch (`__getitems__`)."""

    def __init__(self, split: Split, transform=None):
        self.split, self.transform = split, transform

    def __len__(self) -> int:
        return len(self.split)

    def _row(self, row: dict) -> dict:
        row = {k: torch.from_numpy(np.array(v)) for k, v in numeric(row).items()}
        return self.transform(row) if self.transform else row

    def __getitem__(self, i):
        return self._row(self.split[i])

    def __getitems__(self, idx: list) -> list:
        b = numeric(self.split.batch(idx))
        return [self._row({k: v[j] for k, v in b.items()}) for j in range(len(idx))]


def dataloader(split: Split, batch_size: int, shuffle: bool = False, num_workers: int = 0, drop_last: bool = False, seed: int = 0, transform=None, **kw):
    """torch DataLoader over `split`; `seed` fixes the shuffle order."""
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(Dataset(split, transform), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, generator=g,
                                       persistent_workers=num_workers > 0, **kw)

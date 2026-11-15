"""Grain random-access source and DataLoader over a Split."""
import grain.python as gp

from dbanifaces.data import Split, numeric


class Source:
    """Grain RandomAccessDataSource: a row is a dict of numpy arrays (`rating` as uint8 0 = g, 1 = s). Picklable, so Grain worker processes can use it."""

    def __init__(self, split: Split):
        self.split = split

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, i: int) -> dict:
        return numeric(self.split[int(i)])


def loader(split: Split, batch_size: int, shuffle: bool = True, seed: int = 0, num_epochs: int | None = 1, worker_count: int = 0, drop_remainder: bool = True,
           shard: bool = True, operations=()):
    """Grain DataLoader yielding dicts of numpy batches. `num_epochs=None` repeats forever; `shard=True` gives each JAX process its own slice;
    extra Grain `operations` (maps, random maps) run before batching."""
    sampler = gp.IndexSampler(num_records=len(split), shuffle=shuffle, seed=seed, num_epochs=num_epochs,
                              shard_options=gp.ShardByJaxProcess(drop_remainder=True) if shard else gp.NoSharding())
    return gp.DataLoader(data_source=Source(split), sampler=sampler, operations=[*operations, gp.Batch(batch_size, drop_remainder=drop_remainder)], worker_count=worker_count)

import numpy as np
import pytest

import dbanifaces


def test_core(root):
    ds = dbanifaces.load(root, "val", fields=("image", "tags", "post_id", "flags"))
    assert len(ds) == len(np.load(f"{root}/val/post_id.npy", mmap_mode="r"))
    r = ds[3]
    assert r["image"].shape == (128, 128, 3) and r["image"].dtype == np.uint8 and r["tags"].dtype == np.int8
    b = ds.batch(np.array([5, 1, 5, 0]))
    assert (b["post_id"] == [ds[5]["post_id"], ds[1]["post_id"], ds[5]["post_id"], ds[0]["post_id"]]).all()
    assert len(ds.subset(100)) == 100 and len(ds[2:6]["image"]) == 4
    assert len(dbanifaces.reliable_tags(root)) == 43


def test_pickle_is_small(root):
    import pickle

    ds = dbanifaces.load(root, "val", fields=("image",))
    assert len(pickle.dumps(ds)) < 2000 and (pickle.loads(pickle.dumps(ds))[7]["image"] == ds[7]["image"]).all()


def test_torch(root):
    torch = pytest.importorskip("torch")
    from dbanifaces.torch import dataloader

    ds = dbanifaces.load(root, "val", fields=("image", "rating", "post_id"))
    a = next(iter(dataloader(ds, 16, shuffle=True, seed=1)))
    b = next(iter(dataloader(ds, 16, shuffle=True, seed=1)))
    assert a["image"].shape == (16, 128, 128, 3) and a["image"].dtype == torch.uint8 and a["rating"].dtype == torch.uint8
    assert (a["post_id"] == b["post_id"]).all()
    plain = next(iter(dataloader(ds, 8)))
    assert (plain["post_id"].numpy() == ds.batch(np.arange(8))["post_id"]).all()
    w = next(iter(dataloader(ds, 8, num_workers=2)))
    assert (w["post_id"] == plain["post_id"]).all()


def test_grain(root):
    pytest.importorskip("grain")
    from dbanifaces.jax import loader

    ds = dbanifaces.load(root, "val", fields=("image", "rating", "post_id"))
    first = [b["post_id"] for b in loader(ds, 8, shuffle=False, num_epochs=1, shard=False)]
    assert len(first) == len(ds) // 8 and (first[0] == ds.batch(np.arange(8))["post_id"]).all()
    a = next(iter(loader(ds, 16, seed=3, shard=False)))
    b = next(iter(loader(ds, 16, seed=3, shard=False)))
    assert a["image"].shape == (16, 128, 128, 3) and a["rating"].dtype == np.uint8 and (a["post_id"] == b["post_id"]).all()
    # with workers, Grain gives each worker every second record and batches inside the worker: same rows, different batch order
    w = [np.asarray(b["post_id"]).copy() for b in loader(ds.subset(64), 8, shuffle=False, shard=False, worker_count=2)]
    assert sorted(np.concatenate(w)) == sorted(ds.batch(np.arange(64))["post_id"])

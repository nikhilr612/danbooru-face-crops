"""Load a split, look at a row and a batch, pick the reliable tags. Downloads only the fields asked for (about 1.5 GB for val images)."""
import numpy as np

import dbanifaces

ds = dbanifaces.load("val", fields=("image", "tags", "palette", "rating"))
print(len(ds), ds[0]["image"].shape, ds[0]["palette"].shape)

batch = ds.batch(np.arange(64))  # dict of arrays
print({k: v.shape for k, v in batch.items()})

high = dbanifaces.reliable_tags(("high",))  # tag columns with AUC >= 0.97: safe to condition on
info = dbanifaces.tag_info()
print([info["names"][i] for i in high][:10])

quick = ds.subset(10_000)  # every prefix of a split is a stratified sample
print(len(quick))

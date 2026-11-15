"""Full build, stage 0: candidate id batches from the rating-g tag-filtered pool (pilot/pool_tag_filtered.parquet).

Usage: python build/full_ids.py   -> build_data/ids_g_batch<k>.json (20 batches of 50 shards, shard order seeded, pilot ids excluded)
Batches are processed one at a time, so the funnel can be read off and the run stopped once the quota is met.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("build_data")
OUT.mkdir(exist_ok=True)
pool = pd.read_parquet("pilot/pool_tag_filtered.parquet", columns=["score"])
pilot = set(json.load(open("pilot/pilot_ids.json"))["ids"])
ids = np.array([i for i in pool.index if i not in pilot])
order = np.random.default_rng(1).permutation(1000)
for k in range(20):
    sh = order[k * 50:(k + 1) * 50]
    sel = [int(i) for i in ids[np.isin(ids % 1000, sh)]]
    json.dump({"seed": 1, "shards": sorted(int(s) for s in sh), "ids": sel}, open(OUT / f"ids_g_batch{k:02d}.json", "w"))
    print(k, len(sel))

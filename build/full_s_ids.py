"""Full build, stage 0 for rating s: candidate id batches = rating-s tag-filtered pool, minus the exclusion list (build/s_exclude_tags.txt), keeping only
posts that carry a plain-background tag (they pass the background filter ~2x as often, 48% vs 22% below 0.04 in the g run).

Usage: python build/full_s_ids.py   -> build_data/ids_s_batch<k>.json (10 batches, seeded shard order). Prints per-tag removal counts.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PLAIN = {"simple_background", "white_background", "grey_background", "gradient_background", "blue_background", "pink_background", "yellow_background",
         "red_background", "green_background", "purple_background", "orange_background", "brown_background", "black_background", "beige_background",
         "two-tone_background", "aqua_background", "light_blue_background"}
EX = {l.strip() for l in open("build/s_exclude_tags.txt") if l.strip() and not l.startswith("#")}
d = pd.read_parquet("pilot/pool_s_tag_filtered.parquet", columns=["tag_string_general"])
tok = d.tag_string_general.fillna("").str.split().map(frozenset)
hit = tok.map(lambda s: bool(s & EX))
plain = tok.map(lambda s: bool(s & PLAIN))
print(len(d), "pool |", int(hit.sum()), "removed by exclusion list |", int((~hit).sum()), "clean |", int((~hit & plain).sum()), "clean + plain-background")
print("per-tag removals:", Counter(t for s in tok[hit] for t in s & EX).most_common())
ids = d.index[~hit & plain].to_numpy()
out = Path("build_data")
order = np.random.default_rng(2).permutation(1000)
for k in range(10):
    sh = order[k * 100:(k + 1) * 100]
    sel = [int(i) for i in ids[np.isin(ids % 1000, sh)]]
    json.dump({"seed": 2, "shards": sorted(int(s) for s in sh), "ids": sel}, open(out / f"ids_s_batch{k:02d}.json", "w"))
    print(k, len(sel))

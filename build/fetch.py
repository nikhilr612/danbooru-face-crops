"""Range-fetch single images out of the Danbooru2024-SFW tar shards.

A post lives in shard ``id % 1000``; ``images/NNNN.json`` maps ``<id>.<ext>`` to the
data offset, size and SHA-256 of the file inside ``NNNN.tar``.
"""

import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

BASE = "https://huggingface.co/datasets/deepghs/danbooru2024-sfw/resolve/main/images"
INDEX_DIR = Path("pilot/shard_index")


def shard_index(shard: int) -> dict:
    """The ``files`` map of one shard, cached under ``pilot/shard_index/``."""
    path = INDEX_DIR / f"{shard:04d}.json"
    if not path.exists():
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        r = requests.get(f"{BASE}/{shard:04d}.json", timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
    files = json.loads(path.read_text())["files"]
    return {int(name.split(".")[0]): (name, meta) for name, meta in files.items()}


def fetch_one(post_id: int, out_dir: Path, session: requests.Session, index: dict) -> tuple[int, str]:
    """Download and verify one image; returns ``(post_id, status)``."""
    if post_id not in index:
        return post_id, "not_in_shard"
    name, meta = index[post_id]
    dst = out_dir / name
    if dst.exists():
        return post_id, "cached"
    lo, hi = meta["offset"], meta["offset"] + meta["size"] - 1
    for attempt in range(4):
        try:
            r = session.get(f"{BASE}/{post_id % 1000:04d}.tar", headers={"Range": f"bytes={lo}-{hi}"}, timeout=120)
            if r.status_code == 206 and hashlib.sha256(r.content).hexdigest() == meta["sha256"]:
                dst.write_bytes(r.content)
                return post_id, "ok"
            status = f"bad_{r.status_code}"
        except requests.RequestException as e:
            status = type(e).__name__
        time.sleep(2 ** attempt)
    return post_id, status


def fetch(ids: list[int], out_dir: Path, workers: int = 16) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    indexes = {s: shard_index(s) for s in sorted({i % 1000 for i in ids})}
    counts: dict[str, int] = {}
    with requests.Session() as session, ThreadPoolExecutor(workers) as pool:
        for _, status in pool.map(lambda i: fetch_one(i, out_dir, session, indexes[i % 1000]), ids):
            counts[status] = counts.get(status, 0) + 1
    return counts


if __name__ == "__main__":
    pool_path, out, shards, per_shard = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    df = pd.read_parquet(pool_path, columns=["file_ext"])
    ids = [int(i) for s in range(shards) for i in df.index[df.index % 1000 == s][:per_shard]]
    t = time.time()
    print(len(ids), "ids", fetch(ids, out), f"{time.time() - t:.1f}s")

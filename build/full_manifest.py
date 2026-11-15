"""Full build: manifest.json for a release directory (file sizes and sha256, row counts, version, hash of thresholds.json).
Usage: python build/full_manifest.py DIR THRESHOLDS.json"""
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


d, thr = Path(sys.argv[1]), Path(sys.argv[2])
files = sorted(p for p in d.rglob("*") if p.is_file() and p.name != "manifest.json")
with ThreadPoolExecutor(8) as ex:
    hashes = list(ex.map(sha, files))
import pandas as pd
rows = {k: int(v) for k, v in pd.read_parquet("build_data/selection.parquet", columns=["split"]).split.value_counts().items()}
json.dump({"name": "danbooru-face-crops-128", "version": json.load(open(thr))["version"], "thresholds_sha256": sha(thr),
           "rows": rows, "files": {str(p.relative_to(d)): {"bytes": p.stat().st_size, "sha256": h} for p, h in zip(files, hashes)}},
          open(d / "manifest.json", "w"), indent=1)
print(len(files), "files hashed")

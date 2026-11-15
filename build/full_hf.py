"""Full build: Hugging Face `datasets`-compatible Parquet from the assembled release arrays (streamed in shards, PNG-encoded image columns).

Usage: .envs/metrics/bin/python -I build/full_hf.py RELEASE_DIR HF_DIR [LIMIT_PER_SPLIT]
Writes HF_DIR/data/<split>-NNNNN-of-MMMMM.parquet (splits train / validation / test; the npy directories call the second one `val`), same column names as the
arrays. `image`, `hull`, `parse`, `mouth_fill` are Image columns (lossless PNG; parse and mouth_fill single-channel); the rest are typed columns.
The arrow schema carries the `datasets` feature metadata, so `load_dataset("parquet", data_files=...)` returns the Image feature.
"""
import io
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Features, Image as HFImage, Sequence, Value
from PIL import Image

SHARD = 5000
SPLITS = (("train", "train"), ("val", "validation"), ("test", "test"))
TEXT = ["source_batch", "md5", "file_ext", "tag_string_general", "tag_string_character", "tag_string_copyright", "tag_string_artist", "created_at"]
INTS = {"source_row": "int32", "orig_width": "int32", "orig_height": "int32"}


def features(K: int) -> Features:
    f = {"image": HFImage(), "hull": HFImage(), "parse": HFImage(), "mouth_fill": HFImage(),
         "landmarks": Sequence(Sequence(Value("float32"), length=3), length=28), "palette": Sequence(Value("float32"), length=32),
         "palette_valid": Sequence(Value("uint8"), length=4), "tags": Sequence(Value("int8"), length=K),
         "crop_box": Sequence(Value("float32"), length=4), "group": Sequence(Value("int32"), length=3),
         "post_id": Value("int64"), "rating": Value("string"), "score": Value("int32"), "fav_count": Value("int32"), "created_year": Value("int32"),
         "popularity_pct": Value("float32"), "bg_score": Value("float32"), "flags": Value("uint8"), "idx": Value("int32")}
    f |= {c: Value("string") for c in TEXT} | {c: Value(t) for c, t in INTS.items()}
    return Features(f)


def png(a: np.ndarray) -> bytes:
    b = io.BytesIO()
    Image.fromarray(a).save(b, format="PNG")
    return b.getvalue()


def enc(args):
    return [png(a) for a in args]


def fixed(a: np.ndarray, typ) -> pa.Array:
    """Nested numpy array (n, ...) -> nested FixedSizeList arrow array."""
    flat = pa.array(np.ascontiguousarray(a).reshape(-1))
    for dim in a.shape[:0:-1]:
        flat = pa.FixedSizeListArray.from_arrays(flat, int(dim))
    return flat


def main(rel: str, dst: str, limit: int | None):
    rel, dst = Path(rel), Path(dst)
    (dst / "data").mkdir(parents=True, exist_ok=True)
    tj = json.load(open(rel / "tags.json"))
    feats = features(len(tj["names"]))
    schema = feats.arrow_schema
    pool = mp.get_context("fork").Pool(24)
    for sd, name in SPLITS:
        d = rel / sd
        N = len(np.load(d / "post_id.npy", mmap_mode="r"))
        N = min(N, limit) if limit else N
        meta = pd.read_parquet(d / "meta.parquet").iloc[:N]
        A = {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in ("image", "hull", "parse", "mouth_fill", "landmarks", "palette", "palette_valid", "tags", "crop_box", "group",
                                                                  "post_id", "rating", "score", "fav_count", "created_year", "popularity_pct", "bg_score", "flags")}
        nshard = -(-N // SHARD)
        for s in range(nshard):
            lo, hi = s * SHARD, min((s + 1) * SHARD, N)
            cols = {}
            for k in ("image", "hull", "parse", "mouth_fill"):
                x = A[k][lo:hi]
                chunks = [x[i:i + 250] for i in range(0, len(x), 250)]
                bs = [b for part in pool.map(enc, chunks) for b in part]
                cols[k] = pa.StructArray.from_arrays([pa.array(bs, pa.binary()), pa.array([None] * len(bs), pa.string())], names=["bytes", "path"])
            for k in ("landmarks", "palette", "palette_valid", "tags", "crop_box", "group"):
                cols[k] = fixed(np.asarray(A[k][lo:hi]), None)
            for k in ("post_id", "score", "fav_count", "created_year", "popularity_pct", "bg_score", "flags"):
                cols[k] = pa.array(np.asarray(A[k][lo:hi]))
            cols["rating"] = pa.array([r.decode() for r in np.asarray(A["rating"][lo:hi])])
            cols["idx"] = pa.array(np.arange(lo, hi, dtype=np.int32))
            m = meta.iloc[lo:hi]
            for c in TEXT:
                cols[c] = pa.array(m[c].astype(str).where(m[c].notna(), None).tolist(), pa.string())
            for c, t in INTS.items():
                cols[c] = pa.array(m[c].to_numpy().astype(t))
            t = pa.table([cols[f.name].cast(f.type) if not isinstance(cols[f.name], pa.StructArray) else cols[f.name] for f in schema], schema=schema)
            pq.write_table(t, dst / "data" / f"{name}-{s:05d}-of-{nshard:05d}.parquet", compression="zstd", row_group_size=1000)
            print(name, s + 1, "/", nshard, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)

"""Pilot step 1: slim local copy of the rating-g (or, with a second argument, another rating) rows of the Danbooru2024-SFW metadata.

Reads only the needed columns, one row group at a time, over HTTP range requests.
"""

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

SRC = "datasets/deepghs/danbooru2024-sfw/metadata.parquet"
COLS = [
    "id", "rating", "score", "fav_count", "created_at", "image_width", "image_height",
    "file_ext", "file_size", "md5", "parent_id", "is_deleted", "is_banned", "is_pending",
    "is_flagged", "tag_string_general", "tag_string_character", "tag_string_copyright",
    "tag_string_artist", "tag_string_meta",
]


def main(out: Path, rating: str = "g") -> None:
    fs = HfFileSystem()
    parts = []
    with fs.open(SRC, "rb") as f:
        pf = pq.ParquetFile(f)
        for i in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(i, columns=COLS)
            t = t.filter(pc.equal(t["rating"], rating))
            print(f"row group {i}: kept {t.num_rows}", flush=True)
            parts.append(t)
    table = pa.concat_tables(parts)
    pq.write_table(table, out, compression="zstd")
    print("rows", table.num_rows, "->", out, out.stat().st_size / 1e9, "GB")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "g")

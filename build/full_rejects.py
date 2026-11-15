"""Full build: rejects.parquet, one row per candidate post that went through stage 1 and was not selected, with the stage and reason.

Stages: detect (stage 1: fetch / face rules), frame (stage 2), background (stage 3), select (dedup, tagger flags, text, caps, surplus; from build_data/rejects.parquet).
Posts removed before detection (metadata, tag filters, the rating-s exclusion list and the plain-background tag prefilter) are not listed per id: their counts
are in the funnel files and the dataset card.
Usage: .envs/metrics/bin/python -I build/full_rejects.py OUT.parquet"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("build_data")
MIN_SCORE, MIN_FACE = 0.25, 160


def stage1(path: Path):
    out = []
    for line in open(path):
        r = json.loads(line)
        if r["status"] != "ok":
            out.append((r["id"], "detect", r["status"].split(":")[0]))
            continue
        keep = [d for d in r["dets"] if (d["src"] == "hysts" and d["score"] >= MIN_SCORE) or (d["src"] == "deepghs" and d["match"] < 0)]
        if not keep:
            out.append((r["id"], "detect", "no_face"))
        elif len(keep) > 1:
            out.append((r["id"], "detect", "multiple_faces"))
        elif keep[0]["src"] != "hysts":
            out.append((r["id"], "detect", "face_not_confirmed"))
        elif max(keep[0]["box"][2] - keep[0]["box"][0], keep[0]["box"][3] - keep[0]["box"][1]) < MIN_FACE:
            out.append((r["id"], "detect", "face_too_small"))
    return path.stem.split("_")[1], out


def main(dst: str):
    files = sorted(D.glob("run_[gs]_batch*.jsonl"))
    with mp.get_context("fork").Pool(24) as pool:
        res = pool.map(stage1, files)
    rows = [(i, rating, st, why) for rating, o in res for i, st, why in o]
    for tag in sorted(p.stem.removesuffix("_ids") for p in (D / "crops").glob("*_ids.npy")):
        ids = np.load(D / f"crops/{tag}_ids.npy")
        bg = np.load(D / f"crops/{tag}_bg.npy")
        rows += [(int(i), tag[0], "background", "bg_at_least_0.04" if np.isfinite(b) else "bg_unscored") for i, b in zip(ids, bg) if not b < 0.04]
        for i, why in json.load(open(D / f"crops/{tag}_status.json")).items():
            rows.append((int(i), tag[0], "frame", why))
    d = pd.DataFrame(rows, columns=["post_id", "rating", "stage", "reason"])
    s = pd.read_parquet(D / "rejects.parquet").rename(columns={"pid": "post_id"})
    s["stage"] = "select"
    s["reason"] = s.reason.replace({"text_pending": "not_needed_band"})
    d = pd.concat([d, s[["post_id", "rating", "stage", "reason"]]], ignore_index=True)
    d = d.drop_duplicates("post_id", keep="first")
    d.to_parquet(dst)
    print(len(d)); print(d.groupby(["stage", "reason"]).size().to_string())


if __name__ == "__main__":
    main(sys.argv[1])

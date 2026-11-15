"""Full build, assembly: per-split arrays from build_data/selection.parquet.

Usage: .envs/metrics/bin/python -I build/full_assemble.py OUT_DIR [LIMIT_PER_SPLIT]
Writes OUT_DIR/<split>/*.npy (fields as in SCHEMA.md), OUT_DIR/<split>/meta.parquet, OUT_DIR/tags.json and OUT_DIR/thresholds_assemble.json.
Everything is derived from the selection, so a changed selection (rows dropped or backfilled) is a re-run, not a rebuild of the crops.
Definitions chosen here (recorded in thresholds_assemble.json): hull masks as tinyflow's layouts (convex hulls of face / eyes / mouth landmarks, filled at 4x,
area-averaged); mouth_fill = convex hull of the parse-mouth pieces whose centroid lies within 0.6 inter-ocular distances of the landmark mouth centre
(fallback: the landmark mouth hull, flag bit 1); flag bit 2 = frame shifted (the hysts face-box centre, mapped into the crop, more than 3 px from where the frame rule puts it, i.e. the window was shifted to stay inside the image).
Flag bit 0 (eye_spill / profile) has no definition yet and stays 0.
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from scipy import ndimage

sys.path.insert(0, "build")

from geometry import EYE_A, EYE_B, FACE, MOUTH, convex_hull  # noqa: E402
from afaces.palette import extract as palette_extract  # noqa: E402
import tagspec  # noqa: E402

D = Path("build_data")
SPLITS = ("train", "val", "test")
S = 128
SS = 4
MOUTH_R_IOD = 0.6
SHIFT_PX = 3.0
FACE_TARGET = (64.0, 64.0 + 0.2 / 2.4 * 128)  # where the frame rule puts the face centre in the crop


def fill(canvas: np.ndarray, poly: np.ndarray) -> None:
    """Fill a convex polygon (k, 2) in canvas pixels (sample centres at +0.5) into a 2-D float canvas, inside its bounding box only."""
    if len(poly) < 3:
        return
    h, w = canvas.shape
    x0, x1 = max(int(np.floor(poly[:, 0].min())), 0), min(int(np.ceil(poly[:, 0].max())) + 1, w)
    y0, y1 = max(int(np.floor(poly[:, 1].min())), 0), min(int(np.ceil(poly[:, 1].max())) + 1, h)
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    px, py = xs + 0.5, ys + 0.5
    inside = np.ones(px.shape, bool)
    for i in range(len(poly)):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % len(poly)]
        inside &= (bx - ax) * (py - ay) - (by - ay) * (px - ax) >= 0
    canvas[y0:y1, x0:x1][inside] = 1.0


def hulls(lm: np.ndarray) -> np.ndarray:
    """(28, 2+) landmarks in crop pixels -> (128, 128, 3) uint8: face, eyes, mouth (tinyflow's layout rasteriser, clipped like it, 4x supersampled)."""
    hi = S * SS
    p = (np.clip(lm[:, :2] / S * 2 - 1, -1.2, 1.2) + 1) / 2 * hi
    c = np.zeros((hi, hi, 3), np.float32)
    fill(c[..., 0], convex_hull(p[FACE]))
    fill(c[..., 1], convex_hull(p[EYE_A]))
    fill(c[..., 1], convex_hull(p[EYE_B]))
    fill(c[..., 2], convex_hull(p[MOUTH]))
    return (c.reshape(S, SS, S, SS, 3).mean((1, 3)) * 255 + 0.5).astype(np.uint8)


def mouth_fill(lm: np.ndarray, parse: np.ndarray) -> tuple[np.ndarray, bool]:
    """(128,128) uint8 mouth mask and whether the parse had to be abandoned for the landmark hull."""
    hi = S * SS
    centre = lm[MOUTH, :2].mean(0)
    iod = np.linalg.norm(lm[EYE_A, :2].mean(0) - lm[EYE_B, :2].mean(0))
    lab, n = ndimage.label(parse == 9)
    pts = []
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lab == k)
        if np.hypot(xs.mean() + 0.5 - centre[0], ys.mean() + 0.5 - centre[1]) <= MOUTH_R_IOD * iod:
            pts.append(np.stack([xs, ys], 1) + 0.5)
            pts.append(np.stack([xs + 1, ys + 1], 1) + 0.0)
    c = np.zeros((hi, hi), np.float32)
    ok = bool(pts) and sum(len(x) for x in pts) >= 8
    poly = convex_hull(np.concatenate(pts)) if ok else convex_hull(lm[MOUTH, :2])
    fill(c, np.asarray(poly, float) * SS)
    return (c.reshape(S, SS, S, SS).mean((1, 3)) * 255 + 0.5).astype(np.uint8), not ok


def _boxes_file(args):
    path, want = args
    out = {}
    for line in open(path):
        if not line.startswith('{"id": '):
            continue
        r = json.loads(line)
        if r["id"] in want:
            hy = [d for d in r.get("dets", []) if d["src"] == "hysts" and d["score"] >= 0.25]
            if hy:
                out[r["id"]] = hy[0]["box"][:4]
    return out


def face_boxes(pids: set, pool) -> dict:
    """post id -> hysts face box (x1, y1, x2, y2) in source pixels, from the stage-1 records."""
    out = {}
    for d in pool.map(_boxes_file, [(f, pids) for f in sorted(D.glob("run_[gs]_batch*.jsonl"))]):
        out |= d
    return out


def hull_job(args):
    lm, parse = args
    m, fb = mouth_fill(lm, parse)
    return hulls(lm), m, fb


def pal_job(args):
    return palette_extract(*args)


def popularity() -> pd.Series:
    """Month-cohort percentiles of score and fav_count over the whole source pool (g and s), averaged and re-ranked; indexed by post id."""
    parts = [pd.read_parquet(f"pilot/{f}.parquet", columns=["score", "fav_count", "created_at"]) for f in ("g_meta", "s_meta")]
    d = pd.concat(parts)
    d = d[~d.index.duplicated()]
    d["month"] = pd.to_datetime(d.created_at, utc=True).dt.to_period("M").astype(str)
    sp = d.groupby("month")["score"].rank(pct=True)
    fp = d.groupby("month")["fav_count"].rank(pct=True)
    return ((sp + fp) / 2).rank(pct=True).astype(np.float32)


def metas(sel: pd.DataFrame) -> pd.DataFrame:
    cols = ["score", "fav_count", "created_at", "image_width", "image_height", "file_ext", "md5", "tag_string_general", "tag_string_character",
            "tag_string_copyright", "tag_string_artist"]
    out = []
    for r, f in (("g", "g_meta"), ("s", "s_meta")):
        m = pd.read_parquet(f"pilot/{f}.parquet", columns=cols)
        out.append(m.loc[sel.pid[sel.rating == r]])
    return pd.concat(out).loc[sel.pid]


def main(out_dir: str, limit: int | None):
    out = Path(out_dir)
    sel_all = pd.read_parquet(D / "selection.parquet")
    pop = popularity()
    ts = tagspec.spec()
    names = ts["names"]
    voc = json.load(open(D / "feats/wd_general.json"))
    cols = [voc.index(t) for t in names]
    out.mkdir(exist_ok=True, parents=True)
    json.dump(ts, open(out / "tags.json", "w"), indent=1)
    artist_id = {a: i for i, a in enumerate(sorted(sel_all.artist.unique()))}
    pool = mp.get_context("fork").Pool(24)
    boxes = face_boxes(set(sel_all.pid.tolist()), pool)
    stats = {}
    for split in SPLITS:
        sel = sel_all[sel_all.split == split].sort_values("idx")
        if limit:
            sel = sel.head(limit)
        N = len(sel)
        sd = out / split
        sd.mkdir(exist_ok=True)
        A = {k: open_memmap(sd / f"{k}.npy", "w+", dt, sh) for k, dt, sh in [
            ("image", np.uint8, (N, S, S, 3)), ("hull", np.uint8, (N, S, S, 3)), ("parse", np.uint8, (N, S, S)), ("mouth_fill", np.uint8, (N, S, S)),
            ("landmarks", np.float32, (N, 28, 3)), ("palette", np.float32, (N, 32)), ("palette_valid", np.uint8, (N, 4)),
            ("tags", np.int8, (N, len(names))), ("crop_box", np.float32, (N, 4)), ("bg_score", np.float32, (N,)), ("flags", np.uint8, (N,))]}
        sel = sel.reset_index(drop=True)
        meta = metas(sel)
        for tag, g in sel.groupby("tag"):
            pos = g.index.to_numpy()
            order = np.argsort(g.row.to_numpy())
            pos, rows = pos[order], g.row.to_numpy()[order]
            img = np.load(D / f"crops/{tag}_image.npy", mmap_mode="r")[rows]
            lm = np.load(D / f"crops/{tag}_landmarks.npy", mmap_mode="r")[rows]
            parse = np.load(D / f"crops/{tag}_parse.npy", mmap_mode="r")[rows]
            frows = np.load(D / f"feats/{tag}_rows.npy")
            W = np.load(D / f"feats/{tag}_wd.npy", mmap_mode="r")[np.searchsorted(frows, rows)][:, cols].astype(np.float32)
            res = pool.map(hull_job, list(zip(lm, parse)), chunksize=64)
            pal = pool.map(pal_job, [(img[i:i + 256], parse[i:i + 256]) for i in range(0, len(rows), 256)])
            fb = np.array([r[2] for r in res])
            cb = np.load(D / f"crops/{tag}_crop_box.npy", mmap_mode="r")[rows]
            bx = np.array([boxes.get(int(p), [np.nan] * 4) for p in g.pid.to_numpy()[order]], float)
            cx, cy = ((bx[:, 0] + bx[:, 2]) / 2 - cb[:, 0]) / cb[:, 2] * S, ((bx[:, 1] + bx[:, 3]) / 2 - cb[:, 1]) / cb[:, 2] * S
            shifted = np.nan_to_num(np.hypot(cx - FACE_TARGET[0], cy - FACE_TARGET[1]), nan=99) > SHIFT_PX
            for k, v in (("image", img), ("parse", parse), ("landmarks", lm), ("hull", np.stack([r[0] for r in res])), ("mouth_fill", np.stack([r[1] for r in res])),
                         ("palette", np.concatenate([p[0] for p in pal]).reshape(len(rows), 32)), ("palette_valid", np.concatenate([p[1] for p in pal])),
                         ("tags", tagspec.values(W, ts)), ("crop_box", np.load(D / f"crops/{tag}_crop_box.npy", mmap_mode="r")[rows]),
                         ("bg_score", np.load(D / f"crops/{tag}_bg.npy")[rows]), ("flags", (fb.astype(np.uint8) << 1 | shifted.astype(np.uint8) << 2).astype(np.uint8))):
                A[k][pos] = v
            print(split, tag, len(rows), flush=True)
        for a in A.values():
            a.flush()
        year = pd.to_datetime(meta.created_at, utc=True).dt.year.to_numpy()
        np.save(sd / "post_id.npy", sel.pid.to_numpy().astype(np.int64))
        np.save(sd / "rating.npy", sel.rating.to_numpy().astype("S1"))
        np.save(sd / "score.npy", meta.score.to_numpy().astype(np.int32))
        np.save(sd / "fav_count.npy", meta.fav_count.to_numpy().astype(np.int32))
        np.save(sd / "created_year.npy", year.astype(np.int32))
        np.save(sd / "popularity_pct.npy", pop.loc[sel.pid].to_numpy().astype(np.float32))
        np.save(sd / "group.npy", np.stack([sel.artist.map(artist_id).where(sel.artist != "unk", -1), sel.grp93, sel.cluster], 1).astype(np.int32))
        m = meta.reset_index().rename(columns={"id": "post_id", "image_width": "orig_width", "image_height": "orig_height"})
        m.insert(1, "source_batch", sel.tag.to_numpy())
        m.insert(2, "source_row", sel.row.to_numpy())
        m["rating"] = sel.rating.to_numpy()
        m.to_parquet(sd / "meta.parquet")
        fl = np.load(sd / "flags.npy")
        stats[split] = {"n": N, "mouth_fallback": int(((fl >> 1) & 1).sum()), "frame_shifted": int(((fl >> 2) & 1).sum()),
                        "palette_valid_share": np.load(sd / "palette_valid.npy").mean(0).round(3).tolist()}
        print(split, stats[split], flush=True)
    json.dump({"hull_supersample": SS, "mouth_radius_iod": MOUTH_R_IOD, "frame_shift_px": SHIFT_PX, "face_target": FACE_TARGET, "tag_present": tagspec.PRESENT,
               "tag_absent": tagspec.ABSENT, "stats": stats}, open(out / "thresholds_assemble.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)

"""Full build, stage 2: re-fetch each stage-1 survivor and cut the head-and-neck frame from the original (the stage-1 region is too small for it).

Frame (as in review_fetch.frames "tight"): square window of 2.4x the longer face-box side, centred 0.2 side above the face centre, shifted to stay
inside the image; the post is dropped if the window is larger than the image's short side. The window is area-resized to 128 px.

Usage: python build/full_crop.py TAG [LIMIT]  (TAG = pool letter + batch, e.g. g05, s03)  reads build_data/run_P_batchBB.jsonl, writes build_data/crops/TAG_{image,landmarks,crop_box}.npy,
TAG_ids.npy and TAG_status.json (status of every survivor that was dropped). Run with .envs/metrics/bin/python.
"""
import io
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from fetch import BASE, shard_index  # noqa: E402

S, UP, SIDE = 128, 0.2, 2.4
OUT = Path("build_data/crops")
INDEX: dict = {}


def _decode(data: bytes):
    Image.MAX_IMAGE_PIXELS = 120_000_000
    im = Image.open(io.BytesIO(data))
    if im.mode in ("RGBA", "LA", "P"):
        im = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im.convert("RGBA"))
    return im.convert("RGB")


def work(rec):
    pid = rec["id"]
    det = next(d for d in rec["dets"] if "region" in d)
    try:
        _, meta = INDEX[pid % 1000][pid]
    except KeyError:
        return pid, "not_in_shard", None
    data = None
    for attempt in range(6):
        try:
            r = requests.get(f"{BASE}/{pid % 1000:04d}.tar", timeout=120,
                             headers={"Range": f"bytes={meta['offset']}-{meta['offset'] + meta['size'] - 1}"})
            if r.status_code == 206 and len(r.content) == meta["size"]:
                data = r.content
                break
        except requests.RequestException:
            pass
        time.sleep(min(2 ** attempt * 2, 40))
    if data is None:
        return pid, "fetch_failed", None
    try:
        img = _decode(data)
    except Exception as e:  # noqa: BLE001
        return pid, f"decode_failed:{type(e).__name__}", None
    w, h = img.size
    x0, y0, x1, y1 = det["box"]
    s, cx, cy = max(x1 - x0, y1 - y0), (x0 + x1) / 2, (y0 + y1) / 2
    side = SIDE * s
    if side > min(w, h):
        return pid, "frame_exceeds_image", None
    wx, wy = min(max(cx - side / 2, 0), w - side), min(max(cy - UP * s - side / 2, 0), h - side)
    crop = img.resize((S, S), Image.LANCZOS, box=(wx, wy, wx + side, wy + side), reducing_gap=3.0)
    lm = np.asarray(det["lm"], np.float32)
    lm[:, :2] = (lm[:, :2] - [wx, wy]) * (S / side)
    return pid, "ok", (np.asarray(crop), lm, np.array([wx, wy, side, min(w, h)], np.float32))


def run(recs, ids, imgs, lms, boxes, status, t):
    with mp.get_context("fork").Pool(48) as pool:
        for n, (pid, st, res) in enumerate(pool.imap(work, recs, chunksize=4), 1):
            if st == "ok":
                ids.append(pid), imgs.append(res[0]), lms.append(res[1]), boxes.append(res[2])
                status.pop(pid, None)
            else:
                status[pid] = st
            if n % 2000 == 0:
                print(f"{n}/{len(recs)} {time.time() - t:.0f}s", flush=True)


def main(batch: str, limit: int):
    OUT.mkdir(parents=True, exist_ok=True)
    tag = batch + ("_test" if limit < 10**8 else "")
    recs = [r for r in map(json.loads, open(f"build_data/run_{batch[0]}_batch{int(batch[1:]):02d}.jsonl")) if r["status"] == "ok" and any("region" in d for d in r["dets"])][:limit]
    for s in sorted({r["id"] % 1000 for r in recs}):
        INDEX[s] = shard_index(s)
    t = time.time()
    ids, imgs, lms, boxes, status = [], [], [], [], {}
    if (OUT / f"{tag}_status.json").exists():  # repair mode: keep what exists, redo only the transient failures
        ids = np.load(OUT / f"{tag}_ids.npy").tolist()
        imgs, lms, boxes = list(np.load(OUT / f"{tag}_image.npy")), list(np.load(OUT / f"{tag}_landmarks.npy")), list(np.load(OUT / f"{tag}_crop_box.npy"))
        status = {int(k): v for k, v in json.load(open(OUT / f"{tag}_status.json")).items()}
        recs = [r for r in recs if status.get(r["id"]) in ("fetch_failed", "not_in_shard") or str(status.get(r["id"], "")).startswith("decode_failed:Conn")]
    run(recs, ids, imgs, lms, boxes, status, t)
    again = [r for r in recs if status.get(r["id"]) == "fetch_failed"]
    if again:  # second pass for the posts that still failed, after the server has had time to recover
        print(f"retrying {len(again)} failed fetches", flush=True)
        time.sleep(60)
        run(again, ids, imgs, lms, boxes, status, t)
    np.save(OUT / f"{tag}_ids.npy", np.array(ids, np.int64))
    np.save(OUT / f"{tag}_image.npy", np.stack(imgs))
    np.save(OUT / f"{tag}_landmarks.npy", np.stack(lms))
    np.save(OUT / f"{tag}_crop_box.npy", np.stack(boxes))
    json.dump(status, open(OUT / f"{tag}_status.json", "w"))
    print(f"batch {batch}: {len(recs)} survivors -> {len(ids)} crops, dropped {len(status)} {dict(__import__('collections').Counter(status.values()))}, {time.time() - t:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10**9)

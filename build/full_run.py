"""Full build, stage 1: fetch -> detect (hysts + deepghs) -> store detections, and a face region only for posts that pass the face filter.

Same detector code as pilot_run.py. Differences: a region (~2.5x the box, longest side capped at 256, enough for the 128 px master)
is written only when the post has exactly one hysts detection >= MIN_SCORE, no unmatched deepghs detection and a face box >= MIN_FACE px;
every post's detections are still recorded, so the filter can be re-derived. Resumable; output file per call.

Usage: python build/full_run.py IDS.json OUT.jsonl LIMIT WORKERS    (IDS.json: {"ids": [...]}; regions go to build_data/regions/)
"""

import io
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
sys.path.insert(0, str(Path(__file__).parent))
from fetch import BASE, shard_index  # noqa: E402

REGION_SCALE, REGION_CAP, MIN_SCORE, MIN_FACE = 2.5, 256, 0.25, 160
OUT = Path("build_data")
INDEX: dict = {}
DET = None


def _init():
    global DET
    import warnings

    warnings.filterwarnings("ignore")
    import torch
    from anime_face_detector import create_detector

    import onnxruntime as ort

    # One thread per worker: ORT's default pool per process oversubscribes the CPU ~25x.
    orig = ort.InferenceSession.__init__

    def one_thread(self, path, sess_options=None, *a, **k):
        so = sess_options or ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = 1
        orig(self, path, so, *a, **k)

    ort.InferenceSession.__init__ = one_thread
    torch.set_num_threads(1)
    DET = create_detector("yolov3", device="cpu")


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _decode(data: bytes):
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = 120_000_000
    im = Image.open(io.BytesIO(data))
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return im.convert("RGB")


def process(post_id: int) -> dict:
    import numpy as np
    import requests
    from imgutils.detect import detect_faces
    from PIL import Image

    rec = {"id": post_id}
    try:
        name, meta = INDEX[post_id % 1000][post_id]
    except KeyError:
        return {**rec, "status": "not_in_shard"}
    t0 = time.time()
    data = None
    for attempt in range(4):
        try:
            r = requests.get(f"{BASE}/{post_id % 1000:04d}.tar", timeout=120,
                             headers={"Range": f"bytes={meta['offset']}-{meta['offset'] + meta['size'] - 1}"})
            if r.status_code == 206 and len(r.content) == meta["size"]:
                data = r.content
                break
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    if data is None:
        return {**rec, "status": "fetch_failed"}
    t1 = time.time()
    try:
        img = _decode(data)
    except Exception as e:  # noqa: BLE001
        return {**rec, "status": f"decode_failed:{type(e).__name__}"}
    w, h = img.size
    bgr = np.asarray(img)[..., ::-1].copy()
    dets = []
    for p in DET(bgr):
        dets.append({"src": "hysts", "box": [float(v) for v in p["bbox"][:4]], "score": float(p["bbox"][4]),
                     "lm": np.round(p["keypoints"], 2).tolist(), "match": -1, "iou": 1.0})
    n_h = len(dets)
    for (x0, y0, x1, y1), _, s in detect_faces(img, level="s", version="v1.4", conf_threshold=MIN_SCORE):
        box = [float(x0), float(y0), float(x1), float(y1)]
        ious = [_iou(box, d["box"]) for d in dets[:n_h]]
        best = max(range(n_h), key=ious.__getitem__) if n_h else -1
        d = {"src": "deepghs", "box": box, "score": float(s), "lm": None, "match": best if best >= 0 and ious[best] >= 0.5 else -1,
             "iou": round(ious[best], 3) if best >= 0 else 0.0}
        if d["iou"] < 0.7:  # boxes the two detectors disagree on: landmarks from the deepghs box
            q = DET(bgr, boxes=[np.array(box + [float(s)])])[0]
            d["lm"] = np.round(q["keypoints"], 2).tolist()
        dets.append(d)
    keep = [d for d in dets if (d["src"] == "hysts" and d["score"] >= MIN_SCORE) or (d["src"] == "deepghs" and d["match"] < 0)]
    one = len(keep) == 1 and keep[0]["src"] == "hysts" and max(keep[0]["box"][2] - keep[0]["box"][0], keep[0]["box"][3] - keep[0]["box"][1]) >= MIN_FACE
    for k, d in enumerate(dets):
        if not one:
            break
        if (d["src"] == "hysts" and d["score"] >= MIN_SCORE) or (d["src"] == "deepghs" and d["match"] < 0):
            x0, y0, x1, y1 = d["box"]
            half = REGION_SCALE * max(x1 - x0, y1 - y0) / 2
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            rx0, ry0 = max(0, int(cx - half)), max(0, int(cy - half))
            rx1, ry1 = min(w, int(cx + half)), min(h, int(cy + half))
            if rx1 - rx0 < 8 or ry1 - ry0 < 8:
                continue
            region = img.crop((rx0, ry0, rx1, ry1))
            scale = min(1.0, REGION_CAP / max(region.size))
            if scale < 1.0:
                region = region.resize((max(1, round(region.width * scale)), max(1, round(region.height * scale))), Image.LANCZOS)
            path = OUT / "regions" / f"{post_id}_{k}.png"
            region.save(path)
            d["region"] = {"path": path.name, "x": rx0, "y": ry0, "scale": round(scale, 5)}
    return {**rec, "status": "ok", "name": name, "w": w, "h": h, "bytes": meta["size"], "dets": dets,
            "t_fetch": round(t1 - t0, 2), "t_det": round(time.time() - t1, 2)}


def main(ids_path: str, out_file: str, limit: int, workers: int):
    ids = json.load(open(ids_path))["ids"][:limit]
    out = Path(out_file)
    done = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in out.open()}
    todo = [i for i in ids if i not in done]
    for s in sorted({i % 1000 for i in todo}):
        INDEX[s] = shard_index(s)
    (OUT / "regions").mkdir(parents=True, exist_ok=True)
    print(f"{len(todo)} to do ({len(done)} already done)", flush=True)
    t = time.time()
    with mp.get_context("fork").Pool(workers, initializer=_init) as pool, out.open("a") as f:
        for n, rec in enumerate(pool.imap_unordered(process, todo, chunksize=2), 1):
            f.write(json.dumps(rec) + "\n")
            if n % 250 == 0:
                f.flush()
                print(f"{n}/{len(todo)} {time.time() - t:.0f}s", flush=True)
    print("done", f"{time.time() - t:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))

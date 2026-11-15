"""Full build, text filter on 128 px crops (rule calibrated in build/full_text_calib.py: OCR box score >= 0.7 OR deepghs detect_text >= 0.1 on the
crop upsampled to 512; on the 121 labelled review crops: catches 10/15 text crops, 12/106 false alarms).

Usage: nice .envs/landmarks/bin/python -I build/full_text.py OUT.npz N_PROCS  < list of "TAG ROW" lines on stdin  (default: random 3000 candidates)
Writes OUT.npz with ocr, dg (max scores) and the (tag, row) keys.
"""
import os, sys
os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
import multiprocessing as mp
import numpy as np
from PIL import Image

IMGS = {}


def init():
    import onnxruntime as ort
    orig = ort.InferenceSession.__init__
    def one(self, path, sess_options=None, *a, **k):
        so = sess_options or ort.SessionOptions(); so.intra_op_num_threads = so.inter_op_num_threads = 1
        orig(self, path, so, *a, **k)
    ort.InferenceSession.__init__ = one


def score(key):
    from imgutils.detect.text import detect_text
    from imgutils.ocr import detect_text_with_ocr
    tag, row = key
    if tag not in IMGS:
        IMGS[tag] = np.load(f"build_data/crops/{tag}_image.npy", mmap_mode="r")
    im = Image.fromarray(np.asarray(IMGS[tag][row])).resize((512, 512), Image.LANCZOS)
    o = max([s for _, _, s in detect_text_with_ocr(im, box_threshold=0.5)] or [0.0])
    d = max([s for _, _, s in detect_text(im, threshold=0.05)] or [0.0])
    return o, d


if __name__ == "__main__":
    out, procs = sys.argv[1], int(sys.argv[2])
    keys = [] if sys.stdin.isatty() else [(a, int(b)) for a, b in (l.split() for l in sys.stdin)]
    if not keys:
        rng = np.random.default_rng(0)
        for t in [f"g{k:02d}" for k in range(20)]:
            r = np.load(f"build_data/feats/{t}_rows.npy")
            keys += [(t, int(j)) for j in rng.choice(r, 150, replace=False)]
    chunk = 5000  # saved per chunk (OUT with _NNN before the extension) so a long run can be resumed and watched
    stem = out.removesuffix(".npz")
    with mp.get_context("fork").Pool(procs, initializer=init) as pool:
        for c, i in enumerate(range(0, len(keys), chunk)):
            if os.path.exists(f"{stem}_{c:03d}.npz"):
                continue
            ks = keys[i:i + chunk]
            res = pool.map(score, ks, chunksize=20)
            np.savez(f"{stem}_{c:03d}.npz", ocr=np.array([r[0] for r in res]), dg=np.array([r[1] for r in res]), tag=np.array([k[0] for k in ks]), row=np.array([k[1] for k in ks]))
            o, d = np.array(res).T
            print(f"chunk {c}: {i + len(ks)}/{len(keys)} flagged {float(((o >= .7) | (d >= .1)).mean()):.3f}", flush=True)

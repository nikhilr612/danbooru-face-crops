"""tags.npy specification: the list (build/tags_v1.txt) ordered by reliability (build/tag_reliability.csv: AUC, precision and recall of the WD tagger on the
128 px crops against Danbooru's own tags, 40,000 selected posts, 2026-11-15), the tri-state rule and the metadata written to tags.json.

Columns are sorted by AUC, best first, so a prefix is the most reliable subset. Tiers: high AUC >= 0.97, medium >= 0.93, low below.
Values: 1 present (probability >= 0.5), 0 absent (< 0.3), -1 unknown. Tags with recall@0.5 below 0.5 against Danbooru are "positive-only": their absent
band is also unknown (-1), because a miss there is more often the tagger than a true absence (checked on 24 crops each for six of them).
Exclusive groups keep the highest-probability tag of the group and zero the other present ones.
"""
import numpy as np
import pandas as pd

PRESENT, ABSENT, POS_ONLY_RECALL = 0.5, 0.3, 0.5


def spec() -> dict:
    raw = []
    for line in open("build/tags_v1.txt"):
        if line.startswith("#") or ":" not in line:
            continue
        head, rest = line.split(":", 1)
        h = head.split()
        raw += [(t, h[0], "exclusive" in h, "colour" in h) for t in rest.split()]
    rel = pd.read_csv("build/tag_reliability.csv").set_index("tag")
    raw.sort(key=lambda r: -float(rel.loc[r[0], "auc"]))
    names = [r[0] for r in raw]
    auc = rel.loc[names, "auc"].to_numpy(float)
    rec = rel.loc[names, "rec@.5"].to_numpy(float)
    return {"names": names, "group": [r[1] for r in raw], "exclusive": [r[2] for r in raw], "colour": [r[3] for r in raw],
            "auc": auc.round(3).tolist(), "precision": rel.loc[names, "prec@.5"].round(3).tolist(), "recall": rec.round(3).tolist(),
            "danbooru_rate": rel.loc[names, "danbooru_rate"].round(3).tolist(),
            "tier": ["high" if a >= 0.97 else "medium" if a >= 0.93 else "low" for a in auc],
            "positive_only": (rec < POS_ONLY_RECALL).tolist(), "present": PRESENT, "absent": ABSENT}


def values(W: np.ndarray, s: dict) -> np.ndarray:
    """W (n, K) float probabilities in spec order -> (n, K) int8."""
    out = np.where(W >= PRESENT, 1, np.where(W < ABSENT, 0, -1)).astype(np.int8)
    po = np.array(s["positive_only"])
    out[:, po] = np.where(out[:, po] == 0, -1, out[:, po])
    for g in {g for g, e in zip(s["group"], s["exclusive"]) if e}:
        cols = [i for i, (gg, e) in enumerate(zip(s["group"], s["exclusive"])) if gg == g and e]
        keep = np.zeros((len(W), len(cols)), bool)
        keep[np.arange(len(W)), W[:, cols].argmax(1)] = True
        sub = out[:, cols]
        sub[~keep & (sub == 1)] = 0 if not po[cols].any() else -1
        out[:, cols] = sub
    return out

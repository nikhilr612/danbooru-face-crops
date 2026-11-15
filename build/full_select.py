"""Full build, row selection and split.

Usage: .envs/metrics/bin/python -I build/full_select.py todo BG_MAX      -> build_data/text/todo_BG_MAX.txt: "TAG ROW" lines still needing the text rule
                                                                        (rows with bg < BG_MAX that survive dedup and the tagger flags)
       .envs/metrics/bin/python -I build/full_select.py select [N=210000] -> build_data/selection.parquet, selection_report.json, rejects.parquet

Text scores come from build/full_text.py runs saved as build_data/text/*.npz (keys: tag, row, ocr, dg); rows without a score are treated as
"pending" and are not selected. Selection order: bg < 0.03 first (rating g, then s subsampled to the quota), the 0.03 to 0.04 band (cleanest
first, g before s) only if the quota is not met. Dedup: DINOv2-B cosine >= 0.97 keeps the best-scoring member; >= 0.93 groups stay whole in the split.
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

D = Path("build_data")
BG_FIRST, BG_MAX = 0.03, 0.04
DUP_T, GROUP_T = 0.97, 0.93
CAPS = dict(artist=300, char=1500, copy_share=0.08)  # moderate: per artist, per character, per copyright (share of the quota; "original" is exempt, it is not a series)
SPLITS = (("train", 150_000), ("val", 30_000), ("test", 30_000))
FLAGS = {'text': ['signature', 'watermark', 'artist_name', 'twitter_username', 'web_address', 'logo', 'copyright_name', 'english_text', 'chinese_text',
                  'korean_text', 'speech_bubble', 'character_name', 'dated'],
         'layout': ['border', 'letterboxed', 'pillarboxed', 'multiple_views', 'split_screen', 'comic', 'multiple_girls', '2girls', 'multiple_boys', '2boys', 'no_humans'],
         'style': ['monochrome', 'greyscale', 'sketch', 'lineart', 'chibi', 'realistic'],
         'content': ['blood', 'nude', 'topless', 'nipples', 'covered_nipples', 'cleavage', 'underwear', 'bikini', 'censored']}


def batch_tags() -> list[str]:
    return sorted(p.stem.removesuffix("_dino") for p in (D / "feats").glob("*_dino.npy"))


def load():
    """One row per crop that passed stage 4 (bg < 0.04): provenance, bg, flags, colour class, and the DINO features."""
    voc = json.load(open(D / "feats/wd_general.json"))
    pos = {t: i for i, t in enumerate(voc)}
    flag_cols = [pos[t] for ts in FLAGS.values() for t in ts]
    hair = [(t, i) for t, i in pos.items() if t.endswith("_hair") and t.split("_")[0] in COLOURS]
    eyes = [(t, i) for t, i in pos.items() if t.endswith("_eyes") and t.split("_")[0] in COLOURS]
    parts, dino = [], []
    for tag in batch_tags():
        rows = np.load(D / f"feats/{tag}_rows.npy")
        W = np.load(D / f"feats/{tag}_wd.npy").astype(np.float32)
        h, e = W[:, [i for _, i in hair]], W[:, [i for _, i in eyes]]
        parts.append(pd.DataFrame({
            "tag": tag, "row": rows, "pid": np.load(D / f"crops/{tag}_ids.npy")[rows], "rating": tag[0],
            "bg": np.load(D / f"crops/{tag}_bg.npy")[rows],
            "flag": (W[:, flag_cols] >= 0.5).any(1),
            "hair": np.where(h.max(1) >= 0.3, h.argmax(1), -1), "eyes": np.where(e.max(1) >= 0.3, e.argmax(1), -1)}))
        dino.append(np.load(D / f"feats/{tag}_dino.npy"))
    return pd.concat(parts, ignore_index=True), np.concatenate(dino)


COLOURS = {"black", "brown", "blonde", "blue", "red", "pink", "purple", "green", "white", "grey", "orange", "aqua", "silver", "yellow"}


def add_meta(d: pd.DataFrame) -> pd.DataFrame:
    cols = ["score", "fav_count", "created_at", "tag_string_artist", "tag_string_character", "tag_string_copyright"]
    for r, f in (("g", "g_meta"), ("s", "s_meta")):
        m = pd.read_parquet(f"pilot/{f}.parquet", columns=cols)
        sel = d.rating == r
        got = m.loc[d.pid[sel]]
        for c in cols:
            if c not in d:
                d[c] = None
            d.loc[sel, c] = got[c].to_numpy()
    d["score"] = d.score.astype(np.int64)
    d["artist"] = d.tag_string_artist.fillna("").replace("", "unk").str.split().str[0]
    d["char"] = d.tag_string_character.fillna("").str.split().str[0].fillna("")
    d["copy"] = d.tag_string_copyright.fillna("").str.split().str[0].fillna("")
    return d


@torch.no_grad()
def components(F: np.ndarray, thresholds=(DUP_T, GROUP_T)):
    """Connected-component labels of the cosine-similarity graph at each threshold (DINO features, GPU in chunks)."""
    X = torch.from_numpy(F).cuda().float()
    X = (X / X.norm(dim=1, keepdim=True)).half()
    edges = {t: ([], []) for t in thresholds}
    lo = min(thresholds)
    for i in range(0, len(X), 2048):
        s = X[i:i + 2048] @ X.T
        s[torch.arange(len(s)), torch.arange(i, i + len(s))] = 0
        r, c = torch.nonzero(s >= lo, as_tuple=True)
        v = s[r, c].float().cpu().numpy()
        r, c = r.cpu().numpy() + i, c.cpu().numpy()
        for t in thresholds:
            k = v >= t
            edges[t][0].append(r[k]), edges[t][1].append(c[k])
    out = []
    for t in thresholds:
        r, c = np.concatenate(edges[t][0]), np.concatenate(edges[t][1])
        g = coo_matrix((np.ones(len(r), np.int8), (r, c)), shape=(len(F), len(F)))
        out.append(connected_components(g, directed=False)[1])
    return out


def prepare():
    d, F = load()
    d = add_meta(d)
    dup97, grp93 = components(F)
    d["grp93"] = grp93
    order = np.lexsort((d.bg.to_numpy(), -d.score.to_numpy(), dup97))  # best score first inside each 0.97 component, ties to the cleaner background
    first = np.zeros(len(d), bool)
    first[order[np.unique(dup97[order], return_index=True)[1]]] = True
    d["dup"] = ~first
    return d, F


def load_text() -> pd.DataFrame:
    parts = [np.load(f) for f in sorted(glob.glob(str(D / "text/*.npz")))]
    if not parts:
        return pd.DataFrame(columns=["tag", "row", "ocr", "dg"])
    t = pd.concat([pd.DataFrame({k: p[k] for k in ("tag", "row", "ocr", "dg")}) for p in parts]).drop_duplicates(["tag", "row"], keep="last")
    return t


def todo(bg_max: float):
    d, _ = prepare()
    d = d[(d.bg < bg_max) & ~d.dup & ~d.flag]
    done = set(zip(*[load_text()[c].tolist() for c in ("tag", "row")]))
    keys = [(t, r) for t, r in zip(d.tag, d["row"]) if (t, int(r)) not in done]
    (D / "text").mkdir(exist_ok=True)
    out = D / f"text/todo_{bg_max}.txt"
    out.write_text("".join(f"{t} {r}\n" for t, r in keys))
    print(f"{len(d)} candidates at bg<{bg_max} after dedup and flags; {len(keys)} need text scores -> {out}")


def cap(d: pd.DataFrame, col: str, n: int, reason: str, why: dict) -> pd.DataFrame:
    exempt = d[col].isin(["", "unk", "original"])
    over = (d.groupby(col).cumcount() >= n) & ~exempt
    why.update({int(i): reason for i in d.index[over]})
    return d[~over]


def select(N: int):
    d, F = prepare()
    t = load_text()
    d = d.merge(t, on=["tag", "row"], how="left")
    d["text_bad"] = (d.ocr >= 0.7) | (d.dg >= 0.1)
    if os.environ.get("DRYRUN"):  # preview only: unscored rows count as clean and nothing is written under the real names
        d["ocr"] = d.ocr.fillna(0.0)
    why = {}
    reasons = [("dup", d.dup), ("tagger_flag", d.flag), ("text", d.text_bad.fillna(False)), ("text_pending", d.ocr.isna())]
    d["reason"] = ""
    for name, m in reasons[::-1]:
        d.loc[m, "reason"] = name
    ok = d[d.reason == ""].copy()
    rng = np.random.default_rng(0)
    ok["rnd"] = rng.random(len(ok))
    report = {"candidates_bg<0.04": len(d), **{f"drop_{n}": int(m.sum()) for n, m in reasons}}

    def take(pool: pd.DataFrame, need: int):
        """Caps (artist, character, copyright), in order of popularity score, then g before s."""
        pool = pool.sort_values(["rating", "score"], ascending=[True, False]).copy()
        w = {}
        pool = cap(pool, "artist", CAPS["artist"], "cap_artist", w)
        pool = cap(pool, "char", CAPS["char"], "cap_char", w)
        pool = cap(pool, "copy", int(CAPS["copy_share"] * N), "cap_copyright", w)
        for i, r in w.items():
            d.loc[i, "reason"] = r
        return pool

    first = take(ok[ok.bg < BG_FIRST], N)
    g, s = first[first.rating == "g"], first[first.rating == "s"]
    report["pool_bg<0.03"] = {"g": len(g), "s": len(s)}
    chosen = [g]
    if len(g) < N:
        chosen.append(s.sort_values("rnd").head(N - len(g)))
    sel = pd.concat(chosen)
    if len(sel) < N:
        band = take(ok[(ok.bg >= BG_FIRST) & (ok.rating == "g")], N).sort_values("bg")
        sel = pd.concat([sel, band.head(N - len(sel))])
        if len(sel) < N:
            band_s = take(ok[(ok.bg >= BG_FIRST) & (ok.rating == "s")], N).sort_values("bg")
            sel = pd.concat([sel, band_s.head(N - len(sel))])
    d.loc[ok.index.difference(sel.index), "reason"] = d.loc[ok.index.difference(sel.index), "reason"].replace("", "not_needed")
    report |= {"selected": len(sel), "by_rating": sel.rating.value_counts().to_dict(), "bg_band": int((sel.bg >= BG_FIRST).sum()),
               "bg_max": float(sel.bg.max()) if len(sel) else None}
    if len(sel) < N:
        report["short_by"] = N - len(sel)
    sel = sel.copy()
    sel["split"] = assign_split(sel, N)
    sel = nested_order(sel)
    sel["cluster"] = kmeans_labels(F[sel.index.to_numpy()])
    keep = ["tag", "row", "pid", "rating", "bg", "score", "fav_count", "artist", "char", "copy", "grp93", "cluster", "hair", "eyes", "split", "idx", "ocr", "dg"]
    sel[keep].to_parquet(D / ("selection_dry.parquet" if os.environ.get("DRYRUN") else "selection.parquet"))
    d[d.reason != ""][["pid", "rating", "reason"]].to_parquet(D / ("rejects_dry.parquet" if os.environ.get("DRYRUN") else "rejects.parquet"))
    for name, _ in SPLITS:
        x = sel[sel.split == name]
        report[name] = {"n": len(x), "g_share": float((x.rating == "g").mean()), "artists": int(x.artist.nunique()), "chars": int(x.char.nunique())}
    json.dump(report, open(D / ("selection_report_dry.json" if os.environ.get("DRYRUN") else "selection_report.json"), "w"), indent=1, default=str)
    print(json.dumps(report, indent=1, default=str))


def assign_split(sel: pd.DataFrame, N: int) -> np.ndarray:
    """Near-duplicate groups stay whole; per stratum (rating, hair, eyes) each split gets its proportional share (largest deficit first)."""
    names, sizes = [n for n, _ in SPLITS], np.array([s for _, s in SPLITS], float)
    frac = sizes / sizes.sum()
    rng = np.random.default_rng(1)
    units = sel.groupby("grp93").agg(n=("pid", "size"), rating=("rating", "first"), hair=("hair", "first"), eyes=("eyes", "first"))
    units = units.sample(frac=1, random_state=1)
    got = {}
    out = {}
    for u, r in units.iterrows():
        key = (r.rating, r.hair, r.eyes)
        c = got.setdefault(key, np.zeros(len(names)))
        k = int(np.argmax(frac * (c.sum() + r.n) - c + rng.random(len(names)) * 1e-6))
        c[k] += r.n
        out[u] = names[k]
    return sel.grp93.map(out).to_numpy()


def nested_order(sel: pd.DataFrame) -> pd.DataFrame:
    """Row order within a split such that every prefix is a stratified sample (nested quick-eval subsets): rank inside stratum / stratum size."""
    rng = np.random.default_rng(2)
    parts = []
    for name, _ in SPLITS:
        x = sel[sel.split == name].copy()
        x["_r"] = rng.random(len(x))
        key = x.rating + x.hair.astype(str) + x.eyes.astype(str)
        x["_p"] = (x.groupby(key)["_r"].rank() - 0.5) / key.map(key.value_counts())
        x = x.sort_values("_p")
        x["idx"] = np.arange(len(x))
        parts.append(x.drop(columns=["_r", "_p"]))
    return pd.concat(parts)


def kmeans_labels(F: np.ndarray, k: int = 256) -> np.ndarray:
    from sklearn.cluster import MiniBatchKMeans
    X = F.astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return MiniBatchKMeans(k, random_state=0, n_init=1, batch_size=8192).fit_predict(X).astype(np.int32)


if __name__ == "__main__":
    if sys.argv[1] == "todo":
        todo(float(sys.argv[2]))
    else:
        select(int(sys.argv[2]) if len(sys.argv) > 2 else 210_000)

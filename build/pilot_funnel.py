"""Pilot step 3: sequential tag/metadata funnel on one rating slice.

Usage: python build/pilot_funnel.py META.parquet [POOL_OUT [FUNNEL_JSON]]   (defaults: the rating-g pool paths)
"""

import json
import sys
import urllib.request

import pandas as pd

df = pd.read_parquet(sys.argv[1])
POOL_OUT = sys.argv[2] if len(sys.argv) > 2 else "pilot/pool_tag_filtered.parquet"
FUNNEL_OUT = sys.argv[3] if len(sys.argv) > 3 else "pilot/funnel_tag.json"
RATING = df["rating"].mode()[0]
tok = {c: df[c].fillna("").str.split().map(frozenset) for c in ("tag_string_general", "tag_string_meta", "tag_string_copyright")}
tags = pd.Series([a | b for a, b in zip(tok["tag_string_general"], tok["tag_string_meta"])], index=df.index)
short = df[["image_width", "image_height"]].min(axis=1)
BAD = {"ai-generated", "ai-assisted", "monochrome", "greyscale", "sketch", "lineart", "comic", "3d", "photo_(medium)",
       "realistic", "chibi", "lowres", "jpeg_artifacts", "pixel_art", "animated", "furry", "no_humans", "text_focus", "scan",
       "blurry", "watermark"}
steps = [
    (f"rating {RATING}", pd.Series(True, index=df.index)),
    ("still image (jpg/png/webp)", df["file_ext"].isin(["jpg", "png", "webp"])),
    ("not deleted/pending/flagged", ~(df["is_deleted"] | df["is_pending"] | df["is_flagged"])),
    ("solo and 1girl|1boy", tok["tag_string_general"].map(lambda s: "solo" in s and ("1girl" in s or "1boy" in s))),
    ("no excluded style/quality tag", tags.map(lambda s: not (s & BAD))),
    ("short side >= 512", short >= 512),
    ("short side >= 768", short >= 768),
    ("score >= 5", df["score"] >= 5),
    ("score >= 15", df["score"] >= 15),
]
keep = pd.Series(True, index=df.index)
funnel = []
masks = {}
for name, m in steps:
    keep &= m
    masks[name] = keep.copy()
    funnel.append((name, int(keep.sum())))
    print(f"{name:34s} {int(keep.sum()):>9,}")
print("\nscore percentiles (solo g stills):", df.loc[steps[3][1] & steps[1][1], "score"].quantile([.1, .25, .5, .75, .9]).to_dict())
# Per-tag removal in the style/quality step, among posts that reached it.
reach = steps[0][1] & steps[1][1] & steps[2][1] & steps[3][1]
print("\nremoved by tag (among solo stills):", {t: int(tags[reach].map(lambda s: t in s).sum()) for t in sorted(BAD)})

pool = df[masks["short side >= 512"]].copy()
pool.to_parquet(POOL_OUT)
json.dump(funnel, open(FUNNEL_OUT, "w"), indent=1)

print("\npool rows", len(pool), "| score min", pool["score"].min())
yr = lambda d: pd.to_datetime(d["created_at"], utc=True).dt.year.value_counts(normalize=True).sort_index().round(3)
print(pd.DataFrame({"slice": yr(df), "pool": yr(pool)}).fillna(0).T.to_string())

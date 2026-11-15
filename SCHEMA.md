# Anime faces: dataset schema (draft, 2026-11-05)

Status tags: **[decided]** you decided it, **[proposed]** my proposal for you to edit, **[open]** needs a decision.
Nothing here is built yet. Edit freely; the build scripts will be written against the final version.

## Scope

- Head-and-neck portraits (face, neck, hair, accessories; not torso or clothing) of single characters (`solo`, `1girl` or `1boy`) **[decided]**.
- Plain or soft-gradient backgrounds only; no text, signature or watermark inside the crop **[decided]**.
- Source: deepghs/danbooru2024-sfw; rating `g` first, `s` fills where needed **[decided]**. Frames that would run past the image edge are dropped, not padded **[decided]**.
- 210k rows: train 150k / validation 30k / test 30k, matched category mixes, nested stratified subsets for quick evals **[decided 2026-11-12; was 250k = 190k / 20k / 40k]**. The target dropped because rating g plus all plain-background rating-s posts project to about 231k rows at background < 0.04 after the dedup, text and cap filters (about 177k at < 0.03), and a validation set large enough for model selection was kept.
- Master resolution 128 px **[decided]**.

## Filters (frozen in `thresholds.json`, with a published reject list of ids and reasons) **[proposed]**

| Step | Rule | Evidence so far |
|---|---|---|
| Metadata | still image, not deleted/pending/flagged, `solo` and (`1girl` or `1boy`), no excluded style tag, short side >= 512 | funnel on `g` and `s` |
| Face | exactly one hysts detection >= 0.25 and no second detection; face box >= 160 px | pilot: 76% of crops survive the detector filters |
| Frame | square window of 2.4x the longer face-box side, centred 0.2 side above the face centre, shifted inside the image; drop if it does not fit | review: 120/121 frames ok; ~14% dropped |
| Background | surface-fit residual (fit2_med) of the parse background below **0.03**; if the 210k quota (with caps and category mixes) cannot be met, relax stepwise to **0.04** **[decided 2026-11-08]** | AUC 0.83 vs your 121 labels; < 0.03 kept 43% of ok, let through 0/44 bad; < 0.04 kept 60%, 3/44 bad |
| Text | PaddleOCR `ch_PP-OCRv4_det` box >= 0.7 OR deepghs `detect_text` >= 0.1 flags the crop | on the 128 px crops (as published) caught 10/15 labelled text crops with 12/106 false alarms (mostly decoration); misses stylised handwriting |
| Crop-level content | tagger on the crop; drop what is visibly unsuitable at the frame edge [open: list] | not built |
| Dedup and caps | near-duplicate removal; caps on character, series, artist, hair x eye colour, embedding clusters: **moderate** strength **[decided 2026-11-08]**; the numbers are set from the funnel and reported | not built |

## Rows

Row order is the same in every file of a split. `idx` is the row number within the split.

### Arrays (`<split>/<field>.npy`, fixed-width numeric only; the loader takes `fields=`) **[decided layout]**

| File | Shape and dtype | Content | Status |
|---|---|---|---|
| `image.npy` | (N,128,128,3) uint8 | the head-and-neck crop, resized from the original | decided |
| `hull.npy` | (N,128,128,3) uint8 | face / eyes / mouth hull masks from the landmarks, supersampled then area-averaged | decided |
| `parse.npy` | (N,128,128) uint8 | segmenter class ids (0 background, 1 skin, 2 face, 3 hair, 4/5 eyes, 6/7 brows, 8 nose, 9 mouth, 10 clothes, 11 thin hair) | decided; annotator and thresholds under review |
| `mouth_fill.npy` | (N,128,128) uint8 | mouth mask: convex hull of the parse mouth pieces near the hysts mouth centre, fallback hysts hull | open: pending the mouth round |
| `landmarks.npy` | (N,28,3) float32 | hysts points (x, y in crop pixels at 128, confidence) | decided |
| `palette.npy` | (N,32) float32 | OKLab, 8 values for each of skin, hair, eyes, background | decided |
| `palette_valid.npy` | (N,4) uint8 | validity per region | decided |
| `tags.npy` | (N,K~60) int8 | -1 unknown, 0 absent, 1 present, from the tagger on the crop; mutually exclusive groups; colour tags excluded from conditioning | decided |
| `post_id.npy`, `rating.npy` | int64, `|S1` (`b'g'` or `b's'`, the Danbooru rating itself; a string column in the Parquet release) | provenance | decided 2026-11-12 (was a 0/1 uint8) |
| `score.npy`, `fav_count.npy`, `created_year.npy` | int32 | provenance | proposed |
| `popularity_pct.npy` | float32 | mean of the month-cohort percentiles of score and fav_count, re-ranked | proposed |
| `crop_box.npy` | (N,4) float32 | frame window in source pixels (x, y, side, source short side) | proposed |
| `bg_score.npy`, `flags.npy` | float32, (N,) uint8 bit set | background residual; bits: eye_spill/profile, mouth_parse_missing, frame_shifted | proposed |
| `group.npy` | (N,3) int32 | artist group, near-duplicate group, embedding cluster (used for the split and for caps) | proposed |

### Sidecar (`<split>/meta.parquet`; strings and ragged data only) **[decided layout]**

`post_id`, source shard, `md5`, `sha256`, file extension, original width and height, artist / copyright / character tags and the full Danbooru general tag string (provenance, not conditioning), annotator names and versions. Artist, character and raw `score`/`fav_count` fields ship in the public release **[decided 2026-11-08]**.

### Hugging Face release (`datasets`-compatible Parquet) **[decided]**

One config per split. `image` is an Image feature (128 px, lossless PNG proposed); `hull`, `parse` as Image columns; `landmarks`, `palette`, `tags` and the scalar fields as typed columns; same column names as the arrays above. **Publish the 128 px crops** (not ids-only) **[decided 2026-11-08]**.

## Dataset-level files **[proposed]**

- `manifest.json`: file hashes, row counts, version, `thresholds.json` hash.
- `thresholds.json`: every frozen filter value above.
- `rejects.parquet`: id, reason, for every candidate dropped.
- Dataset card: scope statement (portraits with margin; close-ups that fill the frame are dropped; from_behind dropped; profile views flagged), drop counts, annotator accuracy from the human review, the popularity-percentile caveats.

## Repo (GitHub) **[decided in outline]**

The `dbanifaces` package (src/dbanifaces): backend-free loader core, a PyTorch DataLoader and a Grain source for JAX in separate extras, and a minimal metric set on DINOv2-L CLS features (FD, KID, density/coverage, memorisation fraction, palette adherence) with a shared pinned weight set for both backends; the build scripts. The prototype metrics package is archived in `archive/metrics-prototype/` and is not released.

## Open decisions

1. Crop-level content filter list (tagger on the crop).
2. Mouth mask source (deferred by you; proposal in the notes).
3. Explicit-leaning tag exclusion list for rating `s` posts.
4. **TODO (deferred by you, 2026-11-08):** licence check of annotator weights (hysts, anime-seg, PaddleOCR, deepghs models). Relevant because the crops are published, so it has to be done before any release step; not blocking the build.

Decided 2026-11-08: publish crops; background threshold 0.03 with fallback 0.04; moderate caps; artist, character and raw score/fav fields ship.

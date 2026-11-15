# anime-faces

Tooling for the **Anime Faces 128** dataset: 210,000 head-and-neck anime portraits at 128 px (train 149,994 / validation 29,999 / test 30,007) with hull masks, part segmentation, palettes, 65 tags with measured reliability, landmarks and provenance. The data is published separately (Hugging Face Parquet, and memory-mappable `.npy` files with the same fields). This repository holds the loader and the scripts that built the dataset.

Dataset card: see the Hugging Face dataset page, [nikhilr01/danbooru-anime-faces-128](https://huggingface.co/datasets/nikhilr01/danbooru-anime-faces-128) (`README.md` there), for the field list, filters, known limitations, terms and takedown, and `LICENSES.md` here for the licence audit. `thresholds.json` freezes every filter value; `SCHEMA.md` is the design document.

## Loader

```python
import afaces

ds = afaces.load("anime-faces-128", "train", fields=("image", "tags"))   # splits: train, val, test
ds[0]                      # {"image": (128, 128, 3) uint8, "tags": (65,) int8}
ds.batch(np.arange(64))    # fancy-indexed batch
ds.subset(10_000)          # nested stratified quick-eval subset (every prefix of a split is stratified)
ds.torch()                 # torch Dataset;  ds.grain() -> random-access source for Grain
afaces.reliable_tags("anime-faces-128", tiers=("high",))    # tag columns with AUC >= 0.97
```

The `.npy` files are one per field with the same row order, so there are no joins: `image`, `hull`, `parse`, `mouth_fill`, `landmarks`, `palette`, `palette_valid`, `tags`, `crop_box`, `group`, `bg_score`, `flags`, `post_id`, `rating` (`b"g"` or `b"s"`), `score`, `fav_count`, `created_year`, `popularity_pct`; strings and ragged data are in `meta.parquet` per split. `tags.json` gives tag names (columns are ordered by reliability), groups, AUC / precision / recall, tier and which tags are positive-only. Test: `python tests/test_loader.py anime-faces-128`.

## Build scripts (`build/`)

Run from the repository root with the dataset's working directories (`pilot/` metadata parquet files, `build_data/` intermediates). In order:

1. `pilot_meta.py`, `pilot_funnel.py` (metadata funnel, candidate pools), `full_ids.py`, `full_s_ids.py` (candidate batches; `s_exclude_tags.txt` is the rating-s exclusion list)
2. `full_run.py` (stage 1: fetch, face detection), `full_crop.py` (stage 2: refetch originals, cut the frame), `full_bg.py` (stage 3: parse and background score), `full_feats.py` (stage 4: embeddings and tagger output); `full_*queue.sh` chain them per batch
3. `full_text.py` (text rule), `full_select.py` (dedup, flags, caps, quota, split)
4. `full_assemble.py` (arrays), `tagspec.py` + `full_tags.py` (tags and flags), `full_rejects.py`, `full_hf.py` (Parquet)

The scripts were run as a pipeline on one machine and have not been re-run end to end from a clean checkout. They need `dghs-imgutils`, `anime_face_detector`, `anime_seg` (pin `transformers<5`), `timm`, `torch`, `onnxruntime`, `datasets`.

## Not included

The evaluation metrics package is not part of this release. `build/palette.py` is only the palette extractor used to build the `palette` field.

## Terms

Code: MIT (see `LICENSE`). Data: the images are the original artists' works, derived from the deepghs/danbooru2024-sfw snapshot of 2024-11-18, under that dataset's conditions; see the dataset card and `LICENSES.md`. Not re-checked against live Danbooru.

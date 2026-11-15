# dbanifaces

Loader and metrics for the **Danbooru Face Crops 128** dataset, a curated subset of Danbooru: 210,000 head-and-neck anime face crops at 128 px (train 149,994 / validation 29,999 / test 30,007) with hull masks, part segmentation, palettes, 65 tags with measured reliability, landmarks and provenance. The data is published separately (Hugging Face: [Parquet](https://huggingface.co/datasets/nikhilr01/danbooru-face-crops-128) with the full dataset card, and the memory-mappable [`.npy` release](https://huggingface.co/datasets/nikhilr01/danbooru-face-crops-128-npy) this package reads). This repository holds the Python package and the scripts that built the dataset. `LICENSES.md` is the licence audit, `thresholds.json` freezes every filter value.

```
pip install "dbanifaces[torch] @ git+https://github.com/nikhilr612/danbooru-face-crops"     # PyTorch DataLoader + metrics
pip install "dbanifaces[jax]   @ git+https://github.com/nikhilr612/danbooru-face-crops"     # JAX + Grain loader + metrics
```

The base install needs only numpy and scipy. Everything else is an extra, and no backend is imported until you import it (`dbanifaces.torch`, `dbanifaces.jax`):

| Extra | Installs | For |
|---|---|---|
| `torch` | torch, safetensors, huggingface_hub | `dbanifaces.torch`: DataLoader, DINOv2 features, Evaluator |
| `jax` | jax, grain, safetensors, huggingface_hub | `dbanifaces.jax`: Grain loader, DINOv2 features, Evaluator |
| `hub` | huggingface_hub, safetensors | fetching data or weights without either backend (`dbanifaces.metrics` on features you computed yourself) |
| `data` | pandas, pyarrow | `meta.parquet` as a DataFrame |
| `seg` | anime-seg | palette adherence on generated images (needs a parse map; AnimeSeg has no stated licence, see `LICENSES.md`) |
| `build` | pandas, pyarrow, pillow, requests, huggingface_hub, datasets | the dataset build (`build/`) |
| `dev` | torch and jax extras, pytest, ruff, transformers | the tests |

## Loading

```python
import dbanifaces

ds = dbanifaces.load("train", fields=("image", "tags"))                       # splits: train, val, test; the official release from the Hub
ds = dbanifaces.load("train", fields=("image", "tags"), root="/data/faces")  # a local copy, or another Hub repo id
ds[0]                      # {"image": (128, 128, 3) uint8, "tags": (65,) int8}
ds.batch(np.arange(64))    # fancy-indexed batch (one sorted read)
ds.subset(10_000)          # nested stratified quick-eval subset (every prefix of a split is stratified)
dbanifaces.reliable_tags(("high",))    # tag columns with AUC >= 0.97
```

Without `root`, the `DBANIFACES_ROOT` environment variable is used if set (a local copy or a repo id), else the official release `dbanifaces.DEFAULT_REPO`. Given a Hub repo id, only the files for the split and fields you ask for are downloaded, then memory-mapped from the cache. Fields: `image`, `hull`, `parse`, `mouth_fill`, `landmarks`, `palette`, `palette_valid`, `tags`, `crop_box`, `group`, `bg_score`, `flags`, `post_id`, `rating`, `score`, `fav_count`, `created_year`, `popularity_pct`; strings and ragged data are in `meta.parquet` per split. `tags.json` gives tag names (columns are ordered by reliability), groups, AUC / precision / recall, tier and which tags are positive-only.

**PyTorch** (`dbanifaces.torch`): a `Dataset` that reads whole batches with one sorted index, and a `DataLoader` helper.

```python
from dbanifaces.torch import dataloader
for batch in dataloader(ds, batch_size=128, shuffle=True, num_workers=4, seed=0):   # dict of tensors; rating is uint8, 0 = g, 1 = s
    ...
```

**JAX + Grain** (`dbanifaces.jax`): a picklable random-access `Source` and a configured Grain `DataLoader` (shuffle with a seed, `num_epochs=None` to repeat, per-process sharding via `ShardByJaxProcess`, extra Grain operations for augmentation).

```python
from dbanifaces.jax import loader
for batch in loader(ds, batch_size=128, seed=0, num_epochs=None, worker_count=4):   # dict of numpy arrays
    ...
```

With `worker_count > 0`, Grain gives each worker every second (third, ...) record and batches inside the worker, so batch contents and order differ from `worker_count=0`; the shuffle is still seeded and reproducible.

## Metrics

Four distribution metrics, one per question, a single `headline` combining them, all on DINOv2 ViT-L/14 CLS features (1024-d), plus palette adherence for palette-conditioned models. They are deliberately few.

| Metric | Answers | Notes |
|---|---|---|
| `fd_dinov2` | does the sample distribution match the train distribution? | the headline number; float64; biased upwards at small N, so compare runs at the same N |
| `kid_dinov2` | the same, unbiased in N | read this at 1-2k samples inside a training loop; mean and std over 100 subsets of 1,000 |
| `density`, `coverage` | quality and diversity separately (Naeem et al. 2020) | k = 5 balls around 10,000 fixed real features; `coverage` is the one that drops when modes are missing |
| `memorised_frac` | does it copy training images? | share of samples closer to the train set than 99% of held-out real images are; about 0.01 for a generator that generalises |
| `headline` | one number to minimise | lower is better, about 0 for real data; FD excess over the real-data floor at your N plus a penalty for copying (below) |
| palette adherence | does the output follow the conditioning palette? | `dbanifaces.palette.score`; dE per region against the conditioning palette, read against `FLOOR` and `CHANCE` |

```python
from dbanifaces.torch import Evaluator          # or: from dbanifaces.jax import Evaluator
ev = Evaluator()                            # downloads the reference statistics (about 50 MB; the 300 MB train feature bank only on the first memorisation call)
for images in samples:                      # uint8 [B, H, W, 3], or float in [-1, 1]; 128 px (other sizes: Evaluator(size=...))
    ev.update(images)
ev.compute()    # {"n": ..., "fd_dinov2": ..., "kid_dinov2": ..., "kid_dinov2_std": ..., "density": ..., "coverage": ..., "memorised_frac": ...}
ev.reset()
```

Both backends load the same pinned safetensors (HF `facebook/dinov2-large`, Apache-2.0, revision and SHA-256 checked) and resize with the same fixed bicubic matrix, so they give the same features: JAX versus PyTorch agree to 5e-6 relative on the CLS vector, and both agree with the HF `transformers` model run in float64 to about 1e-4 (`tests/test_dino.py`). The `Evaluator` computes the metrics themselves in numpy float64 from the features, so there is no backend difference beyond the features. `dbanifaces.metrics` also takes features from any extractor: `Evaluator(featurize, Reference.load())`.

JAX note: by default JAX reserves most of the GPU's memory; set `XLA_PYTHON_CLIENT_PREALLOCATE=false` if the training run shares the card.

### The headline number

`ev.compute()["headline"]` (needs the FD and memorisation metrics, so any `compute()` without a metric subset) is one number to minimise:

```
headline = log10(1 + max(0, FD - floor(N))) + u / tau
u        = max(0, memorised_frac - (0.01 + 3 SE)) / 0.99          SE = sqrt(0.01 * 0.99 / N)
```

`floor(N)` is the FD of real held-out images at the same N, measured at N = 1k, 2k, 5k, 10k, 20k and interpolated in log-log between them (9.0 at 10k and 4.8 at 20k for the default train reference), so a perfect generator scores about 0 at any N: 0 to 0.11 on real test images at N from 1k to 30k, against 0.69 for a barely visible blur. The floor is measured on real images that are neither in the reference nor from the test split (the validation rows for the train reference), so the test split was not used to fit anything. `u` is the share of samples copied beyond the calibrated 1% and its sampling noise. The penalty is additive because copying leaves FD at its floor, so a product would stay 0. `tau = 0.279` is solved so that a generator that copies the training set and one that outputs Gaussian noise score the same ("equally worthless"; uniform and pink noise give FD within 2% of Gaussian, so the choice hardly matters). Floors and tau are stored in the reference (`Reference.headline_params`, `scripts/make_headline.py`).

**Selecting versus reporting.** The default reference is built from the train split, so a model is being pushed toward train statistics. To keep a held-out number for the end, build the evaluator on the validation or test split's statistics: `Evaluator(Reference.load(split="val"))`. Select checkpoints and hyperparameters against `"val"`, and report the final model once against `"test"`; neither was seen in training or in fitting the floors. (The memorisation bank and threshold are always the train split's; FD, KID, density and coverage use the chosen split's first 10,000 rows. A val or test reference has a higher floor than the train one, 11.5 against 9.0 at 10k, because 30k reference images estimate the statistics less well than 150k. The headline accounts for it.) Do not tune against the test reference.

| 2,000 test images, | `fd_dinov2` | `memorised_frac` | `headline` |
|---|---|---|---|
| real | 41.7 | 0.009 | 0.00 |
| 5% train copies mixed in | 41.7 | 0.059 | 0.16 |
| 20% train copies mixed in | 41.2 | 0.208 | 0.70 |
| all train copies | 41.7 | 1.000 | 3.56 |
| blur sigma 0.5 / 1 / 2 | 45.6 / 54.9 / 215 | 0.007 / 0.006 / 0.000 | 0.69 / 1.15 / 2.24 |
| flipped upside down | 665 | 0.000 | 2.80 |
| Gaussian noise | 3,776 | 0.000 | 3.57 |

It is a convention, not a measurement: `tau` encodes "copying is as bad as noise", and 20% copies scores like a mild blur. Read `memorised_frac` next to it. Compare headline values only at the same N, and use it to select models rather than as a training loss (it is built on DINOv2 features and will eventually be gamed). It has not been checked against human ratings.

### Reading the numbers

Measured on the **test** split (not used to build the reference), JAX backend, `scripts/floors.py`. The first rows of a split are a stratified sample, so `test[:N]` is the N-image real set.

Real images against the train reference, by sample size. Every metric except KID depends on N, so only compare scores at the same N:

| N | `fd_dinov2` | `kid_dinov2` | `density` | `coverage` | `memorised_frac` |
|---|---|---|---|---|---|
| 1,000 | 82.5 | -0.001 | 1.00 | 0.38 | 0.009 |
| 2,000 | 41.7 | -0.001 | 0.98 | 0.59 | 0.010 |
| 5,000 | 17.5 | 0.000 | 0.99 | 0.87 | 0.010 |
| 10,000 | 9.1 | 0.001 | 0.97 | 0.97 | 0.010 |

The real-data floor is what a perfect generator would score. KID has no floor to subtract (0 +- 0.004 over subsets); `memorised_frac` sits at its 1% calibration (the memorisation threshold, cosine 0.9375, was set on validation and holds on test).

Known problems applied to 2,000 test images (the real floor at N = 2,000 is FD 41.7, KID 0.000, coverage 0.59):

| problem | `fd_dinov2` | `kid_dinov2` | `density` | `coverage` | `memorised_frac` |
|---|---|---|---|---|---|
| Gaussian blur, sigma 0.5 px | 45.6 | 0.043 | 0.96 | 0.58 | 0.008 |
| blur, sigma 1 | 54.9 | 0.125 | 0.73 | 0.48 | 0.006 |
| blur, sigma 2 | 215 | 1.50 | 0.25 | 0.15 | 0.000 |
| 20% of images turned grey | 42.9 | 0.005 | 0.92 | 0.57 | 0.009 |
| flipped upside down | 665 | 4.74 | 0.07 | 0.01 | 0.000 |
| train images instead of generated | 41.7 | -0.004 | 1.08 | 0.70 | 1.000 |
| 20% train images mixed in | 41.2 | -0.002 | 1.03 | 0.63 | 0.209 |

(The brightest-half selection of 1,000 images, a mode drop: FD 96.8 against the N = 1,000 floor of 82.5, KID 0.097.) At this N, mild blur moves FD by about 4 but KID by ten times its noise, so **read KID at small N**; a 20% grey contamination is close to invisible to all of them, and copying is visible only to `memorised_frac` (FD, KID and coverage do not move, as expected).

Palette adherence: a real image scores 4.6 / 5.1 / 7.7 / 1.9 dE (skin / hair / eyes / background) against its own palette, the floor (`dbanifaces.palette.FLOOR`), and 11.0 / 24.1 / 16.0 / 15.6 against another image's palette, chance (`dbanifaces.palette.CHANCE`). Each region's own score is below 0.7x of its shuffled score on 300 validation images (`tests/test_palette.py`).

Cached features: `reference/dinov2l_{train,val,test}.npy` in the npy release hold the CLS features of every row (float16, row for row the same as the split's arrays; `Reference.features(split)`), so floors and your own analyses need no GPU.

Tests (`pytest`; set `DBANIFACES_ROOT` to a local npy release or its Hub repo id for the data tests): loaders (core, torch with workers, Grain with workers), JAX versus PyTorch versus `transformers` DINOv2, the metric formulas (FD against the `sqrtm` formula, KID, density/coverage, memorisation), palette extraction and adherence, and the two backends' Evaluators against each other.

## Build scripts (`build/`)

Run from the repository root with the dataset's working directories (`pilot/` metadata parquet files, `build_data/` intermediates). In order:

1. `pilot_meta.py`, `pilot_funnel.py` (metadata funnel, candidate pools), `full_ids.py`, `full_s_ids.py` (candidate batches; `s_exclude_tags.txt` is the rating-s exclusion list)
2. `full_run.py` (stage 1: fetch, face detection), `full_crop.py` (stage 2: refetch originals, cut the frame), `full_bg.py` (stage 3: parse and background score), `full_feats.py` (stage 4: embeddings and tagger output); `full_*queue.sh` chain them per batch
3. `full_text.py` (text rule), `full_select.py` (dedup, flags, caps, quota, split)
4. `full_assemble.py` (arrays), `tagspec.py` + `full_tags.py` (tags and flags), `full_rejects.py`, `full_hf.py` (Parquet)

The scripts were run as a pipeline on one machine and have not been re-run end to end from a clean checkout. They need `dghs-imgutils`, `anime_face_detector`, `anime_seg` (pin `transformers<5`), `timm`, `torch`, `onnxruntime`, `datasets`.

## Terms

Code: MIT (see `LICENSE`). Data: the images are the original artists' works, derived from the deepghs/danbooru2024-sfw snapshot of 2024-11-18, under that dataset's conditions; see the dataset card and `LICENSES.md`. Not re-checked against live Danbooru.

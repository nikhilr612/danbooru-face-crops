# Licence audit (2026-11-14)

Checked against the model cards, GitHub licence API, PyPI metadata and the dataset's LICENSE file on 2026-11-14. A record of what the sources state, not legal advice.

## Decisions (2026-11-14)

- Host the 128 px crops in the public release, and ship the post ids and the rebuild scripts alongside them.
- Ship `parse.npy` and everything derived from it; the card states that the segmenter has no stated licence.
- The metrics package is not part of the release (not confirmed by the owner).

## Source data

deepghs/danbooru2024-sfw: MIT-style licence ("MIT License-Minimal-Modified", licensor "NoCopyright") with conditions: keep the notice; no use for harassment, false information, harm, unlawful or hateful content; users take responsibility for derived outputs; comply with the opt-out policy of the original booru. The licensor is not the artists, so the grant does not cover the artists' copyright in the images; the crops remain derivative of their works.

Release conditions for the crops:
- carry the deepghs notice and the prohibited-use list in the dataset card;
- re-check every selected post against the current Danbooru state (deleted, banned artist, opt-out) at build time and drop what is gone; record the check date in `manifest.json`;
- takedown contact and process in the card; honour requests by removing rows and bumping the version;
- research-use statement (it states intent; it is not a licence);
- artist tags ship, and post ids ship so any row can be traced to its Danbooru post.

## Models whose outputs enter the published dataset

| Artefact | Model | Stated licence | Handling |
|---|---|---|---|
| `landmarks.npy`, `hull.npy`, face frame | hysts/anime-face-detector (code and weights) | MIT | credit in the card |
| `parse.npy`, `mouth_fill.npy`, `palette*`, `bg_score` | suzukimain/AnimeSeg (`anime_seg` 0.3.8, DINOv2 + U-Net++ backend) | none stated (no LICENSE file in the GitHub repo, none in the HF card, none in PyPI metadata) | the licence concerns the model code and weights, which are not redistributed; the masks are our outputs and contain no model weights. Ship; the card names the model and version and states that it has no stated licence. |
| `tags.npy` | SmilingWolf/wd-vit-tagger-v3 | Apache-2.0 | credit in the card |

## Models used only to filter or deduplicate (outputs not published, only thresholds and reject ids)

| Use | Model | Stated licence |
|---|---|---|
| second face detector (filter) | deepghs `anime_face_detection` | MIT |
| near-duplicate groups, embedding clusters | facebook/dinov2-base | Apache-2.0 |
| `afaces` metrics (FD, KID, density/coverage, memorisation) and the `reference/` features | facebook/dinov2-large, fetched from the Hub at run time (not redistributed in the package); the reference features are numeric outputs of it | Apache-2.0 |
| text rule | deepghs `paddleocr` (PaddleOCR det), deepghs `text_detection` | deepghs "other" (model-redistribution disclaimer, MIT text), MIT |

## Open items

1. Build-time re-check against Danbooru (above) is not written yet.
2. Metrics package: not published (owner decision).

## Update 2026-11-14 (later)

The owner chose to skip the live Danbooru re-check for now. The release conditions above that depend on it (drop deleted, banned and opt-out posts, record the check date) are therefore NOT met. The deepghs snapshot dates from 2024-11-18 and the licence requires compliance with the booru's opt-out policy, so the risk of carrying posts removed since then stays with the release. Post ids ship, so rows can be removed later; the takedown process in the card is the remaining mitigation.

## Update 2026-11-15

- Takedown contact set in the HF card (owner's email).
- The owner accepted the 45% rating-s share, given the filters and tags.

## Update 2026-11-15 (package)

The `afaces` package (loaders, metrics, palette extractor) is MIT. Its metrics use DINOv2-L weights downloaded from the Hub at run time (pinned revision, SHA-256 checked). Palette adherence on generated images can use AnimeSeg through the optional `seg` extra; it is not installed or imported otherwise, and the licence position for AnimeSeg is unchanged (none stated).

"""Per-region colour palettes in OKLab and a palette-adherence metric. Pure numpy/scipy; the parse maps come from anime-seg run on the 128 px image
(`afaces.seg.parse`, extra `seg`), so the dataset field and the metric use one extractor on one input resolution.

Layout, 8 values per region (schema: skin, hair, eyes, background -> 32 dims):
    [L1, a1, b1, L2, a2, b2, w2, area]   primary colour, secondary colour and its weight (0 = flat region, secondary == primary), area share of the crop.
Atoms are canonical: k=2 clustering with a deterministic start, the heavier cluster first. Validity (palette_valid) is stored separately.
Adherence (the metric) compares the CONDITIONING palette (two atoms with weights) with the generated region's full pixel distribution: the cost of the
cheapest transport of the region's pixels onto the two atoms with capacities (1-w2, w2), in OKLab dE (Euclid x 100). For two atoms this is exact: send the
pixels that prefer atom 2 most, up to its weight. It includes the within-region spread, so even the source image scores above 0 against its own palette
(the quantisation floor, reported as the `src` variant)."""
import numpy as np
from scipy.ndimage import binary_erosion

REGIONS = ("skin", "hair", "eyes", "background")
CLASSES = {"skin": (1, 2), "hair": (3, 11), "eyes": (4, 5), "background": (0,)}
ERODE = {"skin": 1, "hair": 1, "eyes": 0, "background": 1}        # px at 128
MIN_PX = {"skin": 150, "hair": 150, "eyes": 12, "background": 150}  # after erosion
FLAT_DE, MIN_W2 = 3.0, 0.05                                        # secondary collapses into the primary below these


def srgb_to_oklab(u8: np.ndarray) -> np.ndarray:
    c = u8.astype(np.float64) / 255
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    l, m, s = (np.cbrt(c @ w) for w in (np.array([.4122214708, .5363325363, .0514459929]), np.array([.2119034982, .6806995451, .1073969566]), np.array([.0883024619, .2817188376, .6299787005])))
    return np.stack([.2104542553 * l + .793617785 * m - .0040720468 * s, 1.9779984951 * l - 2.428592205 * m + .4505937099 * s, .0259040371 * l + .7827717662 * m - .8086757660 * s], -1)


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    l, m, s = (L + .3963377774 * a + .2158037573 * b) ** 3, (L - .1055613458 * a - .0638541728 * b) ** 3, (L - .0894841775 * a - 1.291485548 * b) ** 3
    lin = np.stack([4.0767416621 * l - 3.3077115913 * m + .2309699292 * s, -1.2684380046 * l + 2.6097574011 * m - .3413193965 * s, -.0041960863 * l - .7034186147 * m + 1.707614701 * s], -1)
    lin = np.clip(lin, 0, 1)
    return (np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - .055) * 255).round().astype(np.uint8)


def region_masks(parse: np.ndarray, erode: dict | None = None) -> dict:
    """Boolean masks per region from one (128,128) parse map: (eroded mask, raw pixel count)."""
    erode = ERODE if erode is None else erode
    out = {}
    for r in REGIONS:
        raw = np.isin(parse, CLASSES[r])
        out[r] = (binary_erosion(raw, iterations=erode[r]) if erode[r] else raw, int(raw.sum()))
    return out


def _two_atoms(px: np.ndarray, wt: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Deterministic k=2 (weighted) clustering of OKLab pixels -> (primary, secondary, secondary weight); flat regions give secondary = primary, weight 0."""
    wt = np.ones(len(px)) if wt is None else wt
    c1 = np.array([np.median(px[:, k]) for k in range(3)])
    d = ((px - c1) ** 2).sum(1)
    c2 = px[int(np.argmax(d))]
    for _ in range(12):
        near2 = ((px - c2) ** 2).sum(1) < ((px - c1) ** 2).sum(1)
        if near2.all() or (~near2).all(): break
        n1, n2 = np.average(px[~near2], 0, wt[~near2]), np.average(px[near2], 0, wt[near2])
        if np.allclose(n1, c1) and np.allclose(n2, c2): break
        c1, c2 = n1, n2
    near2 = ((px - c2) ** 2).sum(1) < ((px - c1) ** 2).sum(1)
    w2 = float(wt[near2].sum() / wt.sum())
    if w2 > 0.5: c1, c2, w2 = c2, c1, 1 - w2
    if w2 < MIN_W2 or 100 * np.linalg.norm(c1 - c2) < FLAT_DE: return c1, c1.copy(), 0.0
    return c1, c2, w2


def region_pixels(lab: np.ndarray, parse: np.ndarray, region: str, erode: dict | None = None) -> tuple[np.ndarray, np.ndarray | None, float]:
    """OKLab pixels of one region (eroded; eyes filtered and chroma-weighted) -> (pixels, weights or None, raw area share)."""
    m, raw = region_masks(parse, erode)[region]
    px, wt = lab[m], None
    if region == "eyes" and len(px):  # drop near-white sclera and near-black lashes/pupil, weight the rest by chroma
        chroma = np.hypot(px[:, 1], px[:, 2]); keep = (px[:, 0] < 0.9) & (px[:, 0] > 0.25) & (chroma > 0.02)
        px, wt = px[keep], chroma[keep]
    return px, wt, raw / parse.size


def extract(images: np.ndarray, parses: np.ndarray, erode: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """images (N,128,128,3) uint8, parses (N,128,128) uint8 -> palette (N,4,8) float32 and valid (N,4) uint8."""
    N = len(images); pal = np.zeros((N, 4, 8), np.float32); valid = np.zeros((N, 4), np.uint8); lab_all = srgb_to_oklab(images)
    for i in range(N):
        for k, r in enumerate(REGIONS):
            px, wt, area = region_pixels(lab_all[i], parses[i], r, erode); pal[i, k, 7] = area
            if len(px) < MIN_PX[r]: continue
            c1, c2, w2 = _two_atoms(px, wt)
            pal[i, k, :7] = [*c1, *c2, w2]; valid[i, k] = 1
    return pal, valid


def _transport(px: np.ndarray, wt: np.ndarray | None, c1: np.ndarray, c2: np.ndarray, w2: float) -> float:
    """Exact capacity-constrained transport cost (dE) of weighted pixels onto atoms c1 (mass 1-w2) and c2 (mass w2)."""
    wt = np.ones(len(px)) if wt is None else wt; wt = wt / wt.sum()
    d1, d2 = 100 * np.linalg.norm(px - c1, axis=1), 100 * np.linalg.norm(px - c2, axis=1)
    order = np.argsort(d2 - d1); cum = np.cumsum(wt[order]); to2 = np.clip((w2 - (cum - wt[order])) / wt[order], 0, 1)  # share of each pixel sent to atom 2
    return float((wt[order] * (to2 * d2[order] + (1 - to2) * d1[order])).sum())


def pixel_distance(images: np.ndarray, parses: np.ndarray, cond_pal: np.ndarray, cond_valid: np.ndarray, erode: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per image and region: transport cost (dE) of the generated region's pixels onto the conditioning atoms -> (dist (N,4), gen_valid (N,4)).
    NaN where the condition is invalid or the generated region is missing (fewer than MIN_PX pixels)."""
    N = len(images); dist = np.full((N, 4), np.nan); gen_valid = np.zeros((N, 4), np.uint8); lab_all = srgb_to_oklab(images)
    for i in range(N):
        for k, r in enumerate(REGIONS):
            px, wt, _ = region_pixels(lab_all[i], parses[i], r, erode)
            if len(px) < MIN_PX[r]: continue
            gen_valid[i, k] = 1
            if cond_valid[i, k] == 1: dist[i, k] = _transport(px, wt, cond_pal[i, k, :3].astype(np.float64), cond_pal[i, k, 3:6].astype(np.float64), float(cond_pal[i, k, 6]))
    return dist, gen_valid


def adherence(images, parses, cond_pal, cond_valid, chance: np.ndarray) -> dict:
    """Per region: mean dE over images valid in the condition and in the generated image; the share of condition-valid regions missing in the generated
    image; and a penalised mean where a missing region scores the chance level (a generator cannot gain by dropping a region)."""
    d, gv = pixel_distance(images, parses, cond_pal, cond_valid); cv = cond_valid == 1; out = {}
    for k, r in enumerate(REGIONS):
        miss = cv[:, k] & (gv[:, k] == 0); dd = d[:, k]; pen = np.where(miss, chance[k], dd)[cv[:, k]]
        out[r] = {"dE": float(np.nanmean(dd)) if np.isfinite(dd).any() else float("nan"), "missing_rate": float(miss.sum() / max(cv[:, k].sum(), 1)),
                  "dE_penalised": float(pen.mean()) if len(pen) else float("nan"), "n_cond_valid": int(cv[:, k].sum())}
    return out


# Mean dE on 3,000 validation images, per region: against their own palette (the floor, which is not 0 because the region has spread) and against another image's
# palette (chance). A generator that ignores the conditioning palette scores near CHANCE; a perfect one scores near FLOOR.
FLOOR = {"skin": 4.62, "hair": 5.09, "eyes": 7.66, "background": 1.87}
CHANCE = {"skin": 11.03, "hair": 24.11, "eyes": 16.00, "background": 15.61}


def score(images, cond_palette, cond_valid, parses=None) -> dict:
    """Palette adherence of generated `images` (N,128,128,3) uint8 to the palettes they were conditioned on (`palette` (N,32) or (N,4,8), `palette_valid` (N,4) of the dataset row).
    `parses`: parse maps of the generated images; by default AnimeSeg computes them (extra `seg`). Returns per region dE, missing_rate and dE_penalised (see `adherence`),
    to be read against FLOOR and CHANCE."""
    from afaces.seg import parse

    p = np.asarray(cond_palette, np.float32).reshape(len(images), 4, 8)
    return adherence(images, parse(images) if parses is None else parses, p, np.asarray(cond_valid), np.array([CHANCE[r] for r in REGIONS]))

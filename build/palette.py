"""Per-region colour palettes in OKLab (dataset field palette.npy). Pure numpy/scipy; the parse maps are anime-seg class ids on the 128 px image.

Layout, 8 values per region (skin, hair, eyes, background -> 32 dims):
    [L1, a1, b1, L2, a2, b2, w2, area]   primary colour, secondary colour and its weight (0 = flat region, secondary == primary), area share of the crop.
Atoms are canonical: k=2 clustering with a deterministic start, the heavier cluster first. Validity (palette_valid) is stored separately."""
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

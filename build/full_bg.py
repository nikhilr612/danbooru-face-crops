"""Full build, stage 3: anime-seg parse of the 128 px crops and the background plainness score.

Usage: .envs/seg/bin/python -I build/full_bg.py TAG     (e.g. g05, s03; needs build_data/crops/TAG_image.npy)
Writes build_data/crops/gBB_parse.npy (N,128,128) uint8 class ids and gBB_bg.npy (N,) float32: fit2_med, the median over background pixels of the
worst-channel residual of a quadratic surface fit (nan if fewer than MIN_PX background pixels survive the erosion).
Parse path = metrics_dev_parse.py. The score is the pilot's fit2_med at 128 px (erosion 2, MIN_PX 75); on the 121 labelled review crops it gives
AUC 0.83 like the 256 px original, and <0.03 keeps 42% of the ok backgrounds and passes 0/44 bad ones (original: 43%, 0/44).
"""
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion

from anime_seg import AnimeSegPipeline

S, ERODE, MIN_PX, BATCH, CH = 128, 2, 75, 8, 512


def fit2(args):
    rgb, par, seed = args
    bg = binary_erosion(par == 0, iterations=ERODE)
    if bg.sum() < MIN_PX:
        return np.nan
    ys, xs = np.nonzero(bg)
    if len(ys) > 6000:
        k = np.random.default_rng(seed).choice(len(ys), 6000, replace=False)
        ys, xs = ys[k], xs[k]
    u, v = xs / S - 0.5, ys / S - 0.5
    y = rgb[ys, xs].astype(np.float32) / 255
    D = np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], 1)
    c = np.linalg.lstsq(D, y, rcond=None)[0]
    return float(np.median(np.abs(y - D @ c).max(1)))


def main(batch: str):
    tag = f"build_data/crops/{batch}"
    X = np.load(f"{tag}_image.npy", mmap_mode="r")
    parse = np.zeros(X.shape[:3], np.uint8)
    MEAN, STD = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda(), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
    model = AnimeSegPipeline.from_dinoV2().to("cuda").model.eval()
    jobs = []
    with Pool(12) as pool, torch.no_grad():
        for i in range(0, len(X), BATCH):
            x = torch.from_numpy(np.asarray(X[i:i + BATCH])).cuda().permute(0, 3, 1, 2).float() / 255
            x = (F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False) - MEAN) / STD
            parse[i:i + BATCH] = F.interpolate(model(x).float(), size=(S, S), mode="bilinear", align_corners=False).argmax(1).cpu().numpy().astype(np.uint8)
            if (i + BATCH) % CH == 0 or i + BATCH >= len(X):
                lo = (i // CH) * CH
                jobs.append(pool.map_async(fit2, [(np.asarray(X[j]), parse[j], j) for j in range(lo, min(i + BATCH, len(X)))]))
            if (i // BATCH) % 500 == 0:
                print(f"{i}/{len(X)}", flush=True)
        bg = np.array([v for j in jobs for v in j.get()], np.float32)
    np.save(f"{tag}_parse.npy", parse)
    np.save(f"{tag}_bg.npy", bg)
    ok = ~np.isnan(bg)
    print(f"batch {batch}: {len(X)} crops, scored {int(ok.sum())}, <0.03: {(bg[ok] < 0.03).mean():.3f}, <0.04: {(bg[ok] < 0.04).mean():.3f} of scored")


if __name__ == "__main__":
    main(sys.argv[1])

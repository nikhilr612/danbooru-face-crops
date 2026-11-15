"""Full build, stage 4: per-crop embeddings and tagger output for the crops that pass the background filter (fit2_med < 0.04).

Usage: .envs/metrics/bin/python -I build/full_feats.py TAG      (e.g. g05; needs TAG_image.npy and TAG_bg.npy)
Writes build_data/feats/TAG_rows.npy (indices into the batch arrays), TAG_dino.npy (n,768 float16, DINOv2-B CLS, 224 px) and
TAG_wd.npy (n,V float16 sigmoid probabilities of the wd-vit-tagger-v3 general tags, 448 px BGR); vocabulary in build_data/feats/wd_general.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm
from huggingface_hub import hf_hub_download
from transformers import AutoModel

BG_MAX, BATCH = 0.04, 100
OUT = Path("build_data/feats")


def main(tag: str):
    OUT.mkdir(exist_ok=True, parents=True)
    X = np.load(f"build_data/crops/{tag}_image.npy", mmap_mode="r")
    bg = np.load(f"build_data/crops/{tag}_bg.npy")
    rows = np.nonzero(bg < BG_MAX)[0]
    csv = pd.read_csv(hf_hub_download("SmilingWolf/wd-vit-tagger-v3", "selected_tags.csv"))
    gen = np.nonzero(csv.category.to_numpy() == 0)[0]
    if not (OUT / "wd_general.json").exists():
        json.dump(csv.name.to_numpy()[gen].tolist(), open(OUT / "wd_general.json", "w"))
    tagger = timm.create_model("hf_hub:SmilingWolf/wd-vit-tagger-v3", pretrained=True).cuda().eval().half()
    dino = AutoModel.from_pretrained("facebook/dinov2-base").cuda().eval().half()
    im_m, im_s = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1).cuda().half(), torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1).cuda().half()
    D, W = np.zeros((len(rows), 768), np.float16), np.zeros((len(rows), len(gen)), np.float16)
    with torch.no_grad():
        for i in range(0, len(rows), BATCH):
            x = torch.from_numpy(np.stack([X[j] for j in rows[i:i + BATCH]])).cuda().permute(0, 3, 1, 2).half() / 255
            a = F.interpolate(x, size=(448, 448), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1).flip(1)
            W[i:i + BATCH] = torch.sigmoid(tagger(((a - 0.5) / 0.5))[:, torch.from_numpy(gen).cuda()].float()).half().cpu().numpy()
            b = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
            D[i:i + BATCH] = dino(pixel_values=(b - im_m) / im_s).pooler_output.float().cpu().numpy().astype(np.float16)
    np.save(OUT / f"{tag}_rows.npy", rows), np.save(OUT / f"{tag}_dino.npy", D), np.save(OUT / f"{tag}_wd.npy", W)
    print(f"{tag}: {len(rows)} of {len(bg)} crops pass bg<{BG_MAX}")


if __name__ == "__main__":
    main(sys.argv[1])

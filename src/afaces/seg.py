"""Parse maps of 128 px images with AnimeSeg (extra `seg`: `anime-seg`, which has no stated licence; see LICENSES.md), the annotator that produced the dataset's `parse` field.

Same path as the dataset build: bilinear upsample to 512, the model's logits resized bilinearly to 128, argmax. Class ids: 0 background, 1 skin, 2 face, 3 hair, 4/5 eyes,
6/7 brows, 8 nose, 9 mouth, 10 clothes, 11 thin hair.
"""
import numpy as np


def parse(images, device: str | None = None, batch: int = 16) -> np.ndarray:
    """(N, 128, 128, 3) uint8 -> (N, 128, 128) uint8 class ids."""
    try:
        import torch
        import torch.nn.functional as F
        from anime_seg import AnimeSegPipeline
    except ImportError as e:
        raise ImportError("afaces.seg needs the `seg` extra (pip install 'afaces[seg]'), which installs anime-seg and torch") from e
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = (torch.tensor(v).view(1, 3, 1, 1).to(dev) for v in ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    model = AnimeSegPipeline.from_dinoV2().to(dev).model.eval()
    x = np.asarray(images)
    out = np.zeros(x.shape[:3], np.uint8)
    with torch.no_grad():
        for i in range(0, len(x), batch):
            t = torch.from_numpy(x[i:i + batch]).to(dev).permute(0, 3, 1, 2).float() / 255
            t = (F.interpolate(t, size=(512, 512), mode="bilinear", align_corners=False) - mean) / std
            out[i:i + batch] = F.interpolate(model(t).float(), size=(128, 128), mode="bilinear", align_corners=False).argmax(1).cpu().numpy().astype(np.uint8)
    return out

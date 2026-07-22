"""Image-quality metrics: PSNR, SSIM, LPIPS — optionally restricted to an
object mask so the score measures object reconstruction, not the (uniform,
trivially-correct) background.

Operates on float arrays/tensors in [0, 1], shape (H, W, 3). A `mask` (H, W or
H, W, 1 in [0, 1]) restricts PSNR to object pixels and crops SSIM/LPIPS to the
object bounding box. LPIPS is loaded lazily and cached; if unavailable it is
skipped (returns None).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np


def _to_np(img) -> np.ndarray:
    if hasattr(img, "detach"):  # torch tensor
        img = img.detach().cpu().numpy()
    img = np.asarray(img, dtype=np.float32)
    return np.clip(img, 0.0, 1.0)


def _mask_bool(mask, hw) -> Optional[np.ndarray]:
    if mask is None:
        return None
    m = _to_np(mask)
    if m.ndim == 3:
        m = m[..., 0]
    if m.shape != hw:
        from PIL import Image
        m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize(
            (hw[1], hw[0])), dtype=np.float32) / 255.0
    return m > 0.5


def _bbox(m: np.ndarray):
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def psnr(pred, gt, mask=None) -> float:
    pred, gt = _to_np(pred), _to_np(gt)
    mb = _mask_bool(mask, pred.shape[:2])
    if mb is None:
        mse = float(np.mean((pred - gt) ** 2))
    elif mb.sum() == 0:
        return 99.0
    else:
        mse = float(np.mean((pred[mb] - gt[mb]) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(-10.0 * np.log10(mse))


def ssim(pred, gt, mask=None) -> float:
    from skimage.metrics import structural_similarity as _ssim
    pred, gt = _to_np(pred), _to_np(gt)
    mb = _mask_bool(mask, pred.shape[:2])
    if mb is not None:
        bb = _bbox(mb)
        if bb:
            x0, y0, x1, y1 = bb
            if (x1 - x0) >= 7 and (y1 - y0) >= 7:
                pred, gt = pred[y0:y1, x0:x1], gt[y0:y1, x0:x1]
    return float(_ssim(gt, pred, channel_axis=2, data_range=1.0))


@lru_cache(maxsize=1)
def _lpips_model(net: str = "alex"):
    try:
        import lpips
        import torch
        model = lpips.LPIPS(net=net)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        return model
    except Exception:
        return None


def lpips(pred, gt, mask=None) -> Optional[float]:
    model = _lpips_model()
    if model is None:
        return None
    import torch
    pred, gt = _to_np(pred), _to_np(gt)
    mb = _mask_bool(mask, pred.shape[:2])
    if mb is not None:
        bb = _bbox(mb)
        if bb:
            x0, y0, x1, y1 = bb
            # AlexNet needs a minimum size; only crop when the bbox is big enough
            if (x1 - x0) >= 64 and (y1 - y0) >= 64:
                pred, gt = pred[y0:y1, x0:x1], gt[y0:y1, x0:x1]

    def prep(x):
        t = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None] * 2 - 1
        return t.cuda() if torch.cuda.is_available() else t

    with torch.no_grad():
        return float(model(prep(pred), prep(gt)).item())


def all_metrics(pred, gt, want=("psnr", "ssim", "lpips"), mask=None) -> dict:
    out = {}
    if "psnr" in want:
        out["psnr"] = psnr(pred, gt, mask)
    if "ssim" in want:
        out["ssim"] = ssim(pred, gt, mask)
    if "lpips" in want:
        out["lpips"] = lpips(pred, gt, mask)
    return out

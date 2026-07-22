"""Robust object segmentation: coarse rembg mask -> SAM refinement.

rembg (isnet-general-use) reliably *locates* the object but under-segments hard
angles (top-down views where the object blends with the table). SAM, prompted
with positive points sampled ON the coarse mask + negative points at the image
corners, and with the output mask selected by overlap with the coarse mask,
recovers the complete object and never drifts to the background. A final
fill-holes + largest-connected-component pass cleans stray speckle.

Heavy deps (torch/transformers/rembg/scipy) imported lazily.
"""
from __future__ import annotations

import numpy as np


def _iou(a, b) -> float:
    a, b = a > 0.5, b > 0.5
    return float(np.logical_and(a, b).sum()) / (np.logical_or(a, b).sum() + 1e-6)


def _prompt_points(coarse, W, H, n_pos=6):
    ys, xs = np.where(coarse > 0.5)
    if len(xs):
        sel = np.linspace(0, len(xs) - 1, min(n_pos, len(xs))).astype(int)
        pos = [[int(xs[i]), int(ys[i])] for i in sel]
    else:
        pos = [[W // 2, H // 2]]
    neg = [[8, 8], [W - 8, 8], [8, H - 8], [W - 8, H - 8],
           [W // 2, 8], [W // 2, H - 8]]
    return pos + neg, [1] * len(pos) + [0] * len(neg)


def _cleanup(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes and keep the largest connected component."""
    from scipy import ndimage
    m = mask > 0.5
    lbl, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lbl, range(1, n + 1))
        m = lbl == (int(np.argmax(sizes)) + 1)
    m = ndimage.binary_fill_holes(m)
    return m.astype(np.float32)


class Segmenter:
    def __init__(self, coarse_model="isnet-general-use",
                 sam_model="facebook/sam-vit-base", device=None):
        self.coarse_model = coarse_model
        self.sam_model = sam_model
        self._device = device
        self._sam = self._proc = self._coarse = None

    def _load(self):
        if self._sam is None:
            import torch
            from rembg import new_session
            from transformers import SamModel, SamProcessor
            self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._sam = SamModel.from_pretrained(self.sam_model).to(self._device)
            self._proc = SamProcessor.from_pretrained(self.sam_model)
            self._coarse = new_session(self.coarse_model)

    def coarse_alpha(self, pil_img) -> np.ndarray:
        from rembg import remove
        cut = remove(pil_img.convert("RGBA"), session=self._coarse)
        return np.asarray(cut.split()[-1], dtype=np.float32) / 255.0

    def segment(self, pil_img):
        """Return the object alpha mask as a PIL 'L' image (0/255)."""
        import torch
        from PIL import Image
        self._load()
        img = pil_img.convert("RGB")
        W, H = img.size
        coarse = self.coarse_alpha(img)
        pts, labels = _prompt_points(coarse, W, H)
        inp = self._proc(img, input_points=[[pts]], input_labels=[[labels]],
                         return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._sam(**inp)
        masks = self._proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), inp["original_sizes"].cpu(),
            inp["reshaped_input_sizes"].cpu())[0][0]          # (3, H, W)
        cand = [masks[k].numpy().astype(np.float32) for k in range(masks.shape[0])]
        best = max(cand, key=lambda m: _iou(m, coarse))
        best = _cleanup(best)
        return Image.fromarray((best * 255).astype(np.uint8), mode="L")

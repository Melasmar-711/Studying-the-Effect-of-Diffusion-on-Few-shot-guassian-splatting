"""Stable Zero123 novel-view synthesis of a segmented object.

Loads the community Zero1to3 pipeline with `kxic/zero123-xl` weights by
constructing it from explicit components — diffusers 0.27.2's auto-loader can't
resolve the repo's custom `CCProjection` component. `rembg` isolates the object
(Zero123 expects an object on a clean background). Used for preview now; can back
a GS augmentation strategy later once the pose/background integration is decided.

Heavy deps (torch/diffusers/transformers/rembg/kornia) are imported lazily.
"""
from __future__ import annotations

import importlib.util
import sys
import urllib.request
from pathlib import Path

# community pipeline pinned to our diffusers version for API compatibility
PIPE_URL = ("https://raw.githubusercontent.com/huggingface/diffusers/"
            "v0.27.2/examples/community/pipeline_zero1to3.py")
_CACHE = Path(__file__).resolve().parents[2] / ".cache_zero123_pipe"


def isolate_object(pil_img, size: int = 256, pad: float = 1.25):
    """rembg -> composite on white -> tight square crop -> resize. Returns
    (image, segmented_bool)."""
    from PIL import Image
    try:
        from rembg import remove
        cut = remove(pil_img.convert("RGBA"))
        bbox = cut.getbbox()
        white = Image.new("RGBA", cut.size, (255, 255, 255, 255))
        comp = Image.alpha_composite(white, cut).convert("RGB")
        seg = True
    except Exception:
        comp, bbox, seg = pil_img.convert("RGB"), None, False
    if bbox:
        x0, y0, x1, y1 = bbox
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = max(x1 - x0, y1 - y0) * pad / 2
        crop = comp.crop((int(cx - half), int(cy - half),
                          int(cx + half), int(cy + half)))
    else:
        s = min(comp.size); crop = comp.crop((0, 0, s, s))
    canvas = Image.new("RGB", crop.size, (255, 255, 255))
    canvas.paste(crop, (0, 0))
    return canvas.resize((size, size)), seg


def cond_from_mask(img_path, mask_path, size: int = 256, pad: float = 1.3):
    """Object-on-white 256px conditioning from a frame + its mask (clean, exact —
    no rembg needed since we already have the object mask)."""
    import numpy as np
    from PIL import Image
    img = np.asarray(Image.open(img_path).convert("RGB")).astype(np.uint8)
    m = np.asarray(Image.open(mask_path).convert("L")) > 127
    comp = np.where(m[..., None], img, 255).astype(np.uint8)
    ys, xs = np.where(m)
    if len(xs) == 0:
        return Image.fromarray(comp).resize((size, size))
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) * pad / 2
    x0, y0 = int(cx - half), int(cy - half)
    crop = Image.fromarray(comp).crop((x0, y0, x0 + int(2 * half), y0 + int(2 * half)))
    return crop.resize((size, size))


def rekey_on_black(gen_white, dst_wh, bbox_xywh):
    """Re-key a Zero123 object-on-WHITE output onto BLACK at a target bbox, on a
    full ``dst_wh`` canvas — so it matches the training domain (object-on-black)
    and the object's scale/position in the source frame. Returns a PIL RGB image.

    ``bbox_xywh`` = (cx, cy, w, h) of the source object box in dst-canvas pixels.
    """
    import numpy as np
    from PIL import Image
    W, H = dst_wh
    a = np.asarray(gen_white.convert("RGB")).astype(np.int16)
    # object = pixels not near-white
    obj = (255 - a).max(2) > 22
    ys, xs = np.where(obj)
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    if len(xs) == 0:
        return canvas
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    crop = gen_white.convert("RGB").crop((x0, y0, x1 + 1, y1 + 1))
    mask = Image.fromarray((obj[y0:y1 + 1, x0:x1 + 1] * 255).astype("uint8"), "L")
    cx, cy, bw, bh = bbox_xywh
    s = max(bw, bh) / max(crop.width, crop.height)
    nw, nh = max(1, int(crop.width * s)), max(1, int(crop.height * s))
    crop = crop.resize((nw, nh)); mask = mask.resize((nw, nh))
    canvas.paste(crop, (int(cx - nw / 2), int(cy - nh / 2)), mask)
    return canvas


def _import_module():
    _CACHE.mkdir(exist_ok=True)
    pf = _CACHE / "pipeline_zero1to3.py"
    if not pf.exists():
        urllib.request.urlretrieve(PIPE_URL, pf)
    spec = importlib.util.spec_from_file_location("pipeline_zero1to3", pf)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_zero1to3"] = mod       # register for internal refs
    spec.loader.exec_module(mod)
    return mod


class Zero123Generator:
    def __init__(self, model_id: str = "kxic/zero123-xl"):
        self.model_id = model_id
        self._pipe = None

    def pipe(self):
        if self._pipe is None:
            import torch
            from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
            from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
            mod = _import_module()
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            m = self.model_id
            vae = AutoencoderKL.from_pretrained(m, subfolder="vae", torch_dtype=dtype)
            unet = UNet2DConditionModel.from_pretrained(m, subfolder="unet", torch_dtype=dtype)
            ie = CLIPVisionModelWithProjection.from_pretrained(
                m, subfolder="image_encoder", torch_dtype=dtype)
            fe = CLIPImageProcessor.from_pretrained(m, subfolder="feature_extractor")
            sch = DDIMScheduler.from_pretrained(m, subfolder="scheduler")
            cc = mod.CCProjection.from_pretrained(m, subfolder="cc_projection", torch_dtype=dtype)
            p = mod.Zero1to3StableDiffusionPipeline(
                vae=vae, image_encoder=ie, feature_extractor=fe, unet=unet,
                scheduler=sch, cc_projection=cc,
                requires_safety_checker=False, safety_checker=None)
            if torch.cuda.is_available():
                p.to("cuda")
                try:
                    p.enable_attention_slicing()
                except Exception:
                    pass
            self._pipe = p
        return self._pipe

    def novel_view(self, cond, elevation, azimuth, distance=0.0, steps=50, seed=0):
        """One novel view. `poses=[elevation, azimuth, distance]` (relative deg)."""
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        gen = torch.Generator(dev).manual_seed(seed)
        return self.pipe()(
            input_imgs=cond, prompt_imgs=cond,
            poses=[float(elevation), float(azimuth), float(distance)],
            height=256, width=256, guidance_scale=3.0,
            num_inference_steps=steps, generator=gen).images[0]

    def free(self):
        import torch
        self._pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

#!/usr/bin/env python
"""Feasibility spike for Stable-Zero123 novel-view synthesis of the object.

Isolates the object (rembg -> white bg -> square 256), loads the zero123
community pipeline, and renders a few relative viewpoints. Saves a panel to
results/previews/zero123_spike.png so we can judge whether it's worth wiring in
as a real augmentation strategy.

    python scripts/spike_zero123.py --scene scene01 --n 5 --src 0
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from gsfewshot import load_config
from gsfewshot.splits import load_split_frames

OUT = Path(__file__).resolve().parents[1] / "results" / "previews"


def isolate_object(pil_img, size=256, pad=1.25):
    """rembg -> composite on white -> tight square crop -> resize."""
    from PIL import Image
    try:
        from rembg import remove
        cut = remove(pil_img.convert("RGBA"))
        bbox = cut.getbbox()
        white = Image.new("RGBA", cut.size, (255, 255, 255, 255))
        comp = Image.alpha_composite(white, cut).convert("RGB")
        seg = True
    except Exception as e:
        print(f"[spike] rembg unavailable ({e}); using full frame")
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


PIPE_URL = ("https://raw.githubusercontent.com/huggingface/diffusers/"
            "v0.27.2/examples/community/pipeline_zero1to3.py")


def _import_zero123_module():
    """Download + import the community pipeline module (defines CCProjection +
    Zero1to3StableDiffusionPipeline). Registered in sys.modules under the name
    the weights' model_index expects."""
    import importlib.util
    import sys
    import urllib.request
    pdir = Path(__file__).resolve().parents[1] / ".cache_zero123_pipe"
    pdir.mkdir(exist_ok=True)
    pfile = pdir / "pipeline_zero1to3.py"
    if not pfile.exists():
        print(f"[spike] downloading community pipeline -> {pfile}")
        urllib.request.urlretrieve(PIPE_URL, pfile)
    spec = importlib.util.spec_from_file_location("pipeline_zero1to3", pfile)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_zero1to3"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_pipe(model_id):
    """Build the Zero1to3 pipeline by loading each component explicitly (avoids
    diffusers' auto model_index resolution, which can't find CCProjection)."""
    import torch
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

    mod = _import_zero123_module()
    CCProjection = mod.CCProjection
    Zero1to3 = mod.Zero1to3StableDiffusionPipeline
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"[spike] loading components from {model_id} ...")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        model_id, subfolder="image_encoder", torch_dtype=dtype)
    feat = CLIPImageProcessor.from_pretrained(model_id, subfolder="feature_extractor")
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    cc = CCProjection.from_pretrained(model_id, subfolder="cc_projection", torch_dtype=dtype)

    pipe = Zero1to3(vae=vae, image_encoder=image_encoder, feature_extractor=feat,
                    unet=unet, scheduler=scheduler, cc_projection=cc,
                    requires_safety_checker=False, safety_checker=None)
    if torch.cuda.is_available():
        pipe.to("cuda")
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
    return pipe


def gen_view(pipe, cond, elev, azim, dist, steps, seed):
    import torch
    gen = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
    # the community pipeline takes poses=[elevation, azimuth, distance]
    for kwargs in (
        dict(input_imgs=cond, prompt_imgs=cond, poses=[float(elev), float(azim), float(dist)],
             height=256, width=256, guidance_scale=3.0, num_inference_steps=steps, generator=gen),
        dict(input_imgs=cond, prompt_imgs=cond, poses=[[float(elev), float(azim), float(dist)]],
             height=256, width=256, guidance_scale=3.0, num_inference_steps=steps, generator=gen),
    ):
        try:
            return pipe(**kwargs).images[0]
        except Exception as e:
            print(f"    call form failed: {type(e).__name__}: {str(e)[:160]}")
    raise SystemExit("zero123 __call__ signature mismatch — inspect pipeline")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene01")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--src", type=int, default=0, help="which n-subset frame")
    ap.add_argument("--model", default="kxic/zero123-xl",
                    help="diffusers-format zero123 weights for the community pipeline")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    from PIL import Image
    cfg = load_config(args.config)
    d = cfg.diffusion
    _, frames = load_split_frames(args.scene, f"n{args.n}")
    fr = frames[args.src]
    src = Image.open(fr["file_path"]).convert("RGB")
    cond, seg = isolate_object(src, 256)
    OUT.mkdir(parents=True, exist_ok=True)
    cond.save(OUT / "zero123_cond.png")
    print(f"[spike] source={Path(fr['file_path']).name} segmented={seg}")

    pipe = load_pipe(args.model)
    azims = d.get("zero123_azimuths", [-30, 30, -15, 15])
    elevs = d.get("zero123_elevations", [0, 0, 10, -10])
    steps = int(d.get("zero123_steps", 50))

    tiles = [("cond (segmented)", cond)]
    for i, (az, el) in enumerate(zip(azims, elevs)):
        print(f"[spike] view {i}: azimuth={az} elevation={el}")
        img = gen_view(pipe, cond, el, az, 0.0, steps, seed=i)
        img.save(OUT / f"zero123_view_{i}_az{az}_el{el}.png")
        tiles.append((f"az{az} el{el}", img))

    # panel
    H = 256
    W = sum(t[1].width for t in tiles) + 6 * (len(tiles) - 1)
    from PIL import ImageDraw
    panel = Image.new("RGB", (W, H + 24), (255, 255, 255))
    x = 0
    for label, im in tiles:
        panel.paste(im.resize((H, H)), (x, 24))
        ImageDraw.Draw(panel).text((x + 4, 6), label, fill=(0, 0, 0))
        x += H + 6
    panel.save(OUT / "zero123_spike.png")
    print(f"\n[spike] panel -> {OUT/'zero123_spike.png'}")


if __name__ == "__main__":
    main()

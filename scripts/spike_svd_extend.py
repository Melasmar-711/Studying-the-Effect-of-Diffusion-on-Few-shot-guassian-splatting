#!/usr/bin/env python
"""Spike: extend the real orbit with Stable Video Diffusion (img2vid).

Gate 1 (this script): condition on a real end-of-orbit frame, generate a short
continuation, and SAVE frames so we can eyeball whether SVD keeps the tank RIGID
(a plausible orbit continuation COLMAP could register) or morphs/zooms it.

8 GB-friendly: fp16 + model CPU offload + small decode chunks + 14 frames.
"""
import argparse
from pathlib import Path
import torch
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video

MODEL = "stabilityai/stable-video-diffusion-img2vid"  # 14 frames, lighter than -xt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="scenes/scene02/frames_raw/frame_00319.jpg")
    ap.add_argument("--out", default="results/spikes/svd")
    ap.add_argument("--motion", type=int, nargs="+", default=[127, 180])
    ap.add_argument("--frames", type=int, default=14)
    ap.add_argument("--width", type=int, default=768)     # 16:9, divisible by 8
    ap.add_argument("--height", type=int, default=432)
    ap.add_argument("--fps", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    img = Image.open(args.src).convert("RGB").resize((args.width, args.height))
    tag = Path(args.src).stem

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        MODEL, torch_dtype=torch.float16, variant="fp16")
    pipe.enable_model_cpu_offload()
    # 8 GB: activations (not weights) are the bottleneck -> slice attention and
    # chunk the big feed-forward so peak VRAM stays bounded.
    pipe.enable_attention_slicing()
    pipe.unet.enable_forward_chunking()

    for mb in args.motion:
        gen = torch.manual_seed(args.seed)
        frames = pipe(img, num_frames=args.frames, decode_chunk_size=1,
                      width=args.width, height=args.height,
                      motion_bucket_id=mb, noise_aug_strength=0.02,
                      generator=gen).frames[0]
        clip_dir = out / f"{tag}_mb{mb}"; clip_dir.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames):
            f.save(clip_dir / f"gen_{i:03d}.png")
        export_to_video(frames, str(out / f"{tag}_mb{mb}.mp4"), fps=args.fps)
        print(f"[svd] {tag} mb={mb}: {len(frames)} frames -> {clip_dir}")


if __name__ == "__main__":
    main()

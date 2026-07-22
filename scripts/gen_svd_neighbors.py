#!/usr/bin/env python
"""Stage 1 of the SVD augmentation arm.

Seed a short Stable-Video-Diffusion clip from EACH few-shot training view, so
every real view gets a small halo of nearby frames that COLMAP can later
register onto the orbit (validated: 27/28 register, poses on the ring). Multi-seed
is the lever (one clip only travels ~3-9 deg of azimuth before it morphs).

Segmented dataset is 1-indexed (frame_000K.png); raw frames are 0-indexed, so the
seed for training frame frame_000K.png is raw frame_000(K-1).jpg.
"""
import argparse, json
from pathlib import Path
import torch
from PIL import Image
from diffusers import StableVideoDiffusionPipeline

MODEL = "stabilityai/stable-video-diffusion-img2vid"
RAW = Path("scenes/scene02/frames_raw")


def seed_raw_for(seg_name):
    k = int(Path(seg_name).stem.split("_")[1])      # frame_00320 -> 320
    return RAW / f"frame_{k-1:05d}.jpg"              # -> raw frame_00319.jpg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="data/splits/scene02/n20/transforms.json")
    ap.add_argument("--out", default="data/svd/scene02/gen_raw")
    ap.add_argument("--motion", type=int, default=127)
    ap.add_argument("--frames", type=int, default=14)
    ap.add_argument("--width", type=int, default=896)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train = json.load(open(args.split))["frames"]
    seeds = []
    for f in train:
        raw = seed_raw_for(f["file_path"])
        if raw.exists():
            seeds.append((Path(f["file_path"]).stem, raw))
        else:
            print(f"[warn] no raw for {f['file_path']} -> {raw}")
    print(f"[gen] {len(seeds)} seed frames")

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        MODEL, torch_dtype=torch.float16, variant="fp16")
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()
    pipe.unet.enable_forward_chunking()

    out = Path(args.out)
    manifest = []
    for tag, raw in seeds:
        clip_dir = out / tag; clip_dir.mkdir(parents=True, exist_ok=True)
        img = Image.open(raw).convert("RGB").resize((args.width, args.height))
        gen = torch.manual_seed(args.seed)
        frames = pipe(img, num_frames=args.frames, decode_chunk_size=1,
                      width=args.width, height=args.height,
                      motion_bucket_id=args.motion, noise_aug_strength=0.02,
                      generator=gen).frames[0]
        for i, fr in enumerate(frames):
            fr.save(clip_dir / f"gen_{i:03d}.png")
        manifest.append({"seed_train": tag, "seed_raw": str(raw), "n": len(frames)})
        print(f"[gen] {tag} <- {raw.name}: {len(frames)} frames")
    out.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(out / "manifest.json", "w"), indent=2)
    print(f"[gen] DONE {len(manifest)} clips -> {out}")


if __name__ == "__main__":
    main()

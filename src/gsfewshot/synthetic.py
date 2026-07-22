"""Diffusion-based view augmentation (SD 1.5 stack, 8 GB friendly).

Three strategies from the plan, each producing images *only* from the limited
real subset and linking every synthetic image back to its source real view:

  * outpaint  — widen the field of view (paste source in centre, inpaint border)
  * inpaint   — fill an occlusion mask (simulate/complete missing regions)
  * guided    — ControlNet (canny/depth) at an interpolated nearby pose

Poses/intrinsics for synthetic views are approximate by design (the plan asks
for "approximate nearby viewpoints"); the exact geometry only matters for the
held-out *real* test set used in evaluation. Heavy deps (torch/diffusers/cv2)
are imported lazily so the rest of the package stays light.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config, SYNTHETIC_DIR
from .splits import load_split_frames


# --------------------------------------------------------------------------- #
# Pose / intrinsics helpers
# --------------------------------------------------------------------------- #
def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    d = float(np.dot(q0, q1))
    if d < 0:
        q1 = -q1; d = -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th0 = np.arccos(d); th = th0 * t
    q2 = q1 - q0 * d
    q2 /= np.linalg.norm(q2)
    return q0 * np.cos(th) + q2 * np.sin(th)


def interpolate_pose(m0: list, m1: list, t: float = 0.5) -> list:
    """SLERP rotation + lerp translation between two 4x4 camera-to-world mats."""
    A, B = np.array(m0), np.array(m1)
    q = _slerp(_mat_to_quat(A[:3, :3]), _mat_to_quat(B[:3, :3]), t)
    R = _quat_to_mat(q)
    trans = (1 - t) * A[:3, 3] + t * B[:3, 3]
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = trans
    return M.tolist()


def _cam_center(m: list) -> np.ndarray:
    return np.array(m)[:3, 3]


def _nearest_neighbor(frames: list[dict], idx: int) -> int:
    c = _cam_center(frames[idx]["transform_matrix"])
    best, bd = idx, 1e18
    for j, fr in enumerate(frames):
        if j == idx:
            continue
        d = float(np.linalg.norm(_cam_center(fr["transform_matrix"]) - c))
        if d < bd:
            bd, best = d, j
    return best


def _scaled_intrinsics(meta: dict, inner_frac: float = 1.0) -> dict:
    """Full-resolution intrinsics for a synthetic view.

    assemble.py resizes synthetic images up to the real (W, H) so nerfstudio sees
    one global image size, so the stored intrinsics are full-res; the source FoV
    is preserved (the generated image was the same FoV, just lower resolution).
    `inner_frac` < 1 (outpaint) widens the FoV -> shorter focal, recentred.
    """
    W, H = meta["w"], meta["h"]
    fl_x = meta["fl_x"] * inner_frac
    fl_y = meta["fl_y"] * inner_frac
    if inner_frac < 1.0:                 # outpaint: recentred wider canvas
        cx, cy = W / 2.0, H / 2.0
    else:                                # same FoV as the source view
        cx, cy = meta["cx"], meta["cy"]
    return {"w": W, "h": H, "fl_x": fl_x, "fl_y": fl_y, "cx": cx, "cy": cy}


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #
class SyntheticGenerator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.d = cfg.diffusion
        self._inpaint = None
        self._controlnet = None
        self._controlnet_i2i = None
        self._z123 = None
        self._orbit = None

    def free_pipes(self):
        """Drop cached pipelines and free VRAM (useful between strategies)."""
        import torch
        self._inpaint = self._controlnet = self._controlnet_i2i = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- lazy pipeline builders ------------------------------------------------
    def _torch(self):
        import torch
        return torch

    def _from_pretrained(self, cls, model_id, **kw):
        """Prefer the smaller fp16 safetensors variant; fall back to default."""
        import torch
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        try:
            return cls.from_pretrained(model_id, torch_dtype=dtype,
                                       variant="fp16", use_safetensors=True, **kw)
        except Exception:
            return cls.from_pretrained(model_id, torch_dtype=dtype, **kw)

    def inpaint_pipe(self):
        if self._inpaint is None:
            from diffusers import StableDiffusionInpaintPipeline
            p = self._from_pretrained(StableDiffusionInpaintPipeline,
                                      self.d["inpaint_model"], safety_checker=None)
            self._inpaint = self._prep(p)
        return self._inpaint

    def controlnet_pipe(self):
        if self._controlnet is None:
            from diffusers import (StableDiffusionControlNetPipeline,
                                   ControlNetModel)
            which = self.d.get("guided_control", "canny")
            cn_id = self.d["controlnet_canny"] if which == "canny" else self.d["controlnet_depth"]
            cn = self._from_pretrained(ControlNetModel, cn_id)
            p = self._from_pretrained(StableDiffusionControlNetPipeline,
                                      self.d["base_model"], controlnet=cn,
                                      safety_checker=None)
            self._controlnet = self._prep(p)
        return self._controlnet

    def controlnet_img2img_pipe(self):
        """ControlNet in img2img mode: start from the REAL image and let the
        control map only steer it (low `strength`) -> preserves the object's
        appearance instead of hallucinating it, fixing the 'weird' text2img look.
        """
        if self._controlnet_i2i is None:
            from diffusers import (StableDiffusionControlNetImg2ImgPipeline,
                                   ControlNetModel)
            which = self.d.get("guided_control", "canny")
            cn_id = self.d["controlnet_canny"] if which == "canny" else self.d["controlnet_depth"]
            cn = self._from_pretrained(ControlNetModel, cn_id)
            p = self._from_pretrained(StableDiffusionControlNetImg2ImgPipeline,
                                      self.d["base_model"], controlnet=cn,
                                      safety_checker=None)
            self._controlnet_i2i = self._prep(p)
        return self._controlnet_i2i

    def _prep(self, pipe):
        import torch
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            pipe.enable_attention_slicing()
            try:
                pipe.enable_vae_slicing()
            except Exception:
                pass
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def _generator(self, seed: int):
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.Generator(device=dev).manual_seed(seed)

    # -- control-image helpers -------------------------------------------------
    def _canny(self, pil_img):
        import cv2
        arr = np.array(pil_img)
        edges = cv2.Canny(arr, 100, 200)
        edges = np.stack([edges] * 3, axis=-1)
        from PIL import Image
        return Image.fromarray(edges)

    def _depth(self, pil_img):
        from controlnet_aux import MidasDetector
        if not hasattr(self, "_midas"):
            self._midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
        return self._midas(pil_img)

    def _control_image(self, pil_img):
        which = self.d.get("guided_control", "canny")
        return self._canny(pil_img) if which == "canny" else self._depth(pil_img)

    # -- strategies ------------------------------------------------------------
    def _load_source(self, file_path: str, size):
        from PIL import Image
        return Image.open(file_path).convert("RGB").resize(size)

    def _object_focus_box(self, mask_root, src_fr, gw, gh, pad=0.1):
        """Object bbox (in gen resolution) from its mask, so inpaint degrades the
        object rather than empty background. None if no mask available."""
        if mask_root is None:
            return None
        from PIL import Image
        mp = mask_root / Path(src_fr["file_path"]).name
        if not mp.exists():
            return None
        m = np.asarray(Image.open(mp).convert("L").resize((gw, gh)),
                       dtype=np.float32) / 255.0
        ys, xs = np.where(m > 0.5)
        if len(xs) == 0:
            return None
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        px, py = int((x1 - x0) * pad), int((y1 - y0) * pad)
        return (max(0, x0 - px), max(0, y0 - py),
                min(gw, x1 + px), min(gh, y1 + py))

    def gen_outpaint(self, src_img, prompt, seed):
        """Widen the FoV. The border is seeded with a BLURRED, upscaled copy of
        the real view (not flat grey) so diffusion extends coherent scene context
        rather than hallucinating from nothing."""
        from PIL import Image, ImageDraw, ImageFilter
        gw, gh = int(self.d["gen_width"]), int(self.d["gen_height"])
        blur = float(self.d.get("manip_blur", 18))
        inner = 1.0 - float(self.d["outpaint_expand"])
        iw, ih = int(gw * inner), int(gh * inner)
        inner_img = src_img.resize((iw, ih))
        # real image, manipulated: full-canvas blurred context + sharp centre
        canvas = src_img.resize((gw, gh)).filter(ImageFilter.GaussianBlur(blur))
        ox, oy = (gw - iw) // 2, (gh - ih) // 2
        canvas.paste(inner_img, (ox, oy))
        mask = Image.new("L", (gw, gh), 255)          # 255 = repaint (border)
        ImageDraw.Draw(mask).rectangle([ox, oy, ox + iw, oy + ih], fill=0)
        feather = float(self.d.get("outpaint_feather", 0))
        if feather > 0:                               # blend the inner/outer seam
            mask = mask.filter(ImageFilter.GaussianBlur(feather))
        out = self.inpaint_pipe()(
            prompt=prompt, negative_prompt=self.d.get("negative_prompt"),
            image=canvas, mask_image=mask,
            strength=float(self.d.get("outpaint_strength", 1.0)),
            width=gw, height=gh, num_inference_steps=int(self.d["steps"]),
            guidance_scale=float(self.d["guidance_scale"]),
            generator=self._generator(seed)).images[0]
        return out, inner

    def build_inpaint_input(self, base, seed, mode=None, focus_box=None):
        """Degrade the real image, returning (manipulated_image, repaint_mask).

        mode: "blur"   -> soft-occlude one region (Gaussian blur)
              "remove" -> cut one region to black
              "spots"  -> scatter several black spots across the frame
        focus_box: (x0,y0,x1,y1) to confine the degradation (e.g. the object's
            bounding box) so it lands ON the object, not on empty background.
        The mask marks exactly the degraded pixels for the inpainter to restore.
        """
        from PIL import Image, ImageDraw, ImageFilter
        gw, gh = base.size
        fx0, fy0, fx1, fy1 = focus_box if focus_box else (0, 0, gw, gh)
        mode = mode or self.d.get("inpaint_manip", "blur")
        blur = float(self.d.get("manip_blur", 18))
        rng = np.random.default_rng(seed)
        manipulated = base.copy()
        mask = Image.new("L", (gw, gh), 0)
        mdraw = ImageDraw.Draw(manipulated)
        kdraw = ImageDraw.Draw(mask)

        def randint(lo, hi):
            return int(rng.integers(lo, max(lo + 1, hi)))

        if mode == "spots":
            k = int(self.d.get("spots_count", 7))
            r = int(min(gw, gh) * float(self.d.get("spots_size", 0.09)))
            for _ in range(k):
                cx = randint(max(r, fx0), min(gw - r, fx1))
                cy = randint(max(r, fy0), min(gh - r, fy1))
                box = (cx - r, cy - r, cx + r, cy + r)
                mdraw.ellipse(box, fill=(0, 0, 0))     # black spot on the image
                kdraw.ellipse(box, fill=255)           # ...and mark it for repaint
        else:
            frac = float(self.d.get("inpaint_region", 0.28))
            mw = min(int(gw * frac), fx1 - fx0)
            mh = min(int(gh * frac), fy1 - fy0)
            mx = randint(fx0, fx1 - mw); my = randint(fy0, fy1 - mh)
            box = (mx, my, mx + mw, my + mh)
            if mode == "remove":
                manipulated.paste((0, 0, 0), box)      # black cut-out
            else:
                region = manipulated.crop(box).filter(ImageFilter.GaussianBlur(blur))
                manipulated.paste(region, box)
            kdraw.rectangle(box, fill=255)
        feather = float(self.d.get("inpaint_feather", 0))
        if feather > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(feather))
        return manipulated, mask

    def gen_inpaint(self, src_img, prompt, seed, mode=None, focus_box=None):
        """Occlusion completion: degrade the real image (blur / black / spots),
        then inpaint so the model restores plausible content from the surrounding
        real context — staying close to the true scene, not a free hallucination."""
        gw, gh = int(self.d["gen_width"]), int(self.d["gen_height"])
        base = src_img.resize((gw, gh))
        manipulated, mask = self.build_inpaint_input(base, seed, mode=mode, focus_box=focus_box)
        out = self.inpaint_pipe()(
            prompt=prompt, negative_prompt=self.d.get("negative_prompt"),
            image=manipulated, mask_image=mask,
            strength=float(self.d.get("inpaint_strength", 0.85)),
            width=gw, height=gh, num_inference_steps=int(self.d["steps"]),
            guidance_scale=float(self.d["guidance_scale"]),
            generator=self._generator(seed)).images[0]
        return out

    def gen_guided(self, src_img, prompt, seed):
        """Guided synthesis, mode-selectable via cfg.diffusion.guided_mode:
          "img2img"  (default) -> start from the REAL image, ControlNet steers at
                                   low `guided_strength`; preserves the object.
          "text2img"           -> ControlNet from edges only (invents appearance;
                                   can look 'weird' for object-centric scenes).
        """
        mode = self.d.get("guided_mode", "img2img")
        gw, gh = int(self.d["gen_width"]), int(self.d["gen_height"])
        base = src_img.resize((gw, gh))
        control = self._control_image(base)
        if mode == "text2img":
            return self.controlnet_pipe()(
                prompt=prompt, negative_prompt=self.d.get("negative_prompt"),
                image=control, width=gw, height=gh,
                num_inference_steps=int(self.d["steps"]),
                guidance_scale=float(self.d["guidance_scale"]),
                generator=self._generator(seed)).images[0]
        return self.controlnet_img2img_pipe()(
            prompt=prompt, negative_prompt=self.d.get("negative_prompt"),
            image=base, control_image=control,
            strength=float(self.d.get("guided_strength", 0.45)),
            controlnet_conditioning_scale=float(self.d.get("guided_control_scale", 0.9)),
            num_inference_steps=int(self.d["steps"]),
            guidance_scale=float(self.d["guidance_scale"]),
            generator=self._generator(seed)).images[0]

    # -- driver ----------------------------------------------------------------
    def generate_for_subset(self, scene: str, n: int, strategy: str,
                            count: int, overwrite: bool = False) -> Path:
        """Generate `count` synthetic images for (scene, n{N}, strategy)."""
        meta, frames = load_split_frames(scene, f"n{n}")
        prompt = self.cfg.scene(scene).get("prompt", "a photo of the scene")
        gw, gh = int(self.d["gen_width"]), int(self.d["gen_height"])
        base_seed = int(self.d.get("seed", 0))
        # object-centric scenes: focus the inpaint degradation ON the object
        masks_dir = self.cfg.scene(scene).get("masks_dir")
        mask_root = (self.cfg.scene_source(scene) / masks_dir) if masks_dir else None

        out_dir = SYNTHETIC_DIR / scene / f"n{n}" / strategy
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest: list[dict[str, Any]] = []

        for i in range(count):
            si = i % len(frames)
            src_fr = frames[si]
            seed = base_seed + i
            src_img = self._load_source(src_fr["file_path"], (gw, gh))

            if strategy == "outpaint":
                img, inner = self.gen_outpaint(src_img, prompt, seed)
                intr = _scaled_intrinsics(meta, inner_frac=inner)
                pose = src_fr["transform_matrix"]
            elif strategy == "inpaint":
                focus = self._object_focus_box(mask_root, src_fr, gw, gh)
                img = self.gen_inpaint(src_img, prompt, seed, focus_box=focus)
                intr = _scaled_intrinsics(meta)
                pose = src_fr["transform_matrix"]
            elif strategy == "guided":
                img = self.gen_guided(src_img, prompt, seed)
                intr = _scaled_intrinsics(meta)
                nb = _nearest_neighbor(frames, si)
                pose = interpolate_pose(src_fr["transform_matrix"],
                                        frames[nb]["transform_matrix"], t=0.5)
            elif strategy == "zero123":
                # genuine novel viewpoint: Zero123 azimuth-offset view of the
                # object, re-keyed onto black, placed by the VERIFIED local orbit
                # rotation (azimuth only; elevation is unreliable here).
                from .zero123 import Zero123Generator, cond_from_mask, rekey_on_black
                from .zero123_pose import (orbit_frame, novel_pose_azimuth,
                                           object_centroid, project_point)
                if mask_root is None:
                    raise ValueError("zero123 strategy needs an object mask (object-centric scene)")
                if self._z123 is None:
                    self._z123 = Zero123Generator(self.d.get("zero123_model", "kxic/zero123-xl"))
                    _, _allf = load_split_frames(scene, "full")
                    self._orbit = orbit_frame(_allf)
                    self._obj_O = object_centroid(self.cfg.scene_source(scene) / "sparse_pc.ply")
                mp = mask_root / Path(src_fr["file_path"]).name
                cond = cond_from_mask(src_fr["file_path"], mp, 256)
                azims = self.d.get("zero123_azimuths", [-25, -15, 15, 25])
                d_azim = float(azims[i % len(azims)])
                gen_white = self._z123.novel_view(
                    cond, 0.0, d_azim, steps=int(self.d.get("zero123_steps", 50)), seed=seed)
                C, up = self._orbit
                pose = novel_pose_azimuth(src_fr["transform_matrix"], C, up, d_azim)
                # place object where the NOVEL pose projects the object centroid,
                # at the source object's scale (same radius -> ~same size)
                uf, vf = project_point(self._obj_O, pose, meta["fl_x"], meta["fl_y"],
                                       meta["cx"], meta["cy"])
                W0, H0 = int(meta["w"]), int(meta["h"])
                box = self._object_focus_box(mask_root, src_fr, gw, gh, pad=0.0)
                bw, bh = ((box[2] - box[0], box[3] - box[1]) if box else (gw * 0.5, gh * 0.5))
                bbox = (uf * gw / W0, vf * gh / H0, bw, bh)
                img = rekey_on_black(gen_white, (gw, gh), bbox)
                intr = _scaled_intrinsics(meta)
            else:
                raise ValueError(f"unknown strategy '{strategy}'")

            fname = f"syn_{i:04d}.png"
            img.save(out_dir / fname)
            entry = {"file": fname, "strategy": strategy,
                     "source_file": src_fr["file_path"],
                     "transform_matrix": pose, "prompt": prompt, "seed": seed}
            entry.update(intr)
            with open(out_dir / f"syn_{i:04d}.json", "w") as f:
                json.dump(entry, f, indent=2)
            manifest.append(entry)

        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        return out_dir

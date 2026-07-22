"""Few-Shot Gaussian Splatting with Diffusion Augmentation — experiment harness.

Kept import-light on purpose: importing this package must NOT pull torch /
diffusers / nerfstudio, so lightweight steps (splits, grid, registry) run
without the heavy ML stack. Heavy modules (synthetic, nerf_eval) import their
deps lazily inside functions.
"""

from .config import Config, load_config, PROJECT_ROOT  # noqa: F401

__all__ = ["Config", "load_config", "PROJECT_ROOT"]

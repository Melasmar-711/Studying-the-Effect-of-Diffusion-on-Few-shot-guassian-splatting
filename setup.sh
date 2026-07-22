#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-shot environment setup for the few-shot GS + diffusion workspace.
# Safe to re-run (pip skips already-satisfied packages).
#
#   bash setup.sh
#
# Targets this machine: RTX 4060 (Ada, sm_89), driver 580, system nvcc 11.5.
# Because nvcc 11.5 cannot compile for sm_89, gsplat is installed from its
# PREBUILT cu118 wheel index (built with CUDA 11.8) instead of from source.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

PY=python3
VENV=.venv
TORCH_INDEX="https://download.pytorch.org/whl/cu118"
# PREBUILT gsplat wheel (compiled with CUDA 11.8 -> supports the 4060's sm_89).
# The system nvcc is 11.5 and CANNOT compile for sm_89, so we must NOT let gsplat
# build from source. This exact wheel ships gsplat/csrc.so and needs no nvcc.
GSPLAT_WHL="https://github.com/nerfstudio-project/gsplat/releases/download/v1.4.0/gsplat-1.4.0%2Bpt21cu118-cp310-cp310-linux_x86_64.whl"

echo "==> venv"
[ -d "$VENV" ] || "$PY" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
# setuptools<81 keeps pkg_resources, which torch 2.1.2's cpp_extension imports
# (`from pkg_resources import packaging`) when JIT-compiling gsplat's kernels.
python -m pip install --upgrade pip wheel "setuptools<81"

echo "==> torch 2.1.2 (cu118)"
pip install torch==2.1.2 torchvision==0.16.2 --index-url "$TORCH_INDEX"

echo "==> numpy<2 (torch/nerfstudio need the 1.x ABI)"
pip install "numpy<2"

echo "==> nerfstudio + diffusion + metrics stack"
pip install -r requirements.txt

echo "==> gsplat PREBUILT wheel (force after nerfstudio, else PyPI's source wheel wins)"
pip install --force-reinstall --no-deps "$GSPLAT_WHL"

echo "==> editable install of the gsfewshot package"
pip install -e .

echo "==> re-pin numpy<2 and setuptools<81 (in case a dep bumped them)"
pip install "numpy<2" "setuptools<81"

echo ""
echo "==> sanity check"
python - <<'PY'
from importlib.metadata import version
import torch, gsplat, diffusers, numpy
import nerfstudio  # noqa: F401
from diffusers import (StableDiffusionInpaintPipeline,          # noqa: F401
                       StableDiffusionControlNetPipeline, ControlNetModel)
print("torch      ", torch.__version__, "cuda:", torch.cuda.is_available())
print("gsplat     ", version("gsplat"))
print("nerfstudio ", version("nerfstudio"))
print("diffusers  ", diffusers.__version__)
print("numpy      ", numpy.__version__)
if torch.cuda.is_available():
    print("device     ", torch.cuda.get_device_name(0))
print("OK: all core imports succeeded")
PY

echo ""
echo "Done. Activate with:  source .venv/bin/activate"

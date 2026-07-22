"""Make `import gsfewshot` work whether or not the package is pip-installed,
and pin the CUDA arch BEFORE torch/gsplat get imported anywhere.

This machine's nvcc is 11.5, which cannot target the RTX 4060's Ada (sm_89)
architecture. gsplat 1.x JIT-compiles its CUDA kernels on first use, so we
force compilation for sm_86 + embedded PTX; the CUDA-12 driver then JIT-links
that PTX up to sm_89 at runtime. Every script imports this module first, so the
setting propagates to the `ns-train` subprocess too.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6+PTX")
os.environ.setdefault("MAX_JOBS", "4")           # cap compile RAM (15 GB box)
# HF downloads on this box are flaky. Disable the xet CDN backend and hf_transfer
# (both fail hard without graceful retry); use the plain resumable backend with a
# generous timeout. scripts/prefetch_models.py wraps it in a resume-retry loop.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
# help torch's cpp_extension find the apt-installed toolkit (nvcc in /usr/bin)
if "CUDA_HOME" not in os.environ and Path("/usr/bin/nvcc").exists():
    os.environ["CUDA_HOME"] = "/usr"

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

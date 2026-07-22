"""Train one experiment with splatfacto and capture time + peak VRAM.

Flow: assemble a self-contained data dir (real subset + S synthetic), invoke
``ns-train splatfacto`` as a subprocess with a *fixed* output path, and sample
GPU memory in a side thread so we can report a memory footprint per run.
"""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from .config import Config, EXPERIMENTS_DIR
from .grid import Experiment
from .assemble import assemble_experiment_data


class _VramSampler(threading.Thread):
    """Polls total GPU memory.used and records the peak delta over a baseline."""

    def __init__(self, interval: float = 1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop_evt = threading.Event()      # NB: not `_stop` (Thread uses that)
        self.baseline = self._used()
        self.peak = self.baseline

    @staticmethod
    def _used() -> int:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"], text=True)
            return int(out.strip().splitlines()[0])
        except Exception:
            return 0

    def run(self):
        while not self._stop_evt.is_set():
            self.peak = max(self.peak, self._used())
            time.sleep(self.interval)

    def stop(self) -> int:
        self._stop_evt.set()
        self.join(timeout=3)
        return max(0, self.peak - self.baseline)


def train_config_path(exp: Experiment, out_root: Path) -> Path:
    return out_root / "train" / exp.exp_id / "splatfacto" / "run" / "config.yml"


def train_experiment(cfg: Config, exp: Experiment,
                     max_iterations: int | None = None,
                     out_root: Path | None = None,
                     extra_args: list[str] | None = None) -> dict[str, Any]:
    out_root = out_root or (EXPERIMENTS_DIR / exp.exp_id)
    out_root.mkdir(parents=True, exist_ok=True)
    data_dir = out_root / "data"

    summary = assemble_experiment_data(cfg, exp, data_dir)

    it = max_iterations or int(cfg.training["max_num_iterations"])
    ds = int(cfg.training["downscale_factor"])
    train_out = out_root / "train"

    cmd = [
        "ns-train", cfg.training.get("backend", "splatfacto"),
        "--data", str(data_dir),
        "--output-dir", str(train_out),
        "--experiment-name", exp.exp_id,
        "--timestamp", "run",
        "--max-num-iterations", str(it),
        "--viewer.quit-on-train-completion", "True",
        "--vis", "tensorboard",
        # we run our own held-out eval, so disable nerfstudio's internal eval
        "--steps-per-eval-image", str(it + 1),
        "--steps-per-eval-batch", str(it + 1),
        "--steps-per-eval-all-images", str(it + 1),
    ]
    cmd += cfg.training.get("extra_args", []) or []
    cmd += extra_args or []
    # dataparser: use every provided image for training, downscale for 8 GB
    cmd += ["nerfstudio-data",
            "--downscale-factor", str(ds),
            "--train-split-fraction", "1.0"]

    sampler = _VramSampler()
    sampler.start()
    t0 = time.time()
    proc = subprocess.run(cmd)
    train_time = time.time() - t0
    peak_vram = sampler.stop()

    ok = proc.returncode == 0
    return {
        **summary,
        "exp_id": exp.exp_id,
        "returncode": proc.returncode,
        "status": "trained" if ok else "train_failed",
        "train_time_s": round(train_time, 1),
        "peak_vram_mb": peak_vram,
        "max_num_iterations": it,
        "downscale_factor": ds,
        "config_yml": str(train_config_path(exp, out_root)),
        "cmd": " ".join(cmd),
    }

#!/usr/bin/env python
"""Robustly pre-download the diffusion models into the HF cache.

HF downloads on this box are flaky (read timeouts). snapshot_download resumes
partial files, so we simply retry until each repo completes. Grabs the fp16
safetensors variant (+ configs/tokenizer) to minimise bytes.

    python scripts/prefetch_models.py
"""
import time
import _bootstrap  # noqa: F401  (sets HF env: no xet / no hf_transfer / long timeout)
from huggingface_hub import snapshot_download
from gsfewshot import load_config


def fetch(repo, attempts=40):
    allow = ["*.json", "*.txt", "*.model",
             "*.fp16.safetensors", "*fp16*.safetensors",
             "vocab.json", "merges.txt"]
    for i in range(1, attempts + 1):
        try:
            p = snapshot_download(repo, allow_patterns=allow)
            print(f"  OK  {repo}")
            return p
        except Exception as e:
            print(f"  retry {i}/{attempts} {repo}: {type(e).__name__}: {str(e)[:80]}",
                  flush=True)
            time.sleep(min(5 * i, 30))
    # last resort: no fp16 filter (some components ship only fp32/.bin)
    return snapshot_download(repo)


def main():
    cfg = load_config()
    d = cfg.diffusion
    repos = [d["inpaint_model"], d["base_model"], d["controlnet_canny"]]
    for r in repos:
        print("fetching", r, flush=True)
        fetch(r)
    print("PREFETCH_DONE")


if __name__ == "__main__":
    main()

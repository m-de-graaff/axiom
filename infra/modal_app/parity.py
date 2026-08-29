"""Parity + speed of the KV cache on real weights and a real GPU (P4-02, P4-04).

`tests/test_parity.py` proves the cached loop on a tiny random-weight CPU config in
CI. This is the other half CLAUDE.md asks for: the CUDA leg, on the checkpoints that
actually get evaluated, plus the before/after numbers for P4-09.

    modal run infra/modal_app/parity.py
    modal run infra/modal_app/parity.py --model axiom-zero-base --samples 64
"""

import pathlib

import modal

app = modal.App("axiom-parity")

REPO = pathlib.Path(__file__).resolve().parents[2] if modal.is_local() else pathlib.Path("/root")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "pandas", "numpy", "einops", "huggingface_hub", "safetensors", "tqdm",
    )
    .add_local_dir(str(REPO / "packages" / "axiom_model" / "axiom_model"), "/root/axiom_model")
    .add_local_dir(str(REPO / "packages" / "axiom_data" / "axiom_data"), "/root/axiom_data")
)


@app.function(image=image, gpu="L4", timeout=60 * 60)
def check(model_name: str = "axiom-zero-small", samples: int = 16, horizon: int = 24) -> dict:
    import sys
    import time

    sys.path.insert(0, "/root")
    import numpy as np
    import torch
    from axiom_data.synthetic import random_walk_ohlcv
    from axiom_model import load_predictor, resolve

    spec = resolve(model_name)
    predictor = load_predictor(model_name)
    context = spec.max_context - horizon  # the largest context the cache can serve
    df, timestamps = random_walk_ohlcv(context, horizon)

    from axiom_data import normalization

    x = df[normalization.FEATURES].to_numpy(np.float32)
    mean, std = normalization.fit(x)
    x_norm = normalization.apply(x, mean, std)[None]
    import pandas as pd

    x_stamp = normalization.time_features(pd.Series(timestamps[:context])).to_numpy(np.float32)
    y_stamp = normalization.time_features(pd.Series(timestamps[context:])).to_numpy(np.float32)

    def run(use_cache: bool, seed: int = 1234):
        torch.manual_seed(seed)
        torch.cuda.synchronize()
        started = time.time()
        out = predictor.generate(
            x_norm, x_stamp[None], y_stamp[None], horizon, 1.0, 0, 0.9, samples, False,
            reduce="none", use_cache=use_cache,
        )
        torch.cuda.synchronize()
        return out, time.time() - started

    cached, cached_s = run(True)
    reference, reference_s = run(False)

    identical = bool(np.array_equal(cached, reference))
    close = np.abs(cached - reference).max()
    return {
        "model": model_name,
        "gpu": torch.cuda.get_device_name(0),
        "context": context,
        "horizon": horizon,
        "samples": samples,
        "token_identical": identical,
        "max_abs_diff": float(close),
        "cached_seconds": round(cached_s, 2),
        "reference_seconds": round(reference_s, 2),
        "speedup": round(reference_s / cached_s, 1),
    }


@app.local_entrypoint()
def main(model: str = "axiom-zero-small", samples: int = 16, horizon: int = 24):
    result = check.remote(model_name=model, samples=samples, horizon=horizon)
    for key, value in result.items():
        print(f"{key:>18}: {value}")
    if not result["token_identical"]:
        print("\nNOT token-identical — do not ship this cache (CLAUDE.md rule 2).")

"""Parity + speed of the KV cache on real weights, on whatever device you are on.

Lives in axiom_eval rather than axiom_model because it needs both packages, and
axiom_model deliberately does not depend on axiom_data.

One implementation, three backends: `scripts/rocm_check.py` runs it on the XTX,
`infra/modal_app/parity.py` runs it on a Modal GPU, and either can run it on CPU.
CLAUDE.md wants the same check on CUDA and ROCm before an `axiom-runtime-*` tag, and
a check that is copy-pasted per backend is a check that drifts.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from axiom_data import normalization
from axiom_data.synthetic import random_walk_ohlcv
from axiom_model import load_predictor, resolve


def parity_and_speed(
    model_name: str = "axiom-zero-small",
    samples: int = 64,
    horizon: int = 24,
    device: str | None = None,
    seed: int = 1234,
) -> dict:
    """Generate the same window with and without the cache; compare and time both.

    The context is the largest the cache can serve (`max_context - horizon`), which
    is what the eval harness feeds. Token-identical is the pass condition — never
    "close enough" (CLAUDE.md rule 2).
    """
    spec = resolve(model_name)
    predictor = load_predictor(model_name, device=device)
    context = spec.max_context - horizon
    df, timestamps = random_walk_ohlcv(context, horizon)

    x = df[normalization.FEATURES].to_numpy(np.float32)
    mean, std = normalization.fit(x)
    x_norm = normalization.apply(x, mean, std)[None]
    x_stamp = normalization.time_features(pd.Series(timestamps[:context])).to_numpy(np.float32)
    y_stamp = normalization.time_features(pd.Series(timestamps[context:])).to_numpy(np.float32)

    def run(use_cache: bool):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.time()
        out = predictor.generate(
            x_norm, x_stamp[None], y_stamp[None], horizon, 1.0, 0, 0.9, samples, False,
            reduce="none", use_cache=use_cache,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return out, time.time() - started

    cached, cached_s = run(True)
    reference, reference_s = run(False)
    accelerator = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    return {
        "model": model_name,
        "device": predictor.device,
        "accelerator": accelerator,  # ROCm reports the AMD card here too
        "torch": str(torch.__version__),
        "context": context,
        "horizon": horizon,
        "samples": samples,
        "token_identical": bool(np.array_equal(cached, reference)),
        "max_abs_diff": float(np.abs(cached - reference).max()),
        "cached_seconds": round(cached_s, 2),
        "reference_seconds": round(reference_s, 2),
        "speedup": round(reference_s / cached_s, 1),
    }


__all__ = ["parity_and_speed"]

"""GPU smoke test on Modal — no local GPU needed (P0-07).

Loads Kronos-small on a T4, runs a short forecast on synthetic OHLCV.
Prereq: vendor/kronos present (P0-04 subtree). Cost: pennies (fits free credits).

usage:  modal run infra/modal_app/smoke.py
"""

import modal

app = modal.App("axiom-smoke")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "pandas", "numpy", "einops",
        "huggingface_hub", "safetensors", "tqdm",
    )
    .add_local_dir("vendor/kronos", "/root/kronos")
)


@app.function(image=image, gpu="T4", timeout=900)
def forecast_smoke():
    import sys
    import time

    sys.path.insert(0, "/root/kronos")
    import numpy as np
    import pandas as pd
    import torch

    print("device:", torch.cuda.get_device_name(0))

    from model import Kronos, KronosPredictor, KronosTokenizer  # upstream API

    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    mdl = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    n_params = sum(p.numel() for p in mdl.parameters()) / 1e6
    print(f"loaded Kronos-small: {n_params:.1f}M params")
    pred = KronosPredictor(mdl, tok, device="cuda:0", max_context=512)

    # Synthetic random-walk OHLCV — no external data needed for a smoke test.
    n, h = 400, 24
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.001, n)),
            "close": close,
            "volume": rng.uniform(1e3, 1e4, n),
        }
    )
    df["high"] = df[["open", "close"]].max(axis=1) * (1 + np.abs(rng.normal(0, 0.002, n)))
    df["low"] = df[["open", "close"]].min(axis=1) * (1 - np.abs(rng.normal(0, 0.002, n)))
    df["amount"] = df["close"] * df["volume"]
    ts = pd.date_range("2026-01-01", periods=n + h, freq="1h")

    t0 = time.time()
    try:
        out = pred.predict(
            df=df[["open", "high", "low", "close", "volume", "amount"]],
            x_timestamp=pd.Series(ts[:n]),
            y_timestamp=pd.Series(ts[n:]),
            pred_len=h,
            T=1.0,
            top_p=0.9,
            sample_count=3,
        )
        print(f"forecast OK in {time.time() - t0:.1f}s")
        print(out.head())
        print("SMOKE PASSED — GPU + weights + generation all working.")
    except TypeError as e:
        # Upstream signatures can drift between versions. If this fires, align
        # the call with vendor/kronos/examples/prediction_example.py.
        print("predict() signature mismatch — adapt to upstream example:", e)
        raise

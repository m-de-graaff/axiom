"""GPU smoke test on Modal — no local GPU needed (P0-07).

Runs a short forecast through `AxiomPredictor` + `axiom_model.registry` on a T4
and writes the result to `research/day1/`. Cost: pennies (fits free credits).

usage:  modal run infra/modal_app/smoke.py
        modal run infra/modal_app/smoke.py --model axiom-zero-base
"""

import pathlib

import modal

app = modal.App("axiom-smoke")

# This module is imported remotely too, where it lives at /root/smoke.py and has
# no repo above it — so only reach for repo paths on the local side.
REPO = pathlib.Path(__file__).resolve().parents[2] if modal.is_local() else pathlib.Path("/root")
OUT_DIR = REPO / "research" / "day1"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "pandas", "numpy", "einops",
        "huggingface_hub", "safetensors", "tqdm",
    )
    .add_local_dir(str(REPO / "packages" / "axiom_model" / "axiom_model"), "/root/axiom_model")
    .add_local_dir(str(REPO / "packages" / "axiom_data" / "axiom_data"), "/root/axiom_data")
)


@app.function(image=image, gpu="T4", timeout=900)
def forecast_smoke(model_name: str = "axiom-zero-small", n: int = 400, horizon: int = 24):
    import sys
    import time

    sys.path.insert(0, "/root")
    import pandas as pd
    import torch
    from axiom_data.synthetic import random_walk_ohlcv
    from axiom_model import load_predictor, resolve

    print("device:", torch.cuda.get_device_name(0))

    spec = resolve(model_name)
    pred = load_predictor(model_name)
    n_params = sum(p.numel() for p in pred.model.parameters()) / 1e6
    print(f"loaded {model_name} ({spec.model_source}): {n_params:.1f}M params")

    df, ts = random_walk_ohlcv(n, horizon)
    t0 = time.time()
    out = pred.predict(
        df=df,
        x_timestamp=pd.Series(ts[:n]),
        y_timestamp=pd.Series(ts[n:]),
        pred_len=horizon,
        T=1.0,
        top_p=0.9,
        sample_count=3,
    )
    elapsed = time.time() - t0
    print(f"forecast OK in {elapsed:.1f}s")
    print(out.head())

    assert len(out) == horizon, f"expected {horizon} bars, got {len(out)}"
    assert out.notna().all().all(), "forecast contains NaNs"
    return {
        "model": model_name,
        "source": spec.model_source,
        "params_m": round(n_params, 1),
        "gpu": torch.cuda.get_device_name(0),
        "seconds": round(elapsed, 2),
        "forecast_csv": out.to_csv(),
    }


@app.local_entrypoint()
def main(model: str = "axiom-zero-small"):
    result = forecast_smoke.remote(model_name=model)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"smoke_{model}.csv"
    path.write_text(result.pop("forecast_csv"), encoding="utf-8")
    print(result)
    print(f"SMOKE PASSED — GPU + weights + generation via AxiomPredictor. Wrote {path}")

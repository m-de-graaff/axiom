"""Hourly signal cron on Modal L4 (P6-04). Skeleton — see build order §8.3.

Rules: staleness guard before inference; idempotent upserts on
(symbol, tf, made_at); alert webhook on failure; every run row in `runs`.
"""

import modal

app = modal.App("axiom-signals")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "pandas", "numpy", "ccxt", "psycopg[binary]", "pyyaml")
    .add_local_dir("packages", "/root/packages")
)
ckpt_vol = modal.Volume.from_name("axiom-ckpts", create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",
    schedule=modal.Cron("2 * * * *"),  # hh:02, just after 1h candle close
    volumes={"/ckpts": ckpt_vol},
    secrets=[modal.Secret.from_name("postgres"), modal.Secret.from_name("telegram")],
)
def hourly_signals():
    # TODO P6-03: bars = pull_latest_bars_ccxt(universe, tf="1h", lookback=CTX)
    # TODO P6-03: guard_stale(bars, max_age_bars=2)
    # TODO P6-04: fc = predictor.predict_mc(bars, samples=64, pred_len=24)
    # TODO P6-04: upsert(signals_from_paths(fc, costs)); upsert_fan(fc); log_health()
    raise NotImplementedError("P6-04: wire signal pipeline")

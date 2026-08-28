"""Corpus download + ingest on Modal (P1-04 fallback, P1-11 sync).

Two reasons to run this instead of the local CLI:

1. `data.binance.vision` is unreachable from NL (Binance has exited the EU; the data
   CDN is separate and currently fine, but that is not a guarantee). Modal's egress
   is not in the EU, so this is the fallback path.
2. The corpus needs to be on the `axiom-data` volume anyway for training and the
   hourly inference cron. Building it here skips a multi-GB upload.

    modal run infra/modal_app/download.py                       # whole universe
    modal run infra/modal_app/download.py --symbols BTCUSDT,ETHUSDT --start 2024-01
"""

import pathlib

import modal

app = modal.App("axiom-download")

REPO = pathlib.Path(__file__).resolve().parents[2] if modal.is_local() else pathlib.Path("/root")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pandas", "pyarrow", "duckdb", "httpx", "pyyaml", "numpy")
    .add_local_dir(str(REPO / "packages" / "axiom_data" / "axiom_data"), "/root/axiom_data")
    .add_local_dir(str(REPO / "configs"), "/root/configs")
)
data_vol = modal.Volume.from_name("axiom-data", create_if_missing=True)


@app.function(image=image)
def plan(config: str) -> dict:
    """Resolve the config remotely: `modal run` executes the local entrypoint in the
    Modal CLI's own environment, which has no pandas/yaml."""
    import os
    import sys

    sys.path.insert(0, "/root")
    os.chdir("/root")  # config paths inside the YAML are repo-relative
    from axiom_data.datasets import load_config

    cfg = load_config(config)
    return {"symbols": cfg.symbols, "source_tf": cfg.source_tf, "timeframes": cfg.timeframes}


@app.function(image=image, volumes={"/data": data_vol}, timeout=6 * 60 * 60)
def fetch(symbol: str, source_tf: str, timeframes: list[str], start: str | None, end: str | None):
    """Download one symbol's monthly zips and write its parquet partitions."""
    import asyncio
    import sys

    sys.path.insert(0, "/root")
    import pandas as pd
    from axiom_data import binance, store
    from axiom_data.resample import resample_ohlcv

    raw, root = pathlib.Path("/data/raw"), pathlib.Path("/data/parquet")
    keys = binance.select_keys(binance.SPOT_KLINES, symbol, source_tf, start, end)
    if not keys:
        return {"symbol": symbol, "bars": 0, "note": "no files in range"}
    asyncio.run(binance.download_keys(keys, raw, concurrency=12))

    bars = pd.concat([binance.klines_to_df(raw / k, source_tf) for k in keys])
    bars = bars.sort_values("ts").drop_duplicates(subset="ts").reset_index(drop=True)
    store.write_months(bars, root, "binance", symbol, source_tf)
    for tf in timeframes:
        store.write_months(resample_ohlcv(bars, tf), root, "binance", symbol, tf)
    data_vol.commit()
    return {"symbol": symbol, "bars": len(bars), "first": str(bars.ts.iloc[0])}


@app.function(image=image, volumes={"/data": data_vol}, timeout=2 * 60 * 60)
def build(config: str = "configs/data/crypto_v1.yaml") -> dict:
    """Build the dataset index on the volume. The hash must equal the local one --
    same config, same bars, same hash, or reproducibility is a story we tell.

        modal run infra/modal_app/download.py::build
    """
    import os
    import sys

    sys.path.insert(0, "/root")
    os.chdir("/root")
    from axiom_data.datasets import build as build_dataset

    manifest = build_dataset(
        config, root=pathlib.Path("/data/parquet"), out_dir=pathlib.Path("/data/datasets")
    )
    data_vol.commit()
    summary = {k: manifest[k] for k in ("dataset", "dataset_hash", "splits", "git_sha")}
    print(summary)  # `modal run` does not surface return values
    return summary


@app.local_entrypoint()
def main(
    config: str = "configs/data/crypto_v1.yaml",
    symbols: str = "",
    start: str = "",
    end: str = "",
):
    cfg = plan.remote(config)
    wanted = symbols.split(",") if symbols else cfg["symbols"]
    args = [(s, cfg["source_tf"], cfg["timeframes"], start or None, end or None) for s in wanted]
    for result in fetch.starmap(args):
        print(result)
    print(f"done: {len(wanted)} symbols on the axiom-data volume")

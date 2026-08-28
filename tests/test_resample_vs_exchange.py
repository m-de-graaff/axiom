"""Ground truth for the bar convention: our 1m -> 1h resample must equal Binance's own
1h klines. Unit tests prove the code does what we specified; this proves the spec is the
exchange's. Hits the network, so it is opt-in:

    uv run pytest tests/test_resample_vs_exchange.py --network
"""

import asyncio
from pathlib import Path

import pytest
from axiom_data import binance
from axiom_data.resample import resample_ohlcv

SYMBOL, MONTH = "BTCUSDT", "2024-01"


def fetch(tf: str, raw: Path) -> Path:
    keys = binance.select_keys(binance.SPOT_KLINES, SYMBOL, tf, MONTH, MONTH)
    asyncio.run(binance.download_keys(keys, raw))
    return raw / keys[0]


@pytest.mark.network
def test_resampled_hours_match_the_exchange(tmp_path):
    raw = Path("data/raw") if Path("data/raw").is_dir() else tmp_path
    official = binance.klines_to_df(fetch("1h", raw), "1h")
    ours = resample_ohlcv(binance.klines_to_df(fetch("1m", raw), "1m"), "1h")

    merged = official.merge(ours, on="ts", suffixes=("_ex", "_ax"))
    assert len(merged) == len(official) > 700  # every hour lines up, none shifted

    for col in ["open", "high", "low", "close"]:
        assert (merged[f"{col}_ex"] == merged[f"{col}_ax"]).all()
    for col in ["volume", "amount"]:
        rel = (merged[f"{col}_ex"] - merged[f"{col}_ax"]).abs() / merged[f"{col}_ex"].abs()
        assert rel.max() < 1e-12  # summation order only

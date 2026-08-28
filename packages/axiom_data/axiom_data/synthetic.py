"""Synthetic OHLCV for smoke tests. Never for evaluation — a random walk has no
signal, and any metric computed on it is meaningless by construction.
"""

import numpy as np
import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume", "amount"]


def random_walk_ohlcv(n: int, horizon: int, seed: int = 7, freq: str = "1h"):
    """Return `(df, timestamps)` — `n` bars of random-walk OHLCV plus `n + horizon`
    timestamps, so callers can slice context and forecast stamps."""
    rng = np.random.default_rng(seed)
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
    timestamps = pd.date_range("2026-01-01", periods=n + horizon, freq=freq)
    return df[COLUMNS], timestamps


__all__ = ["COLUMNS", "random_walk_ohlcv"]

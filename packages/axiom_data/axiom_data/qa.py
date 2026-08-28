"""Corpus QA (P1-08). Hard violations fail the build; thresholds come from config.

Hard violations are things that are *wrong* (duplicate or unsorted timestamps, NaNs,
non-positive prices, impossible OHLC, negative volume). Thresholded findings are
things that are merely *suspicious* (missing bars, zero-volume bars) and their limits
live in the `qa:` block of `configs/data/*.yaml` -- never inline here.
"""

from __future__ import annotations

import pandas as pd

from .resample import timeframe_delta

HARD_CHECKS = [
    "duplicate_ts",
    "unsorted_ts",
    "nan_rows",
    "nonpositive_price",
    "ohlc_violations",
    "negative_volume",
]

DEFAULT_THRESHOLDS = {"max_missing_bar_pct": 1.0, "max_zero_volume_pct": 5.0, "min_bars": 1}


def check_frame(df: pd.DataFrame, symbol: str, tf: str) -> dict:
    """One row of the QA report for a single symbol/timeframe."""
    price = df[["open", "high", "low", "close"]]
    n = len(df)
    span = timeframe_delta(tf)
    expected = int((df.ts.iloc[-1] - df.ts.iloc[0]) / span) + 1 if n else 0
    ohlc_bad = (
        (df.high < price.max(axis=1))
        | (df.low > price.min(axis=1))
        | (df.high < df.low)
    ).sum()
    return {
        "symbol": symbol,
        "tf": tf,
        "bars": n,
        "first_ts": df.ts.iloc[0] if n else pd.NaT,
        "last_ts": df.ts.iloc[-1] if n else pd.NaT,
        "duplicate_ts": int(df.ts.duplicated().sum()),
        "unsorted_ts": int(not df.ts.is_monotonic_increasing),
        "nan_rows": int(df.isna().any(axis=1).sum()),
        "nonpositive_price": int((price <= 0).any(axis=1).sum()),
        "ohlc_violations": int(ohlc_bad),
        "negative_volume": int((df.volume < 0).sum()),
        "missing_bars": max(expected - n, 0),
        "missing_bar_pct": round(100 * max(expected - n, 0) / expected, 4) if expected else 0.0,
        "zero_volume_pct": round(100 * (df.volume == 0).sum() / n, 4) if n else 0.0,
    }


def violations(report: pd.DataFrame, thresholds: dict | None = None) -> list[str]:
    """Every reason this corpus must not be built into a dataset."""
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    out = []
    for row in report.to_dict("records"):
        where = f"{row['symbol']}/{row['tf']}"
        for check in HARD_CHECKS:
            if row[check]:
                out.append(f"{where}: {check}={row[check]}")
        if row["bars"] < t["min_bars"]:
            out.append(f"{where}: only {row['bars']} bars (< min_bars={t['min_bars']})")
        if row["missing_bar_pct"] > t["max_missing_bar_pct"]:
            out.append(
                f"{where}: {row['missing_bar_pct']}% bars missing "
                f"(> max_missing_bar_pct={t['max_missing_bar_pct']})"
            )
        if row["zero_volume_pct"] > t["max_zero_volume_pct"]:
            out.append(
                f"{where}: {row['zero_volume_pct']}% zero-volume bars "
                f"(> max_zero_volume_pct={t['max_zero_volume_pct']})"
            )
    return out


__all__ = ["DEFAULT_THRESHOLDS", "HARD_CHECKS", "check_frame", "violations"]

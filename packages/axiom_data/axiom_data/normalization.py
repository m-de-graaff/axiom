"""Window normalization — the single implementation (see CLAUDE.md, docs/normalization.md).

Byte-for-byte the upstream Kronos scheme: per-window z-score computed on the
context window only, epsilon 1e-5, symmetric clip at 5 sigma. Upstream does this
in two places (`finetune/dataset.py` and `KronosPredictor.predict`); this module
is the only place Axiom does it. Training, eval and inference all import from here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["open", "high", "low", "close", "volume", "amount"]
TIME_FEATURES = ["minute", "hour", "weekday", "day", "month"]
PRICE_COLS = ["open", "high", "low", "close"]

EPS = 1e-5
CLIP = 5.0


def fit(context: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/std over the context window only (axis 0)."""
    return np.mean(context, axis=0), np.std(context, axis=0)


def apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray, clip: float = CLIP) -> np.ndarray:
    return np.clip((x - mean) / (std + EPS), -clip, clip)


def invert(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Inverse of `apply` ignoring the clip (upstream denormalizes the same way)."""
    return x * (std + EPS) + mean


def normalize_window(
    x: np.ndarray, context_len: int, clip: float = CLIP
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize a `(context + horizon, n_features)` window using context stats only.

    Returns `(x_norm, mean, std)`. The horizon rows are scaled with the *context*
    statistics — never their own — which is what keeps the target leak-free.
    """
    if context_len <= 0 or context_len > len(x):
        raise ValueError(f"context_len={context_len} outside window of length {len(x)}")
    mean, std = fit(x[:context_len])
    return apply(x, mean, std, clip), mean, std


def time_features(ts: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    """The five calendar features the model consumes, in upstream order."""
    ts = pd.Series(pd.to_datetime(ts))
    return pd.DataFrame(
        {
            "minute": ts.dt.minute,
            "hour": ts.dt.hour,
            "weekday": ts.dt.weekday,
            "day": ts.dt.day,
            "month": ts.dt.month,
        }
    ).reset_index(drop=True)


def ensure_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Fill `amount` the way upstream does when a feed lacks it (quote volume)."""
    if "amount" in df.columns:
        return df
    df = df.copy()
    df["amount"] = df["volume"] * df[PRICE_COLS].mean(axis=1)
    return df


__all__ = [
    "CLIP",
    "EPS",
    "FEATURES",
    "PRICE_COLS",
    "TIME_FEATURES",
    "apply",
    "ensure_amount",
    "fit",
    "invert",
    "normalize_window",
    "time_features",
]

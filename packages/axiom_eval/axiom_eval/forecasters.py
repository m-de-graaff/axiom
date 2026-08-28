"""Everything that turns a context window into a forecast distribution (P2-06..08).

One interface for models and baselines alike:

    forecast(windows, horizons, samples, seed) -> (n_windows, samples, n_horizons)

of **cumulative log returns** measured from the last context close. Log returns
because raw prices are not comparable across symbols (CLAUDE.md rule: never score
on prices); distributions rather than points because calibration (P2-04) is half
of what the harness exists to measure.

Baselines are deliberately dumb — the humiliation panel only works if it is cheap:

  * `persistence` — the last `max_horizon` bars' average drift, carried forward.
    A pure random walk (drift 0) would make cross-sectional RankIC undefined, so
    "persistence" here means *the return persists*, the standard framing.
  * `ewma` — EWMA drift and vol over the context; Gaussian increments.
  * `lightgbm` — gradient boosting on lagged return/vol/volume features, one model
    per horizon, fit on the non-test splits only (see `docs/eval.md`).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from axiom_data import datasets, normalization, store
from axiom_data.resample import timeframe_delta

from .panel import EvalConfig, Window

CLOSE = normalization.FEATURES.index("close")
PRICE_FLOOR = 1e-12  # a 5-sigma denormalized close can land at or below zero


def window_seed(seed: int, name: str, window: Window) -> int:
    """Per-window seed: reproducible regardless of evaluation order or machine."""
    key = f"{seed}|{name}|{window.symbol}|{window.tf}|{window.anchor.isoformat()}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def _gaussian(
    mu: np.ndarray, sigma: np.ndarray, horizons: Sequence[int], samples: int, rng
) -> np.ndarray:
    """Random-walk paths with per-bar drift `mu` and vol `sigma`, sampled at `horizons`.

    `mu` and `sigma` are per window; the returned array is (n_windows, samples, n_h).
    """
    noise = rng.standard_normal((len(mu), samples, max(horizons)))
    steps = np.cumsum(mu[:, None, None] + sigma[:, None, None] * noise, axis=2)
    return steps[:, :, [h - 1 for h in horizons]]


def _log_returns(context: pd.DataFrame) -> np.ndarray:
    return np.diff(np.log(context["close"].to_numpy()))


class Persistence:
    """Last-`max_horizon`-bars drift carried forward; context vol for the fan."""

    name = "persistence"

    def __init__(self, cfg: EvalConfig):
        self.vol_window = cfg.panel["vol_window"]
        self.lookback = cfg.max_horizon

    def forecast(self, windows, horizons, samples, seed) -> np.ndarray:
        rng = np.random.default_rng(window_seed(seed, self.name, windows[0]))
        rets = [_log_returns(w.context) for w in windows]
        mu = np.array([r[-self.lookback :].mean() for r in rets])
        sigma = np.array([r[-self.vol_window :].std(ddof=1) for r in rets])
        return _gaussian(mu, sigma, horizons, samples, rng)


class Ewma:
    """EWMA drift + EWMA vol, Gaussian increments."""

    name = "ewma"

    def __init__(self, cfg: EvalConfig):
        self.halflife = cfg.ewma.get("halflife_bars", 24)

    def forecast(self, windows, horizons, samples, seed) -> np.ndarray:
        rng = np.random.default_rng(window_seed(seed, self.name, windows[0]))
        mu, sigma = [], []
        for w in windows:
            r = pd.Series(_log_returns(w.context))
            ewm = r.ewm(halflife=self.halflife)
            mu.append(ewm.mean().iloc[-1])
            sigma.append(ewm.std().iloc[-1])
        return _gaussian(np.array(mu), np.array(sigma), horizons, samples, rng)


def feature_frame(
    bars: pd.DataFrame, lags: Sequence[int], vol_windows: Sequence[int]
) -> pd.DataFrame:
    """Ex-ante features per bar: nothing here reads a bar later than its own row.

    The same function serves training and inference, which is the only way the two
    stay identical (normalization drift is this project's #1 known failure mode and
    feature drift is its twin).
    """
    close = np.log(bars["close"].to_numpy())
    r = pd.Series(close).diff()
    out = {f"ret_lag{k}": r.shift(k - 1) for k in lags}
    for w in vol_windows:
        out[f"vol{w}"] = r.rolling(w).std()
        out[f"cum{w}"] = pd.Series(close).diff(w)
        logv = np.log1p(bars["volume"].to_numpy())
        out[f"volz{w}"] = (
            pd.Series(logv) - pd.Series(logv).rolling(w).mean()
        ) / pd.Series(logv).rolling(w).std()
    return pd.DataFrame(out)


class LightGbm:
    """Gradient boosting on lagged features — the baseline that has to be beaten.

    Fit once on the configured splits (never on test, CLAUDE.md rule 3). The fan is
    a Gaussian around the point forecast with the training residual sigma; in-sample
    residuals understate it slightly, which flatters this baseline's coverage, not
    Axiom's.
    """

    name = "lightgbm"

    def __init__(
        self,
        cfg: EvalConfig,
        data_cfg: datasets.DataConfig,
        tf: str,
        root: Path = store.DEFAULT_ROOT,
        symbols: Sequence[str] | None = None,
    ):
        self.cfg, self.data_cfg, self.tf, self.root = cfg, data_cfg, tf, root
        self.symbols = list(symbols) if symbols else list(data_cfg.symbols)
        self.lags = cfg.lightgbm["lags"]
        self.vol_windows = cfg.lightgbm["vol_windows"]
        self.models: dict[int, object] = {}
        self.sigma: dict[int, float] = {}

    def _training_rows(self, horizons: Sequence[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
        step = timeframe_delta(self.tf)
        grid = int((self.cfg.lightgbm["train_stride_bars"] * step).value)
        hmax, ctx = max(horizons), self.data_cfg.context_bars
        feats, targets = [], []
        for split in self.cfg.lightgbm["fit_splits"]:
            lo, hi = datasets.split_bounds(self.data_cfg, split, self.tf)
            for symbol in self.symbols:
                bars = store.read(
                    symbol, self.tf, root=self.root, venue=self.data_cfg.venue, start=lo, end=hi
                ).reset_index(drop=True)
                if bars.empty:
                    continue
                for start, stop in datasets.segments(bars.ts, self.tf, ctx + hmax):
                    seg = bars.iloc[start:stop].reset_index(drop=True)
                    x = feature_frame(seg, self.lags, self.vol_windows)
                    close = np.log(seg["close"].to_numpy())
                    y = pd.DataFrame(
                        {h: pd.Series(close).shift(-h) - pd.Series(close) for h in horizons}
                    )
                    keep = (
                        x.notna().all(axis=1)
                        & y.notna().all(axis=1)
                        & (seg.ts.to_numpy("datetime64[ns]").astype("int64") % grid == 0)
                    )
                    feats.append(x[keep])
                    targets.append(y[keep])
        if not feats:
            raise ValueError(f"lightgbm: no training rows for {self.tf}")
        x = pd.concat(feats, ignore_index=True)
        y = pd.concat(targets, ignore_index=True)
        limit = self.cfg.lightgbm["max_train_rows"]
        if len(x) > limit:  # deterministic thinning, spread over the whole period
            idx = np.unique(np.linspace(0, len(x) - 1, limit).round().astype(int))
            x, y = x.iloc[idx], y.iloc[idx]
        return x, y

    def fit(self, horizons: Sequence[int]) -> None:
        """Native LightGBM API — the sklearn wrapper would drag scikit-learn in for nothing."""
        import lightgbm as lgb

        x, y = self._training_rows(horizons)
        params = {"objective": "regression", "verbose": -1, "seed": self.cfg.seed,
                  **self.cfg.lightgbm["params"]}
        for h in horizons:
            model = lgb.train(
                params,
                lgb.Dataset(x, label=y[h]),
                num_boost_round=self.cfg.lightgbm["num_boost_round"],
            )
            self.models[h] = model
            self.sigma[h] = float(np.std(y[h].to_numpy() - model.predict(x), ddof=1))
        self.n_train_rows = len(x)

    def forecast(self, windows, horizons, samples, seed) -> np.ndarray:
        if not self.models:
            self.fit(horizons)
        rng = np.random.default_rng(window_seed(seed, self.name, windows[0]))
        x = pd.DataFrame(
            [feature_frame(w.context, self.lags, self.vol_windows).iloc[-1] for w in windows]
        )
        out = np.empty((len(windows), samples, len(horizons)))
        for j, h in enumerate(horizons):
            mu = self.models[h].predict(x)
            noise = rng.standard_normal((len(windows), samples))
            out[:, :, j] = mu[:, None] + self.sigma[h] * noise
        return out


class AxiomForecaster:
    """Monte-Carlo forecast from a registered Axiom/Kronos checkpoint.

    One `generate` call per window with `sample_count` paths. Batching the MC
    dimension across symbols is P4-03 — deliberately not done here, so the harness
    stays comparable before and after that optimization.
    """

    def __init__(self, name: str, cfg: EvalConfig, device: str | None = None, predictor=None):
        from axiom_model import load_predictor

        self.name = name
        self.cfg = cfg
        self.predictor = predictor if predictor is not None else load_predictor(name, device=device)

    def forecast(self, windows, horizons, samples, seed) -> np.ndarray:
        import torch

        mc, hmax = self.cfg.mc, max(horizons)
        idx = [h - 1 for h in horizons]
        out = np.empty((len(windows), samples, len(horizons)))
        for k, w in enumerate(windows):
            context = normalization.ensure_amount(w.context)
            x = context[normalization.FEATURES].to_numpy(np.float32)
            mean, std = normalization.fit(x)
            x_stamp = normalization.time_features(context.ts).to_numpy(np.float32)
            future_ts = w.anchor + np.arange(1, hmax + 1) * timeframe_delta(w.tf)
            y_stamp = normalization.time_features(pd.Series(future_ts)).to_numpy(np.float32)

            torch.manual_seed(window_seed(seed, self.name, w))
            paths = self.predictor.generate(
                normalization.apply(x, mean, std)[None],
                x_stamp[None],
                y_stamp[None],
                pred_len=hmax,
                T=mc["temperature"],
                top_k=mc.get("top_k", 0),
                top_p=mc["top_p"],
                sample_count=samples,
                verbose=False,
                reduce="none",
            )
            close = normalization.invert(paths[0], mean, std)[..., CLOSE]
            out[k] = np.log(np.maximum(close[:, idx], PRICE_FLOOR) / w.last_close)
        return out


BASELINES = {"persistence": Persistence, "ewma": Ewma, "lightgbm": LightGbm}


def build(
    cfg: EvalConfig,
    data_cfg: datasets.DataConfig,
    tf: str,
    root: Path = store.DEFAULT_ROOT,
    device: str | None = None,
    models: Sequence[str] | None = None,
    symbols: Sequence[str] | None = None,
) -> list:
    """Every forecaster the config asks for, baselines first (they are cheap)."""
    wanted = set(models) if models else None
    out = []
    for name, enabled in cfg.baselines.items():
        if not enabled or (wanted and name not in wanted):
            continue
        if name not in BASELINES:
            raise ValueError(f"baseline {name!r} is not implemented (P2-08 is optional)")
        cls = BASELINES[name]
        out.append(cls(cfg, data_cfg, tf, root, symbols) if name == "lightgbm" else cls(cfg))
    for name in cfg.models:
        if wanted is None or name in wanted:
            out.append(AxiomForecaster(name, cfg, device=device))
    return out


__all__ = [
    "AxiomForecaster",
    "BASELINES",
    "Ewma",
    "LightGbm",
    "Persistence",
    "build",
    "feature_frame",
    "window_seed",
]

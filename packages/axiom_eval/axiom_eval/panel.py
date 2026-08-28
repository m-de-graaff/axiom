"""Which windows get scored, and the leakage rules that decide it (P2-05, P2-11).

Anchors sit on a *shared* epoch-aligned time grid: every symbol is evaluated at the
same timestamps, which is what makes a cross-sectional RankIC meaningful in the
first place. Grid alignment is computed from the bar's epoch nanoseconds, so two
machines pick the same anchors without exchanging anything but the config.

Leakage checklist (CLAUDE.md rule 3), enforced as asserts in `_check`:
  * every window -- context *and* horizon -- lies inside the split bounds returned
    by `axiom_data.datasets.split_bounds` (which is where the embargo is applied);
  * no window spans a data gap (segments come from `axiom_data.datasets.segments`,
    the same function the dataset builder uses);
  * a forecaster is handed `Window.context` only; realized bars live in
    `Window.future_close`, which nothing on the forecasting path may read;
  * normalization statistics come from the context window only -- guaranteed by
    every forecaster going through `axiom_data.normalization`;
  * the universe is fixed ex ante by `configs/universe_v1.yaml` via the data config.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from axiom_data import datasets, store
from axiom_data.resample import timeframe_delta


@dataclass(frozen=True)
class EvalConfig:
    """Resolved `configs/eval/*.yaml`. The harness reads nothing else."""

    name: str
    seed: int
    data: str
    split: str
    models: list[str]
    baselines: dict
    mc: dict
    costs: dict
    metrics: list[str]
    slices: list[str]
    panel: dict
    ewma: dict
    lightgbm: dict
    strategy: dict
    wandb: dict
    report_dir: str
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def round_trip_cost(self) -> float:
        """Round-trip cost as a fraction: in and out, fee plus slippage each way."""
        return 2 * (self.costs["taker_fee_bps"] + self.costs["slippage_bps"]) / 1e4

    @property
    def horizons(self) -> list[int]:
        return list(self.panel["horizons"])

    @property
    def max_horizon(self) -> int:
        return max(self.horizons)


def load_config(path: str | Path) -> tuple[EvalConfig, datasets.DataConfig]:
    """Load the eval config plus the data config it points at."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    data_cfg = datasets.load_config(raw["data"])
    cfg = EvalConfig(
        name=Path(path).stem,
        seed=raw["seed"],
        data=raw["data"],
        split=raw["split"],
        models=list(raw["models"]),
        baselines=dict(raw["baselines"]),
        mc=dict(raw["mc"]),
        costs=dict(raw["costs"]),
        metrics=list(raw["metrics"]),
        slices=list(raw["slices"]),
        panel=dict(raw["panel"]),
        ewma=dict(raw.get("ewma", {})),
        lightgbm=dict(raw.get("lightgbm", {})),
        strategy=dict(raw.get("strategy", {})),
        wandb=dict(raw.get("wandb", {})),
        report_dir=raw["report_dir"],
        raw=raw,
    )
    missing = set(cfg.horizons) - set(data_cfg.horizons)
    if missing:
        raise ValueError(f"eval horizons {sorted(missing)} not in {cfg.data}: {data_cfg.horizons}")
    return cfg, data_cfg


@dataclass(frozen=True)
class Window:
    """One scored forecast origin. Forecasters see `context`; nothing else."""

    symbol: str
    tf: str
    anchor: pd.Timestamp  # timestamp of the last context bar
    context: pd.DataFrame  # `context_bars` rows ending at `anchor`, inclusive
    future_close: np.ndarray  # closes of the `max_horizon` bars after `anchor`

    @property
    def last_close(self) -> float:
        return float(self.context["close"].iloc[-1])

    def realized(self, horizons: Sequence[int]) -> np.ndarray:
        """Realized cumulative log returns at `horizons`, from the last context close."""
        return np.log(self.future_close[[h - 1 for h in horizons]] / self.last_close)

    def context_vol(self, bars: int) -> float:
        """Ex-ante realized vol: std of log returns over the last `bars` context bars."""
        close = self.context["close"].to_numpy()[-(bars + 1) :]
        return float(np.std(np.diff(np.log(close)), ddof=1))


def anchor_positions(
    ts: pd.Series, tf: str, context_bars: int, max_horizon: int, stride_bars: int
) -> np.ndarray:
    """Positions in `ts` that can anchor a leak-free window, on the shared grid."""
    grid = int((stride_bars * timeframe_delta(tf)).value)
    epoch = ts.to_numpy("datetime64[ns]").astype("int64")
    out = []
    for start, stop in datasets.segments(ts, tf, context_bars + max_horizon):
        idx = np.arange(start + context_bars - 1, stop - max_horizon)
        if idx.size:
            out.append(idx[epoch[idx] % grid == 0])
    return np.concatenate(out) if out else np.zeros(0, dtype=int)


def _thin(anchors: np.ndarray, limit: int) -> np.ndarray:
    """Evenly spaced subsample, deterministic — keeps the whole split represented."""
    if limit <= 0 or len(anchors) <= limit:
        return anchors
    return anchors[np.unique(np.linspace(0, len(anchors) - 1, limit).round().astype(int))]


def load_split_bars(
    cfg: EvalConfig,
    data_cfg: datasets.DataConfig,
    tf: str,
    root: Path = store.DEFAULT_ROOT,
    symbols: Sequence[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Bars of the evaluated split, per symbol, clipped to the embargoed bounds."""
    lo, hi = datasets.split_bounds(data_cfg, cfg.split, tf)
    bars = {}
    for symbol in symbols or data_cfg.symbols:
        part = store.read(symbol, tf, root=root, venue=data_cfg.venue, start=lo, end=hi)
        if not part.empty:
            bars[symbol] = part.reset_index(drop=True)
    return bars


def cross_sections(
    cfg: EvalConfig,
    data_cfg: datasets.DataConfig,
    tf: str,
    bars: dict[str, pd.DataFrame],
    chunk: tuple[int, int] | None = None,
) -> Iterator[tuple[pd.Timestamp, list[Window]]]:
    """Yield `(anchor, windows)` — one leak-checked cross-section per anchor timestamp.

    `chunk=(i, n)` takes every n-th anchor starting at `i`, so a long run can be split
    across containers without any of them losing sight of the whole split (each chunk
    is interleaved, not a contiguous block — year and vol slices stay populated).
    Chunking cannot change a number: forecasters are seeded per window.
    """
    lo, hi = datasets.split_bounds(data_cfg, cfg.split, tf)
    ctx, hmax = data_cfg.context_bars, cfg.max_horizon
    per_symbol = {
        symbol: set(
            part.ts.to_numpy()[
                anchor_positions(part.ts, tf, ctx, hmax, cfg.panel["stride_bars"])
            ]
        )
        for symbol, part in bars.items()
    }
    grid = sorted({t for anchors in per_symbol.values() for t in anchors})
    anchors = _thin(np.array(grid), cfg.panel["max_anchors"])
    if chunk is not None:
        index, total = chunk
        anchors = anchors[index::total]
    for anchor in anchors:
        windows = []
        for symbol, valid in per_symbol.items():
            if anchor not in valid:
                continue
            part = bars[symbol]
            i = int(np.searchsorted(part.ts.to_numpy(), anchor))
            window = Window(
                symbol=symbol,
                tf=tf,
                anchor=pd.Timestamp(anchor),
                context=part.iloc[i - ctx + 1 : i + 1].reset_index(drop=True),
                future_close=part["close"].to_numpy()[i + 1 : i + 1 + hmax],
            )
            _check(window, tf, ctx, hmax, lo, hi)
            windows.append(window)
        if windows:
            yield pd.Timestamp(anchor), windows


def _check(
    window: Window, tf: str, context_bars: int, max_horizon: int, lo, hi
) -> None:
    """The leakage checklist, as asserts. Never soften these to make a run finish."""
    step = timeframe_delta(tf)
    assert len(window.context) == context_bars, "short context window"
    assert len(window.future_close) == max_horizon, "short horizon"
    assert window.context.ts.iloc[-1] == window.anchor, "context does not end at the anchor"
    assert (window.context.ts.diff().iloc[1:] == step).all(), "context spans a data gap"
    assert window.context.ts.iloc[0] >= lo, "context starts before the split (embargo breach)"
    assert window.anchor + max_horizon * step <= hi, "horizon runs past the split"
    assert np.isfinite(window.future_close).all(), "non-finite realized closes"


__all__ = [
    "EvalConfig",
    "Window",
    "anchor_positions",
    "cross_sections",
    "load_config",
    "load_split_bars",
]

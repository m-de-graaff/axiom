"""Forecast-quality metrics (P2-01..05) and the cost tripwire (P2-09).

Everything here reads one long DataFrame, the *panel*, with one row per scored
(model, tf, horizon, anchor, symbol):

    model tf horizon ts symbol pred realized q_lo q_hi pit ctx_vol

`pred` and `realized` are cumulative log returns over the horizon. Costs are never
optional: a directional hit that does not clear the round trip is not a hit
(CLAUDE.md rule 4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from axiom_data.resample import timeframe_delta
from scipy import stats

PANEL_COLUMNS = [
    "model",
    "tf",
    "horizon",
    "ts",
    "symbol",
    "pred",
    "realized",
    "q_lo",
    "q_hi",
    "pit",
    "ctx_vol",
]


def rankic(panel: pd.DataFrame, min_symbols: int = 10) -> dict:
    """Cross-sectional Spearman per timestamp, then averaged (P2-01).

    The t-stat is the plain IID one over timestamps: mean / (std / sqrt(n)). With
    non-overlapping anchors that is honest; overlap inflates it, so keep the anchor
    stride at or above the horizon.
    """
    ics = []
    for _, group in panel.groupby("ts", sort=True):
        if len(group) < min_symbols or group["pred"].nunique() < 2:
            continue
        ic = stats.spearmanr(group["pred"], group["realized"]).statistic
        if np.isfinite(ic):
            ics.append(ic)
    if len(ics) < 2:
        return {"rankic": float("nan"), "rankic_t": float("nan"), "n_cross_sections": len(ics)}
    ics = np.array(ics)
    sd = ics.std(ddof=1)
    return {
        "rankic": float(ics.mean()),
        "rankic_std": float(sd),
        "rankic_t": float(ics.mean() / (sd / np.sqrt(len(ics)))) if sd > 0 else float("inf"),
        "rankic_hit_rate": float((ics > 0).mean()),
        "n_cross_sections": len(ics),
    }


def directional(panel: pd.DataFrame, cost: float) -> dict:
    """Directional accuracy *on the forecasts that clear round-trip costs* (P2-02)."""
    signal = panel[panel["pred"].abs() > cost]
    if signal.empty:
        return {"dir_acc_cost": float("nan"), "n_signals": 0, "signal_rate": 0.0,
                "net_edge_bps": float("nan")}
    hit = np.sign(signal["pred"]) == np.sign(signal["realized"])
    net = np.sign(signal["pred"]) * signal["realized"] - cost
    return {
        "dir_acc_cost": float(hit.mean()),
        "dir_acc_all": float((np.sign(panel["pred"]) == np.sign(panel["realized"])).mean()),
        "n_signals": int(len(signal)),
        "signal_rate": float(len(signal) / len(panel)),
        "net_edge_bps": float(net.mean() * 1e4),
    }


def errors(panel: pd.DataFrame) -> dict:
    """MAE/RMSE on log returns (P2-03) — never on prices."""
    err = panel["pred"] - panel["realized"]
    return {"mae_logret": float(err.abs().mean()), "rmse_logret": float(np.sqrt((err**2).mean()))}


def calibration(
    panel: pd.DataFrame, band: tuple[float, float] = (0.1, 0.9), bins: int = 10
) -> dict:
    """Band coverage + PIT uniformity (P2-04).

    Nominal coverage for a 10-90 band is 0.8. A KS statistic far from 0 means the
    MC fan is the wrong shape, which makes every downstream P(up) fiction.
    """
    inside = (panel["realized"] >= panel["q_lo"]) & (panel["realized"] <= panel["q_hi"])
    pit = panel["pit"].dropna().to_numpy()
    ks = stats.kstest(pit, "uniform") if len(pit) > 1 else None
    hist, _ = np.histogram(pit, bins=bins, range=(0.0, 1.0))
    return {
        "coverage_10_90": float(inside.mean()),
        "coverage_nominal": float(band[1] - band[0]),
        "pit_ks": float(ks.statistic) if ks else float("nan"),
        "pit_ks_p": float(ks.pvalue) if ks else float("nan"),
        "pit_hist": (hist / max(hist.sum(), 1)).round(4).tolist(),
    }


def tripwire(panel: pd.DataFrame, cost: float, threshold_mult: float = 1.0) -> dict:
    """Dumb long/flat threshold strategy with fees + slippage (P2-09).

    Go long when the forecast clears `threshold_mult` x round-trip cost, hold the
    horizon out, pay the round trip. Equal weight, no compounding, no sizing, no
    portfolio construction — this is a tripwire that turns "nice RankIC" into "does
    it survive costs at all?", not a backtest. The backtest is nautilus_trader in
    Phase 8. Sharpe here is per-trade, annualized on the number of bars a trade
    holds, and is therefore optimistic about overlap.
    """
    trades = panel[panel["pred"] > threshold_mult * cost]
    if trades.empty:
        return {"n_trades": 0, "net_return_bps": float("nan"), "win_rate": float("nan"),
                "sharpe": float("nan"), "total_net": 0.0}
    net = trades["realized"] - cost
    hold = trades["horizon"].mean() * timeframe_delta(trades["tf"].iloc[0])
    per_year = pd.Timedelta("365D") / hold
    sd = net.std(ddof=1) if len(net) > 1 else 0.0
    return {
        "n_trades": int(len(trades)),
        "net_return_bps": float(net.mean() * 1e4),
        "win_rate": float((net > 0).mean()),
        "sharpe": float(net.mean() / sd * np.sqrt(per_year)) if sd > 0 else float("nan"),
        "total_net": float(net.sum()),
    }


def summarize(panel: pd.DataFrame, cost: float, min_symbols: int = 10, **kwargs) -> dict:
    """Every headline metric for one (model, tf, horizon) slab of the panel."""
    return {
        "n": int(len(panel)),
        "n_symbols": int(panel["symbol"].nunique()),
        **rankic(panel, min_symbols),
        **directional(panel, cost),
        **errors(panel),
        **calibration(panel, **kwargs),
    }


def add_slices(panel: pd.DataFrame) -> pd.DataFrame:
    """Year and realized-vol tercile (P2-05).

    Tercile edges are cut per (tf, horizon) over the whole evaluated panel — a
    post-hoc *slicing* choice, not a fitted one, and `ctx_vol` itself is ex ante.
    """
    panel = panel.copy()
    panel["year"] = pd.to_datetime(panel["ts"]).dt.year
    panel["vol_tercile"] = (
        panel.groupby(["tf", "horizon"])["ctx_vol"]
        .transform(lambda v: pd.qcut(v, 3, labels=["low", "mid", "high"], duplicates="drop"))
        .astype(str)
    )
    return panel


def by_slice(panel: pd.DataFrame, cost: float, key: str, min_symbols: int = 10) -> pd.DataFrame:
    """`summarize` per (model, tf, horizon, <slice>)."""
    rows = []
    for (model, tf, horizon, value), group in panel.groupby(
        ["model", "tf", "horizon", key], sort=True, observed=True
    ):
        rows.append(
            {"model": model, "tf": tf, "horizon": horizon, key: value,
             **summarize(group, cost, min_symbols)}
        )
    return pd.DataFrame(rows)


def table(
    panel: pd.DataFrame, cost: float, min_symbols: int = 10, threshold_mult: float = 1.0
) -> pd.DataFrame:
    """The headline comparison table: one row per model x timeframe x horizon."""
    rows = []
    for (model, tf, horizon), group in panel.groupby(["model", "tf", "horizon"], sort=True):
        rows.append(
            {"model": model, "tf": tf, "horizon": horizon,
             **summarize(group, cost, min_symbols),
             **{f"tw_{k}": v for k, v in tripwire(group, cost, threshold_mult).items()}}
        )
    return pd.DataFrame(rows).sort_values(
        ["tf", "horizon", "rankic"], ascending=[True, True, False]
    )


__all__ = [
    "PANEL_COLUMNS",
    "add_slices",
    "by_slice",
    "calibration",
    "directional",
    "errors",
    "rankic",
    "summarize",
    "table",
    "tripwire",
]

"""Self-contained HTML report (P2-10).

Plain string formatting over `DataFrame.to_html` — a report nobody can open without
a toolchain is a report nobody reads. The caveats are printed at the top on
purpose: every number below them is conditional on things a reader will otherwise
forget.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CAVEATS = [
    "<b>Pretraining leakage.</b> Kronos (<code>axiom-zero-*</code>) was pretrained on 45 "
    "exchanges' history through ~2025. Zero-shot numbers on test data up to 2025 are "
    "optimistic; weight conclusions toward the most recent, definitely-post-training months.",
    "<b>Survivorship.</b> The universe is a survivor set — candidates were pairs Binance "
    "lists today, so coins delisted mid-history are absent. XMRUSDT and WAVESUSDT were "
    "delisted during the test window and contribute no test windows.",
    "<b>The tripwire is not a backtest.</b> Equal weight, no sizing, no portfolio "
    "construction, per-trade Sharpe annualized on holding period and optimistic about "
    "overlapping trades. The real cost-aware replay is nautilus_trader in Phase 8.",
    "<b>All returns are log returns</b>, and every directional number is net of the "
    "round-trip cost stated below.",
]

CSS = """
body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:2rem auto;
     max-width:1200px;color:#111}
h1{margin-bottom:0} h2{margin-top:2.5rem;border-bottom:1px solid #ddd;padding-bottom:.3rem}
table{border-collapse:collapse;font-size:13px;margin:.5rem 0}
th,td{border:1px solid #ddd;padding:4px 8px;text-align:right}
th{background:#f5f5f5} td:first-child,th:first-child{text-align:left}
.meta{color:#555;font-size:13px}
.caveats{background:#fff8e1;border-left:4px solid #f0b400;padding:.75rem 1rem}
.caveats li{margin:.35rem 0} code{background:#f0f0f0;padding:0 3px}
"""


def _table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, float_format=lambda v: f"{v:.4f}", na_rep="—", border=0)


def write_html(
    path: str | Path, meta: dict, headline: pd.DataFrame, slices: dict[str, pd.DataFrame]
) -> Path:
    """Write the report. `meta` is the same dict that goes into `metrics.json`."""
    env = meta["environment"]
    facts = {
        "run": meta["run_id"],
        "config": meta["config"],
        "git SHA": meta["git_sha"],
        "dataset": f"{meta['dataset']} ({meta['dataset_hash'][:12]}…)",
        "split": meta["split"],
        "round-trip cost": f"{meta['round_trip_cost_bps']} bps",
        "machine": f"{env['platform']} · torch {env.get('torch')} · "
                   f"{env.get('gpu', env['device'])}",
    }
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>axiom eval — {meta['run_id']}</title><style>{CSS}</style>",
        f"<h1>Axiom eval</h1><p class='meta'>{meta['created_at']}</p>",
        "<table>"
        + "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in facts.items())
        + "</table>",
        "<div class='caveats'><ul>"
        + "".join(f"<li>{c}</li>" for c in CAVEATS)
        + "</ul></div>",
        "<h2>Headline — model × timeframe × horizon</h2>",
        _table(headline),
    ]
    for key, frame in slices.items():
        parts += [f"<h2>Slice: {key}</h2>", _table(frame)]
    parts += [
        "<h2>Config</h2><pre>",
        json.dumps(meta["eval_config"], indent=2),
        "</pre>",
    ]
    path = Path(path)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


__all__ = ["CAVEATS", "write_html"]

"""Post-clean views: what the corpus actually contains once cleaning has had its say.

The registry answers what we *pulled*. These answer what is **usable**, which is a different and
smaller number, and it is the one v0.5 sizes the tokenizer corpus against. Keeping them here
rather than in :mod:`axiom.registry.reports` is deliberate: they read the segment index, not the
registry, and a report that needs cleaning's output belongs next to cleaning.

Group-bys over a few tens of thousands of rows. numpy, not SQL -- a query engine earns its place
when the question is arbitrary, and `axiom registry query` is where arbitrary lives.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pyarrow as pa

#: Context length the v0.5 tokenizer and v0.7 decoder train at (roadmap §1).
DEFAULT_CONTEXT = 512


def usable_bars(segments: pa.Table) -> list[dict[str, Any]]:
    """Segments, bars and usable windows per source x asset_class x market x frequency.

    A bar count alone overstates the corpus: a hundred segments of 300 bars hold 30 000 bars and
    zero context-512 windows. Both numbers are reported side by side so the gap between them is
    visible rather than discovered later.
    """
    return _group(segments, ("source", "asset_class", "market", "frequency"))


def usable_by_frequency(segments: pa.Table) -> list[dict[str, Any]]:
    """The same, collapsed to source x frequency. The corpus-sizing table."""
    return _group(segments, ("source", "frequency"))


def _group(segments: pa.Table, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    if segments.num_rows == 0:
        return []
    columns = [segments[name].to_pylist() for name in keys]
    n_bars = segments["n_bars"].to_pylist()
    symbols = segments["symbol"].to_pylist()

    groups: dict[tuple, dict[str, Any]] = {}
    members: dict[tuple, set[str]] = defaultdict(set)
    for i, key in enumerate(zip(*columns, strict=True)):
        group = groups.setdefault(
            key,
            {
                **dict(zip(keys, key, strict=True)),
                "segments": 0,
                "bars": 0,
                "windows_512": 0,
                "longest_segment": 0,
            },
        )
        group["segments"] += 1
        group["bars"] += n_bars[i]
        group["windows_512"] += max(n_bars[i] - DEFAULT_CONTEXT + 1, 0)
        group["longest_segment"] = max(group["longest_segment"], n_bars[i])
        members[key].add(symbols[i])
    for key, group in groups.items():
        group["series"] = len(members[key])
    return [groups[k] for k in sorted(groups)]


def drop_rates(dropstats: pa.Table) -> list[dict[str, Any]]:
    """Per source x frequency x rule: bars dropped, runs excised, and the share of the slice.

    The share is of the slice's *total* bars, so the five rule rows for one slice are directly
    comparable and sum to that slice's overall loss.
    """
    if dropstats.num_rows == 0:
        return []
    rows = dropstats.to_pylist()
    totals: dict[tuple[str, str], int] = defaultdict(int)
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        marker = (row["source"], row["frequency"], row["artifact_path"])
        if marker in seen:
            continue
        seen.add(marker)
        totals[(row["source"], row["frequency"])] += row["total_bars"]

    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["source"], row["frequency"], row["rule"])
        group = groups.setdefault(
            key,
            {
                "source": key[0],
                "frequency": key[1],
                "rule": key[2],
                "bars_dropped": 0,
                "runs_excised": 0,
                "segments_dropped": 0,
            },
        )
        group["bars_dropped"] += row["bars_dropped"]
        group["runs_excised"] += row["runs_excised"]
        group["segments_dropped"] += row["segments_dropped"]
    for key, group in groups.items():
        total = totals[(key[0], key[1])]
        group["total_bars"] = total
        group["pct_of_slice"] = round(100.0 * group["bars_dropped"] / total, 4) if total else 0.0
    return [groups[k] for k in sorted(groups)]


def most_cut_series(dropstats: pa.Table, limit: int = 20) -> list[dict[str, Any]]:
    """The series that lost the most, as a share of their own bars.

    The Phase F gate asks for the top twenty to be looked at by a human, one line of verdict each:
    data problem, real market pathology, or rule artifact. This is that list.
    """
    if dropstats.num_rows == 0:
        return []
    per_series: dict[str, dict[str, Any]] = {}
    for row in dropstats.to_pylist():
        entry = per_series.setdefault(
            row["artifact_path"],
            {
                "artifact_path": row["artifact_path"],
                "source": row["source"],
                "symbol": row["symbol"],
                "frequency": row["frequency"],
                "total_bars": row["total_bars"],
                "kept_bars": row["kept_bars"],
                "by_rule": {},
            },
        )
        if row["bars_dropped"]:
            entry["by_rule"][row["rule"]] = row["bars_dropped"]
    entries = []
    for entry in per_series.values():
        total = entry["total_bars"]
        entry["dropped_bars"] = total - entry["kept_bars"]
        entry["pct_dropped"] = round(100.0 * entry["dropped_bars"] / total, 3) if total else 0.0
        entries.append(entry)
    entries.sort(key=lambda e: (-e["pct_dropped"], -e["dropped_bars"], e["artifact_path"]))
    return entries[:limit]


def red_flags(
    dropstats: pa.Table,
    *,
    majors: set[str] | None = None,
    major_limit_pct: float = 1.0,
    slice_limit_pct: float = 15.0,
) -> list[dict[str, Any]]:
    """The Phase F checks that must each be ticked pass or carry a written investigation.

    Three, from the plan: a major losing more than ``major_limit_pct`` of its bars; an FX weekend
    contributing any gap cuts at all (it must not -- the weekend is an expected gap); a source x
    frequency slice losing more than ``slice_limit_pct`` overall.
    """
    majors = majors or {"BTCUSDT", "ETHUSDT", "EURUSD", "USDJPY", "XAUUSD", "SPY", "AAPL"}
    flags: list[dict[str, Any]] = []

    for entry in most_cut_series(dropstats, limit=10_000):
        if entry["symbol"] in majors and entry["pct_dropped"] > major_limit_pct:
            flags.append(
                {
                    "check": "major_series_loss",
                    "subject": entry["artifact_path"],
                    "value": entry["pct_dropped"],
                    "limit": major_limit_pct,
                    "detail": f"{entry['symbol']} lost {entry['pct_dropped']}% of its bars",
                }
            )

    by_slice: dict[tuple[str, str], dict[str, Any]] = {}
    for row in drop_rates(dropstats):
        key = (row["source"], row["frequency"])
        acc = by_slice.setdefault(key, {"dropped": 0, "total": row["total_bars"]})
        acc["dropped"] += row["bars_dropped"]
        if row["rule"] == "gap" and row["source"] == "dukascopy" and row["segments_dropped"]:
            flags.append(
                {
                    "check": "fx_weekend_contributed_cuts",
                    "subject": f"{row['source']}/{row['frequency']}",
                    "value": row["segments_dropped"],
                    "limit": 0,
                    "detail": "the weekend is an expected gap and must produce no dropped segments",
                }
            )
    for (source, frequency), acc in sorted(by_slice.items()):
        pct = 100.0 * acc["dropped"] / acc["total"] if acc["total"] else 0.0
        if pct > slice_limit_pct:
            flags.append(
                {
                    "check": "slice_loss",
                    "subject": f"{source}/{frequency}",
                    "value": round(pct, 3),
                    "limit": slice_limit_pct,
                    "detail": f"{source} {frequency} lost {pct:.2f}% of its bars overall",
                }
            )
    return flags


def _n(value: int) -> str:
    return f"{value:,}"


def clean_summary_markdown(
    segments: pa.Table,
    dropstats: pa.Table,
    *,
    clean_config_hash: str,
    registry_hash: str = "",
) -> str:
    """The committed drop-stats report, built from the two tables alone."""
    usable = usable_bars(segments)
    total_segments = sum(g["segments"] for g in usable)
    total_bars = sum(g["bars"] for g in usable)
    total_windows = sum(g["windows_512"] for g in usable)
    flags = red_flags(dropstats)

    lines = [
        "# axiom clean run",
        "",
        f"`clean_config_hash` **{clean_config_hash}**"
        + (f" · `registry_hash` {registry_hash}" if registry_hash else ""),
        "",
        f"{_n(total_segments)} segments · {_n(total_bars)} usable bars · "
        f"{_n(total_windows)} context-512 windows",
        "",
        "## Usable corpus",
        "",
        "| Source | Asset class | Market | Freq | Series | Segments | Usable bars | "
        "Windows @512 | Longest |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for g in usable:
        lines.append(
            f"| {g['source']} | {g['asset_class']} | {g['market']} | {g['frequency']} | "
            f"{g['series']} | {_n(g['segments'])} | {_n(g['bars'])} | "
            f"{_n(g['windows_512'])} | {_n(g['longest_segment'])} |"
        )

    lines += [
        "",
        "## Drop rates by rule",
        "",
        "| Source | Freq | Rule | Bars dropped | Runs excised | Segments dropped | % of slice |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in drop_rates(dropstats):
        lines.append(
            f"| {row['source']} | {row['frequency']} | {row['rule']} | "
            f"{_n(row['bars_dropped'])} | {_n(row['runs_excised'])} | "
            f"{_n(row['segments_dropped'])} | {row['pct_of_slice']:.3f}% |"
        )

    lines += ["", "## Red flags", ""]
    if flags:
        lines += ["| Check | Subject | Value | Limit | Detail |", "|---|---|---:|---:|---|"]
        for flag in flags:
            lines.append(
                f"| {flag['check']} | `{flag['subject']}` | {flag['value']} | "
                f"{flag['limit']} | {flag['detail']} |"
            )
        lines += [
            "",
            "**Every row above needs an investigation written against it before the v0.3 gate.**",
        ]
    else:
        lines.append("None. All three checks pass.")

    lines += [
        "",
        "## Top 20 most-cut series",
        "",
        "| Artifact | Total | Kept | Dropped | % | By rule |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for entry in most_cut_series(dropstats):
        by_rule = ", ".join(f"{k}: {_n(v)}" for k, v in sorted(entry["by_rule"].items())) or "-"
        lines.append(
            f"| `{entry['artifact_path']}` | {_n(entry['total_bars'])} | "
            f"{_n(entry['kept_bars'])} | {_n(entry['dropped_bars'])} | "
            f"{entry['pct_dropped']:.2f}% | {by_rule} |"
        )

    lines += [
        "",
        "---",
        "",
        "Built by `axiom clean report` from `clean/v1/segments.parquet` and "
        "`clean/v1/dropstats.parquet`. Raw bars are unchanged; cleaning produced only this "
        "metadata (ADR-0018).",
    ]
    return "\n".join(lines) + "\n"

"""The canned questions, answered from the registry table.

Four of them, because four is what the roadmap's gate actually asks for: what do we have, from
where, pulled when, and how much room is it taking. They are plain group-bys over a few thousand
rows, so they are numpy rather than SQL -- a query engine earns its place when the question is
*arbitrary*, which is what `axiom registry query` is for.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime

import pyarrow as pa

MS_PER_DAY = 86_400_000

#: What corpus M0 requires: every slice, at the frequencies the roadmap names for it.
M0_SLICES: dict[tuple[str, str], tuple[str, ...]] = {
    ("crypto", "spot"): ("1h", "1d"),
    ("crypto", "um"): ("1h", "1d"),
    ("fx", "fx"): ("1h", "1d"),
    ("commodity", "commodity"): ("1h", "1d"),
    ("equity", "us"): ("1d",),
}


def _columns(table: pa.Table, *names: str) -> list[list]:
    return [table[name].to_pylist() for name in names]


def coverage_matrix(table: pa.Table) -> list[dict]:
    """Series and bars per source x asset_class x market x frequency, with the ts range.

    This is the answer to "what do we have and from where", and it is the row set the M0 verdict
    is read off.
    """
    keys = _columns(table, "source", "asset_class", "market", "frequency")
    rows, firsts, lasts, sizes = _columns(
        table, "row_count", "first_ts", "last_ts", "artifact_bytes"
    )

    groups: dict[tuple, dict] = {}
    for i, key in enumerate(zip(*keys, strict=True)):
        group = groups.setdefault(
            key,
            {
                "source": key[0],
                "asset_class": key[1],
                "market": key[2],
                "frequency": key[3],
                "series": 0,
                "bars": 0,
                "bytes": 0,
                "first_ts": firsts[i],
                "last_ts": lasts[i],
            },
        )
        group["series"] += 1
        group["bars"] += rows[i]
        group["bytes"] += sizes[i]
        group["first_ts"] = min(group["first_ts"], firsts[i])
        group["last_ts"] = max(group["last_ts"], lasts[i])
    return [groups[k] for k in sorted(groups)]


def storage_by_source(table: pa.Table) -> list[dict]:
    """Bytes and series per source. The headroom check against the 100 GB tier reads this."""
    sources, sizes, rows = _columns(table, "source", "artifact_bytes", "row_count")
    totals: dict[str, dict] = defaultdict(lambda: {"series": 0, "bytes": 0, "bars": 0})
    for source, size, row_count in zip(sources, sizes, rows, strict=True):
        totals[source]["series"] += 1
        totals[source]["bytes"] += size
        totals[source]["bars"] += row_count
    return [{"source": name, **totals[name]} for name in sorted(totals)]


def gappiest(table: pa.Table, limit: int = 15) -> list[dict]:
    """The series with the most missing grid slots.

    A gap is never an error -- it is a weekend, a halt, an outage, a listing that had not happened
    yet. This report exists so the *biggest* ones get looked at by a human once, rather than
    discovered by a model.
    """
    paths, symbols, sources, freqs, gaps, rows = _columns(
        table, "artifact_path", "symbol", "source", "frequency", "gap_count", "row_count"
    )
    entries = [
        {
            "artifact_path": paths[i],
            "symbol": symbols[i],
            "source": sources[i],
            "frequency": freqs[i],
            "gap_count": gaps[i],
            "row_count": rows[i],
        }
        for i in range(table.num_rows)
        if gaps[i] > 0
    ]
    entries.sort(key=lambda e: (-e["gap_count"], e["artifact_path"]))
    return entries[:limit]


def staleness(table: pa.Table, *, now_ms: int | None = None, limit: int = 15) -> list[dict]:
    """Days between each series' last bar and now, worst first.

    Distinguishes a series that stopped from a corpus that stopped being pulled -- the two look
    identical from a row count and have completely different causes.
    """
    now_ms = now_ms if now_ms is not None else int(datetime.now(UTC).timestamp() * 1000)
    paths, sources, symbols, freqs, lasts = _columns(
        table, "artifact_path", "source", "symbol", "frequency", "last_ts"
    )
    entries = [
        {
            "artifact_path": paths[i],
            "source": sources[i],
            "symbol": symbols[i],
            "frequency": freqs[i],
            "last_ts": lasts[i],
            "stale_days": round((now_ms - lasts[i]) / MS_PER_DAY, 1),
        }
        for i in range(table.num_rows)
    ]
    entries.sort(key=lambda e: (-e["stale_days"], e["artifact_path"]))
    return entries[:limit]


def m0_verdict(table: pa.Table) -> list[dict]:
    """Whether every corpus-M0 slice is present at the frequency the roadmap requires.

    Reported per slice rather than as one boolean, because "M0 is not assembled" is useless next
    to "commodities are missing at 1h".
    """
    present: dict[tuple[str, str], set[str]] = defaultdict(set)
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for asset_class, market, frequency in zip(
        *_columns(table, "asset_class", "market", "frequency"), strict=True
    ):
        present[(asset_class, market)].add(frequency)
        counts[(asset_class, market, frequency)] += 1

    verdict = []
    for (asset_class, market), required in M0_SLICES.items():
        have = present.get((asset_class, market), set())
        missing = [f for f in required if f not in have]
        verdict.append(
            {
                "asset_class": asset_class,
                "market": market,
                "required": list(required),
                "missing": missing,
                "series": {f: counts.get((asset_class, market, f), 0) for f in required},
                "ok": not missing,
            }
        )
    return verdict


def _fmt_ts(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).strftime("%Y-%m-%d")


def _gb(nbytes: int) -> str:
    return f"{nbytes / 1e9:.2f} GB"


def summary_markdown(
    table: pa.Table,
    *,
    registry_hash: str,
    bad_count: int = 0,
    now_ms: int | None = None,
) -> str:
    """The committed answer to what/from-where/pulled-when, built from the registry alone."""
    coverage = coverage_matrix(table)
    total_bars = sum(g["bars"] for g in coverage)
    total_bytes = sum(g["bytes"] for g in coverage)

    lines = [
        "# axiom-raw corpus registry",
        "",
        f"`registry_hash` **{registry_hash}** · {table.num_rows} artifacts · "
        f"{total_bars:,} bars · {_gb(total_bytes)}",
        "",
    ]
    if bad_count:
        lines += [
            f"> **{bad_count} sidecar(s) could not be read** and are listed in "
            "`registry/bad_sidecars.json`. They are absent from the table above, so every number "
            "here is a lower bound until they are fixed.",
            "",
        ]

    lines += [
        "## Coverage",
        "",
        "| Source | Asset class | Market | Freq | Series | Bars | First | Last | Size |",
        "|---|---|---|---|---:|---:|---|---|---:|",
    ]
    for g in coverage:
        lines.append(
            f"| {g['source']} | {g['asset_class']} | {g['market']} | {g['frequency']} | "
            f"{g['series']} | {g['bars']:,} | {_fmt_ts(g['first_ts'])} | "
            f"{_fmt_ts(g['last_ts'])} | {_gb(g['bytes'])} |"
        )

    lines += [
        "",
        "## Corpus M0",
        "",
        "| Asset class | Market | Required | Present | Verdict |",
        "|---|---|---|---|---|",
    ]
    for slice_ in m0_verdict(table):
        counts = ", ".join(f"{f}: {n}" for f, n in slice_["series"].items())
        status = "yes" if slice_["ok"] else f"MISSING {', '.join(slice_['missing'])}"
        lines.append(
            f"| {slice_['asset_class']} | {slice_['market']} | "
            f"{', '.join(slice_['required'])} | {counts} | {status} |"
        )

    lines += [
        "",
        "## Storage by source",
        "",
        "| Source | Series | Bars | Size |",
        "|---|---:|---:|---:|",
    ]
    for entry in storage_by_source(table):
        lines.append(
            f"| {entry['source']} | {entry['series']} | {entry['bars']:,} | {_gb(entry['bytes'])} |"
        )

    lines += ["", "## Gappiest series", "", "| Artifact | Gaps | Rows |", "|---|---:|---:|"]
    for entry in gappiest(table):
        lines.append(
            f"| `{entry['artifact_path']}` | {entry['gap_count']:,} | {entry['row_count']:,} |"
        )
    if not gappiest(table):
        lines.append("| _no gaps anywhere_ | 0 | 0 |")

    lines += ["", "## Stalest series", "", "| Artifact | Last bar | Days stale |", "|---|---|---:|"]
    for entry in staleness(table, now_ms=now_ms):
        lines.append(
            f"| `{entry['artifact_path']}` | {_fmt_ts(entry['last_ts'])} | {entry['stale_days']} |"
        )

    lines += [
        "",
        "---",
        "",
        "Built by `axiom registry build` from the sidecar manifests in `axiom-raw`. The sidecars "
        "are the truth; this file is a cache with no authority, and rebuilding it from an "
        "unchanged tier reproduces the same `registry_hash`.",
    ]
    return "\n".join(lines) + "\n"

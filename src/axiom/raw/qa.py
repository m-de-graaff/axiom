"""Verification and QA over the raw tier.

Two questions, both asked of `axiom-raw` rather than of the code that filled it.

`verify_series` asks whether a stored artifact can still be reproduced: re-download exactly the
archives its manifest names, re-verify their checksums, rebuild the Parquet in memory, and compare
the hash. That is the v0.1 exit gate's headline condition, and it is a stronger claim than "the
tests pass" because it re-runs the whole path against the real upstream.

`corpus_stats` asks what is actually in there — series counts, history lengths, gap distributions
— from the sidecars alone, so it costs a few hundred small reads rather than a re-download.

A **drift** is not a failure. A series that has gained days since it was pulled reproduces its
recorded bytes exactly and also has more data available now. Both facts are reported.
"""

from __future__ import annotations

import logging
import random
import statistics
from dataclasses import dataclass, field

from axiom.provenance.manifest import FileManifest, sha256_bytes
from axiom.raw.store import RawStore
from axiom.schema.bars import bars_metadata
from axiom.sources.binance import (
    ASSET_CLASS,
    SOURCE,
    PullTask,
    artifact_path,
    build_table,
    enumerate_sources,
    write_parquet,
)
from axiom.sources.binance_vision import BinanceVision

log = logging.getLogger("axiom.raw.qa")

MS_PER_DAY = 86_400_000


@dataclass
class VerifyResult:
    """What re-deriving one series from its manifest showed."""

    task: PullTask
    status: str  # identical | drifted | mismatch | failed
    byte_identical: bool = False
    new_archives: int = 0
    recorded_last_ts: int = 0
    available_last_ts: int = 0
    recorded_rows: int = 0
    detail: str = ""

    @property
    def ok(self) -> bool:
        """Whether the gate is satisfied. Drift is expected; a hash mismatch is not."""
        return self.status in ("identical", "drifted")

    def line(self) -> str:
        if self.status == "identical":
            return f"{self.task}: byte-identical ({self.recorded_rows} rows)"
        if self.status == "drifted":
            return (
                f"{self.task}: byte-identical, {self.new_archives} new archive(s) since the pull "
                f"(last_ts {self.recorded_last_ts} -> {self.available_last_ts})"
            )
        return f"{self.task}: {self.status.upper()} — {self.detail}"


def verify_series(client: BinanceVision, store: RawStore, task: PullTask) -> VerifyResult:
    """Rebuild one series from exactly the archives its manifest names, and compare the bytes.

    The rebuild uses the manifest's own URL list, not a fresh enumeration. Using a fresh one would
    fold two questions together — "is this reproducible" and "has the upstream grown" — and the
    second one would mask the first every time a day passed.
    """
    path = artifact_path(task.market, task.frequency, task.symbol)
    try:
        recorded = store.read_sidecar(path)
        if recorded is None:
            return VerifyResult(task, "failed", detail="no sidecar in the raw tier")

        table = build_table(client, task, recorded.source_urls, recorded.source_sha256s)
        rebuilt = write_parquet(
            table,
            bars_metadata(
                source=SOURCE,
                asset_class=ASSET_CLASS,
                market=task.market,
                symbol=task.symbol,
                frequency=task.frequency,
                manifest_sha256=recorded.manifest_sha256,
            ),
        )
        identical = sha256_bytes(rebuilt) == recorded.artifact_sha256

        available = enumerate_sources(client, task)
        new_archives = max(0, len(available) - len(recorded.source_urls))

        result = VerifyResult(
            task,
            "identical" if identical else "mismatch",
            byte_identical=identical,
            new_archives=new_archives,
            recorded_last_ts=recorded.last_ts,
            available_last_ts=recorded.last_ts,
            recorded_rows=recorded.row_count,
        )
        if not identical:
            result.detail = (
                f"recorded artifact_sha256={recorded.artifact_sha256}, rebuilt "
                f"{sha256_bytes(rebuilt)} from the same {len(recorded.source_urls)} archives"
            )
            return result

        if new_archives:
            # A grown daily tail is a documented manifest diff, not a broken artifact. The last
            # timestamp the tail would reach is one grid step per new daily archive at most; the
            # honest thing to report is the archive count and the recorded range.
            result.status = "drifted"
            result.available_last_ts = recorded.last_ts + new_archives * MS_PER_DAY
        return result

    except Exception as exc:
        log.warning("verify failed for %s: %s", task, exc)
        return VerifyResult(task, "failed", detail=f"{type(exc).__name__}: {exc}")


def sample_tasks(manifests: list[FileManifest], n: int, seed: int = 1337) -> list[PullTask]:
    """A deterministic sample of series to verify. Same seed, same picks, forever."""
    tasks = sorted(
        (PullTask(m.market, m.symbol, m.frequency) for m in manifests),
        key=lambda t: (t.market, t.frequency, t.symbol),
    )
    return random.Random(seed).sample(tasks, min(n, len(tasks)))


# --- corpus statistics -------------------------------------------------------------------


@dataclass
class GroupStats:
    """One market x frequency slice of the corpus."""

    market: str
    frequency: str
    series: int = 0
    rows: int = 0
    bytes: int = 0
    history_days: list[float] = field(default_factory=list)
    gaps: list[int] = field(default_factory=list)
    off_grid: int = 0
    gappiest: list[tuple[str, int]] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        spans = sorted(self.history_days)
        return {
            "market": self.market,
            "frequency": self.frequency,
            "series": self.series,
            "rows": self.rows,
            "bytes": self.bytes,
            "history_days_min": round(spans[0], 1) if spans else 0.0,
            "history_days_median": round(statistics.median(spans), 1) if spans else 0.0,
            "history_days_max": round(spans[-1], 1) if spans else 0.0,
            "gap_total": sum(self.gaps),
            "off_grid_total": self.off_grid,
            "gap_median": statistics.median(self.gaps) if self.gaps else 0,
            "series_without_gaps": sum(1 for g in self.gaps if g == 0),
        }


def corpus_stats(manifests: list[FileManifest]) -> list[GroupStats]:
    """Group the sidecars by market and frequency and reduce each group."""
    groups: dict[tuple[str, str], GroupStats] = {}
    for manifest in manifests:
        key = (manifest.market, manifest.frequency)
        group = groups.setdefault(key, GroupStats(*key))
        group.series += 1
        group.rows += manifest.row_count
        group.gaps.append(manifest.gap_count)
        group.off_grid += manifest.off_grid_count
        group.gappiest.append((manifest.symbol, manifest.gap_count))
        group.history_days.append((manifest.last_ts - manifest.first_ts) / MS_PER_DAY)
    for group in groups.values():
        group.gappiest = sorted(group.gappiest, key=lambda pair: (-pair[1], pair[0]))[:10]
    return [groups[key] for key in sorted(groups)]


def pairs_meeting_history(manifests: list[FileManifest], min_days: int = 365) -> dict[str, set]:
    """Symbols per market present at **both** 1h and 1d with enough history — the v0.1 gate.

    Both frequencies, because a pair that exists hourly and not daily is not a pair the corpus
    can use: the two frequencies are what the frequency-conditioning embedding contrasts.
    """
    by_market: dict[str, dict[str, set]] = {}
    for manifest in manifests:
        span_days = (manifest.last_ts - manifest.first_ts) / MS_PER_DAY
        if span_days < min_days:
            continue
        by_market.setdefault(manifest.market, {}).setdefault(manifest.frequency, set()).add(
            manifest.symbol
        )
    return {
        market: frequencies.get("1h", set()) & frequencies.get("1d", set())
        for market, frequencies in by_market.items()
    }


def stats_markdown(
    manifests: list[FileManifest],
    verify_results: list[VerifyResult] | None = None,
    *,
    title: str = "v0.1 raw-tier QA report",
) -> str:
    """The committed QA report. Numbers only — the eyeball pass is written in by hand."""
    groups = corpus_stats(manifests)
    lines = [f"# {title}", ""]

    total_rows = sum(g.rows for g in groups)
    total_series = sum(g.series for g in groups)
    lines += [
        f"{total_series} series, {total_rows:,} bars, "
        f"{len({m.symbol for m in manifests})} distinct symbols.",
        "",
        "## Per market and frequency",
        "",
        "| Market | Frequency | Series | Bars | History days (min / median / max) | "
        "Gaps (total / median) | Off-grid bars | Series with no gaps |",
        "|---|---|---:|---:|---|---|---:|---:|",
    ]
    for group in groups:
        s = group.summary()
        lines.append(
            f"| {s['market']} | {s['frequency']} | {s['series']} | {s['rows']:,} | "
            f"{s['history_days_min']} / {s['history_days_median']} / {s['history_days_max']} | "
            f"{s['gap_total']:,} / {s['gap_median']} | {s['off_grid_total']:,} | "
            f"{s['series_without_gaps']} |"
        )

    lines += ["", "## Gappiest series", ""]
    for group in groups:
        top = ", ".join(f"{symbol} ({gaps})" for symbol, gaps in group.gappiest if gaps)
        lines.append(f"- **{group.market} {group.frequency}**: {top or 'no gaps anywhere'}")

    gate = pairs_meeting_history(manifests)
    lines += [
        "",
        "## Exit-gate counts",
        "",
        "Symbols present at both 1h and 1d with at least 365 days of history.",
        "",
        "| Market | Pairs |",
        "|---|---:|",
    ]
    for market, symbols in sorted(gate.items()):
        lines.append(f"| {market} | {len(symbols)} |")
    lines.append(f"| **all markets** | **{len(set().union(*gate.values())) if gate else 0}** |")

    lines += [
        "",
        "## Invariants",
        "",
        "Zero by construction: `validate_bars(..., raise_on_error=True)` runs on every series "
        "before it is written, so a file that broke an invariant would not be in the tier to be "
        "counted. This row exists so its absence is deliberate rather than forgotten.",
        "",
        "Off-grid bars are counted above rather than rejected. They are real bars published on a "
        "shifted phase after an exchange restart — 43 consecutive hourly bars on spot BTCUSDT "
        "from 2018-02-09, each still exactly one hour after the last. Snapping them to the grid "
        "would be imputation, which the raw tier does not do (ADR-0010).",
    ]

    if verify_results:
        identical = sum(1 for r in verify_results if r.byte_identical)
        lines += [
            "",
            "## Re-pull reproducibility",
            "",
            f"{identical}/{len(verify_results)} sampled series rebuilt byte-identically from the "
            "archives their manifests name.",
            "",
        ]
        lines += [f"- {result.line()}" for result in verify_results]

    return "\n".join(lines) + "\n"

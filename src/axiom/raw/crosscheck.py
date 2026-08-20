"""Does Stooq agree with Yahoo about what a stock did?

The v0.2 adjustment audit has two halves. The **split probes** ask whether Stooq's own series has
an N:1 cliff across a known split date -- they need Stooq and a calendar, nothing else, which is
why they survive Yahoo being unavailable. This module is the other half: a direct comparison of
the two vendors' close paths on a sample of tickers, which classifies Stooq as split-only or
split-and-dividend adjusted.

The output is a number and its interpretation, not a pass or fail. Two vendors disagreeing by
0.3% on a two-year window is normal; disagreeing by 40% on one ticker means one of them applied a
corporate action the other did not, and that is the finding.
"""

from __future__ import annotations

import io
import logging
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import pyarrow.parquet as pq

from axiom.provenance.manifest import FileManifest
from axiom.raw.store import RawStore

log = logging.getLogger("axiom.crosscheck")

MS_PER_DAY = 86_400_000


@dataclass
class TickerComparison:
    """One ticker's agreement between the two vendors, over the overlapping dates."""

    symbol: str
    status: str  # compared | skipped | failed
    overlap_days: int = 0
    max_abs_rel_diff: float = 0.0
    median_abs_rel_diff: float = 0.0
    correlation: float = 0.0
    detail: str = ""

    def line(self) -> str:
        if self.status != "compared":
            return f"{self.symbol}: {self.status.upper()} — {self.detail}"
        return (
            f"{self.symbol}: {self.overlap_days} overlapping days, "
            f"max |rel diff| {self.max_abs_rel_diff:.4f}, "
            f"median {self.median_abs_rel_diff:.4f}, corr {self.correlation:.5f}"
        )


def sample_equity_manifests(
    manifests: list[FileManifest], n: int, seed: int = 1337
) -> list[FileManifest]:
    """A deterministic sample of Stooq series. Same seed, same picks, forever."""
    equities = sorted((m for m in manifests if m.source == "stooq"), key=lambda m: m.artifact_path)
    return random.Random(seed).sample(equities, min(n, len(equities)))


def read_closes(store: RawStore, artifact_path: str, *, since_ms: int) -> dict[int, float]:
    """Close price per timestamp for one stored series, from ``since_ms`` onward."""
    data = store.get(artifact_path)
    if data is None:
        return {}
    table = pq.read_table(io.BytesIO(data), columns=["ts", "close"])
    ts = table["ts"].to_numpy(zero_copy_only=False)
    close = table["close"].to_numpy(zero_copy_only=False)
    keep = ts >= since_ms
    return dict(zip(ts[keep].tolist(), close[keep].tolist(), strict=True))


def compare_closes(
    symbol: str, ours: dict[int, float], theirs: dict[int, float], *, min_overlap: int = 60
) -> TickerComparison:
    """Compare two close paths on the dates they share.

    Only the shared dates are used. A holiday one vendor observes and the other does not is a
    calendar difference, not an adjustment difference, and letting it into the comparison would
    put noise where the signal is supposed to be.
    """
    shared = sorted(set(ours) & set(theirs))
    if len(shared) < min_overlap:
        return TickerComparison(
            symbol,
            "skipped",
            overlap_days=len(shared),
            detail=f"only {len(shared)} overlapping days, need {min_overlap}",
        )

    a = np.array([ours[t] for t in shared], dtype=np.float64)
    b = np.array([theirs[t] for t in shared], dtype=np.float64)
    usable = (a > 0) & (b > 0)
    if usable.sum() < min_overlap:
        return TickerComparison(symbol, "skipped", detail="too many non-positive closes")

    a, b = a[usable], b[usable]
    rel = np.abs(a - b) / b
    correlation = float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 else 0.0

    return TickerComparison(
        symbol,
        "compared",
        overlap_days=int(usable.sum()),
        max_abs_rel_diff=float(rel.max()),
        median_abs_rel_diff=float(np.median(rel)),
        correlation=correlation,
    )


def crosscheck_equities(
    store: RawStore,
    manifests: list[FileManifest],
    price_fetcher,
    *,
    sample: int = 25,
    years: int = 2,
    seed: int = 1337,
    now: datetime | None = None,
) -> list[TickerComparison]:
    """Compare a sample of Stooq series against Yahoo's adjusted closes."""
    now = now or datetime.now(UTC)
    since = now - timedelta(days=365 * years)
    since_ms = int(since.timestamp() * 1000)

    results: list[TickerComparison] = []
    for manifest in sample_equity_manifests(manifests, sample, seed):
        symbol = manifest.symbol
        try:
            ours = read_closes(store, manifest.artifact_path, since_ms=since_ms)
            if not ours:
                results.append(
                    TickerComparison(symbol, "skipped", detail="no stored bars in the window")
                )
                continue
            theirs = price_fetcher(manifest.source_symbol or symbol, since, now)
            results.append(compare_closes(symbol, ours, theirs))
        except Exception as exc:
            log.warning("crosscheck failed for %s: %s", symbol, exc)
            results.append(
                TickerComparison(symbol, "failed", detail=f"{type(exc).__name__}: {exc}")
            )
    return results


def crosscheck_markdown(results: list[TickerComparison], *, as_of: str) -> str:
    """The section this comparison contributes to the adjustment audit."""
    compared = [r for r in results if r.status == "compared"]
    lines = [
        "## Stooq versus Yahoo adjusted closes",
        "",
        f"Sampled {len(results)} tickers on {as_of}; {len(compared)} had enough overlap to "
        "compare.",
        "",
    ]
    if not compared:
        lines += [
            "No ticker could be compared. Either the equities tier is not populated or Yahoo "
            "would not answer this backend — both are recorded above rather than inferred from "
            "this absence. The split probes stand independently of this section.",
            "",
        ]
        return "\n".join(lines) + "\n"

    worst = max(r.max_abs_rel_diff for r in compared)
    median_of_medians = float(np.median([r.median_abs_rel_diff for r in compared]))
    lines += [
        f"Median of per-ticker median relative differences: **{median_of_medians:.4f}**. "
        f"Largest single relative difference seen: **{worst:.4f}**.",
        "",
        "| Ticker | Overlap days | Max abs rel diff | Median | Correlation |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in sorted(compared, key=lambda r: -r.max_abs_rel_diff):
        lines.append(
            f"| {r.symbol} | {r.overlap_days} | {r.max_abs_rel_diff:.4f} | "
            f"{r.median_abs_rel_diff:.4f} | {r.correlation:.5f} |"
        )

    skipped = [r for r in results if r.status != "compared"]
    if skipped:
        lines += ["", "Not compared:", ""] + [f"- {r.line()}" for r in skipped]

    lines += [
        "",
        "**Reading this.** Yahoo's `auto_adjust=True` closes are split *and* dividend adjusted. "
        "If Stooq tracks them to within a fraction of a percent, Stooq is dividend-adjusted too. "
        "A persistent drift that grows with the lookback, on dividend-paying tickers only, is "
        "the signature of a split-adjusted-but-not-dividend-adjusted vendor — which is the more "
        "common convention, and the one v0.3 would have to correct for when building total-return "
        "eval labels.",
    ]
    return "\n".join(lines) + "\n"

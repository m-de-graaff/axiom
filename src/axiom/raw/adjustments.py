"""What did Stooq already do to these prices?

The vendor does not document it, and the answer decides what v0.3's policy has to do: an
already-split-adjusted series needs no split handling, a dividend-adjusted one cannot be used for
total-return eval labels without undoing the adjustment, and an unadjusted one needs both.

So it is measured, in two independent ways, and the two are deliberately not equally fragile.

The **split probes** ask whether the vendor's own series has an N:1 cliff across a known split
date. They need the Stooq bars and a calendar, nothing else -- no second vendor, no network -- so
they answer even when Yahoo is unavailable. The **dividend classification** needs the
cross-check, and is reported as unavailable rather than guessed when it cannot run.

The verdict is written down with the evidence beside it. `adjustment_policy` in the manifests is
then a recorded finding rather than an assumption, which is the whole point of the exercise.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import pyarrow.parquet as pq

from axiom.provenance.manifest import FileManifest
from axiom.raw.crosscheck import TickerComparison
from axiom.raw.store import RawStore
from axiom.sources.yahoo_events import detect_split_discontinuity, known_split_probes

log = logging.getLogger("axiom.adjustments")

#: A vendor tracking dividend-adjusted closes stays within a fraction of a percent. A
#: split-adjusted-but-not-dividend-adjusted one drifts, and the drift compounds with the lookback.
DIVIDEND_AGREEMENT_THRESHOLD = 0.02

POLICY_SPLIT_ONLY = "split_adjusted"
POLICY_SPLIT_AND_DIVIDEND = "split_and_dividend_adjusted"
POLICY_UNADJUSTED = "unadjusted"
POLICY_UNKNOWN = "vendor_adjusted_unverified"


@dataclass
class SplitProbe:
    """One known split, and what the stored series did across it."""

    symbol: str
    split_date: str
    ratio: float
    status: str  # adjusted | unadjusted | inconclusive | missing
    measured: float | None = None
    detail: str = ""

    def line(self) -> str:
        head = f"{self.symbol} {self.split_date} ({self.ratio:g}:1)"
        if self.status in ("missing", "inconclusive"):
            return f"{head}: {self.status.upper()} — {self.detail}"
        return f"{head}: {self.status.upper()}, close ratio across the split {self.measured}"


def date_to_ms(iso: str) -> int:
    return int(datetime.fromisoformat(iso).replace(tzinfo=UTC).timestamp() * 1000)


def run_split_probes(
    store: RawStore,
    manifests: list[FileManifest],
    probes: dict[str, tuple[str, float]] | None = None,
) -> list[SplitProbe]:
    """Check each known split against the stored series. Needs no second vendor."""
    probes = probes or known_split_probes()
    by_symbol = {m.symbol: m for m in manifests if m.source == "stooq"}
    results: list[SplitProbe] = []

    for symbol, (split_date, ratio) in sorted(probes.items()):
        manifest = by_symbol.get(symbol)
        if manifest is None:
            results.append(
                SplitProbe(symbol, split_date, ratio, "missing", detail="not in the raw tier")
            )
            continue
        try:
            data = store.get(manifest.artifact_path)
            if data is None:
                results.append(
                    SplitProbe(symbol, split_date, ratio, "missing", detail="artifact unreadable")
                )
                continue
            table = pq.read_table(io.BytesIO(data), columns=["ts", "close"])
            verdict = detect_split_discontinuity(
                table["ts"].to_pylist(), table["close"].to_pylist(), date_to_ms(split_date), ratio
            )
            if verdict["adjusted"] is None:
                results.append(
                    SplitProbe(symbol, split_date, ratio, "inconclusive", detail=verdict["reason"])
                )
            else:
                results.append(
                    SplitProbe(
                        symbol,
                        split_date,
                        ratio,
                        "adjusted" if verdict["adjusted"] else "unadjusted",
                        measured=verdict["measured"],
                    )
                )
        except Exception as exc:
            log.warning("split probe failed for %s: %s", symbol, exc)
            results.append(
                SplitProbe(
                    symbol, split_date, ratio, "inconclusive", detail=f"{type(exc).__name__}: {exc}"
                )
            )
    return results


def classify(probes: list[SplitProbe], comparisons: list[TickerComparison]) -> tuple[str, str]:
    """Turn the evidence into a policy value and the sentence that justifies it.

    The two halves answer different questions and the split half is answered first, because it is
    the one that always has an answer. A corpus can know it is split-adjusted while remaining
    honestly unsure about dividends, and `vendor_adjusted_unverified` would be the wrong label
    for that state -- it would throw away a fact that was established.
    """
    decided = [p for p in probes if p.status in ("adjusted", "unadjusted")]
    if not decided:
        return POLICY_UNKNOWN, (
            "No split probe could be evaluated — the probe tickers are absent from the tier or "
            "their series do not span the split dates. Nothing is claimed."
        )

    adjusted = [p for p in decided if p.status == "adjusted"]
    if len(adjusted) != len(decided):
        disagreeing = ", ".join(p.symbol for p in decided if p.status != "adjusted")
        return POLICY_UNADJUSTED, (
            f"The probes disagree: {disagreeing} shows an unadjusted price cliff across a known "
            "split while others do not. A vendor that adjusts inconsistently is worse than one "
            "that does not adjust, and v0.3 must handle splits itself for every series."
        )

    compared = [c for c in comparisons if c.status == "compared"]
    if not compared:
        return POLICY_SPLIT_ONLY, (
            f"All {len(decided)} split probes show no discontinuity, so the series are "
            "split-adjusted. Whether dividends are also adjusted is **not** established: the "
            "cross-check could not run. Recorded as split-only, which is the weaker of the two "
            "claims and the one the evidence supports."
        )

    median_of_medians = sorted(c.median_abs_rel_diff for c in compared)[len(compared) // 2]
    if median_of_medians <= DIVIDEND_AGREEMENT_THRESHOLD:
        return POLICY_SPLIT_AND_DIVIDEND, (
            f"All {len(decided)} split probes show no discontinuity, and across {len(compared)} "
            f"sampled tickers the median relative difference against Yahoo's dividend-adjusted "
            f"closes is {median_of_medians:.4f}, inside the {DIVIDEND_AGREEMENT_THRESHOLD:.0%} "
            "agreement threshold. The series track a total-return path, so they are split *and* "
            "dividend adjusted."
        )
    return POLICY_SPLIT_ONLY, (
        f"All {len(decided)} split probes show no discontinuity, so the series are split-adjusted. "
        f"Across {len(compared)} sampled tickers the median relative difference against Yahoo's "
        f"dividend-adjusted closes is {median_of_medians:.4f}, outside the "
        f"{DIVIDEND_AGREEMENT_THRESHOLD:.0%} threshold — the signature of a vendor that adjusts "
        "splits but not dividends. v0.3 must build total-return eval labels from the dividend "
        "events rather than from these closes."
    )


def audit_markdown(
    probes: list[SplitProbe],
    comparisons: list[TickerComparison],
    policy: str,
    reasoning: str,
    *,
    as_of: str,
    crosscheck_section: str = "",
) -> str:
    """The committed audit. Evidence first, verdict stated plainly, no hedging either way."""
    lines = [
        "# v0.2 adjustment audit",
        "",
        f"Run {as_of}. **Verdict: `{policy}`.**",
        "",
        reasoning,
        "",
        "## Split probes",
        "",
        "Each of these is a large, recent, unambiguous split. An unadjusted series shows a close "
        "ratio near the split ratio across the date; an adjusted one shows a ratio near 1. No "
        "market move imitates a 75% overnight fall, which is what makes this test decisive "
        "without a second vendor.",
        "",
        "| Ticker | Split date | Ratio | Measured close ratio | Verdict |",
        "|---|---|---:|---:|---|",
    ]
    for probe in probes:
        measured = "—" if probe.measured is None else f"{probe.measured:g}"
        note = probe.status if not probe.detail else f"{probe.status} ({probe.detail})"
        lines.append(
            f"| {probe.symbol} | {probe.split_date} | {probe.ratio:g}:1 | {measured} | {note} |"
        )

    lines += ["", crosscheck_section or "", ""]
    lines += [
        "## What this changes",
        "",
        f"`adjustment_policy` is regenerated as `{policy}` for `raw/stooq/**` only. No other "
        "source is touched: Binance and Dukascopy have no corporate actions to adjust for, and "
        "their manifests keep `none`.",
        "",
        "v0.3 reads this verdict rather than re-deriving it. What it means there: a "
        "`split_adjusted` corpus needs no split handling in the cleaning pass, but its closes are "
        "a price path rather than a total-return path, so eval labels that need total return must "
        "be built from the dividend events captured in `raw/yahoo/adjustments/`.",
        "",
        "## Survivorship, again",
        "",
        "Every number here is measured on tickers that still exist. The bulk dump skews to "
        "currently-listed names (ADR-0016), so this audit says what the vendor did to the "
        "survivors and says nothing about what it did to the delisted. That limitation is "
        "inherited by everything downstream and is repeated in the v0.9 model card.",
    ]
    return "\n".join(line for line in lines) + "\n"

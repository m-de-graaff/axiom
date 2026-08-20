"""The adjustment audit.

The classifier is the piece worth testing hardest, because it turns evidence into a label that
every downstream version reads without re-deriving. The properties that matter are that it never
claims more than the evidence supports, and never throws away a fact it did establish.
"""

from __future__ import annotations

import io

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from axiom.raw.adjustments import (
    POLICY_SPLIT_AND_DIVIDEND,
    POLICY_SPLIT_ONLY,
    POLICY_UNADJUSTED,
    POLICY_UNKNOWN,
    SplitProbe,
    audit_markdown,
    classify,
    date_to_ms,
    run_split_probes,
)
from axiom.raw.crosscheck import TickerComparison
from tests.test_registry import manifest

DAY_MS = 86_400_000


class FakeStore:
    def __init__(self, series: dict[str, tuple[list[int], list[float]]]) -> None:
        self._data = {}
        for path, (ts, close) in series.items():
            buffer = io.BytesIO()
            pq.write_table(
                pa.table({"ts": pa.array(ts, pa.int64()), "close": pa.array(close, pa.float64())}),
                buffer,
            )
            self._data[path] = buffer.getvalue()

    def get(self, artifact_path: str) -> bytes | None:
        return self._data.get(artifact_path)


def stooq(symbol: str):
    return manifest(
        source="stooq", market="us", asset_class="equity", symbol=symbol, frequency="1d"
    )


def probe(status: str, symbol: str = "AAPL") -> SplitProbe:
    return SplitProbe(symbol, "2020-08-31", 4.0, status, measured=1.0)


def comparison(median: float, symbol: str = "AAA") -> TickerComparison:
    return TickerComparison(symbol, "compared", overlap_days=400, median_abs_rel_diff=median)


# --- running the probes ---------------------------------------------------------------------


def test_an_adjusted_series_reads_as_adjusted():
    split = date_to_ms("2020-08-31")
    store = FakeStore(
        {
            "raw/stooq/us/1d/AAPL.parquet": (
                [split - DAY_MS, split, split + DAY_MS],
                [124.0, 125.0, 126.0],
            )
        }
    )
    results = run_split_probes(store, [stooq("AAPL")], {"AAPL": ("2020-08-31", 4.0)})
    assert results[0].status == "adjusted"


def test_an_unadjusted_series_reads_as_unadjusted():
    split = date_to_ms("2020-08-31")
    store = FakeStore(
        {
            "raw/stooq/us/1d/AAPL.parquet": (
                [split - DAY_MS, split, split + DAY_MS],
                [500.0, 125.0, 126.0],
            )
        }
    )
    results = run_split_probes(store, [stooq("AAPL")], {"AAPL": ("2020-08-31", 4.0)})
    assert results[0].status == "unadjusted"
    assert results[0].measured == pytest.approx(4.0)


def test_a_ticker_absent_from_the_tier_is_missing_not_adjusted():
    """The dangerous failure: absence read as a clean result."""
    results = run_split_probes(FakeStore({}), [], {"AAPL": ("2020-08-31", 4.0)})
    assert results[0].status == "missing"


def test_a_series_that_does_not_span_the_split_is_inconclusive():
    split = date_to_ms("2020-08-31")
    store = FakeStore({"raw/stooq/us/1d/AAPL.parquet": ([split + DAY_MS], [125.0])})
    results = run_split_probes(store, [stooq("AAPL")], {"AAPL": ("2020-08-31", 4.0)})
    assert results[0].status == "inconclusive"


# --- classification ---------------------------------------------------------------------------


def test_no_evaluable_probe_claims_nothing():
    policy, reasoning = classify([probe("missing")], [comparison(0.001)])
    assert policy == POLICY_UNKNOWN
    assert "Nothing is claimed" in reasoning


def test_probes_that_disagree_are_treated_as_unadjusted():
    """A vendor that adjusts inconsistently is worse than one that does not adjust at all."""
    policy, reasoning = classify([probe("adjusted"), probe("unadjusted", "TSLA")], [])
    assert policy == POLICY_UNADJUSTED
    assert "TSLA" in reasoning


def test_split_adjusted_with_no_crosscheck_claims_only_split():
    """The weaker of the two claims, and the one the evidence supports."""
    policy, reasoning = classify([probe("adjusted")], [])
    assert policy == POLICY_SPLIT_ONLY
    assert "not** established" in reasoning or "not established" in reasoning


def test_close_agreement_with_yahoo_means_dividends_are_adjusted_too():
    policy, _ = classify([probe("adjusted")], [comparison(0.0005), comparison(0.001, "BBB")])
    assert policy == POLICY_SPLIT_AND_DIVIDEND


def test_persistent_drift_means_split_only():
    policy, reasoning = classify([probe("adjusted")], [comparison(0.09), comparison(0.11, "BBB")])
    assert policy == POLICY_SPLIT_ONLY
    assert "not dividends" in reasoning


def test_a_split_fact_survives_an_unavailable_crosscheck():
    """Establishing splits and staying unsure about dividends is a real state, not 'unknown'."""
    policy, _ = classify([probe("adjusted")], [])
    assert policy != POLICY_UNKNOWN


# --- the report ---------------------------------------------------------------------------------


def test_the_report_states_the_verdict_and_shows_the_evidence():
    text = audit_markdown(
        [probe("adjusted")], [comparison(0.001)], POLICY_SPLIT_ONLY, "because.", as_of="2026-08-21"
    )
    assert f"**Verdict: `{POLICY_SPLIT_ONLY}`.**" in text
    assert "| AAPL | 2020-08-31 |" in text
    assert "2026-08-21" in text


def test_the_report_says_which_manifests_change_and_which_do_not():
    text = audit_markdown(
        [probe("adjusted")], [], POLICY_SPLIT_ONLY, "because.", as_of="2026-08-21"
    )
    assert "raw/stooq/**` only" in text
    assert "Binance and Dukascopy" in text


def test_the_report_repeats_the_survivorship_limitation():
    """Every number here is measured on tickers that still exist."""
    text = audit_markdown(
        [probe("adjusted")], [], POLICY_SPLIT_ONLY, "because.", as_of="2026-08-21"
    )
    assert "Survivorship" in text
    assert "delisted" in text

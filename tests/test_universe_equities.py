"""The equities training universe.

The one universe in the project with a builder that reads the corpus, so the tests are about the
two things that makes fragile: that the cheap filter really runs before the expensive one, and
that the result is reproducible enough to commit.
"""

from __future__ import annotations

import io

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from axiom.registry import build_from_manifests
from axiom.universe.equities import (
    Candidate,
    EquityUniverse,
    build_equity_universe,
    candidates_from_registry,
    load_equity_universe,
    median_dollar_volume,
    rank_candidates,
)
from tests.test_registry import manifest

DAY_MS = 86_400_000
JAN_2024 = 1_704_067_200_000


class FakeStore:
    """A store holding pre-built Parquet bytes, and counting who asked for what."""

    def __init__(self, series: dict[str, tuple[list[float], list[float]]]) -> None:
        self.reads: list[str] = []
        self._data = {
            path: self._parquet(close, volume) for path, (close, volume) in series.items()
        }

    @staticmethod
    def _parquet(close: list[float], volume: list[float]) -> bytes:
        buffer = io.BytesIO()
        pq.write_table(
            pa.table(
                {"close": pa.array(close, pa.float64()), "volume": pa.array(volume, pa.float64())}
            ),
            buffer,
        )
        return buffer.getvalue()

    def get(self, artifact_path: str) -> bytes | None:
        self.reads.append(artifact_path)
        return self._data.get(artifact_path)


def equity(symbol: str, *, days: int) -> object:
    return manifest(
        source="stooq",
        market="us",
        asset_class="equity",
        symbol=symbol,
        frequency="1d",
        rows=days,
        days=days,
    )


def registry_of(*manifests):
    return build_from_manifests(list(manifests), sizes={}).table


# --- the history filter ---------------------------------------------------------------------


def test_only_series_with_enough_history_become_candidates():
    table = registry_of(equity("OLD", days=3000), equity("NEW", days=400))
    candidates = candidates_from_registry(table, min_history_years=5)
    assert [c.symbol for c in candidates] == ["OLD"]


def test_other_sources_are_never_candidates():
    """The equities universe is equities. A crypto pair with ten years of history is not one."""
    table = registry_of(
        equity("OLD", days=3000), manifest(symbol="BTCUSDT", frequency="1d", days=3000)
    )
    assert [c.symbol for c in candidates_from_registry(table)] == ["OLD"]


def test_the_wrong_frequency_is_never_a_candidate():
    table = registry_of(
        equity("OLD", days=3000),
        manifest(
            source="stooq",
            market="us",
            asset_class="equity",
            symbol="HR",
            frequency="1h",
            days=3000,
        ),
    )
    assert [c.symbol for c in candidates_from_registry(table)] == ["OLD"]


def test_candidates_come_back_sorted():
    table = registry_of(equity("ZZZ", days=3000), equity("AAA", days=3000))
    assert [c.symbol for c in candidates_from_registry(table)] == ["AAA", "ZZZ"]


# --- the ranking metric ---------------------------------------------------------------------


def test_dollar_volume_is_price_times_shares():
    table = pa.table({"close": pa.array([10.0, 10.0]), "volume": pa.array([100.0, 100.0])})
    assert median_dollar_volume(table) == 1000.0


def test_the_median_resists_a_single_spike():
    """One earnings day should not carry a ticker into the universe."""
    close = [10.0] * 100
    volume = [100.0] * 99 + [1_000_000.0]
    table = pa.table({"close": pa.array(close), "volume": pa.array(volume)})
    assert median_dollar_volume(table) == 1000.0


def test_only_the_ranking_window_counts():
    """Liquid in 2009 is not liquid, and the window is what says so."""
    close = [10.0] * 300
    volume = [1_000_000.0] * 48 + [100.0] * 252
    table = pa.table({"close": pa.array(close), "volume": pa.array(volume)})
    assert median_dollar_volume(table, window=252) == 1000.0


def test_non_finite_rows_do_not_poison_the_median():
    table = pa.table(
        {"close": pa.array([10.0, 10.0, np.nan]), "volume": pa.array([100.0, 100.0, 100.0])}
    )
    assert median_dollar_volume(table) == 1000.0


def test_an_empty_series_ranks_at_zero_rather_than_raising():
    assert (
        median_dollar_volume(
            pa.table({"close": pa.array([], pa.float64()), "volume": pa.array([], pa.float64())})
        )
        == 0.0
    )


# --- ranking --------------------------------------------------------------------------------


def test_ranking_is_descending_by_dollar_volume():
    store = FakeStore(
        {
            "a.parquet": ([10.0] * 300, [100.0] * 300),
            "b.parquet": ([10.0] * 300, [900.0] * 300),
        }
    )
    candidates = [Candidate("AAA", "a.parquet", 2000.0), Candidate("BBB", "b.parquet", 2000.0)]
    assert [s for s, _ in rank_candidates(store, candidates)] == ["BBB", "AAA"]


def test_ties_break_on_the_symbol_so_the_order_is_total():
    """Two runs a year apart must not disagree about a coin flip."""
    store = FakeStore(
        {"a.parquet": ([10.0] * 300, [100.0] * 300), "b.parquet": ([10.0] * 300, [100.0] * 300)}
    )
    candidates = [Candidate("ZZZ", "b.parquet", 2000.0), Candidate("AAA", "a.parquet", 2000.0)]
    assert [s for s, _ in rank_candidates(store, candidates)] == ["AAA", "ZZZ"]


def test_a_series_that_cannot_be_read_drops_out_rather_than_failing_the_build():
    store = FakeStore({"a.parquet": ([10.0] * 300, [100.0] * 300)})
    candidates = [
        Candidate("AAA", "a.parquet", 2000.0),
        Candidate("GONE", "missing.parquet", 2000.0),
    ]
    assert [s for s, _ in rank_candidates(store, candidates)] == ["AAA"]


def test_only_candidates_are_ever_downloaded():
    """The whole point of filtering on the registry first."""
    table = registry_of(equity("OLD", days=3000), equity("NEW", days=400))
    store = FakeStore({"raw/stooq/us/1d/OLD.parquet": ([10.0] * 300, [100.0] * 300)})
    build_equity_universe(store, table, registry_hash="abc123", generated_at="2026-08-20", top_n=10)
    assert store.reads == ["raw/stooq/us/1d/OLD.parquet"]


# --- the built file ---------------------------------------------------------------------------


def built() -> EquityUniverse:
    table = registry_of(equity("AAA", days=3000), equity("BBB", days=3000), equity("NEW", days=100))
    store = FakeStore(
        {
            "raw/stooq/us/1d/AAA.parquet": ([10.0] * 300, [100.0] * 300),
            "raw/stooq/us/1d/BBB.parquet": ([10.0] * 300, [900.0] * 300),
        }
    )
    return build_equity_universe(
        store, table, registry_hash="reg123", generated_at="2026-08-20", top_n=1
    )


def test_the_cut_keeps_the_top_n_and_records_what_it_considered():
    """'3000 of 3000' and '3000 of 14000' describe very different universes."""
    universe = built()
    assert universe.symbols == ["BBB"]
    assert universe.candidates_considered == 2


def test_the_criteria_and_the_registry_hash_are_echoed():
    """Without the snapshot, 'top N by dollar volume' names a procedure, not a result."""
    universe = built()
    assert universe.criteria.registry_hash == "reg123"
    assert universe.criteria.generated_at == "2026-08-20"
    assert universe.criteria.top_n == 1


def test_the_hash_holds_and_the_file_round_trips(tmp_path):
    universe = built()
    path = tmp_path / "universe_equities_v1.yaml"
    path.write_text(universe.to_yaml(), encoding="utf-8")
    assert load_equity_universe(path).symbols == universe.symbols


def test_an_edited_file_is_refused(tmp_path):
    universe = built()
    path = tmp_path / "u.yaml"
    path.write_text(universe.to_yaml().replace("- BBB", "- CCC"), encoding="utf-8")
    with pytest.raises(ValueError, match="was edited without rebuilding"):
        load_equity_universe(path)


def test_the_header_says_the_corpus_is_a_superset():
    """A universe read as 'what we stored' would make later versions delete data."""
    text = built().to_yaml()
    assert "SUPERSET" in text
    assert "Survivorship" in text

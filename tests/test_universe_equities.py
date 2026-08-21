"""The equities training universe.

It reads the registry and downloads nothing: each series' median dollar volume was computed at
pull time, when its bars were already in memory. So the tests are about the two things that makes
fragile — that a candidate whose statistic is missing is treated as *unmeasured* rather than as
worthless, and that the result is reproducible enough to commit.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from axiom.registry import build_from_manifests
from axiom.sources.base import median_dollar_volume
from axiom.universe.equities import (
    Candidate,
    EquityUniverse,
    IncompleteRanking,
    build_equity_universe,
    candidates_from_registry,
    load_equity_universe,
    rank_candidates,
)
from tests.test_registry import manifest

DAY_MS = 86_400_000


def equity(symbol: str, *, days: int, dollar_volume: float = 1000.0):
    return manifest(
        source="stooq",
        market="us",
        asset_class="equity",
        symbol=symbol,
        frequency="1d",
        rows=days,
        days=days,
        median_dollar_volume=dollar_volume,
    )


def registry_of(*manifests):
    return build_from_manifests(list(manifests), sizes={}).table


def bars(close: list[float], volume: list[float]) -> pa.Table:
    return pa.table(
        {"close": pa.array(close, pa.float64()), "volume": pa.array(volume, pa.float64())}
    )


# --- the history filter ---------------------------------------------------------------------


def test_only_series_with_enough_history_become_candidates():
    table = registry_of(equity("OLD", days=3000), equity("NEW", days=400))
    assert [c.symbol for c in candidates_from_registry(table, min_history_years=5)] == ["OLD"]


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


def test_the_dollar_volume_comes_along_from_the_registry():
    table = registry_of(equity("AAA", days=3000, dollar_volume=12345.0))
    assert candidates_from_registry(table)[0].median_dollar_volume == 12345.0


def test_a_sidecar_predating_the_field_reads_as_unmeasured_not_as_zero():
    """A stale registry must be caught by the guard, not silently ranked at the bottom."""
    table = registry_of(equity("OLD", days=3000, dollar_volume=0.0))
    assert candidates_from_registry(table)[0].median_dollar_volume is None


# --- the ranking metric, computed at pull time ------------------------------------------------


def test_dollar_volume_is_price_times_shares():
    assert median_dollar_volume(bars([10.0, 10.0], [100.0, 100.0])) == 1000.0


def test_the_median_resists_a_single_spike():
    """One earnings day should not carry a ticker into the universe."""
    assert median_dollar_volume(bars([10.0] * 100, [100.0] * 99 + [1_000_000.0])) == 1000.0


def test_only_the_ranking_window_counts():
    """Liquid in 2009 is not liquid, and the window is what says so."""
    table = bars([10.0] * 300, [1_000_000.0] * 48 + [100.0] * 252)
    assert median_dollar_volume(table, window=252) == 1000.0


def test_non_finite_rows_do_not_poison_the_median():
    assert median_dollar_volume(bars([10.0, 10.0, np.nan], [100.0, 100.0, 100.0])) == 1000.0


def test_an_empty_series_measures_zero_rather_than_raising():
    assert median_dollar_volume(bars([], [])) == 0.0


# --- ranking ----------------------------------------------------------------------------------


def candidate(symbol: str, value: float | None) -> Candidate:
    return Candidate(symbol, f"raw/stooq/us/1d/{symbol}.parquet", 3000.0, value)


def test_ranking_is_descending_by_dollar_volume():
    ranking = rank_candidates([candidate("AAA", 100.0), candidate("BBB", 900.0)])
    assert [s for s, _ in ranking.ranked] == ["BBB", "AAA"]


def test_ties_break_on_the_symbol_so_the_order_is_total():
    """Two runs a year apart must not disagree about a coin flip."""
    ranking = rank_candidates([candidate("ZZZ", 100.0), candidate("AAA", 100.0)])
    assert [s for s, _ in ranking.ranked] == ["AAA", "ZZZ"]


def test_an_unmeasured_candidate_is_unrankable_not_worthless():
    """The bug this replaced: a failed read returned 0.0, the `> 0` cut discarded it, and a Hub
    429 became 'this stock has no volume'. One real run measured 1 344 of 6 829 candidates."""
    ranking = rank_candidates([candidate("AAA", 100.0), candidate("GONE", None)])
    assert [s for s, _ in ranking.ranked] == ["AAA"]
    assert ranking.unrankable == ["GONE"]
    assert ranking.zero_volume == []


def test_a_genuinely_untraded_series_is_zero_volume_not_unrankable():
    """The other half of the distinction: measured, and the answer was nothing."""
    ranking = rank_candidates([candidate("AAA", 0.0)])
    assert ranking.zero_volume == ["AAA"]
    assert ranking.unrankable == []


def test_a_universe_is_refused_when_too_much_could_not_be_measured():
    """A universe built on the tenth of the market that answered is not a smaller universe."""
    table = registry_of(*[equity(f"S{i:04d}", days=3000, dollar_volume=0.0) for i in range(50)])
    with pytest.raises(IncompleteRanking, match="could not be measured"):
        build_equity_universe(table, registry_hash="r", generated_at="2026-08-21", top_n=3000)


# --- the built file -----------------------------------------------------------------------------


def built() -> EquityUniverse:
    table = registry_of(
        equity("AAA", days=3000, dollar_volume=100.0),
        equity("BBB", days=3000, dollar_volume=900.0),
        equity("NEW", days=100, dollar_volume=500.0),
    )
    return build_equity_universe(table, registry_hash="reg123", generated_at="2026-08-21", top_n=1)


def test_the_cut_keeps_the_top_n_and_records_what_it_considered():
    """'3000 of 3000' and '3000 of 14000' describe very different universes."""
    universe = built()
    assert universe.symbols == ["BBB"]
    assert universe.candidates_considered == 2


def test_the_file_records_what_could_not_be_measured():
    """Zero here is what makes the universe mean what it says."""
    assert built().candidates_unrankable == 0


def test_the_criteria_and_the_registry_hash_are_echoed():
    """Without the snapshot, 'top N by dollar volume' names a procedure, not a result."""
    universe = built()
    assert universe.criteria.registry_hash == "reg123"
    assert universe.criteria.generated_at == "2026-08-21"
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

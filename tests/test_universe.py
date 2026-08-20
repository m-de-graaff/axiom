"""Universe selection: exclusions, deterministic ranking, and a stable hash (ADR-0011)."""

from __future__ import annotations

import random

import pytest
import yaml

from axiom.universe.binance import (
    FIAT_BASES,
    STABLE_BASES,
    UniverseConfig,
    UniverseCriteria,
    build_universe,
    candidate_symbols,
    has_min_history,
    is_leveraged,
    load_universe,
    rank_symbols,
    shift_month,
)
from tests.fakes import DAY_MS, FakeBucket, kline_zip

MONTH = "2026-07"


@pytest.fixture
def client_factory():
    from axiom.sources.binance_vision import BinanceVision

    created = []

    def make(bucket):
        client = BinanceVision(
            client=bucket.client(),
            concurrency=4,
            backoff_base=0.0,
            sleep=lambda _: None,
            rng=random.Random(5),
        )
        created.append(client)
        return client

    yield make
    for client in created:
        client.close()


def seed(
    bucket: FakeBucket, market: str, volumes: dict[str, float], *, history_months: int = 24
) -> None:
    """Give each symbol enough 1h history to qualify, plus a rankable month of 1d bars."""
    for symbol, volume in volumes.items():
        bucket.put_month(
            market,
            symbol,
            "1d",
            MONTH,
            kline_zip(30, step=DAY_MS, price=1.0, volume=volume),
        )
        bucket.put_month(market, symbol, "1h", shift_month(MONTH, -history_months), kline_zip(24))


# --- exclusions --------------------------------------------------------------------------


def test_leveraged_tokens_are_excluded():
    listed = {"BTCUSDT", "BTCUPUSDT", "BTCDOWNUSDT", "ETHUSDT", "ETHBULLUSDT", "ETHBEARUSDT"}
    assert is_leveraged("BTCUPUSDT", listed)
    assert is_leveraged("ETHBEARUSDT", listed)
    assert not is_leveraged("BTCUSDT", listed)


def test_a_real_token_whose_ticker_ends_in_a_suffix_is_kept():
    # JUP is Jupiter. A bare suffix match reads it as a leveraged JU and drops a real market.
    listed = {"JUPUSDT", "BTCUSDT", "BTCUPUSDT"}
    assert not is_leveraged("JUPUSDT", listed)
    assert is_leveraged("BTCUPUSDT", listed)


def test_a_suffixed_symbol_whose_base_is_not_listed_is_kept():
    assert not is_leveraged("SOMEUPUSDT", {"SOMEUPUSDT"})


@pytest.mark.parametrize("base", ["USDC", "FDUSD", "DAI", "BUSD"])
def test_stable_to_stable_pairs_are_excluded(base):
    assert base in STABLE_BASES
    assert f"{base}USDT" not in candidate_symbols([f"{base}USDT", "BTCUSDT"])


@pytest.mark.parametrize("base", ["EUR", "GBP", "TRY", "BRL"])
def test_fiat_quoted_pairs_are_excluded(base):
    assert base in FIAT_BASES
    assert f"{base}USDT" not in candidate_symbols([f"{base}USDT", "BTCUSDT"])


def test_a_depegged_former_stable_is_kept():
    # Whatever USTC was designed to be, what it prints is real price action.
    assert "USTCUSDT" in candidate_symbols(["USTCUSDT", "BTCUSDT"])


def test_non_usdt_quotes_are_excluded():
    assert candidate_symbols(["BTCUSDT", "BTCBUSD", "ETHBTC"]) == ["BTCUSDT"]


def test_the_quote_asset_alone_is_not_a_candidate():
    assert candidate_symbols(["USDT", "BTCUSDT"]) == ["BTCUSDT"]


# --- minimum history ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("month", "delta", "expected"),
    [("2026-07", -12, "2025-07"), ("2026-01", -1, "2025-12"), ("2025-12", 1, "2026-01")],
)
def test_month_arithmetic(month, delta, expected):
    assert shift_month(month, delta) == expected


def test_a_symbol_younger_than_the_rule_is_rejected(client_factory):
    bucket = FakeBucket()
    seed(bucket, "um", {"OLDUSDT": 1.0}, history_months=24)
    seed(bucket, "um", {"NVDAUSDT": 9.0}, history_months=2)
    client = client_factory(bucket)
    assert has_min_history(client, "um", "OLDUSDT", month=MONTH, min_months=12)
    assert not has_min_history(client, "um", "NVDAUSDT", month=MONTH, min_months=12)


def test_a_symbol_with_no_1h_series_is_rejected(client_factory):
    bucket = FakeBucket()
    seed(bucket, "spot", {"BTCUSDT": 1.0})
    client = client_factory(bucket)
    assert not has_min_history(client, "spot", "GHOSTUSDT", month=MONTH, min_months=12)


def test_a_high_volume_newcomer_does_not_enter_the_universe(client_factory):
    # The July 2026 build put seven tokenized equities in the top ten perpetuals by volume. They
    # trade real size; they are also weeks old, and they are not crypto.
    bucket = FakeBucket()
    seed(bucket, "um", {"BTCUSDT": 1.0}, history_months=24)
    seed(bucket, "um", {"NVDAUSDT": 1000.0}, history_months=2)
    universe = build(bucket, client_factory, top_n={"um": 10})
    assert universe.symbols["um"] == ["BTCUSDT"]


# --- ranking -----------------------------------------------------------------------------


def test_ranking_is_by_quote_volume_descending(client_factory):
    bucket = FakeBucket()
    seed(bucket, "spot", {"AAAUSDT": 1.0, "BBBUSDT": 5.0, "CCCUSDT": 3.0})
    ranked = rank_symbols(client_factory(bucket), "spot", ["AAAUSDT", "BBBUSDT", "CCCUSDT"], MONTH)
    assert [symbol for symbol, _ in ranked] == ["BBBUSDT", "CCCUSDT", "AAAUSDT"]


def test_a_symbol_absent_in_the_selection_month_does_not_rank(client_factory):
    bucket = FakeBucket()
    seed(bucket, "spot", {"AAAUSDT": 1.0})
    ranked = rank_symbols(client_factory(bucket), "spot", ["AAAUSDT", "GHOSTUSDT"], MONTH)
    assert [symbol for symbol, _ in ranked] == ["AAAUSDT"]


def test_ties_break_on_the_symbol_name(client_factory):
    bucket = FakeBucket()
    seed(bucket, "spot", {"ZZZUSDT": 2.0, "AAAUSDT": 2.0})
    ranked = rank_symbols(client_factory(bucket), "spot", ["ZZZUSDT", "AAAUSDT"], MONTH)
    assert [symbol for symbol, _ in ranked] == ["AAAUSDT", "ZZZUSDT"]


# --- building ----------------------------------------------------------------------------


def build(bucket, client_factory, **kwargs):
    return build_universe(
        client_factory(bucket), month=MONTH, top_n=kwargs.pop("top_n", {"spot": 2}), **kwargs
    )


def test_the_build_is_deterministic(client_factory):
    bucket = FakeBucket()
    seed(bucket, "spot", {"AAAUSDT": 1.0, "BBBUSDT": 5.0, "CCCUSDT": 3.0, "BTCUPUSDT": 9.0})
    seed(bucket, "spot", {"BTCUSDT": 7.0})

    first = build(bucket, client_factory)
    second = build(bucket, client_factory)
    assert first.to_yaml() == second.to_yaml()
    assert first.universe_hash == second.universe_hash


def test_the_build_applies_exclusions_and_the_cut(client_factory):
    bucket = FakeBucket()
    seed(
        bucket,
        "spot",
        {"BTCUSDT": 7.0, "BTCUPUSDT": 9.0, "USDCUSDT": 8.0, "BBBUSDT": 5.0, "CCCUSDT": 3.0},
    )
    universe = build(bucket, client_factory, top_n={"spot": 2})
    assert universe.symbols["spot"] == ["BTCUSDT", "BBBUSDT"]


def test_both_markets_are_built_independently(client_factory):
    bucket = FakeBucket()
    seed(bucket, "spot", {"BTCUSDT": 7.0, "ETHUSDT": 5.0})
    seed(bucket, "um", {"SOLUSDT": 4.0})
    universe = build(bucket, client_factory, top_n={"spot": 5, "um": 5})
    assert universe.symbols["spot"] == ["BTCUSDT", "ETHUSDT"]
    assert universe.symbols["um"] == ["SOLUSDT"]


def test_the_criteria_are_echoed_into_the_file(client_factory):
    bucket = FakeBucket()
    seed(bucket, "spot", {"BTCUSDT": 7.0})
    universe = build(bucket, client_factory)
    payload = yaml.safe_load(universe.to_yaml())
    assert payload["criteria"]["selection_month"] == MONTH
    assert payload["criteria"]["min_history_months"] == 12
    assert "USDC" in payload["criteria"]["excluded_bases"]
    assert payload["universe_hash"] == universe.universe_hash


# --- the hash ----------------------------------------------------------------------------


def make_config(**overrides) -> UniverseConfig:
    return UniverseConfig(
        criteria=UniverseCriteria(selection_month=MONTH, top_n={"spot": 2}),
        symbols={"spot": ["BTCUSDT", "ETHUSDT"]},
        **overrides,
    ).with_hash()


def test_the_hash_covers_the_symbol_list():
    other = UniverseConfig(
        criteria=UniverseCriteria(selection_month=MONTH, top_n={"spot": 2}),
        symbols={"spot": ["BTCUSDT", "SOLUSDT"]},
    ).with_hash()
    assert other.universe_hash != make_config().universe_hash


def test_the_hash_covers_the_criteria():
    other = UniverseConfig(
        criteria=UniverseCriteria(selection_month="2026-06", top_n={"spot": 2}),
        symbols={"spot": ["BTCUSDT", "ETHUSDT"]},
    ).with_hash()
    assert other.universe_hash != make_config().universe_hash


def test_symbol_order_is_part_of_the_identity():
    other = UniverseConfig(
        criteria=UniverseCriteria(selection_month=MONTH, top_n={"spot": 2}),
        symbols={"spot": ["ETHUSDT", "BTCUSDT"]},
    ).with_hash()
    assert other.universe_hash != make_config().universe_hash


def test_round_trip_through_yaml(tmp_path):
    path = tmp_path / "universe_v1.yaml"
    path.write_text(make_config().to_yaml(), encoding="utf-8")
    loaded = load_universe(path)
    assert loaded.universe_hash == make_config().universe_hash
    assert loaded.symbols["spot"] == ["BTCUSDT", "ETHUSDT"]


def test_an_edited_universe_file_is_refused(tmp_path):
    path = tmp_path / "universe_v1.yaml"
    payload = yaml.safe_load(make_config().to_yaml())
    payload["symbols"]["spot"].append("DOGEUSDT")
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="edited without rebuilding"):
        load_universe(path)


def test_an_unknown_key_in_the_universe_file_is_refused(tmp_path):
    path = tmp_path / "universe_v1.yaml"
    payload = yaml.safe_load(make_config().to_yaml())
    payload["markets"] = ["spot"]
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(Exception, match="markets"):
        load_universe(path)

"""The pinned Dukascopy universe is committed data, so the tests are about trusting the file.

There is no builder to test -- ADR-0015 pins 27 instruments by hand. What can go wrong is the file
drifting from its hash, gaining a duplicate, or losing the start dates that were measured against
the live feed, and each of those is checked here.
"""

from __future__ import annotations

import pytest
import yaml

from axiom.universe.dukascopy import (
    DukascopyInstrument,
    DukascopyUniverse,
    load_dukascopy_universe,
)

PACKAGED = "universe_dukascopy_v1"


def test_packaged_universe_loads_and_its_hash_holds():
    universe = load_dukascopy_universe(PACKAGED)
    assert universe.universe_hash == universe.compute_hash()
    assert len(universe.instruments) == 27


def test_every_instrument_is_complete_and_plausible():
    universe = load_dukascopy_universe(PACKAGED)
    for instrument in universe.instruments:
        assert instrument.symbol.isupper(), instrument.symbol
        assert instrument.source_symbol, instrument.symbol
        # Dukascopy's feed opens in 2003; nothing may claim history before it existed, and a
        # start date in the future would mean the whole instrument silently pulls nothing.
        assert 2003 <= instrument.start_year <= 2026, instrument.symbol


def test_asset_classes_are_both_present():
    by_class = {i.asset_class for i in load_dukascopy_universe(PACKAGED).instruments}
    assert by_class == {"fx", "commodity"}


def test_by_symbol_is_total_over_the_file():
    universe = load_dukascopy_universe(PACKAGED)
    assert set(universe.by_symbol()) == {i.symbol for i in universe.instruments}


def test_an_edited_file_is_refused(tmp_path):
    """The hash is the point: a hand-edited universe must not load silently."""
    universe = load_dukascopy_universe(PACKAGED)
    payload = yaml.safe_load(universe.to_yaml())
    payload["instruments"][0]["start_date"] = "1999-01-01"

    path = tmp_path / "tampered.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="was edited without rebuilding"):
        load_dukascopy_universe(path)


def test_duplicate_symbols_are_refused(tmp_path):
    universe = load_dukascopy_universe(PACKAGED)
    doubled = universe.model_copy(
        update={"instruments": [*universe.instruments, universe.instruments[0]]}
    )

    path = tmp_path / "doubled.yaml"
    path.write_text(doubled.to_yaml(), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate symbols"):
        load_dukascopy_universe(path)


def test_unknown_asset_class_is_refused():
    with pytest.raises(ValueError, match="unknown asset_class"):
        DukascopyInstrument(
            symbol="EURUSD", source_symbol="EUR/USD", asset_class="forex", start_date="2003-05-04"
        )


def test_hash_ignores_the_recorded_hash_field():
    """Otherwise writing the hash into the file would change the hash of the file."""
    universe = load_dukascopy_universe(PACKAGED)
    stripped = DukascopyUniverse(
        version=universe.version, criteria=universe.criteria, instruments=universe.instruments
    )
    assert stripped.compute_hash() == universe.compute_hash()

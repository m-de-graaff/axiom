"""The Dukascopy universe: 27 hand-pinned instruments (ADR-0015).

Unlike the Binance universe, this one is not computed. Choosing 27 instruments out of 27 does not
need a ranking procedure, so there is no builder here -- only the model that validates the
committed file and the hash that ties every manifest back to it.

What the file does carry that a hand-written list would not is a **measured** `start_date` per
instrument, taken from an actual full-history fetch. Dukascopy's coverage begins at a different
date for each instrument and the differences are years wide, so a guessed start year would either
spend a request per empty year forever or silently truncate history.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

from axiom.config.hashing import SHORT_LEN, canonical_json
from axiom.config.settings import resolve_config_path

ASSET_CLASSES = frozenset({"fx", "commodity"})


class DukascopyInstrument(BaseModel):
    """One pinned instrument: what we call it, what the feed calls it, and where it starts."""

    model_config = ConfigDict(extra="forbid")

    #: Canonical symbol, ours. Goes in the path and in `symbol`.
    symbol: str
    #: What `dukascopy_python.fetch` takes. The library's `INSTRUMENT_*` constants are plain
    #: strings, so this *is* the constant's value -- there is no map in code to disagree with it.
    source_symbol: str
    asset_class: str
    #: First date the daily series has a bar, measured by a full-history fetch. `YYYY-MM-DD`.
    start_date: str

    @field_validator("asset_class")
    @classmethod
    def _known_asset_class(cls, value: str) -> str:
        if value not in ASSET_CLASSES:
            raise ValueError(
                f"unknown asset_class {value!r}; expected one of {sorted(ASSET_CLASSES)}"
            )
        return value

    @property
    def start_year(self) -> int:
        return int(self.start_date[:4])


class DukascopyCriteria(BaseModel):
    """Why these instruments and how they are read. Echoed into the file, not inferred."""

    model_config = ConfigDict(extra="forbid")

    selection: str
    offer_side: str = "bid"
    price_side: str = "bid"
    volume_convention: str = "dukascopy_tick_volume"
    session_id: str = "24x5"
    exchange_tz: str = "UTC"
    frequencies: list[str]
    start_dates_measured_at: str
    index_cfds_included: bool = False


class DukascopyUniverse(BaseModel):
    """The pinned instrument set, its criteria, and their hash."""

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    criteria: DukascopyCriteria
    instruments: list[DukascopyInstrument]
    universe_hash: str = ""

    def compute_hash(self) -> str:
        payload: dict[str, Any] = {
            "version": self.version,
            "criteria": self.criteria.model_dump(mode="json"),
            "instruments": [i.model_dump(mode="json") for i in self.instruments],
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:SHORT_LEN]

    def with_hash(self) -> DukascopyUniverse:
        return self.model_copy(update={"universe_hash": self.compute_hash()})

    def to_yaml(self) -> str:
        payload = self.with_hash().model_dump(mode="json")
        return yaml.safe_dump(payload, sort_keys=True, default_flow_style=False, width=100)

    def by_symbol(self) -> dict[str, DukascopyInstrument]:
        return {i.symbol: i for i in self.instruments}


def load_dukascopy_universe(
    name_or_path: str | Path = "universe_dukascopy_v1",
) -> DukascopyUniverse:
    """Load the pinned file, refusing one whose recorded hash disagrees with its contents."""
    path = resolve_config_path(name_or_path)
    universe = DukascopyUniverse.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    symbols = [i.symbol for i in universe.instruments]
    if len(set(symbols)) != len(symbols):
        duplicated = sorted({s for s in symbols if symbols.count(s) > 1})
        raise ValueError(f"{path}: duplicate symbols {duplicated}")

    actual = universe.compute_hash()
    if universe.universe_hash and universe.universe_hash != actual:
        raise ValueError(
            f"{path}: records universe_hash={universe.universe_hash} but its contents hash to "
            f"{actual}; the file was edited without rebuilding"
        )
    return universe

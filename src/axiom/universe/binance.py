"""The Binance universe: a deterministic, pinned symbol list (ADR-0011).

The universe is code. It is a function of a pinned selection month and an explicit criteria
block, it is committed to git, and it carries a hash that every artifact manifest references.
Re-running the builder on the same month must reproduce the file byte for byte, which is why
nothing here reads the wall clock.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from axiom.config.hashing import SHORT_LEN, canonical_json
from axiom.config.settings import resolve_config_path
from axiom.sources.binance_klines import parse_archive
from axiom.sources.binance_vision import BinanceVision, NotFound, zip_url

log = logging.getLogger("axiom.universe")

QUOTE_ASSET = "USDT"

#: Suffixes Binance appends to a base asset to name a leveraged token.
LEVERAGE_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")

#: Bases that are pegged to the dollar by design. A pair between two of them does not move, and
#: teaching a price model that prices do not move is worse than teaching it nothing.
STABLE_BASES = frozenset(
    {"USDC", "TUSD", "FDUSD", "DAI", "BUSD", "USDP", "PAX", "SUSD", "USDS", "AEUR", "EURI"}
)

#: Fiat bases. These are FX, and FX arrives properly in v0.2 from Dukascopy with session
#: metadata and a real tick history. Admitting them here would put one instrument in the corpus
#: twice under two provenances.
FIAT_BASES = frozenset(
    {
        "EUR", "GBP", "AUD", "JPY", "TRY", "BRL", "ARS", "RUB", "ZAR", "PLN", "RON", "CZK",
        "UAH", "NGN", "IDRT", "BIDR", "VAI", "COP", "MXN", "CHF", "CAD", "NZD",
    }
)  # fmt: skip


class UniverseCriteria(BaseModel):
    """Everything that decides which symbols are in. Echoed into the emitted YAML."""

    model_config = ConfigDict(extra="forbid")

    selection_month: str
    ranking_metric: str = "sum of quote_asset_volume over the selection month's 1d bars"
    quote_asset: str = QUOTE_ASSET
    top_n: dict[str, int]
    exclude_leveraged: bool = True
    excluded_bases: list[str] = Field(default_factory=list)
    min_history_days: int = 365


class UniverseConfig(BaseModel):
    """A pinned universe: the criteria that produced it, the lists, and their hash."""

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    criteria: UniverseCriteria
    symbols: dict[str, list[str]]
    universe_hash: str = ""

    def compute_hash(self) -> str:
        """Identity over the criteria and the lists. Excludes the recorded hash itself."""
        payload: dict[str, Any] = {
            "version": self.version,
            "criteria": self.criteria.model_dump(mode="json"),
            "symbols": {k: list(v) for k, v in self.symbols.items()},
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:SHORT_LEN]

    def with_hash(self) -> UniverseConfig:
        return self.model_copy(update={"universe_hash": self.compute_hash()})

    def to_yaml(self) -> str:
        payload = self.with_hash().model_dump(mode="json")
        return yaml.safe_dump(payload, sort_keys=True, default_flow_style=False, width=100)


def load_universe(name_or_path: str | Path) -> UniverseConfig:
    """Load a universe file, refusing one whose recorded hash does not match its contents."""
    path = resolve_config_path(name_or_path)
    config = UniverseConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    actual = config.compute_hash()
    if config.universe_hash and config.universe_hash != actual:
        raise ValueError(
            f"{path}: records universe_hash={config.universe_hash} but its contents hash to "
            f"{actual}; the file was edited without rebuilding"
        )
    return config


def base_asset(symbol: str, quote: str = QUOTE_ASSET) -> str:
    return symbol[: -len(quote)] if symbol.endswith(quote) else symbol


def is_leveraged(symbol: str, listed: set[str], quote: str = QUOTE_ASSET) -> bool:
    """True for `BTCUPUSDT` and `ETHBEARUSDT`, false for `JUPUSDT`.

    A bare suffix match eats real tokens whose ticker happens to end in those letters — `JUP` is
    Jupiter, not a leveraged JU. A leveraged token is named by appending the suffix to an asset
    that Binance also lists on its own, so the test is whether stripping the suffix leaves a
    symbol the bucket actually has.
    """
    base = base_asset(symbol, quote)
    for suffix in LEVERAGE_SUFFIXES:
        stripped = base[: -len(suffix)]
        if base.endswith(suffix) and len(stripped) >= 2 and f"{stripped}{quote}" in listed:
            return True
    return False


def candidate_symbols(listed: list[str], quote: str = QUOTE_ASSET) -> list[str]:
    """Apply the ADR-0011 filter to a raw listing."""
    universe = set(listed)
    excluded = STABLE_BASES | FIAT_BASES
    return [
        symbol
        for symbol in sorted(listed)
        if symbol.endswith(quote)
        and base_asset(symbol, quote) not in excluded
        and base_asset(symbol, quote) != ""
        and not is_leveraged(symbol, universe, quote)
    ]


def month_quote_volume(client: BinanceVision, market: str, symbol: str, month: str) -> float:
    """Summed quote volume over one month of 1d bars, or 0.0 when the month is not published."""
    url = zip_url(market, "monthly", symbol, "1d", month)
    try:
        archive = client.fetch_verified(url)
    except NotFound:
        return 0.0
    table = parse_archive(archive.data)
    return float(np.nansum(table["amount"].to_numpy(zero_copy_only=False)))


def rank_symbols(
    client: BinanceVision,
    market: str,
    symbols: list[str],
    month: str,
) -> list[tuple[str, float]]:
    """Rank candidates by the selection month's quote volume, descending.

    Ties break on the symbol name so the ordering is total. Two symbols with identical volume is
    vanishingly unlikely, and "vanishingly unlikely" is exactly the kind of thing that makes a
    supposedly reproducible file differ between two runs a year apart.
    """
    volumes = client.run_all(lambda s: month_quote_volume(client, market, s, month), symbols)
    ranked = [(s, v) for s, v in zip(symbols, volumes, strict=True) if v > 0.0]
    return sorted(ranked, key=lambda pair: (-pair[1], pair[0]))


def build_universe(
    client: BinanceVision,
    *,
    month: str,
    top_n: dict[str, int],
    min_history_days: int = 365,
) -> UniverseConfig:
    """Enumerate, filter, rank, and cut to the top N per market."""
    symbols: dict[str, list[str]] = {}
    for market, count in sorted(top_n.items()):
        listed = client.list_symbols(market)
        candidates = candidate_symbols(listed)
        log.info(
            "%s: %d listed, %d candidates after exclusions", market, len(listed), len(candidates)
        )
        ranked = rank_symbols(client, market, candidates, month)
        log.info("%s: %d traded in %s, keeping top %d", market, len(ranked), month, count)
        symbols[market] = [symbol for symbol, _ in ranked[:count]]

    config = UniverseConfig(
        criteria=UniverseCriteria(
            selection_month=month,
            top_n=dict(sorted(top_n.items())),
            excluded_bases=sorted(STABLE_BASES | FIAT_BASES),
            min_history_days=min_history_days,
        ),
        symbols=symbols,
    )
    return config.with_hash()

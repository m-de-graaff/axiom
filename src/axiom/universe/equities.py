"""The US equities training universe: data-driven, not hand-pinned (ADR-0016).

The criteria were fixed in the ADR before any equity data existed; the *list* could not be,
because ranking by dollar volume requires the bars. So this is the one universe in the project
with a builder that reads the corpus.

Two passes, and the split matters. The history filter is answered from the registry -- no
downloads, eighteen thousand rows of metadata. Only the survivors are then downloaded to compute
the ranking metric. Ranking first would mean pulling every artifact in the tier to sort a list
that throws most of them away.

The pulled corpus is a superset of this universe, always. Everything stays in `axiom-raw`; the
universe governs *sampling* from v0.5 onward, not what gets stored. A ticker that falls out of a
later universe does not get deleted, and one that was never in it is still there to be added.
"""

from __future__ import annotations

import hashlib
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel, ConfigDict

from axiom.config.hashing import SHORT_LEN, canonical_json
from axiom.config.settings import resolve_config_path
from axiom.raw.store import RawStore

log = logging.getLogger("axiom.universe.equities")

MS_PER_DAY = 86_400_000

#: ADR-0016's criteria, as defaults. Changing one of these is a new universe version, not an
#: argument tweak -- which is why they are echoed into the emitted file rather than assumed.
MIN_HISTORY_YEARS = 5
TOP_N = 3000
#: Trading days the ranking metric is measured over. One year, so the ranking reflects what is
#: liquid now rather than what was liquid in 2009.
RANKING_WINDOW_DAYS = 252


class EquityCriteria(BaseModel):
    """What decides membership. Echoed into the emitted YAML, never inferred from it."""

    model_config = ConfigDict(extra="forbid")

    min_history_years: int = MIN_HISTORY_YEARS
    top_n: int = TOP_N
    ranking_metric: str = (
        "median of close x volume over the last 252 bars, per ticker, descending; "
        "ties broken on the symbol so the ordering is total"
    )
    ranking_window_days: int = RANKING_WINDOW_DAYS
    source: str = "stooq"
    market: str = "us"
    frequency: str = "1d"
    generated_at: str
    #: Which snapshot of the corpus this was derived from. Without it, "top 3000 by dollar
    #: volume" names a procedure but not a result.
    registry_hash: str


class EquityUniverse(BaseModel):
    """A pinned equities universe: the criteria that produced it, the list, and their hash."""

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    criteria: EquityCriteria
    symbols: list[str]
    #: Candidates that passed the history filter but lost the ranking cut. Recorded as a count,
    #: because "3000 of 3000 candidates" and "3000 of 14000" describe very different universes.
    candidates_considered: int = 0
    universe_hash: str = ""

    def compute_hash(self) -> str:
        payload: dict[str, Any] = {
            "version": self.version,
            "criteria": self.criteria.model_dump(mode="json"),
            "symbols": list(self.symbols),
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:SHORT_LEN]

    def with_hash(self) -> EquityUniverse:
        return self.model_copy(update={"universe_hash": self.compute_hash()})

    def to_yaml(self) -> str:
        header = (
            "# US equities training universe, built by `axiom universe build-equities`.\n"
            "#\n"
            "# The pulled corpus is a SUPERSET of this list. Everything stays in `axiom-raw`;\n"
            "# this file governs sampling from v0.5 onward, not what gets stored. A ticker that\n"
            "# falls out of a later version is not deleted, and one that was never here is still\n"
            "# available to add.\n"
            "#\n"
            "# Survivorship: the source dump skews to currently-listed tickers, so this universe\n"
            "# inherits that skew and every backtest number derived from it is biased upward.\n"
            "# Accepted and documented per ADR-0016; it goes in the v0.9 model card too.\n"
        )
        return header + yaml.safe_dump(
            self.with_hash().model_dump(mode="json"),
            sort_keys=True,
            default_flow_style=False,
            width=100,
        )


def load_equity_universe(name_or_path: str | Path = "universe_equities_v1") -> EquityUniverse:
    """Load a built universe, refusing one whose recorded hash disagrees with its contents."""
    path = resolve_config_path(name_or_path)
    universe = EquityUniverse.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    actual = universe.compute_hash()
    if universe.universe_hash and universe.universe_hash != actual:
        raise ValueError(
            f"{path}: records universe_hash={universe.universe_hash} but its contents hash to "
            f"{actual}; the file was edited without rebuilding"
        )
    return universe


@dataclass(frozen=True)
class Candidate:
    """A ticker that cleared the history filter and is waiting to be ranked."""

    symbol: str
    artifact_path: str
    history_days: float


def candidates_from_registry(
    registry: pa.Table,
    *,
    min_history_years: int = MIN_HISTORY_YEARS,
    source: str = "stooq",
    frequency: str = "1d",
) -> list[Candidate]:
    """The history filter, answered from metadata alone.

    Eighteen thousand rows of registry instead of eighteen thousand downloads. Whatever fraction
    of the tier fails this filter is a download that never happens.
    """
    needed_days = min_history_years * 365.25
    rows = registry.to_pylist()
    candidates = [
        Candidate(r["symbol"], r["artifact_path"], r["history_days"])
        for r in rows
        if r["source"] == source
        and r["frequency"] == frequency
        and r["history_days"] >= needed_days
    ]
    log.info(
        "%d of %d %s series have at least %d years of history",
        len(candidates),
        sum(1 for r in rows if r["source"] == source and r["frequency"] == frequency),
        source,
        min_history_years,
    )
    return sorted(candidates, key=lambda c: c.symbol)


def median_dollar_volume(table: pa.Table, *, window: int = RANKING_WINDOW_DAYS) -> float:
    """Median of `close x volume` over the last ``window`` bars.

    Median rather than mean because a single earnings-day spike should not carry a ticker into
    the universe, and dollar volume rather than share volume because a hundred shares of one
    stock and a hundred of another are not comparable quantities.
    """
    if table.num_rows == 0:
        return 0.0
    close = table["close"].to_numpy(zero_copy_only=False)[-window:]
    volume = table["volume"].to_numpy(zero_copy_only=False)[-window:]
    dollar = close.astype(np.float64) * volume.astype(np.float64)
    finite = dollar[np.isfinite(dollar)]
    return float(np.median(finite)) if finite.size else 0.0


def rank_candidates(
    store: RawStore,
    candidates: list[Candidate],
    *,
    window: int = RANKING_WINDOW_DAYS,
    concurrency: int = 16,
) -> list[tuple[str, float]]:
    """Download each candidate's close and volume columns and rank by median dollar volume.

    Ties break on the symbol, so the ordering is total. Two tickers with identical median dollar
    volume is unlikely and "unlikely" is exactly what makes a supposedly reproducible file differ
    between two runs a year apart.
    """

    def measure(candidate: Candidate) -> tuple[str, float]:
        try:
            data = store.get(candidate.artifact_path)
            if data is None:
                return candidate.symbol, 0.0
            table = pq.read_table(io.BytesIO(data), columns=["close", "volume"])
            return candidate.symbol, median_dollar_volume(table, window=window)
        except Exception as exc:
            log.warning("could not rank %s: %s", candidate.symbol, exc)
            return candidate.symbol, 0.0

    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="rank") as pool:
        measured = list(pool.map(measure, candidates))

    ranked = [(symbol, value) for symbol, value in measured if value > 0.0]
    return sorted(ranked, key=lambda pair: (-pair[1], pair[0]))


def build_equity_universe(
    store: RawStore,
    registry: pa.Table,
    *,
    registry_hash: str,
    generated_at: str,
    min_history_years: int = MIN_HISTORY_YEARS,
    top_n: int = TOP_N,
    window: int = RANKING_WINDOW_DAYS,
    concurrency: int = 16,
) -> EquityUniverse:
    """Filter by history, rank by dollar volume, cut to the top N."""
    candidates = candidates_from_registry(registry, min_history_years=min_history_years)
    ranked = rank_candidates(store, candidates, window=window, concurrency=concurrency)
    log.info("%d candidates ranked, keeping the top %d", len(ranked), top_n)

    universe = EquityUniverse(
        criteria=EquityCriteria(
            min_history_years=min_history_years,
            top_n=top_n,
            ranking_window_days=window,
            generated_at=generated_at,
            registry_hash=registry_hash,
        ),
        symbols=[symbol for symbol, _ in ranked[:top_n]],
        candidates_considered=len(ranked),
    )
    return universe.with_hash()

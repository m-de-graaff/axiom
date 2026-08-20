"""The Binance pull, expressed as a :class:`~axiom.sources.base.Source`.

Everything generic moved to `sources/base.py` in v0.2: the skip test, validation, the Parquet
write, the sidecar, the run manifest, the per-item blast wall. What is left here is what is
actually Binance's -- how a series is enumerated out of the bucket's monthly and daily archives,
and how those archives become a schema-v1 table.

The module-level functions below the class are the v0.1 API, kept working unchanged. That is the
refactor's acceptance test: every v0.1 test still passes without being edited.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pyarrow as pa

from axiom.raw.store import RawStore
from axiom.schema.bars import validate_bars
from axiom.sources.base import (
    PullResult,
    PullRun,
    SourcePlan,
    WorkItem,
    loader_version,
    pull_item,
    write_parquet,
)
from axiom.sources.base import run_pull as _run_pull
from axiom.sources.binance_klines import merge, parse_archive
from axiom.sources.binance_vision import BinanceVision, zip_url

log = logging.getLogger("axiom.pull")

SOURCE = "binance_vision"
ASSET_CLASS = "crypto"

__all__ = [
    "ASSET_CLASS",
    "SOURCE",
    "BinanceSource",
    "PullResult",
    "PullRun",
    "PullTask",
    "artifact_path",
    "build_table",
    "build_tasks",
    "enumerate_sources",
    "loader_version",
    "pull_symbol",
    "run_pull",
    "write_parquet",
]


def artifact_path(market: str, frequency: str, symbol: str) -> str:
    """Where one series lives in `axiom-raw`. The path is half the identity (ADR-0010)."""
    return f"raw/binance/{market}/{frequency}/{symbol}.parquet"


#: v0.1's name for a work item. `WorkItem`'s defaults are the crypto answers -- asset class
#: `crypto`, `exchange_tz` UTC, `session_id` 24x7 -- so `PullTask("spot", "BTCUSDT", "1h")` still
#: means exactly what it did before the framework existed.
PullTask = WorkItem


def enumerate_sources(client: BinanceVision, task: WorkItem) -> list[str]:
    """Every archive URL that makes up this series, monthly first, then the daily tail.

    The daily tail is only the days *after* the last complete month. Binance publishes daily
    archives for days that are already inside a published month too, and taking those as well
    would double the download for no new bars -- the seam dedup would then quietly throw the
    duplicates away, hiding the waste.
    """
    months = client.list_periods(task.market, "monthly", task.symbol, task.frequency)
    days = client.list_periods(task.market, "daily", task.symbol, task.frequency)
    if months:
        last_month = months[-1]
        days = [d for d in days if d[:7] > last_month]

    return [
        zip_url(task.market, "monthly", task.symbol, task.frequency, period) for period in months
    ] + [zip_url(task.market, "daily", task.symbol, task.frequency, period) for period in days]


def build_table(
    client: BinanceVision,
    task: WorkItem,
    urls: list[str],
    digests: list[str] | None = None,
    *,
    validate: bool = True,
) -> pa.Table:
    """Download, verify, parse, merge, validate. Returns the table ready to write.

    ``validate=False`` exists for `axiom raw inspect`, which needs to look at a series precisely
    because it fails validation. Nothing that writes may pass it.
    """
    archives = client.fetch_all(urls, digests)
    tables = [parse_archive(archive.data) for archive in archives]
    table = merge(tables, context=str(task))
    if validate:
        validate_bars(table, task.frequency, session_id=task.session_id, raise_on_error=True)
    return table


class BinanceSource:
    """Binance Vision as the driver sees it."""

    name = SOURCE

    def __init__(self, client: BinanceVision) -> None:
        self.client = client

    def artifact_path(self, item: WorkItem) -> str:
        return artifact_path(item.market, item.frequency, item.symbol)

    def plan(self, item: WorkItem) -> SourcePlan:
        """List the archives, then fetch only their published digests.

        The digests alone decide whether the series needs downloading at all, and they are a few
        dozen bytes each against megabytes of archive -- so a resumed run costs one listing plus a
        handful of tiny CHECKSUM fetches per finished symbol.
        """
        urls = enumerate_sources(self.client, item)
        if not urls:
            return SourcePlan([], [])
        return SourcePlan(urls, self.client.fetch_checksums(urls))

    def build(
        self,
        item: WorkItem,
        plan: SourcePlan,
        load_existing: Callable[[], pa.Table | None],
    ) -> pa.Table:
        """Rebuild the whole series from its archives. Nothing is spliced; nothing is appended."""
        return build_table(self.client, item, plan.source_urls, plan.source_sha256s)

    def manifest_extras(self, item: WorkItem) -> dict[str, Any]:
        """Binance publishes trade prices and its own quote volume, so nothing is synthesized."""
        return {
            "price_side": "trade",
            "volume_convention": "base+quote_native",
            "amount_synthesized": False,
            "adjustment_policy": "none",
            "redistribution_class": "loader_manifest_private_cache",
        }


# --- the v0.1 API, unchanged ------------------------------------------------------------


def pull_symbol(
    client: BinanceVision,
    store: RawStore,
    task: WorkItem,
    *,
    pull_run_id: str,
    universe_hash: str,
    force: bool = False,
) -> PullResult:
    """Bring one series up to date in the raw tier, or report why it could not be."""
    return pull_item(
        BinanceSource(client),
        store,
        task,
        pull_run_id=pull_run_id,
        universe_hash=universe_hash,
        force=force,
    )


def run_pull(
    client: BinanceVision,
    store: RawStore,
    tasks: list[WorkItem],
    manifest,
    *,
    force: bool = False,
) -> PullRun:
    """Walk the work list, one task at a time, flushing the store at the end."""
    return _run_pull(BinanceSource(client), store, tasks, manifest, force=force)


def build_tasks(
    universe: dict[str, list[str]],
    markets: list[str],
    frequencies: list[str],
    *,
    symbols: list[str] | None = None,
    limit: int | None = None,
) -> list[WorkItem]:
    """The work list, in a stable order so a resumed run walks it the same way.

    ``limit`` counts symbols per market, not tasks, so `--limit 40 --frequencies 1h,1d` is
    forty symbols at both frequencies rather than twenty at each.
    """
    tasks: list[WorkItem] = []
    for market in markets:
        chosen = universe.get(market, [])
        if symbols:
            wanted = {s.upper() for s in symbols}
            chosen = [s for s in chosen if s.upper() in wanted]
        if limit is not None:
            chosen = chosen[:limit]
        for symbol in chosen:
            tasks.extend(PullTask(market, symbol, frequency) for frequency in frequencies)
    return tasks

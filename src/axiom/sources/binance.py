"""The Binance pull: one symbol at a time, resumable by construction.

`pull_symbol` is the unit of work and the whole design. It is a function of three things — the
universe entry, what the bucket currently publishes, and what `axiom-raw` already holds — and of
nothing else. There is no checkpoint file, no cursor, no progress database. Restarting a killed
pull works because the second run asks the same three questions and gets "already done" for
everything the first run finished (ADR-0010).
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq

from axiom import __version__
from axiom.ops.logx import git_commit
from axiom.provenance.manifest import (
    FileManifest,
    PullFailure,
    PullRunManifest,
    is_current,
    sha256_bytes,
)
from axiom.raw.store import RawStore
from axiom.schema.bars import ROW_GROUP_SIZE, bars_metadata, count_gaps, validate_bars
from axiom.sources.binance_klines import merge, parse_archive
from axiom.sources.binance_vision import BinanceVision, zip_url

log = logging.getLogger("axiom.pull")

SOURCE = "binance_vision"
ASSET_CLASS = "crypto"


def loader_version() -> str:
    """Package version plus the commit it was built from, for the manifest."""
    return f"{__version__}+{git_commit()}"


def artifact_path(market: str, frequency: str, symbol: str) -> str:
    """Where one series lives in `axiom-raw`. The path is half the identity (ADR-0010)."""
    return f"raw/binance/{market}/{frequency}/{symbol}.parquet"


@dataclass(frozen=True)
class PullTask:
    """One (market, symbol, frequency) series to bring up to date."""

    market: str
    symbol: str
    frequency: str

    def __str__(self) -> str:
        return f"{self.market}/{self.frequency}/{self.symbol}"


@dataclass
class PullResult:
    """What happened to one task. ``status`` is one of ok, skipped, failed."""

    task: PullTask
    status: str
    manifest: FileManifest | None = None
    error: str = ""
    bytes_written: int = 0

    @property
    def rows(self) -> int:
        return self.manifest.row_count if self.manifest else 0


def enumerate_sources(client: BinanceVision, task: PullTask) -> list[str]:
    """Every archive URL that makes up this series, monthly first, then the daily tail.

    The daily tail is only the days *after* the last complete month. Binance publishes daily
    archives for days that are already inside a published month too, and taking those as well
    would double the download for no new bars — the seam dedup would then quietly throw the
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


def build_table(client: BinanceVision, task: PullTask, urls: list[str], digests: list[str]):
    """Download, verify, parse, merge, validate. Returns the table ready to write."""
    archives = client.fetch_all(urls, digests)
    tables = [parse_archive(archive.data) for archive in archives]
    table = merge(tables, context=str(task))
    validate_bars(table, task.frequency, raise_on_error=True)
    return table


def write_parquet(table: pa.Table, metadata: dict[bytes, bytes]) -> bytes:
    """Serialize to Parquet bytes with the ADR-0010 settings.

    In memory rather than to a file because the bytes are what gets hashed and uploaded, and a
    round trip through the container filesystem would add a step that can fail without adding
    anything that can be checked.
    """
    buffer = io.BytesIO()
    pq.write_table(
        table.replace_schema_metadata(metadata),
        buffer,
        compression="zstd",
        row_group_size=ROW_GROUP_SIZE,
    )
    return buffer.getvalue()


def _remote_sidecar(store: RawStore, path: str) -> FileManifest | None:
    """The remote sidecar, or ``None`` when there is not a usable one.

    An unreadable or tampered sidecar is treated as absent rather than as an error. The pull is
    idempotent, so re-pulling overwrites the damage and the series heals; failing the symbol
    instead would leave it stuck behind a corrupt file that only a manual delete could clear.
    """
    try:
        return store.read_sidecar(path)
    except ValueError as exc:
        log.warning("unusable sidecar for %s (%s); re-pulling", path, exc)
        return None


def pull_symbol(
    client: BinanceVision,
    store: RawStore,
    task: PullTask,
    *,
    pull_run_id: str,
    universe_hash: str,
    force: bool = False,
) -> PullResult:
    """Bring one series up to date in the raw tier, or report why it could not be."""
    path = artifact_path(task.market, task.frequency, task.symbol)
    try:
        urls = enumerate_sources(client, task)
        if not urls:
            return PullResult(task, "failed", error="no archives published for this series")

        digests = client.fetch_checksums(urls)
        if not force and is_current(_remote_sidecar(store, path), digests):
            log.info("skip %s (already current, %d source files)", task, len(urls))
            return PullResult(task, "skipped")

        table = build_table(client, task, urls, digests)
        ts = table["ts"].to_numpy(zero_copy_only=False)

        # Two passes, because the Parquet metadata carries the manifest hash and the manifest
        # carries the artifact hash. The identity hash is computed first (it excludes the
        # artifact hash for exactly this reason), then the bytes, then the artifact hash.
        manifest = FileManifest(
            schema_version=1,
            source=SOURCE,
            market=task.market,
            asset_class=ASSET_CLASS,
            symbol=task.symbol,
            frequency=task.frequency,
            pull_run_id=pull_run_id,
            pulled_at=datetime.now(UTC).isoformat(),
            loader_version=loader_version(),
            source_urls=urls,
            source_sha256s=digests,
            artifact_path=path,
            row_count=table.num_rows,
            first_ts=int(ts[0]),
            last_ts=int(ts[-1]),
            gap_count=count_gaps(ts, task.frequency),
            universe_hash=universe_hash,
        )
        data = write_parquet(
            table,
            bars_metadata(
                source=SOURCE,
                asset_class=ASSET_CLASS,
                market=task.market,
                symbol=task.symbol,
                frequency=task.frequency,
                manifest_sha256=manifest.manifest_sha256,
            ),
        )
        manifest = manifest.model_copy(update={"artifact_sha256": sha256_bytes(data)})

        store.put(path, data, manifest)
        log.info(
            "ok %s: %d rows, %d gaps, %d bytes",
            task,
            table.num_rows,
            manifest.gap_count,
            len(data),
        )
        return PullResult(task, "ok", manifest=manifest, bytes_written=len(data))

    except Exception as exc:  # one bad symbol must not end a 300-symbol run
        log.warning("failed %s: %s: %s", task, type(exc).__name__, exc)
        return PullResult(task, "failed", error=f"{type(exc).__name__}: {exc}")


def build_tasks(
    universe: dict[str, list[str]],
    markets: list[str],
    frequencies: list[str],
    *,
    symbols: list[str] | None = None,
    limit: int | None = None,
) -> list[PullTask]:
    """The work list, in a stable order so a resumed run walks it the same way.

    ``limit`` counts symbols per market, not tasks, so `--limit 40 --frequencies 1h,1d` is
    forty symbols at both frequencies rather than twenty at each.
    """
    tasks: list[PullTask] = []
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


@dataclass
class PullRun:
    """Accumulates results into the run manifest as the work list is walked."""

    manifest: PullRunManifest
    results: list[PullResult] = field(default_factory=list)

    def record(self, result: PullResult) -> None:
        self.results.append(result)
        if result.status == "ok":
            self.manifest.ok += 1
            self.manifest.total_rows += result.rows
            self.manifest.total_bytes += result.bytes_written
        elif result.status == "skipped":
            self.manifest.skipped += 1
        else:
            self.manifest.failed += 1
            self.manifest.failures.append(
                PullFailure(
                    market=result.task.market,
                    symbol=result.task.symbol,
                    frequency=result.task.frequency,
                    error=result.error,
                )
            )

    def finish(self) -> PullRunManifest:
        self.manifest.finished_at = datetime.now(UTC).isoformat()
        return self.manifest


def run_pull(
    client: BinanceVision,
    store: RawStore,
    tasks: list[PullTask],
    manifest: PullRunManifest,
    *,
    force: bool = False,
) -> PullRun:
    """Walk the work list, one task at a time, flushing the store at the end.

    Sequential over tasks on purpose. The parallelism that matters lives one level down, where a
    symbol's fifty monthly archives are fetched at once through the client's shared pool; adding
    a second layer here would multiply the two caps together and stop being polite.
    """
    run = PullRun(manifest)
    for index, task in enumerate(tasks, start=1):
        log.info("[%d/%d] %s", index, len(tasks), task)
        run.record(
            pull_symbol(
                client,
                store,
                task,
                pull_run_id=manifest.pull_run_id,
                universe_hash=manifest.universe_hash,
                force=force,
            )
        )
    store.flush()
    return run

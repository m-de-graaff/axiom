"""The source framework: one driver, many sources.

One source is a script. Four sources are a framework, and the difference is that the second
loader must not be a copy of the first with the words changed. So everything that is true of
*every* pull lives here -- the skip test, validation, the Parquet write, the sidecar, the run
manifest, the per-item blast wall -- and a source supplies only the four things that are
genuinely its own.

The driver's contract with a source is deliberately narrow:

``plan(item)``
    What this series is made of right now, as a list of source identifiers and a parallel list of
    digests. Cheap: no bar data is downloaded here, because the digests alone decide whether
    anything needs downloading at all.

``build(item, plan, load_existing)``
    The bars, as a schema-v1 table. ``load_existing`` reads back what the raw tier already holds
    for this series, for a source that extends a series rather than rebuilding it. Sources that
    rebuild ignore it.

``manifest_extras(item)``
    The per-source manifest fields from ADR-0014 -- ``price_side``, ``volume_convention``,
    ``amount_synthesized``, ``adjustment_policy``, ``redistribution_class``.

``artifact_path(item)``
    Where the file goes. Half the identity (ADR-0010).

Retries, backoff and connection concurrency are **not** here. They are transport, they differ per
source in kind rather than in degree -- an S3 bucket, a broker library with its own retry loop, a
single archive URL -- and hoisting them would mean the driver owning an HTTP client that two of
the three sources never use.
"""

from __future__ import annotations

import io
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

import numpy as np
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
from axiom.schema.bars import (
    ROW_GROUP_SIZE,
    bars_metadata,
    count_closed_window,
    count_gaps,
    count_off_grid,
    validate_bars,
)

log = logging.getLogger("axiom.pull")


def _kill_after_items() -> int | None:
    """Fault injection for the kill drill, mirroring the v0.0 loop's `AXIOM_KILL_AT_STEP`.

    A pull's resume story is only worth as much as the drill that tested it, and on a backend
    with no scriptable cancel -- a Kaggle kernel -- there is otherwise no way to kill a run
    mid-flight from a script.
    """
    raw = os.environ.get("AXIOM_KILL_AFTER_ITEMS")
    return int(raw) if raw else None


def loader_version() -> str:
    """Package version plus the commit it was built from, for the manifest."""
    return f"{__version__}+{git_commit()}"


@dataclass(frozen=True)
class WorkItem:
    """One series to bring up to date: its identity, and nothing else.

    Per-source state -- which years a chunked source will fetch, which archive member a bulk
    source will read -- stays inside the source that enumerated the item. The driver only ever
    needs to know what the file will be called and what to write in its manifest.
    """

    market: str
    symbol: str
    frequency: str
    asset_class: str = "crypto"
    #: What the vendor calls this instrument, when that differs. ADR-0014.
    source_symbol: str = ""
    #: File-level metadata, per ADR-0014. Constant within a file, never a column.
    exchange_tz: str = "UTC"
    session_id: str = "24x7"

    def __str__(self) -> str:
        return f"{self.market}/{self.frequency}/{self.symbol}"

    @property
    def vendor_symbol(self) -> str:
        return self.source_symbol or self.symbol


@dataclass(frozen=True)
class SourcePlan:
    """What a series is currently made of: parallel lists of identifiers and their digests.

    ``source_urls`` need not be URLs. They are whatever names a unit of upstream data for this
    source, and their only hard requirement is that the pair of lists changes exactly when the
    data does -- because that comparison is the entire resume mechanism (`provenance.is_current`).
    """

    source_urls: list[str]
    source_sha256s: list[str]

    def __post_init__(self) -> None:
        if len(self.source_urls) != len(self.source_sha256s):
            raise ValueError(
                f"{len(self.source_urls)} source ids but {len(self.source_sha256s)} digests; "
                "they are parallel lists and a mismatch means one of them is wrong"
            )

    def __bool__(self) -> bool:
        return bool(self.source_urls)


class Source(Protocol):
    """What the driver needs from a loader. Four methods and a name."""

    #: Goes into every manifest's `source` field, and into the artifact path.
    name: str

    def artifact_path(self, item: WorkItem) -> str: ...

    def plan(self, item: WorkItem) -> SourcePlan: ...

    def build(
        self,
        item: WorkItem,
        plan: SourcePlan,
        load_existing: Callable[[], pa.Table | None],
    ) -> pa.Table: ...

    def manifest_extras(self, item: WorkItem) -> dict[str, Any]: ...


@dataclass
class PullResult:
    """What happened to one item. ``status`` is one of ok, skipped, failed."""

    task: WorkItem
    status: str
    manifest: FileManifest | None = None
    error: str = ""
    bytes_written: int = 0

    @property
    def rows(self) -> int:
        return self.manifest.row_count if self.manifest else 0


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


#: Bars the dollar-volume statistic is measured over. One trading year, matching what the
#: equities universe ranks on (ADR-0016).
DOLLAR_VOLUME_WINDOW = 252


def median_dollar_volume(table: pa.Table, *, window: int = DOLLAR_VOLUME_WINDOW) -> float:
    """Median of `close x volume` over the last ``window`` bars.

    Median rather than mean, so one earnings-day spike cannot carry a series; dollar volume
    rather than share volume, because a hundred shares of one instrument and a hundred of another
    are not comparable quantities.
    """
    if table.num_rows == 0:
        return 0.0
    close = table["close"].to_numpy(zero_copy_only=False)[-window:]
    volume = table["volume"].to_numpy(zero_copy_only=False)[-window:]
    dollar = close.astype(np.float64) * volume.astype(np.float64)
    finite = dollar[np.isfinite(dollar)]
    return float(np.median(finite)) if finite.size else 0.0


def shard_dir(symbol: str) -> str:
    """The letter bucket a symbol's file lands in (ADR-0016).

    The Hub degrades past roughly ten thousand files in one folder, and the equities tier is
    twelve to eighteen thousand series. Bucketing on the first character splits that into
    twenty-odd folders without anyone maintaining a mapping. Anything that is not a letter or a
    digit -- and there are tickers with dots and dashes -- goes to ``_`` rather than creating a
    folder named after a punctuation mark.
    """
    if not symbol:
        raise ValueError("cannot shard an empty symbol")
    head = symbol[0].upper()
    return head if head.isalnum() else "_"


def bucket_counts(symbols: list[str]) -> dict[str, int]:
    """Series per letter bucket, for the ADR-0016 folder guard.

    The Hub degrades past roughly ten thousand files in one folder and each series lands two
    files -- the Parquet and its sidecar -- so the count that matters is twice this.
    """
    counts: dict[str, int] = {}
    for symbol in symbols:
        bucket = shard_dir(symbol)
        counts[bucket] = counts.get(bucket, 0) + 1
    return counts


def read_existing_table(store: RawStore, path: str) -> pa.Table | None:
    """Read back an artifact the raw tier already holds, or ``None`` if it holds none.

    Only a source that *extends* a series needs this. It is a plain read: a source that splices
    onto it is responsible for deciding which of its rows to keep.
    """
    data = store.get(path)
    if data is None:
        return None
    return pq.read_table(io.BytesIO(data))


def _remote_sidecar(store: RawStore, path: str) -> FileManifest | None:
    """The remote sidecar, or ``None`` when there is not a usable one.

    An unreadable or tampered sidecar is treated as absent rather than as an error. The pull is
    idempotent, so re-pulling overwrites the damage and the series heals; failing the item
    instead would leave it stuck behind a corrupt file that only a manual delete could clear.
    """
    try:
        return store.read_sidecar(path)
    except ValueError as exc:
        log.warning("unusable sidecar for %s (%s); re-pulling", path, exc)
        return None


def pull_item(
    source: Source,
    store: RawStore,
    item: WorkItem,
    *,
    pull_run_id: str,
    universe_hash: str,
    force: bool = False,
) -> PullResult:
    """Bring one series up to date in the raw tier, or report why it could not be.

    This is the whole design. It is a function of three things -- the work item, what the source
    currently publishes, and what `axiom-raw` already holds -- and of nothing else. There is no
    checkpoint file, no cursor, no progress database. Restarting a killed pull works because the
    second run asks the same three questions and gets "already done" for everything the first run
    finished (ADR-0010).
    """
    path = source.artifact_path(item)
    try:
        plan = source.plan(item)
        if not plan:
            return PullResult(
                item, "failed", error="no archives published upstream for this series"
            )

        if not force and is_current(_remote_sidecar(store, path), plan.source_sha256s):
            log.info("skip %s (already current, %d source file(s))", item, len(plan.source_urls))
            return PullResult(item, "skipped")

        table = source.build(item, plan, lambda: read_existing_table(store, path))
        validate_bars(table, item.frequency, session_id=item.session_id, raise_on_error=True)
        ts = table["ts"].to_numpy(zero_copy_only=False)

        # Two passes, because the Parquet metadata carries the manifest hash and the manifest
        # carries the artifact hash. The identity hash is computed first (it excludes the
        # artifact hash for exactly this reason), then the bytes, then the artifact hash.
        manifest = FileManifest(
            schema_version=1,
            source=source.name,
            market=item.market,
            asset_class=item.asset_class,
            symbol=item.symbol,
            frequency=item.frequency,
            pull_run_id=pull_run_id,
            pulled_at=datetime.now(UTC).isoformat(),
            loader_version=loader_version(),
            source_urls=plan.source_urls,
            source_sha256s=plan.source_sha256s,
            artifact_path=path,
            row_count=table.num_rows,
            first_ts=int(ts[0]),
            last_ts=int(ts[-1]),
            gap_count=count_gaps(ts, item.frequency),
            off_grid_count=count_off_grid(ts, item.frequency),
            closed_window_count=count_closed_window(ts, item.session_id, item.frequency),
            median_dollar_volume=median_dollar_volume(table),
            source_symbol=item.vendor_symbol,
            universe_hash=universe_hash,
            **source.manifest_extras(item),
        )
        data = write_parquet(
            table,
            bars_metadata(
                source=source.name,
                asset_class=item.asset_class,
                market=item.market,
                symbol=item.symbol,
                frequency=item.frequency,
                manifest_sha256=manifest.manifest_sha256,
                exchange_tz=item.exchange_tz,
                session_id=item.session_id,
            ),
        )
        manifest = manifest.model_copy(update={"artifact_sha256": sha256_bytes(data)})

        store.put(path, data, manifest)
        log.info(
            "ok %s: %d rows, %d gaps, %d off-grid, %d bytes",
            item,
            table.num_rows,
            manifest.gap_count,
            manifest.off_grid_count,
            len(data),
        )
        return PullResult(item, "ok", manifest=manifest, bytes_written=len(data))

    except Exception as exc:  # one bad item must not end a fifteen-thousand-item run
        log.warning("failed %s: %s: %s", item, type(exc).__name__, exc)
        return PullResult(item, "failed", error=f"{type(exc).__name__}: {exc}")


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
    source: Source,
    store: RawStore,
    items: list[WorkItem],
    manifest: PullRunManifest,
    *,
    force: bool = False,
) -> PullRun:
    """Walk the work list, one item at a time, flushing the store at the end.

    Sequential over items on purpose. The parallelism that matters lives one level down, inside
    a source's own transport, where a symbol's fifty monthly archives are fetched at once through
    a shared pool; adding a second layer here would multiply the two caps together and stop being
    polite.
    """
    run = PullRun(manifest)
    kill_after = _kill_after_items()
    for index, item in enumerate(items, start=1):
        log.info("[%d/%d] %s", index, len(items), item)
        run.record(
            pull_item(
                source,
                store,
                item,
                pull_run_id=manifest.pull_run_id,
                universe_hash=manifest.universe_hash,
                force=force,
            )
        )
        if kill_after is not None and index >= kill_after:
            # `os._exit`, not an exception: a real session death does not unwind, does not flush
            # the store's pending batch, and does not write a run manifest. Simulating it with
            # anything gentler would test a kinder failure than the one that actually happens.
            log.warning("AXIOM_KILL_AFTER_ITEMS=%d reached; dying without flushing", kill_after)
            os._exit(137)

    # The final commit is as fallible as every other one -- it is the same Hub call the loop has
    # been making all along -- and letting it escape would throw away the run manifest that
    # records what just happened. A pull that landed twelve thousand series and could not commit
    # its last batch has still landed twelve thousand series.
    try:
        store.flush()
    except Exception as exc:
        log.error("final flush failed: %s: %s", type(exc).__name__, exc)
        run.manifest.flush_error = f"{type(exc).__name__}: {exc}"
    return run

"""The corpus clean run: fan out over every bar artifact, reduce to one segment index.

The driver is here and the compute is in :mod:`axiom.clean.engine`, so the same code path runs
under Modal's `.map()`, under a Kaggle kernel, and over a local raw tier in a test. What changes
is who calls :func:`clean_artifact` and how many at once.

Two guards make a rerun trustworthy rather than merely repeatable:

- **Staleness.** A segment is bound to the `sha256` of the raw file it came from. A raw file that
  changed since the last clean invalidates its segments, and `--incremental` re-cleans exactly
  those. Adjusted equity history really does get restated by a later corporate action (ADR-0019),
  so this is a normal event, not a corruption alarm.
- **Config identity.** `--incremental` is refused across a config-hash change. Half a corpus
  cleaned at one threshold and half at another is not a corpus, and the failure would be
  invisible in the output.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from axiom.clean.config import CleanConfig
from axiom.clean.engine import (
    SeriesIdentity,
    clean_series,
    dropstats_table,
    segments_table,
    verify_corpus_invariants,
)

log = logging.getLogger("axiom.clean")

#: Registry rows that are not bars and must not be cleaned. Yahoo's event series is not a time
#: grid, and the derived tier is an output of cleaning's sibling rather than an input to it.
NON_BAR_SOURCES = frozenset({"yahoo"})
NON_BAR_PREFIXES = ("derived/", "clean/", "registry/", "staging/")

#: Fallback session per (source, asset_class), for a file whose Parquet metadata predates the
#: `session_id` key. ADR-0014 fixes these; the file's own metadata still wins when it has any.
FALLBACK_SESSIONS: dict[tuple[str, str], str] = {
    ("binance", "crypto"): "24x7",
    ("dukascopy", "fx"): "24x5",
    ("dukascopy", "commodity"): "24x5",
    ("dukascopy", "index"): "24x5",
    ("stooq", "equity"): "XNYS-regular",
}


def clean_paths(clean_version: int) -> dict[str, str]:
    root = f"clean/v{clean_version}"
    return {
        "root": root,
        "segments": f"{root}/segments.parquet",
        "dropstats": f"{root}/dropstats.parquet",
        "manifest": f"{root}/run_manifest.json",
    }


@dataclass
class ArtifactRef:
    """One bar file the run must clean, as the registry describes it."""

    artifact_path: str
    source: str
    market: str
    asset_class: str
    symbol: str
    frequency: str
    artifact_sha256: str


@dataclass
class CleanRun:
    """What one corpus clean produced."""

    clean_version: int
    clean_config_hash: str
    registry_hash: str = ""
    incremental: bool = False
    segments: list[dict[str, Any]] = field(default_factory=list)
    dropstats: list[dict[str, Any]] = field(default_factory=list)
    ok: int = 0
    failed: int = 0
    failures: list[dict[str, str]] = field(default_factory=list)
    total_bars: int = 0
    kept_bars: int = 0
    reused_artifacts: int = 0
    wall_seconds: float = 0.0

    @property
    def dropped_bars(self) -> int:
        return self.total_bars - self.kept_bars

    def line(self) -> str:
        pct = 100.0 * self.dropped_bars / self.total_bars if self.total_bars else 0.0
        return (
            f"clean v{self.clean_version} ({self.clean_config_hash}): {self.ok} series, "
            f"{len(self.segments)} segments, {self.kept_bars}/{self.total_bars} bars kept "
            f"({pct:.2f}% dropped), {self.failed} failed"
        )

    def to_json(self) -> str:
        return (
            json.dumps(
                {
                    "clean_version": self.clean_version,
                    "clean_config_hash": self.clean_config_hash,
                    "registry_hash": self.registry_hash,
                    "incremental": self.incremental,
                    "series_ok": self.ok,
                    "series_failed": self.failed,
                    "series_reused": self.reused_artifacts,
                    "segments": len(self.segments),
                    "total_bars": self.total_bars,
                    "kept_bars": self.kept_bars,
                    "dropped_bars": self.dropped_bars,
                    "coverage": _coverage(self.dropstats),
                    "wall_seconds": round(self.wall_seconds, 1),
                    "failures": self.failures,
                },
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )


def _coverage(dropstats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per source and frequency: series, bars in, bars kept. One row per slice, sorted."""
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    seen: set[tuple[str, str, str]] = set()
    for row in dropstats:
        key = (row["source"], row["frequency"])
        group = groups.setdefault(
            key, {"source": key[0], "frequency": key[1], "series": 0, "bars": 0, "kept": 0}
        )
        marker = (row["source"], row["frequency"], row["artifact_path"])
        if marker in seen:
            continue
        seen.add(marker)
        group["series"] += 1
        group["bars"] += row["total_bars"]
        group["kept"] += row["kept_bars"]
    return [groups[k] for k in sorted(groups)]


def bar_artifacts(registry: pa.Table) -> list[ArtifactRef]:
    """Every registry row that is a bar series, in a deterministic order.

    The filter is by source *and* by path prefix. Either alone would let something through: a new
    non-bar source would slip past a prefix check, and a bar source writing into `derived/` would
    slip past a source check.
    """
    rows = registry.select(
        [
            "artifact_path",
            "source",
            "market",
            "asset_class",
            "symbol",
            "frequency",
            "artifact_sha256",
        ]
    ).to_pylist()
    refs = [
        ArtifactRef(**row)
        for row in rows
        if row["source"] not in NON_BAR_SOURCES
        and not row["artifact_path"].startswith(NON_BAR_PREFIXES)
    ]
    return sorted(refs, key=lambda r: r.artifact_path)


def session_id_for(table: pa.Table, ref: ArtifactRef) -> str:
    """The file's declared session, or the ADR-0014 default for its source.

    Read from the file rather than inferred, because the file is what a reader will have. The
    fallback exists for artifacts written before the metadata key did, and it raises rather than
    guessing `24x7` -- defaulting an equity series to a crypto calendar would partition it into
    single days and look like a data problem.
    """
    metadata = table.schema.metadata or {}
    declared = metadata.get(b"session_id")
    if declared:
        return declared.decode("utf-8")
    try:
        return FALLBACK_SESSIONS[(ref.source, ref.asset_class)]
    except KeyError:
        raise ValueError(
            f"{ref.artifact_path}: no session_id in the file metadata and no fallback for "
            f"({ref.source}, {ref.asset_class}); add one to FALLBACK_SESSIONS or re-pull"
        ) from None


def clean_artifact(data: bytes, ref: ArtifactRef, config: CleanConfig):
    """Clean one artifact's bytes. The unit a `.map()` fans out over."""
    table = pq.read_table(io.BytesIO(data))
    identity = SeriesIdentity(
        source=ref.source,
        market=ref.market,
        asset_class=ref.asset_class,
        symbol=ref.symbol,
        frequency=ref.frequency,
        session_id=config.session_id_for(ref.source, ref.asset_class, session_id_for(table, ref)),
        artifact_path=ref.artifact_path,
        raw_artifact_sha256=ref.artifact_sha256,
    )
    return clean_series(table, identity, config)


# --- staleness --------------------------------------------------------------------------


def stale_artifacts(refs: list[ArtifactRef], existing: pa.Table | None) -> set[str]:
    """Artifacts whose segments are missing or were derived from different bytes.

    An artifact absent from the existing index is stale by definition -- it has never been
    cleaned. One present under a different `raw_artifact_sha256` is stale because the file moved
    under it.
    """
    if existing is None or existing.num_rows == 0:
        return {r.artifact_path for r in refs}
    known: dict[str, set[str]] = {}
    for path, digest in zip(
        existing["artifact_path"].to_pylist(),
        existing["raw_artifact_sha256"].to_pylist(),
        strict=True,
    ):
        known.setdefault(path, set()).add(digest)
    return {
        r.artifact_path for r in refs if r.artifact_sha256 not in known.get(r.artifact_path, ())
    }


class ConfigHashChanged(RuntimeError):
    """Raised when `--incremental` is asked for across a config change."""


def check_incremental_allowed(existing: pa.Table | None, config: CleanConfig) -> None:
    """Refuse an incremental run whose config differs from the one already in the index."""
    if existing is None or existing.num_rows == 0:
        return
    hashes = set(existing["clean_config_hash"].to_pylist())
    if hashes and hashes != {config.config_hash}:
        raise ConfigHashChanged(
            f"the existing segment index was built at config hash(es) {sorted(hashes)} and this "
            f"config hashes to {config.config_hash}. Segments are never trusted across a config "
            "change -- rerun in full, without --incremental."
        )


# --- the driver -------------------------------------------------------------------------


def clean_corpus(
    refs: list[ArtifactRef],
    read: Any,
    config: CleanConfig,
    *,
    existing: pa.Table | None = None,
    existing_dropstats: pa.Table | None = None,
    incremental: bool = False,
    registry_hash: str = "",
    concurrency: int = 1,
    now: Any = None,
) -> CleanRun:
    """Clean every artifact in ``refs`` and reduce the results into one run.

    ``read`` maps an :class:`ArtifactRef` to its bytes; injecting it is what lets this run against
    the Hub, a local directory, or a Modal volume without knowing which. ``now`` is a clock,
    injected so a test can assert on the manifest without the wall time in it.

    ``concurrency`` threads the read-and-clean loop. Fourteen thousand artifacts against the Hub
    is fourteen thousand HTTPS round trips where the cost is latency, the same shape the registry
    build has; numpy drops the GIL for the vectorized part. Results are consumed in input order,
    so the output does not depend on which worker finished first -- and the tables are sorted on
    the way out regardless.

    An incremental run carries forward both the segments *and* the drop stats of the artifacts it
    skipped. Carrying only the segments would leave the dropstats table describing whichever
    subset happened to be stale, which is a worse lie than having no dropstats at all.
    """
    if incremental:
        check_incremental_allowed(existing, config)

    run = CleanRun(
        clean_version=config.clean_version,
        clean_config_hash=config.config_hash,
        registry_hash=registry_hash,
        incremental=incremental,
    )

    todo = refs
    carried: list[dict[str, Any]] = []
    carried_drops: list[dict[str, Any]] = []
    if incremental and existing is not None and existing.num_rows:
        stale = stale_artifacts(refs, existing)
        todo = [r for r in refs if r.artifact_path in stale]
        fresh = {r.artifact_path for r in refs} - stale
        carried = [row for row in existing.to_pylist() if row["artifact_path"] in fresh]
        if existing_dropstats is not None:
            carried_drops = [
                row for row in existing_dropstats.to_pylist() if row["artifact_path"] in fresh
            ]
        run.reused_artifacts = len(fresh)
        log.info("incremental: %d stale, %d reused", len(todo), len(fresh))

    def clean_one(ref: ArtifactRef):
        """Returns the result, or the exception, so one bad file cannot end the run."""
        try:
            data = read(ref)
            if data is None:
                raise FileNotFoundError(ref.artifact_path)
            return ref, clean_artifact(data, ref, config), None
        except Exception as exc:
            return ref, None, exc

    start = now() if now else _monotonic()
    if concurrency > 1 and len(todo) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="clean") as pool:
            outcomes = list(pool.map(clean_one, todo))
    else:
        outcomes = [clean_one(ref) for ref in todo]

    for ref, result, error in outcomes:
        if error is not None:
            log.warning("clean failed for %s: %s", ref.artifact_path, error)
            run.failed += 1
            run.failures.append(
                {"artifact_path": ref.artifact_path, "error": f"{type(error).__name__}: {error}"}
            )
            continue
        run.ok += 1
        run.segments.extend(result.segments)
        run.dropstats.extend(result.dropstats)
        run.total_bars += result.total_bars
        run.kept_bars += result.kept_bars

    run.segments.extend(carried)
    run.dropstats.extend(carried_drops)
    run.wall_seconds = (now() if now else _monotonic()) - start
    return run


def _monotonic() -> float:
    import time

    return time.monotonic()


def write_outputs(run: CleanRun) -> dict[str, bytes]:
    """Serialize a run into the three files that land in `axiom-raw`.

    The corpus-wide invariants are checked here rather than at the caller, so no path can upload
    a segment index that overlaps itself.
    """
    segments = segments_table(run.segments)
    problems = verify_corpus_invariants(segments)
    if problems:
        raise ValueError("the segment index breaks its own invariants: " + "; ".join(problems))

    paths = clean_paths(run.clean_version)
    return {
        paths["segments"]: _parquet(segments),
        paths["dropstats"]: _parquet(dropstats_table(run.dropstats)),
        paths["manifest"]: run.to_json().encode("utf-8"),
    }


def _parquet(table: pa.Table) -> bytes:
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression="zstd")
    return buffer.getvalue()

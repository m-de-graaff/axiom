"""Build the total-return tier over the equities universe (ADR-0019).

What gets written depends on the recorded verdict, and that is the whole design:

- **Identity verdict** (`split_and_dividend_adjusted`, `none`) -- `tr_close` equals `close`, so
  materializing it would write twelve thousand byte-for-byte copies of a column that is already
  in the file next to it. Only `derived/tr_close/manifest.json` is written: the verdict, the
  coverage, and `tr_available` per ticker. Readers get `tr_close` from
  :func:`axiom.adjust.policy.tr_close` at read time, uniformly for every source.
- **Accumulation verdict** (`split_adjusted`) -- `tr_close` genuinely differs from `close`, and
  recomputing it per read would be both slow and a reproducibility hazard. The letter-sharded
  Parquet tier is written.

The interface downstream sees is the same either way. Which one ran is recorded in the manifest,
so a reader that finds no bar files knows why rather than guessing.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import asdict, dataclass, field

import pyarrow.parquet as pq

from axiom.adjust.policy import needs_events, tr_available, tr_close
from axiom.provenance.manifest import FileManifest, sha256_bytes
from axiom.raw.store import RawStore
from axiom.schema.bars import ROW_GROUP_SIZE
from axiom.sources.base import shard_dir
from axiom.sources.yahoo_events import SOURCE as EVENTS_SOURCE
from axiom.sources.yahoo_events import artifact_path as events_path

log = logging.getLogger("axiom.adjust")

TR_ROOT = "derived/tr_close"
TR_MANIFEST_PATH = f"{TR_ROOT}/manifest.json"

#: Which raw tier the total-return series is derived from. Equities are the only asset class in
#: the corpus with corporate actions; everything else is `tr_close == close` by definition and
#: does not appear here at all.
TR_SOURCE = "stooq"
TR_MARKET = "us"
TR_FREQUENCY = "1d"


def tr_artifact_path(symbol: str) -> str:
    return f"{TR_ROOT}/{TR_MARKET}/{TR_FREQUENCY}/{shard_dir(symbol)}/{symbol}.parquet"


@dataclass
class TickerCoverage:
    """One ticker's place in the derived tier."""

    symbol: str
    tr_available: bool
    materialized: bool
    n_bars: int
    dividend_events: int
    derived_from_sha256: str
    events_sha256: str = ""
    error: str = ""


@dataclass
class DeriveRun:
    """What one `axiom derive tr` invocation did."""

    verdict: str
    policy_by_source: dict[str, str]
    clean_config_hash: str = ""
    materialized: bool = False
    tickers: list[TickerCoverage] = field(default_factory=list)

    @property
    def available(self) -> int:
        return sum(1 for t in self.tickers if t.tr_available)

    @property
    def failed(self) -> list[TickerCoverage]:
        return [t for t in self.tickers if t.error]

    def coverage_pct(self) -> float:
        return 100.0 * self.available / len(self.tickers) if self.tickers else 0.0

    def to_json(self) -> str:
        payload = {
            "verdict": self.verdict,
            "policy_by_source": self.policy_by_source,
            "materialized": self.materialized,
            "tickers_total": len(self.tickers),
            "tr_available": self.available,
            "coverage_pct": round(self.coverage_pct(), 3),
            "failed": len(self.failed),
            "tickers": [asdict(t) for t in sorted(self.tickers, key=lambda t: t.symbol)],
        }
        return json.dumps(payload, sort_keys=True, indent=2) + "\n"

    def line(self) -> str:
        mode = "materialized" if self.materialized else "identity (manifest only)"
        return (
            f"verdict={self.verdict} {mode}: {self.available}/{len(self.tickers)} tickers "
            f"tr_available ({self.coverage_pct():.1f}%), {len(self.failed)} failed"
        )


def _read_table(store: RawStore, path: str):
    data = store.get(path)
    return None if data is None else pq.read_table(io.BytesIO(data))


def derive_tr(
    store: RawStore,
    manifests: list[FileManifest],
    *,
    verdict: str,
    limit: int | None = None,
) -> DeriveRun:
    """Build the total-return tier from the raw bars and the captured events.

    ``verdict`` comes from the recorded audit -- `adjustment_policy` on the Stooq manifests -- and
    is passed in rather than re-derived, because re-deriving it here would mean two places that
    disagree about what the vendor did.
    """
    materialize = needs_events(verdict)
    bars = sorted(
        (
            m
            for m in manifests
            if m.source == TR_SOURCE and m.market == TR_MARKET and m.frequency == TR_FREQUENCY
        ),
        key=lambda m: m.symbol,
    )
    if limit is not None:
        bars = bars[:limit]

    events_by_symbol = {
        m.symbol: m for m in manifests if m.source == EVENTS_SOURCE and m.frequency == "events"
    }
    run = DeriveRun(
        verdict=verdict,
        policy_by_source={TR_SOURCE: verdict, "binance": "none", "dukascopy": "none"},
        materialized=materialize,
    )

    for manifest in bars:
        symbol = manifest.symbol
        events_manifest = events_by_symbol.get(symbol)
        events = None
        if events_manifest is not None:
            events = _read_table(store, events_path(symbol))
        dividends = 0
        if events is not None and events.num_rows:
            dividends = sum(1 for t in events["event_type"].to_pylist() if t == "dividend")

        coverage = TickerCoverage(
            symbol=symbol,
            tr_available=tr_available(events, verdict),
            materialized=False,
            n_bars=manifest.row_count,
            dividend_events=dividends,
            derived_from_sha256=manifest.artifact_sha256,
            events_sha256=events_manifest.artifact_sha256 if events_manifest else "",
        )

        if materialize and coverage.tr_available:
            try:
                table = _read_table(store, manifest.artifact_path)
                if table is None:
                    raise FileNotFoundError(manifest.artifact_path)
                tr = tr_close(table, events, verdict)
                buffer = io.BytesIO()
                pq.write_table(tr, buffer, compression="zstd", row_group_size=ROW_GROUP_SIZE)
                payload = buffer.getvalue()
                store.put(
                    tr_artifact_path(symbol),
                    payload,
                    _derived_manifest(manifest, payload, verdict),
                )
                coverage.materialized = True
            except Exception as exc:  # one bad ticker must not lose the other twelve thousand
                log.warning("tr derive failed for %s: %s", symbol, exc)
                coverage.error = f"{type(exc).__name__}: {exc}"
                coverage.tr_available = False

        run.tickers.append(coverage)

    if materialize:
        store.flush()
    return run


def _derived_manifest(source: FileManifest, payload: bytes, verdict: str) -> FileManifest:
    """A sidecar for a derived file: what it came from, and under which policy.

    ``source_urls`` names the raw artifact rather than a vendor URL, because that *is* where these
    bytes came from -- the derived tier's provenance chain ends at the raw tier, which has its own.
    """
    return FileManifest(
        schema_version=source.schema_version,
        source="derived",
        market=TR_MARKET,
        asset_class=source.asset_class,
        symbol=source.symbol,
        frequency=TR_FREQUENCY,
        pull_run_id=source.pull_run_id,
        pulled_at=source.pulled_at,
        loader_version=source.loader_version,
        source_urls=[source.artifact_path],
        source_sha256s=[source.artifact_sha256],
        artifact_path=tr_artifact_path(source.symbol),
        artifact_sha256=sha256_bytes(payload),
        row_count=source.row_count,
        first_ts=source.first_ts,
        last_ts=source.last_ts,
        gap_count=source.gap_count,
        adjustment_policy=verdict,
        universe_hash=source.universe_hash,
    )

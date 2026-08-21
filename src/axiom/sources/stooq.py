"""US daily equities from the Stooq bulk archive (ADR-0016).

Every other source in the corpus is a query interface. This one is a single zip holding the whole
US market -- twelve to eighteen thousand tickers, one text file each -- which inverts the usual
shape: enumeration is reading a table of contents rather than listing a bucket, and every series
in the run shares one provenance record, because they all came out of one download.

That sharing is what makes the idempotence work without any new machinery. Each series' plan is
the archive's URL and its sha256, so a re-run against the same archive skips everything already
landed, and a newer archive changes the digest for every ticker at once -- which is correct, since
a new dump is a new copy of all of them.

The archive itself never touches the laptop. A human solves the CAPTCHA and hands over a URL; the
cloud job downloads it. The one sanctioned exception, and its accounting, is in ADR-0016.
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from axiom.schema.bars import BARS_SCHEMA_V1
from axiom.sources.base import SourcePlan, WorkItem, shard_dir

log = logging.getLogger("axiom.stooq")

SOURCE = "stooq"
MARKET = "us"
ASSET_CLASS = "equity"

#: Series shorter than this are recorded as skipped rather than landed. A ticker that listed last
#: month is a fact about the market, and a twelve-bar series is not usable by anything downstream.
MIN_ROWS = 30

#: Share of a file's data lines that may be malformed before the file is failed outright. Vendor
#: text dumps carry occasional damage, and failing a nine-thousand-row series over one truncated
#: line would cost more than it protects.
MAX_MALFORMED_FRACTION = 0.001

#: Directory names under `data/daily/us/` that hold tradeable instruments. Matched as substrings
#: of the containing directory, because Stooq numbers its subdirectories (`nasdaq stocks/1/`) and
#: has renumbered them before.
KEPT_DIRECTORY_MARKERS = ("stocks", "etfs")

#: Explicitly not kept. An index is not a tradeable instrument with a volume, and futures open a
#: contract-rollover problem v0.2 does not (ADR-0016).
SKIPPED_DIRECTORY_MARKERS = ("indices", "futures", "currencies", "bonds", "commodities")

#: Stooq's own column order. The header line names them in angle brackets; the data lines do not.
COLUMNS = ("TICKER", "PER", "DATE", "TIME", "OPEN", "HIGH", "LOW", "CLOSE", "VOL", "OPENINT")


class MalformedFile(ValueError):
    """Too many unparseable lines in one ticker's file for the rest to be trusted."""


def is_kept_member(name: str) -> bool:
    """Whether one archive member is a US daily stock or ETF series.

    The test is on the *directory*, not the filename, because the filename is just a ticker and
    a ticker says nothing about which instrument class it belongs to.
    """
    lowered = name.lower().replace("\\", "/")
    if not lowered.endswith(".txt") or "/us/" not in lowered:
        return False
    directory = lowered.rsplit("/", 1)[0]
    if any(marker in directory for marker in SKIPPED_DIRECTORY_MARKERS):
        return False
    return any(marker in directory for marker in KEPT_DIRECTORY_MARKERS)


def symbol_from_member(name: str) -> str:
    """`.../nasdaq stocks/1/aapl.us.txt` -> `AAPL`.

    The `.us` suffix is Stooq's market tag rather than part of the ticker, so it is stripped into
    `symbol` and kept whole in `source_symbol` -- which is what makes the vendor's name for the
    series recoverable from the manifest alone.
    """
    stem = name.replace("\\", "/").rsplit("/", 1)[-1]
    stem = stem[: -len(".txt")] if stem.lower().endswith(".txt") else stem
    return stem[: -len(".us")].upper() if stem.lower().endswith(".us") else stem.upper()


def source_symbol_from_member(name: str) -> str:
    """The vendor's spelling, suffix included."""
    stem = name.replace("\\", "/").rsplit("/", 1)[-1]
    return stem[: -len(".txt")] if stem.lower().endswith(".txt") else stem


def date_to_ms(yyyymmdd: str) -> int:
    """`20240610` -> 00:00:00 UTC of that calendar date, in milliseconds (ADR-0014)."""
    stamp = datetime(int(yyyymmdd[0:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8]), tzinfo=UTC)
    return int(stamp.timestamp() * 1000)


@dataclass
class ParseCounts:
    """What one file's parse ran into. Carried into the run's log rather than swallowed."""

    lines: int = 0
    malformed: int = 0

    @property
    def malformed_fraction(self) -> float:
        return self.malformed / self.lines if self.lines else 0.0


def parse_ticker_file(text: str, *, context: str = "stooq") -> tuple[pa.Table, ParseCounts]:
    """Parse one ticker's daily text file into schema v1.

    Rows that cannot be read at all are counted and dropped, up to
    :data:`MAX_MALFORMED_FRACTION` of the file. Above that the file is failed: past some density
    of damage, what survived is not a series, it is a sample of one.

    Duplicate dates are *not* handled here. They are left in, sorted, and caught by
    `validate_bars` as a non-increasing timestamp -- a hard failure with no tolerance, because
    every other defect in this file is absence of information and a duplicate date is a
    contradiction (ADR-0016).
    """
    counts = ParseCounts()
    ts: list[int] = []
    ohlc: list[tuple[float, float, float, float]] = []
    volume: list[float] = []

    for row in csv.reader(io.StringIO(text)):
        if not row or row[0].startswith("<"):  # the header line, named in angle brackets
            continue
        counts.lines += 1
        try:
            if row[1].strip().upper() != "D":
                raise ValueError(f"period {row[1]!r} is not daily")
            ts.append(date_to_ms(row[2].strip()))
            ohlc.append((float(row[4]), float(row[5]), float(row[6]), float(row[7])))
            volume.append(float(row[8]))
        except (IndexError, ValueError):
            counts.malformed += 1

    if counts.malformed_fraction > MAX_MALFORMED_FRACTION:
        raise MalformedFile(
            f"{context}: {counts.malformed}/{counts.lines} lines unparseable "
            f"({counts.malformed_fraction:.2%}), over the {MAX_MALFORMED_FRACTION:.1%} tolerance"
        )

    order = np.argsort(np.asarray(ts, dtype=np.int64), kind="stable")
    ts_sorted = np.asarray(ts, dtype=np.int64)[order]
    prices = np.asarray(ohlc, dtype=np.float64).reshape(-1, 4)[order] if ohlc else np.zeros((0, 4))
    vol = np.asarray(volume, dtype=np.float64)[order]

    open_, high, low, close = (prices[:, i] for i in range(4))
    n = len(ts_sorted)
    table = pa.table(
        {
            "ts": pa.array(ts_sorted, pa.int64()),
            "open": pa.array(open_, pa.float64()),
            "high": pa.array(high, pa.float64()),
            "low": pa.array(low, pa.float64()),
            "close": pa.array(close, pa.float64()),
            "volume": pa.array(vol, pa.float64()),
            # No native quote volume, so `amount` is synthesized and flagged (ADR-0010/0014).
            "amount": pa.array(vol * ((open_ + high + low + close) / 4.0), pa.float64()),
            "n_trades": pa.array([None] * n, pa.int64()),
            "taker_buy_volume": pa.array([None] * n, pa.float64()),
            "taker_buy_quote_volume": pa.array([None] * n, pa.float64()),
        },
        schema=BARS_SCHEMA_V1,
    )
    return table, counts


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Digest a file without holding it in memory. The archive is measured in gigabytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def download_archive(url: str, dest: Path, *, client: Any = None) -> Path:
    """Stream the archive to ``dest``, cloud-side.

    Streaming rather than reading into memory: the dump is a multi-gigabyte zip and a runner has
    single-digit gigabytes of RAM. Stooq ships no checksum of its own, so the digest recorded in
    every manifest is self-computed -- weaker than Binance's vendor-published one, and named as
    such in `docs/DATA_LICENSING.md`.
    """
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    owned = client is None
    client = client or httpx.Client(timeout=httpx.Timeout(60.0, read=300.0), follow_redirects=True)
    try:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with dest.open("wb") as handle:
                for block in response.iter_bytes(1 << 20):
                    handle.write(block)
    finally:
        if owned:
            client.close()
    return dest


@dataclass
class StooqArchive:
    """One downloaded bulk dump: where it came from, what it hashes to, and where it sits now."""

    url: str
    path: Path
    sha256: str = ""
    #: True when the archive reached the cloud via the laptop, under ADR-0016's one exception.
    staging_exception_used: bool = False

    def __post_init__(self) -> None:
        if not self.sha256:
            self.sha256 = sha256_file(self.path)


class StooqSource:
    """The bulk archive as the driver sees it."""

    name = SOURCE

    def __init__(self, archive: StooqArchive) -> None:
        self.archive = archive
        self._zip = zipfile.ZipFile(archive.path)
        #: Tickers found but not landed because they are too short to be useful. Reported rather
        #: than silently absent -- a corpus that quietly drops series is a corpus nobody can audit.
        self.skipped_short: list[str] = []
        self._members: dict[str, str] = {}

    def close(self) -> None:
        self._zip.close()

    def __enter__(self) -> StooqSource:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- work list -------------------------------------------------------------------

    def work_items(
        self, *, symbols: list[str] | None = None, limit: int | None = None
    ) -> list[WorkItem]:
        """One item per kept ticker, sorted, with the too-short ones filtered out and recorded.

        Length is judged from the member's line count rather than by parsing, so enumerating
        eighteen thousand tickers costs a walk of the zip's central directory plus one cheap read
        each, not a full parse of the corpus before the corpus is written.
        """
        wanted = {s.upper() for s in symbols} if symbols else None
        items: list[WorkItem] = []

        for info in sorted(self._zip.infolist(), key=lambda i: i.filename):
            if info.is_dir() or not is_kept_member(info.filename):
                continue
            symbol = symbol_from_member(info.filename)
            if wanted is not None and symbol not in wanted:
                continue
            if self._line_count(info) - 1 < MIN_ROWS:  # less the header
                self.skipped_short.append(symbol)
                continue

            self._members[symbol] = info.filename
            items.append(
                WorkItem(
                    market=MARKET,
                    symbol=symbol,
                    frequency="1d",
                    asset_class=ASSET_CLASS,
                    source_symbol=source_symbol_from_member(info.filename),
                    exchange_tz="America/New_York",
                    session_id="XNYS-regular",
                )
            )
            if limit is not None and len(items) >= limit:
                break

        log.info("%d series to pull, %d skipped as too short", len(items), len(self.skipped_short))
        return items

    def _line_count(self, info: zipfile.ZipInfo) -> int:
        with self._zip.open(info) as handle:
            return sum(block.count(b"\n") for block in iter(lambda: handle.read(1 << 16), b""))

    # --- the Source protocol ---------------------------------------------------------

    def artifact_path(self, item: WorkItem) -> str:
        return f"raw/{SOURCE}/{MARKET}/1d/{shard_dir(item.symbol)}/{item.symbol}.parquet"

    def plan(self, item: WorkItem) -> SourcePlan:
        """Every series in the run shares the archive's provenance, because they all came from it.

        A newer dump changes this digest for every ticker at once, which is exactly right: it is
        a new copy of all of them, not an extension of any one.
        """
        return SourcePlan([self.archive.url], [self.archive.sha256])

    def build(
        self,
        item: WorkItem,
        plan: SourcePlan,
        load_existing: Callable[[], pa.Table | None],
    ) -> pa.Table:
        """Read one member out of the archive and parse it. Nothing is spliced or extended."""
        member = self._members[item.symbol]
        with self._zip.open(member) as handle:
            text = handle.read().decode("utf-8", errors="replace")
        table, counts = parse_ticker_file(text, context=f"{item.symbol} ({member})")
        if counts.malformed:
            log.info(
                "%s: dropped %d malformed line(s) of %d",
                item.symbol,
                counts.malformed,
                counts.lines,
            )
        return table

    def manifest_extras(self, item: WorkItem) -> dict[str, Any]:
        """Trade prices and share volume; the adjustment verdict is not in yet (ADR-0016)."""
        return {
            "price_side": "trade",
            "volume_convention": "shares",
            "amount_synthesized": True,
            # The audit's finding, measured 2026-08-21 and recorded in
            # `docs/reports/v0.2-adjustment-audit.md`: three known splits show no discontinuity,
            # and 20 sampled tickers track Yahoo's dividend-adjusted closes to a median 1.29%.
            # This started as `vendor_adjusted_unverified` and stayed there until it was checked,
            # which is the only reason the value can be trusted now.
            "adjustment_policy": "split_and_dividend_adjusted",
            "redistribution_class": "loader_manifest_private_cache",
        }

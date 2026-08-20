"""Fetch layer for `data.binance.vision` (ADR-0012).

The bucket publishes one zip per (market, cadence, symbol, frequency, period), each with a
`.CHECKSUM` sibling. This module turns those conventions into URLs, enumerates what actually
exists from the S3 listing rather than from a date range, and downloads with the checksum
verified before anybody looks inside the archive.

Availability comes from the listing because listing and delisting leave real holes. Walking a
date range would turn every hole into a 404 that has to be interpreted, and "interpreted 404"
is how a corpus quietly loses six months of a symbol.
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from xml.etree import ElementTree

import httpx

log = logging.getLogger("axiom.binance")

#: Where the files are served from.
DOWNLOAD_BASE = "https://data.binance.vision"

#: Where the bucket is *listed* from. The same bucket, addressed as S3 so it answers with XML.
LISTING_BASE = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"

#: Path segment per market. `um` is USDT-margined perpetuals; COIN-M is out of scope for v0.1.
MARKET_PREFIX = {"spot": "data/spot", "um": "data/futures/um"}

MARKETS = tuple(MARKET_PREFIX)
CADENCES = ("monthly", "daily")

#: Politeness cap on simultaneous requests. Binance Vision publishes no rate limit, which is a
#: reason to pick a conservative number rather than a licence to pick none.
DEFAULT_CONCURRENCY = 12

#: Status codes worth trying again. Everything else is a fact about the request, not the weather.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class NotFound(Exception):
    """A 404. Expected for an unlisted daily probe, a hard error for a listed month."""


class ChecksumMismatch(Exception):
    """Downloaded bytes do not match the published sha256. Never extracted, always retried once."""


def _localname(tag: str) -> str:
    """Strip the XML namespace: ``{http://s3...}Key`` -> ``Key``."""
    return tag.rsplit("}", 1)[-1]


def market_prefix(market: str) -> str:
    try:
        return MARKET_PREFIX[market]
    except KeyError:
        raise ValueError(f"unknown market {market!r}; v0.1 carries {list(MARKET_PREFIX)}") from None


def klines_dir(market: str, cadence: str, symbol: str, frequency: str) -> str:
    """The bucket key prefix holding one symbol's zips at one cadence and frequency."""
    if cadence not in CADENCES:
        raise ValueError(f"unknown cadence {cadence!r}; expected one of {CADENCES}")
    return f"{market_prefix(market)}/{cadence}/klines/{symbol}/{frequency}/"


def zip_key(market: str, cadence: str, symbol: str, frequency: str, period: str) -> str:
    """Bucket key for one archive. ``period`` is ``YYYY-MM`` monthly, ``YYYY-MM-DD`` daily."""
    return f"{klines_dir(market, cadence, symbol, frequency)}{symbol}-{frequency}-{period}.zip"


def zip_url(market: str, cadence: str, symbol: str, frequency: str, period: str) -> str:
    return f"{DOWNLOAD_BASE}/{zip_key(market, cadence, symbol, frequency, period)}"


def checksum_url(url: str) -> str:
    return f"{url}.CHECKSUM"


def period_from_key(key: str) -> str:
    """``.../BTCUSDT-1h-2024-01.zip`` -> ``2024-01``, and the daily form -> ``2024-01-15``.

    The name is ``SYMBOL-FREQUENCY-PERIOD.zip`` and only the period contains hyphens of its own,
    so everything from the third field on is the period regardless of cadence.
    """
    name = key.rsplit("/", 1)[-1]
    if not name.endswith(".zip"):
        raise ValueError(f"not an archive key: {key!r}")
    return "-".join(name[: -len(".zip")].split("-")[2:])


def parse_checksum(text: str) -> str:
    """Pull the digest out of a ``.CHECKSUM`` file (``<sha256>  <filename>``)."""
    first = text.strip().split("\n", 1)[0]
    digest = first.split()[0].strip().lower()
    if len(digest) != 64 or not all(c in "0123456789abcdef" for c in digest):
        raise ValueError(f"not a sha256 digest: {first!r}")
    return digest


@dataclass(frozen=True)
class Archive:
    """One verified source zip: where it came from, its published digest, and its bytes."""

    url: str
    sha256: str
    data: bytes


class BinanceVision:
    """A polite, retrying, checksum-verifying client for the bucket.

    Holds one connection pool and one thread pool, both sized by ``concurrency``. The thread pool
    *is* the semaphore: nothing can be in flight that is not occupying one of its workers, so the
    cap holds however many callers fan out through it at once.
    """

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        concurrency: int = DEFAULT_CONCURRENCY,
        max_attempts: int = 5,
        backoff_base: float = 0.5,
        sleep=time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self.concurrency = concurrency
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self._sleep = sleep
        self._rng = rng or random.Random(0)
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(30.0, connect=10.0, read=60.0),
            limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency),
            follow_redirects=True,
            headers={"user-agent": "axiom-loader/0.1 (private research corpus)"},
        )
        self._pool = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="binance")

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        self._client.close()

    def __enter__(self) -> BinanceVision:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- raw transport ---------------------------------------------------------------

    def run_all(self, fn, items: Iterable) -> list:
        """Run ``fn`` over ``items`` on the shared pool, in order.

        Public because the universe builder fans out over symbols too, and every caller sharing
        one pool is what makes ``concurrency`` a global cap rather than a per-caller suggestion.
        """
        return list(self._pool.map(fn, list(items)))

    def get(self, url: str) -> bytes:
        """GET with exponential backoff and jitter. Raises :class:`NotFound` on a 404."""
        last: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                response = self._client.get(url)
            except httpx.HTTPError as exc:  # connection reset, timeout, DNS
                last = exc
            else:
                if response.status_code == 404:
                    raise NotFound(url)
                if response.status_code not in RETRYABLE_STATUS:
                    response.raise_for_status()
                    return response.content
                last = httpx.HTTPStatusError(
                    f"{response.status_code} for {url}", request=response.request, response=response
                )

            if attempt + 1 < self.max_attempts:
                delay = self.backoff_base * (2**attempt) * (1.0 + self._rng.random())
                log.debug("retry %d/%d in %.2fs: %s", attempt + 1, self.max_attempts, delay, url)
                self._sleep(delay)

        raise RuntimeError(f"giving up on {url} after {self.max_attempts} attempts") from last

    def get_text(self, url: str) -> str:
        return self.get(url).decode("utf-8")

    # --- listing ---------------------------------------------------------------------

    def _list(self, prefix: str, *, delimiter: str = "/") -> Iterator[tuple[str, str]]:
        """Yield ``(tag, value)`` for every key and common prefix under ``prefix``, paginated.

        ``tag`` is ``"Key"`` for a file and ``"Prefix"`` for a directory-like common prefix.

        Only ``<Contents>`` and ``<CommonPrefixes>`` children are read. A flat scan for any
        element named ``Prefix`` would also pick up the response's own echo of the request
        prefix, which would arrive looking like a symbol named "".
        """
        marker = ""
        while True:
            url = f"{LISTING_BASE}?delimiter={delimiter}&prefix={prefix}"
            if marker:
                url = f"{url}&marker={marker}"
            root = ElementTree.fromstring(self.get_text(url))

            fields = {_localname(child.tag): (child.text or "") for child in root}
            last = ""
            for child in root:
                name = _localname(child.tag)
                if name not in ("Contents", "CommonPrefixes"):
                    continue
                wanted = "Key" if name == "Contents" else "Prefix"
                value = next(
                    (g.text for g in child if _localname(g.tag) == wanted and g.text), None
                )
                if value:
                    yield wanted, value
                    last = value

            if fields.get("IsTruncated", "false").lower() != "true":
                return
            marker = fields.get("NextMarker") or last
            if not marker:
                return

    def list_symbols(self, market: str, *, cadence: str = "monthly") -> list[str]:
        """Every symbol the bucket has klines for, in listing order."""
        prefix = f"{market_prefix(market)}/{cadence}/klines/"
        return sorted(
            value[len(prefix) :].rstrip("/")
            for tag, value in self._list(prefix)
            if tag == "Prefix" and value.startswith(prefix)
        )

    def list_periods(self, market: str, cadence: str, symbol: str, frequency: str) -> list[str]:
        """Every period available for one series, sorted. Empty when the series does not exist.

        Sorted lexicographically, which for ``YYYY-MM`` and ``YYYY-MM-DD`` is chronological.
        """
        prefix = klines_dir(market, cadence, symbol, frequency)
        return sorted(
            period_from_key(value)
            for tag, value in self._list(prefix)
            if tag == "Key" and value.endswith(".zip")
        )

    # --- verified downloads ----------------------------------------------------------

    def fetch_checksum(self, url: str) -> str:
        return parse_checksum(self.get_text(checksum_url(url)))

    def fetch_checksums(self, urls: Iterable[str]) -> list[str]:
        """Published digests for many URLs at once, in the order given.

        Called before anything is downloaded: the digests alone decide whether a symbol is
        already current, and they are a few dozen bytes each against megabytes of archive.
        """
        return self.run_all(self.fetch_checksum, urls)

    def fetch_verified(self, url: str, expected: str | None = None) -> Archive:
        """Download and verify against the published digest, before anything is extracted.

        A mismatch is retried exactly once. Twice-corrupt bytes are a fact about the bucket or
        the network path, not a flake, and failing loudly is better than silently caching them
        into a corpus that will be trusted for months.
        """
        expected = expected or self.fetch_checksum(url)
        for attempt in (1, 2):
            data = self.get(url)
            actual = hashlib.sha256(data).hexdigest()
            if actual == expected:
                return Archive(url=url, sha256=expected, data=data)
            log.warning("checksum mismatch on %s (attempt %d)", url, attempt)
        raise ChecksumMismatch(f"{url}: published {expected}, downloaded {actual}")

    def fetch_all(self, urls: Iterable[str], digests: Iterable[str] | None = None) -> list[Archive]:
        """Verified downloads for many URLs at once, in the order given.

        Ordering matters downstream: the parser concatenates in period order and the manifest
        records source URLs in the same order, which is what makes the idempotence comparison a
        list equality rather than a set comparison.
        """
        wanted = list(urls)
        expected: list[str | None] = list(digests) if digests is not None else [None] * len(wanted)
        return list(self._pool.map(self.fetch_verified, wanted, expected))

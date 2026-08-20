"""Fetch layer: URLs, enumeration, retries, checksum verification, 404 semantics.

Everything runs against `tests.fakes.FakeBucket` over an httpx mock transport. No socket is
opened and no market-data byte exists anywhere in this file.
"""

from __future__ import annotations

import hashlib
import random
import threading

import pytest

from axiom.sources.binance_vision import (
    BinanceVision,
    ChecksumMismatch,
    NotFound,
    checksum_url,
    klines_dir,
    parse_checksum,
    period_from_key,
    zip_key,
    zip_url,
)
from tests.fakes import FakeBucket, kline_zip


@pytest.fixture
def bucket() -> FakeBucket:
    bucket = FakeBucket()
    bucket.put_month("spot", "BTCUSDT", "1h", "2024-01", kline_zip(744))
    bucket.put_month("spot", "BTCUSDT", "1h", "2024-02", kline_zip(696))
    bucket.put_month("spot", "ETHUSDT", "1h", "2024-01", kline_zip(744))
    bucket.put_month("um", "BTCUSDT", "1h", "2024-01", kline_zip(744))
    bucket.put_day("spot", "BTCUSDT", "1h", "2024-03-01", kline_zip(24))
    return bucket


@pytest.fixture
def client(bucket):
    with BinanceVision(
        client=bucket.client(),
        concurrency=4,
        backoff_base=0.0,
        sleep=lambda _: None,
        rng=random.Random(1),
    ) as instance:
        yield instance


# --- URL construction --------------------------------------------------------------------


def test_spot_and_um_urls_match_the_published_scheme():
    assert zip_url("spot", "monthly", "BTCUSDT", "1h", "2024-01") == (
        "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2024-01.zip"
    )
    assert zip_url("um", "daily", "BTCUSDT", "1d", "2024-01-15") == (
        "https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/1d/"
        "BTCUSDT-1d-2024-01-15.zip"
    )
    assert checksum_url(zip_url("spot", "monthly", "BTCUSDT", "1h", "2024-01")).endswith(
        ".zip.CHECKSUM"
    )


def test_unknown_market_and_cadence_are_refused():
    with pytest.raises(ValueError, match="unknown market"):
        zip_key("cm", "monthly", "BTCUSD_PERP", "1h", "2024-01")
    with pytest.raises(ValueError, match="unknown cadence"):
        klines_dir("spot", "hourly", "BTCUSDT", "1h")


@pytest.mark.parametrize(
    ("key", "period"),
    [
        ("data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2024-01.zip", "2024-01"),
        ("data/spot/daily/klines/BTCUSDT/1h/BTCUSDT-1h-2024-01-15.zip", "2024-01-15"),
        ("BTCUSDT-1d-2019-12.zip", "2019-12"),
    ],
)
def test_period_is_recovered_from_the_key(key, period):
    assert period_from_key(key) == period


def test_checksum_parsing():
    digest = "a" * 64
    assert parse_checksum(f"{digest}  BTCUSDT-1h-2024-01.zip\n") == digest
    assert parse_checksum(f"{digest.upper()}  x.zip") == digest
    with pytest.raises(ValueError, match="not a sha256"):
        parse_checksum("not-a-digest  x.zip")


# --- enumeration -------------------------------------------------------------------------


def test_symbols_come_from_the_listing(client):
    assert client.list_symbols("spot") == ["BTCUSDT", "ETHUSDT"]
    assert client.list_symbols("um") == ["BTCUSDT"]


def test_the_echoed_request_prefix_is_not_mistaken_for_a_symbol(client):
    # The S3 response repeats the request prefix in a <Prefix> element of its own. Reading it as
    # a common prefix invents a symbol named "".
    assert "" not in client.list_symbols("spot")


def test_periods_are_listed_per_cadence_and_sorted(client):
    assert client.list_periods("spot", "monthly", "BTCUSDT", "1h") == ["2024-01", "2024-02"]
    assert client.list_periods("spot", "daily", "BTCUSDT", "1h") == ["2024-03-01"]


def test_a_series_that_does_not_exist_lists_empty_rather_than_raising(client):
    assert client.list_periods("spot", "monthly", "NOPEUSDT", "1h") == []


def test_listing_paginates(bucket):
    for i in range(25):
        bucket.put_month("spot", f"SYM{i:02d}USDT", "1h", "2024-01", kline_zip(2))
    bucket.page_size = 4
    with BinanceVision(client=bucket.client(), concurrency=2, sleep=lambda _: None) as client:
        symbols = client.list_symbols("spot")
    assert len(symbols) == 27  # 25 new plus BTCUSDT and ETHUSDT
    assert symbols == sorted(symbols)


# --- retries and status handling ---------------------------------------------------------


def test_retryable_status_is_retried_then_succeeds(bucket, client):
    key = zip_key("spot", "monthly", "BTCUSDT", "1h", "2024-01")
    bucket.failures[key] = [503, 429]
    archive = client.fetch_verified(zip_url("spot", "monthly", "BTCUSDT", "1h", "2024-01"))
    assert hashlib.sha256(archive.data).hexdigest() == archive.sha256


def test_retries_are_bounded(bucket, client):
    key = zip_key("spot", "monthly", "BTCUSDT", "1h", "2024-01")
    bucket.failures[key] = [503] * 20
    with pytest.raises(RuntimeError, match="giving up"):
        client.get(zip_url("spot", "monthly", "BTCUSDT", "1h", "2024-01"))


def test_backoff_grows_and_is_jittered(bucket):
    delays: list[float] = []
    bucket.failures[zip_key("spot", "monthly", "BTCUSDT", "1h", "2024-01")] = [503] * 3
    with BinanceVision(
        client=bucket.client(),
        concurrency=1,
        backoff_base=1.0,
        sleep=delays.append,
        rng=random.Random(7),
    ) as client:
        client.get(zip_url("spot", "monthly", "BTCUSDT", "1h", "2024-01"))
    assert len(delays) == 3
    assert delays[0] < delays[1] < delays[2]
    # Jitter means no delay is exactly the un-jittered doubling.
    assert all(delay not in (1.0, 2.0, 4.0) for delay in delays)


def test_404_is_not_retried(bucket, client):
    url = zip_url("spot", "monthly", "GHOSTUSDT", "1h", "2024-01")
    before = len(bucket.requests)
    with pytest.raises(NotFound):
        client.get(url)
    assert len(bucket.requests) - before == 1


def test_a_non_retryable_error_raises_immediately(bucket, client):
    key = zip_key("spot", "monthly", "BTCUSDT", "1h", "2024-01")
    bucket.failures[key] = [403]
    with pytest.raises(Exception, match="403"):
        client.get(zip_url("spot", "monthly", "BTCUSDT", "1h", "2024-01"))


# --- checksum verification ---------------------------------------------------------------


def test_verified_download_returns_the_published_digest(client):
    archive = client.fetch_verified(zip_url("spot", "monthly", "BTCUSDT", "1h", "2024-01"))
    assert archive.sha256 == hashlib.sha256(archive.data).hexdigest()
    assert archive.url.endswith("BTCUSDT-1h-2024-01.zip")


def test_corrupt_bytes_are_retried_once_then_fail_loudly(bucket, client):
    key = zip_key("spot", "monthly", "BTCUSDT", "1h", "2024-01")
    bucket.corrupt.add(key)
    before = sum(1 for url in bucket.requests if url.endswith(key))
    with pytest.raises(ChecksumMismatch):
        client.fetch_verified(zip_url("spot", "monthly", "BTCUSDT", "1h", "2024-01"))
    attempts = sum(1 for url in bucket.requests if url.endswith(key)) - before
    assert attempts == 2


def test_a_supplied_digest_skips_the_checksum_fetch(bucket, client):
    url = zip_url("spot", "monthly", "BTCUSDT", "1h", "2024-01")
    digest = client.fetch_checksum(url)
    before = sum(1 for u in bucket.requests if u.endswith(".CHECKSUM"))
    client.fetch_verified(url, digest)
    assert sum(1 for u in bucket.requests if u.endswith(".CHECKSUM")) == before


def test_fetch_all_preserves_order(client):
    urls = [
        zip_url("spot", "monthly", "BTCUSDT", "1h", "2024-02"),
        zip_url("spot", "monthly", "BTCUSDT", "1h", "2024-01"),
    ]
    assert [archive.url for archive in client.fetch_all(urls)] == urls


# --- the politeness cap ------------------------------------------------------------------


def test_concurrency_cap_is_honoured(bucket):
    peak = 0
    live = 0
    lock = threading.Lock()

    def slow(request):
        nonlocal peak, live
        with lock:
            live += 1
            peak = max(peak, live)
        try:
            return bucket.handler(request)
        finally:
            with lock:
                live -= 1

    import httpx

    for i in range(40):
        bucket.put_month("spot", f"CAP{i:02d}USDT", "1h", "2024-01", kline_zip(2))

    urls = [zip_url("spot", "monthly", f"CAP{i:02d}USDT", "1h", "2024-01") for i in range(40)]
    with BinanceVision(
        client=httpx.Client(transport=httpx.MockTransport(slow)),
        concurrency=3,
        sleep=lambda _: None,
    ) as client:
        client.fetch_all(urls)

    assert peak <= 3

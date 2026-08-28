"""data.binance.vision bulk downloader + zip->DataFrame conversion (P1-02/03/04).

Public dataset host, no API key. Monthly zips per symbol/timeframe, each with a
`.CHECKSUM` sibling. The downloader is resume-safe: a file already on disk whose
SHA-256 matches its CHECKSUM is skipped, a mismatching one is re-fetched.

Binance the *exchange* has exited the EU; this data site is a separate CDN and is
reachable from NL. If that changes, run the same code on Modal (P1-04):
`modal run infra/modal_app/download.py`.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import httpx
import pandas as pd

from .resample import timeframe_delta

BASE = "https://data.binance.vision"
LIST_ENDPOINT = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]
FUNDING_COLS = ["calc_time", "funding_interval_hours", "last_funding_rate"]

# Binance switched kline timestamps from milliseconds to microseconds for data
# from 2025-01 onward. Anything past this magnitude is microseconds.
_US_THRESHOLD = 1e14


@dataclass(frozen=True)
class Feed:
    """One (market, kind) stream of monthly zips."""

    prefix: str
    name: str

    def key_prefix(self, symbol: str, tf: str) -> str:
        return self.prefix.format(symbol=symbol, tf=tf)


SPOT_KLINES = Feed("data/spot/monthly/klines/{symbol}/{tf}/", "spot_klines")
UM_KLINES = Feed("data/futures/um/monthly/klines/{symbol}/{tf}/", "um_klines")
UM_FUNDING = Feed("data/futures/um/monthly/fundingRate/{symbol}/", "um_funding")


def list_keys(prefix: str, timeout: float = 30.0) -> list[str]:
    """List every `.zip` object under `prefix` (paginated S3 XML listing)."""
    keys: list[str] = []
    marker = ""
    with httpx.Client(timeout=timeout) as client:
        while True:
            r = client.get(
                LIST_ENDPOINT, params={"delimiter": "/", "prefix": prefix, "marker": marker}
            )
            r.raise_for_status()
            root = ElementTree.fromstring(r.text)
            page = [
                el.text
                for el in root.iter(f"{S3_NS}Key")
                if el.text and el.text.endswith(".zip")
            ]
            keys.extend(page)
            truncated = root.findtext(f"{S3_NS}IsTruncated") == "true"
            if not truncated:
                break
            last = [el.text for el in root.iter(f"{S3_NS}Key")][-1]
            if last == marker:  # defensive: never spin on a stuck marker
                break
            marker = last
    return sorted(keys)


def month_of(key: str) -> str:
    """`.../BTCUSDT-1m-2024-03.zip` -> `2024-03`."""
    return "-".join(Path(key).stem.split("-")[-2:])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


async def _fetch(client: httpx.AsyncClient, url: str, dest: Path, retries: int = 3) -> None:
    for attempt in range(retries):
        try:
            r = await client.get(url)
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(r.content)
            tmp.replace(dest)
            return
        except (httpx.HTTPError, OSError):
            if attempt == retries - 1:
                raise
            await asyncio.sleep(2**attempt)


async def _download_one(
    client: httpx.AsyncClient, key: str, out_root: Path, sem: asyncio.Semaphore
) -> tuple[str, str]:
    dest = out_root / key
    checksum_dest = dest.with_suffix(".zip.CHECKSUM")
    async with sem:
        if not checksum_dest.exists():
            await _fetch(client, f"{BASE}/{key}.CHECKSUM", checksum_dest)
        expected = checksum_dest.read_text().split()[0]

        if dest.exists() and _sha256(dest) == expected:
            return key, "cached"

        await _fetch(client, f"{BASE}/{key}", dest)
        if _sha256(dest) != expected:
            dest.unlink(missing_ok=True)
            raise ValueError(f"checksum mismatch for {key}")
        return key, "downloaded"


async def download_keys(
    keys: list[str], out_root: Path, concurrency: int = 8, timeout: float = 120.0
) -> dict[str, str]:
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        results = await asyncio.gather(
            *(_download_one(client, k, out_root, sem) for k in keys)
        )
    return dict(results)


def select_keys(
    feed: Feed,
    symbol: str,
    tf: str = "1m",
    start: str | None = None,
    end: str | None = None,
) -> list[str]:
    """Keys for `symbol` between `start` and `end` months (inclusive, `YYYY-MM`)."""
    keys = list_keys(feed.key_prefix(symbol, tf))
    return [
        k
        for k in keys
        if (start is None or month_of(k) >= start) and (end is None or month_of(k) <= end)
    ]


def _read_zip_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0])
    head = raw[:64].decode("utf-8", "ignore").lower()
    header = 0 if head.startswith(columns[0]) else None
    df = pd.read_csv(io.BytesIO(raw), header=header, names=None if header == 0 else columns)
    df.columns = columns[: len(df.columns)]
    return df


def klines_to_df(path: Path, tf: str = "1m") -> pd.DataFrame:
    """Monthly kline zip -> `ts, open, high, low, close, volume, amount`.

    `ts` is shifted to the bar **close** (Binance labels with open time); `amount`
    is quote-asset volume, which is what Kronos/Axiom expects in that column.
    """
    df = _read_zip_csv(path, KLINE_COLS)
    unit = "us" if df["open_time"].max() > _US_THRESHOLD else "ms"
    ts = pd.to_datetime(df["open_time"], unit=unit) + timeframe_delta(tf)
    out = pd.DataFrame(
        {
            "ts": ts,
            "open": df["open"].astype("float64"),
            "high": df["high"].astype("float64"),
            "low": df["low"].astype("float64"),
            "close": df["close"].astype("float64"),
            "volume": df["volume"].astype("float64"),
            "amount": df["quote_volume"].astype("float64"),
        }
    )
    return out.sort_values("ts").drop_duplicates(subset="ts").reset_index(drop=True)


def funding_to_df(path: Path) -> pd.DataFrame:
    """Monthly fundingRate zip -> `ts, funding_rate, funding_interval_hours`."""
    df = _read_zip_csv(path, FUNDING_COLS)
    unit = "us" if df["calc_time"].max() > _US_THRESHOLD else "ms"
    out = pd.DataFrame(
        {
            "ts": pd.to_datetime(df["calc_time"], unit=unit),
            "funding_rate": df["last_funding_rate"].astype("float64"),
            "funding_interval_hours": df["funding_interval_hours"].astype("float64"),
        }
    )
    return out.sort_values("ts").drop_duplicates(subset="ts").reset_index(drop=True)


__all__ = [
    "BASE",
    "SPOT_KLINES",
    "UM_FUNDING",
    "UM_KLINES",
    "Feed",
    "download_keys",
    "funding_to_df",
    "klines_to_df",
    "list_keys",
    "month_of",
    "select_keys",
]

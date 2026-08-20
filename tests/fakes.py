"""A synthetic Binance Vision bucket, served over an httpx mock transport.

Real archives never enter this repo -- not the test suite, not a fixture directory, not a
gitignored cache. The bucket below builds the same file format from scratch: a zip holding a
twelve-column CSV, a `.CHECKSUM` sibling with the real sha256 of those bytes, and an S3 XML
listing that paginates the way the real one does.

Using `httpx.MockTransport` rather than monkeypatching `BinanceVision.get` means the retry
policy, the status handling and the 404 semantics are all exercised for real. The only thing
faked is what is on the other end of the socket.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlparse

import httpx

from axiom.sources.binance_vision import DOWNLOAD_BASE, zip_key

S3_NS = "http://s3.amazonaws.com/doc/2006-03-01/"

#: The header Binance writes on newer archives.
CSV_HEADER = (
    "open_time,open,high,low,close,volume,close_time,quote_asset_volume,count,"
    "taker_buy_volume,taker_buy_quote_volume,ignore"
)

HOUR_MS = 3_600_000
DAY_MS = 86_400_000

#: 2024-01-01T00:00:00Z.
EPOCH = 1_704_067_200_000


def kline_rows(
    n: int,
    *,
    start: int = EPOCH,
    step: int = HOUR_MS,
    price: float = 100.0,
    volume: float = 3.0,
    unit: str = "ms",
) -> list[list[str]]:
    """``n`` well-formed source rows. ``unit='us'`` writes microsecond open times.

    Every field is a function of the bar's absolute grid position rather than its position in
    this call, so two calls that overlap in time produce identical rows for the overlapping
    bars. That is what makes a seam fixture a seam rather than a conflict.
    """
    scale = 1000 if unit == "us" else 1
    rows = []
    for offset in range(n):
        ts = start + offset * step
        i = (ts - EPOCH) // step
        o = price + i
        c = o + 0.5
        rows.append(
            [
                str(ts * scale),
                f"{o:.8f}",
                f"{max(o, c) + 1:.8f}",
                f"{min(o, c) - 1:.8f}",
                f"{c:.8f}",
                f"{volume:.8f}",
                str((ts + step - 1) * scale),
                f"{volume * o:.8f}",
                str(10 + i),
                f"{volume / 2:.8f}",
                f"{volume * o / 2:.8f}",
                "0",
            ]
        )
    return rows


def csv_bytes(rows: list[list[str]], *, header: bool = False) -> bytes:
    lines = [",".join(row) for row in rows]
    if header:
        lines.insert(0, CSV_HEADER)
    return ("\n".join(lines) + "\n").encode("utf-8")


def make_zip(payload: bytes, member: str = "klines.csv") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, payload)
    return buffer.getvalue()


def kline_zip(n: int, **kwargs) -> bytes:
    """One archive holding ``n`` synthetic bars."""
    header = kwargs.pop("header", False)
    return make_zip(csv_bytes(kline_rows(n, **kwargs), header=header))


@dataclass
class FakeBucket:
    """Keys to bytes, plus enough S3 semantics to satisfy the enumerator.

    ``page_size`` forces pagination so the marker handling is exercised on a bucket small enough
    to reason about; the real one paginates at 1000.
    """

    objects: dict[str, bytes] = field(default_factory=dict)
    page_size: int = 1000
    #: Keys that answer with these statuses before succeeding, popped one per request.
    failures: dict[str, list[int]] = field(default_factory=dict)
    #: Keys whose bytes are served corrupted, so the published checksum will not match.
    corrupt: set[str] = field(default_factory=set)
    requests: list[str] = field(default_factory=list)

    # --- population -------------------------------------------------------------------

    def put_month(self, market: str, symbol: str, frequency: str, period: str, data: bytes) -> None:
        self.objects[zip_key(market, "monthly", symbol, frequency, period)] = data

    def put_day(self, market: str, symbol: str, frequency: str, period: str, data: bytes) -> None:
        self.objects[zip_key(market, "daily", symbol, frequency, period)] = data

    # --- serving ----------------------------------------------------------------------

    def _listing(self, prefix: str, delimiter: str, marker: str) -> str:
        keys = sorted(k for k in self.objects if k.startswith(prefix))
        entries: list[tuple[str, str]] = []
        seen: set[str] = set()
        for key in keys:
            rest = key[len(prefix) :]
            if delimiter and delimiter in rest:
                common = prefix + rest.split(delimiter, 1)[0] + delimiter
                if common not in seen:
                    seen.add(common)
                    entries.append(("CommonPrefixes", common))
            else:
                entries.append(("Contents", key))

        entries.sort(key=lambda pair: pair[1])
        if marker:
            entries = [e for e in entries if e[1] > marker]
        page, truncated = entries[: self.page_size], len(entries) > self.page_size

        body = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<ListBucketResult xmlns="{S3_NS}">',
            "<Name>data.binance.vision</Name>",
            # The echo of the request prefix: an enumerator that scans for any element named
            # Prefix picks this up and invents a symbol named "".
            f"<Prefix>{prefix}</Prefix>",
            f"<Delimiter>{delimiter}</Delimiter>",
            f"<IsTruncated>{'true' if truncated else 'false'}</IsTruncated>",
        ]
        for kind, value in page:
            tag = "Key" if kind == "Contents" else "Prefix"
            body.append(f"<{kind}><{tag}>{value}</{tag}></{kind}>")
        body.append("</ListBucketResult>")
        return "".join(body)

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        parsed = urlparse(url)

        if "s3-" in parsed.netloc:
            query = parse_qs(parsed.query)
            return httpx.Response(
                200,
                text=self._listing(
                    query.get("prefix", [""])[0],
                    query.get("delimiter", ["/"])[0],
                    query.get("marker", [""])[0],
                ),
            )

        key = parsed.path.lstrip("/")
        pending = self.failures.get(key.removesuffix(".CHECKSUM"))
        if pending:
            return httpx.Response(pending.pop(0))

        if key.endswith(".CHECKSUM"):
            target = key.removesuffix(".CHECKSUM")
            if target not in self.objects:
                return httpx.Response(404)
            digest = hashlib.sha256(self.objects[target]).hexdigest()
            return httpx.Response(200, text=f"{digest}  {target.rsplit('/', 1)[-1]}\n")

        if key not in self.objects:
            return httpx.Response(404)
        data = self.objects[key]
        if key in self.corrupt:
            data = data + b"tampered"
        return httpx.Response(200, content=data)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler), base_url=DOWNLOAD_BASE)

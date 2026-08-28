"""Regenerate `configs/universe_v1.yaml` (P1-01).

Selects Binance USDT spot pairs by **median daily quote volume measured over a window
inside the training split**, not by a live 24h snapshot. That distinction matters: a
snapshot ranks a coin that pumped yesterday above one that has been liquid for years,
and the pumped coin turns out to have no trades in 40% of its minutes. Ranking on
train-period medians is robust to that and uses no information from val or test.

Each survivor is annotated with the first month available on data.binance.vision (the
start of usable history) and whether a USD-M perp exists for funding/OI work (M4).

Known bias, stated rather than hidden: candidates come from the pairs Binance lists
*today*, so coins delisted since the training period are absent and the set is a
survivor set. The fix would be to enumerate every symbol ever published on
data.binance.vision; it is in the backlog, not in this script.

    uv run python scripts/build_universe.py --top 50 --out configs/universe_v1.yaml
"""

from __future__ import annotations

import argparse
import statistics
import time
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
from axiom_data import binance

TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr"
KLINES_URL = "https://api.binance.com/api/v3/klines"

# Quote-for-quote pairs carry no directional signal; leveraged tokens are derivatives
# with their own decay dynamics and do not belong in an OHLCV corpus.
STABLE_BASES = {
    "USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "USD1", "USDS", "USDE", "PYUSD",
    "RLUSD", "UST", "EUR", "EURI", "GBP", "AEUR", "TRY", "BRL", "ARS", "JPY", "AUD",
}
LEVERAGED_SUFFIXES = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")


def _ms(day: str) -> int:
    return int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp() * 1000)


def _get_json(client: httpx.Client, url: str, params: dict | None = None, retries: int = 5):
    """Binance drops connections and rate-limits under a long sequential sweep."""
    for attempt in range(retries):
        try:
            r = client.get(url, params=params, timeout=60)
            if r.status_code in (418, 429):
                time.sleep(5 * (attempt + 1))
                continue
            return r.raise_for_status().json()
        except httpx.HTTPError:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError(f"gave up on {url} {params}")


def candidates(client: httpx.Client) -> list[str]:
    """Every USDT spot pair worth considering (order here does not matter)."""
    rows = _get_json(client, TICKER_URL)
    return [
        r["symbol"]
        for r in rows
        if r["symbol"].endswith("USDT")
        and r["symbol"][: -len("USDT")] not in STABLE_BASES
        and not r["symbol"].endswith(LEVERAGED_SUFFIXES)
    ]


def liquidity(client: httpx.Client, symbol: str, start: str, end: str) -> tuple[float, float]:
    """`(median daily quote volume, continuity)` over `[start, end)` from daily klines.

    The median, not the mean: one 50x volume day should not buy a symbol a place in
    the universe. Days the symbol did not trade count as **zero**, not as absent -- a
    coin that was liquid for eight months and then delisted is not a liquid coin over
    this window, and letting absent days drop out of the median would say it was.

    Continuity is traded days over the span between the symbol's *own* first and last
    traded day in the window. Listing late is not a defect and is not penalised; going
    dark for months in the middle is, and that is what this catches (FTTUSDT after the
    FTX collapse: halted, 13% of its bars missing, still liquid enough on the median).
    """
    volumes: list[float] = []
    cursor, stop = _ms(start), _ms(end)
    days = (stop - cursor) // 86_400_000
    traded_days: list[int] = []
    while cursor < stop:
        rows = _get_json(
            client,
            KLINES_URL,
            {
                "symbol": symbol,
                "interval": "1d",
                "startTime": cursor,
                "endTime": stop,
                "limit": 1000,
            },
        )
        if not rows:
            break
        volumes += [float(r[7]) for r in rows]
        traded_days += [r[0] // 86_400_000 for r in rows]
        cursor = rows[-1][0] + 86_400_000
        if len(rows) < 1000:
            break
    volumes += [0.0] * (days - len(volumes))
    if not traded_days:
        return 0.0, 0.0
    span = traded_days[-1] - traded_days[0] + 1
    return statistics.median(volumes), len(traded_days) / span


def annotate(symbol: str) -> dict | None:
    """History start and perp availability, straight from the bulk-data listing."""
    months = binance.list_keys(binance.SPOT_KLINES.key_prefix(symbol, "1m"))
    if not months:
        return None
    return {
        "symbol": symbol,
        "listed": binance.month_of(months[0]),
        "perp": bool(binance.list_keys(binance.UM_KLINES.key_prefix(symbol, "1m"))),
    }


def render(entries: list[dict], args: argparse.Namespace) -> str:
    lines = [
        "# Axiom universe v1 -- Binance USDT spot pairs (+ matching USD-M perps).",
        "#",
        f"# Generated by scripts/build_universe.py on {date.today()}:",
        f"#   --top {args.top} --listed-before {args.listed_before}"
        f" --liquidity-window {args.liquidity_start}..{args.liquidity_end}"
        f" --min-median-daily-usd {args.min_median_daily_usd:,.0f}"
        f" --min-continuity {args.min_continuity}",
        "#",
        "# Ranked by MEDIAN DAILY QUOTE VOLUME over a window inside the TRAIN split, so",
        "# no val/test information selects the universe and a single pump day cannot buy",
        "# a listing. Symbols that went dark mid-window (a halt, not a late listing) are",
        "# dropped at the same 98% bar the QA gap check uses, so selection and QA agree.",
        "# `med_daily_musd` is that median in millions of USD. `listed` is the",
        "# first month on data.binance.vision (start of usable history, not necessarily",
        "# the exchange listing date). `perp` marks bases with a USD-M perpetual (M4).",
        "#",
        "# BIAS, on the record: candidates are the pairs Binance lists today, so coins",
        "# delisted since the training period are missing and this is a survivor set.",
        "# Evaluation reports must say so and weight recent months (build order 4.4).",
        "#",
        "# FROZEN before Phase 2. Changing it invalidates every comparison made with it.",
        "version: 1",
        "venue: binance",
        "quote: USDT",
        "perps: true",
        "symbols:",
    ]
    lines += [
        f"  - {{symbol: {e['symbol']}, listed: \"{e['listed']}\","
        f" perp: {str(e['perp']).lower()}, med_daily_musd: {e['med_daily_musd']}}}"
        for e in entries
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--top", type=int, default=50, help="how many symbols to keep")
    p.add_argument(
        "--listed-before",
        default="2022-01",
        help="skip pairs whose history starts later; they contribute few training windows",
    )
    p.add_argument("--liquidity-start", default="2021-01-01", help="inside the train split")
    p.add_argument("--liquidity-end", default="2024-01-01", help="train split end, exclusive")
    p.add_argument(
        "--min-median-daily-usd",
        type=float,
        default=5e6,
        help="liquidity floor; below this, fee/slippage assumptions stop being credible",
    )
    p.add_argument(
        "--min-continuity",
        type=float,
        default=0.98,
        help="traded days / days spanned, within the window; same bar as the QA gap check",
    )
    p.add_argument("--out", default="configs/universe_v1.yaml")
    args = p.parse_args()

    with httpx.Client() as client:
        symbols = candidates(client)
        print(f"{len(symbols)} candidate USDT pairs; measuring train-period liquidity")
        measured = {
            s: liquidity(client, s, args.liquidity_start, args.liquidity_end) for s in symbols
        }
    ranked = sorted(((v, c, s) for s, (v, c) in measured.items()), reverse=True)

    entries: list[dict] = []
    for volume, continuity, symbol in ranked:
        if len(entries) == args.top:
            break
        if volume < args.min_median_daily_usd:
            print(f"stopping at {symbol}: median {volume:,.0f} below the floor")
            break
        if continuity < args.min_continuity:
            print(f"  {symbol}: traded {continuity:.1%} of its own span, halted, dropped")
            continue
        info = annotate(symbol)
        if info is None:
            print(f"  {symbol}: no bulk history, dropped")
            continue
        if info["listed"] >= args.listed_before:
            print(f"  {symbol}: history starts {info['listed']}, too young, dropped")
            continue
        info["med_daily_musd"] = round(volume / 1e6, 2)
        entries.append(info)
        print(f"  {symbol}: listed {info['listed']} perp={info['perp']} median {volume:,.0f}")

    Path(args.out).write_text(render(entries, args), encoding="utf-8")
    print(f"\n{len(entries)} symbols -> {args.out}")


if __name__ == "__main__":
    main()

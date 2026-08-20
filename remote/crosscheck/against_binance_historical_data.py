"""The ADR-0012 safety net: diff `axiom-raw` against an independent implementation.

A custom fetcher's real risk is not a bug that a test would catch. It is a systematic
misunderstanding — bar open time read as close time, a column one position off, the wrong
quote-volume field — which every test written by the same author against the same assumption
would agree with. The only thing that catches that is a second implementation that made its own
mistakes.

So this downloads the same series with `binance_historical_data`, an unrelated package by an
unrelated author, and compares row counts over the overlap plus every OHLCV value on a sampled
day. It runs in the cloud, on demand, and writes nothing to the Hub.

Not part of the package: it needs a dependency nothing else needs, and it is a check that runs a
handful of times per version rather than code anything imports.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import sys
import tempfile

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from axiom.config.settings import AxiomSettings
from axiom.sources.binance import artifact_path
from axiom.sources.binance_klines import SOURCE_COLUMNS

#: (our column, the source column it came from). Compared exactly. Volumes are in here
#: deliberately: a base/quote mix-up is precisely the systematic error this exists to catch, and
#: it would leave every price identical.
COMPARED_COLUMNS = (
    ("open", "open"),
    ("high", "high"),
    ("low", "low"),
    ("close", "close"),
    ("volume", "volume"),
    ("amount", "quote_asset_volume"),
)


def load_theirs(root: str, symbol: str, frequency: str) -> pa.Table:
    """Read every CSV the dumper wrote for one series, whatever tree it chose."""
    paths = sorted(
        p
        for p in glob.glob(f"{root}/**/*.csv", recursive=True)
        if f"{os.sep}{symbol}{os.sep}" in p and f"{os.sep}{frequency}{os.sep}" in p
    )
    if not paths:
        raise SystemExit(f"the dumper wrote no CSV for {symbol} {frequency} under {root}")

    tables = []
    for path in paths:
        with open(path, "rb") as handle:
            raw = handle.read()
        first = raw.split(b"\n", 1)[0].split(b",", 1)[0].strip().strip(b'"')
        try:
            float(first)
            skip = 0
        except ValueError:
            skip = 1
        tables.append(
            pacsv.read_csv(
                path,
                read_options=pacsv.ReadOptions(column_names=list(SOURCE_COLUMNS), skip_rows=skip),
                convert_options=pacsv.ConvertOptions(
                    column_types=dict.fromkeys(SOURCE_COLUMNS, pa.float64()),
                    include_columns=[
                        "open_time",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "quote_asset_volume",
                    ],
                ),
            )
        )
    combined = pa.concat_tables(tables).combine_chunks()
    return combined.sort_by("open_time")


def compare(symbol: str, frequency: str, market: str, theirs_root: str, sample_day: str) -> bool:
    """Diff one series. Returns True when the two implementations agree."""
    settings = AxiomSettings()
    path = artifact_path(market, frequency, symbol)
    ours = pq.read_table(
        hf_hub_download(
            repo_id=settings.raw_repo_id,
            filename=path,
            repo_type="dataset",
            token=settings.hf_token.get_secret_value() if settings.hf_token else None,
        )
    )
    theirs = load_theirs(theirs_root, symbol, frequency)

    their_ts = [int(v) for v in theirs["open_time"].to_pylist()]
    our_ts = {ts: i for i, ts in enumerate(ours["ts"].to_pylist())}
    lo, hi = min(their_ts), max(their_ts)
    ours_in_window = [ts for ts in our_ts if lo <= ts <= hi]

    print(f"\n=== {market}/{frequency}/{symbol} ===")
    print(f"  ours:   {ours.num_rows} rows, window {lo}..{hi} holds {len(ours_in_window)}")
    print(f"  theirs: {theirs.num_rows} rows")

    agreed = True
    missing = sorted(set(their_ts) - set(our_ts))
    extra = sorted(set(ours_in_window) - set(their_ts))
    if missing:
        print(f"  MISSING from ours: {len(missing)}, first {missing[:3]}")
        agreed = False
    if extra:
        # Off-grid bars from an exchange restart legitimately appear here: the other package
        # reads the same archives, so a difference means one of us dropped or invented a row.
        print(f"  EXTRA in ours: {len(extra)}, first {extra[:3]}")
        agreed = False

    day = dt.datetime.strptime(sample_day, "%Y-%m-%d").replace(tzinfo=dt.UTC)
    start = int(day.timestamp() * 1000)
    end = start + 86_400_000
    checked = 0
    for row, ts in enumerate(their_ts):
        if not (start <= ts < end) or ts not in our_ts:
            continue
        checked += 1
        for our_name, their_name in COMPARED_COLUMNS:
            mine = ours[our_name][our_ts[ts]].as_py()
            yours = theirs[their_name][row].as_py()
            if mine != yours:
                print(f"  VALUE MISMATCH ts={ts} {our_name}: ours={mine} theirs={yours}")
                agreed = False
    print(f"  sampled {sample_day}: {checked} bars compared on {[c for c, _ in COMPARED_COLUMNS]}")
    if checked == 0:
        print("  WARNING: the sampled day held no overlapping bars; pick another")

    print(f"  verdict: {'AGREE' if agreed else 'DISAGREE'}")
    return agreed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--frequency", default="1h")
    parser.add_argument("--market", default="spot", choices=["spot", "um"])
    parser.add_argument("--date-start", default="2024-01-01")
    parser.add_argument("--date-end", default="2024-03-31")
    parser.add_argument("--sample-day", default="2024-02-14")
    args = parser.parse_args()

    from binance_historical_data import BinanceDataDumper

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    with tempfile.TemporaryDirectory() as root:
        BinanceDataDumper(
            path_dir_where_to_dump=root,
            asset_class=args.market,
            data_type="klines",
            data_frequency=args.frequency,
        ).dump_data(
            tickers=symbols,
            date_start=dt.date.fromisoformat(args.date_start),
            date_end=dt.date.fromisoformat(args.date_end),
        )
        verdicts = [
            compare(symbol, args.frequency, args.market, root, args.sample_day)
            for symbol in symbols
        ]

    agreed = sum(verdicts)
    print(f"\n{agreed}/{len(verdicts)} series agree with binance_historical_data")
    return 0 if agreed == len(verdicts) else 1


if __name__ == "__main__":
    sys.exit(main())

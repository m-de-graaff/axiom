"""`axiom-data` CLI: download -> ingest -> qa -> build.

    axiom-data download --config configs/universe_v1.yaml
    axiom-data ingest   --config configs/data/crypto_v1.yaml
    axiom-data qa       --config configs/data/crypto_v1.yaml
    axiom-data build    --config configs/data/crypto_v1.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import pandas as pd
import yaml

from . import binance, datasets, qa, store
from .resample import resample_ohlcv

FEEDS = {"spot": binance.SPOT_KLINES, "futures": binance.UM_KLINES, "funding": binance.UM_FUNDING}


def _universe(path: Path) -> tuple[list[str], dict]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    symbols = [s["symbol"] if isinstance(s, dict) else s for s in cfg["symbols"]]
    return symbols, cfg


def cmd_download(args: argparse.Namespace) -> int:
    symbols, cfg = _universe(Path(args.config))
    if args.symbols:
        symbols = args.symbols
    out = Path(args.out)
    total = {"downloaded": 0, "cached": 0}
    for feed_name in args.feeds:
        feed = FEEDS[feed_name]
        tf = args.source_tf if feed is not binance.UM_FUNDING else ""
        for symbol in symbols:
            keys = binance.select_keys(feed, symbol, tf, args.start, args.end)
            if not keys:
                print(f"  {feed_name}/{symbol}: no files in range", file=sys.stderr)
                continue
            result = asyncio.run(binance.download_keys(keys, out, args.concurrency))
            got = sum(v == "downloaded" for v in result.values())
            total["downloaded"] += got
            total["cached"] += len(result) - got
            print(f"  {feed_name}/{symbol}: {len(keys)} files ({got} new)")
    print(f"download: {total['downloaded']} new, {total['cached']} already verified -> {out}")
    return 0


def _raw_months(raw: Path, feed: binance.Feed, symbol: str, tf: str) -> list[Path]:
    return sorted((raw / feed.key_prefix(symbol, tf)).glob("*.zip"))


def cmd_ingest(args: argparse.Namespace) -> int:
    """Raw zips -> parquet. Futures land under a separate venue so spot stays spot."""
    cfg = datasets.load_config(args.config)
    feed = FEEDS[args.feed]
    venue = cfg.venue if args.feed == "spot" else f"{cfg.venue}-um"
    raw, root = Path(args.raw), Path(args.root)

    for symbol in args.symbols or cfg.symbols:
        source_tf = "" if feed is binance.UM_FUNDING else cfg.source_tf
        months = _raw_months(raw, feed, symbol, source_tf)
        if not months:
            print(f"  {symbol}: no raw zips, skipped", file=sys.stderr)
            continue

        if feed is binance.UM_FUNDING:
            rates = pd.concat([binance.funding_to_df(m) for m in months])
            rates = rates.sort_values("ts").drop_duplicates(subset="ts")
            store.write_months(rates, root, venue, symbol, "funding")
            print(f"  {symbol}: funding:{len(rates)}")
            continue

        bars = pd.concat([binance.klines_to_df(m, cfg.source_tf) for m in months])
        bars = bars.sort_values("ts").drop_duplicates(subset="ts").reset_index(drop=True)
        store.write_months(bars, root, venue, symbol, cfg.source_tf)
        line = [f"{cfg.source_tf}:{len(bars)}"]
        for tf in cfg.timeframes:
            resampled = resample_ohlcv(bars, tf)
            store.write_months(resampled, root, venue, symbol, tf)
            line.append(f"{tf}:{len(resampled)}")
        print(f"  {symbol}: {' '.join(line)}")

    print(f"ingest: {args.feed} written under {root / venue}")
    return 0


def cmd_qa(args: argparse.Namespace) -> int:
    cfg = datasets.load_config(args.config)
    rows = [
        qa.check_frame(bars, symbol, tf)
        for tf in [cfg.source_tf, *cfg.timeframes]
        for symbol in cfg.symbols
        if not (bars := store.read(symbol, tf, root=Path(args.root), venue=cfg.venue)).empty
    ]
    report = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(report.to_string(index=False))
    failures = qa.violations(report, cfg.qa_thresholds)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.out, index=False)
    if failures:
        print("\nQA FAILED:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    print("\nQA clean.")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    manifest = datasets.build(args.config, root=Path(args.root), out_dir=Path(args.out))
    for split, stats in manifest["splits"].items():
        print(
            f"  {split:5s} {stats['windows']:>10,} windows  "
            f"{stats['segments']:>5} segments  {stats['symbols']:>3} symbols"
        )
    print(f"\ndataset  {manifest['dataset']}")
    print(f"hash     {manifest['dataset_hash']}")
    print(f"written  {manifest.get('path', '(not written)')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="axiom-data", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="fetch monthly zips from data.binance.vision")
    d.add_argument("--config", default="configs/universe_v1.yaml")
    d.add_argument("--out", default="data/raw")
    d.add_argument("--symbols", nargs="*", help="override the universe (debugging)")
    d.add_argument("--feeds", nargs="+", default=["spot"], choices=sorted(FEEDS))
    d.add_argument("--source-tf", default="1m")
    d.add_argument("--start", help="first month, YYYY-MM")
    d.add_argument("--end", help="last month, YYYY-MM")
    d.add_argument("--concurrency", type=int, default=8)
    d.set_defaults(func=cmd_download)

    i = sub.add_parser("ingest", help="raw zips -> partitioned parquet + resampled timeframes")
    i.add_argument("--config", default="configs/data/crypto_v1.yaml")
    i.add_argument("--raw", default="data/raw")
    i.add_argument("--root", default=str(store.DEFAULT_ROOT))
    i.add_argument("--feed", default="spot", choices=sorted(FEEDS))
    i.add_argument("--symbols", nargs="*", help="override the universe (debugging)")
    i.set_defaults(func=cmd_ingest)

    q = sub.add_parser("qa", help="corpus QA report")
    q.add_argument("--config", default="configs/data/crypto_v1.yaml")
    q.add_argument("--root", default=str(store.DEFAULT_ROOT))
    q.add_argument("--out", help="also write the report as CSV")
    q.set_defaults(func=cmd_qa)

    b = sub.add_parser("build", help="splits + embargo + window index + dataset hash")
    b.add_argument("--config", default="configs/data/crypto_v1.yaml")
    b.add_argument("--root", default=str(store.DEFAULT_ROOT))
    b.add_argument("--out", default="data/datasets")
    b.set_defaults(func=cmd_build)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

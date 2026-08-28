"""Bulk downloader for data.binance.vision (P1-02/03).

Plan:
  - spot monthly 1m klines:   data/spot/monthly/klines/{SYMBOL}/1m/{SYMBOL}-1m-{YYYY-MM}.zip
  - USD-M futures klines:     data/futures/um/monthly/klines/{SYMBOL}/1m/...
  - USD-M funding rates:      data/futures/um/monthly/fundingRate/{SYMBOL}/...
  - verify .CHECKSUM files; resume-safe; async (httpx); write raw zips to
    data/raw/, extract+convert to Parquet via axiom_data.
Fallback (P1-04): run this on a Modal function / non-EU VPS if unreachable from NL.
"""

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/universe_v1.yaml")
    p.add_argument("--out", default="data/raw")
    p.parse_args()
    raise SystemExit("TODO P1-02: implement (see docstring + build order §3.1)")


if __name__ == "__main__":
    main()

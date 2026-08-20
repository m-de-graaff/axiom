"""Does Dukascopy answer a Kaggle kernel?

GitHub Actions runners get HTTP 403 from `freeserv.dukascopy.com` — the JSON chart endpoint
`dukascopy-python` reads — so the v0.2 plan's cloud pull has no backend yet. ADR-0015's fallback
ladder names a Kaggle CPU kernel as the next rung. This is that rung, asked directly.

Prints the egress IP, the raw HTTP status, and a real fetch's row count. Costs no GPU quota and
writes nothing anywhere.
"""

import datetime as dt
import subprocess
import sys
import urllib.request

URL = (
    "https://freeserv.dukascopy.com/2.0/index.php?path=chart/json3&instrument=EUR/USD"
    "&offer_side=B&interval=1DAY&splits=true&stocks=true&time_direction=N"
    "&timestamp=1704067200000&jsonp="
)

try:
    ip = urllib.request.urlopen("https://api.ipify.org", timeout=20).read().decode()
except Exception as exc:
    ip = f"unknown ({exc})"
print("kernel egress IP:", ip, flush=True)

try:
    with urllib.request.urlopen(URL, timeout=40) as response:
        body = response.read()
    print(f"freeserv.dukascopy.com HTTP {response.status}, {len(body)} bytes", flush=True)
    print(body[:300], flush=True)
except Exception as exc:
    print(f"freeserv.dukascopy.com FAILED: {type(exc).__name__}: {exc}", flush=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "dukascopy-python"], check=True)

import dukascopy_python  # noqa: E402
from dukascopy_python.instruments import INSTRUMENT_FX_MAJORS_EUR_USD as EURUSD  # noqa: E402

frame = dukascopy_python.fetch(
    EURUSD,
    dukascopy_python.INTERVAL_DAY_1,
    dukascopy_python.OFFER_SIDE_BID,
    dt.datetime(2024, 1, 1),
    dt.datetime(2025, 1, 1),
)
print(f"VERDICT: dukascopy-python returned {len(frame)} daily bars for 2024", flush=True)
print("REACHABLE" if len(frame) > 200 else "BLOCKED", flush=True)

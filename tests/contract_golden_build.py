"""Regenerate the golden fixtures under ``tests/fixtures/contract/``.

Run with ``uv run python tests/contract_golden_build.py``. It is not a test and pytest does not
collect it: the fixtures are the frozen thing, and a generator that ran on every test run would
freeze nothing.

Regenerating is a deliberate act with a paper trail — a contract change that alters these numbers
is a ``schema_version`` bump (ADR-0020), and the diff on these files is the evidence for it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from axiom.contract import load_spec, transform
from axiom.contract.spec import ContractSpec
from axiom.testing import synth
from axiom.testing.contract import constants

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "contract"
SPEC_NAMES = ("contract_geo_v1", "contract_ret_v1")


def _cases() -> list[tuple[str, str, str, synth.SynthSeries]]:
    """Six shapes, each chosen because it is where an implementation goes wrong."""
    plain = synth.walk("1h", 40)

    gapped = synth.walk("1h", 60)
    # A 3 % jump between close_{t-1} and open_t: the gap feature is the only one that spans two
    # bars, so this is the fixture that catches an off-by-one in the previous-close alignment.
    open_ = gapped.column("open")
    high = gapped.column("high")
    low = gapped.column("low")
    close = gapped.column("close")
    for i in range(20, 40):
        for arr in (open_, high, low, close):
            arr[i] *= 1.03
    gapped = synth._rebuild(gapped, open_=open_, high=high, low=low, close=close)

    # high == open: the upper wick is exactly 0, the boundary of `upper >= max(0, body)`.
    touch_high = synth.walk("1h", 40)
    o = touch_high.column("open")
    c = touch_high.column("close")
    h = touch_high.column("high")
    h[10:15] = np.maximum(o[10:15], c[10:15])
    o[10:15] = h[10:15]
    c[10:15] = np.minimum(c[10:15], h[10:15])
    touch_high = synth._rebuild(touch_high, open_=o, high=h, close=c)

    # low == open: the lower wick is exactly 0.
    touch_low = synth.walk("1h", 40)
    o = touch_low.column("open")
    c = touch_low.column("close")
    lo = touch_low.column("low")
    lo[10:15] = np.minimum(o[10:15], c[10:15])
    o[10:15] = lo[10:15]
    c[10:15] = np.maximum(c[10:15], lo[10:15])
    touch_low = synth._rebuild(touch_low, open_=o, low=lo, close=c)

    # A run of exactly-zero volume: log1p(0) = 0, and a median over a window of zeros is zero,
    # so the flow feature is exactly 0 there rather than -inf. The fixture proves it.
    dead = synth.with_illiquid(synth.walk("1h", 40), at=15, n=8)
    v = dead.column("volume")
    v[15:23] = 0.0
    dead = synth._rebuild(dead, volume=v)

    # 258 bars: 257 feature rows, so rows 1..255 are expanding, row 256 is the first full window
    # and row 257 is the first that rolls. Every boundary of the strictly-past median, once.
    boundary = synth.walk("1h", 258)

    return [
        ("plain_walk", "crypto", "a clean 1h random walk, no pathology", plain),
        ("gap_edge", "crypto", "a 3 % overnight-style gap at bar 20", gapped),
        ("high_equals_open", "crypto", "five bars whose high is exactly the open", touch_high),
        ("low_equals_open", "crypto", "five bars whose low is exactly the open", touch_low),
        ("zero_volume_run", "crypto", "an eight-bar run of exactly zero volume", dead),
        (
            "window_boundary",
            "crypto",
            "258 bars: expanding, full and rolling median in one",
            boundary,
        ),
    ]


def _columns(series: synth.SynthSeries) -> dict[str, list]:
    table = series.table
    return {
        name: [
            int(v) if name == "ts" else float(v) for v in table[name].to_numpy(zero_copy_only=False)
        ]
        for name in ("ts", "open", "high", "low", "close", "volume", "amount")
    }


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    specs: list[ContractSpec] = [load_spec(name) for name in SPEC_NAMES]
    table = constants(specs)
    for name, asset_class, description, series in _cases():
        expected = {}
        for spec in specs:
            block = transform(
                series.table,
                spec,
                table,
                asset_class=asset_class,
                frequency=series.frequency,
            )
            expected[spec.spec_id] = [[float(x) for x in row] for row in block.values]
        payload = {
            "name": name,
            "description": description,
            "verification": (
                "Values are frozen output. tests/test_contract_golden.py recomputes rows from a "
                "second implementation written from the ADR-0020 formulas rather than from "
                "axiom.contract, and both must agree before the frozen numbers are trusted."
            ),
            "asset_class": asset_class,
            "frequency": series.frequency,
            "constants": table.model_dump(mode="json"),
            "bars": _columns(series),
            "expected": expected,
        }
        path = FIXTURE_DIR / f"{name}.json"
        path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {path} ({len(payload['bars']['ts'])} bars)")


if __name__ == "__main__":
    main()

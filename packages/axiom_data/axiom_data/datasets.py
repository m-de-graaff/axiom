"""Dataset builder (P1-10): chronological splits, embargo, sliding windows, hash.

What gets written is a *segment index*, not materialized windows. A segment is a
maximal gap-free run of bars inside one split for one symbol/timeframe; every
sliding window is an offset into a segment, so `n_windows` is exact while the
index stays a few thousand rows instead of millions. Loaders expand segments with
`expand`; nothing else may enumerate windows.

Leakage rules enforced here (see CLAUDE.md rule 3):
  * splits are chronological and must not overlap;
  * every window -- context *and* horizon -- lies entirely inside its own split, so
    no validation or test window can reach back into training bars. With
    `embargo_bars <= context_bars` that containment *is* the embargo; a larger
    embargo additionally shifts each split's usable start forward;
  * windows never span a data gap.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from . import qa, store
from .resample import timeframe_delta

SEGMENT_COLUMNS = ["symbol", "tf", "split", "start_ts", "end_ts", "bars", "windows"]


@dataclass(frozen=True)
class DataConfig:
    """Resolved `configs/data/*.yaml`."""

    name: str
    symbols: list[str]
    venue: str
    source_tf: str
    timeframes: list[str]
    context_bars: int
    horizons: list[int]
    embargo_bars: int
    normalization: str
    resample: str
    splits: dict[str, dict[str, str]]
    qa_thresholds: dict

    @property
    def window_bars(self) -> int:
        """Context plus the longest horizon -- the span one training sample covers."""
        return self.context_bars + max(self.horizons)

    def spec(self) -> dict:
        """The part of the config the dataset hash is computed over."""
        return {
            "symbols": sorted(self.symbols),
            "venue": self.venue,
            "source_tf": self.source_tf,
            "timeframes": list(self.timeframes),
            "context_bars": self.context_bars,
            "horizons": list(self.horizons),
            "embargo_bars": self.embargo_bars,
            "normalization": self.normalization,
            "resample": self.resample,
            "splits": self.splits,
        }


def load_config(path: str | Path) -> DataConfig:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    universe = yaml.safe_load(Path(cfg["universe"]).read_text(encoding="utf-8"))
    symbols = [s["symbol"] if isinstance(s, dict) else s for s in universe["symbols"]]
    return DataConfig(
        name=Path(path).stem,
        symbols=symbols,
        venue=universe.get("venue", "binance"),
        source_tf=cfg["source_tf"],
        timeframes=cfg["timeframes"],
        context_bars=cfg["context_bars"],
        horizons=cfg["horizons"],
        embargo_bars=cfg["embargo_bars"],
        normalization=cfg["normalization"],
        resample=cfg["resample"],
        splits=cfg["splits"],
        qa_thresholds={**qa.DEFAULT_THRESHOLDS, **cfg.get("qa", {})},
    )


def check_splits(splits: dict[str, dict[str, str]]) -> None:
    """Splits must be chronological and disjoint. Overlap is a hard build failure."""
    bounds = [(k, pd.Timestamp(v["start"]), pd.Timestamp(v["end"])) for k, v in splits.items()]
    bounds.sort(key=lambda b: b[1])
    for (a, _, a_end), (b, b_start, _) in zip(bounds, bounds[1:], strict=False):
        if b_start <= a_end:
            raise ValueError(f"splits {a!r} and {b!r} overlap: {a_end} >= {b_start}")


def split_bounds(cfg: DataConfig, split: str, tf: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Usable range for a split, with the extra embargo lead applied if configured."""
    start = pd.Timestamp(cfg.splits[split]["start"])
    end = pd.Timestamp(cfg.splits[split]["end"])
    extra = max(0, cfg.embargo_bars - cfg.context_bars)
    return start + extra * timeframe_delta(tf), end


def segments(ts: pd.Series, tf: str, window_bars: int) -> list[tuple[int, int]]:
    """Maximal gap-free runs of at least `window_bars` bars, as `(start, stop)` slices."""
    if ts.empty:
        return []
    step = timeframe_delta(tf)
    breaks = (ts.diff() != step).to_numpy(copy=True)
    breaks[0] = True
    starts = [i for i, b in enumerate(breaks) if b]
    stops = starts[1:] + [len(ts)]
    return [(s, e) for s, e in zip(starts, stops, strict=True) if e - s >= window_bars]


def expand(start_ts: pd.Timestamp, bars: int, tf: str, window_bars: int):
    """Yield the context-start timestamp of every window in a segment."""
    step = timeframe_delta(tf)
    for i in range(bars - window_bars + 1):
        yield start_ts + i * step


def _digest(df: pd.DataFrame) -> str:
    """Content hash of the exact bars used; stable across machines and rebuilds."""
    h = hashlib.sha256()
    h.update(df["ts"].to_numpy("datetime64[ns]").astype("int64").tobytes())
    cols = ["open", "high", "low", "close", "volume", "amount"]
    h.update(df[cols].to_numpy("float64").tobytes())
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def build(
    config_path: str | Path,
    root: Path = store.DEFAULT_ROOT,
    out_dir: Path = Path("data/datasets"),
    write: bool = True,
) -> dict:
    """Build the split/segment index and return the manifest (printing is the CLI's job).

    Raises on QA violations or overlapping splits -- a dirty corpus never becomes a
    dataset.
    """
    cfg = load_config(config_path)
    check_splits(cfg.splits)

    rows: list[dict] = []
    qa_rows: list[dict] = []
    digests: dict[str, str] = {}

    for tf in cfg.timeframes:
        for symbol in cfg.symbols:
            bars = store.read(symbol, tf, root=root, venue=cfg.venue)
            if bars.empty:
                raise FileNotFoundError(f"no bars for {symbol}/{tf} under {root}")
            qa_rows.append(qa.check_frame(bars, symbol, tf))

            for split in cfg.splits:
                lo, hi = split_bounds(cfg, split, tf)
                part = bars[(bars.ts >= lo) & (bars.ts <= hi)].reset_index(drop=True)
                if part.empty:
                    continue
                for start, stop in segments(part.ts, tf, cfg.window_bars):
                    seg = part.iloc[start:stop]
                    rows.append(
                        {
                            "symbol": symbol,
                            "tf": tf,
                            "split": split,
                            "start_ts": seg.ts.iloc[0],
                            "end_ts": seg.ts.iloc[-1],
                            "bars": len(seg),
                            "windows": len(seg) - cfg.window_bars + 1,
                        }
                    )
                    key = f"{tf}/{symbol}/{split}/{seg.ts.iloc[0]:%Y%m%dT%H%M}"
                    digests[key] = _digest(seg)

    report = pd.DataFrame(qa_rows)
    failures = qa.violations(report, cfg.qa_thresholds)
    if failures:
        raise ValueError("QA violations (build refused):\n  " + "\n  ".join(failures))

    index = pd.DataFrame(rows, columns=SEGMENT_COLUMNS).sort_values(
        ["tf", "symbol", "split", "start_ts"]
    )
    spec = cfg.spec()
    payload = json.dumps({"spec": spec, "digests": dict(sorted(digests.items()))}, sort_keys=True)
    dataset_hash = hashlib.sha256(payload.encode()).hexdigest()

    manifest = {
        "dataset": cfg.name,
        "dataset_hash": dataset_hash,
        "spec": spec,
        "window_bars": cfg.window_bars,
        "splits": {
            split: {
                "windows": int(index.loc[index.split == split, "windows"].sum()),
                "segments": int((index.split == split).sum()),
                "symbols": int(index.loc[index.split == split, "symbol"].nunique()),
            }
            for split in cfg.splits
        },
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
    }

    if write:
        dest = Path(out_dir) / cfg.name
        dest.mkdir(parents=True, exist_ok=True)
        index.to_parquet(dest / "segments.parquet", index=False)
        report.to_csv(dest / "qa_report.csv", index=False)
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest["path"] = str(dest)

    manifest["qa_report"] = report
    manifest["segments"] = index
    return manifest


__all__ = [
    "SEGMENT_COLUMNS",
    "DataConfig",
    "build",
    "check_splits",
    "expand",
    "load_config",
    "segments",
    "split_bounds",
]

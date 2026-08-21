"""The two corpus passes: fit the constants (Phase B), then measure what they produce (Phase E).

Both walk the v0.3 segment index rather than raw files — a segment is the unit the contract is
defined on, and reading a whole artifact would feed the transform bars that cleaning already
ruled out. Both are pure: they take bytes and return sketches. Who downloads the bytes is the
job's problem, not this module's.

The fit pass differs from every other pass in the project in one way that matters, and it is
enforced here rather than trusted: **it sees no bar at or after the firewall.** Trimming happens
on the bar sequence before a single feature is computed, and the job writes the resulting
`max(ts)` into the constants manifest so the claim is checkable after the fact (ADR-0021).
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from axiom.contract.spec import ContractSpec
from axiom.contract.stats import REPORT_QUANTILES, SketchSet
from axiom.contract.transform import ContractError, raw_features, transform

#: Where the fitted constants live once committed. A packaged config, so a cloud kernel with no
#: checkout loads it by bare name.
CONSTANTS_CONFIG = "contract_constants_v1"

#: Significant digits every emitted constant is rounded to. Not for precision — for determinism.
#: Two runs of the fit produce identical sketches and therefore identical floats, but a repr that
#: differs by an ulp between platforms would make the byte-identity check a platform check.
FLOAT_DIGITS = 12


@dataclass(frozen=True)
class SegmentRef:
    """One row of `clean/v1/segments.parquet`, as the contract passes need it."""

    segment_id: str
    artifact_path: str
    source: str
    market: str
    symbol: str
    asset_class: str
    frequency: str
    start_ts: int
    end_ts: int
    n_bars: int

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> SegmentRef:
        return cls(
            segment_id=row["segment_id"],
            artifact_path=row["artifact_path"],
            source=row["source"],
            market=row["market"],
            symbol=row["symbol"],
            asset_class=row["asset_class"],
            frequency=row["frequency"],
            start_ts=int(row["start_ts"]),
            end_ts=int(row["end_ts"]),
            n_bars=int(row["n_bars"]),
        )

    @property
    def series_key(self) -> str:
        """`SYMBOL-frequency`, the name the pinned regression snapshots are keyed by."""
        return f"{self.symbol}-{self.frequency}"


def read_segment_refs(data: bytes) -> list[SegmentRef]:
    table = pq.read_table(io.BytesIO(data))
    return [SegmentRef.from_row(row) for row in table.to_pylist()]


def group_by_artifact(refs: list[SegmentRef]) -> dict[str, list[SegmentRef]]:
    """One download per artifact, however many segments it holds. Order is stable."""
    grouped: dict[str, list[SegmentRef]] = {}
    for ref in refs:
        grouped.setdefault(ref.artifact_path, []).append(ref)
    for segments in grouped.values():
        segments.sort(key=lambda s: s.start_ts)
    return grouped


def slice_segment(bars: pa.Table, ref: SegmentRef, *, before_ts: int | None = None) -> pa.Table:
    """The bars of one segment, optionally truncated at a firewall.

    Half-open on the right when ``before_ts`` is given: a bar exactly at the firewall is
    post-firewall. Off-by-one here would put one bar of the sealed period into every fitted
    constant in the project, which is the sort of leak that is invisible for four versions.
    """
    ts = bars["ts"].to_numpy(zero_copy_only=False)
    keep = (ts >= ref.start_ts) & (ts <= ref.end_ts)
    if before_ts is not None:
        keep &= ts < before_ts
    return bars.filter(pa.array(keep))


def usable_windows(n_bars: int, context: int = 512) -> int:
    """Training windows a segment yields under the anchor-bar rule.

    The v0.3 table computed ``max(0, n_bars - 511)`` over bars. Bar 0 of every segment produces no
    feature row — it is consumed as the anchor — so the count is over ``n_bars - 1`` rows, and
    every segment in the corpus yields exactly one window fewer than v0.3 published.
    """
    return max(0, (n_bars - 1) - (context - 1))


# --- Phase B: fit the constants ----------------------------------------------------------


def fit_artifact(
    data: bytes,
    refs: list[SegmentRef],
    specs: list[ContractSpec],
    firewall_ts: int,
) -> tuple[SketchSet, list[str]]:
    """Sketch the raw feature distributions of one artifact's pre-firewall segments."""
    bars = pq.read_table(io.BytesIO(data))
    out = SketchSet()
    skipped: list[str] = []
    for ref in refs:
        segment = slice_segment(bars, ref, before_ts=firewall_ts)
        if segment.num_rows < 2:
            skipped.append(ref.segment_id)
            continue
        try:
            for spec in specs:
                raw, columns = raw_features(segment, spec)
                for i, name in enumerate(spec.feature_names):
                    out.sketch((spec.spec_id, ref.asset_class, ref.frequency, name)).add(raw[:, i])
                out.max_ts = max(out.max_ts, int(columns["ts"][-1]))
            out.segments += 1
            out.bars += segment.num_rows
        except ContractError as exc:
            skipped.append(f"{ref.segment_id}:{exc.code}")
    return out, skipped


def _round(value: float) -> float:
    return float(f"%.{FLOAT_DIGITS}g" % value)


def constants_tables(
    sketches: SketchSet, specs: list[ContractSpec]
) -> dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]]:
    """Turn sketches into the nested `spec -> class -> frequency -> feature` scaling table."""
    tables: dict[str, Any] = {}
    for (spec_id, asset_class, frequency, feature), sketch in sorted(sketches.sketches.items()):
        center, scale = sketch.center_scale()
        slot = tables.setdefault(spec_id, {}).setdefault(asset_class, {}).setdefault(frequency, {})
        slot[feature] = {"center": _round(center), "scale": _round(scale)}
    for spec in specs:
        for asset_class, frequencies in tables.get(spec.spec_id, {}).items():
            for frequency, features in frequencies.items():
                missing = [n for n in spec.feature_names if n not in features]
                if missing:
                    raise ValueError(
                        f"{spec.spec_id}/{asset_class}/{frequency} is missing {missing}: the fit "
                        "saw bars for this slice but produced no distribution for every feature"
                    )
    return tables


def constants_yaml(tables: dict[str, Any], manifest: dict[str, Any], schema_version: int) -> bytes:
    """The committed constants file. Byte-identical across runs with identical inputs."""
    header = (
        "# Frozen affine scaling constants for the v0.4 preprocessing contract (ADR-0020).\n"
        "#\n"
        "# Generated by `axiom contract fit-constants`, never edited by hand. `center` is a\n"
        "# robust median and `scale` an IQR/1.349, fitted over pre-firewall bars only -- the\n"
        "# manifest below records which bars, and the assertion that none of them was at or\n"
        "# after the firewall (ADR-0021).\n"
        "#\n"
        "# These constants are part of the contract. Refitting them is a schema_version bump,\n"
        "# new golden vectors and new regression snapshots, not a config edit.\n"
    )
    payload = {"schema_version": schema_version, "manifest": manifest, "tables": tables}
    body = yaml.safe_dump(payload, sort_keys=True, default_flow_style=False, width=100)
    return (header + body).encode("utf-8")


# --- Phase E: measure what the constants produce -----------------------------------------


@dataclass
class DryrunResult:
    """One artifact's contribution to the Phase E report."""

    sketches: SketchSet = field(default_factory=SketchSet)
    rows: int = 0
    seconds: float = 0.0
    n_nan: int = 0
    audits_run: int = 0
    audits_passed: int = 0
    failures: list[str] = field(default_factory=list)

    def merge(self, other: DryrunResult) -> DryrunResult:
        self.sketches.merge(other.sketches)
        self.rows += other.rows
        self.seconds += other.seconds
        self.n_nan += other.n_nan
        self.audits_run += other.audits_run
        self.audits_passed += other.audits_passed
        self.failures.extend(other.failures)
        return self

    @property
    def bars_per_second(self) -> float:
        return self.rows / self.seconds if self.seconds else 0.0

    def to_dict(self) -> dict:
        return {
            "sketches": self.sketches.to_dict(),
            "rows": self.rows,
            "seconds": self.seconds,
            "n_nan": self.n_nan,
            "audits_run": self.audits_run,
            "audits_passed": self.audits_passed,
            "failures": self.failures,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> DryrunResult:
        return cls(
            sketches=SketchSet.from_dict(payload["sketches"]),
            rows=int(payload["rows"]),
            seconds=float(payload["seconds"]),
            n_nan=int(payload["n_nan"]),
            audits_run=int(payload["audits_run"]),
            audits_passed=int(payload["audits_passed"]),
            failures=list(payload["failures"]),
        )


def prefix_audit(
    bars: pa.Table,
    spec: ContractSpec,
    constants,
    *,
    asset_class: str,
    frequency: str,
    splits: list[int],
) -> list[bool]:
    """Prefix-consistency on real bars: does a prefix's features prefix the whole's?

    The CI property battery runs this on synthetic series, which proves the arithmetic. Running it
    cloud-side on real segments proves the arithmetic *and* that no real pathology — a weekend
    gap, a stale run, a phase-shifted hour — routes around it. Bit-exact, not within tolerance:
    the contract has no reason to produce a different last digit for the same window.
    """
    full = transform(bars, spec, constants, asset_class=asset_class, frequency=frequency)
    results = []
    for split in splits:
        prefix = transform(
            bars.slice(0, split + 1), spec, constants, asset_class=asset_class, frequency=frequency
        )
        results.append(
            prefix.n_rows == split
            and bool(np.array_equal(prefix.values, full.values[:split]))
            and bool(np.array_equal(prefix.ts, full.ts[:split]))
        )
    return results


def dryrun_artifact(
    data: bytes,
    refs: list[SegmentRef],
    specs: list[ContractSpec],
    constants,
    *,
    audit_splits: dict[str, list[int]] | None = None,
    snapshot_series: tuple[str, ...] = (),
) -> tuple[DryrunResult, dict[str, str]]:
    """Stream one artifact's segments through every spec, keeping statistics and nothing else.

    Features are computed and discarded. v0.6 is what stores them; v0.4 only needs to know what
    they look like, and materializing forty million feature rows to find that out would be a
    corpus nobody asked for.
    """
    bars = pq.read_table(io.BytesIO(data))
    out = DryrunResult()
    snapshots: dict[str, str] = {}
    audit_splits = audit_splits or {}
    # One segment per pinned series gets hashed: the longest, so the snapshot covers the most
    # arithmetic, and deterministic so a rerun hashes the same rows.
    pinned = [r for r in refs if r.series_key in snapshot_series]
    pinned_id = max(pinned, key=lambda r: (r.n_bars, r.segment_id)).segment_id if pinned else None
    for ref in refs:
        segment = slice_segment(bars, ref)
        if segment.num_rows < 2:
            continue
        for spec in specs:
            started = time.perf_counter()
            try:
                block = transform(
                    segment,
                    spec,
                    constants,
                    asset_class=ref.asset_class,
                    frequency=ref.frequency,
                )
            except ContractError as exc:
                out.failures.append(f"{ref.segment_id}/{spec.spec_id}: {exc.code}")
                continue
            out.seconds += time.perf_counter() - started
            out.rows += block.n_rows
            for i, name in enumerate(spec.feature_names):
                key = (spec.spec_id, ref.asset_class, ref.frequency, name)
                out.sketches.sketch(key).add(block.values[:, i])
                out.sketches.clipped[key] = (
                    out.sketches.clipped.get(key, 0) + block.clip_counts[name]
                )
            out.sketches.segments += 1
            out.sketches.bars += segment.num_rows
            out.n_nan += int((~np.isfinite(block.values)).sum())
            if ref.segment_id == pinned_id:
                snapshots[f"{ref.series_key}/{spec.spec_id}"] = block_sha256(block)

            splits = audit_splits.get(ref.segment_id, [])
            if splits:
                passed = prefix_audit(
                    segment,
                    spec,
                    constants,
                    asset_class=ref.asset_class,
                    frequency=ref.frequency,
                    splits=splits,
                )
                out.audits_run += len(passed)
                out.audits_passed += sum(passed)
                if not all(passed):
                    out.failures.append(
                        f"{ref.segment_id}/{spec.spec_id}: prefix-consistency FAILED at splits "
                        f"{[s for s, ok in zip(splits, passed, strict=True) if not ok]}"
                    )
    return out, snapshots


def quantile_rows(result: DryrunResult) -> list[dict[str, Any]]:
    """The Phase E distribution table, one row per (spec, class, frequency, feature)."""
    rows = []
    for key, sketch in sorted(result.sketches.sketches.items()):
        spec_id, asset_class, frequency, feature = key
        total = sketch.total
        clipped = result.sketches.clipped.get(key, 0)
        row: dict[str, Any] = {
            "spec_id": spec_id,
            "asset_class": asset_class,
            "frequency": frequency,
            "feature": feature,
            "n": total,
            "clipped": clipped,
            "clip_rate": clipped / total if total else 0.0,
            "n_nan": sketch.n_nan,
        }
        for q in REPORT_QUANTILES:
            row[f"q{q:g}"] = sketch.quantile(q)
        rows.append(row)
    return rows


# --- pinned regression snapshots ---------------------------------------------------------

#: Five series, one per corpus slice, whose feature blocks are hashed and committed. Any future
#: change to the contract that moves a number moves these hashes, which is the cheapest possible
#: tripwire: no data leaves the cloud, and the diff is five lines.
PINNED_SERIES: tuple[str, ...] = (
    "BTCUSDT-1h",
    "EURUSD-1h",
    "AAPL-1d",
    "XAUUSD-1d",
    "ETHUSDT-1d",
)

SNAPSHOT_PATH = "tests/snapshots/contract_v1.json"


def block_sha256(block) -> str:
    """Content hash of a feature block: the array bytes, in C order, plus its shape.

    Over the float32 emission rather than the float64 compute, because the emission is what v0.6
    stores and v0.9 reads. A change the cast swallows is a change nothing downstream can see.
    """
    import hashlib

    digest = hashlib.sha256()
    digest.update(f"{block.spec_id}|{block.n_rows}|{len(block.feature_names)}|".encode())
    digest.update(np.ascontiguousarray(block.values, dtype=np.float32).tobytes())
    return digest.hexdigest()


def pick_audit_segments(
    refs: list[SegmentRef], *, n_segments: int = 50, n_splits: int = 3, seed: int = 20260404
) -> dict[str, list[int]]:
    """Choose the segments and split points for the corpus-level prefix-consistency audit.

    Seeded, so a rerun audits the same segments and a failure can be reproduced by name. Only
    segments long enough to have interesting split points are eligible — auditing a 130-bar
    segment would never exercise the rolling phase of the 256-bar median window.
    """
    import random

    rng = random.Random(seed)
    eligible = sorted((r for r in refs if r.n_bars >= 600), key=lambda r: r.segment_id)
    chosen = rng.sample(eligible, min(n_segments, len(eligible)))
    return {
        ref.segment_id: sorted(rng.sample(range(1, ref.n_bars - 1), min(n_splits, ref.n_bars - 2)))
        for ref in chosen
    }


# --- drivers ------------------------------------------------------------------------------


def _fan_out(grouped, read, work, concurrency: int, log=None):
    """Run ``work(bytes, refs)`` over every artifact, in threads, yielding results as they land.

    Threads rather than processes: every hot loop below is numpy, which drops the GIL, and a
    process pool would pay to pickle a bar table per artifact.

    **Submission is bounded, and that is not a tuning detail.** Each result carries up to twelve
    quantile sketches, which is about 6 MB. Submitting all ten thousand artifacts up front means
    every completed result stays alive inside its `Future` until the consumer reaches it, and the
    consumer cannot keep up with the pool — so the run climbs to tens of gigabytes and the runner
    kills it, which is exactly what happened the first time this ran over the whole corpus. Keeping
    a few times `concurrency` in flight and dropping each future as it is consumed holds the
    footprint flat.
    """
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    def one(path, refs):
        data = read(path)
        if data is None:
            return path, None, f"{path}: not found"
        try:
            return path, work(data, refs), ""
        except Exception as exc:  # a bad file must not take the corpus down
            return path, None, f"{path}: {type(exc).__name__}: {exc}"

    total = len(grouped)
    in_flight = max(2, concurrency * 3)
    done = 0

    def drain(finished):
        nonlocal done
        for future in finished:
            done += 1
            if log and done % 1000 == 0:
                log(f"  {done}/{total} artifacts")
            yield future.result()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending: set = set()
        for path, refs in grouped.items():
            pending.add(pool.submit(one, path, refs))
            if len(pending) >= in_flight:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                yield from drain(finished)
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            yield from drain(finished)


@dataclass
class FitRun:
    sketches: SketchSet = field(default_factory=SketchSet)
    ok: int = 0
    failed: int = 0
    failures: list[str] = field(default_factory=list)
    skipped_segments: int = 0

    def line(self) -> str:
        return (
            f"fit: {self.ok} artifact(s), {self.sketches.segments} segments, "
            f"{self.sketches.bars} bars, {self.skipped_segments} segments skipped, "
            f"{self.failed} failed"
        )


def fit_corpus(
    grouped: dict[str, list[SegmentRef]],
    read,
    specs: list[ContractSpec],
    firewall_ts: int,
    *,
    concurrency: int = 16,
    log=None,
) -> FitRun:
    run = FitRun()
    work = lambda data, refs: fit_artifact(data, refs, specs, firewall_ts)  # noqa: E731
    for _path, result, error in _fan_out(grouped, read, work, concurrency, log):
        if error:
            run.failed += 1
            run.failures.append(error)
            continue
        sketches, skipped = result
        run.sketches.merge(sketches)
        run.skipped_segments += len(skipped)
        run.ok += 1
    return run


def dryrun_corpus(
    grouped: dict[str, list[SegmentRef]],
    read,
    specs: list[ContractSpec],
    constants,
    *,
    audit_splits: dict[str, list[int]] | None = None,
    snapshot_series: tuple[str, ...] = PINNED_SERIES,
    concurrency: int = 16,
    log=None,
) -> tuple[DryrunResult, dict[str, str], list[str]]:
    """Stream the whole corpus through both specs. Returns stats, snapshot hashes, failures."""
    total = DryrunResult()
    snapshots: dict[str, str] = {}
    failures: list[str] = []
    work = lambda data, refs: dryrun_artifact(  # noqa: E731
        data, refs, specs, constants, audit_splits=audit_splits, snapshot_series=snapshot_series
    )
    for _path, result, error in _fan_out(grouped, read, work, concurrency, log):
        if error:
            failures.append(error)
            continue
        partial, artifact_snapshots = result
        total.merge(partial)
        snapshots.update(artifact_snapshots)
    return total, snapshots, failures

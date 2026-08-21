"""Bars in, six causal features out. The whole contract is this function and its inverse.

The rule the module is built around: ``transform`` is a **pure function of the bars it was handed
and the frozen constants**. Not of the corpus, not of the file the bars came from, not of anything
fitted at call time. Everything else in v0.4 -- the frozen constants, the strictly-past median, the
refusal to fill a NaN -- is that rule applied somewhere specific.

The rule buys one testable property, and ADR-0020 calls it the definition of causal:
``transform(bars[:t+1])`` is exactly the first ``t`` rows of ``transform(bars)``. Not close to.
Exactly, bit for bit, on the same platform.

Pure in the v0.3 sense as well: no file is read, no socket is opened, no module-level state is
touched. A test enforces it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from axiom.contract.rolling import strictly_past_median
from axiom.contract.spec import ContractConstants, ContractSpec
from axiom.schema.bars import BARS_SCHEMA_V1, OHLCVA

#: What every feature row is computed in. Emission is float32 (ADR-0020): the model reads half
#: precision anyway, and a float64 corpus would double the shard bill for digits nothing uses.
COMPUTE_DTYPE = np.float64
EMIT_DTYPE = np.float32


class ContractError(ValueError):
    """A refusal, with a code the caller can branch on.

    The contract never fills, coerces or drops. Every failure in here is the caller handing it
    something that is not a valid bar sequence, and the only honest response is to stop: a
    silently repaired input is a silently wrong feature block, and it surfaces four versions
    later as an evaluation number nobody can explain.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class AnchorBar:
    """Bar 0 of a segment, which produces no feature row of its own.

    It is consumed instead: ``close`` seeds the first gap feature and the Predictor's price
    inversion, and ``volume``/``amount`` seed the first strictly-past median. Everything needed
    to walk the features back to prices is here, which is what makes :func:`inverse` exact.
    """

    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float

    @classmethod
    def from_table(cls, bars: pa.Table, row: int = 0) -> AnchorBar:
        return cls(
            ts=int(bars["ts"][row].as_py()),
            open=float(bars["open"][row].as_py()),
            high=float(bars["high"][row].as_py()),
            low=float(bars["low"][row].as_py()),
            close=float(bars["close"][row].as_py()),
            volume=float(bars["volume"][row].as_py()),
            amount=float(bars["amount"][row].as_py()),
        )


@dataclass(frozen=True)
class FeatureBlock:
    """The output of one transform: the array, and everything needed to interpret or undo it."""

    values: np.ndarray
    feature_names: tuple[str, ...]
    ts: np.ndarray
    anchor: AnchorBar
    spec_id: str
    schema_version: int
    asset_class: str
    frequency: str
    spec_hash: str
    constants_hash: str
    #: Per feature, how many values the clip actually moved. A rate above 0.5 % in any slice is
    #: a red flag the Phase E report is required to investigate rather than note.
    clip_counts: dict[str, int]

    @property
    def n_rows(self) -> int:
        return int(self.values.shape[0])

    def column(self, name: str) -> np.ndarray:
        return self.values[:, self.feature_names.index(name)]


def _validate(bars: pa.Table) -> dict[str, np.ndarray]:
    """Every reason to refuse, checked before a single feature is computed."""
    missing = [name for name in ("ts", *OHLCVA) if name not in bars.column_names]
    if missing:
        raise ContractError("missing_column", f"bars are missing columns {missing}")
    for name in ("ts", *OHLCVA):
        expected = BARS_SCHEMA_V1.field(name).type
        actual = bars.schema.field(name).type
        if actual != expected:
            raise ContractError("bad_dtype", f"column {name!r} is {actual}, expected {expected}")

    if bars.num_rows < 2:
        raise ContractError(
            "too_short",
            f"a segment needs an anchor bar and at least one feature bar; got {bars.num_rows}",
        )

    columns = {
        name: np.asarray(
            bars[name].to_numpy(zero_copy_only=False),
            dtype=np.int64 if name == "ts" else np.float64,
        )
        for name in ("ts", *OHLCVA)
    }

    for name in OHLCVA:
        if not np.isfinite(columns[name]).all():
            bad = int(np.flatnonzero(~np.isfinite(columns[name]))[0])
            raise ContractError("non_finite", f"{name} is NaN or Inf at row {bad}")

    for name in ("open", "high", "low", "close"):
        if (columns[name] <= 0.0).any():
            bad = int(np.flatnonzero(columns[name] <= 0.0)[0])
            raise ContractError(
                "non_positive_price",
                f"{name} is {columns[name][bad]} at row {bad}; the contract is built on logs of "
                "prices and has no meaning at or below zero",
            )
    for name in ("volume", "amount"):
        if (columns[name] < 0.0).any():
            bad = int(np.flatnonzero(columns[name] < 0.0)[0])
            raise ContractError("negative_flow", f"{name} is negative at row {bad}")

    ts = columns["ts"]
    if (np.diff(ts) <= 0).any():
        bad = int(np.flatnonzero(np.diff(ts) <= 0)[0]) + 1
        raise ContractError(
            "ts_not_increasing",
            f"ts does not strictly increase at row {bad}: {ts[bad - 1]} -> {ts[bad]}",
        )

    open_, high, low, close = (columns[k] for k in ("open", "high", "low", "close"))
    if (high < np.maximum(open_, close)).any() or (low > np.minimum(open_, close)).any():
        bad = int(
            np.flatnonzero((high < np.maximum(open_, close)) | (low > np.minimum(open_, close)))[0]
        )
        raise ContractError(
            "ohlc_inconsistent",
            f"row {bad} has high or low outside the open/close range; the structural identities "
            "the geometry parameterization encodes do not hold for it",
        )
    return columns


def _price_features(spec: ContractSpec, columns: dict[str, np.ndarray]) -> list[np.ndarray]:
    """The four price features, for rows 1..n-1, in the spec's declared order."""
    open_, high, low, close = (columns[k] for k in ("open", "high", "low", "close"))
    prev_close = close[:-1]
    if spec.parameterization == "geo":
        return [
            np.log(open_[1:] / prev_close),  # gap
            np.log(close[1:] / open_[1:]),  # body
            np.log(high[1:] / open_[1:]),  # upper wick
            np.log(low[1:] / open_[1:]),  # lower wick
        ]
    return [
        np.log(open_[1:] / prev_close),
        np.log(high[1:] / prev_close),
        np.log(low[1:] / prev_close),
        np.log(close[1:] / prev_close),
    ]


def _flow_feature(values: np.ndarray, window: int) -> np.ndarray:
    """``log1p(x_t)`` minus the median of ``log1p(x)`` over the bars strictly before t.

    Relative to its own past, so a series that trades in units of one and a series that trades in
    units of a million land on the same axis without either of them needing a per-series constant
    fitted from data the contract is not allowed to see.
    """
    logged = np.log1p(values)
    return (logged - strictly_past_median(logged, window))[1:]


def raw_features(bars: pa.Table, spec: ContractSpec) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """The six features before any scaling, plus the validated columns they came from.

    Split out because the Phase B constants job needs exactly this: the distribution the scaling
    is about to be fitted to, which by definition does not exist yet when the job runs. Nothing
    else may call it — a consumer that scales features itself is the second implementation
    ADR-0020 exists to forbid.
    """
    columns = _validate(bars)
    raw = np.empty((bars.num_rows - 1, len(spec.feature_names)), dtype=COMPUTE_DTYPE)
    for i, values in enumerate(_price_features(spec, columns)):
        raw[:, i] = values
    raw[:, 4] = _flow_feature(columns["volume"], spec.volume_window)
    raw[:, 5] = _flow_feature(columns["amount"], spec.volume_window)
    return raw, columns


def transform(
    bars: pa.Table,
    spec: ContractSpec,
    constants: ContractConstants | None,
    *,
    asset_class: str,
    frequency: str,
    allow_leaky: bool = False,
) -> FeatureBlock:
    """Turn one segment's bars into its feature block.

    ``bars`` is a contiguous segment as the v0.3 index defines one: strictly increasing ``ts``,
    every bar real. Gaps in the timestamp grid are *not* rejected — a 24x5 series legitimately
    skips a weekend, and the clean layer already adjudicated which absences are boundaries and
    which are sessions (ADR-0018). Re-litigating that here would reject every FX segment in the
    corpus.

    ``asset_class`` and ``frequency`` select the frozen scaling row. They are not columns of the
    bar schema — identity lives in the path and the manifest (ADR-0010) — so the caller passes
    them, and passing the wrong one is visible as a clip rate that jumps.
    """
    if spec.leaky and not allow_leaky:
        raise ContractError(
            "leaky_spec",
            f"spec {spec.spec_id!r} normalizes against the window it is normalizing and fails "
            "the causality audit by construction. Pass allow_leaky=True only from a study that "
            "means to measure the leak (ADR-0020).",
        )
    raw, columns = raw_features(bars, spec)

    if spec.leaky:
        # Kronos's normalization, reproduced exactly: the mean and standard deviation of the
        # window being normalized. Every row of the output knows about every other row.
        centers = raw.mean(axis=0)
        scales = raw.std(axis=0)
        scales[scales == 0.0] = 1.0
    elif constants is None:
        raise ContractError("missing_constants", f"spec {spec.spec_id!r} needs a constants table")
    else:
        scaling = constants.scaling_for(spec, asset_class, frequency)
        centers = np.array([s.center for s in scaling], dtype=COMPUTE_DTYPE)
        scales = np.array([s.scale for s in scaling], dtype=COMPUTE_DTYPE)

    scaled = (raw - centers) / scales
    clipped_mask = (scaled < spec.clip_low) | (scaled > spec.clip_high)
    clip_counts = {name: int(clipped_mask[:, i].sum()) for i, name in enumerate(spec.feature_names)}
    np.clip(scaled, spec.clip_low, spec.clip_high, out=scaled)

    if not np.isfinite(scaled).all():
        raise ContractError(
            "non_finite_output",
            "the contract produced a NaN or Inf from a valid input, which is a bug in the "
            "contract and not a property of the data",
        )

    return FeatureBlock(
        values=scaled.astype(EMIT_DTYPE),
        feature_names=spec.feature_names,
        ts=columns["ts"][1:].copy(),
        anchor=AnchorBar.from_table(bars),
        spec_id=spec.spec_id,
        schema_version=spec.schema_version,
        asset_class=asset_class,
        frequency=frequency,
        spec_hash=spec.config_hash,
        constants_hash="" if constants is None else constants.config_hash,
        clip_counts=clip_counts,
    )

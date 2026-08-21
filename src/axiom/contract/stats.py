"""Mergeable quantile sketches, and the two corpus passes built on them.

Both cloud jobs in v0.4 ask the same question of forty million bars — where do these feature
values actually sit — and neither can hold the answer in memory. Phase B wants a median and an
IQR to freeze as scaling constants; Phase E wants nine quantiles and a clip rate to publish. So
there is one sketch here and both jobs use it.

A histogram, not a sample. The reason is mergeability: a worker's histogram plus another worker's
histogram is the histogram of both, exactly, with no ordering to preserve and no random seed to
thread through a `.map()`.

**Bins are uniform in `asinh(x / 1e-4)`, not in `x`.** A uniform grid over raw feature units
cannot do this job: an hourly crypto body has an interquartile range around 10⁻³ and a flow
feature reaches ±20, so any single bin width is either too coarse to resolve the IQR the scaling
constants are fitted from or too fine to span the support. `asinh` is linear near zero and
logarithmic far from it, which buys a constant **relative** resolution of about 0.05 % everywhere
outside ±10⁻⁴ while covering ±163 in 60 000 bins. Monotone, so quantiles map through it exactly:
take the quantile in bin space, `sinh` it back.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field

import numpy as np

#: Where `asinh` turns over from linear to logarithmic. Below this the sketch resolves absolutely,
#: above it proportionally. Set two orders of magnitude under the smallest IQR any real slice is
#: expected to have.
ASINH_SCALE = 1e-4

#: The support, in bin space. `sinh(15) * 1e-4` is about 163 raw units, comfortably past the ±20
#: a flow feature reaches on a series that went from dust to billions. Values outside are counted
#: at the edges rather than dropped, and a quantile landing in an overflow bucket returns the edge.
BIN_LOW = -15.0
BIN_HIGH = 15.0
N_BINS = 60_000
BIN_WIDTH = (BIN_HIGH - BIN_LOW) / N_BINS


def to_bin_space(values: np.ndarray) -> np.ndarray:
    return np.arcsinh(np.asarray(values, dtype=np.float64) / ASINH_SCALE)


def from_bin_space(value: float) -> float:
    return float(np.sinh(value) * ASINH_SCALE)


#: The quantiles the Phase E report publishes.
REPORT_QUANTILES = (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999)

#: IQR of a normal distribution, in standard deviations. Dividing by it makes `scale` a
#: standard-deviation estimate that a fat tail cannot drag around.
IQR_TO_SIGMA = 1.349


@dataclass
class Sketch:
    """One feature's distribution, as counts per bin plus the two overflow buckets."""

    counts: np.ndarray = field(default_factory=lambda: np.zeros(N_BINS, dtype=np.int64))
    under: int = 0
    over: int = 0
    n_nan: int = 0

    @property
    def total(self) -> int:
        return int(self.counts.sum()) + self.under + self.over

    def add(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).ravel()
        finite = np.isfinite(values)
        self.n_nan += int((~finite).sum())
        values = values[finite]
        if values.size == 0:
            return
        idx = np.floor((to_bin_space(values) - BIN_LOW) / BIN_WIDTH).astype(np.int64)
        self.under += int((idx < 0).sum())
        self.over += int((idx >= N_BINS).sum())
        inside = idx[(idx >= 0) & (idx < N_BINS)]
        if inside.size:
            self.counts += np.bincount(inside, minlength=N_BINS).astype(np.int64)

    def merge(self, other: Sketch) -> Sketch:
        self.counts += other.counts
        self.under += other.under
        self.over += other.over
        self.n_nan += other.n_nan
        return self

    def _quantile_bin(self, q: float) -> float:
        """Where ``q`` of the mass falls, in bin space, interpolated inside its bin.

        Overflow buckets are counted in the mass, so a quantile that falls in one returns the
        support edge. That is visible in the report as a value pinned at ±163, which is the right
        way for a distribution the sketch cannot see to announce itself.
        """
        total = self.total
        if total == 0:
            return float("nan")
        target = q * total
        if target <= self.under:
            return BIN_LOW
        cumulative = np.cumsum(self.counts) + self.under
        if target > cumulative[-1]:
            return BIN_HIGH
        b = int(np.searchsorted(cumulative, target, side="left"))
        before = float(cumulative[b - 1]) if b else float(self.under)
        in_bin = float(self.counts[b])
        frac = 0.0 if in_bin == 0 else (target - before) / in_bin
        return BIN_LOW + (b + frac) * BIN_WIDTH

    def quantile(self, q: float) -> float:
        """The feature value below which ``q`` of the mass sits, in raw units."""
        return from_bin_space(self._quantile_bin(q))

    def _spread(self, lo: float, hi: float) -> float | None:
        """The distance between two quantiles, or ``None`` if the sketch cannot resolve it.

        Both quantiles landing in the same bin is the case to catch. Interpolating inside a bin
        would still return a positive number — a fraction of the bin width — and that number is
        an artifact of the bin, not a measurement of the data. Dividing a whole asset class's
        features by it would be worse than failing.
        """
        low_bin = int((self._quantile_bin(lo) - BIN_LOW) / BIN_WIDTH)
        high_bin = int((self._quantile_bin(hi) - BIN_LOW) / BIN_WIDTH)
        if low_bin == high_bin:
            return None
        return self.quantile(hi) - self.quantile(lo)

    def center_scale(self) -> tuple[float, float]:
        """Robust location and spread: the median, and IQR/1.349.

        An unresolvable IQR is not a fit failure to paper over with an epsilon — it means half the
        mass of a whole asset class sits inside one bin, which for a gap feature would mean the
        class opens within a hundredth of a percent of where it closed more often than not.
        Widen to the 10-90 range once, then refuse.
        """
        center = self.quantile(0.5)
        iqr = self._spread(0.25, 0.75)
        scale = iqr / IQR_TO_SIGMA if iqr is not None else None
        if scale is None:
            decile = self._spread(0.1, 0.9)
            scale = decile / 2.563 if decile is not None else None
        if scale is None or scale <= 0.0 or not np.isfinite(scale):
            raise ValueError(
                "degenerate distribution: the 10-90 range does not span a single histogram bin, "
                f"so no affine scaling can be fitted to it (n={self.total}, median={center})"
            )
        return center, scale

    def to_dict(self) -> dict:
        """Transport form. Counts go over as base64 so a `.map()` result stays small."""
        return {
            "counts": base64.b64encode(self.counts.tobytes()).decode("ascii"),
            "under": self.under,
            "over": self.over,
            "n_nan": self.n_nan,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Sketch:
        return cls(
            counts=np.frombuffer(base64.b64decode(payload["counts"]), dtype=np.int64).copy(),
            under=int(payload["under"]),
            over=int(payload["over"]),
            n_nan=int(payload["n_nan"]),
        )


@dataclass
class SketchSet:
    """Sketches keyed ``(spec_id, asset_class, frequency, feature)``, plus what was consumed."""

    sketches: dict[tuple[str, str, str, str], Sketch] = field(default_factory=dict)
    clipped: dict[tuple[str, str, str, str], int] = field(default_factory=dict)
    segments: int = 0
    bars: int = 0
    max_ts: int = 0

    def sketch(self, key: tuple[str, str, str, str]) -> Sketch:
        return self.sketches.setdefault(key, Sketch())

    def merge(self, other: SketchSet) -> SketchSet:
        for key, sketch in other.sketches.items():
            self.sketch(key).merge(sketch)
        for key, count in other.clipped.items():
            self.clipped[key] = self.clipped.get(key, 0) + count
        self.segments += other.segments
        self.bars += other.bars
        self.max_ts = max(self.max_ts, other.max_ts)
        return self

    def to_dict(self) -> dict:
        return {
            "sketches": [
                {"key": list(key), **sketch.to_dict()} for key, sketch in self.sketches.items()
            ],
            "clipped": [{"key": list(key), "n": n} for key, n in self.clipped.items()],
            "segments": self.segments,
            "bars": self.bars,
            "max_ts": self.max_ts,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> SketchSet:
        return cls(
            sketches={tuple(item["key"]): Sketch.from_dict(item) for item in payload["sketches"]},
            clipped={tuple(item["key"]): int(item["n"]) for item in payload["clipped"]},
            segments=int(payload["segments"]),
            bars=int(payload["bars"]),
            max_ts=int(payload["max_ts"]),
        )

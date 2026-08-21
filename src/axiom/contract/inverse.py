"""Features back to bars. The Predictor's other half.

v0.9 samples feature rows out of the model and has to turn them into a candle chart, which means
the contract has to be invertible in the arithmetic sense: every step in :mod:`axiom.contract
.transform` is a log, an affine, or a subtraction of a value the inverse can rebuild from what it
already knows. Nothing here is fitted or approximated.

Three things do not come back exactly, and all three are honest limits rather than bugs:

- **Clipping is not invertible.** A feature that hit the bound inverts to the bound. The round-trip
  property is therefore asserted on inputs whose features land inside the clip range, and a run
  that clips is a run whose reconstruction is a projection.
- **The optional columns** — ``n_trades`` and the taker splits — never entered the contract, so
  they come back null. The bar schema allows that; the six-feature contract is what v0.5 onwards
  actually consumes.
- **Float32 emission bounds the error in log space, not in price space.** Prices come back to
  about 1e-5 relative. Volume and amount come back to about 1e-4 *of their log1p*, which on a
  zero-volume bar following a very large one reads as 1e-6 units of volume rather than 0. That is
  one ulp of a float32 log, not a defect in the algebra.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from axiom.contract.spec import ContractConstants, ContractSpec
from axiom.contract.transform import AnchorBar, ContractError, FeatureBlock
from axiom.schema.bars import BARS_SCHEMA_V1


def _unscale(
    features: np.ndarray,
    spec: ContractSpec,
    constants: ContractConstants,
    asset_class: str,
    frequency: str,
) -> np.ndarray:
    scaling = constants.scaling_for(spec, asset_class, frequency)
    centers = np.array([s.center for s in scaling], dtype=np.float64)
    scales = np.array([s.scale for s in scaling], dtype=np.float64)
    return np.asarray(features, dtype=np.float64) * scales + centers


def _prices(spec: ContractSpec, raw: np.ndarray, anchor_close: float) -> dict[str, np.ndarray]:
    """Walk the price features forward from the anchor close.

    Both parameterizations are a cumulative sum in log space, so neither needs a loop: the close
    path is fixed first, and every other price on a bar is that bar's reference close times an
    exponent.
    """
    log_anchor = np.log(anchor_close)
    prices = _walk(spec, raw, log_anchor)
    # Project onto the set of valid candles. Both parameterizations guarantee
    # `high >= max(open, close) >= min(open, close) >= low` *analytically* — it follows from the
    # feature identities — but the features arrive as float32, and rounding a wick of exactly
    # zero can put `high` an ulp under `open`. That reconstructs a bar the schema refuses over a
    # rounding artifact of our own emission.
    #
    # The clamp also matters for v0.9 in a way rounding does not: a model samples feature rows
    # freely, and nothing stops it sampling an upper wick below the body. Projecting is the
    # honest response — a candle is what comes out either way, and the alternative is refusing to
    # draw the chart.
    body_high = np.maximum(prices["open"], prices["close"])
    body_low = np.minimum(prices["open"], prices["close"])
    high = np.maximum(prices["high"], body_high)
    low = np.minimum(prices["low"], body_low)
    return {"open": prices["open"], "high": high, "low": low, "close": prices["close"]}


def _walk(spec: ContractSpec, raw: np.ndarray, log_anchor: float) -> dict[str, np.ndarray]:
    """The unprojected price path. Cumulative sums in log space, so neither spec needs a loop."""
    if spec.parameterization == "geo":
        gap, body, upper, lower = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
        log_close = log_anchor + np.cumsum(gap + body)
        log_prev_close = np.concatenate(([log_anchor], log_close[:-1]))
        log_open = log_prev_close + gap
        return {
            "open": np.exp(log_open),
            "high": np.exp(log_open + upper),
            "low": np.exp(log_open + lower),
            "close": np.exp(log_close),
        }
    r_open, r_high, r_low, r_close = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
    log_close = log_anchor + np.cumsum(r_close)
    log_prev_close = np.concatenate(([log_anchor], log_close[:-1]))
    return {
        "open": np.exp(log_prev_close + r_open),
        "high": np.exp(log_prev_close + r_high),
        "low": np.exp(log_prev_close + r_low),
        "close": np.exp(log_close),
    }


def _flow(raw: np.ndarray, anchor_value: float, window: int) -> np.ndarray:
    """Undo ``log1p(x_t) - median(past)`` one bar at a time.

    Sequential by nature: bar t's median is taken over reconstructed values, so bar t cannot be
    recovered before bar t-1 exists. At the context lengths this runs on — 512 at inference, a
    segment in a test — the loop is not worth removing.

    ponytail: O(n·window) Python loop. If v0.9 ever inverts a whole corpus rather than a context,
    swap the median for an incremental order-statistic structure.
    """
    logged = np.empty(raw.size + 1, dtype=np.float64)
    logged[0] = np.log1p(anchor_value)
    for t in range(1, logged.size):
        logged[t] = raw[t - 1] + np.median(logged[max(0, t - window) : t])
    # A traded quantity is not negative. `log1p` of an exactly-zero-volume bar round-trips
    # through float32 to a value a hair under zero, and `expm1` faithfully returns -1.5e-10 --
    # which the bar schema refuses, correctly.
    return np.maximum(np.expm1(logged[1:]), 0.0)


def inverse(
    features: np.ndarray | FeatureBlock,
    spec: ContractSpec,
    constants: ContractConstants,
    *,
    anchor: AnchorBar | None = None,
    ts: np.ndarray | None = None,
    asset_class: str = "",
    frequency: str = "",
) -> pa.Table:
    """Rebuild the bar table a feature block came from, anchor row included.

    The output has ``len(ts) + 1`` rows: the anchor bar verbatim, then one reconstructed bar per
    feature row. Passing a :class:`~axiom.contract.transform.FeatureBlock` supplies ``anchor``,
    ``ts``, ``asset_class`` and ``frequency`` from the block itself.
    """
    if isinstance(features, FeatureBlock):
        anchor, ts = features.anchor, features.ts
        asset_class, frequency = features.asset_class, features.frequency
        features = features.values
    if anchor is None or ts is None:
        raise ContractError(
            "missing_anchor",
            "inverting a bare array needs the anchor bar and the row timestamps; pass the "
            "FeatureBlock instead if you still have it",
        )
    if spec.leaky:
        raise ContractError(
            "leaky_spec",
            f"spec {spec.spec_id!r} scales against per-window statistics that are not carried "
            "with the features, so there is nothing to invert against",
        )
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(spec.feature_names):
        raise ContractError(
            "bad_shape",
            f"expected an [n, {len(spec.feature_names)}] block, got {values.shape}",
        )
    ts = np.asarray(ts, dtype=np.int64)
    if ts.size != values.shape[0]:
        raise ContractError("bad_shape", f"{ts.size} timestamps for {values.shape[0]} feature rows")

    raw = _unscale(values, spec, constants, asset_class, frequency)
    prices = _prices(spec, raw, anchor.close)
    volume = _flow(raw[:, 4], anchor.volume, spec.volume_window)
    amount = _flow(raw[:, 5], anchor.amount, spec.volume_window)

    n = ts.size + 1
    columns = {
        "ts": np.concatenate(([anchor.ts], ts)).astype(np.int64),
        "open": np.concatenate(([anchor.open], prices["open"])),
        "high": np.concatenate(([anchor.high], prices["high"])),
        "low": np.concatenate(([anchor.low], prices["low"])),
        "close": np.concatenate(([anchor.close], prices["close"])),
        "volume": np.concatenate(([anchor.volume], volume)),
        "amount": np.concatenate(([anchor.amount], amount)),
    }
    return pa.table(
        {
            **{
                name: pa.array(values_, type=BARS_SCHEMA_V1.field(name).type)
                for name, values_ in columns.items()
            },
            "n_trades": pa.nulls(n, pa.int64()),
            "taker_buy_volume": pa.nulls(n, pa.float64()),
            "taker_buy_quote_volume": pa.nulls(n, pa.float64()),
        },
        schema=BARS_SCHEMA_V1,
    )

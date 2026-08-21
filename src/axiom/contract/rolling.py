"""The strictly-past median: the one place in the contract where a feature sees more than its bar.

Every causality argument in v0.4 comes down to this function's window. The value at row ``t`` is
the median of ``a[max(0, t - window) : t]`` — a half-open interval that **ends before t**. Include
``t`` and prefix-consistency still passes at most split points and fails at some, which is the
worst possible failure mode for a leak: intermittent.

The window expands from the segment start until it reaches ``window`` bars, then rolls. There is
no prior blending and no cold-start constant: the expanding phase *is* the warm-up, and it is a
phase the model also sees in training, because training segments start too (ADR-0020).
"""

from __future__ import annotations

import numpy as np

#: Rows per block when taking medians over materialized sliding windows. At window 256 this is
#: about 8 MB of float64 in flight, which keeps a 30-million-bar streaming job off the swap.
_BLOCK = 4096


def strictly_past_median(a: np.ndarray, window: int) -> np.ndarray:
    """Median over the ``window`` values before each index, expanding at the start.

    Returns an array the same length as ``a``. Index 0 is ``nan``: no bar precedes the first one,
    and returning a number there would invent a statistic. Callers only ever read indices >= 1,
    because feature rows only exist from bar 1 (the anchor-bar rule).

    Each output depends on a contiguous slice of ``a`` that ends strictly before its own index, so
    the result over a prefix of ``a`` is a prefix of the result over all of ``a``, exactly. That
    is the property the whole contract rests on, and it is why the implementation may not do
    anything clever with running state that a prefix would not have built the same way.
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    a = np.asarray(a, dtype=np.float64)
    n = a.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return out

    # Expanding phase: indices 1..min(window, n) - 1 see fewer than `window` predecessors.
    # At most `window - 1` medians over short slices, so a loop costs nothing measurable.
    for t in range(1, min(window, n)):
        out[t] = np.median(a[:t])

    if n > window:
        # Rolling phase. `windows[i]` is `a[i : i + window]`, which is the window for row
        # `i + window`. Blocked so the materialized view stays small on long series.
        windows = np.lib.stride_tricks.sliding_window_view(a, window)
        rows = n - window
        for start in range(0, rows, _BLOCK):
            stop = min(start + _BLOCK, rows)
            out[window + start : window + stop] = np.median(windows[start:stop], axis=1)
    return out


def reference_past_median(a: np.ndarray, window: int) -> np.ndarray:
    """The obvious O(n·window) version, written to be read rather than run.

    Ships in the library instead of a test module because it is the definition
    :func:`strictly_past_median` is checked against, and a definition that lives in one test file
    is a definition the next version reimplements slightly differently.
    """
    a = np.asarray(a, dtype=np.float64)
    out = np.full(a.size, np.nan, dtype=np.float64)
    for t in range(1, a.size):
        out[t] = np.median(a[max(0, t - window) : t])
    return out

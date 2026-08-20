"""The canonical bar schema. Pure data shape and validation — no I/O, no network."""

from axiom.schema.bars import (
    BARS_SCHEMA_V1,
    BARS_SCHEMA_VERSION,
    FREQUENCIES,
    ROW_GROUP_SIZE,
    ValidationReport,
    Violation,
    bars_metadata,
    count_gaps,
    count_off_grid,
    grid_step_ms,
    normalize_ts_ms,
    validate_bars,
)

__all__ = [
    "BARS_SCHEMA_V1",
    "BARS_SCHEMA_VERSION",
    "FREQUENCIES",
    "ROW_GROUP_SIZE",
    "ValidationReport",
    "Violation",
    "bars_metadata",
    "count_gaps",
    "count_off_grid",
    "grid_step_ms",
    "normalize_ts_ms",
    "validate_bars",
]

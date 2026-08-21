"""The cleaning config: Kronos Table 4 thresholds, stage order, and session gap rules.

Everything here is data. The rules that consume it live in :mod:`axiom.clean.stages`, and the
semantics behind every field are pinned in ADR-0018.

The config hash is the identity of a clean run. It is stamped into every segment row, and a
segment carrying a hash other than the current one is not valid for the current config -- which
is what lets thresholds be tuned without a corpus rewrite.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from axiom.config.hashing import SHORT_LEN, canonical_json
from axiom.config.settings import resolve_config_path

#: The five stages of Algorithm 1, in the only order this config is allowed to declare.
#:
#: Partitioning runs before excision so that a run of dead bars interrupted by an outage counts
#: as two runs. Counting it as one would excise bars on the strength of an absence (ADR-0018).
CANONICAL_STAGE_ORDER: tuple[str, ...] = (
    "session_filter",
    "gap",
    "jump",
    "illiquid",
    "stagnant",
    "min_length",
)

#: Session kinds a config may declare, matching the `session_id` values in the bar schema.
SESSION_KINDS = frozenset({"strict", "weekend", "exchange_calendar"})


class FrequencyRule(BaseModel):
    """One row of Kronos Table 4."""

    model_config = ConfigDict(extra="forbid")

    min_bars: int = Field(gt=0)
    jump_threshold: float = Field(gt=0.0)
    #: Longest run of illiquid bars that survives. A run of exactly this length is kept.
    max_illiquid: int = Field(ge=0)
    #: Longest run of equal closes that survives. A run of exactly this length is kept.
    max_stagnant: int = Field(ge=0)
    #: False for rows extracted from a secondary source and not yet re-read against the paper.
    #: :meth:`CleanConfig.rule_for` refuses to hand one out.
    verified: bool = True


class SessionRule(BaseModel):
    """When a missing grid step is expected rather than a boundary."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    #: `weekend` only: the UTC hour from which Friday counts as shut.
    friday_close_hour_utc: int = Field(default=20, ge=0, le=23)
    #: `weekend` only: the UTC hour before which Sunday counts as shut.
    sunday_open_hour_utc: int = Field(default=23, ge=0, le=24)
    #: `weekend` only: a daily settlement break, `[start, end)` in UTC hours, wrapping midnight
    #: when end <= start. -1 on either means the market runs straight through the day.
    #:
    #: Only the **gap** rule consults this. Bars inside the break are kept: most CFDs do print
    #: some, and deleting a real bar to satisfy a window would be worse than tolerating a gap.
    break_start_hour_utc: int = Field(default=-1, ge=-1, le=23)
    break_end_hour_utc: int = Field(default=-1, ge=-1, le=23)
    #: `exchange_calendar` only: the `exchange_calendars` code, e.g. `XNYS`.
    calendar: str = ""

    @property
    def has_break(self) -> bool:
        return self.break_start_hour_utc >= 0 and self.break_end_hour_utc >= 0

    def model_post_init(self, _: Any) -> None:
        if self.kind not in SESSION_KINDS:
            raise ValueError(
                f"unknown session kind {self.kind!r}; expected {sorted(SESSION_KINDS)}"
            )
        if self.kind == "exchange_calendar" and not self.calendar:
            raise ValueError("an exchange_calendar session must name a calendar")


class CleanConfig(BaseModel):
    """Everything a clean run needs, and nothing that identifies the run."""

    model_config = ConfigDict(extra="forbid")

    clean_version: int = Field(gt=0)
    stage_order: list[str]
    illiquid_eps: float = Field(ge=0.0)
    frequencies: dict[str, FrequencyRule]
    sessions: dict[str, SessionRule]
    #: `"{source}/{asset_class}" -> session_id`, overriding what a file's own metadata declares.
    #:
    #: Needed because `session_id` is stamped into the Parquet at pull time and a mistake there
    #: is only visible once cleaning runs. Re-pulling four million bars to correct a label would
    #: be the expensive way to fix a one-line classification error.
    session_overrides: dict[str, str] = Field(default_factory=dict)

    def session_id_for(self, source: str, asset_class: str, declared: str) -> str:
        """The session to clean under: an override if one applies, else what the file says."""
        return self.session_overrides.get(f"{source}/{asset_class}", declared)

    def model_post_init(self, _: Any) -> None:
        if tuple(self.stage_order) != CANONICAL_STAGE_ORDER:
            raise ValueError(
                f"stage_order is {self.stage_order}, but ADR-0018 fixes it at "
                f"{list(CANONICAL_STAGE_ORDER)}. Changing the order is a new ADR, not a "
                "config edit."
            )

    @property
    def config_hash(self) -> str:
        """Content identity of the **thresholds**. Stamped into every segment row.

        ``verified`` is excluded, and only ``verified``. It records how much anybody trusts a
        Table 4 row, not what the row says, and including it would mean that re-reading the 2H
        threshold against the paper invalidates every 1h and 1d segment in the corpus — a full
        recompute for a change that cannot have altered a single cut.
        """
        payload = self.model_dump(mode="json")
        payload["frequencies"] = {
            name: {k: v for k, v in rule.items() if k != "verified"}
            for name, rule in payload["frequencies"].items()
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:SHORT_LEN]

    def rule_for(self, frequency: str) -> FrequencyRule:
        """The Table 4 row for a frequency, refusing the ones nobody has checked.

        An unverified row is a transcription of a transcription. Handing one out silently would
        clean fifty million bars against a number that may not be in the paper.
        """
        try:
            rule = self.frequencies[frequency]
        except KeyError:
            raise ValueError(
                f"no cleaning rule for frequency {frequency!r}; config carries "
                f"{sorted(self.frequencies)}"
            ) from None
        if not rule.verified:
            raise ValueError(
                f"the {frequency} row is marked verified: false -- it came from an earlier "
                "extraction and has not been re-read against the Kronos PDF. Verify it and flip "
                "the flag before cleaning anything with it."
            )
        return rule

    def session_for(self, session_id: str) -> SessionRule:
        try:
            return self.sessions[session_id]
        except KeyError:
            raise ValueError(
                f"no session rule for {session_id!r}; config carries {sorted(self.sessions)}"
            ) from None


def load_clean_config(name_or_path: str | Path = "clean_v1") -> CleanConfig:
    """Load a cleaning config from YAML, rejecting unknown keys.

    Resolves a bare name against the packaged configs, so a cloud kernel with no checkout can
    run `axiom clean run --config clean_v1`.
    """
    path = resolve_config_path(name_or_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping, got {type(raw).__name__}")
    return CleanConfig.model_validate(raw)

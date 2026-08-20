"""Provenance manifests: what a file is, where it came from, and whether it is still current.

Two manifests, with different jobs.

:class:`FileManifest` is a sidecar written next to every Parquet file. It is the answer to "what
is this and where did it come from", and it is also the loader's resume mechanism: a pull compares
the source checksums it just enumerated against the ones the remote sidecar records, and skips the
symbol when they match. That is the only checkpoint state the pull job has.

:class:`PullRunManifest` is one file per pull run. It is the answer to "what happened that time",
including which symbols failed and whether the run was a partial smoke run rather than a full pull.

Everything here is pure data and hashing. No network, no filesystem beyond the two explicit
write/read helpers at the bottom.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from axiom.config.hashing import canonical_json

#: Suffix appended to a Parquet path to get its sidecar. Adjacency is the point: a file and its
#: provenance travel together, and a sync that moves one moves the other.
SIDECAR_SUFFIX = ".manifest.json"

#: Fields excluded from ``manifest_sha256``.
#:
#: The hash answers "is this the same data", not "was it produced by the same run at the same
#: time by the same build". Excluding these is what makes a re-pull byte-identical: the hash goes
#: into the Parquet key-value metadata, so if the wall clock or the git commit fed it, no two
#: pulls of the same month could ever produce the same bytes and the v0.1 exit gate would be
#: unmeetable by construction.
#:
#: ``artifact_sha256`` is excluded for a second reason as well -- it is the hash *of* the file
#: whose metadata carries ``manifest_sha256``, so including it would be circular.
VOLATILE_MANIFEST_FIELDS: frozenset[str] = frozenset(
    {"pull_run_id", "pulled_at", "artifact_sha256", "loader_version"}
)


class FileManifest(BaseModel):
    """Provenance for exactly one Parquet file in the raw tier."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    source: str
    market: str
    asset_class: str
    symbol: str
    frequency: str

    # Which run produced this copy, and when. Deliberately outside the identity hash.
    pull_run_id: str
    pulled_at: str
    loader_version: str

    # Every source file consumed, and the checksum published alongside it. The two lists are
    # parallel and ordered; `is_current` compares them against a fresh enumeration.
    source_urls: list[str]
    source_sha256s: list[str]

    artifact_path: str
    artifact_sha256: str = ""

    row_count: int
    first_ts: int
    last_ts: int
    gap_count: int
    #: Bars whose open time is not on the frequency grid. Real bars from an exchange restart,
    #: counted rather than repaired (ADR-0010).
    off_grid_count: int = 0

    # How to read the numbers. `volume` is base asset and `amount` is quote asset for Binance,
    # natively; a source without a native quote volume gets `amount` synthesized and says so.
    volume_convention: str = "base+quote_native"
    amount_synthesized: bool = False
    adjustment_policy: str = "none"

    universe_hash: str

    @model_validator(mode="after")
    def _checksums_pair_up(self) -> FileManifest:
        if len(self.source_urls) != len(self.source_sha256s):
            raise ValueError(
                f"{len(self.source_urls)} source urls but {len(self.source_sha256s)} checksums; "
                "they are parallel lists and a mismatch means one of them is wrong"
            )
        return self

    def identity_payload(self) -> dict[str, Any]:
        """The subset of fields the identity hash is taken over."""
        return {
            k: v
            for k, v in self.model_dump(mode="json").items()
            if k not in VOLATILE_MANIFEST_FIELDS
        }

    @property
    def manifest_sha256(self) -> str:
        """Content identity: changes when the data changes, not when the run does."""
        return hashlib.sha256(canonical_json(self.identity_payload()).encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        """Canonical serialization: sorted keys, trailing newline, stable across runs."""
        payload = self.model_dump(mode="json")
        payload["manifest_sha256"] = self.manifest_sha256
        return json.dumps(payload, sort_keys=True, indent=2) + "\n"

    @classmethod
    def from_json(cls, text: str | bytes) -> FileManifest:
        """Parse a sidecar, verifying the hash it carries against the one it implies."""
        payload = json.loads(text)
        recorded = payload.pop("manifest_sha256", None)
        manifest = cls.model_validate(payload)
        if recorded is not None and recorded != manifest.manifest_sha256:
            raise ValueError(
                f"{manifest.artifact_path}: sidecar records manifest_sha256={recorded} but its "
                f"own fields hash to {manifest.manifest_sha256}; the file has been edited"
            )
        return manifest


class PullFailure(BaseModel):
    """One symbol that did not land, and why. Kept in the run manifest so it is not just a log."""

    model_config = ConfigDict(extra="forbid")

    market: str
    symbol: str
    frequency: str
    error: str


class PullRunManifest(BaseModel):
    """What one invocation of the pull job did."""

    model_config = ConfigDict(extra="forbid")

    pull_run_id: str
    started_at: str
    finished_at: str = ""
    loader_version: str
    backend_tag: str
    universe_hash: str
    universe_path: str

    markets: list[str]
    frequencies: list[str]

    # A partial pull must never be mistaken for a full one, so the narrowing flags are recorded
    # rather than only appearing in somebody's shell history.
    limit: int | None = None
    symbols_filter: list[str] = Field(default_factory=list)

    ok: int = 0
    skipped: int = 0
    failed: int = 0
    total_rows: int = 0
    total_bytes: int = 0
    failures: list[PullFailure] = Field(default_factory=list)

    @property
    def is_partial(self) -> bool:
        return self.limit is not None or bool(self.symbols_filter)

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, indent=2) + "\n"


def is_current(remote: FileManifest | None, expected_source_sha256s: list[str]) -> bool:
    """The loader's skip test: does the remote copy already cover exactly these source files?

    Order matters. The enumeration is chronological, so a reordering would mean the enumerator
    changed its mind about what came first, which is worth re-pulling over.

    A grown daily tail changes the list and correctly forces a re-pull -- the series has genuinely
    gained bars since the last run.
    """
    if remote is None:
        return False
    return remote.source_sha256s == expected_source_sha256s


def sidecar_path(artifact_path: str | Path) -> Path:
    """The sidecar that belongs to a Parquet path."""
    return Path(str(artifact_path) + SIDECAR_SUFFIX)


def write_sidecar(manifest: FileManifest, artifact_path: str | Path) -> Path:
    """Write the sidecar next to its artifact and return where it went."""
    path = sidecar_path(artifact_path)
    path.write_text(manifest.to_json(), encoding="utf-8")
    return path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

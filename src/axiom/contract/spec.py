"""The frozen models behind the contract: what a spec says, and what the constants say.

Everything here is data plus validation. The arithmetic lives in :mod:`axiom.contract.transform`
and :mod:`axiom.contract.inverse`, and the reasons behind every field are pinned in ADR-0020.

Two identities matter and they are different. A **spec** identifies the parameterization —
which six features, which window, which clip bounds. **Constants** identify the affine scaling
those features are pushed through, and they are fitted once, on pre-firewall bars only, and then
committed. A consumer logs both hashes; a feature block produced under one pair is not comparable
to one produced under another.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from axiom.config.hashing import SHORT_LEN, canonical_json
from axiom.config.settings import resolve_config_path

#: The contract's version. Bumping it is a new tokenizer, new shards and new snapshots, not a
#: migration -- every artifact downstream of the contract is keyed by the features it saw.
SCHEMA_VERSION = 1

#: Parameterizations this schema version knows how to compute. `geo` is the designated primary
#: (ADR-0005); `ret` is the A/B challenger that v0.5's reconstruction study decides against.
PARAMETERIZATIONS = frozenset({"geo", "ret"})

#: Feature order per parameterization. Order is part of the contract: v0.6 writes columns in it
#: and v0.9 reads them back by position.
FEATURE_ORDER: dict[str, tuple[str, ...]] = {
    "geo": ("gap", "body", "upper", "lower", "volume", "amount"),
    "ret": ("ret_open", "ret_high", "ret_low", "ret_close", "volume", "amount"),
}

#: The two features built from a strictly-past rolling median rather than from the bar alone.
FLOW_FEATURES: tuple[str, str] = ("volume", "amount")


class ContractSpec(BaseModel):
    """One parameterization of a bar sequence into six causal features."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    spec_id: str
    schema_version: int = Field(ge=0)
    parameterization: str
    #: Length of the strictly-past window the flow features take their median over. The window
    #: expands from the segment start until it reaches this, and rolls afterwards.
    volume_window: int = Field(gt=0)
    clip_low: float
    clip_high: float
    #: True only for `kronos-zscore-v0`, which normalizes against the window it is normalizing.
    #: Production paths refuse it; it exists so v0.5 can measure what the leak buys.
    leaky: bool = False

    def model_post_init(self, _: Any) -> None:
        if self.parameterization not in PARAMETERIZATIONS:
            raise ValueError(
                f"unknown parameterization {self.parameterization!r}; "
                f"expected one of {sorted(PARAMETERIZATIONS)}"
            )
        if self.clip_high <= self.clip_low:
            raise ValueError(f"clip bounds are inverted: [{self.clip_low}, {self.clip_high}]")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"spec {self.spec_id!r} declares schema_version {self.schema_version}, but this "
                f"build of the contract is version {SCHEMA_VERSION}. Reading it would produce "
                "features that do not mean what the file says they mean."
            )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return FEATURE_ORDER[self.parameterization]

    @property
    def config_hash(self) -> str:
        """Content identity of the spec. Logged by every consumer, stamped into every report."""
        payload = self.model_dump(mode="json")
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:SHORT_LEN]


class Scaling(BaseModel):
    """The affine constants for one (asset class, frequency, feature).

    ``center`` is a robust median and ``scale`` an IQR/1.349 -- a normal-consistent standard
    deviation estimate that a fat tail cannot drag around. Both are fitted once and frozen; the
    contract never fits anything at run time, which is the property that makes pre-tokenization
    and streaming inference produce the same numbers.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    center: float
    scale: float = Field(gt=0.0)

    def model_post_init(self, _: Any) -> None:
        import math

        if not math.isfinite(self.center) or not math.isfinite(self.scale):
            raise ValueError(
                f"non-finite scaling constant: center={self.center} scale={self.scale}"
            )


class GenerationManifest(BaseModel):
    """What the constants were fitted from. Without it the numbers are unfalsifiable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    generated_utc: str
    git_commit: str
    registry_hash: str
    clean_config_hash: str
    firewall_ts: int
    firewall_config_sha256: str
    #: The job asserts `max(ts consumed) < firewall_ts` and writes the answer here. A `false`
    #: in a committed constants file is a failed build, not a note.
    firewall_respected: bool
    segments_consumed: int
    bars_consumed: int
    #: True when the fit ran over a subset. A partial fit may not be committed as the corpus fit.
    partial: bool = False


class ContractConstants(BaseModel):
    """Every spec's scaling table, plus the manifest saying where the numbers came from.

    Keyed `spec_id -> asset_class -> frequency -> feature`. A missing key is an error at lookup
    rather than a default, because a silent fallback to some other asset class's scale is
    exactly the kind of thing that shows up as an unexplainable evaluation number in v0.8.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(ge=0)
    manifest: GenerationManifest
    tables: dict[str, dict[str, dict[str, dict[str, Scaling]]]]

    def model_post_init(self, _: Any) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"constants declare schema_version {self.schema_version}, contract is "
                f"{SCHEMA_VERSION}"
            )
        if not self.manifest.firewall_respected:
            raise ValueError(
                "these constants were fitted over bars at or after the firewall. They are "
                "contaminated and no production path may load them (ADR-0021)."
            )
        if self.manifest.partial:
            raise ValueError(
                "these constants come from a partial fit (--limit). A partial fit is a smoke "
                "test; committing one as the corpus fit would freeze the contract against a "
                "sample nobody chose."
            )

    @property
    def config_hash(self) -> str:
        payload = self.model_dump(mode="json")
        payload.pop("manifest")
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:SHORT_LEN]

    def scaling_for(
        self, spec: ContractSpec, asset_class: str, frequency: str
    ) -> tuple[Scaling, ...]:
        """The six scalings for a spec and slice, in feature order.

        Returned as a tuple rather than a dict so the caller cannot accidentally apply them in
        a different order than the one the features were built in.
        """
        try:
            table = self.tables[spec.spec_id][asset_class][frequency]
        except KeyError as exc:
            known = sorted(self.tables.get(spec.spec_id, {}))
            raise KeyError(
                f"no constants for spec {spec.spec_id!r} / asset_class {asset_class!r} / "
                f"frequency {frequency!r} (missing {exc}); file carries asset classes {known}"
            ) from None
        missing = [name for name in spec.feature_names if name not in table]
        if missing:
            raise KeyError(
                f"constants for {spec.spec_id}/{asset_class}/{frequency} are missing "
                f"features {missing}"
            )
        return tuple(table[name] for name in spec.feature_names)


class FirewallConfig(BaseModel):
    """The date before which every fitted number in this project must live (ADR-0021)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    firewall_ts: int = Field(gt=0)
    firewall_date_utc: str
    rationale: str
    registry_hash: str


def _load_yaml(name_or_path: str | Path) -> dict[str, Any]:
    path = resolve_config_path(name_or_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping, got {type(raw).__name__}")
    return raw


def load_spec(name_or_path: str | Path) -> ContractSpec:
    """Load a spec from YAML, rejecting unknown keys and foreign schema versions.

    A bare name resolves against the packaged configs, so a cloud kernel with no checkout can
    run ``axiom contract dryrun --spec contract_geo_v1``.
    """
    return ContractSpec.model_validate(_load_yaml(name_or_path))


def load_constants(name_or_path: str | Path = "contract_constants_v1") -> ContractConstants:
    """Load the frozen scaling table, refusing anything contaminated or partial."""
    return ContractConstants.model_validate(_load_yaml(name_or_path))


def load_firewall(name_or_path: str | Path = "firewall") -> FirewallConfig:
    return FirewallConfig.model_validate(_load_yaml(name_or_path))


def firewall_sha256(name_or_path: str | Path = "firewall") -> str:
    """The hash ADR-0021 commits to. Taken over the file's bytes, not its parsed content."""
    path = resolve_config_path(name_or_path)
    return hashlib.sha256(path.read_bytes()).hexdigest()

"""Constants tables for tests, with a distinct row per feature on purpose.

The committed constants are fitted from the corpus and a test may not depend on them: a refit
would break every assertion for reasons that have nothing to do with the assertion. So tests
build their own here.

Every feature gets a **different** center and scale. A transform that wrote the body feature into
the gap column would still pass every test if all six rows were `(0, 1)`, and column-order bugs
are the ones that survive longest — the array has the right shape, the right dtype, and plausible
numbers in it.
"""

from __future__ import annotations

from axiom.contract.spec import (
    SCHEMA_VERSION,
    ContractConstants,
    ContractSpec,
    GenerationManifest,
    Scaling,
)

ASSET_CLASSES = ("crypto", "fx", "commodity", "equity")
FREQUENCIES = ("1h", "1d")


def _manifest() -> GenerationManifest:
    return GenerationManifest(
        generated_utc="2026-01-01T00:00:00Z",
        git_commit="0" * 40,
        registry_hash="test",
        clean_config_hash="test",
        firewall_ts=1_735_689_600_000,
        firewall_config_sha256="0" * 64,
        firewall_respected=True,
        segments_consumed=1,
        bars_consumed=1,
        partial=False,
    )


def constants(
    specs: ContractSpec | list[ContractSpec],
    *,
    asset_classes: tuple[str, ...] = ASSET_CLASSES,
    frequencies: tuple[str, ...] = FREQUENCIES,
    nudge: float = 0.0,
) -> ContractConstants:
    """A constants table covering every (spec, class, frequency) a test is likely to ask for.

    ``nudge`` shifts every center, so a test can assert that changing a constant changes the
    output — the guard against a transform that reads the table and ignores it.
    """
    if isinstance(specs, ContractSpec):
        specs = [specs]
    tables: dict = {}
    for spec in specs:
        for asset_class in asset_classes:
            for frequency in frequencies:
                row = {}
                for i, name in enumerate(spec.feature_names):
                    # Small, distinct, and signed: 0.001 * (i + 1) with alternating sign for the
                    # center, and a scale that is never 1.0 so a missing division is visible.
                    sign = 1.0 if i % 2 == 0 else -1.0
                    row[name] = Scaling(
                        center=sign * 0.001 * (i + 1) + nudge,
                        scale=0.01 * (i + 2),
                    )
                tables.setdefault(spec.spec_id, {}).setdefault(asset_class, {})[frequency] = row
    return ContractConstants(schema_version=SCHEMA_VERSION, manifest=_manifest(), tables=tables)

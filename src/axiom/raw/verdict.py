"""Stamp a measured adjustment verdict into the sidecars it applies to (ADR-0019).

The audit measured what Stooq had already done to its prices, and wrote the answer into a report.
The sidecars kept saying `vendor_adjusted_unverified`, which is what the loader honestly believed
before the audit ran — true, permanently, and also the first thing a reader of `axiom-raw` sees.

Correcting `adjustment_policy` in place is not possible: it is inside `manifest_sha256`, which is
stamped into each Parquet's own metadata, so editing it would break the file-to-sidecar link on
twelve thousand artifacts and turn a label fix into a re-pull.

So the verdict goes in a **second** field that lives outside the identity hash. Both facts are
kept — what was believed at pull time, what was measured after — and stamping costs a sidecar
rewrite and nothing else. No Parquet is touched, `artifact_sha256` does not move, and the segment
index bound to it stays valid.

Idempotent: a sidecar that already carries the verdict is left alone, so a re-run after a partial
one costs only the files it did not reach.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from axiom.provenance.manifest import SIDECAR_SUFFIX, FileManifest

log = logging.getLogger("axiom.raw")


@dataclass
class StampRun:
    """What one stamping pass did."""

    source: str
    verdict: str
    stamped: int = 0
    already: int = 0
    skipped: int = 0
    failures: list[dict[str, str]] = field(default_factory=list)

    def line(self) -> str:
        return (
            f"{self.source}: stamped {self.stamped}, already correct {self.already}, "
            f"not this source {self.skipped}, failed {len(self.failures)} "
            f"(verdict {self.verdict})"
        )


def needs_stamp(manifest: FileManifest, source: str, verdict: str) -> bool:
    return manifest.source == source and manifest.adjustment_policy_verified != verdict


def stamped(manifest: FileManifest, verdict: str) -> FileManifest:
    """A copy carrying the verdict. The identity hash is unchanged, and a test asserts it."""
    return manifest.model_copy(update={"adjustment_policy_verified": verdict})


def stamp_verdict(
    store: Any,
    manifests: list[FileManifest],
    *,
    source: str,
    verdict: str,
    dry_run: bool = False,
) -> StampRun:
    """Write the verdict into every sidecar belonging to ``source``.

    Sidecars go through the store's **batched** path. Twelve thousand direct writes would be
    twelve thousand Hub commits against a limit of 128 an hour; batched they are about seven.
    """
    run = StampRun(source=source, verdict=verdict)
    for manifest in manifests:
        if manifest.source != source:
            run.skipped += 1
            continue
        if manifest.adjustment_policy_verified == verdict:
            run.already += 1
            continue
        updated = stamped(manifest, verdict)
        if updated.manifest_sha256 != manifest.manifest_sha256:
            # Belt and braces. If this ever fires, the field has drifted into the identity hash
            # and stamping would silently break every Parquet's link to its sidecar.
            run.failures.append(
                {
                    "artifact_path": manifest.artifact_path,
                    "error": "stamping changed manifest_sha256; the verdict field is not volatile",
                }
            )
            continue
        if not dry_run:
            try:
                store.stage_bytes(
                    manifest.artifact_path + SIDECAR_SUFFIX,
                    updated.to_json().encode("utf-8"),
                )
            except Exception as exc:
                log.warning("could not stamp %s: %s", manifest.artifact_path, exc)
                run.failures.append(
                    {
                        "artifact_path": manifest.artifact_path,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
        run.stamped += 1
    if not dry_run:
        store.flush()
    return run

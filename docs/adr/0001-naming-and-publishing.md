# ADR-0001: Naming and publishing

**Status:** Accepted (v0.0)

## Context

The project needs a name to import, a name to distribute under, and a rule about when anything
becomes public. These are three separate decisions that get conflated. `axiom` on PyPI is already
taken by a squatted placeholder, so the distribution name cannot match the import name for free.

Reserving a distribution name now would mean creating a public PyPI project before there is
anything to publish, which contradicts the privacy rule below.

## Decision

The working title is Axiom and the import name is `axiom`. Code says `import axiom` from day one
and never changes.

The distribution name is deferred to the Publish Gate. Candidates are `axiom-kline`, `axiom-fm`,
and `axiom-quant`, and their availability is re-verified at that moment rather than now. We
reserve nothing, and we accept the risk that a candidate is taken in the interim — renaming a
distribution is a metadata change, and the import name is unaffected.

Nothing is public before the Publish Gate: no PyPI, no public GitHub, no public Hugging Face
repos. The Publish Gate is a separate explicit decision after v1.0, not a step inside it.

## Consequences

`pyproject.toml` carries `name = "axiom"` throughout the 0.x line. That is only ever built and
installed from git, so the PyPI collision cannot bite before the rename.

Every version's TODO inherits a hard non-goal: publish nothing. Anything that would create a
public artifact is out of scope until the gate.

# ADR-0021: The temporal firewall — 2025-01-01, fixed before anything was fitted

**Status:** Accepted (v0.4). Schedule deviation from the roadmap, deliberately.

## Context

The roadmap puts the firewall declaration in v0.5, alongside the tokenizer that first trains on
data. But v0.4's scaling constants are fitted numbers, and a fitted number that has seen the test
period is contaminated in exactly the way the firewall exists to prevent. Fitting first and
declaring the boundary afterwards would let the boundary be chosen — even unconsciously — to suit
what was already measured.

So the **date** moves to v0.4. The full sealed-holdout *governance* — embargo width, purging rules,
the hash-commit of the holdout itself — stays in v0.5, where the tokenizer needs it.

## Decision

### `firewall_ts = 1735689600000`, which is 2025-01-01T00:00:00Z

No statistic, constant, threshold, or model parameter in this project may be derived from a bar at
or after that instant, until v0.8 opens the sealed evaluation.

The boundary is **half-open on the right**: a bar stamped exactly `firewall_ts` is post-firewall. An
off-by-one here would put the sealed period's first bar into every fitted constant in the project.

### Why this date

Checked against registry `56873db71b6d`. The five M0 slices end between 2026-08-19 and 2026-08-21,
so the shortest post-firewall span — crypto UM 1h, ending 2026-08-19 — is **19.6 months**, against
the 18-month floor an evaluation window needs.

A calendar-year boundary because a round date is one fewer thing to mistype across a year of
manifests, and because a boundary chosen to land on a market event is a boundary chosen.

### The declaration is hash-committed

`src/axiom/configs/firewall.yaml` holds the timestamp, the rationale, and the registry hash it was
checked against. Its sha256 is:

```
94dd8b5072b01f746c03537450b6559180f21e87e3031fe22daad6c04719e871
```

A test asserts it. A firewall that can be edited without anybody noticing is not a firewall — moving
it means changing this ADR first, then refitting every constant that depended on it.

### Enforcement is in code, not by convention

`axiom contract fit-constants` truncates every segment at `firewall_ts` *before* computing a single
feature, records the `max(ts)` it actually consumed, and writes `firewall_respected` into the
constants file's generation manifest. A constants file whose manifest says the assertion failed
**does not load at all** — `ContractConstants` refuses it during validation, so no production path
can pick it up by accident.

The same manifest records the git commit, the registry hash, the clean-config hash, the firewall
file's own sha256, and the segment and bar counts consumed. Every number in the eventual model card
traces back through it.

## Consequences

v0.5 inherits a date it did not choose, which is the point: by the time the tokenizer's holdout
governance is written, the boundary is already fixed and already used, so it cannot be tuned to
flatter a result.

v0.8's leakage tripwires get a cheap, mechanical check: read the constants manifest, compare
`max(ts consumed)` against `firewall_ts`, and fail if the file ever loaded without that assertion
passing.

The cost is that 19.6 months of the corpus's most recent history is unavailable to every fitted
quantity until v0.8. For scaling constants estimated over tens of millions of bars, that costs
nothing measurable. It would matter for a model, which is why the model never sees it either.

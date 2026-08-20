# ADR-0009: Execution backend #2

**Status:** Accepted (v0.0). Supersedes the Modal row of ADR-0008 for v0.0 only.

## Context

Backend #2 exists so Kaggle is not a single point of failure, and so the CLI is forced to stay
backend-agnostic rather than quietly growing Kaggle assumptions. The roadmap named Modal.

Modal turned out to be unavailable: the account sits behind a review gate, and `modal token new`
expires its browser flow before the gate clears (`Token flow did not complete in time`). Beam is a
plausible substitute, but it means another account, another token, and another gate that might
close the same way.

Meanwhile GitHub Actions was already provisioned. The repo is private and live, CI is green,
`ubuntu-24.04` runners are CPU boxes, and repository secrets hold credentials the same way
Kaggle's secret store does. Nothing needed creating.

## Decision

**GitHub Actions is backend #2 for v0.0**, as `.github/workflows/loop.yml` — dispatched by hand,
never on push, calling the same `axiom loop run` the laptop and Kaggle call.

`remote/modal/loop_test.py` stays in the repo, unrun. When the Modal gate clears it works as
written, and Modal resumes its roadmap role.

**The Modal-versus-Beam-versus-anything-else decision moves to v0.6**, where the pre-tokenization
map job over roughly 50 million bars sets requirements a dummy trainer cannot. Choosing a compute
vendor now, to run a loop that increments a float, would be choosing on no evidence.

## Consequences

This backend is not vendor-independent. It shares an outage and an account with the code host, so
it does not deliver the resilience the roadmap wanted from backend #2. What it does deliver is the
part v0.0 actually tests: that a third backend runs the identical CLI path with no special case.
The independence requirement is deferred to v0.6 along with the vendor choice, and this paragraph
is here so that deferral is not mistaken for the requirement having been met.

One thing it does better than Modal would have. `gh run cancel` is a real SIGKILL of the runner,
scriptable from the laptop, so the kill-and-resume drill on this backend is genuine rather than
simulated and does not need a human clicking cancel in a UI.

It also needs one fewer secret than Kaggle: the job is already inside the repo, so `checkout`
supplies the code and no GitHub PAT is involved. Only `AXIOM_HF_TOKEN` is configured.

Actions minutes are metered on private repos (2,000/month on Free, 3,000 on Pro). A loop drill
costs single-digit minutes, so this is not a constraint at v0.0 and would become one at v0.7 —
another reason the real backend decision belongs later.

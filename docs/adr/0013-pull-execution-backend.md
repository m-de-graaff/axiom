# ADR-0013: The v0.1 pull runs on GitHub Actions, not Modal

**Status:** Accepted (v0.1). Extends ADR-0009 from the v0.0 loop to the v0.1 data jobs.

## Context

The roadmap gives Modal the data and map jobs, and the v0.1 plan is written around
`remote/modal/pull_binance.py`. ADR-0009 already recorded that Modal's account sits behind a
review gate and that GitHub Actions stood in for the v0.0 loop. The gate has not cleared: there
is still no Modal token on this machine.

The v0.1 pull needs the same three things the loop needed — a CPU box, a Hugging Face token, and
a way to kill it on purpose — plus outbound access to a public S3 bucket.

## Decision

**The pull, the universe build, and the v0.1 verification jobs run on GitHub Actions**, as
`.github/workflows/pull.yml` and `.github/workflows/universe.yml`, dispatched by hand.

The work itself is not in the workflow. `axiom pull binance` is a CLI command, and the workflow
is fifteen lines that call it. That is the same shape v0.0 settled on, and it is what makes the
choice of backend cheap to revisit: when Modal opens, the Modal function is a handful of lines
that call the same command.

The cost fits. A full v0.1 pull is roughly a gigabyte of small archives across some six hundred
series, an hour or so of runner time against the 2 000 minutes a month a private repo gets on the
Free plan.

## Consequences

The vendor-independence half of what backend #2 was for is still not delivered, for the same
reason ADR-0009 gave: Actions shares an account and an outage with the code host. That deferral
now spans two versions rather than one. It is still deferred to v0.6, where the pre-tokenization
map job sets requirements a data pull does not, and where the vendor decision will be made on
evidence rather than on availability.

One thing this does better than Modal would have, and it matters more here than it did for the
loop: `gh run cancel` SIGKILLs the runner from the laptop, scriptably. The v0.1 kill drill —
launch a partial pull, kill it mid-flight, relaunch, watch the finished symbols skip — is a
one-liner rather than a human clicking Cancel in somebody's dashboard.

The six-hour job limit on a hosted runner is not a constraint at v0.1 volumes, and would not be
one even if it were: a pull that runs out of clock is resumed by dispatching it again, because
resuming is what the pull does on every start anyway.

`remote/modal/loop_test.py` stays in the repo, unrun, as ADR-0009 left it. No Modal pull job is
written until there is a Modal account to run it on; writing one now would be writing code that
cannot be tested, which is how a file rots into a liability.

# Axiom task runner. `just` with no argument lists these.

default:
    @just --list

# --- local quality gates (same commands CI runs) ---

fmt:
    uv run ruff format .

lint:
    uv run ruff format --check .
    uv run ruff check .

type:
    uv run ty check src

test:
    uv run pytest -q

# Everything CI runs, in CI's order: cheapest first.
check: lint type test

# --- the v0.0 loop ---

# Run the loop locally without touching the Hub.
loop-local config="loop_test":
    AXIOM_DISABLE_TRACKING=1 uv run axiom loop run --config {{config}} --no-push

# The determinism drill: clean run vs killed-and-resumed run, compared exactly.
loop-verify config="loop_test":
    AXIOM_DISABLE_TRACKING=1 uv run axiom loop verify --config {{config}}

# Run locally against the real Hub, so a checkpoint tree exists to inspect.
loop-hub config="loop_test":
    uv run axiom loop run --config {{config}} --backend-tag local-hub

# Dispatch to Kaggle (CPU kernel, zero GPU quota). See docs/RUNBOOK.md for the kill drill.
loop-kaggle:
    uv run kaggle kernels push -p remote/kaggle/loop_test

loop-kaggle-status:
    uv run kaggle kernels status markdgraaff/axiom-loop-test

loop-kaggle-log:
    uv run kaggle kernels output markdgraaff/axiom-loop-test -p .artifacts/kaggle-out

# Dispatch to GitHub Actions (CPU, backend #2 for v0.0 per ADR-0009).
loop-github run_id="loop-test-github-001":
    gh workflow run loop.yml -f run_id={{run_id}}

loop-github-watch:
    gh run watch $(gh run list --workflow=loop.yml --limit 1 --json databaseId -q '.[0].databaseId') --exit-status

# Kill the newest loop run mid-flight. A real SIGKILL of the runner, for the resume drill.
loop-github-kill:
    gh run cancel $(gh run list --workflow=loop.yml --limit 1 --json databaseId -q '.[0].databaseId')

# Dispatch to Modal. Blocked on the account review gate; see ADR-0009.
loop-modal:
    uv run modal run remote/modal/loop_test.py

# Delete local checkpoint trees. The durable copies live in axiom-runs.
clean-checkpoints:
    rm -rf checkpoints

# --- the v0.1 data jobs ---

# Build the pinned universe in the cloud, then fetch the YAML it produced.
universe-build month:
    gh workflow run universe.yml -f month={{month}}

universe-fetch:
    gh run download $(gh run list --workflow=universe.yml --limit 1 --json databaseId -q '.[0].databaseId') -n universe_v1 -D src/axiom/configs

# Print a universe's criteria and counts, verifying its hash.
universe-show path="universe_v1":
    uv run axiom universe show {{path}}

# Dispatch the Binance pull. Extra flags go through as workflow inputs; see .github/workflows/pull.yml.
pull-binance *ARGS:
    gh workflow run pull.yml {{ARGS}}

# Smoke run: two majors, spot only, both frequencies.
pull-smoke:
    gh workflow run pull.yml -f markets=spot -f symbols=BTCUSDT,ETHUSDT

pull-watch:
    gh run watch $(gh run list --workflow=pull.yml --limit 1 --json databaseId -q '.[0].databaseId') --exit-status

pull-log:
    gh run view $(gh run list --workflow=pull.yml --limit 1 --json databaseId -q '.[0].databaseId') --log

# Kill the newest pull mid-flight. A real SIGKILL of the runner, for the resume drill.
pull-kill:
    gh run cancel $(gh run list --workflow=pull.yml --limit 1 --json databaseId -q '.[0].databaseId')

# Pull into a local directory instead of the Hub. Development only — this writes market data to
# the machine it runs on, which the laptop must never do.
pull-local symbols="BTCUSDT":
    uv run axiom pull binance --symbols {{symbols}} --markets spot --dest .artifacts/raw-local

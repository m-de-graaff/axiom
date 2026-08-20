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

# Dispatch to Modal (CPU, backend #2).
loop-modal:
    uv run modal run remote/modal/loop_test.py

# Delete local checkpoint trees. The durable copies live in axiom-runs.
clean-checkpoints:
    rm -rf checkpoints

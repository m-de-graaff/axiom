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

# Same path against the real bucket, writing to the runner's disk and publishing nothing.
pull-dryrun symbols="BTCUSDT,ETHUSDT":
    gh workflow run pull.yml -f dry_run=true -f symbols={{symbols}}

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

# --- the v0.2 Dukascopy pull ---

# Reachability smoke: one instrument against the real feed, written to the runner's disk only.
dukascopy-smoke symbols="EURUSD":
    gh workflow run pull-dukascopy.yml -f dry_run=true -f symbols={{symbols}} -f frequencies=1d

# The full pull. Extra flags go through as workflow inputs; see .github/workflows/pull-dukascopy.yml.
pull-dukascopy *ARGS:
    gh workflow run pull-dukascopy.yml {{ARGS}}

dukascopy-watch:
    gh run watch $(gh run list --workflow=pull-dukascopy.yml --limit 1 --json databaseId -q '.[0].databaseId') --exit-status

dukascopy-log:
    gh run view $(gh run list --workflow=pull-dukascopy.yml --limit 1 --json databaseId -q '.[0].databaseId') --log

# Kill the newest Dukascopy pull mid-flight, for the resume drill.
dukascopy-kill:
    gh run cancel $(gh run list --workflow=pull-dukascopy.yml --limit 1 --json databaseId -q '.[0].databaseId')

# --- the v0.2 Stooq pull ---

# The URL-handoff smoke: download from the runner and parse, publishing nothing.
stooq-smoke url limit="500":
    gh workflow run pull-stooq.yml -f dry_run=true -f archive_url='{{url}}' -f limit={{limit}}

# The real pull. Solve the CAPTCHA at https://stooq.com/db/h/ and pass the resulting direct URL.
pull-stooq url:
    gh workflow run pull-stooq.yml -f archive_url='{{url}}'

stooq-watch:
    gh run watch $(gh run list --workflow=pull-stooq.yml --limit 1 --json databaseId -q '.[0].databaseId') --exit-status

stooq-log:
    gh run view $(gh run list --workflow=pull-stooq.yml --limit 1 --json databaseId -q '.[0].databaseId') --log

# --- the v0.2 yfinance adjunct ---

# Reachability smoke: twenty tickers, publishing nothing. Yahoo may simply refuse; that is data.
yahoo-smoke limit="20":
    gh workflow run pull-yahoo.yml -f dry_run=true -f limit={{limit}}

# Capture splits and dividends for the whole pinned list. Takes about two hours, paced.
pull-yahoo *ARGS:
    gh workflow run pull-yahoo.yml {{ARGS}}

yahoo-watch:
    gh run watch $(gh run list --workflow=pull-yahoo.yml --limit 1 --json databaseId -q '.[0].databaseId') --exit-status

# --- the v0.2 corpus registry ---

# Rebuild the registry over every sidecar in axiom-raw. Run after every pull.
registry-build:
    gh workflow run registry.yml

# Build over a local raw tier instead, for development.
registry-local dest=".artifacts/raw-local":
    uv run axiom registry build --dest {{dest}}

# Arbitrary SQL over the registry. Needs `uv sync --extra query`.
registry-query sql:
    uv run axiom registry query "{{sql}}"

# --- the v0.2 corpus artifacts ---

# Build the equities universe and run the adjustment audit in the cloud.
corpus job="both":
    gh workflow run corpus.yml -f job={{job}}

# Fetch what the corpus job produced, into the paths they belong in.
corpus-fetch:
    gh run download $(gh run list --workflow=corpus.yml --limit 1 --json databaseId -q '.[0].databaseId') -n corpus-outputs -D .

corpus-watch:
    gh run watch $(gh run list --workflow=corpus.yml --limit 1 --json databaseId -q '.[0].databaseId') --exit-status

# Create the private axiom-raw dataset and seed its front page. Idempotent.
bootstrap-raw:
    gh workflow run bootstrap.yml -f repo=axiom-raw

# Fetch one series and print what failed validation. Writes nothing anywhere.
raw-inspect symbol market="spot" frequency="1h":
    gh workflow run raw.yml -f command=inspect -f symbol={{symbol}} -f market={{market}} -f frequency={{frequency}}

# Re-derive a sample of series and compare the bytes against their manifests.
raw-verify sample="10":
    gh workflow run raw.yml -f command=verify -f sample={{sample}}

# Regenerate the QA report from the sidecars.
raw-stats:
    gh workflow run raw.yml -f command=stats

# Download the newest QA report the raw workflow produced.
raw-report-fetch:
    gh run download $(gh run list --workflow=raw.yml --limit 1 --json databaseId -q '.[0].databaseId') -n v0.1-raw-qa -D docs/reports

# ADR-0012 safety net: diff axiom-raw against an independent implementation.
raw-crosscheck symbols="BTCUSDT,ETHUSDT,SOLUSDT":
    gh workflow run raw.yml -f command=crosscheck -f symbol={{symbols}}

# --- the v0.3 cleaning pass ---

# Clean the whole corpus on Modal: registry fan-out -> clean/v1/ in axiom-raw.
clean-corpus *ARGS:
    uv run modal run remote/modal/clean_run.py {{ARGS}}

# Smoke run: fifty artifacts, same code path.
clean-smoke:
    uv run modal run remote/modal/clean_run.py --limit 50

# Clean a local raw tier instead, for development.
clean-local dest=".artifacts/raw-local":
    uv run axiom clean run --dest {{dest}}

# Render the drop-stats report and drop it in docs/reports/.
clean-report:
    uv run axiom clean report --out docs/reports/v0.3-clean-qa.md

# Build the derived total-return tier per the recorded adjustment verdict (ADR-0019).
derive-tr *ARGS:
    uv run axiom derive tr {{ARGS}}

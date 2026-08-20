# Axiom v0.0 — "Spine & Loop" — Phased Development TODO

**Goal:** a private, fully-tooled monorepo plus a *proven* develop-local / execute-remote loop: one command on the laptop dispatches a dummy trainer to the cloud, which pulls the code, runs, checkpoints full state to Hugging Face, survives a deliberate mid-run kill, and resumes **bit-identically**. Zero market data. Zero GPU minutes.

**Starting state:** an empty folder. No git, no remotes, nothing published.
**Exit gate:** **G1** (see roadmap §6).
**Total budget:** ~6 focused days (≈ 1–1.5 calendar weeks).

**Non-goals for v0.0 (do not let scope creep):** no market data downloaded anywhere (not even a sample CSV); no tokenizer/model/eval code; no GPU kernels; no PyPI, no public repos, no GCP/TRC; no Cloudflare R2.

**Repos/services this version creates (all private, all documented in `docs/REPOS.md` as part of Phase D):**
- GitHub `m-de-graaff/axiom` (private) — the monorepo remote, required so Kaggle/Modal can install the package.
- HF dataset `m-de-graaff/axiom-runs` (private) — checkpoints + `latest.json` resume pointers.
- HF Space `m-de-graaff/axiom-trackio` (private, optional) — trackio dashboard sync; trackio may auto-create a backing dataset, which is accepted and documented.

---

## Phase A — Decisions & accounts (budget: 0.5 day)

### A1. Accounts & prerequisites
- [ ] Kaggle account phone-verified (required for internet-enabled kernels and secrets). Confirm "Internet" toggle is available in a scratch notebook.
- [ ] Hugging Face account ready; note the namespace is `m-de-graaff` (or record the actual HF username in `docs/REPOS.md` if different).
- [ ] Modal account created; workspace confirmed on the free Starter plan ($30/mo credits).
- [ ] Local toolchain on laptop: `git`, `uv` (latest), `gh` CLI (optional but convenient), `kaggle` CLI, `modal` CLI. Record versions in `docs/RUNBOOK.md` later (D4).
- [ ] Explicitly NOT done: GCP project, TRC application, PyPI anything.

### A2. Architecture Decision Records — lock the six open decisions
Create `docs/adr/` with one short file each (context → decision → consequences). These are the research's "open design decisions" plus toolchain/topology, resolved per its recommendations:
- [ ] `0001-naming-and-publishing.md` — working title Axiom; **import name `axiom`**; distribution name deferred to the Publish Gate (PyPI `axiom` is squatted; candidates `axiom-kline`/`axiom-fm`/`axiom-quant`, re-verified at publish time; no reservation now by policy). Nothing public before the Publish Gate.
- [ ] `0002-tokenizer-hierarchy.md` — BSQ (vendored from Kronos, MIT) is the hierarchy default; flat FSQ via `vector-quantize-pytorch` is an ablation only; novel hierarchical-FSQ is post-1.0 research.
- [ ] `0003-frequency-scope.md` — corpus M0 = 1h + 1d across crypto/FX/commodities/equities; M1 (crypto 15m/5m toward ~0.3 B bars) decided at Gate G3.
- [ ] `0004-compute-target.md` — GPU-only for v1.0 (Kaggle P100 first, 2×T4 DDP only if throughput-bound); TPU/TRC is a stretch branch gated at G3/G4; paid burst only by explicit decision.
- [ ] `0005-preprocessing-parameterization.md` — candle-geometry (log h/o, log l/o, log c/o, gap) is the default contract; plain per-field log-returns is the A/B; causal normalization only (no per-window future stats).
- [ ] `0006-corpus-ambition.md` — M0 (~50 M bars) mandatory floor; M1 stretch; undertraining relative to Kronos's 12 B bars is a documented, accepted limitation.
- [ ] `0007-toolchain.md` — uv + Ruff (lint+format) + **one** type checker as CI source of truth (start with `ty`; if it blocks on any real code, switch to `mypy` and amend this ADR) + pytest + hypothesis + typer (CLI) + pydantic-settings (config) + trackio (tracking). Python floor `>=3.11` (Kaggle-image compatible; re-verify actual Kaggle Python in Phase F and amend if needed).
- [ ] `0008-repo-topology.md` — the table from roadmap §3: private monorepo + HF `axiom-runs` now; `axiom-raw` (v0.1), `axiom-tokenized` (v0.6), `axiom-model` (v0.9) later; all private until the Publish Gate.

**Phase A acceptance:** all 8 ADRs written; accounts verified; nothing created online yet.

---

## Phase B — Repo skeleton, local only (budget: 0.5 day)

### B1. Initialize
- [ ] In the empty folder: `git init` (branch `main`).
- [ ] `uv init --lib --name axiom` (src layout). Verify tree: `src/axiom/__init__.py`, `pyproject.toml`, `.python-version` (set to 3.12), `README.md`.
- [ ] `.gitignore`: `.venv/`, `__pycache__/`, `.env`, `*.env`, `data/`, `checkpoints/`, `runs/`, `.kaggle/`, `dist/`, `.pytest_cache/`, `.ruff_cache/`, `.hypothesis/`, `*.parquet`, `*.safetensors`. (Data and secrets can never enter history — this is the day-one rule.)
- [ ] `LICENSE` = MIT (Mark). `NOTICE` = placeholder stating Kronos (MIT, © shiyu-coder) attribution will accompany any vendored code from v0.5 onward.
- [ ] `README.md` (private-facing): one-paragraph project statement + the honesty banner + pointer to `docs/`.
- [ ] `CHANGELOG.md` in Keep-a-Changelog format with an `[Unreleased]` section.

### B2. `pyproject.toml`
- [ ] `[project]`: name `axiom` (fine for a never-published private package; the *distribution* rename happens at the Publish Gate), `requires-python = ">=3.11"`, version `0.0.0`.
- [ ] Core deps (keep minimal): `pydantic-settings`, `pyyaml`, `typer`, `huggingface_hub`, `trackio`.
- [ ] Optional extras: `train = ["torch"]`, `kaggle = ["kaggle"]`, `modalrun = ["modal"]`, `dev = ["pytest", "hypothesis", "ruff", "pre-commit"]` (+ `ty` or `mypy` per ADR-0007).
- [ ] `[tool.uv]`: add a `pytorch-cpu` index (`https://download.pytorch.org/whl/cpu`) and pin `torch` to it via `[tool.uv.sources]` so laptop + CI never pull CUDA wheels. Cloud images bring their own torch.
- [ ] `[tool.ruff]`: line-length 100, `lint.select = ["E","F","I","UP","B","SIM","RUF"]`; format enabled.
- [ ] Type checker config per ADR-0007 (`[tool.ty]` or `[tool.mypy]` strict-ish on `src/`).
- [ ] `uv sync --all-extras` succeeds; `uv lock` committed.

### B3. Directory scaffold (empty `__init__.py` where needed)
- [ ]
```
src/axiom/
  cli.py                # typer app: version, config, loop subcommands
  config/               # settings.py, hashing.py
  ops/                  # seeding.py, logx.py, checkpoint.py, hub.py
  loop/                 # dummy_trainer.py  (v0.0's stand-in "model")
configs/loop_test.yaml
remote/kaggle/loop_test/   # kernel-metadata.json + run.py (Phase F)
remote/modal/loop_test.py  # (Phase F)
tests/
docs/adr/  docs/REPOS.md  docs/RUNBOOK.md  docs/ARCHITECTURE.md
.github/workflows/ci.yml   # (Phase E)
justfile                   # fmt, lint, type, test, loop-* recipes
```
- [ ] `justfile` recipes: `fmt`, `lint`, `type`, `test`, `check` (all three), `loop-local`, `loop-verify`, `loop-kaggle`, `loop-modal`.
- [ ] First commit: `chore: repo skeleton (v0.0 phase B)`.

**Phase B acceptance:** `uv run python -c "import axiom"` works; `just check` runs (even with zero tests); tree matches scaffold; committed.

---

## Phase C — Config & reproducibility core (budget: 1 day)

### C1. Settings & run config (`src/axiom/config/settings.py`)
- [ ] `AxiomSettings(BaseSettings)`: `hf_token: SecretStr | None`, `hf_namespace: str = "m-de-graaff"`, `runs_repo: str = "axiom-runs"`, env-prefix `AXIOM_`, loads `.env` locally, plain env vars in cloud.
- [ ] `LoopConfig` (pydantic model, YAML-loadable): `run_id: str`, `seed: int = 1337`, `total_steps: int`, `save_every: int`, `sleep_s: float = 0.05`, `backend_tag: str`, `schema_version: int = 0`.
- [ ] YAML loader: `load_config(path) -> LoopConfig` with unknown-key rejection (`model_config = ConfigDict(extra="forbid")`).

### C2. Config hashing (`src/axiom/config/hashing.py`)
- [ ] `config_hash(cfg) -> str`: sha256 over canonical JSON (sorted keys, no whitespace) of the model dump, excluding volatile fields (`run_id`, `backend_tag`). First 12 hex chars used in artifact paths.
- [ ] Test: hash stable across field order / reload; changes when any substantive field changes; volatile fields don't affect it.

### C3. Determinism (`src/axiom/ops/seeding.py`)
- [ ] `seed_all(seed)` seeding `random`, `numpy`, `torch` (+ cuda if present, harmless on CPU).
- [ ] `capture_rng_state() -> dict` and `restore_rng_state(dict)` covering all three generators (torch state as bytes/tensor serialized safely).
- [ ] Test: seed → generate → capture → generate seq A; restore → generate seq A′; assert A == A′ exactly.

### C4. Logging & tracking (`src/axiom/ops/logx.py`)
- [ ] Stdlib `logging` setup (console, level from env), plus `init_tracking(cfg)` wrapping `trackio.init(project="axiom", name=run_id, config=...)` with a no-op fallback when trackio/env is absent (cloud kernels must never crash on tracking).
- [ ] Log at run start: config hash, git commit (env-provided in cloud, `git rev-parse` locally), package version, python/torch versions, backend tag.

### C5. CLI (`src/axiom/cli.py`)
- [ ] `axiom version` — package version + git commit if known.
- [ ] `axiom config hash configs/loop_test.yaml` — prints the hash (smoke-tests C1+C2).
- [ ] Wire console script `axiom = "axiom.cli:app"` in `pyproject.toml`; `uv run axiom version` works.

**Phase C acceptance:** tests for C2/C3 green; `uv run axiom config hash` prints a stable hash; commit `feat: config + reproducibility core`.

---

## Phase D — Secrets & remote repos (budget: 0.5 day)

> This phase creates the private online infrastructure. Every item created is recorded in `docs/REPOS.md` the same day — that file is the living version of roadmap §3.

### D1. GitHub (private monorepo)
- [ ] Create **private** repo `m-de-graaff/axiom`; `git remote add origin …`; push `main`.
- [ ] Create fine-grained PAT `axiom-kaggle-read`: repository access = only `axiom`, permissions = Contents: Read-only, expiry 90 days. Store nowhere on disk except the password manager.
- [ ] Document in `docs/REPOS.md` + note the **no-GitHub fallback** for the record: `uv build` a wheel and attach it as a private Kaggle Dataset instead of cloning (kept as documented plan B, not implemented).

### D2. Hugging Face
- [ ] Create fine-grained token `axiom-write`: write access scoped to repos matching `axiom-*` only, expiry 90 days.
- [ ] Create **private dataset** repo `m-de-graaff/axiom-runs` (`huggingface_hub.create_repo(..., repo_type="dataset", private=True)`).
- [ ] Optional: create private Space `m-de-graaff/axiom-trackio` for trackio sync; if trackio auto-creates a backing dataset, add it to `docs/REPOS.md`.
- [ ] Laptop: `.env` with `AXIOM_HF_TOKEN=…` (gitignored). Verify `whoami` via `HfApi().whoami()`.

### D3. Kaggle & Modal secrets
- [ ] Kaggle → Account → Create API token → `~/.kaggle/kaggle.json` (laptop only, chmod 600).
- [ ] Kaggle user secrets (attached later to the kernel): `GH_PAT` = the D1 token, `HF_TOKEN` = the D2 token.
- [ ] Modal: `modal token new`; create secrets `axiom-gh` (`GH_PAT`) and `axiom-hf` (`HF_TOKEN`).

### D4. RUNBOOK
- [ ] `docs/RUNBOOK.md` first sections: token inventory table (name, scope, where stored, expiry, rotation = 90 days), how to rotate each, how to revoke fast, and the rule that tokens never appear in code, configs, notebook cells' outputs, or git history.

**Phase D acceptance:** `git push` works; `axiom-runs` visible (private) on HF; a 1-line test script uploads + deletes a dummy file in `axiom-runs` using the env token; `docs/REPOS.md` and `docs/RUNBOOK.md` committed.

---

## Phase E — CI (budget: 0.5 day)

- [ ] `.github/workflows/ci.yml`, triggers: push + PR on `main`.
- [ ] Job `lint` (ubuntu-latest, `astral-sh/setup-uv`): `uv sync --frozen --extra dev` → `ruff format --check .` → `ruff check .`.
- [ ] Job `types`: run the ADR-0007 checker over `src/` (`uvx ty check src` or `uv run mypy src`).
- [ ] Job `test` (matrix Python 3.11 / 3.12 / 3.13): `uv sync --frozen --extra dev --extra train` (torch = CPU wheel via the B2 index pin) → `uv run pytest -q`.
- [ ] Keep total CI < ~5 min (uv cache action on).
- [ ] Badge-free (private); note in `docs/RUNBOOK.md` that a CPU tiny-model smoke-train job gets added to this workflow in v0.7.

**Phase E acceptance:** first push shows all three jobs green on GitHub.

---

## Phase F — The Loop (budget: 2.5 days) — the heart of v0.0

The dummy trainer stands in for the real model so that all checkpoint/resume machinery built now is reused unchanged in v0.5/v0.7. Design rule: *the loop code may not know it's a dummy* — same `TrainState`, same writer, same resume path the AR model will use.

### F1. Dummy trainer (`src/axiom/loop/dummy_trainer.py`)
- [ ] `TrainState` (dataclass): `step: int`, `acc: float`, `rng: dict` (from C3 `capture_rng_state`), `config_hash: str`, `run_id: str`, `schema_version: int`.
- [ ] Step function: `acc += torch.rand((), generator=g).item()` with a dedicated seeded `torch.Generator`; `time.sleep(cfg.sleep_s)`. The RNG-driven `acc` makes any resume divergence detectable to the last bit.
- [ ] `run(cfg, state|None)`: loop `state.step → cfg.total_steps`; every `save_every` steps call the checkpoint writer (F2) and `trackio.log({"step": …, "acc": …})`.
- [ ] Fault injection for tests: env `AXIOM_KILL_AT_STEP=<n>` raises `SystemExit(137)` at step n (used by F4 and the Kaggle kill drill).

### F2. Checkpoint writer/reader (`src/axiom/ops/checkpoint.py`, `hub.py`)
- [ ] Local write is atomic: serialize `TrainState` (via `torch.save` of a plain dict) to `checkpoints/{run_id}/step_{n:08d}/state.pt` + `meta.json` (step, sha256 of state.pt, wall time, config hash) using tmp-file + `os.replace`.
- [ ] Keep-last-K=3 local pruning.
- [ ] `hub.py`: `push_checkpoint(local_dir, repo="{ns}/axiom-runs", path_in_repo=f"loop-test/{run_id}/step_{n:08d}")` via `HfApi.upload_folder(run_as_future=True)` (non-blocking); then upload/overwrite `loop-test/{run_id}/latest.json` = `{step, path_in_repo, sha256}`. Block on futures at run end.
- [ ] `pull_latest(run_id) -> TrainState | None`: download `latest.json` → download that step dir → verify sha256 → deserialize → `restore_rng_state`.
- [ ] Unit test (local FS only, HF mocked/skipped): write → read roundtrip preserves every field bit-exactly, including RNG state (continue 10 steps from restored vs original — identical `acc`).

### F3. CLI wiring
- [ ] `axiom loop run --config configs/loop_test.yaml [--resume] [--backend-tag …]` — runs F1 end-to-end; with `--resume`, calls `pull_latest` first.
- [ ] `axiom loop verify` — the local determinism drill (F4) as one command.
- [ ] `configs/loop_test.yaml`: seed 1337, total_steps 1000, save_every 100, sleep_s 0.05.

### F4. Local determinism drill
- [ ] Run A: fresh, uninterrupted, 1000 steps → record final `acc` (full `repr`) and final RNG state → `H_A`.
- [ ] Run B: fresh, `AXIOM_KILL_AT_STEP=437` → dies at 437 (last checkpoint 400) → `axiom loop run --resume` → completes 1000 → `H_B`.
- [ ] Assert `H_A == H_B` exactly. Encode as `tests/test_loop_determinism.py` (marked `slow`; sleep_s=0 in test).
- [ ] Also verify: resume after kill *between* checkpoints correctly replays from step 400 (checkpoint floor), not 437 — i.e., recomputation is deterministic, not skipped.

### F5. Kaggle backend (CPU kernel — zero GPU quota)
- [ ] `remote/kaggle/loop_test/kernel-metadata.json`: `id: m-de-graaff/axiom-loop-test`, `code_file: run.py`, `kernel_type: script`, `enable_gpu: false`, `enable_internet: true`.
- [ ] `run.py`: read `GH_PAT` + `HF_TOKEN` via `kaggle_secrets.UserSecretsClient`; export `AXIOM_HF_TOKEN`; `pip install "git+https://x-access-token:${GH_PAT}@github.com/m-de-graaff/axiom.git@main"`; log installed commit; run `axiom loop run --config … --resume --backend-tag kaggle` with `total_steps 2000, save_every 200`.
- [ ] Attach both Kaggle secrets to the kernel; dispatch with `kaggle kernels push -p remote/kaggle/loop_test`; watch with `kaggle kernels status`.
- [ ] **Kill drill on Kaggle:** cancel the running kernel from the Kaggle UI mid-run (past step ~600); re-push; confirm the log shows `resumed from step N` and the run completes to 2000; confirm final `acc` equals a clean local run of the same config (seed-identical, sleep-independent).
- [ ] Confirm on HF: `loop-test/{run_id}/…` checkpoint dirs + `latest.json` present; nothing but logs on Kaggle's side matters.
- [ ] Record the Kaggle image's Python + torch versions in `docs/RUNBOOK.md`; amend ADR-0007 floor if needed.

### F6. Modal backend (backend #2)
- [ ] `remote/modal/loop_test.py`: `modal.App("axiom-loop")`, image = `debian_slim().pip_install("git+https://…@main")` using `modal.Secret.from_name("axiom-gh")` at build, run function with `axiom-hf` secret, CPU only, calls the same CLI path with `--backend-tag modal`, `total_steps 500`.
- [ ] `modal run remote/modal/loop_test.py` → checkpoints land under a distinct `run_id` in `axiom-runs`.
- [ ] Note cost in RUNBOOK (should be cents of the $30 credit).

### F7. Loop hygiene
- [ ] Prove the laptop stayed clean: `checkpoints/` exists locally only for the local drill and is gitignored; delete it; `git status` clean; no data anywhere.
- [ ] trackio dashboard shows local + kaggle + modal runs of project `axiom` (or the no-op fallback is confirmed logged when disabled).

**Phase F acceptance:** F4 test green locally; Kaggle kill-drill resumed bit-identically; Modal run landed checkpoints; all runs visible in tracking; HF `axiom-runs` contains three run trees; RUNBOOK updated with dispatch instructions.

---

## Phase G — Docs, tag, exit review (budget: 0.5 day)

- [ ] `docs/ARCHITECTURE.md`: one page — the C1–C11 component map from the research, which components exist as of v0.0 (C1, C11) and which version delivers each of the rest (mirror of roadmap §2).
- [ ] `docs/REPOS.md` final check: every online thing created in v0.0 listed with visibility + purpose + created-in-version.
- [ ] `docs/RUNBOOK.md` final check: tokens, rotation, dispatch recipes (`just loop-kaggle`, `just loop-modal`), kill/resume procedure, "what to do when a Kaggle session dies."
- [ ] `CHANGELOG.md`: move v0.0 content from `[Unreleased]` to `[0.0.0] - <date>`.
- [ ] Bump `pyproject.toml` version to `0.0.0` finality; `git tag v0.0.0` (local tag pushed to the private remote; no release object needed).
- [ ] **G1 exit checklist** (all must be true):
  - [ ] CI green on `main` (lint, types, tests × 3 Python versions).
  - [ ] 8 ADRs merged; the six design decisions are closed.
  - [ ] Kill→resume bit-identity proven locally (test) **and** on Kaggle (drill).
  - [ ] Checkpoints + `latest.json` for ≥ 3 runs live in private `axiom-runs`.
  - [ ] Both backends (Kaggle CPU, Modal CPU) executed the identical CLI path.
  - [ ] Zero market data touched; zero Kaggle GPU-hours consumed; Modal spend < $2.
  - [ ] No secret in git history (`git log -p | grep -iE "hf_|ghp_|token"` returns nothing); `.env` untracked.
  - [ ] `docs/REPOS.md` documents every repo created; nothing is public.

**Then:** open the v0.1 TODO (next session) — "Schema & First Bars": canonical bar schema, provenance manifests, Binance loader with CHECKSUM verification, `axiom-raw` created, first cloud-to-cloud pull.

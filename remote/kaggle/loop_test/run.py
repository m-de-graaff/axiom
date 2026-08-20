"""Kaggle CPU kernel for the v0.0 loop drill.

Deliberately a CPU kernel: v0.0 spends zero GPU quota. What this proves is the dispatch path, not
throughput.

The kernel installs the private monorepo from GitHub and then calls the same `axiom loop run` the
laptop calls, against the same packaged config. Nothing here reimplements training logic; if this
file starts growing logic, the loop has stopped being backend-agnostic.

Attach secrets `GH_PAT` and `HF_TOKEN` to the kernel in the Kaggle UI (Add-ons -> Secrets).
Neither is printed. The install URL carries the PAT, so it is passed to pip as a list element and
never echoed.
"""

import os
import platform
import subprocess
import sys

REPO = "m-de-graaff/axiom"
BRANCH = "main"

# Larger than the local drill so a mid-run cancel from the Kaggle UI lands somewhere useful.
TOTAL_STEPS = 2000
SAVE_EVERY = 200

# A stable run id is what makes a re-push resume rather than start over.
RUN_ID = "loop-test-kaggle-001"


def read_secrets() -> tuple[str, str]:
    from kaggle_secrets import UserSecretsClient

    client = UserSecretsClient()
    return client.get_secret("GH_PAT"), client.get_secret("HF_TOKEN")


def report_image() -> None:
    """ADR-0007 leaves the Python floor provisional until a real kernel reports its version."""
    print(f"kaggle python: {platform.python_version()}", flush=True)
    try:
        import torch

        print(f"kaggle torch: {torch.__version__}", flush=True)
    except ImportError:
        print("kaggle torch: absent", flush=True)


def install(gh_pat: str) -> None:
    url = f"git+https://x-access-token:{gh_pat}@github.com/{REPO}.git@{BRANCH}"
    print(f"installing {REPO}@{BRANCH} (token redacted)", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", url],
        check=True,
    )
    subprocess.run([sys.executable, "-m", "axiom.cli", "version"], check=True)


def main() -> None:
    gh_pat, hf_token = read_secrets()
    report_image()
    install(gh_pat)

    os.environ["AXIOM_HF_TOKEN"] = hf_token

    subprocess.run(
        [
            sys.executable,
            "-m",
            "axiom.cli",
            "loop",
            "run",
            "--config",
            "loop_test",  # resolved from the installed package; the kernel has no checkout
            "--resume",
            "--backend-tag",
            "kaggle",
            "--run-id",
            RUN_ID,
            "--total-steps",
            str(TOTAL_STEPS),
            "--save-every",
            str(SAVE_EVERY),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()

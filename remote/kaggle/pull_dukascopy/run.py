"""Kaggle CPU kernel for the v0.2 Dukascopy pull.

This is the only pull in the project that does not run on GitHub Actions, and the reason is
measured rather than assumed: Dukascopy's chart endpoint returns HTTP 403 to Actions runner IPs
and answers a Kaggle kernel. ADR-0015 records all three hosts and their answers.

A CPU kernel, so it spends no GPU quota. Like the v0.0 loop kernel, it installs the private
monorepo and then calls the same CLI command the laptop would; if this file grows logic, the pull
has stopped being backend-agnostic.

Attach secrets `GH_PAT` and `HF_TOKEN` in the Kaggle UI (Add-ons -> Secrets). Neither is printed.

Re-pushing this kernel resumes rather than restarting: an instrument whose sidecar already covers
today's as-of date is skipped, because a year that has ended cannot gain bars (ADR-0015).
"""

import os
import subprocess
import sys

REPO = "m-de-graaff/axiom"
BRANCH = "main"

#: Narrow these for a smoke run. Empty means the whole pinned universe.
SYMBOLS = ""
LIMIT = ""

#: Set to a small integer to run the kill drill: the pull dies without flushing after that many
#: items, exactly as a session death would. Re-push with it cleared to watch the finished
#: instruments skip. Left empty for a real pull.
KILL_AFTER_ITEMS = ""


def read_secrets() -> tuple[str, str]:
    from kaggle_secrets import UserSecretsClient

    client = UserSecretsClient()
    return client.get_secret("GH_PAT"), client.get_secret("HF_TOKEN")


def install(gh_pat: str) -> None:
    # The `data` extra brings `dukascopy-python`. It is not a base dependency: CI drives every
    # loader through a synthetic fetcher and must not acquire a network dependency.
    url = f"axiom[data] @ git+https://x-access-token:{gh_pat}@github.com/{REPO}.git@{BRANCH}"
    print(f"installing {REPO}@{BRANCH} with the data extra (token redacted)", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", url], check=True
    )
    subprocess.run([sys.executable, "-m", "axiom.cli", "version"], check=True)


def main() -> None:
    gh_pat, hf_token = read_secrets()
    install(gh_pat)

    os.environ["AXIOM_HF_TOKEN"] = hf_token
    os.environ["AXIOM_DISABLE_TRACKING"] = "1"
    os.environ["AXIOM_STAGING_DIR"] = "/kaggle/tmp/axiom-raw-staging"
    if KILL_AFTER_ITEMS:
        os.environ["AXIOM_KILL_AFTER_ITEMS"] = KILL_AFTER_ITEMS

    argv = [
        sys.executable,
        "-m",
        "axiom.cli",
        "pull",
        "dukascopy",
        "--universe",
        "universe_dukascopy_v1",  # resolved from the installed package; no checkout here
        "--frequencies",
        "1h,1d",
        "--backend-tag",
        "kaggle",
    ]
    if SYMBOLS:
        argv += ["--symbols", SYMBOLS]
    if LIMIT:
        argv += ["--limit", LIMIT]

    subprocess.run(argv, check=True)


if __name__ == "__main__":
    main()

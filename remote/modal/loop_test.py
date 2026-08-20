"""Modal CPU job for the v0.0 loop drill: execution backend #2.

Backend #2 exists so Kaggle is not a single point of failure. Its value is that it runs the
*identical* CLI path — if this file ever needs a special case, the loop has a portability bug.

Run it with `modal run remote/modal/loop_test.py`, or `just loop-modal`.

Secrets `axiom-gh` (GH_PAT) and `axiom-hf` (HF_TOKEN) must exist in the Modal workspace. The
GitHub PAT is needed at image build time to install the private repo; the Hugging Face token is
needed at run time to push checkpoints. Keeping them separate means the runtime container never
holds a credential that can read source.
"""

import modal

REPO = "m-de-graaff/axiom"
BRANCH = "main"

# Shorter than the Kaggle drill: this proves portability, not endurance. Cents of the monthly
# credit, per docs/RUNBOOK.md.
TOTAL_STEPS = 500
SAVE_EVERY = 100
RUN_ID = "loop-test-modal-001"

app = modal.App("axiom-loop")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        f"git+https://x-access-token:$GH_PAT@github.com/{REPO}.git@{BRANCH}",
        secrets=[modal.Secret.from_name("axiom-gh")],
    )
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("axiom-hf")],
    timeout=60 * 30,
)
def loop() -> str:
    import os
    import subprocess
    import sys

    # The secret arrives as HF_TOKEN; the package reads AXIOM_HF_TOKEN.
    os.environ["AXIOM_HF_TOKEN"] = os.environ["HF_TOKEN"]

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "axiom.cli",
            "loop",
            "run",
            "--config",
            "loop_test",
            "--resume",
            "--backend-tag",
            "modal",
            "--run-id",
            RUN_ID,
            "--total-steps",
            str(TOTAL_STEPS),
            "--save-every",
            str(SAVE_EVERY),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    return result.stdout.strip().splitlines()[-1]


@app.local_entrypoint()
def main() -> None:
    print(loop.remote())

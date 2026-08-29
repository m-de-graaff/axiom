"""The ROCm leg (P4-02), to run on the RX 7900 XTX box.

    uv run python scripts/rocm_check.py                    # small, then base
    uv run python scripts/rocm_check.py --models axiom-zero-base --samples 64

Prints a markdown row to paste into `docs/rocm-notes.md`. Pass condition is
`token_identical: True` — the cached generation loop must produce the same bars on
ROCm that it does on CPU and CUDA. If it does not, that is a finding worth a note in
the incidents table, not something to work around silently (CLAUDE.md).

Reference numbers to compare against, Modal L4, 64 samples, 24 steps, 488 context:
    axiom-zero-small   cached 1.4 s     uncached  ~4 s
    axiom-zero-base    cached 2.8 s     uncached 25.8 s   (9.1x, max abs diff 0.0)
"""

from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=["axiom-zero-small", "axiom-zero-base"])
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--device", help="default: cuda:0 when present (ROCm reports as cuda)")
    parser.add_argument("--json", help="also write the raw results here")
    args = parser.parse_args(argv)

    import torch
    from axiom_eval.bench import parity_and_speed

    print(f"torch {torch.__version__} · cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")
        print(f"hip: {getattr(torch.version, 'hip', None)} · cuda: {torch.version.cuda}")

    results = []
    for model in args.models:
        result = parity_and_speed(model, args.samples, args.horizon, args.device)
        results.append(result)
        print(f"\n{model}")
        for key, value in result.items():
            print(f"  {key:>18}: {value}")
        if not result["token_identical"]:
            print("  FAIL — not token-identical on this backend; record it in docs/rocm-notes.md")

    print("\nPaste into docs/rocm-notes.md:\n")
    print("| model | backend | cached | uncached | speedup | token-identical |")
    print("|---|---|---|---|---|---|")
    for r in results:
        print(
            f"| `{r['model']}` | {r['accelerator']} (torch {r['torch']}) | {r['cached_seconds']}s "
            f"| {r['reference_seconds']}s | {r['speedup']}x | {r['token_identical']} |"
        )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)

    return 0 if all(r["token_identical"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

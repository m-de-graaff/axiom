# ADR-0004: Compute target

**Status:** Accepted (v0.0)

## Context

Free compute comes in two incompatible shapes. Kaggle offers GPUs (P100, or 2×T4) at roughly
30 GPU-hours a week in 12-hour sessions. Kaggle and TRC offer TPUs, which need JAX or
XLA-friendly PyTorch and a rewrite of anything using `torch.multinomial` in the sampling path.

The dual-head decoder samples the coarse token before predicting the fine one. That multinomial
call is on the hot path, and porting it to XLA is the kind of work that eats a week and produces
a model that trains slower than the GPU version.

## Decision

v1.0 is GPU-only. Kaggle P100 single-GPU is the baseline; 2×T4 with DDP is used only if the run
turns out to be throughput-bound, not by default.

TPU and TRC are a stretch branch, gated at G3 and G4. The TRC application is filed at G3 only if
the tokenizer passes early, because TRC takes about two weeks to grant and expires whether or not
it is used.

Paid compute — Vast.ai or RunPod at roughly $0.30/hour — requires an explicit decision. The
trigger is v0.7 stalling more than two weeks on quota, and even then it is opt-in.

## Consequences

Training code targets CUDA semantics and fp16 with GradScaler. Nothing in the v1.0 path may
depend on XLA, so the TPU branch stays a branch.

The whole project is built resume-first, because a 12-hour session cap makes interruption the
normal case rather than the failure case. That discipline is what v0.0 exists to prove.

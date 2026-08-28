# Machine / backend notes

Record here: torch versions + install commands per machine (laptop CPU wheel,
XTX ROCm wheel), and every "works on one backend, breaks on another" incident
with its workaround. Rules of the road live in CLAUDE.md (SDPA-only attention,
no CUDA-only deps, keep --no-compile working, never assume a local GPU).

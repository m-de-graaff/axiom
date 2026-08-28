"""axiom_eval -- the frozen measuring stick (Phase 2).

No model or preprocessing change merges without before/after numbers from here
(CLAUDE.md rule 1). Layout:

    panel        which windows get scored + the leakage asserts (P2-05, P2-11)
    forecasters  models and the baseline humiliation panel     (P2-06..08)
    metrics      RankIC, cost-aware direction, errors, calibration, tripwire
    run          the harness: config in, reports/{run_id}/ out (P2-10, P2-12)
    report       the HTML it writes

Submodules are imported explicitly (`from axiom_eval import metrics`) so that
importing the package does not drag in torch or lightgbm.
"""

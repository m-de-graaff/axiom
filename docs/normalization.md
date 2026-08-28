# Normalization spec (P1-09)

Document the upstream Kronos per-window normalization scheme here after reading
vendor/kronos source, then implement it ONCE in axiom_data.normalization.
Training, eval, and inference must all import that single module.
Mismatch here is the project's #1 known silent failure mode.

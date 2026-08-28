"""axiom_data -- corpus download, storage, QA and dataset building.

Submodules are imported explicitly (`from axiom_data import store`) rather than
re-exported here: the Modal smoke image installs only what the model needs, so
importing this package must not drag in duckdb/httpx.

Normalization lives in exactly one place (`axiom_data.normalization`), resampling in
exactly one place (`axiom_data.resample`). Do not re-implement either elsewhere.
"""

"""Read-only signal API (P6-06). Deploy via Modal ASGI or a small VPS.
Only this API touches the DB from the outside world; dashboard + AI chat
consume it. Token auth required before exposure."""

from fastapi import FastAPI

app = FastAPI(title="axiom-signal-api")


@app.get("/health")
def health() -> dict:
    return {"ok": True}


# TODO P6-06: /signals?tf=1h · /forecast/{symbol} · /health/model · /universe

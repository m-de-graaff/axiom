-- Axiom Postgres schema (see build order §8.2).
-- With TimescaleDB, optionally: SELECT create_hypertable('candles','ts');

CREATE TABLE IF NOT EXISTS candles (
  symbol text NOT NULL,
  tf     text NOT NULL,
  ts     timestamptz NOT NULL,
  o double precision, h double precision, l double precision,
  c double precision, v double precision,
  PRIMARY KEY (symbol, tf, ts)
);

CREATE TABLE IF NOT EXISTS runs (
  run_id   uuid PRIMARY KEY,
  kind     text NOT NULL,            -- 'infer_cron' | 'train' | 'eval' | ...
  git_sha  text,
  config   jsonb,
  started  timestamptz NOT NULL DEFAULT now(),
  finished timestamptz,
  status   text NOT NULL DEFAULT 'running',
  error    text
);

CREATE TABLE IF NOT EXISTS forecasts (
  run_id   uuid NOT NULL REFERENCES runs(run_id),
  symbol   text NOT NULL,
  tf       text NOT NULL,
  made_at  timestamptz NOT NULL,     -- close time of last observed bar
  horizon  int  NOT NULL,            -- bars ahead
  quantiles jsonb NOT NULL,          -- {"q10":..,"q50":..,"q90":..} per step or terminal
  mc_summary jsonb,                  -- samples, T, top_p, dispersion stats
  model    text NOT NULL,
  PRIMARY KEY (run_id, symbol, tf, horizon)
);
CREATE INDEX IF NOT EXISTS forecasts_lookup ON forecasts (symbol, tf, made_at);

CREATE TABLE IF NOT EXISTS signals (
  symbol   text NOT NULL,
  tf       text NOT NULL,
  made_at  timestamptz NOT NULL,
  horizon  int  NOT NULL,
  p_up     real NOT NULL,
  exp_ret  real NOT NULL,
  conf     real NOT NULL,
  stance   text NOT NULL,            -- 'BULL' | 'BEAR' | 'NEUTRAL'
  model    text NOT NULL,
  PRIMARY KEY (symbol, tf, made_at, horizon)
);
CREATE INDEX IF NOT EXISTS signals_recent ON signals (made_at DESC);

CREATE TABLE IF NOT EXISTS model_health (
  day        date NOT NULL,
  model      text NOT NULL,
  rankic     real,
  hit_rate   real,
  coverage80 real,
  notes      text,
  PRIMARY KEY (day, model)
);

-- Phase 8 adds: orders, fills, positions, pnl_daily.

# Proposal — btc-news-signal

## Intent
Build a first educational slice that proves a small local model can map
a BTC news headline plus BTC/EUR context to buy / sell / hold, with
honest temporal evaluation. Learning outcome, not trading performance.
Without pipeline and honest evaluation, any accuracy is noise.

## Users (dual purpose)
- Learner (you): learn end-to-end ingest -> dataset -> model -> eval -> CLI.
- Hiring reviewer (portfolio): hires for decisions, not returns — why Kaggle
  before GDELT, why 24h, how leakage was avoided. Hence Engram memory +
  `DECISION_LOG.md` as audit trail.

## Business rules (R1-R6 per teacher)
- R1 (label looks forward, train-time only): `ret_24h = close(t+24h)/close(t)-1`.
  `buy` if r > +thr, `sell` if r < -thr, else `hold`. Historical truth only.
- R2 (features look backward only): backward join-asof, each news gets last
  candle at-or-before its timestamp, UTC. Never the future candle. Prevents
  decision leakage.
- R3 (params, not hardcoded): `flat_threshold` init ±0.5% tunable [0.3/0.5/1.0%],
  `embargo_days` init 1-2 tunable [1 vs 2]. Val-only tuning, never test-peeked.
- R4 (mandatory temporal eval): chronological split + 24h overlap purge +
  embargo. Random split forbidden (inflates 1x-6x).
- R5-R6 (ethical/legal): no real trading, always disclaimer, allowed sources
  only. Saying no to dangerous scope is portfolio signal.

## Scope — first slice (6 steps)
In:
- 1 `price-ingest`: Kraken `XXBTZEUR` daily OHLC bulk + incremental update.
- 2 `news-ingest`: Kaggle crypto-news 2018+, 30-50k, normalize to UTC, dedup.
- 3 `dataset` (core intelligence): align + label without leakage, join-asof +
  24h labels + temporal split with purge+embargo + `data_card.md` contract.
- 4 `baseline`: TF-IDF + LogisticRegression (balanced), CPU explainable.
- 5 `eval`: per-class PR (not global accuracy) + confusion matrix +
  simulated return vs buy-hold (informational).
- 6 `cli`: inference-only, headline + price -> action + probs + disclaimer.
  No retrain in CLI.
- Constraints: CPU-only, <100MB local, parquet fixed schema, fixed seed,
  TDD `python -m pytest`, 400-line budget, `DECISION_LOG.md` trail.

Out (explicit non-goals):
- No live trading, no financial advice, no real-money execution.
- No LLM fine-tune in this slice.
- No GDELT full backfiles (phase 2), no aggressive scraping.
- No intraday horizon in slice 1.

## Evaluation params (tunable, not hardcoded)
- `flat_threshold`: initial ±0.5%, tunable [0.3%, 0.5%, 1.0%]. Decided in `sdd/btc-news-signal/evaluation` (mem id 4).
- `embargo_days`: initial 1-2, tunable [1 vs 2]. Same decision record.
- Both flow to `spec` as config params with val-only tuning, never test-peeked.

## Affected areas
- New: `src/ingest_kraken.py`, `src/ingest_news.py`, `src/dataset.py`,
  `src/train.py`, `src/cli.py`, `tests/*`, `data/` (gitignored payloads).
- No existing production code to migrate (greenfield).

## Risks
- Kaggle license/quality variance -> allowlist + data_card gate.
- TZ misalignment -> UTC-only join, tests for asof correctness.
- Leakage via overlapping 24h windows -> purge+embargo enforced + test.
- Duplicates -> url+title hash dedup, tested.
- Hold dominance -> balanced weights, report per-class, not accuracy alone.
- SDD executor model instability (init/explore/proposal delegates failed)
  -> parent-inline fallback used, recorded here for audit.

## Tradeoffs and assumptions
- Low thr (0.3%) more signals but noisy vs high thr (1.0%) cleaner but few.
- Short embargo (1d) more data vs long (2d) safer.
- Kaggle fast/small vs GDELT complete/heavy — Kaggle wins slice 1.
- Assume mostly English; baseline barely above random is success (learning).
  No perfect option, conscious decision + logged.

## Rollback
Delete `openspec/changes/btc-news-signal/` outputs and `data/interim/*`;
no migration, no external state. Re-run from bulk CSVs.

## Success criteria
- Dataset builds reproducibly from raw CSVs with data_card counts.
- Temporal split has zero future leakage by test.
- Baseline trains on CPU, macro-F1 reported on test, CLI returns action
  with disclaimer for a sample headline.
- All within 400-line review budget or flagged ask-on-risk.

## Next
` sdd-spec` then `sdd-design` then `sdd-tasks` for this change.

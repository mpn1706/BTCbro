# Explore — btc-news-signal

## Intent
Educational local classifier: BTC news headline + BTC/EUR price -> buy / sell / hold.
No advice, no live trading.

## Current state
- Repo: clean Python 3.14.7, only pip. No dataset, no code.
- `openspec/config.yaml`: strict TDD, `python -m pytest`, budget 400.
- Teacher ask: classifier from specialized portals (yahoo/finance/crypto).

## Confirmed product decisions
- Corte 1: Kaggle crypto-news ~2018+, 30-50k + Kraken XXBTZEUR daily OHLC.
- Fase 2 (out of scope): GDELT backfiles from 2015.
- Horizon: 24h forward return -> up / down / flat.
- Baseline: TF-IDF + LogisticRegression, temporal split, CLI with disclaimer.
- Formal `sdd-research`: unselected (runtime has no evidence grants).

## Parts and integrations
1. **price-ingest (Kraken)**: public OHLC `XXBTZEUR`, no key. Bulk CSV first, then incremental API. Output: `data/raw/kraken_xxbtzeur_1d.csv` (timestamp, open, high, low, close, volume).
2. **news-ingest (Kaggle)**: 30-50k crypto-news 2018+. Output: `data/raw/news.csv` (id, published_at, source, title, body, url). Normalize to UTC, dedup by url+title hash.
3. **normalize**: UTC timestamps, lowercase/strip, language flag, keep raw + clean. Contract: `data/interim/data_card.md` (counts, date range, nulls, dup rate).
4. **join-asof**: each news -> last close at-or-before `published_at` + close 24h after. Backward-asof only, never forward. Timezone-aware.
5. **label-24h**: forward return `r = close_T+24h / close_T - 1`. Thresholds e.g. `r > +1.5% => buy`, `r < -1.5% => sell`, else `hold`. Thresholds are config, tuned on train only.
6. **temporal split**: train (2018-2022) / val (2023) / test (2024+), purged overlap + 1-2d embargo because 24h labels overlap. No random split.
7. **train-baseline**: TF-IDF (title+body, 5-20k feats) + LogisticRegression balanced. No LLM fine-tune in slice 1. Metrics: macro-F1, per-class precision/recall, confusion matrix, backtest return vs buy-hold (informational only).
8. **cli-demo**: input headline + fetch current price -> output action + probabilities + disclaimer. Read-only model load, `<100MB` local.

Data flow:
`Kaggle + Kraken -> normalize+dedup -> join-asof -> label-24h -> temporal-split -> TF-IDF+LogReg -> metrics -> CLI`

## Risks
- Kaggle license/quality varies; need source allowlist + data_card.
- Timestamp misalignment (news TZ vs Kraken UTC) -> join bias.
- Leakage via random split or overlapping 24h windows; must purge+embargo.
- Class imbalance (hold dominates) -> stratified temporal eval, balanced weights.
- GDELT API only covers recent window; deep history needs heavy backfiles -> correctly deferred to phase 2.

## Scope boundaries
- In: daily horizon, English (+ES if present), CPU-only, local files.
- Out: intraday, real-money execution, aggressive scraping, LLM fine-tune.

## Next
Ready for `sdd-proposal` on same change. Formal research remains unselected.

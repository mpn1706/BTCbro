# Bitcoin News Classifier — Project Context

## Goal
Learn to train a small local model that, given a BTC news headline and the
current BTC/EUR price, suggests buy / sell / hold. Educational only.

## Non-goals
- No financial advice.
- No automated trading with real money.
- No LLM fine-tuning in the first slice.
- No aggressive scraping against ToS.

## Scope (first slice)
1. Ingest BTC/EUR price history (Kraken public OHLC, pair XXBTZEUR).
2. Ingest BTC news history (GDELT 2.0 filtered by bitcoin/BTC, Kaggle backup).
3. Label each news item by 24h forward return (up / down / flat).
4. Baseline: TF-IDF + LogisticRegression with temporal split, no leakage.
5. CLI demo: headline + current price -> action + disclaimer + metrics.

## Constraints
- Local CPU only, small model, explainable.
- Strict TDD: `python -m pytest`, RED/GREEN/TRIANGULATE/REFACTOR.
- Temporal split mandatory. No future leakage.
- Review budget: 400 lines. Delivery: ask-on-risk. Mode: interactive.

## Data notes
- Price: Kraken OHLC XXBTZEUR (free, no key). Alternative: CoinGecko.
- News history: GDELT 2.0 (free, no key). RSS (CoinDesk/Cointelegraph) only
  for recent inference, not for history.
- Alignment: join news to price by timestamp. Horizon: 24h default.

## Disclaimer
This project is for learning. Output is not investment advice.

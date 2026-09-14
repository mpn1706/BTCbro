# Bitcoin News Signal (educational slice)

Small local model mapping a BTC news headline (+ BTC/EUR price context)
to `buy` / `sell` / `hold`. Learning outcome, not trading performance.
Output is never investment advice; live trading is out of scope.

## Pipeline (leakage-first)

Kaggle news + Kraken `XXBTZEUR` daily → UTC normalize + dedup →
backward join-asof (features past-only) → 24h forward label →
temporal split + purge + embargo → TF-IDF + balanced LogReg (CPU,
`SEED=42`) → per-class eval → inference-only CLI.

- Train 2018–2022 learns weights; val 2023 chooses
  `flat_threshold ∈ {0.003, 0.005, 0.01}` / `embargo ∈ {1, 2}` /
  TF-IDF / `C` by macro-F1; test 2024+ is read exactly once.
- Tuning on test is failure regardless of score (see `DECISION_LOG.md`).

## Run

```bash
python -m pytest
python -m src.cli --title "Bitcoin ETF inflows hit record" --price 60000 --model models/latest
btc-signal --title "Bitcoin ETF inflows hit record" --price 60000
```

`--price` is display context only, not a trained feature (`--help`).
Exit codes: `0` ok · `2` bad input · `3` missing/corrupt artifact · `1` internal.

## Layout

`src/config.py` (seed/params) · `src/ingest_*.py` · `src/dataset.py`
(join/label/split + `data_card.md`) · `src/train.py` · `src/evaluate.py`
(per-class P/R, macro-F1, 3×3 matrix) · `src/cli.py` → `btc-signal`.

EDUCATIONAL DEMO ONLY — NOT INVESTMENT ADVICE. NO LIVE TRADING.

# Apply progress — btc-news-signal — PR1 (ingest)

Scope: PR1 ONLY (PR2 dataset, PR3 model/eval, PR4 CLI untouched).

## Completed (PR1 RED → GREEN → TRIANGULATE → REFACTOR)

- Fixtures `tests/fixtures/news_tiny.csv` (7 rows: tz-mixed, 1 dup, 1 missing-field) + `prices_tiny.csv` (6 daily rows).
- `tests/test_ingest.py`: 9 tests (UTC, dedup, quarantine, wrong-pair, polite pagination mocked + Madrid-naive, empty-source→unknown, OHLCV coercion, de-overlap).
- `src/config.py`: SEED=42, thr 0.005∈{0.003,0.005,0.01}, embargo 1∈{1,2}, PAIR XXBTZEUR, train_end/val_end.
- `src/ingest_news.py`: required-field quarantine+count, →UTC (naive assumed Europe/Madrid), sha1(url‖title) dedup keep-earliest, lang report.
- `src/ingest_kraken.py`: bulk CSV + daily UTC validation, incremental from max+1d (≤720/call, ≥1s sleep), pair gate aborts naming XXBTZEUR.

## TDD Cycle Evidence

| Cycle | Command | Result |
|---|---|---|
| RED | `python -m pytest tests/test_ingest.py -v` (no src/) | collection ERROR, `No module named 'src.config'` |
| GREEN-fix | same (after src/ + 2 test fixes: finite-history fake, tz-dtype assert) | 5 passed |
| TRIANGULATE | same (4 edge tests added; 1 test-logic fix: overlap row served with new day) | 9 passed |
| REFACTOR | same (shared `_price_batch` helper, simpler missing-field mask) | 9 passed |

## Verification

- `python -m pytest tests/test_ingest.py -v` → 9 passed.
- Manual bulk load: `python -m src.ingest_news` + `python -m src.ingest_kraken` on local CSV copies → `data/interim/{news_clean,prices_clean}.parquet` + expected counts (5 kept/1 missing/1 dup; 6 prices).
- Size: 398 new lines (src 222 + tests 161 + fixtures 15) — within 400-line PR budget.

## Deviations

- `data/.gitignore` NOT created: root `.gitignore` already ignores `data/raw/`, `data/clean/`, `data/proc/`, `data/interim/*.{parquet,csv}`; a second ignore file would conflict. No `.gitignore` overwritten.
- `tasks.md` checkboxes NOT ticked: file is outside this run's allowed edit surfaces; parent owns the update.
- Deps installed into environment: `pytest pandas pyarrow requests` (were absent).

## Remaining (out of scope for this run)

- PR2: `tests/test_dataset.py` + `src/dataset.py` + card. PR3: train/eval. PR4: CLI/docs. Final slice gate.
- Rollback PR1: delete `data/interim/news_clean.parquet`, `data/interim/prices_clean.parquet` (+ local `data/raw/` copies).

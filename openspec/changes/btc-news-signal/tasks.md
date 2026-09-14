# Tasks — btc-news-signal

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | ~900–1200 (src ~600 + tests ~300 + fixtures/config/packaging ~100) |
| 400-line budget risk | High |
| Chained PRs recommended | Yes |
| Suggested split | PR1 ingesta → PR2 label+split → PR3 modelo+eval → PR4 CLI+docs |
| Delivery strategy | ask-on-risk |
| Chain strategy | stacked-to-main |

```text
Decision needed before apply: No
Chained PRs recommended: Yes
Chain strategy: stacked-to-main
400-line budget risk: High
```

> 4-PR stacked chain approved (professor + user). Single PR would exceed 400 lines. Implement as 4 chainable work units stacked-to-main, each ideally <400 lines, each independently verifiable with rollback = delete its outputs. Do NOT merge PR2/3/4 until prior PR is reviewed.

## Temporal Discipline — EXPLICIT (binding for all tasks)

- **Train 2018-2022 learns weights only.** Model parameters (TF-IDF vocabulary, LogisticRegression coefficients) are fit on train split exclusively (`train_end=2022-12-31`).
- **Valid 2023 chooses hyperparameters only.** `flat_threshold ∈ {0.003,0.005,0.01}`, `embargo_days ∈ {1,2}`, TF-IDF (`max_features`, `ngram_range`), `C` are selected on validation (`2023-01-01..2023-12-31`) by macro-F1. Record choice + score in `DECISION_LOG.md`.
- **Test 2024 locked until final single report.** No threshold/embargo/vectorizer/C decision may read test data or test metrics. Exactly ONE final evaluation on test (`2024-01-01..end`); no re-tuning after seeing it.
- **Tuning on test is forbidden** and counts as failure regardless of score (spec: honest near-random passes, dishonest high score fails).
- Every task touching tuning or eval MUST assert/re-affirm this discipline in its verification step.

## Global Artifact & Data-Card Contracts (binding)

- **Run artifacts:** `models/<run_id>/{vectorizer.pkl, model.pkl, config.json, data_card.md, metrics.json}` where `<run_id>` is `YYYYMMDD-HHMMSS-seed<N>-thr<X>-emb<Y>`. `config.json` stamps `flat_threshold, embargo_days, seed, TF-IDF params, C, split boundaries, package versions`.
- **`models/latest` pointer for CLI:** file or dir copy/symlink to the frozen run; CLI default `--model models/latest`, never a bare `models/tfidf_logreg.joblib` without run_id provenance.
- **`data_card.md` exact fields (every build):** sources with hash (Kaggle dataset name/version + sha256 of raw CSVs, Kraken pair `XXBTZEUR` + raw CSV hash), date ranges (raw news range, raw price range, per-split ranges), counts (raw/clean/purged: `raw_news_rows, kept_news_rows, dropped_missing_fields, duplicate_rate, dropped_no_price, dropped_no_forward, purged_train_rows, embargoed_rows`, per-split counts + per-split label distribution), params (`flat_threshold, embargo_days, seed`), versions (python, pandas, sklearn + `pip freeze` excerpt or `versions` block), reproduce command (`python -m src.dataset --flat-threshold X --embargo-days Y --seed N`).
- **Purge+embargo asserts:** train rows with `published_at > B - 24h - embargo_days` for each boundary B MUST be absent; test asserts this arithmetically on fixtures.
- **CLI exit codes:** `0` success; `2` missing/invalid input (`--title`/`--price`, non-numeric or non-positive price, empty title) with stderr naming the argument + usage; `3` model artifact missing/corrupt with stderr naming expected path; `1` unexpected internal error.
- **Mirror tests with small fixtures:** `tests/fixtures/` tiny hand-made news/price CSVs; mirror coverage for label (`test_dataset.py`), split/purge (`test_dataset.py`), CLI (`test_train_eval_cli.py`); no real data committed; no network in tests (mock Kraken).

---

## PR1 — Ingest (ingest_news + ingest_kraken + config) — target <200 lines

### PR1 RED — write failing mirror tests first

- [x] RED: Add `tests/fixtures/news_tiny.csv` + `tests/fixtures/prices_tiny.csv` (5–10 rows, tz-mixed timestamps, 1 intentional duplicate, 1 missing-field row) and failing `tests/test_ingest.py` covering UTC normalization, case/whitespace dedup, missing-field quarantine count, wrong-pair fail, polite pagination (mocked) in `tests/test_ingest.py`. <!-- sdd-owner: implementation -->
- [x] RED: Verify PR1 RED fails as expected via `python -m pytest tests/test_ingest.py -v` (new tests red, no src yet). <!-- sdd-owner: implementation -->

### PR1 GREEN — minimal implementation

- [x] GREEN: Implement `src/config.py` with `SEED=42`, `FLAT_THRESHOLD_DEFAULT=0.005`, `FLAT_THRESHOLD_CHOICES={0.003,0.005,0.01}`, `EMBARGO_DEFAULT=1`, `EMBARGO_CHOICES={1,2}`, `PAIR="XXBTZEUR"`, `train_end="2022-12-31"`, `val_end="2023-12-31"` in `src/config.py`. <!-- sdd-owner: implementation -->
- [x] GREEN: Implement `src/ingest_news.py` (require title/published_at/url → quarantine+count otherwise; published_at → UTC tz-aware; sha1(url_norm + \x00 + title_norm) dedup keep-earliest; lang report) emitting `data/interim/news_clean.parquet` per design §5.1 in `src/ingest_news.py`. <!-- sdd-owner: implementation -->
- [x] GREEN: Implement `src/ingest_kraken.py` (bulk CSV load + daily UTC validation per §5.2; incremental from max+1d with ≤720 candles/call and ≥1s sleep; pair gate `!=XXBTZEUR` aborts naming expected pair) in `src/ingest_kraken.py`. <!-- sdd-owner: implementation -->

### PR1 TRIANGULATE + REFACTOR

- [x] TRIANGULATE: Extend `tests/test_ingest.py` edge cases (Europe/Madrid naive timestamp, empty source → "unknown", numeric OHLCV coercion, de-overlap on incremental append) and confirm green via `python -m pytest tests/test_ingest.py -v` in `tests/test_ingest.py`. <!-- sdd-owner: implementation -->
- [x] REFACTOR: Deduplicate normalize/validate helpers, add `data/.gitignore` (ignore `data/raw/*`, `data/interim/*` except card template), keep PR1 diff <400 lines (`git diff --stat`) in `src/ingest_news.py`, `src/ingest_kraken.py`, `src/config.py`. <!-- sdd-owner: implementation -->

**PR1 verification:** `python -m pytest tests/test_ingest.py -v` green; manual bulk load on a local CSV copy works; rollback = delete `data/interim/news_clean.parquet`, `data/interim/prices_clean.parquet`.

---

## PR2 — Dataset + data_card (core intelligence) — target <250 lines

### PR2 RED — failing dataset mirror tests

- [x] RED: Write failing `tests/test_dataset.py` on fixtures for asof-backward (`test_asof_uses_last_candle_at_or_before`), future-candle-never-feature, buy boundary `60000→60330 @0.005 = buy`, hold `60000→60100 @0.005 = hold`, thr configurable `r=+0.008 → hold @0.01 / buy @0.005`, temporal order, purge+embargo math `T > B-24h-embargo absent`, random-split forbidden, rebuild-reproducible in `tests/test_dataset.py`. <!-- sdd-owner: implementation -->
- [x] RED: Confirm PR2 RED state via `python -m pytest tests/test_dataset.py -v` (all new tests fail, no `src/dataset.py` yet). <!-- sdd-owner: implementation -->

### PR2 GREEN — join + label + split + card

- [x] GREEN: Implement `join_asof_backward` + `label_forward` in `src/dataset.py` (sorted `merge_asof(direction='backward')` UTC; `ret_24h=close(t+24h)/close(t)-1`; strict `>` so `r==thr → hold`; invalid thr raises `ValueError` naming allowed values; drop+count `dropped_no_price`/`dropped_no_forward`; stamp `flat_threshold` per row). <!-- sdd-owner: implementation -->
- [x] GREEN: Implement `temporal_split` + `DatasetConfig` in `src/dataset.py` (chronological train 2018-2022 / val 2023 / test 2024+, overridable `train_end`/`val_end`; purge `T > B-24h` + embargo `T > B-24h-embargo_days` at both boundaries with `purged_train_rows`/`embargoed_rows` counts; `embargo_days ∈ {1,2}` else `ValueError`; `split_mode != "temporal"` → explicit failure). <!-- sdd-owner: implementation -->
- [x] GREEN: Implement `build_dataset(config)` + `data_card.md` writer in `src/dataset.py` emitting full contract fields (sources+hashes, ranges, raw/clean/purged counts, params, versions, split boundaries + purge cutoffs + gaps + per-split counts/label distribution, rebuild command `python -m src.dataset --flat-threshold X --embargo-days Y`) to `data/interim/data_card.md` and `models/<run_id>/data_card.md` copy hook. <!-- sdd-owner: implementation -->

### PR2 TRIANGULATE + REFACTOR

- [x] TRIANGULATE: Add boundary/edge tests (exact-equality `r==thr → hold`, sell mirror `r<-thr`, embargo=2 vs 1 row-count delta, news-before-first-candle dropped) and confirm `python -m pytest tests/test_ingest.py tests/test_dataset.py -v` green in `tests/test_dataset.py`. <!-- sdd-owner: implementation -->
- [x] REFACTOR: Consolidate leakage-sensitive code in `src/dataset.py` only (no join/label logic elsewhere), enforce temporal-discipline docstring (train learns / val chooses / test locked / no test tuning), keep PR2 diff <400 lines in `src/dataset.py`. <!-- sdd-owner: implementation -->

**PR2 verification:** `python -m pytest tests/test_dataset.py -v` green; `build_dataset` on fixtures reproduces identical labels/splits + card on rebuild; rollback = delete `data/interim/labeled.parquet`, `data/interim/data_card.md`.

---

## PR3 — Modelo + eval + run artifacts — target <250 lines

### PR3 RED — failing train/eval tests

- [x] RED: Write failing `tests/test_train_eval.py` (balanced weights set, seed reproducibility, eval requires per-class P/R + confusion matrix and rejects accuracy-only) in `tests/test_train_eval.py`. <!-- sdd-owner: implementation -->
- [x] RED: Confirm PR3 RED via `python -m pytest tests/test_train_eval.py -v` (failures, no `src/train.py`/`src/evaluate.py` yet). <!-- sdd-owner: implementation -->

### PR3 GREEN — model + eval

- [x] GREEN: Implement `src/train.py` (TF-IDF title+body `max_features 5k–20k, ngram (1,2), sublinear_tf=True` + `LogisticRegression(class_weight='balanced', random_state=SEED, max_iter=1000, solver='lbfgs')` CPU-only; train fit on 2018-2022 only; val-2023 loop over thr/embargo/TFIDF/C by macro-F1 with `DECISION_LOG.md` entry; freeze best; write `models/<run_id>/{vectorizer.pkl, model.pkl, config.json, metrics.json}` + refresh `models/latest`; single final test-2024 report only). <!-- sdd-owner: implementation -->
- [x] GREEN: Implement `src/evaluate.py` (per-class precision/recall + macro-F1 + 3×3 confusion matrix buy/hold/sell; accuracy-alone → incomplete/fail gate; near-random+honest = PASS rule documented) in `src/evaluate.py`. <!-- sdd-owner: implementation -->

### PR3 TRIANGULATE + REFACTOR

- [x] TRIANGULATE: Add eval edge tests (hold-dominance, single-class val fold) and run `python -m pytest tests/test_train_eval.py -v` green in `tests/test_train_eval.py`. <!-- sdd-owner: implementation -->
- [x] REFACTOR: Keep PR3 diff <400 lines in `src/train.py`, `src/evaluate.py`. <!-- sdd-owner: implementation -->

**PR3 verification:** `python -m pytest tests/test_train_eval.py -v` green; `models/<run_id>/{vectorizer.pkl, model.pkl, config.json, metrics.json}` exists; rollback = delete `models/<run_id>/` + restore prior `models/latest`.

---

## PR4 — CLI + docs + portfolio — target <200 lines

### PR4 RED — failing CLI tests

- [x] RED: Write failing `tests/test_cli.py` (CLI happy path action+probs+disclaimer exit 0, missing/invalid input exit 2, missing artifact exit 3, never-retrains via mtime/hash guard) in `tests/test_cli.py`. <!-- sdd-owner: implementation -->
- [x] RED: Confirm PR4 RED via `python -m pytest tests/test_cli.py -v` (failures, no `src/cli.py` yet). <!-- sdd-owner: implementation -->

### PR4 GREEN — CLI + docs

- [x] GREEN: Implement `src/cli.py` + `btc-signal` entry in `pyproject.toml` (inference-only read-only loads from `models/latest`; `--title --price [--model --json]`; stdout action+probs+price_ctx+DISCLAIMER; exit 2/3/1 per contract; `--help` notes price is display context not a trained feature; never writes `data/`/`models/`) in `src/cli.py`, `pyproject.toml`. <!-- sdd-owner: implementation -->

### PR4 TRIANGULATE + REFACTOR + audit

- [x] TRIANGULATE: Add CLI edge tests (non-numeric/zero/negative price → exit 2 with usage, corrupt model → exit 3 naming path, probs sum 1.0±1e-6, `--json` shape if implemented) and run full `python -m pytest -v` green in `tests/test_cli.py`. <!-- sdd-owner: implementation -->
- [x] REFACTOR + audit: Append `DECISION_LOG.md` (Kaggle-vs-GDELT, thr/embargo val-only choice, TF-IDF/C choice, single test report, CLI inference-only), write portfolio README section, verify `models/<run_id>/` completeness + `models/latest` resolves + `data_card.md` exact fields present, full `python -m pytest` + `btc-signal --title "Bitcoin ETF inflows hit record" --price 60000` demo, keep PR4 diff <400 lines in `src/cli.py`, `pyproject.toml`, `DECISION_LOG.md`. <!-- sdd-owner: implementation -->

**PR4 verification:** `python -m pytest -v` all green; `btc-signal` happy path exit 0 with disclaimer; rollback = delete CLI entry + restore prior `models/latest`.

---

## Definition of Done (slice gate)

- [x] Final check: full `python -m pytest` green, temporal asserts hold, `data_card.md` exact fields + reproduce command verified by fixture rebuild, CLI 2/3 codes demonstrated, test-2024 evaluated exactly once with per-class report, all within chained-PR budget in `openspec/changes/btc-news-signal/tasks.md`. <!-- sdd-owner: implementation -->

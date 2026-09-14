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

---

# Apply progress — btc-news-signal — PR2 (dataset, stacked-to-main PR2/4, branch pr2-label-split)

Scope: PR2 ONLY (PR1 untouched, PR3 model/eval and PR4 CLI untouched).

## Completed (PR2 RED → GREEN → TRIANGULATE → REFACTOR)

- `tests/test_dataset.py`: 13 tests (asof-backward, future-candle-never-feature, buy 60000→60330 @0.005, hold 60000→60100 @0.005, thr configurable r=+0.008 hold @0.01/buy @0.005 + invalid-thr ValueError, temporal order, purge+embargo math T>B-24h-embargo absent + invalid-embargo ValueError, random-split forbidden, rebuild-reproducible + card exact-fields; triangulate: r==thr hold, sell mirror, embargo 2 vs 1 delta, news-before-first-candle dropped).
- `src/dataset.py`: `DatasetConfig` (thr 0.005∈{0.003,0.005,0.01}, embargo 1∈{1,2}, split_mode temporal-only, seed 42, train_end/val_end overridable) + `join_asof_backward` (sorted merge_asof backward UTC, close_t24 via +1d lookup) + `label_forward` (ret_24h, strict > so r==thr holds, invalid thr ValueError naming allowed, drop+count, stamp flat_threshold) + `temporal_split` (train 2018-2022/val 2023/test 2024+ defaults, purge T>B-24h + embargo T>B-24h-embargo_days at both boundaries, purged_train_rows/embargoed_rows, embargo∈{1,2} else ValueError, split_mode!=temporal explicit failure) + `build_dataset` + data_card.md writer with exact contract fields + `models/<run_id>/data_card.md` copy hook + `python -m src.dataset` CLI.
- Temporal discipline affirmed: train learns / val chooses / test locked / no test tuning (module docstring + val-only thr/embargo, single final test report deferred to PR3).
- Leakage consolidation: `merge_asof`/`ret_24h`/label/split logic only in `src/dataset.py` (grep verified, no join/label in ingest/config).

## TDD Cycle Evidence

| Cycle | Command | Result |
|---|---|---|
| RED | `python -m pytest tests/test_dataset.py -v` (no src/dataset.py) | collection ERROR, `No module named 'src.dataset'` |
| GREEN | same (after `src/dataset.py` + 1 fix: f-string `{{}}` → `dict()` for empty label dist) | 9 passed |
| TRIANGULATE | `python -m pytest tests/test_ingest.py tests/test_dataset.py -v` (4 edge tests added) | 22 passed (9 ingest + 13 dataset) |
| REFACTOR | same (no code change needed; grep confirms single-owner + docstring already enforces discipline) | 22 passed |

## Verification

- `python -m pytest tests/test_ingest.py tests/test_dataset.py -v` → 22 passed.
- Fixture rebuild: `build_dataset(DatasetConfig(train_end="2024-01-17", val_end="2024-01-19"))` twice on `news_tiny`/`prices_tiny` → identical train/val/test frames + card contains `raw_news_rows, flat_threshold, embargo_days, seed, split boundaries, rebuild command, dropped_no_price, dropped_no_forward, purged_train_rows, embargoed_rows`.
- Purge math asserted: train rows with `T > B-24h-embargo` absent; embargo=2 strictly removes more than embargo=1 on probe set; `split_mode="random"/"stratified-shuffle"` → ValueError naming chronological/temporal.
- Size: 472 new lines (`src/dataset.py` 235 + `tests/test_dataset.py` 237 via `wc -l`) — exceeds 400-line max and 250-line PR2 target. Cannot shrink without deleting required tests/contract fields/docs (forbidden by budget rule); recommend `size:exception` for PR2 or accept stacked-PR boundary as-is. `src/dataset.py` alone 235 lines is within 400.
- Rollback PR2: delete `src/dataset.py`, `tests/test_dataset.py`, `data/interim/data_card.md` (+ optional `models/<run_id>/data_card.md` copy).

## Deviations

- `build_dataset` accepts optional `news_df`/`prices_df` (default loads `data/interim/*.parquet`) to enable deterministic fixture rebuild tests; CLI `python -m src.dataset --flat-threshold X --embargo-days Y --seed N` satisfies reproduce command.
- `dropped_no_price`/`dropped_no_forward` computed in `build_dataset` from join/label length deltas (label drops both classes); `purged_train_rows` = rows with `T>B-24h` across both boundaries, `embargoed_rows` = embargo-only strip `(B-24h-embargo, B-24h]`.
- `tasks.md` checkboxes NOT ticked: file is outside this run's allowed edit surfaces; parent owns the update.
- Generated `data/interim/data_card.md` left in working tree as build evidence (test side effect, not a source edit).

## Remaining (out of scope for this run)

- PR3: `tests/test_train_eval.py` + `src/train.py` + `src/evaluate.py` + run artifacts. PR4: CLI/docs/portfolio + final slice gate.
- Structured status: change `btc-news-signal`, artifactStore `openspec`, chain `stacked-to-main` PR2/4, `actionContext.mode=repo-local`, allowedEditRoots respected (only `src/dataset.py`, `tests/test_dataset.py`, `apply-progress.md` edited), no warnings.

---

# Apply progress — btc-news-signal — PR3 (model+eval, stacked-to-main PR3/4, branch pr2-label-split)

Scope: PR3 ONLY (PR1/PR2 untouched, PR4 CLI/docs untouched).

## Completed (PR3 RED → GREEN → TRIANGULATE → REFACTOR)

- `tests/test_train_eval.py`: 9 tests (balanced weights + SEED/max_iter/solver contract, TF-IDF max_features 5k–20k + ngram(1,2) + sublinear + invalid-max_features ValueError, seed reproducibility via coef_ + predict equality, train-fits-train-only vocab guard, combine_text title+body, eval per-class P/R + macro-F1 + 3x3 matrix + gate pass; triangulate: hold-dominance eval, single-class val fold macro_F1≈1/3).
- `src/train.py` (84 lines): `combine_text`/`texts_from_df` (title+body, price NOT a feature) + `make_vectorizer` (5k–20k validated, ngram (1,2), sublinear True) + `train_baseline` (TF-IDF fit on train texts only + LogisticRegression balanced/random_state=SEED/max_iter=1000/solver=lbfgs/C) + `make_run_id` (YYYYMMDD-HHMMSS-seedN-thrX-embY) + `save_run` (models/<run_id>/{vectorizer.pkl, model.pkl, config.json, metrics.json} + models/latest pointer).
- `src/evaluate.py` (64 lines): `evaluate_predictions` (per-class precision/recall/F1/support via zero_division=0, macro-F1, accuracy alongside only, 3x3 confusion matrix rows=true cols=pred order buy/hold/sell) + `check_report_complete` (accuracy-alone → ValueError naming per-class + confusion matrix; non-3x3 or missing buy/hold/sell → ValueError) + near-random+honest=PASS rule in docstring.
- Temporal discipline affirmed: vectorizer+model fit on train only (val-only-token vocab test), val-2023 chooses thr/embargo/TFIDF/C by macro-F1 (evaluate_predictions), single final test-2024 report deferred (no test read in PR3), no leakage logic outside `src/dataset.py` (grep verified: no merge_asof/ret_24h in train/evaluate).

## TDD Cycle Evidence

| Cycle | Command | Result |
|---|---|---|
| RED | `python -m pytest tests/test_train_eval.py -v` (no src/train.py/src/evaluate.py) | collection ERROR, `No module named 'src.evaluate'` |
| GREEN | same (after `src/train.py` + `src/evaluate.py` + 1 fix: texts_from_df `[\"\"]*len` typo) | 9 passed |
| TRIANGULATE | `python -m pytest tests/test_ingest.py tests/test_dataset.py tests/test_train_eval.py -v` (hold-dominance + single-class fold verified, no code change) | 31 passed (9 ingest + 13 dataset + 9 train/eval) |
| REFACTOR | same (no duplication to remove; train vs eval concerns separated; grep confirms leakage code only in dataset.py) | 31 passed |

## Verification

- `python -m pytest tests/test_ingest.py tests/test_dataset.py tests/test_train_eval.py -v` → 31 passed.
- Smoke (tempdir, no repo pollution): toy 12-row fit → preds [buy, hold] sane, macro_F1 1.0 on separable toy, `save_run` writes {vectorizer.pkl, model.pkl, config.json, metrics.json} + `models/latest` pointer resolves to run_id `YYYYMMDD-HHMMSS-seed42-thr0.005-emb1`.
- Size: 270 new lines (`src/train.py` 84 + `src/evaluate.py` 64 + `tests/test_train_eval.py` 122 via `wc -l`) — over PR3 <250 target by 20 lines but within 400-line max; cannot shrink without deleting required contract tests/guards (forbidden by budget rule). No commit/push/PR per instructions.
- Rollback PR3: delete `src/train.py`, `src/evaluate.py`, `tests/test_train_eval.py` (+ optional `models/<run_id>/` + `models/latest` if a real run was saved; smoke used tempdir so none created).

## Deviations

- `tasks.md` checkboxes NOT ticked: file is outside this run's allowed edit surfaces; parent owns the update.
- `DECISION_LOG.md` NOT appended: file is outside allowed edit surfaces; val-loop choice (thr/embargo/TFIDF/C by macro-F1) is implemented via `train_baseline` + `evaluate_predictions` return values ready for the parent/PR4 logging step. No test-peeking: PR3 performs no test-2024 read.
- Deps installed into environment: `scikit-learn 1.9.1, scipy, joblib` (were absent; `numpy/pandas/pytest` already present). No `pyproject.toml` touched (absent from repo; PR4 owns packaging).

## Remaining (out of scope for this run)

- PR4: `tests/test_cli.py` + `src/cli.py` + `btc-signal` entry + DECISION_LOG/portfolio/README + final slice gate (full `pytest`, CLI 2/3 codes, single test-2024 report).
- Structured status: change `btc-news-signal`, artifactStore `openspec`, chain `stacked-to-main` PR3/4 branch `pr2-label-split`, `actionContext.mode=repo-local`, allowedEditRoots respected (only `src/train.py`, `src/evaluate.py`, `tests/test_train_eval.py`, `apply-progress.md` edited), delivery `ask-on-risk` with `size:within-budget` (270 < 400 max), strict TDD followed, no warnings.

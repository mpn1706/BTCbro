```yaml
schema: gentle-ai.verify-result/v1
evidence_revision: sha256:3deac8b8a15916cbe7deff3926d63f319053b94f87aa6f224770354d61496fe7
verdict: fail
blockers: 26
critical_findings: 26
requirements: 8/8
scenarios: 23/23
test_command: python -m pytest tests/ -v
test_exit_code: 0
test_output_hash: sha256:afd0b3606d2b5df39895f2edd84e7520c04f120e01b08a3e2a5eaef07967349b
build_command: python -c "import src.config, src.ingest_news, src.ingest_kraken, src.dataset, src.train, src.evaluate, src.cli; print('imports ok')"
build_exit_code: 0
build_output_hash: sha256:47233ac20971137b78bf82759638a60672195955c3eb57326d48e2ead5c2b6b5
```

# Verify — btc-news-signal (full 4-PR slice, branch pr2-label-split)

**Status: FAIL — archive blocked (procedural, not functional).**
Code and tests for all 8 requirements verified by reading implementation and re-running tests (42 passed).
FAIL is required because `openspec/changes/btc-news-signal/tasks.md` still shows 0/26 implementation
checkboxes checked (parent owns the file; apply-progress records implementation as done but checkboxes
were never ticked due to allowedEditRoots). Per verify contract, unchecked implementation tasks are
CRITICAL archive blockers — no clean PASS while any `- [ ]` remains, even when code proves the work.
Stale-checkbox reconciliation is proven below, but it does not convert this to a clean pass.

Branch verified: `pr2-label-split` (stacked-to-main PR4/4 per apply-progress).
Workspace root: `C:/Users/usuario/Documents/ARCHIVOS DOMÉSTICOS/CURSO EOI/José Antonio/Proyecto Bitcoin`
Artifact store: `openspec`. Date (UTC): 2026-09-14.

## 1. Spec coverage — REQ-1..REQ-8 (all PASS by code + test evidence)

Requirement totals for envelope: 8 requirements, 23 scenarios (counted from
`openspec/changes/btc-news-signal/specs/btc-news-signal/spec.md`).

| REQ | Result | Implementation evidence (read) | Test evidence (re-run, all green) |
|-----|--------|--------------------------------|-----------------------------------|
| REQ-1 News ingest schema + dedup | PASS | `src/ingest_news.py`: `REQUIRED_FIELDS=(title,published_at,url)` quarantine+count; `_to_utc` naive→Europe/Madrid→UTC, offsets honored; `sha1(url_norm+\x00+title_norm)` keep-earliest; `lang` per-record + `lang_distribution`; emits `news_clean.parquet` (§5.1 cols incl. `dedup_hash`) | `test_news_timestamp_to_utc` (12:00+01:00→11:00Z, tz-aware UTC dtype), `test_dedup_case_whitespace` (earliest wins, dup≥1), `test_missing_field_quarantined` (5 kept = 7−1 missing−1 dup, quarantine==dropped), `test_naive_madrid_timestamp_to_utc` (15:30 Madrid→14:30Z), `test_empty_source_becomes_unknown` |
| REQ-2 Price ingest Kraken XXBTZEUR daily OHLC | PASS | `src/ingest_kraken.py`: `validate_pair` aborts naming `XXBTZEUR`; `clean_prices` daily UTC-normalized + numeric OHLCV coercion (`>0` open/high/low/close, `>=0` volume); `incremental_update` from max+1d, `MAX_PER_CALL=720`, `SLEEP_SECS=1`, de-overlap via timestamp anti-join | `test_wrong_pair_fails` (XXBTZUSD + BTC/USD raise naming XXBTZEUR), `test_incremental_pagination_polite` (mocked: count≤720, sleep≥1s, +1 row), `test_ohlcv_numeric_coercion` (string→float, UTC), `test_incremental_deoverlap_on_append` (overlap+new → new_rows==1, unique timestamps) |
| REQ-3 Backward join + 24h forward label | PASS | `src/dataset.py:join_asof_backward` sorted `merge_asof(direction='backward')` UTC, `close_t24` via +1d lookup; `label_forward` `ret_24h=close(t+24h)/close(t)-1`, strict `>` so `r==thr→hold`, stamps `flat_threshold` per row, drops+counts; `src/config.py`: `FLAT_THRESHOLD_DEFAULT=0.005`, `CHOICES={0.003,0.005,0.01}` | `test_asof_uses_last_candle_at_or_before` (C0=60800), `test_future_candle_never_feature` (C1 61000→99999 leaves feature 60000), `test_label_buy_boundary` (60000→60330@0.005=buy, r≈0.0055), `test_label_hold_inside` (60000→60100=hold), `test_threshold_configurable` (r=+0.008 hold@0.01/buy@0.005 + invalid 0.007 ValueError naming allowed), `test_exact_equality_holds`, `test_sell_mirror` (60000→59400=sell), `test_news_before_first_candle_dropped` |
| REQ-4 Chronological split + purge + embargo | PASS | `src/dataset.py:temporal_split` + `DatasetConfig(train_end=2022-12-31,val_end=2023-12-31,overridable)`: train<val<test modulo gap; purge `T>B-24h` + embargo `T>B-24h-embargo_days` at both boundaries with `purged_train_rows`/`embargoed_rows`; `embargo∈{1,2}` else ValueError; `split_mode!="temporal"` explicit failure naming chronological/temporal | `test_temporal_order` (train<val<test), `test_purge_plus_embargo_math` (cutoff 2022-12-30T00Z, embargo=5 ValueError), `test_random_split_forbidden` (random + stratified-shuffle → ValueError chronological/temporal), `test_embargo2_vs_1_delta` (emb2 strictly fewer than emb1) |
| REQ-5 Baseline TF-IDF + LogReg | PASS | `src/train.py`: `combine_text` title+body (price NOT a feature); `make_vectorizer` 5k–20k validated, ngram(1,2), sublinear True; `train_baseline` TF-IDF fit on passed texts only + `LogisticRegression(class_weight='balanced',random_state=SEED(42),max_iter=1000,solver='lbfgs')` CPU-only; `make_run_id` YYYYMMDD-HHMMSS-seedN-thrX-embY; `save_run` writes `models/<run_id>/{vectorizer.pkl,model.pkl,config.json,metrics.json}` + `models/latest` pointer | `test_balanced_weights_set` (balanced/SEED/max_iter/solver + 3 classes exposed under hold-dominance), `test_tfidf_params_in_contract` (ngram/sublinear/range + 500→ValueError), `test_seed_reproducible` (coef_ + predict equality), `test_train_fits_train_only` (valonlytokenxyz absent, common present), `test_combine_text_title_body` |
| REQ-6 Honest per-class eval | PASS | `src/evaluate.py`: `evaluate_predictions` per-class P/R/F1/support (zero_division=0), macro-F1, accuracy alongside only, 3×3 matrix rows=true cols=pred order buy/hold/sell; `check_report_complete` accuracy-alone→ValueError naming per-class+matrix, non-3×3 or missing label→ValueError; near-random+honest=PASS rule in docstring | `test_eval_requires_per_class` (P/R/macro-F1/3×3 + gate pass, sum==n), `test_accuracy_only_rejected` ({"accuracy":0.9}→ValueError), `test_hold_dominance_eval` (18 hold+buy+sell all-held → 3 entries, gate pass), `test_single_class_val_fold` (macro_F1≈1/3) |
| REQ-7 Inference CLI + disclaimer | PASS | `src/cli.py` (123 lines): `main(argv)->int`, parser `--title --price [--body --model --json]`, `--help` notes price display-context-only; `resolve_run_dir` latest-pointer→run dir; `load_artifacts` read-only, missing/corrupt→exit 3 naming path; `predict_probs` via `combine_text` (price never vectorized); stdout action+probs+price_ctx+DISCLAIMER, `--json` variant; exits 0/2/3/1 per contract; `pyproject.toml` `btc-signal = "src.cli:main"` | `test_cli_happy_path` (exit 0, action+3 probs+DISCLAIMER/not-investment-advice), `test_cli_missing_title_exit2`, `test_cli_invalid_price_exit2`, `test_cli_missing_artifact_exit3` (names path), `test_cli_never_retrains` (mtime+hash unchanged), `test_cli_bad_prices_exit2` (abc/0/-5/empty→2), `test_cli_corrupt_model_exit3` (names dir), `test_cli_probs_sum_to_one` (±1e-6), `test_cli_json_shape` (action/probs/disclaimer), `test_cli_help_notes_price_context`, `test_cli_latest_pointer_resolves` — plus live re-check: `--price abc`→exit 2 naming price+usage; missing model→exit 3 naming path; `--help` shows display-context note |
| REQ-8 Reproducibility via data_card | PASS (tasks contract; 2 design extras missing — see §7 WARNING) | `src/dataset.py:build_dataset` + card writer to `data/interim/data_card.md` + `models/<run_id>/data_card.md` copy hook + `python -m src.dataset --flat-threshold X --embargo-days Y --seed N` CLI; card verified present with all tasks-contract fields (see §7 audit) | `test_rebuild_reproducible` (fixture rebuild twice → identical train/val/test frames + card contains raw_news_rows, flat_threshold, embargo_days, seed, split boundaries, rebuild command, dropped_no_price, dropped_no_forward, purged_train_rows, embargoed_rows) |

Scenario count check: 3+3+4+3+2+3+3+2 = 23. All 23 scenarios above map to at least one passing test; no scenario is uncovered.

## 2. Valid-chooses / test-certifies rule (PASS)

Tunable params are never hardcoded; invalid values fail loudly and tests certify the gate.
Live re-verified 2026-09-14 (not trusting prior reports):

- `flat_threshold ∈ {0.003, 0.005, 0.01}` default 0.005 (`src/config.py`, `DatasetConfig`): `label_forward(...,0.007)` → `ValueError: invalid flat_threshold 0.007: allowed values are [0.003, 0.005, 0.01]`; certified by `test_threshold_configurable`.
- `embargo_days ∈ {1, 2}` default 1: `temporal_split(...,embargo_days=5)` → `ValueError: invalid embargo_days 5: allowed values are [1, 2]`; certified by `test_purge_plus_embargo_math`.
- `split_mode == "temporal"` only: `"random"` / `"stratified-shuffle"` → `ValueError: only chronological temporal split ... (random/shuffle forbidden)`; certified by `test_random_split_forbidden`.
- `max_features ∈ [5000, 20000]`: `make_vectorizer(500)` → `ValueError` naming max_features; certified by `test_tfidf_params_in_contract`.
- Val-chooses / test-locked: `train_baseline` fits TF-IDF + LogReg on caller-passed train texts only (certified by `test_train_fits_train_only` vocab guard); no test-2024 read exists in `src/train.py`/`src/evaluate.py`; `DECISION_LOG.md` D2/D4 records val-only selection and single final test report. No test-peeking code found (grep: leakage tokens only in `src/dataset.py`).

## 3. Test counts (re-run, not trusted)

- Command: `python -m pytest tests/ -v` → exit 0, **42 passed** (9 ingest + 13 dataset + 9 train/eval + 11 CLI), 0 failed/skipped.
- Observed output hash `sha256:afd0b3606d2b5df39895f2edd84e7520c04f120e01b08a3e2a5eaef07967349b` covers the full 42-line run ending `42 passed in 3.60s` (timing varies per run; content otherwise identical to prior 42-passed reports).
- Build/import check: `python -c "import src.config, src.ingest_news, src.ingest_kraken, src.dataset, src.train, src.evaluate, src.cli; print('imports ok')"` → exit 0, `imports ok`, hash `sha256:47233ac20971137b78bf82759638a60672195955c3eb57326d48e2ead5c2b6b5`.
- Full output saved by verifier run; `tests/fixtures/{news_tiny.csv (7 rows), prices_tiny.csv (6 rows)}` present with tz-mixed, dup, missing-field rows as required.

## 4. Task completion status (26 unchecked — CRITICAL, parent owns)

`openspec/changes/btc-news-signal/tasks.md` shows 26/26 implementation tasks unchecked (`grep -c "^ - [ ]"` = 26).
Per contract each is a CRITICAL completeness issue and archive blocker. Exact unchecked lines (verbatim):

- [ ] RED: Add `tests/fixtures/news_tiny.csv` + `tests/fixtures/prices_tiny.csv` (5–10 rows, tz-mixed timestamps, 1 intentional duplicate, 1 missing-field row) and failing `tests/test_ingest.py` covering UTC normalization, case/whitespace dedup, missing-field quarantine count, wrong-pair fail, polite pagination (mocked) in `tests/test_ingest.py`. <!-- sdd-owner: implementation -->
- [ ] RED: Verify PR1 RED fails as expected via `python -m pytest tests/test_ingest.py -v` (new tests red, no src yet). <!-- sdd-owner: implementation -->
- [ ] GREEN: Implement `src/config.py` with `SEED=42`, `FLAT_THRESHOLD_DEFAULT=0.005`, `FLAT_THRESHOLD_CHOICES={0.003,0.005,0.01}`, `EMBARGO_DEFAULT=1`, `EMBARGO_CHOICES={1,2}`, `PAIR="XXBTZEUR"`, `train_end="2022-12-31"`, `val_end="2023-12-31"` in `src/config.py`. <!-- sdd-owner: implementation -->
- [ ] GREEN: Implement `src/ingest_news.py` (require title/published_at/url → quarantine+count otherwise; published_at → UTC tz-aware; sha1(url_norm + \x00 + title_norm) dedup keep-earliest; lang report) emitting `data/interim/news_clean.parquet` per design §5.1 in `src/ingest_news.py`. <!-- sdd-owner: implementation -->
- [ ] GREEN: Implement `src/ingest_kraken.py` (bulk CSV load + daily UTC validation per §5.2; incremental from max+1d with ≤720 candles/call and ≥1s sleep; pair gate `!=XXBTZEUR` aborts naming expected pair) in `src/ingest_kraken.py`. <!-- sdd-owner: implementation -->
- [ ] TRIANGULATE: Extend `tests/test_ingest.py` edge cases (Europe/Madrid naive timestamp, empty source → "unknown", numeric OHLCV coercion, de-overlap on incremental append) and confirm green via `python -m pytest tests/test_ingest.py -v` in `tests/test_ingest.py`. <!-- sdd-owner: implementation -->
- [ ] REFACTOR: Deduplicate normalize/validate helpers, add `data/.gitignore` (ignore `data/raw/*`, `data/interim/*` except card template), keep PR1 diff <400 lines (`git diff --stat`) in `src/ingest_news.py`, `src/ingest_kraken.py`, `src/config.py`. <!-- sdd-owner: implementation -->
- [ ] RED: Write failing `tests/test_dataset.py` on fixtures for asof-backward (`test_asof_uses_last_candle_at_or_before`), future-candle-never-feature, buy boundary `60000→60330 @0.005 = buy`, hold `60000→60100 @0.005 = hold`, thr configurable `r=+0.008 → hold @0.01 / buy @0.005`, temporal order, purge+embargo math `T > B-24h-embargo absent`, random-split forbidden, rebuild-reproducible in `tests/test_dataset.py`. <!-- sdd-owner: implementation -->
- [ ] RED: Confirm PR2 RED state via `python -m pytest tests/test_dataset.py -v` (all new tests fail, no `src/dataset.py` yet). <!-- sdd-owner: implementation -->
- [ ] GREEN: Implement `join_asof_backward` + `label_forward` in `src/dataset.py` (sorted `merge_asof(direction='backward')` UTC; `ret_24h=close(t+24h)/close(t)-1`; strict `>` so `r==thr → hold`; invalid thr raises `ValueError` naming allowed values; drop+count `dropped_no_price`/`dropped_no_forward`; stamp `flat_threshold` per row). <!-- sdd-owner: implementation -->
- [ ] GREEN: Implement `temporal_split` + `DatasetConfig` in `src/dataset.py` (chronological train 2018-2022 / val 2023 / test 2024+, overridable `train_end`/`val_end`; purge `T > B-24h` + embargo `T > B-24h-embargo_days` at both boundaries with `purged_train_rows`/`embargoed_rows` counts; `embargo_days ∈ {1,2}` else `ValueError`; `split_mode != "temporal"` → explicit failure). <!-- sdd-owner: implementation -->
- [ ] GREEN: Implement `build_dataset(config)` + `data_card.md` writer in `src/dataset.py` emitting full contract fields (sources+hashes, ranges, raw/clean/purged counts, params, versions, split boundaries + purge cutoffs + gaps + per-split counts/label distribution, rebuild command `python -m src.dataset --flat-threshold X --embargo-days Y`) to `data/interim/data_card.md` and `models/<run_id>/data_card.md` copy hook. <!-- sdd-owner: implementation -->
- [ ] TRIANGULATE: Add boundary/edge tests (exact-equality `r==thr → hold`, sell mirror `r<-thr`, embargo=2 vs 1 row-count delta, news-before-first-candle dropped) and confirm `python -m pytest tests/test_ingest.py tests/test_dataset.py -v` green in `tests/test_dataset.py`. <!-- sdd-owner: implementation -->
- [ ] REFACTOR: Consolidate leakage-sensitive code in `src/dataset.py` only (no join/label logic elsewhere), enforce temporal-discipline docstring (train learns / val chooses / test locked / no test tuning), keep PR2 diff <400 lines in `src/dataset.py`. <!-- sdd-owner: implementation -->
- [ ] RED: Write failing `tests/test_train_eval.py` (balanced weights set, seed reproducibility, eval requires per-class P/R + confusion matrix and rejects accuracy-only) in `tests/test_train_eval.py`. <!-- sdd-owner: implementation -->
- [ ] RED: Confirm PR3 RED via `python -m pytest tests/test_train_eval.py -v` (failures, no `src/train.py`/`src/evaluate.py` yet). <!-- sdd-owner: implementation -->
- [ ] GREEN: Implement `src/train.py` (TF-IDF title+body `max_features 5k–20k, ngram (1,2), sublinear_tf=True` + `LogisticRegression(class_weight='balanced', random_state=SEED, max_iter=1000, solver='lbfgs')` CPU-only; train fit on 2018-2022 only; val-2023 loop over thr/embargo/TFIDF/C by macro-F1 with `DECISION_LOG.md` entry; freeze best; write `models/<run_id>/{vectorizer.pkl, model.pkl, config.json, metrics.json}` + refresh `models/latest`; single final test-2024 report only). <!-- sdd-owner: implementation -->
- [ ] GREEN: Implement `src/evaluate.py` (per-class precision/recall + macro-F1 + 3×3 confusion matrix buy/hold/sell; accuracy-alone → incomplete/fail gate; near-random+honest = PASS rule documented) in `src/evaluate.py`. <!-- sdd-owner: implementation -->
- [ ] TRIANGULATE: Add eval edge tests (hold-dominance, single-class val fold) and run `python -m pytest tests/test_train_eval.py -v` green in `tests/test_train_eval.py`. <!-- sdd-owner: implementation -->
- [ ] REFACTOR: Keep PR3 diff <400 lines in `src/train.py`, `src/evaluate.py`. <!-- sdd-owner: implementation -->
- [ ] RED: Write failing `tests/test_cli.py` (CLI happy path action+probs+disclaimer exit 0, missing/invalid input exit 2, missing artifact exit 3, never-retrains via mtime/hash guard) in `tests/test_cli.py`. <!-- sdd-owner: implementation -->
- [ ] RED: Confirm PR4 RED via `python -m pytest tests/test_cli.py -v` (failures, no `src/cli.py` yet). <!-- sdd-owner: implementation -->
- [ ] GREEN: Implement `src/cli.py` + `btc-signal` entry in `pyproject.toml` (inference-only read-only loads from `models/latest`; `--title --price [--model --json]`; stdout action+probs+price_ctx+DISCLAIMER; exit 2/3/1 per contract; `--help` notes price is display context not a trained feature; never writes `data/`/`models/`) in `src/cli.py`, `pyproject.toml`. <!-- sdd-owner: implementation -->
- [ ] TRIANGULATE: Add CLI edge tests (non-numeric/zero/negative price → exit 2 with usage, corrupt model → exit 3 naming path, probs sum 1.0±1e-6, `--json` shape if implemented) and run full `python -m pytest -v` green in `tests/test_cli.py`. <!-- sdd-owner: implementation -->
- [ ] REFACTOR + audit: Append `DECISION_LOG.md` (Kaggle-vs-GDELT, thr/embargo val-only choice, TF-IDF/C choice, single test report, CLI inference-only), write portfolio README section, verify `models/<run_id>/` completeness + `models/latest` resolves + `data_card.md` exact fields present, full `python -m pytest` + `btc-signal --title "Bitcoin ETF inflows hit record" --price 60000` demo, keep PR4 diff <400 lines in `src/cli.py`, `pyproject.toml`, `DECISION_LOG.md`. <!-- sdd-owner: implementation -->
- [ ] Final check: full `python -m pytest` green, temporal asserts hold, `data_card.md` exact fields + reproduce command verified by fixture rebuild, CLI 2/3 codes demonstrated, test-2024 evaluated exactly once with per-class report, all within chained-PR budget in `openspec/changes/btc-news-signal/tasks.md`. <!-- sdd-owner: implementation -->

Stale-checkbox assessment (honest): every item above is functionally present in the working tree
(fixtures + 4 test files + 7 src modules + card + CLI entry + docs, all re-read and re-executed here;
apply-progress PR1→PR4 documents RED→GREEN→TRIANGULATE→REFACTOR with commands and sizes).
Checkboxes remain unchecked only because `tasks.md` was outside each apply run's allowedEditRoots
(parent owns the update — stated in all four apply-progress Deviations). This qualifies as
stale-checkbox reconciliation proven by apply-progress + this report, but per contract it still
blocks archive until the parent ticks the boxes. Verifier did NOT modify `tasks.md`.

## 5. Structured status and actionContext

- Native status consumed: change `btc-news-signal`, artifactStore `openspec`, `tasks 0/26 complete`,
  `applyState ready`, `dependencies.verify ready / sync blocked / archive blocked`, `nextRecommended sdd-apply`
  (status is authoritative; verify ran because verify is `ready`).
- `actionContext.mode=repo-local`, workspaceRoot as above, allowedEditRoots respected (verify wrote only
  `openspec/changes/btc-news-signal/verify.md` + `verify-report.md` mirror; no src/tests/data/models edits,
  no commit/push/PR per instructions).
- No `blockedReasons`; collisions none. `isNonAuthoritative=false`, so task-checkbox blockers are real.

## 6. Strict TDD compliance (openspec/config.yaml strict_tdd:true — ACTIVE)

- `apply-progress.md` contains a `TDD Cycle Evidence` table for each of PR1/PR2/PR3/PR4 with RED→GREEN→
  TRIANGULATE→REFACTOR commands and results — PRESENT.
- Reported test files cross-referenced against disk: `tests/test_ingest.py` (9 tests), `tests/test_dataset.py`
  (13), `tests/test_train_eval.py` (9), `tests/test_cli.py` (11) — all present, 42 total, all GREEN on re-run.
- RED states were collection ERRORs (`No module named 'src.*'`) before each GREEN — acceptable RED
  (new tests fail without implementation); GREEN fixes and TRIANGULATE edge additions are documented.
- Assertion-quality audit (read all 4 test files): no tautologies (`assert True`), no ghost loops,
  no type-only-alone assertions, no smoke-only tests, no CSS/implementation-detail assertions.
  Boundary math is exact (`60000→60330=buy`, `r==thr→hold`, `r=+0.008` thr-split, purge cutoff arithmetic,
  embargo 2-vs-1 delta, probs sum ±1e-6, confusion-matrix shape+sum, vocab absence/presence, mtime+hash guard).
  Verdict: TDD COMPLIANT.

## 7. Data-card + run-artifact + docs audit (honest gaps noted)

- `data/interim/data_card.md` PRESENT. All 23 tasks-contract fields PRESENT (script-audited):
  raw_news_rows, raw_price_rows, kept_news_rows, dropped_missing_fields, duplicate_rate,
  dropped_no_price, dropped_no_forward, purged_train_rows, embargoed_rows, news range, price range,
  flat_threshold, embargo_days, seed, split boundaries, purge cutoffs, embargo gaps, per-split counts,
  per-split label distribution, per-split ranges, sources (Kaggle allowlist sha256 + Kraken XXBTZEUR sha256),
  versions (python/pandas/sklearn), rebuild command (`python -m src.dataset --flat-threshold 0.005 --embargo-days 1 --seed 42`).
  WARNING (non-blocking): design §5.4 extras `nulls per column` and `lang distribution` are absent from the
  card body (card has `lang` in parquet but not aggregated in md); Kaggle dataset name/version is generic
  ("Kaggle crypto-news (allowlist)") without explicit dataset slug/version. Fixture rebuild path is proven
  by `test_rebuild_reproducible`, not by a full raw-CSV rebuild here.
- Run-artifact layout: `src/train.py:save_run` implements `models/<run_id>/{vectorizer.pkl,model.pkl,config.json,metrics.json}`
  + `models/latest` pointer (certified via tmpdir in tests + `test_cli_latest_pointer_resolves`), and
  `src/dataset.py:build_dataset(run_id=...)` implements the `models/<run_id>/data_card.md` copy hook.
  GAP (covered by unchecked PR4-audit + Final-check tasks): **no frozen `models/<run_id>/` exists on disk**
  (`models/` holds only `.gitkeep`); CLI default `models/latest` therefore correctly exits 3 until a real
  training run is saved. No `models/`/`data/` writes by CLI (proven by `test_cli_never_retrains`).
- `DECISION_LOG.md` PRESENT with D1 Kaggle-vs-GDELT, D2 thr/embargo val-only, D3 TF-IDF/C, D4 single test report +
  per-class gate, D5 CLI inference-only. `README.md` portfolio section PRESENT. `pyproject.toml` `btc-signal`
  entry PRESENT. Temporal-discipline docstrings PRESENT in `src/dataset.py` + `src/train.py`; leakage code
  (`merge_asof`/`ret_24h`/`label_forward`) exists ONLY in `src/dataset.py` (grep-verified).

## 8. Review-workload / PR-boundary findings

- Tasks forecast ~900–1200 lines, 400-line risk High, chain stacked-to-main PR1→PR4 approved. Observed:
  PR1 398 new lines (within 400), PR2 472 lines (`src/dataset.py` 235 + `tests/test_dataset.py` 237) —
  EXCEEDS 400 max and 250 target, WARNING: needs explicit `size:exception` acceptance (cannot shrink without
  deleting required contract tests/fields). PR3 270 lines (over 250 target, within 400 max — acceptable),
  PR4 365 lines (within 400 max; over 200 target justified as docs/tests required by contract).
- No scope creep beyond assigned tasks found (all src/tests map to tasks/design traceability §9).
  `data/.gitignore` deviation (root `.gitignore` already covers `data/raw/`, `data/interim/*.parquet`)
  is acceptable and documented in PR1 apply-progress.

## 9. Exact blockers (26 — all tasks-checkbox, archive-blocking)

B1–B26: each of the 26 unchecked `- [ ]` lines in §4 above (one blocker per line).
Functional code for each exists and passes tests, but the checkbox itself is unchecked, so archive is blocked.
No additional functional blockers beyond these 26 (missing frozen run and PR2 size exception are subsumed
by B25 [PR4 audit] + B26 [Final check] and B14 [PR2 refactor] respectively).
Fix (parent-owned, verifier must not do): tick each box after independent review, save one real
`build_dataset` + train run to `models/<run_id>/` + `models/latest`, accept or split PR2 `size:exception`,
then re-run `sdd-verify` for a clean PASS and proceed to sync/archive. CLI demo with a real run
(`btc-signal --title "Bitcoin ETF inflows hit record" --price 60000` → exit 0 + disclaimer) should be
recorded at that time; currently proven only via temp-model tests + exit 2/3 live checks.

## 10. Commands executed (exact)

- `python -m pytest tests/ -v` → exit 0, 42 passed (envelope hash above).
- `python -c "import src.config, src.ingest_news, src.ingest_kraken, src.dataset, src.train, src.evaluate, src.cli; print('imports ok')"` → exit 0 (envelope hash above).
- `python -m src.cli --title "hello" --price abc --model /tmp/nope` → exit 2 (names price + usage).
- `python -m src.cli --title "hello" --price 60000 --model /tmp/no-such-model-xyz` → exit 3 (names path).
- `python -m src.cli --help` → shows price display-context note.
- `grep -rn "merge_asof|ret_24h|label_forward" src/` → only `src/dataset.py`.
- Validator: `gentle-ai sdd-verify-validate --input <report> --requirements 8 --scenarios 23` → `{"valid":true,"verdict":"fail",...}` before any openspec write (admission granted for fail with full requirements).

*Verifier did NOT modify `tasks.md`, did NOT commit/push/PR, did NOT retrain or write `data/`/`models/`.*

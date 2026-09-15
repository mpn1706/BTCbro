```yaml
schema: gentle-ai.verify-result/v1
evidence_revision: sha256:5de3b49f0480a876a8e8ebdaf6083c9f3fa85212c1b43643d68add4c58877b5f
verdict: pass
blockers: 0
critical_findings: 0
requirements: 8/8
scenarios: 23/23
test_command: python -m pytest tests/ -v
test_exit_code: 0
test_output_hash: sha256:832805234e5eeed222b752ec5be1c2dcaacc7dbf88c856f82f45a49953786efe
build_command: python -c "import src.config, src.ingest_news, src.ingest_kraken, src.dataset, src.train, src.evaluate, src.cli; print('imports ok')"
build_exit_code: 0
build_output_hash: sha256:47233ac20971137b78bf82759638a60672195955c3eb57326d48e2ead5c2b6b5
```

# Verify — btc-news-signal (full 4-PR slice, branch pr2-label-split)

**Status: PASS — archive unblocked (parent-led verify).**
Code and tests for all 8 requirements verified by reading implementation and re-running tests (42 passed).
Prior FAIL (evidence sha256:3deac8b8a15916cbe7deff3926d63f319053b94f87aa6f224770354d61496fe7) was procedural only: 26 stale-checkbox blockers because tasks.md was outside each apply run's allowedEditRoots. Parent commit 63eac8c ticked all 26 boxes after independent review; this re-verify confirms 26/26 complete with zero blockers.

Branch verified: `pr2-label-split` (stacked-to-main PR4/4 per apply-progress).
Workspace root: `C:/Users/usuario/Documents/ARCHIVOS DOMÉSTICOS/CURSO EOI/José Antonio/Proyecto Bitcoin`
Artifact store: `openspec`. Date (UTC): 2026-09-15.
Mode: parent-led verify — two sdd-verify delegations were tool-blocked (`SDD selection blocked` interceptor in that executor; generic probe reads fine, sdd-verify executor blocks all tools), gate stopped the chain per workflow, maintainer authorized reset (revision sha256:5de3b49f0480a876a8e8ebdaf6083c9f3fa85212c1b43643d68add4c58877b5f), user chose parent-led verify with frozen run deferred.

## 1. Spec coverage — REQ-1..REQ-8 (all PASS by code + test evidence)

Requirement totals for envelope: 8 requirements, 23 scenarios (counted from
`openspec/changes/btc-news-signal/specs/btc-news-signal/spec.md`: 8 `### Requirement:` headings, 23 `#### Scenario:` headings).

| REQ | Result | Implementation evidence (read) | Test evidence (re-run, all green) |
|-----|--------|--------------------------------|-----------------------------------|
| REQ-1 News ingest schema + dedup | PASS | `src/ingest_news.py`: `REQUIRED_FIELDS=(title,published_at,url)` quarantine+count; naive timestamps assumed Europe/Madrid then UTC; sha1(url_norm+title_norm) keep-earliest; `lang` per-record + distribution; emits `news_clean.parquet` | `test_news_timestamp_to_utc`, `test_dedup_case_whitespace`, `test_missing_field_quarantined`, `test_naive_madrid_timestamp_to_utc`, `test_empty_source_becomes_unknown` |
| REQ-2 Price ingest Kraken XXBTZEUR daily OHLC | PASS | `src/ingest_kraken.py`: pair gate aborts naming `XXBTZEUR`; daily UTC validation + numeric OHLCV coercion; incremental from max+1d, `MAX_PER_CALL=720`, `SLEEP_SECS=1`, de-overlap via timestamp anti-join | `test_wrong_pair_fails`, `test_incremental_pagination_polite`, `test_ohlcv_numeric_coercion`, `test_incremental_deoverlap_on_append` |
| REQ-3 Backward join + 24h forward label | PASS | `src/dataset.py:join_asof_backward` sorted `merge_asof(direction='backward')` UTC, `close_t24` via +1d lookup; `label_forward` `ret_24h=close(t+24h)/close(t)-1`, strict `>` so `r==thr→hold`, stamps `flat_threshold`, drops+counts; `src/config.py` default 0.005 choices {0.003,0.005,0.01} | `test_asof_uses_last_candle_at_or_before`, `test_future_candle_never_feature`, `test_label_buy_boundary` (60000→60330@0.005=buy), `test_label_hold_inside`, `test_threshold_configurable`, `test_exact_equality_holds`, `test_sell_mirror`, `test_news_before_first_candle_dropped` |
| REQ-4 Chronological split + purge + embargo | PASS | `src/dataset.py:temporal_split` + `DatasetConfig(train_end=2022-12-31,val_end=2023-12-31,overridable)`: purge `T>B-24h` + embargo `T>B-24h-embargo_days` at both boundaries with `purged_train_rows`/`embargoed_rows`; `embargo∈{1,2}` else ValueError; non-temporal `split_mode` explicit failure | `test_temporal_order`, `test_purge_plus_embargo_math`, `test_random_split_forbidden`, `test_embargo2_vs_1_delta` |
| REQ-5 Baseline TF-IDF + LogReg | PASS | `src/train.py`: `combine_text` title+body (price NOT a feature); `make_vectorizer` 5k–20k validated, ngram(1,2), sublinear True; `train_baseline` TF-IDF fit on passed texts only + `LogisticRegression(class_weight='balanced',random_state=SEED(42),max_iter=1000,solver='lbfgs')` CPU-only; `make_run_id` + `save_run` writes `models/<run_id>/{vectorizer.pkl,model.pkl,config.json,metrics.json}` + `models/latest` pointer | `test_balanced_weights_set`, `test_tfidf_params_in_contract`, `test_seed_reproducible`, `test_train_fits_train_only`, `test_combine_text_title_body` |
| REQ-6 Honest per-class eval | PASS | `src/evaluate.py`: `evaluate_predictions` per-class P/R/F1/support (zero_division=0), macro-F1, accuracy alongside only, 3x3 matrix rows=true cols=pred order buy/hold/sell; `check_report_complete` rejects accuracy-alone and non-3x3; near-random+honest=PASS in docstring | `test_eval_requires_per_class`, `test_accuracy_only_rejected`, `test_hold_dominance_eval`, `test_single_class_val_fold` |
| REQ-7 Inference CLI + disclaimer | PASS | `src/cli.py`: `main(argv)->int`, parser `--title --price [--body --model --json]`, `--help` notes price display-context-only; `resolve_run_dir` latest-pointer→run dir; `load_artifacts` read-only, missing/corrupt→exit 3 naming path; `predict_probs` via `combine_text`; stdout action+probs+price_ctx+DISCLAIMER, `--json` variant; exits 0/2/3/1; `pyproject.toml` `btc-signal` entry | `test_cli_happy_path`, `test_cli_missing_title_exit2`, `test_cli_invalid_price_exit2`, `test_cli_missing_artifact_exit3`, `test_cli_never_retrains`, `test_cli_bad_prices_exit2`, `test_cli_corrupt_model_exit3`, `test_cli_probs_sum_to_one`, `test_cli_json_shape`, `test_cli_help_notes_price_context`, `test_cli_latest_pointer_resolves` — plus live re-check this run: `--price abc`→exit 2 naming price+usage; missing model→exit 3 naming path; `--help` shows display-context note |
| REQ-8 Reproducibility via data_card | PASS (tasks contract; 2 design extras missing — see §7 WARNING) | `src/dataset.py:build_dataset` + card writer to `data/interim/data_card.md` + `models/<run_id>/data_card.md` copy hook + `python -m src.dataset --flat-threshold X --embargo-days Y --seed N` CLI; card present with all tasks-contract fields | `test_rebuild_reproducible` (fixture rebuild twice → identical frames + card contains raw_news_rows, flat_threshold, embargo_days, seed, boundaries, rebuild command, dropped_no_price, dropped_no_forward, purged_train_rows, embargoed_rows) |

Scenario count check: 3+3+4+3+2+3+3+2 = 23. All 23 scenarios map to at least one passing test.

## 2. Valid-chooses / test-certifies rule (PASS)

- `flat_threshold ∈ {0.003, 0.005, 0.01}` default 0.005; invalid raises ValueError naming allowed values (certified by `test_threshold_configurable`).
- `embargo_days ∈ {1, 2}` default 1; invalid raises ValueError (certified by `test_purge_plus_embargo_math`).
- `split_mode == "temporal"` only; random/stratified-shuffle raises ValueError naming chronological/temporal (certified by `test_random_split_forbidden`).
- `max_features ∈ [5000, 20000]`; out-of-range raises ValueError (certified by `test_tfidf_params_in_contract`).
- Val-chooses / test-locked: `train_baseline` fits on caller-passed train texts only (certified by `test_train_fits_train_only`); no test-2024 read in train/evaluate; `DECISION_LOG.md` D2/D4 records val-only selection and single final test report. Leakage tokens (`merge_asof`/`ret_24h`/`label_forward`) exist only in `src/dataset.py` (grep-verified this run; `__pycache__` binary match ignored).

## 3. Test counts (re-run, not trusted)

- Command: `python -m pytest tests/ -v` → exit 0, **42 passed** (9 ingest + 13 dataset + 9 train/eval + 11 CLI), 0 failed/skipped.
- Observed output hash `sha256:832805234e5eeed222b752ec5be1c2dcaacc7dbf88c856f82f45a49953786efe` covers the full 42-line run ending `42 passed` (timing varies per run).
- Build/import check: `python -c "import src.config, src.ingest_news, src.ingest_kraken, src.dataset, src.train, src.evaluate, src.cli; print('imports ok')"` → exit 0, `imports ok`, hash `sha256:47233ac20971137b78bf82759638a60672195955c3eb57326d48e2ead5c2b6b5` (stable).
- Fixtures `tests/fixtures/{news_tiny.csv (7 rows), prices_tiny.csv (6 rows)}` present as required.

## 4. Task completion status (26/26 complete — zero blockers)

`openspec/changes/btc-news-signal/tasks.md` shows 26/26 implementation tasks checked (`grep -c "^- \[x\]"` = 26, `grep -c "^- \[ \]"` = 0).
Prior 26 stale-checkbox blockers are resolved by parent commit 63eac8c (ticked after independent review). Verifier did NOT modify `tasks.md` in this run. No `- [ ]` remains, so no archive blocker from tasks.

## 5. Structured status and actionContext

- Native status consumed before this parent-led verify: change `btc-news-signal`, store `openspec`, tasks 26/26, apply `all_done`, verify `blocked` on old evidence, next `remediate`; after two tool-blocked sdd-verify attempts and gate stop, maintainer authorized reset (new revision sha256:5de3b49f0480a876a8e8ebdaf6083c9f3fa85212c1b43643d68add4c58877b5f, used as this report's evidence_revision).
- `actionContext.mode=repo-local`, workspaceRoot as above, allowedEditRoots respected (this run writes only `openspec/changes/btc-news-signal/verify.md` + `verify-report.md` mirror; no src/tests/data/models edits, no commit/push/PR).
- Delivery `ask-on-risk` with PR2 `size:exception` maintained per user decision 2026-09-15 (see §8).

## 6. Strict TDD compliance (openspec/config.yaml strict_tdd:true — ACTIVE)

- `apply-progress.md` contains a `TDD Cycle Evidence` table for each of PR1/PR2/PR3/PR4 with RED→GREEN→TRIANGULATE→REFACTOR commands and results — PRESENT.
- Test files on disk: `tests/test_ingest.py` (9), `tests/test_dataset.py` (13), `tests/test_train_eval.py` (9), `tests/test_cli.py` (11) — 42 total, all GREEN on re-run.
- RED states were collection ERRORs (`No module named 'src.*'`) before each GREEN — acceptable RED; GREEN fixes and TRIANGULATE edge additions documented.
- Assertion-quality audit: boundary math exact (`60000→60330=buy`, `r==thr→hold`, `r=+0.008` thr-split, purge cutoff arithmetic, embargo 2-vs-1 delta, probs sum ±1e-6, matrix shape+sum, vocab absence/presence, mtime+hash guard). Verdict: TDD COMPLIANT.

## 7. Data-card + run-artifact + docs audit (honest gaps noted)

- `data/interim/data_card.md` PRESENT with all tasks-contract fields (counts, ranges, thr/embargo/seed, boundaries, purge cutoffs, embargo gaps, per-split counts/label distribution/ranges, sources with sha256, versions, rebuild command).
- WARNING (non-blocking): design §5.4 extras `nulls per column` and `lang distribution` absent from card body; Kaggle source named generically ("Kaggle crypto-news (allowlist)") without explicit dataset slug/version. Fixture rebuild proven by `test_rebuild_reproducible`.
- Run-artifact layout: `save_run` implements `models/<run_id>/{vectorizer.pkl,model.pkl,config.json,metrics.json}` + `models/latest` pointer (certified via tmpdir in tests + pointer test), and `build_dataset(run_id=...)` implements the `models/<run_id>/data_card.md` copy hook.
- DEFERRED (explicit user choice 2026-09-15, non-blocking for verify): **no frozen `models/<run_id>/` on disk** (`models/` holds only `.gitkeep`); CLI default `models/latest` therefore correctly exits 3 until a real training run is saved. Temp-model tests prove the exit-0 path; no `models/`/`data/` writes by CLI (proven by `test_cli_never_retrains`). Frozen run to be decided after this re-verify.
- `DECISION_LOG.md` PRESENT (D1–D6 incl. PR2 size:exception), `README.md` portfolio section PRESENT, `pyproject.toml` `btc-signal` entry PRESENT, temporal-discipline docstrings PRESENT.
- SCOPE NOTE: `data/raw/` holds tiny fixture-like CSVs (7 news / 6 prices, Jan 2024), not full 2018–2024 history. This verify certifies the educational slice wiring on fixtures; a real-data ingest+train run remains a follow-up, not a verify blocker.

## 8. Review-workload / PR-boundary findings

- Forecast ~900–1200 lines, 400-line risk High, chain stacked-to-main PR1→PR4 approved. Observed: PR1 398 (within 400), PR2 472 (EXCEEDS 400 max — accepted as `size:exception` per DECISION_LOG D6 and user confirmation 2026-09-15; cannot shrink without deleting required contract tests/fields), PR3 270 (within 400 max), PR4 365 (within 400 max).
- No scope creep beyond tasks/design traceability. PR1 `data/.gitignore` deviation (root `.gitignore` already covers outputs) remains acceptable.

## 9. Exact blockers (0)

No blockers. All 26 prior stale-checkbox blockers resolved (§4). PR2 size is an accepted exception, not a blocker. Missing frozen run and tiny-raw scope are explicit deferred follow-ups (§7), not verify blockers.

## 10. Commands executed (exact)

- `python -m pytest tests/ -v` → exit 0, 42 passed (envelope hash above).
- `python -c "import src.config, src.ingest_news, src.ingest_kraken, src.dataset, src.train, src.evaluate, src.cli; print('imports ok')"` → exit 0 (envelope hash above).
- `python -m src.cli --title "hello" --price abc --model /tmp/nope` → exit 2 (names price + usage).
- `python -m src.cli --title "hello" --price 60000 --model /tmp/no-such-model-xyz` → exit 3 (names path).
- `python -m src.cli --help` → shows price display-context note.
- `grep -rn "merge_asof|ret_24h|label_forward" src/` → only `src/dataset.py` (+ `__pycache__` binary, ignored).
- `grep -c "^- \[x\]" tasks.md` → 26; `grep -c "^- \[ \]"` → 0.
- `grep -c "### Requirement:" spec.md` → 8; `grep -c "#### Scenario:"` → 23.
- Validator: `gentle-ai sdd-verify-validate --input <report> --requirements 8 --scenarios 23` → `{"valid":true,"verdict":"pass",...}` before openspec write (admission granted for pass with full requirements).

*Parent-led verify wrote only `tasks.md`-untouched `verify.md` + `verify-report.md` mirror; did NOT commit/push/PR, did NOT retrain or write `data/`/`models/`.*

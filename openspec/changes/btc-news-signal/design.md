# Design — btc-news-signal

## 1. Overview

**Change:** `btc-news-signal`
**Goal:** Educational first slice proving a small local model can map BTC news headline + BTC/EUR price context to `buy` / `sell` / `hold` with honest temporal evaluation.
**Non-goals:** No live trading, no financial advice, no LLM fine-tune, no GDELT backfiles, no intraday horizon (see proposal).

**Architecture thesis:** A linear, reproducible batch pipeline with leakage prevention as the central invariant:

```
Kaggle news + Kraken XXBTZEUR
  -> normalize (UTC) + dedup
  -> join-asof backward (features past-only)
  -> label 24h forward (train-time truth only)
  -> temporal split + purge + embargo
  -> TF-IDF + LogReg (seed fixed, CPU)
  -> per-class metrics
  -> CLI inference-only + data_card.md + DECISION_LOG.md
```

Leakage prevention is enforced at three independent layers: join direction (R2), split geometry (R4), and tuning discipline (R3, val-only). Tests enforce each layer.

## 2. Data Flow (normative)

### Stage 0 — Raw inputs (gitignored `data/`, never committed)

| Input | Source | File | Grain |
|---|---|---|---|
| News history | Kaggle crypto-news 2018+, 30–50k rows | `data/raw/news.csv` | 1 row / article |
| Price history | Kraken `XXBTZEUR` daily OHLC | `data/raw/kraken_xxbtzeur_1d.csv` | 1 row / UTC day |

`data/` is gitignored. Only small fixtures under `tests/fixtures/` are committed.

### Stage 1 — Normalize + dedup (`src/ingest_news.py`)

1. Parse each raw news record; require `title`, `published_at`, `url`. Missing any → reject/quarantine + count, continue (REQ-1).
2. Normalize: `published_at` → UTC timezone-aware datetime; `title`/`url` → strip + casefold for hashing, preserve originals for features; detect/report language (per-record `lang` field + aggregate distribution in card).
3. Dedup: `sha1(normalized_url + '\x00' + normalized_title)`. Keep first occurrence (earliest `published_at`), count remainder as `duplicate_rate`.
4. Emit `data/interim/news_clean.parquet` (schema §5.1).

### Stage 2 — Price ingest (`src/ingest_kraken.py`)

1. **Bulk-first:** load `kraken_xxbtzeur_1d.csv` → validate daily UTC-aware `timestamp`, numeric OHLCV.
2. **Incremental:** from `max(stored timestamp)+1d`, call Kraken OHLC API (`pair=XXBTZEUR`, `interval=1440`), ≤720 candles/call, ≥1s sleep between calls, append + de-overlap.
3. **Pair gate:** any `pair != "XXBTZEUR"` → abort with explicit error naming expected pair, no partial output presented as valid (REQ-2 wrong-pair scenario).
4. Emit `data/interim/prices_clean.parquet` (schema §5.2).

### Stage 3 — Join + label + split (`src/dataset.py`, core intelligence)

This single module owns all three leakage-sensitive operations so the invariant can be reviewed and tested in one place. Public functions:

- `join_asof_backward(news, prices) -> joined`
- `label_forward(joined, flat_threshold) -> labeled`
- `temporal_split(labeled, embargo_days) -> (train, val, test)`
- `build_dataset(config) -> (train, val, test) + data_card.md`

**3a. Backward join-asof (R2):**
- Sort both frames by UTC time. For each news `published_at=T`, feature price = last price candle with `timestamp <= T` (`merge_asof(direction='backward')`).
- Future candle `C1 > T` is never a feature. `close(t+24h)` is resolved only for the label (train-time truth), never concatenated into `X`.
- News earlier than first candle, or with no `close(t+24h)` available (last 24h of price history) → dropped + counted in card (`dropped_no_price`, `dropped_no_forward`).

**3b. 24h forward label (R1, R3):**
- `ret_24h = close(t+24h) / close(t) - 1` using daily closes.
- `buy` if `r > +thr`, `sell` if `r < -thr`, else `hold`.
- `flat_threshold` is a `build_dataset` / CLI-config param, default `0.005`, allowed `{0.003, 0.005, 0.01}`. Any other value → explicit `ValueError`. Tuning on **val only**, never test (evaluator asserts this by convention + DECISION_LOG entry).
- Boundary contract (default thr): `60000 → 60330` (`r=0.0055`) = `buy`; `60000 → 60100` (`r≈0.00167`) = `hold`. Exact equality `r == thr` → `hold` (strict `>`).

**3c. Temporal split + purge + embargo (R4):**
- Default boundaries (config-overridable, recorded in card): train `2018-01-01..2022-12-31`, val `2023-01-01..2023-12-31`, test `2024-01-01..end`.
- Invariant: `max(train.published_at) < min(val.published_at) <= max(val) < min(test)`, modulo enforced gap.
- **Purge:** drop any train item with `published_at > B - 24h` for boundary `B` (its 24h label window overlaps the next split).
- **Embargo:** additionally drop `published_at > B - 24h - embargo_days`. `embargo_days ∈ {1, 2}`, default `1`.
- Same purge+embargo applied at val/test boundary.
- `split_mode` param accepts only `"temporal"`. Any `"random"` / `"shuffle"` / `"stratified-shuffle"` value → explicit failure stating only chronological+purge+embargo is allowed. No shuffle flag is offered in CLI.

**3d. `data_card.md`:** written to `data/interim/data_card.md` on every build (§5.4).

### Stage 4 — Train baseline (`src/train.py`)

- Input: train (+ val for threshold selection) splits from `build_dataset`.
- Pipeline: `TfidfVectorizer(title + " " + body, max_features 5k–20k, ngram (1,2), sublinear_tf=True)` → `LogisticRegression(class_weight='balanced', random_state=SEED, max_iter=1000, solver='lbfgs')` on CPU.
- `SEED` fixed in one `src/config.py` (e.g. `SEED=42`), recorded in card; two runs same seed+data → identical label mapping + artifacts within deterministic tolerance.
- No LLM, no GPU, artifact `<100MB`: `models/tfidf_logreg.joblib` + `models/label_map.json`.
- Threshold tuning loop (if run): evaluate `thr ∈ {0.003,0.005,0.01}` on **val only**, pick best macro-F1, freeze, then single final eval on test.

### Stage 5 — Evaluate (`src/evaluate.py`)

- Mandatory outputs: per-class precision, per-class recall, macro-F1, full 3×3 confusion matrix (rows=true, cols=pred, order `buy/hold/sell`).
- Forbidden: reporting global accuracy alone → treated as incomplete, fails slice gate. Accuracy may appear only alongside per-class block.
- Success rule: near-random + honest protocol = PASS; high score via random split / forward join / test-peeked tuning = FAIL regardless of number.
- Optional informational: simulated return vs buy-hold (clearly labeled non-advisory, not a gate).

### Stage 6 — CLI inference-only (`src/cli.py` → `btc-signal`)

- Loads frozen artifacts read-only. Never imports train paths, never writes `data/` or `models/`.
- Feature path at inference: vectorize `--title` (+ optional `--body`), combine with `--price` context display only (price is context, not a trained feature in slice 1 — documented in `--help` and card to avoid implying price-predicts-price).
- Always prints disclaimer.

## 3. File Changes (within 400-line review budget)

| File | Change | REQ |
|---|---|---|
| `src/config.py` (new, ~25 lines) | `SEED`, `FLAT_THRESHOLD_DEFAULT=0.005`, `FLAT_THRESHOLD_CHOICES`, `EMBARGO_DEFAULT=1`, `EMBARGO_CHOICES`, `PAIR="XXBTZEUR"`, split boundaries | R3, R4 |
| `src/ingest_news.py` (new, ~70 lines) | Raw CSV → UTC normalize + url+title hash dedup + quarantine counts → `news_clean.parquet` | REQ-1 |
| `src/ingest_kraken.py` (new, ~60 lines) | Bulk CSV + polite incremental (720/call, 1s gap) + pair gate → `prices_clean.parquet` | REQ-2 |
| `src/dataset.py` (new, ~110 lines) | `join_asof_backward` + `label_forward` + `temporal_split` + `build_dataset` + `data_card.md` writer | REQ-3, REQ-4, REQ-8 |
| `src/train.py` (new, ~50 lines) | TF-IDF + balanced LogReg, seed-fixed, CPU | REQ-5 |
| `src/evaluate.py` (new, ~45 lines) | Per-class P/R, macro-F1, confusion matrix, accuracy-alone rejection | REQ-6 |
| `src/cli.py` (new, ~40 lines) | `btc-signal --title --price`, inference-only, disclaimer, exit codes §5.3 | REQ-7 |
| `tests/test_dataset.py` (new, ~80 lines) | asof-backward, buy/hold boundaries, thr=0.01 vs 0.005, purge+embargo math, random-split forbidden, determinism | REQ-3, REQ-4 |
| `tests/test_ingest.py` (new, ~50 lines) | UTC normalization, case/space dedup, missing-field quarantine, wrong-pair fail, pagination contract (mocked) | REQ-1, REQ-2 |
| `tests/fixtures/*` | Tiny hand-made news/price CSVs (no real data committed) | support |
| `data/` | gitignored payloads; `data/interim/data_card.md` generated | REQ-8 |
| `DECISION_LOG.md` (append) | thr/embargo choices, val-only tuning record, Kaggle-vs-GDELT rationale | audit |
| `pyproject.toml` | `btc-signal` entry point, deps (`pandas`, `pyarrow`, `scikit-learn`, `joblib`, `requests`) | packaging |

Budget note: ~530 lines of new code enumerated above exceeds the 400-line review budget — per `ask-on-risk` strategy the implementation MUST be delivered as a chain (e.g. PR1: ingest+tests, PR2: dataset+card+tests, PR3: train/eval/CLI) or an explicit `size:exception` accepted by the human. No single PR over 400 lines.

## 4. Module Contracts

### 4.1 `build_dataset(config)` signature

```python
@dataclass(frozen=True)
class DatasetConfig:
    flat_threshold: float = 0.005      # ∈ {0.003, 0.005, 0.01}
    embargo_days: int = 1              # ∈ {1, 2}
    split_mode: str = "temporal"       # only value accepted
    seed: int = 42
    train_end: str = "2022-12-31"
    val_end: str = "2023-12-31"
```

- Invalid `flat_threshold` / `embargo_days` / `split_mode != "temporal"` → `ValueError` with message naming allowed values (test-asserted).

### 4.2 CLI contract (`btc-signal`)

**Invocation:**
```
btc-signal --title <headline> --price <current-btc-eur> [--model models/tfidf_logreg.joblib]
```

**Stdout (happy path, exit 0):**
```
action: hold
probs: buy=0.21 sell=0.18 hold=0.61
price_ctx: 60000.00 EUR
DISCLAIMER: Educational demo only — not investment advice. No live trading.
```

- `action ∈ {buy, sell, hold}` (argmax of probs).
- `probs` sum to 1.0 ± 1e-6, three entries always present in `buy/sell/hold` order.
- Machine-readable variant: `--json` emits `{"action":..,"probs":{..},"price_ctx":..,"disclaimer":..}` (optional, test only if implemented).

**Exit codes:**

| Code | Condition |
|---|---|
| 0 | inference success |
| 2 | missing `--title` / `--price`, non-numeric or non-positive `--price`, empty title → stderr names the argument + prints usage |
| 3 | model artifact missing/corrupt → stderr names expected path |
| 1 | any unexpected internal error |

**Inference-only guarantee:** CLI opens model files read-only; a test asserts mtimes/hashes of `data/` + `models/` unchanged after CLI runs.

## 5. Data Contracts

### 5.1 `news_clean.parquet`

| col | type | constraint |
|---|---|---|
| `id` | string | unique, non-null |
| `published_at` | tz-aware datetime (UTC) | non-null |
| `source` | string | nullable, empty→`"unknown"` |
| `title` | string | non-null, stripped, non-empty |
| `body` | string | nullable, default `""` |
| `url` | string | non-null |
| `lang` | string | non-null (e.g. `en`), `"unknown"` allowed but counted |
| `dedup_hash` | string | `sha1(url_norm + \x00 + title_norm)`, unique |

### 5.2 `prices_clean.parquet`

| col | type | constraint |
|---|---|---|
| `timestamp` | tz-aware datetime (UTC, 00:00) | unique, sorted, daily, non-null |
| `open/high/low/close` | float64 | `> 0`, non-null |
| `volume` | float64 | `>= 0`, non-null |
| `pair` | string | always `"XXBTZEUR"` |

### 5.3 `labeled.parquet` (interim, pre-split)

`news_clean` cols + `close_t: float`, `close_t24: float`, `ret_24h: float`, `label ∈ {buy,sell,hold}`, `flat_threshold: float` (build value stamped per row for audit).

### 5.4 `data_card.md` fields (mandatory, REQ-8)

```
# Data Card — build <iso-timestamp>
- raw_news_rows, raw_price_rows (date range raw)
- kept_news_rows, dropped_missing_fields, duplicate_rate
- nulls per column, lang distribution
- price range (min/max date, missing days)
- flat_threshold, embargo_days, seed
- split boundaries + purge cutoffs + embargo gaps + per-split counts + per-split label distribution
- dropped_no_price, dropped_no_forward, purged_train_rows, embargoed_rows
- source allowlist + Kaggle dataset name/version + Kraken pair
- rebuild command: python -m src.dataset --flat-threshold X --embargo-days Y
```

Rebuild with same raw CSVs + same params → identical counts/labels/splits (test-asserted on fixtures).

## 6. TDD Plan (`python -m pytest`)

Order: RED → GREEN → TRIANGULATE → REFACTOR per `openspec/config.yaml` strict TDD.

**`tests/test_ingest.py`:**
1. `test_news_timestamp_to_utc` — naive/`Europe/Madrid` input → UTC-aware output.
2. `test_dedup_case_whitespace` — same url, title differing by case/whitespace → 1 kept, dup rate 0.5.
3. `test_missing_field_quarantined` — missing `url`/`title`/`published_at` → quarantined + counted, valid rows pass.
4. `test_wrong_pair_fails` — `pair="XXBTZUSD"` → error naming `XXBTZEUR`, no output marked valid.
5. `test_incremental_pagination_polite` (mocked HTTP) — asserts `count<=720` per call and sleep ≥1s between calls.

**`tests/test_dataset.py`:**
6. `test_asof_uses_last_candle_at_or_before` — news between C0/C1 → feature price C0.
7. `test_future_candle_never_feature` — C1 changes label inputs only, feature vector unchanged.
8. `test_label_buy_boundary` — `60000→60330 @0.005` → `buy`.
9. `test_label_hold_inside` — `60000→60100 @0.005` → `hold`.
10. `test_threshold_configurable` — `r=+0.008` → `hold @0.01`, `buy @0.005`.
11. `test_temporal_order` — train < val < test timestamps.
12. `test_purge_plus_embargo_math` — boundary B, `embargo=1`: train rows with `T > B-24h-1d` absent.
13. `test_random_split_forbidden` — `split_mode="random"` → explicit failure.
14. `test_rebuild_reproducible` — same fixtures + params → identical labels/splits + card.

**`tests/test_train_eval_cli.py` (in PR3):**
15. `test_balanced_weights_set`, `test_seed_reproducible`, `test_eval_requires_per_class` (accuracy-only → fail gate), `test_cli_happy_path` (action+probs+disclaimer, exit 0), `test_cli_missing_input_exit2`, `test_cli_never_retrains` (hashes unchanged).

## 7. Decisions, Tradeoffs, Risks

### Decisions (from proposal/explore, locked here)

| # | Decision | Why | Alternatives rejected |
|---|---|---|---|
| D1 | Kaggle news for slice 1, GDELT phase 2 | 30–50k rows fast/small vs GDELT backfiles heavy; slice-1 budget | GDELT full history (phase 2) |
| D2 | Kraken `XXBTZEUR` daily, bulk-first | Free/no-key, EUR-native, polite incremental | CoinGecko (rate limits), intraday (scope) |
| D3 | 24h horizon daily | Matches daily candles, teacher ask, simple label | Intraday (noisy, needs finer candles) |
| D4 | `flat_threshold` default 0.005 ∈ {0.003,0.005,0.01}, val-only | 0.5% sensible for daily BTC/EUR; tunable without test peek | Hardcoded 1.5% (explore draft) — rejected, violates R3 |
| D5 | Embargo default 1 ∈ {1,2} | 1d preserves data, 2d safer; experiment both on val | No embargo (leaks via 24h overlap) |
| D6 | TF-IDF+LogReg balanced, seed-fixed, CPU | Explainable, <100MB, reproducible, portfolio-legible | LLM fine-tune (out of slice) |
| D7 | Per-class metrics gate, near-random passes | Honest eval is the learning outcome | Accuracy-alone (forbidden) |
| D8 | CLI inference-only | Prevents accidental retrain/contamination | CLI with `--retrain` (rejected) |

### Tradeoffs

- **thr 0.3% vs 1.0%:** low thr → more buy/sell signals but noisy/imbalanced-light; high thr → cleaner but hold-dominated, minority classes starve. Resolve empirically on val macro-F1, log in DECISION_LOG.
- **embargo 1d vs 2d:** 1d keeps ~1 extra day of train per boundary; 2d safer against multi-day news-effect autocorrelation. Same val-based resolution.
- **title+body vs title-only:** title+body richer but noisier + larger vector; start title+body, fall back to title-only if val shows degradation (recorded, not test-peeked).
- **English-only vs +ES:** assume mostly English; keep `lang` column so a later slice can stratify without re-ingest.

### Risks → mitigations (traceability)

| Risk | Mitigation | Test/gate |
|---|---|---|
| Kaggle license/quality variance | Source allowlist + card gate | card lists dataset name/version; quarantine counts |
| TZ misalignment | UTC-only, tz-aware dtypes, fail on naive | UTC test |
| 24h overlap leakage | purge + embargo in `dataset.py` | purge-math test |
| Direction leakage (future candle as feature) | backward-asof only | asof + future-candle tests |
| Random-split inflation (1–6×) | `split_mode` gate | forbidden-split test |
| Test-peeked tuning | val-only rule + DECISION_LOG | review gate, card stamps thr per row |
| Hold dominance | balanced weights + per-class gate | weights + eval tests |
| Budget overrun | chain PRs, ask-on-risk | §3 budget note |

## 8. Rollout

1. **PR1 (ingest):** `config.py`, `ingest_news.py`, `ingest_kraken.py`, `test_ingest.py`, `data/.gitignore`. Local run: `python -m pytest tests/test_ingest.py`.
2. **PR2 (dataset):** `dataset.py`, `test_dataset.py`, `data_card.md` template. Local run: build on fixtures, inspect card.
3. **PR3 (model+eval+CLI):** `train.py`, `evaluate.py`, `cli.py`, entry point, DECISION_LOG entries. Local run: `python -m pytest` full + `btc-signal --title "…" --price 60000`.
- Each PR <400 lines or flagged. CPU-only, no secrets, no network in tests (mock Kraken). Real Kraken calls only in manual incremental update.
- Rollback: delete `openspec/changes/btc-news-signal/` outputs + `data/interim/*`; no migration, re-run from bulk CSVs.

## 9. Traceability (spec → design → test)

| Spec REQ | Design § | Tests |
|---|---|---|
| News schema+dedup | §2-stage1, §5.1 | ingest 1–3 |
| Kraken XXBTZEUR | §2-stage2, §5.2 | ingest 4–5 |
| Backward join + 24h label | §2-stage3a/b, §4.1, §5.3 | dataset 6–10 |
| Temporal split purge+embargo | §2-stage3c | dataset 11–13 |
| TF-IDF+LogReg | §2-stage4 | train 15 |
| Per-class eval | §2-stage5 | eval 15 |
| CLI + disclaimer | §4.2 | cli 15 |
| data_card.md | §5.4 | dataset 14 |

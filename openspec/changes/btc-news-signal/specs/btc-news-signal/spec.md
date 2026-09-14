# BTC News Signal Specification

## Purpose

Provide an educational, reproducible first slice that maps a BTC news headline plus BTC/EUR price context to `buy` / `sell` / `hold`, using a small local CPU model with honest temporal evaluation. Learning outcome and decision auditability are the goals; trading performance is explicitly not a goal. Output is never financial advice and never triggers live trading.

## Requirements

### Requirement: News Ingest Schema and Deduplication

The system MUST ingest news history into a fixed schema with fields `id`, `published_at` (UTC, timezone-aware), `source`, `title`, `body`, `url`, and MUST deduplicate records by hash of normalized `url` + normalized `title`, and MUST report language per record or in aggregate.

#### Scenario: Valid news record normalizes to UTC

- GIVEN a raw Kaggle crypto-news record with a local-time or naive timestamp
- WHEN news ingest runs
- THEN the stored record has all six schema fields with `published_at` converted to UTC and a reported language value

#### Scenario: Duplicate url plus title is removed once

- GIVEN two raw records with the same URL and title differing only by case or surrounding whitespace
- WHEN news ingest runs
- THEN only one record is retained and the duplicate is counted in the duplicate rate

#### Scenario: Missing schema field is rejected

- GIVEN a raw record missing `url` or `title` or `published_at`
- WHEN news ingest runs
- THEN the record is rejected or quarantined and counted, and ingest continues for valid records

### Requirement: Price Ingest for Kraken XXBTZEUR Daily OHLC

The system MUST ingest daily OHLC price history in UTC for the Kraken pair `XXBTZEUR` only, with columns `timestamp`, `open`, `high`, `low`, `close`, `volume`, loading bulk CSV first and then incremental API updates using pagination of 720 candles per call with a 1-second delay between calls. The system MUST fail explicitly with a usable error when a non-`XXBTZEUR` pair is requested.

#### Scenario: Bulk CSV loads daily UTC candles

- GIVEN a bulk `kraken_xxbtzeur_1d.csv` file with daily UTC OHLC rows
- WHEN price ingest runs
- THEN all rows are stored with UTC timezone-aware daily timestamps and numeric OHLCV fields

#### Scenario: Incremental update paginates politely

- GIVEN stored history ending at day D and new days available via the Kraken API
- WHEN incremental price ingest runs
- THEN requests use at most 720 candles per call with at least 1 second between calls until history is current

#### Scenario: Wrong pair fails explicitly

- GIVEN a request for any pair other than `XXBTZEUR` (e.g. `XXBTZUSD`, `BTC/USD`)
- WHEN price ingest is invoked
- THEN ingest aborts with an explicit error naming the expected pair and no partial dataset is presented as valid

### Requirement: Backward Join and 24h Forward Label

The system MUST align each news item to price using a backward join-asof (the last candle at-or-before `published_at` in UTC, never a future candle) and MUST compute `ret_24h = close(t+24h) / close(t) - 1` and MUST label `buy` if `ret_24h > +flat_threshold`, `sell` if `ret_24h < -flat_threshold`, else `hold`. `flat_threshold` MUST be a tunable parameter with default `0.005` and supported values `0.003` / `0.005` / `0.01`, tuned on validation data only.

#### Scenario: Buy boundary example at default threshold

- GIVEN `flat_threshold = 0.005`, a news item joined to `close(t) = 60000` and `close(t+24h) = 60330`
- WHEN labeling runs (`ret_24h = 0.0055`)
- THEN the label is `buy`

#### Scenario: Hold inside threshold

- GIVEN `flat_threshold = 0.005`, a news item joined to `close(t) = 60000` and `close(t+24h) = 60100`
- WHEN labeling runs (`ret_24h ≈ 0.00167`)
- THEN the label is `hold`

#### Scenario: Future candle is never used as feature context

- GIVEN a news item published at time T between daily candles C0 (at-or-before T) and C1 (after T)
- WHEN the join runs
- THEN the feature price is from C0 and C1 is used only for the train-time forward label, never as an input feature

#### Scenario: Threshold is configurable

- GIVEN a dataset build configured with `flat_threshold = 0.01`
- WHEN labeling runs on a return of `+0.008`
- THEN the label is `hold`, whereas the same return under `0.005` would be `buy`

### Requirement: Chronological Split with Purge and Embargo

The system MUST split data chronologically (train before validation before test), MUST purge training items whose 24h label window overlaps the validation/test boundary, and MUST apply a configurable `embargo_days` of `1` or `2` after each boundary. Random or shuffled splits are forbidden and MUST NOT be offered as a valid evaluation mode.

#### Scenario: Temporal order is preserved

- GIVEN labeled news from 2018 through 2024+
- WHEN the split runs
- THEN every training timestamp precedes every validation timestamp, which precedes every test timestamp, modulo the purged and embargoed gap

#### Scenario: Overlapping 24h windows are purged plus embargo

- GIVEN `embargo_days = 1` and a train/test boundary at time B
- WHEN the split runs
- THEN any training item with `published_at` later than `B - 24h - embargo_days` is excluded from training

#### Scenario: Random split is rejected

- GIVEN a request to evaluate with a random or stratified-shuffle split
- WHEN dataset build or evaluation is invoked with that option
- THEN the system fails explicitly stating that only chronological split with purge and embargo is allowed

### Requirement: Baseline Model TF-IDF plus Logistic Regression

The system MUST train the slice-1 baseline as TF-IDF features over news text plus LogisticRegression with balanced class weights, using a fixed configured seed, running on local CPU only. The system MUST NOT require LLM fine-tuning for this slice.

#### Scenario: Reproducible CPU training

- GIVEN a fixed seed and the same training split run twice on CPU
- WHEN training runs
- THEN both runs produce identical label mappings and identical model artifacts within deterministic tolerances

#### Scenario: Hold dominance does not silently win

- GIVEN a training set where `hold` is the majority class
- WHEN training runs
- THEN the classifier uses balanced class weights so minority `buy`/`sell` classes are not ignored by default

### Requirement: Honest Per-Class Evaluation

The system MUST report per-class precision and recall plus the confusion matrix (and macro-F1), and MUST NEVER present global accuracy as the sole success metric. A result barely above random MUST count as success when the temporal protocol (backward join, purge plus embargo, no test peeking) was followed honestly.

#### Scenario: Per-class report is mandatory

- GIVEN a trained baseline and a held-out test split
- WHEN evaluation runs
- THEN output includes per-class precision, per-class recall, and the full confusion matrix, not only a single accuracy number

#### Scenario: Accuracy-alone report is rejected

- GIVEN an evaluation output containing only global accuracy
- WHEN the evaluation gate is checked
- THEN the output is treated as incomplete and fails the slice success criteria

#### Scenario: Honest near-random passes, dishonest high score fails

- GIVEN a baseline scoring barely above random with chronological split, purge, and embargo intact
- WHEN success is judged
- THEN it counts as slice success, whereas any high score obtained via random split, forward join, or test-peeked threshold tuning counts as failure regardless of the number

### Requirement: Inference CLI with Disclaimer

The system MUST provide a `btc-signal --title <headline> --price <current-btc-eur>` command that outputs a predicted action (`buy` / `sell` / `hold`) plus class probabilities plus an educational disclaimer stating the output is not investment advice, and MUST return usable errors for missing or invalid inputs. The CLI MUST be inference-only and MUST NOT retrain the model.

#### Scenario: Happy-path inference

- GIVEN a trained model artifact and input `--title "Bitcoin ETF inflows hit record" --price 60000`
- WHEN `btc-signal` runs
- THEN it outputs one action, the probability distribution over `buy`/`sell`/`hold`, and a disclaimer that the output is for learning only and is not investment advice

#### Scenario: Missing input gives usable error

- GIVEN invocation without `--title` or without `--price` or with a non-numeric price
- WHEN `btc-signal` runs
- THEN it exits with a non-zero status and a message naming the missing or invalid argument and showing correct usage

#### Scenario: CLI never retrains

- GIVEN a CLI invocation with a headline and price
- WHEN `btc-signal` runs
- THEN no training data, model weights, or data card files are modified

### Requirement: Reproducibility via Data Card

The system MUST produce a `data_card.md` alongside every dataset build recording record counts, date range, null counts, duplicate rate, the `flat_threshold` and `embargo_days` values used, the split boundaries, and the fixed seed, such that the dataset rebuilds reproducibly from raw CSVs.

#### Scenario: Data card documents a build

- GIVEN a completed dataset build from raw Kraken and news CSVs
- WHEN the build finishes
- THEN `data_card.md` exists and lists input counts, post-dedup counts, date range, nulls, duplicate rate, `flat_threshold`, `embargo_days`, split boundaries with purge/embargo gaps, and seed

#### Scenario: Rebuild reproduces the card

- GIVEN the same raw CSVs plus the same `data_card.md` parameters
- WHEN the dataset is rebuilt
- THEN record counts, labels, and splits match the original build

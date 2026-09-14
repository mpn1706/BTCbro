# DECISION_LOG — btc-news-signal (audit trail)

## D1 — Kaggle news for slice 1, GDELT in phase 2
Kaggle crypto-news (30–50k rows, 2018+) is small/fast enough for the
400-line CPU slice; GDELT full backfiles are complete but heavy.
Revisit in slice 2 without re-ingest (`lang` column already kept).

## D2 — flat_threshold / embargo chosen on val-2023 only
`flat_threshold ∈ {0.003, 0.005, 0.01}` (default 0.005) and
`embargo_days ∈ {1, 2}` (default 1) are selected by macro-F1 on val
2023. Test 2024+ stays locked until one final report; any test-peeked
tuning counts as failure even with a high score.

## D3 — TF-IDF(title+body) + balanced LogReg, CPU-only
`TfidfVectorizer(max_features 5k–20k, ngram (1,2), sublinear_tf=True)` +
`LogisticRegression(class_weight='balanced', random_state=42,
max_iter=1000, solver='lbfgs')`. Balanced weights stop hold-dominance
from silently winning; seed-fixed for reproducibility; no LLM/GPU.

## D4 — Single final test-2024 report, per-class gate
Eval must show per-class precision/recall + macro-F1 + 3×3 confusion
matrix (buy/hold/sell); accuracy-alone is incomplete/FAIL.
Near-random + honest protocol = PASS.

## D5 — CLI is inference-only
`btc-signal` loads `models/latest` read-only and never writes
`data/`/`models/`. `--price` is display context, not a trained
feature. Exits: 0 ok, 2 bad input, 3 missing/corrupt artifact, 1 internal.

## D6 — PR2 size:exception accepted
PR2 (dataset) landed at 472 new lines over the 400-line PR budget because
threshold/split/purge tests and the data_card contract cannot be cut without
losing guarantees. Accepted as `size:exception` under the approved 4-PR
stacked chain; PR1/PR3/PR4 stayed within budget.

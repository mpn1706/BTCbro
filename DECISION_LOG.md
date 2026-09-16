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

## D7 — Mini-demo real 2026 (NewsAPI 97 + Kraken live) honest, test-once 2026-09-15
train 44 / val 20 / test 24 (thr 0.005, emb 1, bounds 06-10/06-20).
Baseline test macro-F1 0.216 (never predicts buy) vs LLM v2 macro-F1 0.301,
parse 24/24, 7.9s/news. Both near-random = honest PASS, no test repeek.
Balabaskar discarded for labeling (day-10 timestamp clamping); kept for text only.
Prices: live Kraken XXBTZEUR 721d. See data/interim/mini_compare.json.

## D8 — PR2 done CPU-only + big honest split (2026-09-15)
PR2 TDD: RED (no src/compare) -> GREEN (src/compare.py + CLI --backend) -> 59/59
suite green, no baseline regression. CLI: default baseline unchanged; --backend llm
same contract (exit 0/2/3). Manual demo llm exit 0, missing-baseline exit 3.
No GPU anywhere: --n-gpu-layers 0, no device param (box has no GPU to target).
Sample: kashnitsky 40k discarded (year-only, no timestamps); balabaskar discarded
(day-10 clamping). imadallal-2021-2024 (11295) IS the professor's source
(train 5375/val 4572 match his 5374/4570). Test=165 (his 1334) — gap Mar-Sep 2024
without prices (krairy ends 03-19, live starts 09-25; Stooq blocked). Big run
launched background w/ cache resume; see logs/big_llm.log, data/interim/big_compare/.

## D9 — Veredicto final muestra grande (STEP=4, n=334, thr 0.005, test-once)
Baseline macro-F1 0.316 vs LLM 0.283 → gana baseline, confirma al profe
(él 0.31 vs 0.26 con thr 0.003). 333/334 parseadas. Capital 1.0: HODL 1.32 >
baseline 0.92 > LLM 0.89. Incidente encoding Windows (charmap 0x9d) fijado con
utf-8/errors=replace; 1 abstención honesta por output roto. Web final en
web/data.json (submuestra documentada). Ver data/interim/big_compare/compare.json.

## D10 — Corrección de etiqueta + veredicto extractor (2026-09-16)
La línea de consola del export etiquetaba baseline/LLM cruzados en capital
(solo el print; tablas/JS siempre por clave). Valores reales web-sub:
HODL 1.316 > EMB 1.17 > LLM 0.922 > baseline 0.89.
Extractor (Qwen3-embeds + LogReg C=0.5, C elegido en val): test virgen n=1000
macro-F1 0.368 vs baseline 0.336 en las mismas filas; web-sub 0.339 vs 0.316/0.283.
Tercer método real en la web (emb); LoRA sigue pendiente de GPU.

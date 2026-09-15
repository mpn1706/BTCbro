# Tasks — btc-llm-direct (experimental, alongside frozen baseline)

## Review Workload Forecast

| Field | Value |
|---|---|
| Estimated changed lines | ~280-360 (PR1 ~150 + PR2 ~150, docs included) |
| 400-line budget risk | Medium (fits single PR but chain preferred for review) |
| Chained PRs recommended | Yes (professor parity: PR1 backend+cache, PR2 CLI+compare+docs) |
| Suggested split | PR1 `src/llm_backend.py` + prompt v2 + parser/cache tests; PR2 `--backend` + `compare.py` + metrics + docs |
| Delivery strategy | ask-on-risk |
| Chain strategy | stacked-to-main |

> Chain 2 PRs stacked-to-main. Slice 1 (`src/dataset.py`, baseline weights,
> `btc-news-signal` specs) frozen — no edits there. Each PR independently
> verifiable, rollback = delete its new files + restore CLI flag default.

## Temporal Discipline — EXPLICIT (inherits slice 1, binding)

- Train learns weights only (baseline already frozen; LLM has no weights).
- Valid chooses prompt version + threshold only, by macro-F1.
- Test locked until ONE final `compare` report; second test read = failure.
- Every task touching prompt/eval MUST re-affirm this in verification.

## PR1 — LLM backend + cache + prompt v2 (target <200 lines)

### PR1 RED — failing parser/cache tests first

- [x] RED: `tests/test_llm_backend.py` with captured raw logs: strict
  `ACTION: BUY + REASON`, bare `BUY` without prefix, broken output → abstain
  (`hold, parse_ok=false`), banner-immunity (`Loading model...` never parsed
  as reason), timeout → abstain (mocked `subprocess`). Fails, no `src/` yet.
- [x] RED: confirm via `python -m pytest tests/test_llm_backend.py -v` (red).

### PR1 GREEN — minimal backend

- [x] GREEN: `prompts/llm_v2.txt` (ACTION system + user template, params
  `temp 0, seed 42, reasoning off, -t 4, -c 1024` stamped).
- [x] GREEN: `src/llm_backend.py` — `predict_llm(title, body, price)`:
  CPU-básica `llama-cli` call (`--n-gpu-layers 0 -st --simple-io </dev/null`,
  timeout 120s), windowed tolerant parser (strict + bare fallback), abstain
  on fail/timeout, `llm_cache.jsonl` read/append, no writes elsewhere.
- [x] GREEN: `python -m pytest tests/test_llm_backend.py -v` green (mocked,
  no GGUF, no network; real-GGUF path skips with explicit message).

### PR1 TRIANGULATE + REFACTOR

- [x] TRIANGULATE: edge cases (lowercase `action: sell`, `REASON` missing →
  second-line fallback, `BUY` inside title not parsed as action, cache hit
  avoids subprocess call asserted by mock).
- [x] REFACTOR: parser pure + window fn unit-tested; call builder single
  place; keep PR1 diff <400 lines.

**PR1 verification:** mocked pytest green; manual
`bash tools/run_llm_smoke.sh` exit 0 ~7s; `python spikes/llm_direct.py --run 5`
5/5 parsed; rollback = delete `src/llm_backend.py`, `prompts/llm_v2.txt`.

## PR2 — CLI switch + same-split compare + docs (target <200 lines)

### PR2 RED — failing CLI/compare tests

- [x] RED: `tests/test_llm_cli_compare.py`: `--backend baseline` unchanged
  behaviour, `--backend llm` same JSON shape + `backend/prompt_version/
  parse_ok` fields, missing GGUF → exit 3 naming path (mocked), compare on
  fixtures emits both 3x3 + both macro-F1 + shared `split_hash`, accuracy-alone
  rejected. Fails, no CLI/compare yet.
- [x] RED: confirm via `python -m pytest tests/test_llm_cli_compare.py -v`.

### PR2 GREEN — switch + report

- [x] GREEN: `src/cli.py --backend baseline|llm` (default `baseline`;
  llm path calls `predict_llm`, records wall time; inference-only + cache-append
  exception documented).
- [x] GREEN: `src/compare.py` — loads frozen splits, runs both backends,
  writes `metrics_compare.json` (per-class F1 x2, 3x3 x2,
  `parse_failure_rate`, `mean_wall_s`, `split_hash`, `prompt_version`).
- [x] GREEN: `python -m pytest tests/test_llm_cli_compare.py -v` green.

### PR2 TRIANGULATE + REFACTOR + audit

- [x] TRIANGULATE: `--backend typo` → exit 2 usage; corrupt cache line skipped
  + counted; probs-shape parity baseline vs llm.
- [x] REFACTOR + audit: `DECISION_LOG.md` entry (ACTION vs JSON, `-st` hang
  root-cause, `--jinja+schema` Qwen3 bug, `-t 4 -c 1024` envelope, discards
  Llama-3.2-1B/Qwen3.5-2B); README spike section; full `python -m pytest` green;
  keep PR2 diff <400 lines.

**PR2 verification:** full pytest green; demo both backends on one fixture;
`metrics_compare.json` has both matrices + shared hash; rollback = revert CLI
flag + delete `src/compare.py`.

## Definition of Done (parity gate with professor)

- [ ] `spikes/llm_direct.py --run 5` → 5/5 parsed, ~6-7s/news (done, 2026-09-15).
- [ ] Full 50 synthetic deferred to PR1 GREEN manual (5-6 min, close apps).
- [ ] `metrics_compare.json` on fixtures: both backends, shared split hash.
- [ ] Real-data compare (Kaggle + Kraken) explicitly OUT of these tasks —
  needs `kaggle auth login` + price-window decision (professor: overlap
  train 2024-09→2025-06, valid →2025-12, test 2026→hoy; or rethink source if
  dataset is pre-2024). Tracked as next change, not here.

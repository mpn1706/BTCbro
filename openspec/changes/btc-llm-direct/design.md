# Design — btc-llm-direct (Windows CPU adaptation)

## 1. Overview

**Change:** `btc-llm-direct` (experimental, alongside frozen baseline).
**Goal:** direct LLM classifier on this box with professor-track parity:
`ACTION: BUY|SELL|HOLD + REASON`, greedy + thinking off, tolerant parser,
CPU-básica tuning, honest same-split comparison.
**Non-goals:** no fine-tune, no API, no trading, no replacing TF-IDF yet,
no torch/transformers/ollama, no frozen `models/<run_id>/` for the LLM until
it beats chance honestly.

**Thesis:** keep the LLM path OUT of `src/dataset.py` leakage core. Dataset
stays frozen; the LLM is a pure `predict(title, body, price) -> action`
function with cache + abstain, compared on identical splits.

```
fixtures or real news+prices
  -> build_dataset (FROZEN, slice 1) -> (train,val,test) + split_hash
  -> baseline predict (TF-IDF, frozen) ─┐
  -> llm predict (llama-cli CPU) ───────┤-> compare.py -> metrics.json
        ^ prompt vX, cache, abstain      │   per-class F1 x2 + 3x3 x2
        └ models/llm/*.gguf (gitignored)┘   + parse_failure_rate
```

## 2. Where Things Live

| Path | What | Git? |
|---|---|---|
| `models/llm/Qwen3-1.7B-Q4_K_M.gguf` (~1.1 GB) | Qwen3-1.7B Q4_K_M, Apache-2.0, no gate | gitignored |
| `tools/llama.cpp/bin/llama-cli(.exe)` + DLLs | prebuilt CPU runtime, build b10964 | gitignored (keep `win-cpu.zip` URL in card) |
| `tools/run_llm_smoke.sh` | 1-news smoke: `-t 4 -c 1024 --n-gpu-layers 0 -st --simple-io --reasoning off` | committed |
| `spikes/llm_direct.py` | throwaway 50-synthetic generator + tolerant parser + runner | committed (spike only) |
| `spikes/llm_prompt_v1.txt` | JSON prompt v1 (superseded, kept for history) | committed |
| `prompts/llm_v2.txt` (on promotion) | ACTION prompt v2 frozen | committed on PR1 |
| `src/llm_backend.py` (PR1) | `predict_llm()` + cache + parser + timeout/abstain | committed |
| `src/compare.py` (PR2) | same-split baseline-vs-llm report | committed |
| `src/cli.py --backend` (PR2) | `baseline\|llm` switch, same JSON+disclaimer | committed |
| `data/`, `models/<run_id>/` | payloads, never committed | gitignored |

Discards recorded (professor parity): Llama-3.2-1B needs Meta gated accept
(403 for current account); Qwen3.5-2B is multimodal + video preprocessor,
overkill for text-only.

## 3. LLM Call (CPU-básica, single-turn)

```
llama-cli -m models/llm/Qwen3-1.7B-Q4_K_M.gguf
  --n-gpu-layers 0 -c 1024 -t 4
  --temp 0 --seed 42 -n 60
  --reasoning off --no-display-prompt --simple-io -st
  -sys "<SYSTEM v2>" -p "<TITLE/BODY/PRICE + Respond with ACTION and REASON lines only.>"
  </dev/null   # never leave stdin open (was the hang: missing -st waited forever)
```

- Measured on this box: ~6-7s/news, prompt ~40-55 t/s, gen ~10-13 t/s.
- 50 news ≈ 5-6 min. Run 5 first, then 50. Close heavy apps (peak ~4-5 GB).
- `--jinja + --json-schema` with Qwen3 breaks
  (`empty grammar stack ... <|im_start|>`). Spike uses prompt-JSON + tolerant
  parser. Grammar only with `--no-jinja` if ever needed — not for the spike.

## 4. Parser (tolerant, windowed)

1. Cut generation window: after last `"> "`/prompt-echo line, before `"[ Prompt"`.
   (Avoids parsing `Loading model...` banner.)
2. Strict: `ACTION\s*:\s*(BUY|SELL|HOLD)` (case-insensitive).
3. Fallback: first bare `^(BUY|SELL|HOLD)\b` in window (model sometimes drops
   the prefix — observed 4/5 cases before the fix).
4. `REASON:` strict, else first non-action content line of the window.
5. No action found / timeout → `hold, parse_ok=false, conf=0.0`.
   Never a silent confident default. `parse_failure_rate` is a first-class metric.

Reference implementation: `spikes/llm_direct.py:parse_output` (+ window fn).

## 5. Cache

- Key: `sha1(prompt_version + \x00 + title_norm + \x00 + body_norm + \x00 + price_bucket)`.
- Store: `data/interim/llm_cache.jsonl` (gitignored) `{key, action, reason_short,
  parse_ok, wall_s, prompt_version, model_sha16}`.
- Reruns hit cache; `--no-cache` forces re-inference. Cache file hash goes in
  the comparison card so numbers are reproducible without re-running the model.

## 6. CLI Contract (`--backend`)

```
btc-signal --title "..." --price 60000 [--backend baseline|llm] [--json]
```

- Default `--backend baseline` = slice-1 behaviour unchanged.
- `--backend llm`: same stdout shape + `backend: llm`, `prompt_version`,
  `parse_ok`, `wall_s`. Same exit codes as slice 1 (0 ok / 2 bad input /
  3 missing model / 1 internal). Missing GGUF → exit 3 naming expected path,
  never auto-download.
- Inference-only: no writes to `data/`/`models/` except the allowed
  `llm_cache.jsonl` append (documented, hash-guarded in tests).

## 7. Comparison Report

`python -m src.compare --split-hash <h> --out models/<run_id>/metrics_compare.json`:

```json
{
  "split_hash": "…", "prompt_version": "v2",
  "baseline": {"macro_f1": 0.33, "per_class_f1": {}, "cm_3x3": []},
  "llm": {"macro_f1": 0.0, "per_class_f1": {}, "cm_3x3": [],
          "parse_failure_rate": 0.0, "mean_wall_s": 6.4},
  "n_val": 0, "n_test": 0
}
```

Val chooses prompt/threshold; test runs once. Accuracy informational only.

## 8. TDD Plan (maps to tasks.md chain)

- PR1 (backend+cache): parser unit tests on captured raw logs (strict,
  bare-token, broken→abstain, banner-immunity) + cache hit/miss + timeout-abstain
  (mocked subprocess) + GGUF-missing skip rule. No network, no real model.
- PR2 (CLI+compare): `--backend` parity tests, exit-3 on missing GGUF,
  same-split compare test on fixtures (both matrices present, split hash equal).

## 9. Risks

| Risk | Mitigation |
|---|---|
| 8 GB RAM pressure | `-c 1024 -t 4`, 5-first then 50, close apps, timeout 120s |
| Interactive hang regression | `-st --simple-io </dev/null`, smoke asserts `Exiting...` |
| Grammar/template bug | no `--json-schema` with `--jinja` on Qwen3; parser over prompt-JSON |
| Prompt brittleness | versioned prompt + `parse_failure_rate` gate; val-only tuning |
| Test contamination | LLM tests mocked; real GGUF never in `pytest` path |
| Scope creep into baseline | `src/dataset.py` frozen; LLM is a side predictor only |

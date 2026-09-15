# BTC LLM Direct Specification (experimental, Windows CPU box)

## Purpose

Spike-then-compare a direct LLM classifier alongside the frozen TF-IDF baseline,
on THIS box (Windows 11 Intel, 8 GB RAM, no GPU/torch/ollama), with identical
honesty gates. Slice 1 (`btc-news-signal`) is untouched. No fine-tuning, no API,
no trading. Educational only.

## Requirements

### Requirement: LLM Backend Contract (ACTION format)

The system MUST expose an LLM backend mapping `(title, body, price_at_publish)`
to `buy | sell | hold` via `llama.cpp` CPU with greedy decoding and thinking off,
emitting EXACTLY `ACTION: BUY|SELL|HOLD` + `REASON: <=12 words`. Parse failures
and timeouts MUST abstain to `hold` with `parse_ok=false`, never a silent
confident default. Parse-failure rate MUST be reported as a metric.

#### Scenario: Happy-path bullish news

- GIVEN title "Bitcoin ETF inflows hit record" + body + price 60000
- WHEN the LLM backend runs (`--temp 0 --reasoning off -t 4 -c 1024`)
- THEN output is `action=buy`, `conf` in `[0,1]`, `reason_short` non-empty,
  `parse_ok=true`, wall time recorded

#### Scenario: Broken output abstains honestly

- GIVEN model output without any `BUY|SELL|HOLD` token (e.g. empty/timeout)
- WHEN parsing runs (strict `ACTION:` + tolerant bare-token fallback on the
  generation window only)
- THEN result is `action=hold`, `parse_ok=false`, `conf=0.0`, counted in
  `parse_failure_rate`

#### Scenario: Timeout abstains

- GIVEN `llama-cli` exceeds 120s for one news
- WHEN the runner enforces timeout
- THEN the item is marked `timeout=true`, `parse_ok=false`, `action=hold`

### Requirement: Versioned Prompt

The system MUST keep the system prompt + user template in a versioned file
(e.g. `spikes/llm_prompt_v2.txt` for the spike, `prompts/llm_v1.txt` frozen on
promotion), recording `temp`, `seed`, `reasoning=off`, `-t/-c`, and backend
version. Prompt changes MUST bump the version and re-run the fixtures.

#### Scenario: Prompt bump is traceable

- GIVEN a prompt edit
- WHEN evaluation runs
- THEN `metrics.json` stamps `prompt_version` and results are not compared
  across versions without noting the bump

### Requirement: CPU-Only Runtime Budget

The system MUST run CPU-only (`--n-gpu-layers 0`), single-turn (`-st`,
`--simple-io`, stdin closed), context `<=1024`, threads `<=4`, tokens `<=100`
per news on this box. Expected envelope: `<=15s/news`, `<=6 GB` peak RAM.
Heavy apps closed during the spike. No torch/transformers/ollama.

#### Scenario: Smoke fits the box

- GIVEN 1 fixture news via `tools/run_llm_smoke.sh`
- WHEN run on the 8 GB laptop
- THEN exit 0 within 120s, `Exiting...` reached (no interactive hang)

### Requirement: Model + Binary Provenance (no large blobs in git)

The system MUST NOT commit the GGUF (`~1.1 GB`) nor `tools/llama.cpp/bin/`
binaries. `models/llm/*.gguf` and `tools/llama.cpp/bin/*` MUST be gitignored
(except `.gitkeep`/checksums). The data card / metrics MUST record model name
(`Qwen3-1.7B-Q4_K_M`), file size + sha256 (first 16 ok in spike), and
`llama.cpp` build id. Download requires explicit user OK.

#### Scenario: Fresh clone reproduces without blobs

- GIVEN a fresh clone without `models/llm/*.gguf`
- WHEN tests run (`python -m pytest`)
- THEN all tests pass with the LLM backend mocked; real-GGUF tests skip
  with an explicit message (no network, no download in tests)

### Requirement: CLI Backend Switch (baseline untouched)

The CLI MUST gain `--backend baseline|llm` (default `baseline`), with identical
JSON + disclaimer contract for both backends. Baseline path MUST be byte-equal
behaviour to slice 1. `--backend llm` MUST record `backend`, `prompt_version`,
`parse_ok`, and wall time per call.

#### Scenario: Backend parity on CLI contract

- GIVEN `--title "Bitcoin ETF inflows hit record" --price 60000`
- WHEN run with `--backend baseline` vs `--backend llm`
- THEN both emit `{action, conf/probs, reason_short, disclaimer}` with exit 0,
  differing only in predicted values + `backend` field

### Requirement: Honest Comparison on the Same Split

Comparison MUST reuse slice-1 honesty: asof-backward join (past only),
`ret_24h` labels, purge 24h + embargo, valid chooses prompt/threshold, test
certifies exactly once. Both backends evaluate on the SAME frozen
`(train,val,test)` splits. `metrics.json` MUST contain per-class F1 for both
backends + 3x3 matrix each; accuracy informational only.

#### Scenario: Same-split comparability

- GIVEN frozen splits from `build_dataset`
- WHEN `evaluate --compare baseline llm` runs
- THEN output includes `baseline.macro_f1`, `llm.macro_f1`,
  `llm.parse_failure_rate`, both 3x3 matrices, and the shared split hash

#### Scenario: No test-peeked prompt tuning

- GIVEN test split locked
- WHEN prompt/threshold is selected on val
- THEN test is read exactly once for the final report; any second read is a
  failure even with high score

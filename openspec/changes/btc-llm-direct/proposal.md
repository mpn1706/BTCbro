# Proposal — btc-llm-direct (experimental, adapted to Windows CPU box)

## Problem
Slice 1 (`btc-news-signal`, verify PASS, archive ready) proves the honest pipeline
with TF-IDF + LogReg. The professor's track moves to a direct LLM classifier
(headline + price → buy/sell/hold). That track was designed for a Linux + ROCm
box (AMD 780M, 61 GiB RAM, torch HIP). This box is Windows 11 Intel, 8 GB RAM,
no torch/transformers/ollama — so the change is adapted, not copied.

## Goal
Spike a direct LLM classifier on THIS box without touching the baseline:
- Model: Qwen3-1.7B Q4_K_M GGUF (~1.1–1.3 GB, Apache-2.0, no gate — Llama-3.2-1B
  is Meta-gated, 403 for the current account; Qwen3.5-2B is multimodal and
  discarded for this text-only task).
- Runtime: llama.cpp prebuilt Windows binary (CPU, AVX2, 8 threads). No torch,
  no transformers, no ollama. Context kept small (2k) to fit 8 GB RAM.
- Qwen3 thinks by default → spike uses `/no_think` + JSON-constrained output
  `{action: buy|sell|hold, conf: 0-1, reason_short: str}` with a versioned
  prompt file and a robust parser (retry-or-abstain, never silent default).
- Same honesty gates as slice 1: asof-backward join (past only), purge 24h +
  embargo, valid chooses prompt/threshold, test certifies exactly once.
- CLI gains `--backend baseline|llm` with the same JSON + disclaimer contract.
- Eval: `metrics.json` with per-class F1 for both backends + 3×3 matrix;
  accuracy informational only.

## Non-goals
- No fine-tuning. No external API. No real trading. No replacing TF-IDF yet.
- No ollama / torch / transformers on this box. No multimodal models.
- No frozen `models/<run_id>/` for the LLM until the spike beats chance honestly.

## Scope step 0 (done, zero-install)
Keyword-sentiment heuristic as a backend stand-in: validates the score →
threshold → label → per-class F1 wiring with no downloads (`spikes/` only,
throwaway, baseline untouched).

## Scope step 1 (needs explicit download OK, ~1.3 GB)
Download GGUF + llama.cpp binary, run the 50-news spike (fixtures first, then
real news if ingested), measuring wall-time and peak RAM. If the 1.7B does not
separate above chance, the finding is recorded and nothing breaks.

## Risks
- 8 GB RAM (2.4 free): keep context ≤2k, close heavy apps during the spike.
- CPU speed: ~10s/news expected → 50 news ≈ 10 min. Acceptable for a spike.
- Prompt brittleness: versioned prompt + parse-failure rate reported as metric.

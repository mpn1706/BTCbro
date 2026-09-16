#!/usr/bin/env bash
# Smoke liviano CPU-basica: Win11 Intel 8GB sin GPU.
# Fluidez: -t 4, -c 1024, --n-gpu-layers 0, -st (single-turn), --simple-io.
# Qwen3: --reasoning off + prompt JSON (sin --json-schema con --jinja por bug <|im_start|>).
# Uso: bash tools/run_llm_smoke.sh
set -u
timeout 120 tools/llama.cpp/bin/llama-cli \
  -m models/llm/Qwen3-1.7B-Q4_K_M.gguf \
  --n-gpu-layers 0 -c 1024 -t 4 \
  --temp 0 --seed 42 -n 100 \
  --reasoning off --no-display-prompt --simple-io -st \
  -sys 'You are an educational BTC news classifier. Output ONLY JSON: {"action":"buy"|"hold"|"sell","conf":0..1,"reason_short":"<=12 words"}. buy=bullish 24h, sell=bearish, else hold.' \
  -p 'Classify this news. Return ONLY {"action","conf","reason_short"}. TITLE: Bitcoin ETF inflows hit record BODY: ETF bodies attract record flows. PRICE_AT_PUBLISH_EUR: 60000' \
  </dev/null 2>&1 | tail -20

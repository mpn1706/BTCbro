# ₿ BTC News Signal + LLM directo (educativo)

Modelo local que analiza noticias de Bitcoin y sugiere **comprar**, **vender** o
**mantener**, con web comparativa de 4 métodos, backtest y simulador.
Proyecto de curso de *vibe coding*: el objetivo es demostrar un proceso de ML
honesto y reproducible, no lograr un modelo de trading real.

> ⚠️ **Aviso**: proyecto educativo. No es asesoramiento financiero. Nada opera
> de verdad. En ventanas alcistas, HODL le gana a tradear señales flojas
> (la propia web lo muestra: HODL 1.32x > EMB 1.17x > LLM 0.92x > baseline 0.89x;
> el LoRA colapsado queda en cash 1.00x).

## Qué hace

1. Descarga noticias (Kaggle) y precio diario (Kraken `XXBTZEUR`, Kraken en vivo).
2. Cruza cada noticia con el precio: etiqueta `buy`/`sell`/`hold` según el
   retorno 24h (`ret_24h` vs umbral `flat_threshold`).
3. Entrena un baseline TF-IDF + Regresión Logística (scikit-learn, CPU).
4. Corre un LLM local (Qwen3-1.7B, llama.cpp, CPU) como clasificador directo.
5. Suma un extractor Qwen-embeds + LogReg y un fine-tune LoRA (Qwen3-1.7B, GPU en Kaggle).
6. Compara los 4 en el mismo split honesto y lo muestra en la web con velas,
   señales, backtest, simulador de monto y testigo HODL.

## Regla anti-trampa (lo que vale del proyecto)

Join-asof pasado (nunca vela futura) · split cronológico + purge 24h + embargo ·
valid elige, test se lee **una sola vez** · accuracy-alone no vale (solo
per-class + macro-F1 + matriz 3×3). Un negativo honesto > un positivo con leakage.

## Resultado

Test grande (1334; LoRA en complemento disjunto de 1000, thr 0.005):

| Método | n | macro-F1 | Capital 1.0 |
|---|---|---|---|
| Extractor Qwen-embeds + LogReg | 1000 | 0.368 | 1.17x |
| Baseline TF-IDF + LogReg | 1000 | 0.336 | 0.89x |
| LLM Qwen3 zero-shot | 334 | 0.283 | 0.92x |
| LoRA Qwen3 (r=16, 2 épocas, T4) | 1000 | 0.197 | 1.00x |
| HODL (testigo) | — | — | 1.32x |

En la submuestra web-334 el F1 es 0.316 / 0.283 / 0.339 (ver `web/data.json`
`meta.n`). El extractor gana entre los modelos, pero HODL le gana a todos:
con señales flojas, tradear pierde contra no hacer nada. El LoRA colapsó a
predecir casi todo sell — negativo honesto (D11). Ver
`cloud/lora_compare.json` y `DECISION_LOG.md`.

## Cómo correrlo en tu computadora

```bash
git clone https://github.com/mpn1706/BTCbro.git
cd BTCbro
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
pip install -e .                  # instala btc-signal
pip install pytest                # si no lo tenés
python -m pytest                  # 84 tests (sin red, sin GPU)
```

CLI (baseline y LLM):

```bash
btc-signal --title "Bitcoin ETF inflows hit record" --price 60000
btc-signal --title "..." --price 60000 --backend llm
```

Web comparativa (sin servidor no carga el `data.json`; con servidor sí):

```bash
cd web && python -m http.server 8000
# abrir http://localhost:8000
```

Regenerar datos de la web (determinista):

```bash
PYTHONPATH=. python tools/export_web.py
```

El modelo LLM (~1.1 GB) y los CSV crudos **no** viajan en git (ver `.gitignore`);
la guía dice cómo conseguirlos (`tools/real_data.md`).

## Estructura

```
src/            # config, ingesta, dataset honesto, train, evaluate, cli,
                # llm_backend (Qwen3 CPU), compare, backtest
tests/          # 84 tests espejo del contrato (incluye fixtures tiny)
web/            # index.html fijo + data.json generado (velas, backtest, precedentes)
tools/          # export_web, corridas, spikes documentados, guías de datos
prompts/        # prompts versionados del LLM (v2, v3 en duelo)
docs/           # guia-usuario.md + DECISION_LOG.md en raíz
openspec/       # artefactos SDD (proposal/spec/design/tasks/verify)
```

## Tecnologías

Python (pandas, scikit-learn, joblib, requests) · llama.cpp (CPU) · HTML/CSS/JS
vainilla (sin dependencias) · pytest con TDD estricto.

## Guía de usuario

Ver [docs/guia-usuario.md](docs/guia-usuario.md) (web, CLI, metodología y límites).

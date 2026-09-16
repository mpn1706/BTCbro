# ₿ BTC News Signal + LLM directo (educativo)

Modelo local que analiza noticias de Bitcoin y sugiere **comprar**, **vender** o
**mantener**, con web comparativa de 3 métodos, backtest y simulador.
Proyecto de curso de *vibe coding*: el objetivo es demostrar un proceso de ML
honesto y reproducible, no lograr un modelo de trading real.

> ⚠️ **Aviso**: proyecto educativo. No es asesoramiento financiero. Nada opera
> de verdad. En ventanas alcistas, HODL le gana a tradear señales flojas
> (la propia web lo muestra: 1.32x > 0.92x > 0.89x).

## Qué hace

1. Descarga noticias (Kaggle) y precio diario (Kraken `XXBTZEUR`, Kraken en vivo).
2. Cruza cada noticia con el precio: etiqueta `buy`/`sell`/`hold` según el
   retorno 24h (`ret_24h` vs umbral `flat_threshold`).
3. Entrena un baseline TF-IDF + Regresión Logística (scikit-learn, CPU).
4. Corre un LLM local (Qwen3-1.7B, llama.cpp, CPU) como clasificador directo.
5. Compara ambos en el mismo split honesto y lo muestra en la web con velas,
   señales, backtest, simulador de monto y testigo HODL.

## Regla anti-trampa (lo que vale del proyecto)

Join-asof pasado (nunca vela futura) · split cronológico + purge 24h + embargo ·
valid elige, test se lee **una sola vez** · accuracy-alone no vale (solo
per-class + macro-F1 + matriz 3×3). Un negativo honesto > un positivo con leakage.

## Resultado

Test grande (1334 → submuestra 334, thr 0.005): baseline macro-F1 **0.316** vs
LLM **0.283** → gana el baseline; el LLM zero-shot queda bajo el azar (0.33).
Ver `data/interim/big_compare/compare.json` y `DECISION_LOG.md`.

## Cómo correrlo en tu computadora

```bash
git clone https://github.com/mpn1706/BTCbro.git
cd BTCbro
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
pip install -e .                  # instala btc-signal
pip install pytest                # si no lo tenés
python -m pytest                  # 70+ tests (sin red, sin GPU)
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
tests/          # 70+ tests espejo del contrato (incluye fixtures tiny)
web/            # index.html fijo + data.json generado (velas, backtest, precedentes)
tools/          # export_web, corridas, spkes documentados, guías de datos
prompts/        # prompts versionados del LLM (v2, v3 en duelo)
docs/           # guia-usuario.md + DECISION_LOG.md en raíz
openspec/       # artefactos SDD (proposal/spec/design/tasks/verify)
```

## Tecnologías

Python (pandas, scikit-learn, joblib, requests) · llama.cpp (CPU) · HTML/CSS/JS
vainilla (sin dependencias) · pytest con TDD estricto.

## Guía de usuario

Ver [docs/guia-usuario.md](docs/guia-usuario.md) (web, CLI, metodología y límites).

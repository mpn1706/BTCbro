# Guía de usuario — comparativa de 4 métodos

> Educativo. No es asesoría financiera. Nada acá opera de verdad (disclaimer).

## 1. Qué es y qué NO es

Clasificador de noticias BTC → `buy`/`hold`/`sell` con cuatro motores: **baseline**
(TF-IDF + LogReg, entrenado), **llm** (Qwen3-1.7B zero-shot, CPU), **emb**
(Qwen-embeds + LogReg, entrenado) y **lora** (Qwen3-1.7B con fine-tune LoRA en
GPU). El LoRA figuraba como pendiente hasta D11: colapsó a predecir casi todo
sell (F1 0.197) y su curva queda plana en 1.00x — negativo honesto, sin curva inventada.

## 2. La web (`web/index.html`, doble clic, sin servidor)

- **Velas**: verde sube / roja baja, ventana del test honesto.
- **Marcadores**: ▲ buy / ▼ sell por motor (toggles); HOLD no marca.
- **Simulador**: poné el monto en EUR y todo se reescala. Curvas: baseline,
  llm, emb, lora y **HODL** (comprar el día 1 y no tocar).
- **Lectura incómoda**: HODL 1.32x > EMB 1.17x > LoRA 1.00x (cash) > LLM 0.92x >
  baseline 0.89x con capital 1.0. Es el testigo haciendo su trabajo: con señales
  flojas, tradear pierde contra no hacer nada.

## 3. El CLI

```
btc-signal --title "..." --price 60000 [--backend baseline|llm] [--json]
```

Salidas (exit codes): `0` ok · `2` input inválido · `3` modelo faltante · `1` interno.
`--backend llm` suma `parse_ok` y `prompt_version`. JSON con `--json`.

## 4. Regenerar datos

```
PYTHONPATH=. python tools/export_web.py   # reescribe web/data.json (determinista)
```

La matemática vive en `src/backtest.py` (testeada); el JS solo dibuja.
El adapter LoRA ya corre como 4ª serie (evaluado en muestra disjunta n=1000, D11);
este comando la regenera.

## 5. Metodología y límites honestos

- Join-asof pasado, etiqueta 24h futura, purge 24h + embargo, valid elige,
  test se lee una vez. Submuestra documentada cuando aplica.
- Ventana grande: train 5375 / val 4572 / test 1334 con submuestra
  sistemática 1-de-4 (334, mismo protocolo del track de referencia).
  Gap Mar-Sep 2024 tapado con serie belbino (solo fechas faltantes).
  Precios mezclan USD+EUR en retornos (solo afecta display).
- Veredicto final: extractor 0.368 > baseline 0.336 > LLM zero-shot 0.283 >
  LoRA 0.197 (macro-F1; baseline/extractor/LoRA en n=1000, LLM en n=334).
  Capital 1.0: HODL 1.32 > EMB 1.17 > LoRA 1.00 (cash) > LLM 0.92 > baseline 0.89
  (el extractor es el único modelo que gana plata sin ser HODL; el LoRA colapsó a sell).
- Sin fees, all-in. Micro-comentarios con precios citados favorecen al baseline.

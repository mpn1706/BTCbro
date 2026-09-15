# Guía de usuario — comparativa baseline vs LLM

> Educativo. No es asesoría financiera. Nada acá opera de verdad (disclaimer).

## 1. Qué es y qué NO es

Clasificador de noticias BTC → `buy`/`hold`/`sell` con dos motores: **baseline**
(TF-IDF + LogReg, entrenado) y **llm** (Qwen3-1.7B zero-shot, CPU). Un tercer
motor **ft (LoRA) figura como pendiente**: requiere GPU y no existe en esta
máquina — la web lo dice, sin curva inventada.

## 2. La web (`web/index.html`, doble clic, sin servidor)

- **Velas**: verde sube / roja baja, ventana del test honesto.
- **Marcadores**: ▲ buy / ▼ sell por motor (toggles); HOLD no marca.
- **Simulador**: poné el monto en EUR y todo se reescala. Curvas: baseline,
  llm y **HODL** (comprar el día 1 y no tocar).
- **Lectura incómoda**: en ventanas alcistas HODL gana (acá 1.53x > 1.47x >
  1.13x con capital 1.0). Es el testigo haciendo su trabajo: con señales
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
Cuando exista el adapter LoRA, se agrega la 3ª serie acá.

## 5. Metodología y límites honestos

- Join-asof pasado, etiqueta 24h futura, purge 24h + embargo, valid elige,
  test se lee una vez. Submuestra documentada cuando aplica.
- Ventana grande: train 5375 / val 4572 / test 1334 con submuestra
  sistemática 1-de-4 (334, mismo protocolo del track de referencia).
  Gap Mar-Sep 2024 tapado con serie belbino (solo fechas faltantes).
  Precios mezclan USD+EUR en retornos (solo afecta display).
- Veredicto final: baseline macro-F1 0.316 vs LLM 0.283 (333/334 parseadas).
  Capital 1.0: HODL 1.32 > baseline 0.92 > LLM 0.89.
- Sin fees, all-in. Micro-comentarios con precios citados favorecen al baseline.

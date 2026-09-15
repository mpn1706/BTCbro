# Datos reales — setup pendiente de tu auth (1 minuto tuyo)

Estado: `kaggle` CLI instalado vía `pip install kaggle` (kagglesdk 0.1.37).
Verificado: `python -m kaggle` responde `Authentication required` = CLI ok,
falta credencial. Sin auth ni siquiera lista datasets (como le pasó al profe).

## Tu parte (elegí UNA)

**Opción A — OAuth (lo que pide el profe):**
```
python -m kaggle auth login
```
Te abre el browser para autorizar. Avisame cuando esté verde.

**Opción B — kaggle.json (la clásica):**
1. Entrá a kaggle.com/settings/api → Create New Token → baja `kaggle.json`.
2. Dejalo en `C:\Users\usuario\.kaggle\kaggle.json` (crear la carpeta si no existe).

Verificación:
```
python -m kaggle datasets list --search "bitcoin news" | head
```

## Candidatos del profe (de su captura)

- `kashnitsky/news-about-major-cryptocurrencies-20132018-40k` (40k, 2013-2018)
- `balabaskar/bitcoin-news-articles-text-corpora`
- `inadallal/sentiment-analysis-of-bitcoin-news-2021-2024`
- `gsanthoshkumar01/bitcoin-news-dataset-newsapi` (fresco, 2026-06)

Criterio: 2018+, con `title + published_at (+body/url)`, 30-50k filas ideal.
El de 2013-2018 sirve pero recorta ventana; el 2021-2024 calza con nuestro
split (train ≤2022 / val 2023 / test 2024+).

## Ventana de precios (decisión Kraken)

Kraken API devuelve solo las últimas 720 velas diarias (~2 años). Nuestro
`data/raw/kraken_xxbtzeur_1d.csv` actual es fixture (6 filas, ene-2024).
Al bajar noticias reales, hay dos caminos (decisión tuya, la anoto en DECISION_LOG):

1. **Overlap moderno (recomendado por el profe):** bajar precio actual
   (2024-09→hoy) y usar train 2024-09→2025-06, val →2025-12, test 2026→hoy.
   Requiere redefinir boundaries del DatasetConfig solo para ese run.
2. **Histórico clásico:** si el dataset es pre-2024, buscar fuente de precio
   con más historia (CSV bulk de Kraken descargado aparte, no API).

## Cuando el auth esté verde, corro yo

```
python -m kaggle datasets download -d <dataset> -p data/raw/ --unzip
python -m src.dataset --flat-threshold 0.005 --embargo-days 1  # rebuild + card
python -m pytest -q  # regresión baseline intacto
```

Después: PR2 (CLI `--backend` + `compare.py`) sobre el mismo split honesto.

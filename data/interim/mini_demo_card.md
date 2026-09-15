# Data Card — build dataset
- raw_news_rows: 97
- raw_price_rows: 721
- kept_news_rows: 97
- dropped_missing_fields: 0
- duplicate_rate: 0.0
- dropped_no_price: 0
- dropped_no_forward: 0
- purged_train_rows: 6
- embargoed_rows: 3
- news range: 2026-05-28 17:10:33+00:00 to 2026-06-27 18:11:04+00:00
- price range: 2024-09-25 00:00:00+00:00 to 2026-09-15 00:00:00+00:00
- flat_threshold: 0.005
- embargo_days: 1
- seed: 42
- split boundaries: train_end=2026-06-10 (B1=2026-06-11 00:00:00+00:00), val_end=2026-06-20 (B2=2026-06-21 00:00:00+00:00)
- purge cutoffs: 2026-06-10 00:00:00+00:00, 2026-06-20 00:00:00+00:00; embargo gaps: 1d -> cutoffs 2026-06-09 00:00:00+00:00, 2026-06-19 00:00:00+00:00
- per-split counts: train=44, val=20, test=24
- per-split label distribution: train={'sell': 22, 'hold': 15, 'buy': 7}, val={'buy': 8, 'sell': 8, 'hold': 4}, test={'sell': 16, 'hold': 6, 'buy': 2}
- per-split ranges: train=[2026-05-28 17:10:33+00:00..2026-06-08 21:55:04+00:00], val=[2026-06-11 10:49:51+00:00..2026-06-18 23:00:00+00:00], test=[2026-06-21 09:00:32+00:00..2026-06-27 18:11:04+00:00]
- sources: Kaggle crypto-news (allowlist) sha256=6de45c9af98513a6, Kraken XXBTZEUR sha256=90c7e5acbbf230b4
- versions: python=3.14.7, pandas=3.0.5, sklearn=1.9.1
- rebuild command: python -m src.dataset --flat-threshold 0.005 --embargo-days 1 --seed 42


> MINI-DEMO: NewsAPI 97 (2026-05-28->06-27, genuine ts) + live Kraken XXBTZEUR 721d. Balabaskar discarded for labeling (day-10 timestamp clamping). Boundaries 06-10/06-20.

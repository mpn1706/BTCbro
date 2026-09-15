# Data Card — build dataset
- raw_news_rows: 2402
- raw_price_rows: 4994
- kept_news_rows: 2402
- dropped_missing_fields: 0
- duplicate_rate: 0.0
- dropped_no_price: 0
- dropped_no_forward: 0
- purged_train_rows: 0
- embargoed_rows: 0
- news range: 2022-03-10 00:00:00+00:00 to 2022-09-10 23:56:00+00:00
- price range: 2010-07-18 00:00:00+00:00 to 2024-03-19 00:00:00+00:00
- flat_threshold: 0.005
- embargo_days: 1
- seed: 42
- split boundaries: train_end=2022-05-31 (B1=2022-06-01 00:00:00+00:00), val_end=2022-07-31 (B2=2022-08-01 00:00:00+00:00)
- purge cutoffs: 2022-05-31 00:00:00+00:00, 2022-07-31 00:00:00+00:00; embargo gaps: 1d -> cutoffs 2022-05-30 00:00:00+00:00, 2022-07-30 00:00:00+00:00
- per-split counts: train=1341, val=744, test=317
- per-split label distribution: train={'sell': 1341}, val={'sell': 744}, test={'hold': 317}
- per-split ranges: train=[2022-03-10 00:00:00+00:00..2022-05-10 23:55:00+00:00], val=[2022-06-10 00:00:00+00:00..2022-07-10 23:54:00+00:00], test=[2022-08-10 00:00:00+00:00..2022-09-10 23:56:00+00:00]
- sources: Kaggle crypto-news (allowlist) sha256=e0f551a8221f9afb, Kraken XXBTZEUR sha256=b8be3047e372fb86
- versions: python=3.14.7, pandas=3.0.5, sklearn=1.9.1
- rebuild command: python -m src.dataset --flat-threshold 0.005 --embargo-days 1 --seed 42


> DEMO 2022-only. Prices USD-sourced (krairy btc.csv); labels are return-based. Custom boundaries train->2022-05-31, val->2022-07-31. Fixtures untouched.

"""Dataset: backward join + 24h forward label + temporal split + data card.

Temporal discipline (binding): train learns weights only (2018-2022),
val chooses hyperparameters only (thr/embargo, 2023), test stays locked
until the single final report (2024+). Tuning on test is forbidden.
All leakage-sensitive code lives in this module only.
"""

import argparse
import hashlib
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import (
    EMBARGO_CHOICES,
    EMBARGO_DEFAULT,
    FLAT_THRESHOLD_CHOICES,
    FLAT_THRESHOLD_DEFAULT,
    SEED,
    TRAIN_END,
    VAL_END,
)


@dataclass(frozen=True)
class DatasetConfig:
    flat_threshold: float = FLAT_THRESHOLD_DEFAULT
    embargo_days: int = EMBARGO_DEFAULT
    split_mode: str = "temporal"
    seed: int = SEED
    train_end: str = TRAIN_END
    val_end: str = VAL_END


def _validate_config(config):
    if config.flat_threshold not in FLAT_THRESHOLD_CHOICES:
        allowed = sorted(FLAT_THRESHOLD_CHOICES)
        raise ValueError(f"invalid flat_threshold {config.flat_threshold!r}: "
                         f"allowed values are {allowed}")
    if config.embargo_days not in EMBARGO_CHOICES:
        allowed = sorted(EMBARGO_CHOICES)
        raise ValueError(f"invalid embargo_days {config.embargo_days!r}: "
                         f"allowed values are {allowed}")
    if config.split_mode != "temporal":
        raise ValueError(
            f"invalid split_mode {config.split_mode!r}: only chronological "
            "temporal split with purge+embargo is allowed (random/shuffle forbidden)"
        )
    return config


def _validate_threshold(thr):
    if thr not in FLAT_THRESHOLD_CHOICES:
        allowed = sorted(FLAT_THRESHOLD_CHOICES)
        raise ValueError(f"invalid flat_threshold {thr!r}: allowed values are {allowed}")
    return thr


def join_asof_backward(news, prices):
    """Backward asof: feature price = last candle at-or-before published_at (UTC)."""
    n = news.copy()
    p = prices.copy()
    n["published_at"] = pd.to_datetime(n["published_at"], utc=True)
    p["timestamp"] = pd.to_datetime(p["timestamp"], utc=True)
    n = n.sort_values("published_at")
    p = p.sort_values("timestamp")
    close_map = dict(zip(p["timestamp"], p["close"]))
    px = p[["timestamp", "close"]].rename(
        columns={"timestamp": "price_timestamp", "close": "close_t"})
    joined = pd.merge_asof(
        n, px.sort_values("price_timestamp"),
        left_on="published_at", right_on="price_timestamp",
        direction="backward",
    )
    fwd = (joined["price_timestamp"] + pd.Timedelta(days=1)).map(close_map)
    joined["close_t24"] = pd.to_numeric(fwd, errors="coerce")
    return joined.sort_values("published_at").reset_index(drop=True)


def label_forward(joined, flat_threshold=FLAT_THRESHOLD_DEFAULT, prices=None):
    """24h forward label: ret=close(t+24h)/close(t)-1; strict > so r==thr holds."""
    _validate_threshold(flat_threshold)
    df = joined.copy()
    if "close_t24" not in df.columns or df["close_t24"].isna().all():
        if prices is not None:
            pmap = dict(zip(
                pd.to_datetime(prices["timestamp"], utc=True),
                pd.to_numeric(prices["close"], errors="coerce"),
            ))
            df["close_t24"] = (
                pd.to_datetime(df["price_timestamp"], utc=True)
                + pd.Timedelta(days=1)
            ).map(pmap)
    df["close_t"] = pd.to_numeric(df["close_t"], errors="coerce")
    df["close_t24"] = pd.to_numeric(df["close_t24"], errors="coerce")
    df["ret_24h"] = df["close_t24"] / df["close_t"] - 1
    valid = df["close_t"].notna() & df["close_t24"].notna() & (df["close_t"] > 0)
    df = df.loc[valid].copy()
    thr = flat_threshold
    df["label"] = "hold"
    df.loc[df["ret_24h"] > thr, "label"] = "buy"
    df.loc[df["ret_24h"] < -thr, "label"] = "sell"
    df["flat_threshold"] = thr
    return df.reset_index(drop=True)


def _boundary(ts_date):
    return pd.Timestamp(ts_date, tz="UTC") + pd.Timedelta(days=1)


def temporal_split(labeled, config=None):
    """Chronological split with purge (T>B-24h) + embargo (T>B-24h-embargo)."""
    if config is None:
        config = DatasetConfig()
    _validate_config(config)
    df = labeled.copy()
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    b1 = _boundary(config.train_end)
    b2 = _boundary(config.val_end)
    purge1, purge2 = b1 - pd.Timedelta(days=1), b2 - pd.Timedelta(days=1)
    cut1 = purge1 - pd.Timedelta(days=int(config.embargo_days))
    cut2 = purge2 - pd.Timedelta(days=int(config.embargo_days))
    train_cand = df[df["published_at"] < b1]
    val_cand = df[(df["published_at"] >= b1) & (df["published_at"] < b2)]
    test = df[df["published_at"] >= b2].copy()
    train = train_cand[train_cand["published_at"] <= cut1].copy()
    val = val_cand[val_cand["published_at"] <= cut2].copy()
    purged = int((train_cand["published_at"] > purge1).sum()
                 + (val_cand["published_at"] > purge2).sum())
    embargoed = int(
        ((train_cand["published_at"] > cut1) & (train_cand["published_at"] <= purge1)).sum()
        + ((val_cand["published_at"] > cut2) & (val_cand["published_at"] <= purge2)).sum()
    )
    info = {"b1": b1, "b2": b2, "purge1": purge1, "purge2": purge2,
            "cutoff1": cut1, "cutoff2": cut2,
            "purged_train_rows": purged, "embargoed_rows": embargoed,
            "train_end": config.train_end, "val_end": config.val_end}
    return train.reset_index(drop=True), val.reset_index(drop=True), \
        test.reset_index(drop=True), info


def _sha256_df(df):
    return hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()


def build_dataset(config=None, news_df=None, prices_df=None,
                  news_report=None, price_report=None,
                  output_card="data/interim/data_card.md", run_id=None):
    """Join+label+split and write data_card.md; returns (train, val, test, card)."""
    if config is None:
        config = DatasetConfig()
    _validate_config(config)
    if news_df is None or prices_df is None:
        npath = Path("data/interim/news_clean.parquet")
        ppath = Path("data/interim/prices_clean.parquet")
        news_df = pd.read_parquet(npath)
        prices_df = pd.read_parquet(ppath)
    joined = join_asof_backward(news_df, prices_df)
    dropped_no_price = int(joined["close_t"].isna().sum())
    labeled = label_forward(joined, flat_threshold=config.flat_threshold,
                            prices=prices_df)
    dropped_no_forward = int(len(joined) - dropped_no_price - len(labeled))
    train, val, test, info = temporal_split(labeled, config)
    try:
        import sklearn  # noqa: F401
        skl = sklearn.__version__
    except Exception:
        skl = "not-installed"
    train_dist = train['label'].value_counts().to_dict() if len(train) else dict()
    val_dist = val['label'].value_counts().to_dict() if len(val) else dict()
    test_dist = test['label'].value_counts().to_dict() if len(test) else dict()
    card = (
        f"# Data Card — build dataset\n"
        f"- raw_news_rows: {(news_report or {}).get('raw_news_rows', len(news_df))}\n"
        f"- raw_price_rows: {(price_report or {}).get('raw_price_rows', len(prices_df))}\n"
        f"- kept_news_rows: {len(labeled)}\n"
        f"- dropped_missing_fields: {(news_report or {}).get('dropped_missing_fields', 0)}\n"
        f"- duplicate_rate: {(news_report or {}).get('duplicate_rate', 0.0)}\n"
        f"- dropped_no_price: {dropped_no_price}\n"
        f"- dropped_no_forward: {dropped_no_forward}\n"
        f"- purged_train_rows: {info['purged_train_rows']}\n"
        f"- embargoed_rows: {info['embargoed_rows']}\n"
        f"- news range: {pd.to_datetime(news_df['published_at'], utc=True).min()} "
        f"to {pd.to_datetime(news_df['published_at'], utc=True).max()}\n"
        f"- price range: {pd.to_datetime(prices_df['timestamp'], utc=True).min()} "
        f"to {pd.to_datetime(prices_df['timestamp'], utc=True).max()}\n"
        f"- flat_threshold: {config.flat_threshold}\n"
        f"- embargo_days: {config.embargo_days}\n"
        f"- seed: {config.seed}\n"
        f"- split boundaries: train_end={config.train_end} (B1={info['b1']}), "
        f"val_end={config.val_end} (B2={info['b2']})\n"
        f"- purge cutoffs: {info['purge1']}, {info['purge2']}; "
        f"embargo gaps: {config.embargo_days}d -> cutoffs {info['cutoff1']}, {info['cutoff2']}\n"
        f"- per-split counts: train={len(train)}, val={len(val)}, test={len(test)}\n"
        f"- per-split label distribution: train={train_dist}, val={val_dist}, test={test_dist}\n"
        f"- per-split ranges: train=[{train['published_at'].min() if len(train) else None}.."
        f"{train['published_at'].max() if len(train) else None}], "
        f"val=[{val['published_at'].min() if len(val) else None}.."
        f"{val['published_at'].max() if len(val) else None}], "
        f"test=[{test['published_at'].min() if len(test) else None}.."
        f"{test['published_at'].max() if len(test) else None}]\n"
        f"- sources: Kaggle crypto-news (allowlist) sha256={_sha256_df(news_df)[:16]}, "
        f"Kraken XXBTZEUR sha256={_sha256_df(prices_df)[:16]}\n"
        f"- versions: python={platform.python_version()}, pandas={pd.__version__}, "
        f"sklearn={skl}\n"
        f"- rebuild command: python -m src.dataset --flat-threshold {config.flat_threshold} "
        f"--embargo-days {config.embargo_days} --seed {config.seed}\n"
    )
    Path(output_card).parent.mkdir(parents=True, exist_ok=True)
    Path(output_card).write_text(card, encoding="utf-8")
    if run_id:
        dest = Path(f"models/{run_id}/data_card.md")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(card, encoding="utf-8")
    return train, val, test, card


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--flat-threshold", type=float, default=FLAT_THRESHOLD_DEFAULT)
    ap.add_argument("--embargo-days", type=int, default=EMBARGO_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    cfg = DatasetConfig(flat_threshold=args.flat_threshold,
                        embargo_days=args.embargo_days, seed=args.seed)
    train, val, test, _card = build_dataset(cfg)
    print(f"train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    sys.exit(main())

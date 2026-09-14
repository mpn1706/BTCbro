"""Kraken price ingest: bulk CSV first, then polite incremental.

Emits data/interim/prices_clean.parquet (schema design §5.2).
Pair gate: anything != XXBTZEUR aborts naming the expected pair.
"""

import time
from pathlib import Path

import pandas as pd

from src.config import PAIR as EXPECTED_PAIR

MAX_PER_CALL = 720
SLEEP_SECS = 1
OHLCV_COLS = ("open", "high", "low", "close", "volume")


def validate_pair(pair):
    if pair != EXPECTED_PAIR:
        raise ValueError(
            f"unsupported pair {pair!r}: expected {EXPECTED_PAIR!r} "
            "(slice 1 is XXBTZEUR-only)"
        )
    return pair


def clean_prices(df, pair=EXPECTED_PAIR):
    validate_pair(pair)
    raw_rows = len(df)
    out = df.copy()
    out["timestamp"] = pd.to_datetime(
        out["timestamp"], errors="coerce", utc=True
    ).dt.tz_convert("UTC").dt.normalize()
    for col in OHLCV_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    valid = (
        out["timestamp"].notna()
        & out["open"].gt(0) & out["high"].gt(0)
        & out["low"].gt(0) & out["close"].gt(0)
        & out["volume"].ge(0)
    )
    dropped = int((~valid).sum())
    out = out.loc[valid].copy()
    out["pair"] = EXPECTED_PAIR
    out = out.sort_values("timestamp").drop_duplicates(
        subset=["timestamp"], keep="last"
    ).reset_index(drop=True)
    report = {
        "raw_price_rows": raw_rows,
        "kept_price_rows": len(out),
        "dropped_invalid_rows": dropped,
        "pair": EXPECTED_PAIR,
    }
    return out, report


def load_prices_csv(path, pair=EXPECTED_PAIR):
    return clean_prices(pd.read_csv(Path(path)), pair=pair)


def incremental_update(prices, pair=EXPECTED_PAIR, fetch_fn=None,
                       sleep_fn=time.sleep,
                       max_per_call=MAX_PER_CALL,
                       sleep_secs=SLEEP_SECS):
    """Append daily candles from max(stored)+1d via paginated fetch_fn.

    fetch_fn(since: 'YYYY-MM-DD', count: int) -> DataFrame with
    timestamp/open/high/low/close/volume. Sleeps >=1s per call.
    """
    validate_pair(pair)
    if fetch_fn is None:
        raise ValueError("fetch_fn is required (no implicit network)")
    prices = prices.sort_values("timestamp").reset_index(drop=True)
    calls, new_rows = 0, 0
    while True:
        since = (prices["timestamp"].max() + pd.Timedelta(days=1)).date()
        count = min(max_per_call, MAX_PER_CALL)
        batch = fetch_fn(since=since.isoformat(), count=count)
        calls += 1
        sleep_fn(max(sleep_secs, SLEEP_SECS))
        if batch is None or len(batch) == 0:
            break
        clean, _ = clean_prices(batch, pair=pair)
        fresh = clean[~clean["timestamp"].isin(prices["timestamp"])]
        if len(fresh) == 0:
            break
        prices = pd.concat([prices, fresh], ignore_index=True)
        prices = prices.sort_values("timestamp").drop_duplicates(
            subset=["timestamp"], keep="last"
        ).reset_index(drop=True)
        new_rows += len(fresh)
        if calls >= 50:
            break
    stats = {"calls": calls, "max_per_call": MAX_PER_CALL,
             "new_rows": int(new_rows), "total_rows": len(prices)}
    return prices, stats


def main(src="data/raw/kraken_xxbtzeur_1d.csv",
         dst="data/interim/prices_clean.parquet"):
    clean, report = load_prices_csv(src)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(dst, index=False)
    print(report)


if __name__ == "__main__":
    main()

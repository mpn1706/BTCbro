"""PR1 ingest tests (RED first, strict TDD).

Covers REQ-1 (news schema/UTC/dedup/quarantine) and REQ-2
(Kraken XXBTZEUR bulk/incremental/pair gate). No network: Kraken
paginated fetch is mocked.
"""

from pathlib import Path

import pandas as pd

from src.config import PAIR
from src.ingest_kraken import clean_prices, incremental_update, validate_pair
from src.ingest_news import clean_news, load_news_csv

FIXTURES = Path(__file__).parent / "fixtures"
PRICE_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def _price_batch(rows):
    return pd.DataFrame(rows, columns=PRICE_COLS)


def _empty_batch():
    return pd.DataFrame(columns=PRICE_COLS)


def test_news_timestamp_to_utc():
    df = load_news_csv(FIXTURES / "news_tiny.csv")
    clean, report = clean_news(df)
    assert len(clean) > 0
    # All stored timestamps are tz-aware UTC.
    assert isinstance(clean["published_at"].dtype, pd.DatetimeTZDtype)
    assert str(clean["published_at"].dt.tz) == "UTC"
    # Explicit +01:00 offset normalizes to UTC (12:00+01:00 -> 11:00Z).
    rally = clean[clean["url"] == "https://example.com/etf-rally"]
    assert len(rally) == 1
    assert rally.iloc[0]["published_at"] == pd.Timestamp(
        "2024-01-16 11:00:00+00:00"
    )


def test_dedup_case_whitespace():
    df = load_news_csv(FIXTURES / "news_tiny.csv")
    clean, report = clean_news(df)
    # n2/n3 share url+title modulo case/whitespace -> one kept, earliest wins.
    urls = clean["url"].tolist()
    assert urls.count("https://example.com/etf-rally") == 1
    assert report["duplicate_count"] >= 1
    kept = clean[clean["url"] == "https://example.com/etf-rally"].iloc[0]
    assert kept["published_at"] == pd.Timestamp("2024-01-16 11:00:00+00:00")


def test_missing_field_quarantined():
    df = load_news_csv(FIXTURES / "news_tiny.csv")
    clean, report = clean_news(df)
    # n4 has empty url -> quarantined + counted, valid rows still pass.
    assert report["dropped_missing_fields"] >= 1
    assert "quarantine_count" in report
    assert report["quarantine_count"] == report["dropped_missing_fields"]
    assert clean["title"].notna().all()
    assert (clean["title"].str.strip() != "").all()
    assert clean["url"].notna().all()
    assert (clean["url"].str.strip() != "").all()
    assert len(clean) == 5  # 7 raw - 1 missing - 1 duplicate


def test_wrong_pair_fails():
    import pytest

    df = pd.read_csv(FIXTURES / "prices_tiny.csv")
    with pytest.raises(ValueError, match="XXBTZEUR"):
        clean_prices(df, pair="XXBTZUSD")
    with pytest.raises(ValueError, match="XXBTZEUR"):
        validate_pair("BTC/USD")


def test_incremental_pagination_polite():
    df = pd.read_csv(FIXTURES / "prices_tiny.csv")
    clean, _ = clean_prices(df, pair=PAIR)

    calls = []
    sleeps = []

    def fake_fetch(since, count):
        calls.append({"since": since, "count": count})
        if len(calls) > 1:
            return _empty_batch()
        return _price_batch(
            [{"timestamp": since, "open": 62100, "high": 63000,
              "low": 61500, "close": 62800, "volume": 180.0}]
        )

    def fake_sleep(secs):
        sleeps.append(secs)

    updated, stats = incremental_update(
        clean, pair=PAIR, fetch_fn=fake_fetch, sleep_fn=fake_sleep
    )
    assert len(updated) == len(clean) + 1
    assert calls, "expected at least one paginated fetch"
    assert all(c["count"] <= 720 for c in calls)
    assert stats["max_per_call"] <= 720
    assert sleeps, "expected polite sleep between calls"
    assert all(s >= 1 for s in sleeps)


def test_naive_madrid_timestamp_to_utc():
    # Naive 15:30 Madrid (CET, UTC+1 in January) -> 14:30 UTC.
    df = load_news_csv(FIXTURES / "news_tiny.csv")
    clean, _ = clean_news(df)
    row = clean[clean["id"] == "n6"].iloc[0]
    assert row["published_at"] == pd.Timestamp("2024-01-20 14:30:00+00:00")


def test_empty_source_becomes_unknown():
    df = load_news_csv(FIXTURES / "news_tiny.csv")
    clean, _ = clean_news(df)
    row = clean[clean["id"] == "n5"].iloc[0]
    assert row["source"] == "unknown"


def test_ohlcv_numeric_coercion():
    df = _price_batch(
        [{"timestamp": "2024-01-15", "open": "59500",
          "high": "60500", "low": "59000", "close": "60000",
          "volume": "123.4"}]
    )
    clean, _ = clean_prices(df, pair=PAIR)
    assert len(clean) == 1
    assert clean.iloc[0]["close"] == 60000.0
    assert str(clean["timestamp"].dt.tz) == "UTC"


def test_incremental_deoverlap_on_append():
    df = pd.read_csv(FIXTURES / "prices_tiny.csv")
    clean, _ = clean_prices(df, pair=PAIR)
    overlap_day = clean["timestamp"].max().strftime("%Y-%m-%d")

    def fake_fetch(since, count):
        # Cursor starts at max(stored)+1d; the API overlaps one stored
        # day in its response plus one genuinely new day.
        if since == (pd.Timestamp(overlap_day) + pd.Timedelta(days=1))\
                .strftime("%Y-%m-%d"):
            return _price_batch(
                [
                    {"timestamp": overlap_day, "open": 1, "high": 2,
                     "low": 1, "close": 62100, "volume": 1.0},
                    {"timestamp": since, "open": 62100,
                     "high": 63000, "low": 61500, "close": 62800,
                     "volume": 180.0},
                ]
            )
        return _empty_batch()

    updated, stats = incremental_update(
        clean, pair=PAIR, fetch_fn=fake_fetch, sleep_fn=lambda s: None
    )
    assert stats["new_rows"] == 1
    assert len(updated) == len(clean) + 1
    assert updated["timestamp"].is_unique

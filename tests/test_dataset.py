"""PR2 dataset tests (RED first, strict TDD).

Covers REQ-3 (backward join + 24h forward label) and REQ-4
(chronological split + purge + embargo) + REQ-8 rebuild.
No network, no model, no CLI.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.dataset import (
    DatasetConfig,
    build_dataset,
    join_asof_backward,
    label_forward,
    temporal_split,
)
from src.ingest_kraken import clean_prices
from src.ingest_news import clean_news, load_news_csv

FIXTURES = Path(__file__).parent / "fixtures"


def _prices(dates, closes):
    n = len(dates)
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(dates, utc=True),
            "open": list(closes),
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": list(closes),
            "volume": [100.0] * n,
            "pair": ["XXBTZEUR"] * n,
        }
    )


def _news(times, prefix="t"):
    rows = []
    for i, t in enumerate(times):
        rows.append(
            {
                "id": f"{prefix}{i}",
                "published_at": t,
                "source": "test",
                "title": f"Headline {prefix}{i}",
                "body": "body text",
                "url": f"https://example.com/{prefix}{i}",
                "lang": "en",
                "dedup_hash": f"h{prefix}{i}",
            }
        )
    df = pd.DataFrame(rows)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    return df


def test_asof_uses_last_candle_at_or_before():
    prices = _prices(
        ["2024-01-15", "2024-01-16", "2024-01-17"],
        [60000.0, 60800.0, 60330.0],
    )
    news = _news(["2024-01-16T12:00:00Z"])
    joined = join_asof_backward(news, prices)
    assert len(joined) == 1
    assert joined.iloc[0]["close_t"] == 60800.0
    assert pd.Timestamp(joined.iloc[0]["price_timestamp"]) == pd.Timestamp(
        "2024-01-16 00:00:00+00:00"
    )


def test_future_candle_never_feature():
    base = _prices(["2024-01-15", "2024-01-16"], [60000.0, 61000.0])
    news = _news(["2024-01-15T12:00:00Z"])
    j1 = join_asof_backward(news, base)
    assert j1.iloc[0]["close_t"] == 60000.0
    # Move the future candle drastically: feature must not change.
    moved = _prices(["2024-01-15", "2024-01-16"], [60000.0, 99999.0])
    j2 = join_asof_backward(news, moved)
    assert j2.iloc[0]["close_t"] == 60000.0
    assert j2.iloc[0]["close_t"] == j1.iloc[0]["close_t"]


def test_label_buy_boundary():
    # 60000 -> 60330 @0.005: r=0.0055 -> buy.
    prices = _prices(["2024-01-15", "2024-01-16"], [60000.0, 60330.0])
    news = _news(["2024-01-15T12:00:00Z"])
    labeled = label_forward(join_asof_backward(news, prices), flat_threshold=0.005)
    assert labeled.iloc[0]["label"] == "buy"
    assert labeled.iloc[0]["ret_24h"] == pytest.approx(0.0055)


def test_label_hold_inside():
    # 60000 -> 60100 @0.005: r~=0.00167 -> hold.
    prices = _prices(["2024-01-15", "2024-01-16"], [60000.0, 60100.0])
    news = _news(["2024-01-15T12:00:00Z"])
    labeled = label_forward(join_asof_backward(news, prices), flat_threshold=0.005)
    assert labeled.iloc[0]["label"] == "hold"


def test_threshold_configurable():
    # r=+0.008 -> hold @0.01, buy @0.005. Invalid thr names allowed values.
    prices = _prices(["2024-01-15", "2024-01-16"], [60000.0, 60480.0])
    news = _news(["2024-01-15T12:00:00Z"])
    joined = join_asof_backward(news, prices)
    hi = label_forward(joined, flat_threshold=0.01)
    lo = label_forward(joined, flat_threshold=0.005)
    assert hi.iloc[0]["label"] == "hold"
    assert lo.iloc[0]["label"] == "buy"
    with pytest.raises(ValueError, match="0.003.*0.005.*0.01|allowed"):
        label_forward(joined, flat_threshold=0.007)


def test_temporal_order():
    labeled = _news(
        [
            "2021-06-01T12:00:00Z",
            "2022-06-01T12:00:00Z",
            "2023-06-01T12:00:00Z",
            "2024-06-01T12:00:00Z",
        ]
    )
    labeled["label"] = "hold"
    cfg = DatasetConfig()
    train, val, test, _info = temporal_split(labeled, cfg)
    assert len(train) > 0 and len(val) > 0 and len(test) > 0
    assert train["published_at"].max() < val["published_at"].min()
    assert val["published_at"].max() < test["published_at"].min()


def test_purge_plus_embargo_math():
    # Boundary B1 = 2023-01-01 00:00Z; cutoff = B - 24h - 1d.
    cfg = DatasetConfig(embargo_days=1)
    labeled = _news(
        [
            "2022-12-29T12:00:00Z",  # kept (<= cutoff 2022-12-30 00:00)
            "2022-12-30T12:00:00Z",  # purged (> cutoff)
            "2022-12-31T12:00:00Z",  # purged (> cutoff)
            "2023-06-01T12:00:00Z",
            "2024-06-01T12:00:00Z",
        ]
    )
    labeled["label"] = "hold"
    train, _val, _test, _info = temporal_split(labeled, cfg)
    cutoff = pd.Timestamp("2022-12-30 00:00:00+00:00")
    assert (train["published_at"] <= cutoff).all()
    assert not (train["published_at"] > cutoff).any()
    with pytest.raises(ValueError, match="1.*2|allowed"):
        temporal_split(labeled, DatasetConfig(embargo_days=5))


def test_random_split_forbidden():
    labeled = _news(["2023-06-01T12:00:00Z", "2024-06-01T12:00:00Z"])
    labeled["label"] = "hold"
    with pytest.raises(ValueError, match="[Cc]hronological|temporal"):
        temporal_split(labeled, DatasetConfig(split_mode="random"))
    with pytest.raises(ValueError, match="[Cc]hronological|temporal"):
        build_dataset(
            DatasetConfig(split_mode="stratified-shuffle"),
            news_df=labeled,
            prices_df=_prices(["2023-01-01"], [60000.0]),
        )


def test_rebuild_reproducible():
    news_clean, _nr = clean_news(load_news_csv(FIXTURES / "news_tiny.csv"))
    prices_clean, _pr = clean_prices(
        pd.read_csv(FIXTURES / "prices_tiny.csv")
    )
    cfg = DatasetConfig(
        flat_threshold=0.005,
        embargo_days=1,
        train_end="2024-01-17",
        val_end="2024-01-19",
    )
    t1, v1, s1, card1 = build_dataset(cfg, news_df=news_clean, prices_df=prices_clean)
    t2, v2, s2, card2 = build_dataset(cfg, news_df=news_clean, prices_df=prices_clean)
    pd.testing.assert_frame_equal(t1.reset_index(drop=True), t2.reset_index(drop=True))
    pd.testing.assert_frame_equal(v1.reset_index(drop=True), v2.reset_index(drop=True))
    pd.testing.assert_frame_equal(s1.reset_index(drop=True), s2.reset_index(drop=True))
    # Card carries the exact contract fields (timestamp line may differ).
    for field in (
        "raw_news_rows",
        "flat_threshold",
        "embargo_days",
        "seed",
        "split boundaries",
        "rebuild command",
        "dropped_no_price",
        "dropped_no_forward",
        "purged_train_rows",
        "embargoed_rows",
    ):
        assert field in card1, f"missing card field: {field}"


def test_exact_equality_holds():
    # Math r == thr (60000 -> 60300 @0.005) must be hold via strict >.
    prices = _prices(["2024-01-15", "2024-01-16"], [60000.0, 60300.0])
    news = _news(["2024-01-15T12:00:00Z"])
    labeled = label_forward(join_asof_backward(news, prices), flat_threshold=0.005)
    assert labeled.iloc[0]["label"] == "hold"


def test_sell_mirror():
    # 60000 -> 59400 @0.005: r=-0.01 < -thr -> sell.
    prices = _prices(["2024-01-15", "2024-01-16"], [60000.0, 59400.0])
    news = _news(["2024-01-15T12:00:00Z"])
    labeled = label_forward(join_asof_backward(news, prices), flat_threshold=0.005)
    assert labeled.iloc[0]["label"] == "sell"


def test_embargo2_vs_1_delta():
    base = _news(
        [
            "2022-12-29T12:00:00Z",
            "2022-12-30T12:00:00Z",
            "2022-12-31T12:00:00Z",
            "2023-06-01T12:00:00Z",
            "2024-06-01T12:00:00Z",
        ]
    )
    base["label"] = "hold"
    t1, _v1, _s1, _i1 = temporal_split(base, DatasetConfig(embargo_days=1))
    t2, _v2, _s2, _i2 = temporal_split(base, DatasetConfig(embargo_days=2))
    assert len(t2) <= len(t1)
    assert len(t2) < len(t1)  # embargo=2 strictly removes more here


def test_news_before_first_candle_dropped():
    prices = _prices(["2024-01-16", "2024-01-17"], [60800.0, 60330.0])
    news = _news(["2024-01-15T12:00:00Z"], prefix="early")
    labeled = label_forward(join_asof_backward(news, prices), flat_threshold=0.005)
    assert len(labeled) == 0  # no prior candle -> dropped_no_price

"""Build REAL-DATA DEMO (2022-only, honest geometry) without touching fixtures.

News: data/raw/kaggle_balabaskar/bitcoin_articles.csv (2500, 2022-03->09)
Prices: data/raw/kaggle_krairy/btc.csv (daily 2010->2024-03, USD — disclosed;
labels use relative returns so currency only affects price_ctx display).
Boundaries (custom, recorded): train ->2022-05-31 / val ->2022-07-31 / test rest.
Outputs: data/raw/real_news.csv, data/raw/real_prices.csv (new files),
data/interim/real_demo_card.md. Fixtures (news.csv, kraken_xxbtzeur_1d.csv) untouched.
Usage: python tools/build_real_demo.py
"""
from pathlib import Path
import pandas as pd

from src.dataset import DatasetConfig, build_dataset

RAW_NEWS = Path("data/raw/kaggle_balabaskar/bitcoin_articles.csv")
RAW_PX = Path("data/raw/kaggle_krairy/btc.csv")
OUT_NEWS = Path("data/raw/real_news.csv")
OUT_PX = Path("data/raw/real_prices.csv")
CARD = Path("data/interim/real_demo_card.md")


def main() -> None:
    raw_n = pd.read_csv(RAW_NEWS)
    news = pd.DataFrame({
        "id": raw_n["article_id"].astype(str),
        "published_at": pd.to_datetime(raw_n["published_date"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": raw_n["media"].fillna("unknown").astype(str),
        "title": raw_n["title"].fillna("").astype(str),
        "body": raw_n["excerpt"].fillna(raw_n["summary"]).fillna("").astype(str),
        "url": raw_n["link"].fillna("").astype(str),
    })
    news = news[news["title"].str.strip() != ""].drop_duplicates(subset=["id"])
    raw_p = pd.read_csv(RAW_PX)
    px = pd.DataFrame({
        "timestamp": pd.to_datetime(raw_p["Date"], utc=True).dt.strftime("%Y-%m-%d"),
        "open": pd.to_numeric(raw_p["Open"], errors="coerce"),
        "high": pd.to_numeric(raw_p["High"], errors="coerce"),
        "low": pd.to_numeric(raw_p["Low"], errors="coerce"),
        "close": pd.to_numeric(raw_p["Close"], errors="coerce"),
        "volume": pd.to_numeric(raw_p["Volume"], errors="coerce").fillna(0),
    }).dropna(subset=["close"]).sort_values("timestamp")
    OUT_NEWS.parent.mkdir(parents=True, exist_ok=True)
    news.to_csv(OUT_NEWS, index=False)
    px.to_csv(OUT_PX, index=False)

    cfg = DatasetConfig(flat_threshold=0.005, embargo_days=1,
                        train_end="2022-05-31", val_end="2022-07-31")
    train, val, test, card = build_dataset(config=cfg, news_df=news, prices_df=px)
    for name, df in (("train", train), ("val", val), ("test", test)):
        dist = df["label"].value_counts().to_dict() if len(df) else {}
        print(f"{name}: n={len(df)} dist={dist}")
    print(f"years: {pd.to_datetime(train['published_at']).dt.year.value_counts().to_dict() if len(train) else {}} "
          f"/ val / test")
    CARD.parent.mkdir(parents=True, exist_ok=True)
    CARD.write_text(card + "\n\n> DEMO 2022-only. Prices USD-sourced (krairy btc.csv); "
                           "labels are return-based. Custom boundaries train->2022-05-31, "
                           "val->2022-07-31. Fixtures untouched.\n", encoding="utf-8")
    print(f"wrote {OUT_NEWS} ({len(news)}), {OUT_PX} ({len(px)}), card")


if __name__ == "__main__":
    main()

"""Build BIG honest split: imadallal 2021-2024 (11295, genuine ts) + krairy/live prices.

Writes data/raw/big_news.csv + data/raw/big_prices.csv (fixtures untouched),
prints thr/embargo sweep counts like the professor's capture. No training here.
Usage: PYTHONPATH=. python tools/build_big_demo.py
"""
from pathlib import Path
import pandas as pd

from src.dataset import DatasetConfig, build_dataset

OUT_N = Path("data/raw/big_news.csv")
OUT_P = Path("data/raw/big_prices.csv")


def main() -> None:
    raw = pd.read_csv("data/raw/kaggle_imad/bitcoin_sentiments_21_24.csv")
    news = pd.DataFrame({
        "id": ["imad-" + str(i) for i in range(len(raw))],
        "published_at": pd.to_datetime(raw["Date"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "imadallal-21-24",
        "title": raw["Short Description"].fillna("").astype(str).str.slice(0, 300),
        "body": "",
        "url": ["https://example.invalid/imad-" + str(i) for i in range(len(raw))],
    })
    news = news[news["title"].str.strip() != ""]
    k = pd.read_csv("data/raw/kaggle_krairy/btc.csv")
    kpx = pd.DataFrame({
        "timestamp": pd.to_datetime(k["Date"], utc=True).dt.strftime("%Y-%m-%d"),
        "open": pd.to_numeric(k["Open"], errors="coerce"),
        "high": pd.to_numeric(k["High"], errors="coerce"),
        "low": pd.to_numeric(k["Low"], errors="coerce"),
        "close": pd.to_numeric(k["Close"], errors="coerce"),
        "volume": pd.to_numeric(k["Volume"], errors="coerce").fillna(0),
    })
    live = pd.read_csv("data/raw/kraken_xxbtzeur_live.csv")
    live["timestamp"] = pd.to_datetime(live["timestamp"], utc=True).dt.strftime("%Y-%m-%d")
    # Gap-fill Mar-Sep 2024 (krairy ends 03-19, live starts 09-25, Kraken API only
    # serves last 720d): belbino safe_haven Bitcoin daily, gap dates ONLY.
    bel = pd.read_csv("data/raw/kaggle_belbino/safe_haven_prices.csv")
    bel = bel[bel["asset"] == "Bitcoin"]
    gap = pd.DataFrame({
        "timestamp": pd.to_datetime(bel["date"], utc=True).dt.strftime("%Y-%m-%d"),
        "open": pd.to_numeric(bel["open"], errors="coerce"),
        "high": pd.to_numeric(bel["high"], errors="coerce"),
        "low": pd.to_numeric(bel["low"], errors="coerce"),
        "close": pd.to_numeric(bel["close"], errors="coerce"),
        "volume": pd.to_numeric(bel["volume"], errors="coerce").fillna(0),
    })
    have = set(kpx["timestamp"]).union(live["timestamp"])
    gap = gap[~gap["timestamp"].isin(have)]
    print(f"gap-fill belbino: {len(gap)} candles "
          f"{gap['timestamp'].min()} -> {gap['timestamp'].max()}")
    px = pd.concat([kpx, live, gap], ignore_index=True
                   ).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    OUT_N.parent.mkdir(parents=True, exist_ok=True)
    news.to_csv(OUT_N, index=False)
    px.to_csv(OUT_P, index=False)
    print(f"news {len(news)} range {news['published_at'].min()} -> {news['published_at'].max()}")
    print(f"prices {len(px)} range {px['timestamp'].min()} -> {px['timestamp'].max()}")
    for thr in (0.003, 0.005, 0.01):
        for emb in (1, 2):
            cfg = DatasetConfig(flat_threshold=thr, embargo_days=emb)
            tr, va, te, _ = build_dataset(config=cfg, news_df=news, prices_df=px)
            d = lambda df: df["label"].value_counts().to_dict() if len(df) else {}
            print(f"thr={thr} emb={emb}: train={len(tr)} {d(tr)} / val={len(va)} {d(va)} / test={len(te)} {d(te)}")


if __name__ == "__main__":
    main()

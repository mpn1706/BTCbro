"""Honest MINI-DEMO: real NewsAPI 2026 (97, genuine timestamps) + live Kraken EUR.

Boundaries (recorded): train ->2026-06-10 / val ->2026-06-20 / test rest.
Fixture files untouched. Usage: PYTHONPATH=. python tools/build_mini_demo.py
"""
from pathlib import Path
import pandas as pd

from src.dataset import DatasetConfig, build_dataset

OUT_N = Path("data/raw/mini_news.csv")
OUT_P = Path("data/raw/mini_prices.csv")
CARD = Path("data/interim/mini_demo_card.md")


def main() -> None:
    raw = pd.read_csv("data/raw/kaggle_newsapi/bitcoin.csv")
    body = raw["description"].fillna(raw["content"]).fillna("")
    news = pd.DataFrame({
        "id": ["newsapi-" + str(i) for i in range(len(raw))],
        "published_at": pd.to_datetime(raw["publishedAt"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": raw["source_name"].fillna("unknown").astype(str),
        "title": raw["title"].fillna("").astype(str),
        "body": body.astype(str),
        "url": raw["url"].fillna("").astype(str),
    })
    news = news[news["title"].str.strip() != ""]
    live = pd.read_csv("data/raw/kraken_xxbtzeur_live.csv")
    OUT_N.parent.mkdir(parents=True, exist_ok=True)
    news.to_csv(OUT_N, index=False)
    live.to_csv(OUT_P, index=False)
    for thr in (0.003, 0.005, 0.01):
        cfg = DatasetConfig(flat_threshold=thr, embargo_days=1,
                            train_end="2026-06-10", val_end="2026-06-20")
        tr, va, te, _ = build_dataset(config=cfg, news_df=news, prices_df=live)
        d = lambda df: df["label"].value_counts().to_dict() if len(df) else {}
        print(f"thr={thr}: train n={len(tr)} {d(tr)} / val n={len(va)} {d(va)} / test n={len(te)} {d(te)}")
    cfg = DatasetConfig(flat_threshold=0.005, embargo_days=1,
                        train_end="2026-06-10", val_end="2026-06-20")
    tr, va, te, card = build_dataset(config=cfg, news_df=news, prices_df=live)
    CARD.parent.mkdir(parents=True, exist_ok=True)
    CARD.write_text(card + "\n\n> MINI-DEMO: NewsAPI 97 (2026-05-28->06-27, genuine ts) + "
                           "live Kraken XXBTZEUR 721d. Balabaskar discarded for labeling "
                           "(day-10 timestamp clamping). Boundaries 06-10/06-20.\n",
                    encoding="utf-8")
    print("card ->", CARD)


if __name__ == "__main__":
    main()

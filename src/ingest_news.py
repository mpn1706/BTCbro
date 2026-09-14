"""News ingest: raw Kaggle CSV -> UTC normalize + url+title dedup.

Emits data/interim/news_clean.parquet (schema design §5.1).
Naive timestamps are assumed Europe/Madrid local time then
converted to UTC; explicit offsets are honored as-is.
"""

import hashlib
import re
from pathlib import Path

import pandas as pd

NAIVE_TZ = "Europe/Madrid"
REQUIRED_FIELDS = ("title", "published_at", "url")

_ES_RE = re.compile(r"[ñ¡¿]|bitcóin|\b(el|la|los|las|una|para|con|que)\b", re.I)


def _to_utc(value, naive_tz=NAIVE_TZ):
    ts = pd.to_datetime(value, errors="coerce", utc=False)
    if pd.isna(ts):
        return pd.NaT
    if ts.tzinfo is None:
        ts = ts.tz_localize(naive_tz)
    return ts.tz_convert("UTC")


def detect_lang(title, body):
    text = f"{title or ''} {body or ''}"
    return "es" if _ES_RE.search(text) else "en"


def load_news_csv(path):
    return pd.read_csv(Path(path), dtype=str, keep_default_na=True)


def clean_news(df, naive_tz=NAIVE_TZ):
    raw_rows = len(df)
    df = df.copy()
    df["published_at"] = df.get("published_at").apply(
        lambda v: _to_utc(v, naive_tz)
    )
    missing = df["published_at"].isna()
    for col in ("title", "url"):
        vals = df.get(col)
        missing = missing | vals.isna() | (vals.astype(str).str.strip() == "")
    report_missing = int(missing.sum())
    df = df.loc[~missing].copy()

    df["source"] = df.get("source").fillna("").astype(str).str.strip()
    df.loc[df["source"] == "", "source"] = "unknown"
    df["body"] = df.get("body").fillna("").astype(str)
    df["title"] = df["title"].astype(str).str.strip()
    df["url"] = df["url"].astype(str).str.strip()
    df["lang"] = [
        detect_lang(t, b) for t, b in zip(df["title"], df["body"])
    ]
    url_norm = df["url"].str.strip().str.casefold()
    title_norm = df["title"].str.strip().str.casefold()
    df["dedup_hash"] = [
        hashlib.sha1(f"{u}\x00{t}".encode("utf-8")).hexdigest()
        for u, t in zip(url_norm, title_norm)
    ]
    if "id" not in df.columns:
        df["id"] = df["dedup_hash"].str[:12]
    else:
        df["id"] = df["id"].fillna("").astype(str).str.strip()
        blank = df["id"] == ""
        df.loc[blank, "id"] = df.loc[blank, "dedup_hash"].str[:12]
    df = df.sort_values("published_at")
    before = len(df)
    df = df.drop_duplicates(subset=["dedup_hash"], keep="first")
    dupes = before - len(df)
    clean = df[
        ["id", "published_at", "source", "title", "body", "url",
         "lang", "dedup_hash"]
    ].sort_values("published_at").reset_index(drop=True)
    report = {
        "raw_news_rows": raw_rows,
        "kept_news_rows": len(clean),
        "dropped_missing_fields": report_missing,
        "quarantine_count": report_missing,
        "duplicate_count": int(dupes),
        "duplicate_rate": (dupes / before) if before else 0.0,
        "lang_distribution": clean["lang"].value_counts().to_dict(),
    }
    return clean, report


def main(src="data/raw/news.csv", dst="data/interim/news_clean.parquet"):
    clean, report = clean_news(load_news_csv(src))
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(dst, index=False)
    print(report)


if __name__ == "__main__":
    main()

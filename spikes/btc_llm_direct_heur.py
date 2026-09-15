"""THROWAWAY SPIKE — step 0 of btc-llm-direct. NOT shipped code, no tests.

Keyword-sentiment heuristic as a stand-in for the future LLM backend.
Purpose: validate the score -> threshold -> label -> per-class F1 wiring
with zero installs, on fixtures. Baseline (src/, tests/) untouched.
Interpretation: wiring check only. n is tiny; any F1 here proves nothing
about signal, only that the harness measures honestly.
"""

import time

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from src.dataset import build_dataset, join_asof_backward, label_forward
from src.ingest_kraken import clean_prices, load_prices_csv
from src.ingest_news import clean_news, load_news_csv

BULL = {
    "record", "rally", "rallies", "surge", "surges", "inflows", "inflow",
    "optimism", "upgrade", "bull", "soars", "jumps", "gains", "etf",
    "sue", "sube", "fuerza", "récord", "record",
}
BEAR = {
    "crash", "crashes", "plunge", "hack", "ban", "bans", "lawsuit",
    "sells", "selloff", "drops", "falls", "fear", "cae", "caída",
}


def heur_score(title: str, body: str = "") -> float:
    toks = (f"{title} {body}").lower().split()
    if not toks:
        return 0.0
    pos = sum(1 for t in toks if t.strip(".,!?()\"'") in BULL)
    neg = sum(1 for t in toks if t.strip(".,!?()\"'") in BEAR)
    return (pos - neg) / max(1, pos + neg)


def predict(score: float, thr: float) -> str:
    if score > thr:
        return "buy"
    if score < -thr:
        return "sell"
    return "hold"


def main() -> None:
    t0 = time.time()
    # Faithful path: raw CSVs -> real ingest cleaners -> join -> label.
    news, _news_rep = clean_news(load_news_csv("tests/fixtures/news_tiny.csv"))
    prices, _px_rep = load_prices_csv("tests/fixtures/prices_tiny.csv")
    joined = join_asof_backward(news, prices)
    labeled = label_forward(joined, flat_threshold=0.005, prices=prices)
    pool = labeled.copy()
    pool["score"] = [
        heur_score(str(t), str(b)) for t, b in zip(pool["title"], pool.get("body", ""))
    ]
    print(f"labeled rows: {len(pool)}, label dist: {pool['label'].value_counts().to_dict()}")
    for thr in (0.0, 0.25, 0.5):
        preds = [predict(s, thr) for s in pool["score"]]
        macro = f1_score(pool["label"], preds, average="macro", zero_division=0)
        cm = confusion_matrix(pool["label"], preds, labels=["buy", "hold", "sell"])
        print(f"thr={thr}: macro-F1={macro:.3f} preds={pd.Series(preds).value_counts().to_dict()}")
        print(f"  3x3 rows=true cols=pred [buy,hold,sell]:\n{cm}")
    print(f"wall: {time.time() - t0:.1f}s (wiring only, n={len(pool)})")


if __name__ == "__main__":
    main()

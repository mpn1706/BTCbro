"""Exporta web/data.json determinista (fijo -> mismo bytes siempre).

test congelado + baseline + preds LLM (cache) + closes -> backtest testeado.
Uso: PYTHONPATH=. python tools/export_web.py
"""
import json
import os
from pathlib import Path
import pandas as pd

from src.backtest import daily_vote, equity, hodl
from src.compare import build_compare_report
from src.evaluate import evaluate_predictions
from src.dataset import DatasetConfig, build_dataset
from src.llm_backend import PROMPT_VERSION, predict_llm
from src.train import texts_from_df, train_baseline

OUT = Path("web/data.json")
CACHE = Path("data/interim/llm_cache_big.jsonl")
STEP = int(os.environ.get("STEP", "4"))


def r6(x: float) -> float:
    return round(float(x), 6)


def main() -> None:
    news = pd.read_csv("data/raw/big_news.csv")
    px = pd.read_csv("data/raw/big_prices.csv")
    cfg = DatasetConfig(flat_threshold=0.005, embargo_days=1)
    tr, _, te, _ = build_dataset(config=cfg, news_df=news, prices_df=px)
    te = te.reset_index(drop=True)
    te = te.iloc[::STEP].reset_index(drop=True)

    vec, model = train_baseline(texts_from_df(tr), tr["label"].tolist())
    base_pred = [str(a) for a in model.predict(vec.transform(texts_from_df(te)))]
    llm_outs = [predict_llm(str(r["title"]), str(r.get("body", "")),
                            float(r["close_t"]), cache_path=CACHE)
                for _, r in te.iterrows()]
    llm_pred = [o["action"] for o in llm_outs]
    oks = sum(1 for o in llm_outs if o["parse_ok"])
    walls = [o["wall_s"] for o in llm_outs]
    llm_meta = {"parse_ok": f"{oks}/{len(llm_outs)}",
                "mean_wall_s": round(sum(walls) / len(walls), 1) if walls else 0}

    pxd = px.assign(d=pd.to_datetime(px["timestamp"], utc=True).dt.strftime("%Y-%m-%d"))
    pxd = pxd.drop_duplicates("d").set_index("d")
    days = sorted(pd.to_datetime(te["price_timestamp"], utc=True).dt.strftime("%Y-%m-%d").unique())
    ohlc = [{"d": d, "o": r6(pxd.loc[d]["open"]), "h": r6(pxd.loc[d]["high"]),
             "l": r6(pxd.loc[d]["low"]), "c": r6(pxd.loc[d]["close"])}
            for d in days if d in pxd.index]
    closes = [c["c"] for c in ohlc]
    day_of = [pd.to_datetime(t, utc=True).strftime("%Y-%m-%d") for t in te["published_at"]]
    votes = {"baseline": daily_vote(list(zip(day_of, base_pred)), days),
             "llm": daily_vote(list(zip(day_of, llm_pred)), days)}
    curves = {b: [r6(v) for v in equity(votes[b], closes, 1.0)["curve"]] for b in votes}
    curves["HODL"] = [r6(v) for v in hodl(closes, 1.0)["curve"]]
    y_test = te["label"].tolist()
    per_class = {}
    for name, preds in (("baseline", base_pred), ("llm", llm_pred)):
        rep = evaluate_predictions(y_test, preds)
        per_class[name] = {k: round(v["f1"], 3) for k, v in rep["per_class"].items()}
    comp = build_compare_report(te["label"].tolist(), base_pred, llm_pred,
                                split_hash="big-imadallal-thr0005-emb1",
                                sampling=f"systematic_step_{STEP} (n={len(te)})",
                                out_dir="data/interim/big_compare",
                                prompt_version=PROMPT_VERSION,
                                llm_meta=llm_meta)
    prec = [{"t": str(r["title"])[:200], "y": str(r["label"]),
               "r": round(float(r["ret_24h"]), 4),
               "b": b, "l": l}
              for r, b, l in zip(te.to_dict("records"), base_pred, llm_pred)]
    data = {"precedents": prec,
            "meta": {"window": f"{days[0]} → {days[-1]}", "n_test": len(te),
                     "prompt_version": PROMPT_VERSION,
                     "ft_status": "pendiente (requiere GPU)",
                     "f1": {"baseline": comp["baseline"]["macro_f1"],
                            "llm": comp["llm"]["macro_f1"]},
                     "per_class": per_class,
                     "sampling": comp["sampling"]},
            "days": days, "ohlc": ohlc, "votes": votes, "curves_1_0": curves}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
    h, l, b = curves["HODL"][-1], curves["baseline"][-1], curves["llm"][-1]
    print(f"HODL {h:.2f} > baseline {b:.2f} > LLM {l:.2f} (capital 1.0)" if h >= b >= l
          else f"HODL {h:.2f} / baseline {b:.2f} / LLM {l:.2f} (capital 1.0)")
    print("wrote", OUT)


if __name__ == "__main__":
    main()

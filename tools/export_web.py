"""Exporta web/data.json determinista (fijo -> mismo bytes siempre).

test congelado + baseline + preds LLM (cache) + closes -> backtest testeado.
Uso: PYTHONPATH=. python tools/export_web.py
"""
import csv
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
LORA_PREDS = Path("cloud/predictions_lora_test.csv")
LORA_TEST = Path("cloud/lora_test.jsonl")
STEP = int(os.environ.get("STEP", "4"))


def r6(x: float) -> float:
    return round(float(x), 6)


def main() -> None:
    news = pd.read_csv("data/raw/big_news.csv")
    px = pd.read_csv("data/raw/big_prices.csv")
    cfg = DatasetConfig(flat_threshold=0.005, embargo_days=1)
    tr, _, te, _ = build_dataset(config=cfg, news_df=news, prices_df=px)
    te_full = te.reset_index(drop=True)
    day_by_id = dict(zip(te_full["id"], pd.to_datetime(te_full["published_at"], utc=True).dt.strftime("%Y-%m-%d")))
    te = te_full.iloc[::STEP].reset_index(drop=True)

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
    emb_pred = pd.read_csv("data/interim/embeddings/pred_web334.csv").iloc[:, 0].tolist()
    assert len(emb_pred) == len(te), "emb preds must align to web subsample"
    # LoRA corre en el complemento disjunto del web-subsample (1000 + 334 = 1334).
    # Se alinea por dia a la misma ventana: dias sin noticia LoRA = hold (contrato daily_vote).
    lora_rows = list(csv.DictReader(LORA_PREDS.open(encoding="utf-8")))
    assert len(lora_rows) == 1000, f"lora preds incompletas: {len(lora_rows)}"
    lora_map = {r["id"]: r["action"].strip().lower() for r in lora_rows}
    assert set(lora_map) <= set(day_by_id), "lora ids fuera del split"
    lora_test = [json.loads(l) for l in LORA_TEST.open(encoding="utf-8")]
    lora_y = [r["label"] for r in lora_test]
    lora_p = [lora_map[r["id"]] for r in lora_test]
    votes = {"baseline": daily_vote(list(zip(day_of, base_pred)), days),
             "llm": daily_vote(list(zip(day_of, llm_pred)), days),
             "emb": daily_vote(list(zip(day_of, emb_pred)), days),
             "lora": daily_vote([(day_by_id[i], lora_map[i]) for i in lora_map], days)}
    curves = {b: [r6(v) for v in equity(votes[b], closes, 1.0)["curve"]] for b in votes}
    curves["HODL"] = [r6(v) for v in hodl(closes, 1.0)["curve"]]
    y_test = te["label"].tolist()
    per_class = {}
    for name, preds in (("baseline", base_pred), ("llm", llm_pred)):
        rep = evaluate_predictions(y_test, preds)
        per_class[name] = {k: round(v["f1"], 3) for k, v in rep["per_class"].items()}
    emb_rep = evaluate_predictions(y_test, emb_pred)
    lora_rep = evaluate_predictions(lora_y, lora_p)
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
                     "ft_status": "LoRA completado en GPU (T4): F1 0.197 en muestra disjunta n=1000, 4.º de 4 (D11).",
                     "f1": {"baseline": comp["baseline"]["macro_f1"],
                            "llm": comp["llm"]["macro_f1"],
                            "emb": round(emb_rep["macro_f1"], 3),
                            "lora": round(lora_rep["macro_f1"], 3)},
                     "n": {"baseline": len(te), "llm": len(te),
                           "emb": len(te), "lora": len(lora_test)},
                     "per_class": {**per_class,
                                      "emb": {k: round(v["f1"], 3) for k, v in emb_rep["per_class"].items()},
                                      "lora": {k: round(v["f1"], 3) for k, v in lora_rep["per_class"].items()}},
                     "sampling": comp["sampling"],
                     "lora_sampling": "disjoint complement of web subsample (n=1000); date-aligned daily votes"},
            "days": days, "ohlc": ohlc, "votes": votes, "curves_1_0": curves}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
    h, l, b, e, o = (curves[k][-1] for k in ("HODL", "llm", "baseline", "emb", "lora"))
    print(f"HODL {h:.2f} / baseline {b:.2f} / LLM {l:.2f} / EMB {e:.2f} / LORA {o:.2f} (capital 1.0)")
    print("wrote", OUT)


if __name__ == "__main__":
    main()

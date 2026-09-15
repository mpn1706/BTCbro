"""DUELO v2 vs v3 en VALID (nunca test). Regla pre-registrada: v3 promociona
solo si macro-F1(v3) > macro-F1(v2) en la MISMA submuestra de valid.
Si promociona, la evaluación final será en filas del test NUNCA leídas
(complemento fuera del STEP=4), en una corrida aparte. Test actual ni se mira.
Usage: STEP_V=20 PYTHONPATH=. python tools/run_prompt_duel.py (background + log)
"""
import os
import time
from pathlib import Path
import pandas as pd

from src.dataset import DatasetConfig, build_dataset
from src.evaluate import evaluate_predictions
from src.llm_backend import predict_llm

STEP_V = int(os.environ.get("STEP_V", "20"))
CACHE = Path("data/interim/llm_cache_big.jsonl")
OUT = Path("data/interim/duel.json")


def run_version(val: pd.DataFrame, version: str) -> tuple[list, dict]:
    preds, oks, walls = [], 0, []
    t0 = time.time()
    for i, r in val.iterrows():
        out = predict_llm(str(r["title"]), str(r.get("body", "")), float(r["close_t"]),
                          cache_path=CACHE, prompt_version=version)
        preds.append(out["action"])
        oks += out["parse_ok"]
        walls.append(out["wall_s"])
        n = len(preds)
        if n % 25 == 0 or n == len(val):
            print(f"[{version} {n}/{len(val)}] ok={oks}/{n}", flush=True)
    meta = {"parse_ok": f"{oks}/{len(val)}",
            "mean_wall_s": round(sum(walls) / len(walls), 1) if walls else 0}
    return preds, meta


def main() -> None:
    news = pd.read_csv("data/raw/big_news.csv")
    px = pd.read_csv("data/raw/big_prices.csv")
    cfg = DatasetConfig(flat_threshold=0.005, embargo_days=1)
    _, va, _, _ = build_dataset(config=cfg, news_df=news, prices_df=px)
    va = va.reset_index(drop=True).iloc[::STEP_V].reset_index(drop=True)
    print(f"duel on val: n={len(va)} (STEP_V={STEP_V})", flush=True)
    res = {}
    for v in ("v2", "v3"):
        preds, meta = run_version(va, v)
        rep = evaluate_predictions(va["label"].tolist(), preds)
        res[v] = {"macro_f1": round(rep["macro_f1"], 3), **meta}
        print(f"{v}: macro-F1={res[v]['macro_f1']}", flush=True)
    verdict = "PROMOTE v3" if res["v3"]["macro_f1"] > res["v2"]["macro_f1"] else "HOLD v2"
    import json
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rule": "promote iff F1(v3) > F1(v2) on same val subsample",
                               "n_val": len(va), **res, "verdict": verdict}, indent=1))
    print("VERDICT:", verdict, flush=True)


if __name__ == "__main__":
    main()

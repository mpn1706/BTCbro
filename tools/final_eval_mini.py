"""FINAL EVAL mini-demo — espejo del final_eval.py del profe, a nuestra escala.

El profe corrigio el bug clave: evaluar las preds del LLM contra test_sub
(la submuestra predicha), no contra el test completo. Aca el test es n=24 y
ya corrio ENTERO (183s), asi que no hace falta submuestrear; igual aplicamos
sus tres garantias: assert de conteo, assert de alineacion por posicion y
baseline en las MISMAS filas (manzana con manzana), todo en el compare.json.
Usage: PYTHONPATH=. python tools/final_eval_mini.py
"""
import json
import time
from pathlib import Path

import pandas as pd

from src.dataset import DatasetConfig, build_dataset
from src.evaluate import check_report_complete, evaluate_predictions
from src.llm_backend import PROMPT_VERSION, cache_key, predict_llm
from src.train import texts_from_df, train_baseline

SAMPLING = "full (n=24, no subsample needed at this scale)"
OUT = Path("data/interim/mini_compare_final.json")


def main() -> None:
    news = pd.read_csv("data/raw/mini_news.csv")
    px = pd.read_csv("data/raw/mini_prices.csv")
    cfg = DatasetConfig(flat_threshold=0.005, embargo_days=1,
                        train_end="2026-06-10", val_end="2026-06-20")
    tr, _, te, _ = build_dataset(config=cfg, news_df=news, prices_df=px)
    te = te.reset_index(drop=True)

    # Baseline en las mismas filas del test (manzana con manzana).
    vec, model = train_baseline(texts_from_df(tr), tr["label"].tolist())
    base_pred = list(model.predict(vec.transform(texts_from_df(te))))
    assert len(base_pred) == len(te), f"baseline {len(base_pred)} vs test {len(te)}"
    base_rep = evaluate_predictions(te["label"].tolist(), base_pred)
    check_report_complete(base_rep)

    # LLM desde cache (sin re-inferir): asserts de conteo + alineacion.
    llm_pred, walls, oks = [], [], 0
    for _, r in te.iterrows():
        key = cache_key(PROMPT_VERSION, str(r["title"]), str(r.get("body", "")), float(r["close_t"]))
        out = predict_llm(str(r["title"]), str(r.get("body", "")),
                          float(r["close_t"]), cache_path="data/interim/llm_cache.jsonl")
        assert cache_key(PROMPT_VERSION, str(r["title"]), str(r.get("body", "")), float(r["close_t"])) == key
        llm_pred.append(out["action"])
        oks += out["parse_ok"]
        walls.append(out["wall_s"])
    assert len(llm_pred) == len(te), f"cache {len(llm_pred)} vs test {len(te)}"
    llm_rep = evaluate_predictions(te["label"].tolist(), llm_pred)
    check_report_complete(llm_rep)

    comp = {
        "sampling": SAMPLING,
        "prompt_version": PROMPT_VERSION,
        "model": "Qwen3-1.7B-Q4_K_M",
        "n_test": len(te),
        "baseline": {"macro_f1": round(base_rep["macro_f1"], 3),
                     "accuracy": round(base_rep["accuracy"], 3),
                     "cm_3x3_buy_hold_sell": base_rep["confusion_matrix"]},
        "llm": {"macro_f1": round(llm_rep["macro_f1"], 3),
                "accuracy": round(llm_rep["accuracy"], 3),
                "cm_3x3_buy_hold_sell": llm_rep["confusion_matrix"],
                "parse_ok": f"{oks}/{len(te)}",
                "mean_wall_s": round(sum(walls) / len(walls), 1) if walls else 0},
        "verdict": ("LLM supera a baseline en macro-F1" if llm_rep["macro_f1"] > base_rep["macro_f1"]
                    else "baseline >= LLM en macro-F1") + " (ambos near-random, PASS honesto)",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(comp, indent=1), encoding="utf-8")
    print(json.dumps(comp, indent=1))


if __name__ == "__main__":
    main()

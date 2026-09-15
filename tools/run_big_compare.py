"""BIG honest compare: imadallal 2021-2024 + krairy/live, thr 0.005 emb 1.

Baseline trains on 5375 (seconds); LLM runs test 165 (~8s each, ~22 min) with
per-item log + cache resume (checkpoint = llm_cache_big.jsonl). Professor parity:
same rows both backends, sampling documented. Safe to re-run (resumes).
Usage: PYTHONPATH=. python tools/run_big_compare.py   (or nohup … & for background)
"""
import os
import time
from pathlib import Path
import pandas as pd

STEP = int(os.environ.get("STEP", "1"))  # systematic 1-in-STEP (professor parity: 4)

from src.compare import build_compare_report
from src.dataset import DatasetConfig, build_dataset
from src.llm_backend import PROMPT_VERSION, predict_llm
from src.train import texts_from_df, train_baseline

CACHE = Path("data/interim/llm_cache_big.jsonl")
OUTDIR = Path("data/interim/big_compare")
LOG_EVERY = 25


def main() -> None:
    news = pd.read_csv("data/raw/big_news.csv")
    px = pd.read_csv("data/raw/big_prices.csv")
    cfg = DatasetConfig(flat_threshold=0.005, embargo_days=1)
    tr, va, te, _ = build_dataset(config=cfg, news_df=news, prices_df=px)
    te = te.reset_index(drop=True)
    if STEP > 1:
        te = te.iloc[::STEP].reset_index(drop=True)
    print(f"split: train={len(tr)} val={len(va)} test_sub={len(te)} (STEP={STEP})", flush=True)

    vec, model = train_baseline(texts_from_df(tr), tr["label"].tolist())
    base_pred = list(model.predict(vec.transform(texts_from_df(te))))

    llm_pred, oks, walls = [], 0, []
    t0 = time.time()
    for i, r in te.iterrows():
        out = predict_llm(str(r["title"]), str(r.get("body", "")),
                          float(r["close_t"]), cache_path=CACHE)
        llm_pred.append(out["action"])
        oks += out["parse_ok"]
        walls.append(out["wall_s"])
        n = i + 1
        if n % LOG_EVERY == 0 or n == len(te):
            el = time.time() - t0
            rate = n / el if el else 0
            eta = (len(te) - n) / rate / 60 if rate else -1
            print(f"[{n}/{len(te)}] rate={rate:.2f}/s eta={eta:.0f}min parse_ok={oks}/{n}", flush=True)

    meta = {"parse_ok": f"{oks}/{len(te)}",
            "mean_wall_s": round(sum(walls) / len(walls), 1) if walls else 0}
    comp = build_compare_report(te["label"].tolist(), base_pred, llm_pred,
                                split_hash="big-imadallal-thr0005-emb1",
                                sampling=f"systematic_step_{STEP} (professor parity)",
                                out_dir=OUTDIR, prompt_version=PROMPT_VERSION, llm_meta=meta)
    print(f"BASELINE macro-F1={comp['baseline']['macro_f1']} | "
          f"LLM macro-F1={comp['llm']['macro_f1']} | {comp['verdict']}", flush=True)


if __name__ == "__main__":
    main()

"""Extrae embeddings (server residente) para train/val/test-complemento.

El complemento del test (filas fuera del STEP=4 ya evaluado) sigue virgen para
la evaluación final del extractor. Reanudable por caché .npz.
Usage: PYTHONPATH=. python tools/run_embed_extract.py (background + log)
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd

from src.dataset import DatasetConfig, build_dataset
from src.llm_embed import embed_texts
from src.train import texts_from_df

OUT = Path("data/interim/embeddings")
STEP_TEST_USED = 4


def dump(name: str, X: np.ndarray, y: list) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(str(OUT / f"X_{name}.npy"), X)
    pd.Series(y).to_csv(str(OUT / f"y_{name}.csv"), index=False)
    print(f"saved {name}: {X.shape}", flush=True)


def main() -> None:
    news = pd.read_csv("data/raw/big_news.csv")
    px = pd.read_csv("data/raw/big_prices.csv")
    cfg = DatasetConfig(flat_threshold=0.005, embargo_days=1)
    tr, va, te, _ = build_dataset(config=cfg, news_df=news, prices_df=px)
    te = te.reset_index(drop=True)
    used = set(te.iloc[::STEP_TEST_USED].index)
    te_new = te.loc[~te.index.isin(used)].reset_index(drop=True)
    print(f"train={len(tr)} val={len(va)} test_complement={len(te_new)}", flush=True)
    t0 = time.time()
    for name, df in (("train", tr), ("val", va), ("test_new", te_new)):
        X = embed_texts(list(texts_from_df(df)))
        dump(name, X, df["label"].tolist())
    print(f"done {(time.time()-t0)/60:.0f}min", flush=True)


if __name__ == "__main__":
    main()

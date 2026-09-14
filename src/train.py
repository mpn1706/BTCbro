"""Train baseline: TF-IDF + balanced LogisticRegression (CPU-only).

Temporal discipline (binding): fit vectorizer + model on TRAIN split
only (2018-2022). Val (2023) chooses hyperparameters (thr/embargo/
TF-IDF/C) by macro-F1; test (2024+) stays locked until the single
final report. Tuning on test is forbidden.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.config import SEED

MIN_FEATURES = 5000
MAX_FEATURES = 20000


def combine_text(title, body=""):
    """Combine title + body for TF-IDF; body optional, price is NOT a feature."""
    title = (title or "").strip()
    body = (body or "").strip()
    return f"{title} {body}".strip() if body else title


def texts_from_df(df):
    """Build TF-IDF input strings from a labeled split (title + body)."""
    titles = df["title"].fillna("") if "title" in df else [""] * len(df)
    bodies = df["body"].fillna("") if "body" in df else [""] * len(df)
    return [combine_text(str(t), str(b)) for t, b in zip(titles, bodies)]


def make_vectorizer(max_features=5000):
    """TF-IDF(title+body, max_features 5k-20k, ngram (1,2), sublinear)."""
    if not MIN_FEATURES <= int(max_features) <= MAX_FEATURES:
        raise ValueError(
            f"invalid max_features {max_features!r}: "
            f"choose within [{MIN_FEATURES}, {MAX_FEATURES}]"
        )
    return TfidfVectorizer(
        max_features=int(max_features),
        ngram_range=(1, 2),
        sublinear_tf=True,
    )


def train_baseline(texts, labels, max_features=5000, C=1.0, seed=SEED):
    """Fit TF-IDF + balanced LogReg on train texts only. Returns (vec, model)."""
    vec = make_vectorizer(max_features=max_features)
    X = vec.fit_transform(list(texts))
    model = LogisticRegression(
        class_weight="balanced",
        random_state=int(seed),
        max_iter=1000,
        solver="lbfgs",
        C=float(C),
    )
    model.fit(X, list(labels))
    return vec, model


def make_run_id(seed=SEED, flat_threshold=0.005, embargo_days=1):
    """Run id: YYYYMMDD-HHMMSS-seed<N>-thr<X>-emb<Y>."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-seed{seed}-thr{flat_threshold}-emb{embargo_days}"


def save_run(vectorizer, model, config, metrics, run_id, base="models"):
    """Write models/<run_id>/{vectorizer.pkl, model.pkl, config.json,
    metrics.json} + refresh models/latest pointer (text file with run_id)."""
    run_dir = Path(base) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, run_dir / "vectorizer.pkl")
    joblib.dump(model, run_dir / "model.pkl")
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (Path(base) / "latest").write_text(run_id, encoding="utf-8")
    return run_dir

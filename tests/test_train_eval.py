"""PR3 train/eval tests (RED first, strict TDD).

Covers REQ-5 (TF-IDF + balanced LogReg, seed-fixed, CPU) and REQ-6
(per-class P/R + macro-F1 + 3x3 matrix; accuracy-alone rejected).
No network, no CLI. Temporal discipline: train learns / val chooses /
test locked (vectorizer fit on train only).
"""

import numpy as np
import pytest

from src.config import SEED
from src.evaluate import check_report_complete, evaluate_predictions
from src.train import combine_text, make_vectorizer, train_baseline

BUY = "Bitcoin ETF inflows hit record rally surge"
SELL = "Bitcoin crash plunge dump losses fear"
HOLD = "Bitcoin price steady flat market update"


def _toy_texts_labels():
    texts = (
        [BUY + f" breakout {i}" for i in range(4)]
        + [SELL + f" downturn {i}" for i in range(4)]
        + [HOLD + f" mixed {i}" for i in range(12)]
    )
    labels = ["buy"] * 4 + ["sell"] * 4 + ["hold"] * 12
    return texts, labels


def test_balanced_weights_set():
    texts, labels = _toy_texts_labels()
    vec, model = train_baseline(texts, labels)
    params = model.get_params()
    assert params["class_weight"] == "balanced"
    assert params["random_state"] == SEED
    assert params["max_iter"] == 1000
    assert params["solver"] == "lbfgs"
    # Hold-dominated input must still expose all three classes.
    assert set(model.classes_) == {"buy", "hold", "sell"}


def test_tfidf_params_in_contract():
    vec = make_vectorizer(max_features=5000)
    assert vec.ngram_range == (1, 2)
    assert vec.sublinear_tf is True
    assert 5000 <= vec.max_features <= 20000
    with pytest.raises(ValueError, match="max_features"):
        make_vectorizer(max_features=500)


def test_seed_reproducible():
    texts, labels = _toy_texts_labels()
    _, m1 = train_baseline(texts, labels, seed=SEED)
    _, m2 = train_baseline(texts, labels, seed=SEED)
    probe = [BUY, SELL, HOLD]
    assert list(m1.predict(_vec_transform(probe, texts, labels))) == list(
        m2.predict(_vec_transform(probe, texts, labels))
    )
    np.testing.assert_allclose(m1.coef_, m2.coef_)


def _vec_transform(probe, texts, labels):
    vec, _ = train_baseline(texts, labels, seed=SEED)
    return vec.transform(probe)


def test_train_fits_train_only():
    train_texts = [BUY + " trainonly", HOLD + " common"]
    vec, _ = train_baseline(train_texts, ["buy", "hold"])
    assert "valonlytokenxyz" not in (vec.vocabulary_ or {})
    assert "common" in (vec.vocabulary_ or {})


def test_combine_text_title_body():
    assert "hello" in combine_text("hello", "world")
    assert "world" in combine_text("hello", "world")
    assert combine_text("hello", "") == "hello"


def test_eval_requires_per_class():
    y_true = ["buy", "buy", "hold", "sell", "hold", "sell"]
    y_pred = ["buy", "hold", "hold", "sell", "sell", "hold"]
    report = evaluate_predictions(y_true, y_pred)
    for label in ("buy", "hold", "sell"):
        assert label in report["per_class"]
        assert "precision" in report["per_class"][label]
        assert "recall" in report["per_class"][label]
    assert "macro_f1" in report
    assert "confusion_matrix" in report
    assert report["labels"] == ["buy", "hold", "sell"]
    matrix = report["confusion_matrix"]
    assert len(matrix) == 3 and all(len(row) == 3 for row in matrix)
    assert sum(sum(row) for row in matrix) == len(y_true)
    # Complete report passes the gate.
    check_report_complete(report)


def test_accuracy_only_rejected():
    with pytest.raises(ValueError, match="accuracy.*incomplete|per-class|confusion"):
        check_report_complete({"accuracy": 0.9})


def test_hold_dominance_eval():
    # 90% hold: per-class entries must still exist, no crash.
    y_true = ["hold"] * 18 + ["buy", "sell"]
    y_pred = ["hold"] * 20
    report = evaluate_predictions(y_true, y_pred)
    for label in ("buy", "hold", "sell"):
        assert label in report["per_class"]
    assert len(report["confusion_matrix"]) == 3
    check_report_complete(report)


def test_single_class_val_fold():
    y_true = ["hold"] * 6
    y_pred = ["hold"] * 6
    report = evaluate_predictions(y_true, y_pred)
    for label in ("buy", "hold", "sell"):
        assert label in report["per_class"]
    assert report["macro_f1"] == pytest.approx(1 / 3, abs=0.05)
    check_report_complete(report)

"""Honest per-class evaluation (REQ-6).

Mandatory: per-class precision/recall (+F1), macro-F1, 3x3 confusion
matrix (rows=true, cols=pred, order buy/hold/sell). Accuracy alone is
incomplete and fails the slice gate. Success rule: near-random +
honest temporal protocol = PASS; high score via leakage = FAIL.
"""

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

LABELS = ["buy", "hold", "sell"]


def evaluate_predictions(y_true, y_pred, labels=None):
    """Score predictions; never crashes on hold-dominance/single-class folds."""
    labels = list(labels or LABELS)
    y_true, y_pred = list(y_true), list(y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }
    macro_f1 = float(sum(f1) / len(f1)) if len(f1) else 0.0
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {
        "labels": labels,
        "per_class": per_class,
        "macro_f1": macro_f1,
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "confusion_matrix": matrix,
    }


def check_report_complete(report):
    """Gate: accuracy-alone reports are incomplete and fail."""
    if (
        not isinstance(report, dict)
        or "per_class" not in report
        or "confusion_matrix" not in report
        or "macro_f1" not in report
    ):
        raise ValueError(
            "incomplete report: accuracy alone is insufficient; "
            "per-class precision/recall + macro-F1 + confusion matrix required"
        )
    if set(report["per_class"]) != set(LABELS):
        raise ValueError(
            f"incomplete report: per-class entries must cover {LABELS}"
        )
    matrix = report["confusion_matrix"]
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError("incomplete report: confusion matrix must be 3x3")
    return True

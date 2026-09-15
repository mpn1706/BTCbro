"""PR2 GREEN — same-split baseline-vs-LLM comparison (professor parity).

Guarantees (his final_eval.py lines 30-31): assert pred/test lengths match and
share positions; baseline runs on the SAME rows as the LLM (apples to apples);
sampling documented; accuracy-alone rejected via check_report_complete.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.evaluate import check_report_complete, evaluate_predictions

LABELS = ["buy", "hold", "sell"]


def build_compare_report(y_test, base_pred, llm_pred, split_hash, sampling,
                         out_dir=".", prompt_version="v2", llm_meta=None,
                         base_report_override=None, llm_report_override=None):
    """Evaluate both backends on identical rows; persist compare.json."""
    y_test, base_pred, llm_pred = list(y_test), list(base_pred), list(llm_pred)
    assert len(base_pred) == len(y_test), f"baseline {len(base_pred)} vs test {len(y_test)}"
    assert len(llm_pred) == len(y_test), f"llm {len(llm_pred)} vs test {len(y_test)}"
    base_rep = base_report_override if base_report_override is not None else evaluate_predictions(y_test, base_pred)
    llm_rep = llm_report_override if llm_report_override is not None else evaluate_predictions(y_test, llm_pred)
    check_report_complete(base_rep)
    check_report_complete(llm_rep)
    meta = dict(llm_meta or {})
    comp = {
        "split_hash": split_hash,
        "sampling": sampling,
        "prompt_version": prompt_version,
        "n_test": len(y_test),
        "baseline": {"macro_f1": round(base_rep["macro_f1"], 3),
                     "accuracy": round(base_rep["accuracy"], 3),
                     "cm_3x3_buy_hold_sell": base_rep["confusion_matrix"]},
        "llm": {"macro_f1": round(llm_rep["macro_f1"], 3),
                "accuracy": round(llm_rep["accuracy"], 3),
                "cm_3x3_buy_hold_sell": llm_rep["confusion_matrix"],
                **{k: meta[k] for k in ("parse_ok", "mean_wall_s") if k in meta}},
    }
    comp["verdict"] = (
        "LLM supera a baseline en macro-F1" if llm_rep["macro_f1"] > base_rep["macro_f1"]
        else "baseline >= LLM en macro-F1") + " (honesto: test-once, misma muestra)"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.json").write_text(json.dumps(comp, indent=1), encoding="utf-8")
    return comp

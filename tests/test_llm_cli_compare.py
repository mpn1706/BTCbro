"""PR2 RED — CLI --backend + compare (mocked LLM, no GGUF, no network)."""

import json
from unittest.mock import patch, MagicMock

import pytest

from src.compare import build_compare_report
from src.cli import build_parser


def _rep(macro, acc=0.4):
    return {"labels": ["buy", "hold", "sell"],
            "per_class": {lab: {"precision": 0.3, "recall": 0.3, "f1": macro, "support": 5}
                          for lab in ("buy", "hold", "sell")},
            "macro_f1": macro, "accuracy": acc,
            "confusion_matrix": [[1, 1, 1], [1, 1, 1], [1, 1, 1]]}


def test_compare_same_split_asserts_lengths():
    with pytest.raises(AssertionError):
        build_compare_report(y_test=[1, 2], base_pred=[1], llm_pred=[1, 2],
                             split_hash="h", sampling="full")


def test_compare_report_shape_and_sampling_documented(tmp_path):
    out = build_compare_report(y_test=["buy", "sell"], base_pred=["buy", "sell"],
                               llm_pred=["hold", "sell"], split_hash="abc123",
                               sampling="systematic_step_4", out_dir=tmp_path,
                               prompt_version="v2", llm_meta={"parse_ok": "2/2", "mean_wall_s": 7.0})
    assert out["sampling"] == "systematic_step_4"
    assert out["baseline"]["macro_f1"] >= 0 and out["llm"]["macro_f1"] >= 0
    assert "cm_3x3_buy_hold_sell" in out["baseline"] and "cm_3x3_buy_hold_sell" in out["llm"]
    assert (tmp_path / "compare.json").is_file()


def test_compare_rejects_accuracy_only():
    with pytest.raises(ValueError, match="incomplete"):
        build_compare_report(y_test=["buy"], base_pred=["buy"], llm_pred=["buy"],
                             split_hash="h", sampling="full",
                             base_report_override={"accuracy": 0.9})


def test_cli_has_backend_flag_default_baseline():
    args = build_parser().parse_args(["--title", "x", "--price", "60000"])
    assert args.backend == "baseline"


def test_cli_llm_missing_model_exits_3(capsys):
    from src.cli import main
    with patch("src.cli.predict_llm", side_effect=FileNotFoundError("LLM model missing: models/llm/x.gguf")):
        code = main(["--title", "x", "--price", "60000", "--backend", "llm"])
    assert code == 3
    assert "models/llm/x.gguf" in capsys.readouterr().err


def test_cli_llm_happy_path_shape(capsys):
    from src.cli import main
    fake = {"action": "buy", "conf": 1.0, "reason_short": "strong demand",
            "parse_ok": True, "timeout": False, "wall_s": 7.0, "prompt_version": "v2"}
    with patch("src.cli.predict_llm", return_value=fake):
        code = main(["--title", "t", "--price", "60000", "--backend", "llm"])
    assert code == 0
    out = capsys.readouterr().out
    assert "action: buy" in out and "DISCLAIMER" in out and "parse_ok" in out


def test_cli_bad_backend_exits_2(capsys):
    from src.cli import main
    assert main(["--title", "x", "--price", "60000", "--backend", "gpt"]) == 2

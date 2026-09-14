"""PR4 CLI tests (RED first, strict TDD).

REQ-7: btc-signal --title --price outputs action + probs + disclaimer,
exit 2 on bad input, exit 3 on missing artifact, never retrains.
"""

import hashlib
from pathlib import Path

import joblib

from src.cli import main

BUY = "Bitcoin ETF inflows hit record rally surge"
SELL = "Bitcoin crash plunge dump losses fear"
HOLD = "Bitcoin price steady flat market update"


def _make_model_dir(tmp_path):
    from src.train import train_baseline

    texts = (
        [BUY + f" breakout {i}" for i in range(4)]
        + [SELL + f" downturn {i}" for i in range(4)]
        + [HOLD + f" mixed {i}" for i in range(8)]
    )
    labels = ["buy"] * 4 + ["sell"] * 4 + ["hold"] * 8
    vec, model = train_baseline(texts, labels)
    d = tmp_path / "run"
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, d / "vectorizer.pkl")
    joblib.dump(model, d / "model.pkl")
    return d


def _hash_tree(root):
    out = {}
    for p in Path(root).rglob("*"):
        if p.is_file():
            out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def test_cli_happy_path(tmp_path, capsys):
    model_dir = _make_model_dir(tmp_path)
    code = main(["--title", "Bitcoin ETF inflows hit record", "--price", "60000",
                 "--model", str(model_dir)])
    assert code == 0
    out = capsys.readouterr().out
    assert "action:" in out
    assert "buy" in out or "hold" in out or "sell" in out
    for label in ("buy", "hold", "sell"):
        assert label in out
    assert "DISCLAIMER" in out
    assert "not investment advice" in out


def test_cli_missing_title_exit2(tmp_path, capsys):
    model_dir = _make_model_dir(tmp_path)
    code = main(["--price", "60000", "--model", str(model_dir)])
    assert code == 2
    err = capsys.readouterr().err
    assert "title" in err.lower()


def test_cli_invalid_price_exit2(tmp_path, capsys):
    model_dir = _make_model_dir(tmp_path)
    code = main(["--title", "hello", "--price", "abc", "--model", str(model_dir)])
    assert code == 2
    err = capsys.readouterr().err
    assert "price" in err.lower()


def test_cli_missing_artifact_exit3(tmp_path, capsys):
    code = main(["--title", "hello", "--price", "60000",
                 "--model", str(tmp_path / "no-such-model")])
    assert code == 3
    err = capsys.readouterr().err
    assert "no-such-model" in err


def test_cli_never_retrains(tmp_path, capsys):
    model_dir = _make_model_dir(tmp_path)
    before_mtime = {str(p): p.stat().st_mtime for p in model_dir.rglob("*") if p.is_file()}
    before_hash = _hash_tree(model_dir)
    code = main(["--title", "Bitcoin ETF inflows hit record", "--price", "60000",
                 "--model", str(model_dir)])
    assert code == 0
    capsys.readouterr()
    after_hash = _hash_tree(model_dir)
    assert before_hash == after_hash
    for p, mtime in before_mtime.items():
        assert Path(p).stat().st_mtime == mtime


def test_cli_bad_prices_exit2(tmp_path, capsys):
    model_dir = _make_model_dir(tmp_path)
    for bad in ("abc", "0", "-5", ""):
        code = main(["--title", "hello", "--price", bad,
                     "--model", str(model_dir)])
        assert code == 2
        err = capsys.readouterr().err
        assert "price" in err.lower()


def test_cli_corrupt_model_exit3(tmp_path, capsys):
    bad = tmp_path / "badrun"
    bad.mkdir()
    (bad / "vectorizer.pkl").write_bytes(b"not-a-pickle")
    (bad / "model.pkl").write_bytes(b"not-a-pickle")
    code = main(["--title", "hello", "--price", "60000",
                 "--model", str(bad)])
    assert code == 3
    assert str(bad) in capsys.readouterr().err


def test_cli_probs_sum_to_one(tmp_path, capsys):
    import re

    model_dir = _make_model_dir(tmp_path)
    assert main(["--title", "Bitcoin ETF inflows hit record",
                 "--price", "60000", "--model", str(model_dir)]) == 0
    out = capsys.readouterr().out
    vals = [float(v) for v in re.findall(r"(?:buy|hold|sell)=(\d+\.\d+)", out)]
    assert len(vals) == 3
    assert abs(sum(vals) - 1.0) < 1e-6


def test_cli_json_shape(tmp_path, capsys):
    import json

    model_dir = _make_model_dir(tmp_path)
    code = main(["--title", "Bitcoin ETF inflows hit record",
                 "--price", "60000", "--model", str(model_dir), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["action"] in ("buy", "hold", "sell")
    assert set(payload["probs"]) == {"buy", "hold", "sell"}
    assert abs(sum(payload["probs"].values()) - 1.0) < 1e-6
    assert "disclaimer" in payload


def test_cli_help_notes_price_context():
    from src.cli import build_parser

    assert "display context" in build_parser().format_help()


def test_cli_latest_pointer_resolves(tmp_path, capsys):
    import shutil

    run = _make_model_dir(tmp_path)
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    dest = models / "demo-seed42-thr0.005-emb1"
    shutil.move(str(run), str(dest))
    (models / "latest").write_text("demo-seed42-thr0.005-emb1", encoding="utf-8")
    code = main(["--title", "Bitcoin ETF inflows hit record", "--price", "60000",
                 "--model", str(models / "latest")])
    assert code == 0
    assert "action:" in capsys.readouterr().out

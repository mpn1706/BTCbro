"""WEB-BACKTEST RED — matemática testeada en Python (el JS solo dibuja)."""

import pytest

from src.backtest import daily_vote, equity, hodl


def test_daily_vote_last_wins_and_hold_default():
    rows = [("2024-01-02", "buy"), ("2024-01-02", "sell"), ("2024-01-03", "hold")]
    assert daily_vote(rows, ["2024-01-02", "2024-01-03", "2024-01-04"]) == ["sell", "hold", "hold"]


def test_equity_hold_keeps_cash():
    assert equity(["hold", "hold"], [100.0, 110.0], 1000.0)["final"] == 1000.0


def test_equity_buy_sell_math():
    r = equity(["buy", "hold", "sell"], [100.0, 200.0, 50.0], 1000.0)
    assert r["final"] == 500.0 and len(r["curve"]) == 3


def test_hodl_reference():
    assert hodl([100.0, 50.0], 1000.0)["final"] == 500.0


def test_equity_ignores_case():
    assert equity(["BUY", "SELL"], [100.0, 200.0], 1000.0)["final"] == 2000.0


def test_export_has_precedents():
    import json
    d = json.load(open("web/data.json", encoding="utf-8"))
    assert len(d["precedents"]) == d["meta"]["n_test"] == 334
    assert set(d["precedents"][0]) == {"t", "y", "r", "b", "l"}


def test_equity_defaults_reproduce_legacy_exact():
    sigs = ["buy", "hold", "sell"]
    closes = [100.0, 200.0, 50.0]
    legacy = equity(sigs, closes, 1000.0)
    explicit = equity(sigs, closes, 1000.0, 0.0, 1.0)
    assert explicit == legacy
    assert explicit["final"] == 500.0
    assert explicit["curve"] == [1000.0, 2000.0, 500.0]


def test_equity_fractional_buy_math():
    r = equity(["buy", "hold"], [100.0, 200.0], 1000.0, 0.0, 0.5)
    assert r["curve"] == [1000.0, 1500.0]
    assert r["final"] == 1500.0


def test_equity_fractional_sell_stairstep():
    r = equity(["sell", "hold"], [100.0, 200.0], 0.0, 1.0, 0.5)
    assert r["curve"] == [100.0, 150.0]
    assert r["final"] == 150.0


def test_equity_validation_errors():
    with pytest.raises(ValueError):
        equity(["hold"], [100.0], 1000.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        equity(["hold"], [100.0], 1000.0, 0.0, 1.5)
    with pytest.raises(ValueError):
        equity(["hold"], [100.0], -1.0)
    with pytest.raises(ValueError):
        equity(["hold"], [100.0], 1000.0, -0.1)
    with pytest.raises(ValueError):
        equity(["hold"], [0.0], 1000.0)
    with pytest.raises(ValueError):
        equity(["hold"], [-5.0], 1000.0)


def test_equity_hold_keeps_btc_initial():
    r = equity(["hold", "hold"], [100.0, 200.0], 500.0, 1.0, 1.0)
    assert r["curve"] == [600.0, 700.0]
    assert r["final"] == 700.0

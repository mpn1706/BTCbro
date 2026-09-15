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

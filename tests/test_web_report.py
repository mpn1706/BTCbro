"""WEB RED — simulación parametrizable + reporte HTML (sin servidor)."""

import json

from tools.build_web_report import daily_signals, simulate


def test_simulate_all_hold_keeps_cash():
    closes = [100.0, 110.0, 90.0]
    eq = simulate(["hold", "hold", "hold"], closes, 1000.0)
    assert eq["final"] == 1000.0 and eq["curve"] == [1000.0, 1000.0, 1000.0]


def test_simulate_buy_then_sell():
    closes = [100.0, 200.0, 50.0]
    eq = simulate(["buy", "hold", "sell"], closes, 1000.0)
    # 1000/100=10 BTC -> hold -> sell 10*50=500
    assert eq["final"] == 500.0


def test_simulate_buy_and_hodl():
    closes = [100.0, 200.0]
    eq = simulate(["buy", "hold"], closes, 1000.0)
    assert eq["final"] == 2000.0


def test_hodl_reference():
    from tools.build_web_report import hodl
    assert hodl([100.0, 50.0], 1000.0)["final"] == 500.0


def test_daily_signals_last_of_day_wins():
    rows = [("2024-01-02", "buy"), ("2024-01-02", "sell"), ("2024-01-03", "hold")]
    assert daily_signals(rows) == {"2024-01-02": "sell", "2024-01-03": "hold"}


def test_report_embeds_parametrizable_amount_and_ft_pending(tmp_path):
    from tools.build_web_report import build_report
    from pathlib import Path
    p = build_report(
        candles=[{"d": "2024-01-02", "o": 1, "h": 2, "l": 0.5, "c": 1.5}],
        series={"baseline": {"2024-01-02": "buy"}, "llm": {"2024-01-02": "hold"}},
        out=tmp_path / "r.html", amount_default=1000.0)
    html = Path(p).read_text(encoding="utf-8")
    assert "amount" in html and "HODL" in html and "lora" in html.lower()
    assert "pendiente" in html.lower() or "pending" in html.lower()


def test_index_has_zoom_cards_f1_and_disclaimer():
    from pathlib import Path
    html = Path("web/index.html").read_text(encoding="utf-8")
    for kw in ("wheel", "dblclick", "kpis", 'id="f1"', "data.json",
               "ft (LoRA)", "completado", "asesor"):
        assert kw in html, f"index no trae: {kw}"


def test_index_has_side_by_side_tables_and_precedents():
    from pathlib import Path
    html = Path("web/index.html").read_text(encoding="utf-8")
    assert "grid-template-columns:1fr 1fr" in html
    assert "Precedentes" in html and "precSearch" in html and "btc-signal" in html


def test_index_covers_emb_series():
    from pathlib import Path
    import json
    html = Path("web/index.html").read_text(encoding="utf-8")
    assert 'id="tE"' in html
    assert 'id="tO"' in html
    d = json.load(open("web/data.json", encoding="utf-8"))
    assert set(d["votes"]) == {"baseline", "llm", "emb", "lora"}
    assert d["meta"]["f1"]["emb"] > d["meta"]["f1"]["llm"]
    assert d["meta"]["n"]["lora"] == 1000
    assert "disjoint" in d["meta"]["lora_sampling"]

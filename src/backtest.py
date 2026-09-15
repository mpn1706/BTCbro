"""Backtest educativo (matemática testeada; el JS de la web solo dibuja).

Reglas all-in sin fees: BUY gasta toda la caja al cierre, SELL liquida todo al
cierre, HOLD nada. HODL compra el primer día y mantiene. No es asesoría.
"""

from __future__ import annotations

HOLD = "hold"


def daily_vote(rows: list[tuple[str, str]], days: list[str]) -> list[str]:
    """Una señal por día (última gana); días sin noticia = hold."""
    last = {}
    for day, action in rows:
        last[day] = (action or HOLD).lower()
    return [last.get(d, HOLD) for d in days]


def equity(signals: list[str], closes: list[float], amount: float) -> dict:
    """Curva de capital normalizada a `amount`."""
    cash, btc, curve = float(amount), 0.0, []
    for sig, px in zip(signals, closes):
        sig = (sig or HOLD).lower()
        if sig == "buy" and cash > 0:
            btc, cash = cash / px, 0.0
        elif sig == "sell" and btc > 0:
            cash, btc = btc * px, 0.0
        curve.append(cash + btc * px)
    return {"final": curve[-1] if curve else float(amount), "curve": curve}


def hodl(closes: list[float], amount: float) -> dict:
    """Testigo comprar-y-mantener."""
    if not closes:
        return {"final": float(amount), "curve": []}
    n = float(amount) / closes[0]
    return {"final": n * closes[-1], "curve": [n * px for px in closes]}

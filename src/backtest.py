"""Educational backtest (math tested; web JS only draws).

Fractional rules without fees: BUY spends cash*trade_pct at close,
SELL sells btc*trade_pct at close, HOLD does nothing. HODL buys day one
and holds. Not financial advice.
"""

from __future__ import annotations

HOLD = "hold"


def daily_vote(rows: list[tuple[str, str]], days: list[str]) -> list[str]:
    """One signal per day (last wins); days without news = hold."""
    last = {}
    for day, action in rows:
        last[day] = (action or HOLD).lower()
    return [last.get(d, HOLD) for d in days]


def equity(
    signals: list[str],
    closes: list[float],
    amount: float,
    initial_btc: float = 0.0,
    trade_pct: float = 1.0,
) -> dict:
    """Absolute capital curve for custom initial capital and fractional size.

    `amount` is the initial cash in EUR. The initial total is
    `amount + initial_btc * closes[0]` (or `amount` when empty).
    Defaults (0.0, 1.0) reproduce the legacy all-in behavior exactly.
    """
    # Fail fast on invalid inputs.
    try:
        cash = float(amount)
    except (TypeError, ValueError):
        raise ValueError("amount must be a number >= 0")
    try:
        btc = float(initial_btc)
    except (TypeError, ValueError):
        raise ValueError("initial_btc must be a number >= 0")
    try:
        pct = float(trade_pct)
    except (TypeError, ValueError):
        raise ValueError("trade_pct must be in (0, 1]")
    if not 0.0 < pct <= 1.0:
        raise ValueError("trade_pct must be in (0, 1]")
    if cash < 0.0:
        raise ValueError("amount must be >= 0")
    if btc < 0.0:
        raise ValueError("initial_btc must be >= 0")
    prices: list[float] = []
    for c in closes:
        try:
            px = float(c)
        except (TypeError, ValueError):
            raise ValueError("closes must contain numbers > 0")
        if px <= 0.0:
            raise ValueError("closes must contain numbers > 0")
        prices.append(px)
    if not prices:
        return {"final": float(cash), "curve": []}
    curve: list[float] = []
    for sig, px in zip(signals, prices):
        action = (sig or HOLD).lower()
        if action == "buy" and cash > 0.0:
            spend = cash * pct
            btc += spend / px
            cash *= 1.0 - pct
        elif action == "sell" and btc > 0.0:
            cash += btc * pct * px
            btc *= 1.0 - pct
        # HOLD / none / unknown signals keep positions unchanged.
        curve.append(cash + btc * px)
    return {"final": curve[-1] if curve else float(cash), "curve": curve}


def hodl(closes: list[float], amount: float) -> dict:
    """Testigo comprar-y-mantener."""
    if not closes:
        return {"final": float(amount), "curve": []}
    n = float(amount) / closes[0]
    return {"final": n * closes[-1], "curve": [n * px for px in closes]}

"""THROWAWAY SPIKE — btc-llm-direct parity with professor track. NOT shipped code.
Format: ACTION: BUY|SELL|HOLD + REASON (plain text, regex parse, no grammar).
Greedy (temp 0) + thinking off (--reasoning off). Baseline (src/, tests/) untouched.
50 synthetic news from bull/bear/flat templates in fixture style.
Runner uses CPU-basica tuning: --n-gpu-layers 0 -c 1024 -t 4 -st --simple-io.
Usage:
  python spikes/llm_direct.py --list            # show 50 titles
  python spikes/llm_direct.py --run 5           # smoke 5 via llama-cli (~35s)
  python spikes/llm_direct.py --run 50          # full spike (~6min on 8GB box)
"""
from __future__ import annotations

import re
import subprocess
import sys
import time

ACTION_RE = re.compile(r"ACTION\s*:\s*(BUY|SELL|HOLD)\b", re.IGNORECASE)
REASON_RE = re.compile(r"REASON\s*[:\-]\s*(.+)", re.IGNORECASE)

SYSTEM = (
    "You are an educational BTC news classifier. "
    "Output EXACTLY two lines:\n"
    "ACTION: BUY or SELL or HOLD\n"
    "REASON: <=12 words\n"
    "BUY = clearly bullish for next 24h; SELL = clearly bearish; else HOLD. "
    "No other text. Educational exercise, not financial advice."
)

BULL_T = [
    ("Bitcoin ETF inflows hit record", "ETF bodies attract record flows."),
    ("Bitcoin price rallies on ETF optimism", "Second day rally body with strong volume."),
    ("Institutional demand surges for Bitcoin", "Funds report record allocations this week."),
    ("Bitcoin breaks resistance on strong volume", "Momentum traders join the breakout move."),
    ("Analysts upgrade Bitcoin outlook", "Upgrades cite adoption and ETF demand."),
]
BEAR_T = [
    ("Exchange hack sparks selloff fears", "Users withdraw funds after breach report."),
    ("Regulator bans crypto ads", "Ban raises fear of tighter rules ahead."),
    ("Bitcoin plunges on lawsuit news", "Lawsuit triggers broad market drop."),
    ("Whale sells spark crash fears", "Large transfer to exchange spooks market."),
    ("Bitcoin drops as fear spreads", "Fear index spikes on negative headlines."),
]
FLAT_T = [
    ("Bitcoin holds steady above 60000", "Holds body text with narrow range."),
    ("Ethereum upgrade announced", "Eth body text, unrelated to BTC flow."),
    ("El precio de Bitcoin sube con fuerza", "Texto en español sobre bitcoin."),
    ("Market waits for macro data", "Traders sit out ahead of CPI release."),
    ("Bitcoin trades flat into weekend", "Low volume, no clear catalyst."),
]


def build_50() -> list[dict]:
    items: list[dict] = []
    # 10 BUY / 20 SELL / 20 HOLD like professor distribution (template round-robin)
    for i in range(10):
        t, b = BULL_T[i % len(BULL_T)]
        items.append({"id": f"syn-bull-{i+1}", "title": t, "body": b, "expect": "BUY"})
    for i in range(20):
        t, b = BEAR_T[i % len(BEAR_T)]
        items.append({"id": f"syn-bear-{i+1}", "title": t, "body": b, "expect": "SELL"})
    for i in range(20):
        t, b = FLAT_T[i % len(FLAT_T)]
        items.append({"id": f"syn-flat-{i+1}", "title": t, "body": b, "expect": "HOLD"})
    return items


def _generation_window(text: str) -> str:
    """Recorta la ventana generada: despues del prompt ecoado, antes de '[ Prompt'.
    Evita que el parser coma 'Loading model...' o el banner."""
    txt = text or ""
    # Fin: antes de timings
    end = txt.find("[ Prompt")
    has_end = end != -1
    if has_end:
        txt = txt[:end]
    # Inicio: despues de la ultima linea que contiene el prompt ecoado
    lines = txt.splitlines()
    start = 0
    found = False
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("> ") or "Respond with ACTION" in s:
            start = i + 1
            found = True
    if not found and not has_end:
        return ""
    return "\n".join(lines[start:])


def parse_output(text: str) -> tuple[str | None, str]:
    win = _generation_window(text)
    m = ACTION_RE.search(win)
    if m:
        action = m.group(1).upper()
    else:
        # Fallback tolerante estilo profe: primer BUY/SELL/HOLD suelto en la
        # ventana generada (el modelo a veces omite el prefijo ACTION:).
        action = None
        for line in win.splitlines()[:8]:
            mm = re.search(r"^\s*(BUY|SELL|HOLD)\b", line.strip(), re.IGNORECASE)
            if mm:
                action = mm.group(1).upper()
                break
        if not action:
            mm = re.search(r"\b(BUY|SELL|HOLD)\b", win, re.IGNORECASE)
            action = mm.group(1).upper() if mm else None
    r = REASON_RE.search(win)
    if r:
        reason = r.group(1).strip()[:120]
    else:
        # Fallback: primera linea no vacia de la ventana que no sea la accion
        reason = ""
        for ln in [ln.strip() for ln in win.splitlines() if ln.strip()]:
            if action and ln.upper().strip() == action:
                continue
            if action and ln.upper().startswith(action) and len(ln) < 25:
                # "BUY" + resto corto -> el resto es la reason sin prefijo
                rest = ln[len(action):].strip(" :-")
                if rest:
                    reason = rest[:120]
                    break
                continue
            reason = ln[:120]
            break
    return action, reason


def run_one(title: str, body: str, price: int = 60000, timeout_s: int = 120) -> tuple[str, float]:
    prompt = (
        f"TITLE: {title}\nBODY: {body}\nPRICE_AT_PUBLISH_EUR: {price}\n"
        "Respond with ACTION and REASON lines only."
    )
    cmd = [
        "tools/llama.cpp/bin/llama-cli",
        "-m", "models/llm/Qwen3-1.7B-Q4_K_M.gguf",
        "--n-gpu-layers", "0", "-c", "1024", "-t", "4",
        "--temp", "0", "--seed", "42", "-n", "60",
        "--reasoning", "off", "--no-display-prompt", "--simple-io", "-st",
        "-sys", SYSTEM, "-p", prompt,
    ]
    t0 = time.time()
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, input="",
        )
        txt = (out.stdout or "") + (out.stderr or "")
    except subprocess.TimeoutExpired as e:
        txt = str(e.stdout or "") + " TIMEOUT"
    return txt, time.time() - t0


def main() -> None:
    args = sys.argv[1:]
    items = build_50()
    if "--list" in args or not args:
        for it in items:
            print(f"{it['id']}: {it['title']} [{it['expect']}]")
        print(f"total={len(items)}")
        return
    n = 5
    if "--run" in args:
        try:
            n = int(args[args.index("--run") + 1])
        except (IndexError, ValueError):
            n = 5
    n = max(1, min(n, len(items)))
    ok = 0
    t_all = time.time()
    for it in items[:n]:
        txt, dt = run_one(it["title"], it["body"])
        action, reason = parse_output(txt)
        status = "OK" if action else "BROKEN"
        if action:
            ok += 1
        print(f"{it['id']} [{it['expect']}] -> {action or '??'} | {reason[:60]} | {dt:.1f}s {status}")
    print(f"parsed {ok}/{n}, wall {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()

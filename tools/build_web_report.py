"""WEB GREEN — reporte HTML estatico offline: velas + marcadores + simulador.

Sin servidor ni dependencias (SVG + JS vanilla). Los datos van embebidos como
JSON; el monto es parametrizable en el navegador. El 3er backend (llm+lora)
aparece como pendiente (requiere GPU): arquitectura lista, sin numeros
inventados. Uso: build_report(candles, series, out, amount_default).
"""
from __future__ import annotations

import json
from pathlib import Path

# Canonical math lives in src/backtest (tested there too); these are thin
# aliases so the two modules can never drift apart.
from src.backtest import equity as simulate  # noqa: F401
from src.backtest import hodl  # noqa: F401

HOLD = "hold"


def daily_signals(rows: list[tuple[str, str]]) -> dict[str, str]:
    """Collapse (date, action) rows to one signal per day: last wins."""
    out: dict[str, str] = {}
    for day, action in rows:
        out[day] = (action or HOLD).lower()
    return out


_CSS = ("body{font-family:system-ui,sans-serif;max-width:1000px;margin:auto;padding:16px}"
        ".up{stroke:#16a34a}.dn{stroke:#dc2626}.mk{font-size:10px;font-weight:bold}"
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px}"
        ".banner{background:#fef9c3;padding:8px;border:1px solid #eab308}")


def build_report(candles: list[dict], series: dict[str, dict[str, str]],
                 out, amount_default: float = 1000.0):
    """Render report.html. candles: [{d,o,h,l,c}]; series: {backend: {day: action}}."""
    days = [c["d"] for c in candles]
    closes = [float(c["c"]) for c in candles]
    W, H, P = 940, 380, 40
    lo = min(float(c["l"]) for c in candles)
    hi = max(float(c["h"]) for c in candles)
    span = (hi - lo) or 1.0

    def X(i: int) -> float:
        return P + i * (W - 2 * P) / max(1, len(candles) - 1)

    def Y(v: float) -> float:
        return H - P - (float(v) - lo) / span * (H - 2 * P)

    parts = []
    n = len(candles)
    for i, c in enumerate(candles):
        x = X(i)
        col = "up" if float(c["c"]) >= float(c["o"]) else "dn"
        parts.append(f'<line x1="{x:.1f}" y1="{Y(c["h"]):.1f}" x2="{x:.1f}" '
                     f'y2="{Y(c["l"]):.1f}" class="{col}" stroke-width="1.5"/>')
        w = max(2.0, (W - 2 * P) / max(1, n) * 0.5)
        y1, y2 = Y(c["o"]), Y(c["c"])
        parts.append(f'<rect x="{x - w / 2:.1f}" y="{min(y1, y2):.1f}" width="{w:.1f}" '
                     f'height="{max(1.5, abs(y2 - y1)):.1f}" fill="{"#16a34a" if col == "up" else "#dc2626"}"/>')
    colors = {"baseline": "#2563eb", "llm": "#9333ea"}
    for b, sig in series.items():
        col = colors.get(b, "#666")
        for i, c in enumerate(candles):
            a = (sig.get(c["d"], HOLD)).upper()
            if a == "HOLD":
                continue
            y = Y(c["h"]) - 12 if a == "SELL" else Y(c["l"]) + 16
            parts.append(f'<text x="{X(i):.1f}" y="{y:.1f}" text-anchor="middle" '
                         f'class="mk" fill="{col}" data-b="{b}">{a[0]}</text>')
    svg = (f'<svg viewBox="0 0 {W} {H}" width="100%">{"".join(parts)}'
           f'<text x="{P}" y="16">Test window ({days[0]} → {days[-1]}) '
           f'· B=buy S=sell · hold sin marca</text></svg>')

    payload = json.dumps({"days": days, "closes": closes,
                          "signals": series, "amount_default": amount_default})
    js = """<script>
const D=%s;
function sim(sig,closes,amt){let cash=amt,btc=0;const curve=[];
for(let i=0;i<closes.length;i++){const s=(sig[i]||'hold');
if(s==='buy'&&cash>0){btc=cash/closes[i];cash=0;}
else if(s==='sell'&&btc>0){cash=btc*closes[i];btc=0;}
curve.push(cash+btc*closes[i]);}return curve;}
function run(){const amt=parseFloat(document.getElementById('amount').value)||D.amount_default;
const rows=D.days.map((d,i)=>({d,c:D.closes[i]}));
let html='<tr><th>método</th><th>final (EUR)</th><th>retorno</th></tr>';
const curves={};
for(const b of Object.keys(D.signals)){const s=D.days.map(d=>(D.signals[b][d]||'hold'));
const cv=sim(s,D.closes,amt);curves[b]=cv;
const f=cv[cv.length-1];html+=`<tr><td>${b}</td><td>${f.toFixed(2)}</td><td>${((f/amt-1)*100).toFixed(1)}%%</td></tr>`;}
const h=D.closes.map((c,i)=>amt/D.closes[0]*c);curves['HODL']=h;
html+=`<tr><td>HODL</td><td>${h[h.length-1].toFixed(2)}</td><td>${((h[h.length-1]/amt-1)*100).toFixed(1)}%%</td></tr>`;
document.getElementById('simtab').innerHTML=html;}
function toggle(b,cb){document.querySelectorAll(`[data-b="${b}"]`).forEach(e=>e.style.display=cb.checked?'':'none');}
window.onload=run;
</script>""" % payload
    checks = "".join(
        f'<label><input type="checkbox" checked onchange="toggle(\'{b}\',this)">{b}</label> '
        for b in series)
    html = (f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{_CSS}</style></head><body>"
            f"<h1>Comparativa baseline vs LLM (test honesto)</h1>"
            f"<div class='banner'>llm+lora: <b>pendiente</b> — requiere GPU para el fine-tune; "
            f"la arquitectura ya prevé la 3ª serie. Sin números inventados.</div>"
            f"<p>{checks}</p>{svg}"
            f"<h2>Simulación (educativa, sin fees, all-in)</h2>"
            f"<label>Monto EUR: <input id='amount' type='number' value='{amount_default}' "
            f"oninput='run()'></label><table id='simtab'></table>"
            f"<p>Reglas: BUY compra con toda la caja al cierre · SELL liquida todo al cierre · "
            f"HOLD nada · HODL compra el primer día y mantiene. No es asesoría financiera.</p>"
            f"{js}</body></html>")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return str(out)

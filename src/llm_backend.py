"""PR1 GREEN — LLM backend: CPU-only llama-cli + tolerant parser + cache + abstain.

Slice 1 (src/dataset.py, baseline) frozen. This module is a side predictor only.
No network, no auto-download. Real GGUF never touched in unit tests (mocked).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

PROMPT_VERSION = "v2"
MODEL_PATH = Path("models/llm/Qwen3-1.7B-Q4_K_M.gguf")
DEFAULT_CACHE = Path("data/interim/llm_cache.jsonl")
TIMEOUT_S = 120

SYSTEM_DEFAULT = (
    "You are an educational BTC news classifier. Output EXACTLY two lines:\n"
    "ACTION: BUY or SELL or HOLD\n"
    "REASON: <=12 words\n"
    "BUY = clearly bullish for next 24h; SELL = clearly bearish; else HOLD. "
    "No other text. Educational exercise, not financial advice."
)

ACTION_RE = re.compile(r"ACTION\s*:\s*(BUY|SELL|HOLD)\b", re.IGNORECASE)
REASON_RE = re.compile(r"REASON\s*[:\-]\s*(.+)", re.IGNORECASE)
BARE_RE = re.compile(r"^\s*(BUY|SELL|HOLD)\b", re.IGNORECASE)


SYSTEMS = {
    "v2": SYSTEM_DEFAULT,
    "v3": (
        "You are an educational BTC news classifier. Decide ONLY what the NEWS implies "
        "for BTC over the NEXT 24h. Ignore any price/level quoted in the text (entry, "
        "support, resistance): judge the event, not the number.\n"
        "Output EXACTLY two lines:\n"
        "ACTION: BUY or SELL or HOLD\n"
        "REASON: <=12 words\n"
        "BUY = event clearly bullish for next 24h; SELL = clearly bearish; else HOLD. "
        "No other text. Educational exercise, not financial advice."
    ),
}


def load_system_prompt(version: str = PROMPT_VERSION) -> str:
    """Frozen system text per prompt version (file wins for v2 if present)."""
    if version == "v2":
        p = Path("prompts/llm_v2.txt")
        if p.is_file():
            text = p.read_text(encoding="utf-8")
            if "ACTION: BUY" in text:
                return SYSTEM_DEFAULT
        return SYSTEM_DEFAULT
    if version in SYSTEMS:
        return SYSTEMS[version]
    raise ValueError(f"unknown prompt_version {version!r}: allowed {sorted(SYSTEMS)}")


def generation_window(text: str) -> str:
    """Cut the generated window: after prompt echo, before timings.

    Avoids parsing the 'Loading model...' banner or build info as content.
    """
    txt = text or ""
    end = txt.find("[ Prompt")
    has_end = end != -1
    if has_end:
        txt = txt[:end]
    lines = txt.splitlines()
    start = 0
    found_start = False
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("> ") or "Respond with ACTION" in s:
            start = i + 1
            found_start = True
    if not found_start and not has_end:
        return ""  # no generation markers at all (banner only) -> nothing
    return "\n".join(lines[start:])


def parse_output(text: str) -> tuple[str | None, str]:
    """Parse (ACTION, reason) from raw llama-cli output.

    Strict ACTION: prefix first; tolerant bare BUY/SELL/HOLD fallback
    (model often drops the prefix — observed 4/5 pre-fix). Returns
    (None, "")-ish on unparseable so callers abstain honestly.
    """
    win = generation_window(text)
    m = ACTION_RE.search(win)
    if m:
        action: str | None = m.group(1).upper()
    else:
        action = None
        for line in win.splitlines()[:8]:
            mm = BARE_RE.search(line.strip())
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
        reason = ""
        for ln in [x.strip() for x in win.splitlines() if x.strip()]:
            if action and ln.upper().strip() == action:
                continue
            if action and ln.upper().startswith(action) and len(ln) < 25:
                rest = ln[len(action):].strip(" :-")
                if rest:
                    reason = rest[:120]
                    break
                continue
            reason = ln[:120]
            break
    return action, reason


def cache_key(prompt_version: str, title: str, body: str, price: float) -> str:
    """Stable cache key incl. prompt version (bump = miss, never mix)."""
    norm = "\x00".join([
        prompt_version,
        (title or "").strip().lower(),
        (body or "").strip().lower(),
        str(price),
    ])
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def _read_cache(cache_path: Path, key: str) -> dict | None:
    if not cache_path.is_file():
        return None
    try:
        with cache_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # corrupt line skipped + counted upstream
                if rec.get("key") == key:
                    return rec
    except OSError:
        return None
    return None


def _append_cache(cache_path: Path, rec: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def build_command(title: str, body: str, price: float,
                  prompt_version: str = PROMPT_VERSION) -> list[str]:
    """CPU-basica single-turn call. Never leaves stdin open (was the hang)."""
    prompt = (
        f"TITLE: {title}\nBODY: {body}\nPRICE_AT_PUBLISH_EUR: {price}\n"
        "Respond with ACTION and REASON lines only."
    )
    return [
        "tools/llama.cpp/bin/llama-cli",
        "-m", str(MODEL_PATH),
        "--n-gpu-layers", "0", "-c", "1024", "-t", "4",
        "--temp", "0", "--seed", "42", "-n", "60",
        "--reasoning", "off", "--no-display-prompt", "--simple-io", "-st",
        "-sys", load_system_prompt(prompt_version), "-p", prompt,
    ]


def predict_llm(title: str, body: str = "", price: float = 60000,
                cache_path: Path | None = None,
                timeout_s: int = TIMEOUT_S,
                prompt_version: str = PROMPT_VERSION) -> dict:
    """Predict one news item. Abstains (hold, parse_ok=False) on any failure.

    Raises FileNotFoundError naming the GGUF when the model file is absent
    (CLI maps to exit 3; never auto-downloads).
    """
    cache = Path(cache_path) if cache_path is not None else DEFAULT_CACHE
    key = cache_key(prompt_version, title, body or "", price)
    hit = _read_cache(cache, key)
    if hit:
        return {k: v for k, v in hit.items() if k != "key"}

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"LLM model missing: {MODEL_PATH}")

    t0 = time.time()
    try:
        proc = subprocess.run(
            build_command(title, body or "", price, prompt_version),
            capture_output=True, text=True, timeout=timeout_s, input="",
            encoding="utf-8", errors="replace",  # Windows cp1252 breaks on smart quotes
        )
        raw = (proc.stdout or "") + (proc.stderr or "")
        timeout = False
    except subprocess.TimeoutExpired:
        raw, timeout = "", True
    wall = time.time() - t0

    action, reason = (parse_output(raw) if not timeout else (None, ""))
    if timeout or not action:
        result = {"action": "hold", "conf": 0.0, "reason_short": reason,
                  "parse_ok": False, "timeout": timeout, "wall_s": round(wall, 2),
                  "prompt_version": prompt_version}
    else:
        result = {"action": action.lower(), "conf": 1.0,
                  "reason_short": reason, "parse_ok": True, "timeout": False,
                  "wall_s": round(wall, 2), "prompt_version": prompt_version}
    _append_cache(cache, {"key": key, **result})
    return result

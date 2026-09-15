"""PR1 RED — LLM backend parser/cache/timeout (mocked, no GGUF, no network)."""

from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess

import pytest

from src.llm_backend import (
    PROMPT_VERSION,
    cache_key,
    parse_output,
    predict_llm,
)


RAW_STRICT = """Loading model...
> TITLE: Bitcoin ETF inflows hit record
BODY: flows.
PRICE_AT_PUBLISH_EUR: 60000
Respond with ACTION and REASON lines only.
ACTION: BUY
REASON: Record inflows show strong demand

[ Prompt: 43.0 t/s | Generation: 11.4 t/s ]
Exiting...
"""

RAW_BARE = """> TITLE: Breakout
BODY: momentum join.
Respond with ACTION and REASON lines only.
BUY
Momentum traders join the breakout move

[ Prompt: 48.7 t/s | Generation: 12.8 t/s ]
"""

RAW_BROKEN = """> TITLE: foo
BODY: bar
Respond with ACTION and REASON lines only.
I am not sure about markets today, maybe later.

[ Prompt: 50.0 t/s | Generation: 12.0 t/s ]
"""

RAW_BANNER_ONLY = """Loading model...
build : b10964
Exiting...
"""


def test_strict_action_parsed():
    action, reason = parse_output(RAW_STRICT)
    assert action == "BUY"
    assert "inflows" in reason.lower()


def test_bare_token_fallback():
    action, reason = parse_output(RAW_BARE)
    assert action == "BUY"
    assert "momentum" in reason.lower()


def test_broken_output_abstains():
    action, reason = parse_output(RAW_BROKEN)
    assert action is None


def test_banner_never_parsed_as_reason():
    action, reason = parse_output(RAW_BANNER_ONLY)
    assert action is None
    assert "loading" not in reason.lower()


def test_lowercase_action():
    action, _ = parse_output("> t\nRespond.\naction: sell\nREASON: ban fears\n[ Prompt:")
    assert action == "SELL"


def test_cache_key_stable_and_versioned():
    k1 = cache_key("v2", "Title", "Body", 60000)
    k2 = cache_key("v2", "Title", "Body", 60000)
    k3 = cache_key("v3", "Title", "Body", 60000)
    assert k1 == k2 and k1 != k3 and len(k1) == 40


def _fake_ok(stdout):
    m = MagicMock()
    m.stdout = stdout
    m.stderr = ""
    m.returncode = 0
    return m


def test_predict_happy_path_mocked(tmp_path):
    cache = tmp_path / "cache.jsonl"
    with patch("src.llm_backend.subprocess.run", return_value=_fake_ok(RAW_STRICT)) as r:
        out = predict_llm("T", "B", 60000, cache_path=cache)
    assert out["action"] == "buy" and out["parse_ok"] is True
    assert out["prompt_version"] == PROMPT_VERSION
    assert r.call_count == 1
    # second call hits cache, no subprocess
    with patch("src.llm_backend.subprocess.run") as r2:
        out2 = predict_llm("T", "B", 60000, cache_path=cache)
    assert out2["action"] == "buy" and r2.call_count == 0


def test_predict_timeout_abstains(tmp_path):
    cache = tmp_path / "cache.jsonl"
    with patch("src.llm_backend.subprocess.run",
               side_effect=subprocess.TimeoutExpired(cmd="x", timeout=120)):
        out = predict_llm("T", "B", 60000, cache_path=cache)
    assert out["action"] == "hold" and out["parse_ok"] is False
    assert out["timeout"] is True


def test_predict_broken_abstains_zero_conf(tmp_path):
    cache = tmp_path / "cache.jsonl"
    with patch("src.llm_backend.subprocess.run", return_value=_fake_ok(RAW_BROKEN)):
        out = predict_llm("T", "B", 60000, cache_path=cache)
    assert out["action"] == "hold" and out["parse_ok"] is False
    assert out["conf"] == 0.0


def test_missing_gguf_raises_naming_path(tmp_path):
    with patch("src.llm_backend.MODEL_PATH", tmp_path / "nope.gguf"):
        with pytest.raises(FileNotFoundError, match="nope.gguf"):
            predict_llm("T", "B", 60000, cache_path=tmp_path / "c.jsonl")

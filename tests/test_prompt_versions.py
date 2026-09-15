"""Versiones de prompt aisladas: v3 no contamina v2 ni su caché."""
from unittest.mock import patch, MagicMock
from src import llm_backend as B

def test_versions_have_distinct_system_text():
    assert B.SYSTEMS["v2"] != B.SYSTEMS["v3"]
    assert "24h" in B.SYSTEMS["v3"] or "NEXT 24h" in B.SYSTEMS["v3"]

def test_cache_key_separates_versions():
    assert B.cache_key("v2", "t", "b", 1) != B.cache_key("v3", "t", "b", 1)

def test_predict_sends_versioned_prompt(tmp_path):
    seen = {}
    m = MagicMock(); m.stdout = "> x\nRespond.\nACTION: BUY\nREASON: strong demand\n[ Prompt:"; m.stderr = ""
    def fake_run(cmd, **kw):
        seen["sys"] = cmd[cmd.index("-sys") + 1]
        return m
    with patch.object(B, "MODEL_PATH", tmp_path / "m.gguf"):
        B.MODEL_PATH.touch()
        with patch("src.llm_backend.subprocess.run", side_effect=fake_run):
            out = B.predict_llm("t", "b", 1, cache_path=tmp_path / "c.jsonl", prompt_version="v3")
    assert out["prompt_version"] == "v3" and seen["sys"] == B.SYSTEMS["v3"]

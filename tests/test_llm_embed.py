"""Extractor de embeddings via llama-server (mockeado, sin red)."""
import numpy as np
import requests as rq
from unittest.mock import patch, MagicMock
from src import llm_embed as E

def _resp(vecs):
    m = MagicMock(); m.raise_for_status.return_value = None
    m.json.return_value = [{"index": 0, "embedding": vecs}]
    return m

def test_pooled_single_vector(tmp_path):
    with patch("src.llm_embed.requests.post", return_value=_resp([[0.0, 3.0, 4.0]])):
        out = E.embed_texts(["uniq-a"], cache_path=tmp_path / "c.npz", base_url="http://x")
    assert out.shape == (1, 3)
    assert abs(float((out[0] ** 2).sum() ** 0.5) - 1.0) < 1e-6

def test_token_list_mean_pooled(tmp_path):
    with patch("src.llm_embed.requests.post", return_value=_resp([[1.0, 0.0], [0.0, 1.0]])):
        out = E.embed_texts(["uniq-b"], cache_path=tmp_path / "c.npz", base_url="http://x")
    assert out.shape == (1, 2)
    assert abs(out[0, 0] - out[0, 1]) < 1e-6

def test_cache_avoids_resend(tmp_path):
    cache = tmp_path / "c.npz"
    with patch("src.llm_embed.requests.post", return_value=_resp([[1.0, 0.0]])) as p:
        E.embed_texts(["a", "b"], cache_path=cache, base_url="http://x")
        assert p.call_count == 2
    with patch("src.llm_embed.requests.post", return_value=_resp([[9.0, 9.0]])) as p2:
        out = E.embed_texts(["a", "b"], cache_path=cache, base_url="http://x")
        assert p2.call_count == 0 and out.shape == (2, 2)
        assert abs(out[0, 0] - 1.0) < 1e-6

def test_server_down_raises(tmp_path):
    with patch("src.llm_embed.requests.post", side_effect=rq.ConnectionError("down")):
        try:
            E.embed_texts(["uniq-c"], cache_path=tmp_path / "c.npz", base_url="http://x")
            raise SystemExit("debio fallar")
        except ConnectionError:
            pass

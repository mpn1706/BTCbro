"""Extractor LLM-as-features: embeddings del server residente + caché .npz.

El server (llama-server --embeddings) mantiene el modelo cargado: ~0.3s/texto
vs ~8s generando. Respuesta: [{"index":0,"embedding":[[vec-pooled]]}] o lista
de vectores por token -> mean client-side + L2. Sin red en tests (mock).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import requests

DEFAULT_URL = "http://127.0.0.1:8899"
DEFAULT_CACHE = Path("data/interim/embed_cache.npz")


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _fetch(text: str, base_url: str, timeout_s: int = 120) -> np.ndarray:
    try:
        r = requests.post(f"{base_url}/embedding", json={"content": text},
                          timeout=timeout_s)
        r.raise_for_status()
        payload = r.json()
    except requests.ConnectionError as exc:
        raise ConnectionError(f"embedding server caído en {base_url}: {exc}") from exc
    vecs = np.array(payload[0]["embedding"], dtype=float)
    if vecs.ndim == 1:
        vec = vecs
    else:
        vec = vecs.mean(axis=0)
    n = float(np.linalg.norm(vec)) or 1.0
    return (vec / n).astype(np.float32)


def _load_cache(cache_path: Path) -> dict[str, np.ndarray]:
    if not cache_path.is_file():
        return {}
    try:
        z = np.load(str(cache_path), allow_pickle=True)
        return {str(k): z[k] for k in z.files}
    except Exception:
        return {}


def embed_texts(texts: list[str], cache_path: Path | None = None,
                base_url: str = DEFAULT_URL, timeout_s: int = 120) -> np.ndarray:
    """Embed lista de textos (2048-d Qwen3). Cachea por sha1 en .npz."""
    cache = Path(cache_path) if cache_path is not None else DEFAULT_CACHE
    store = _load_cache(cache)
    rows: list[np.ndarray] = []
    new_count = 0
    for t in texts:
        k = "k" + _key(t or "")
        if k not in store:
            store[k] = _fetch(t or "", base_url, timeout_s)
            new_count += 1
            if new_count % 100 == 0:  # batch saves: per-item rewrite is O(n^2)
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(str(cache), **{k: np.asarray(v) for k, v in store.items()})
        rows.append(store[k])
    if new_count % 100 != 0:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(cache), **{k: np.asarray(v) for k, v in store.items()})
    return np.stack(rows).astype(np.float32)

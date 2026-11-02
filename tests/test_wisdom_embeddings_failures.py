"""Tail-110 pins for nomorals/wisdom/embeddings.py — 4 exclusions, 0 converts.

All four sites are probes: an availability/dimension probe's exception
IS its honest negative answer. Each pin proves the negative plus a
control half where the probe answers positively. Requesting a backend
that probed unavailable raises EmbeddingError with install guidance
(the __init__ guards sit outside these except blocks) — pinned by the
available() contract itself, so nothing here fabricates capability.

Sites:
* 149 E104 — FastEmbedBackend.available(): import probe for
  fastembed/onnxruntime; any import failure -> False.
* 222 E104 — SentenceTransformersBackend.available(): import probe
  for sentence_transformers -> False.
* 267 E104 — SentenceTransformersBackend.dim: a loaded model whose
  dimension call raises degrades to None, the property's documented
  ``int | None`` unknown; unloaded known models answer from the
  static table instead (control).
* 308 E104 — OllamaBackend.available(): HTTP probe of the local
  Ollama /api/tags endpoint; unreachable/error -> False.
"""

from __future__ import annotations

import json
import sys
import types
import urllib.request
from unittest import mock

from nomorals.wisdom import embeddings as emb


# ── 149: FastEmbed availability probe ───────────────────────────────────

def test_149_missing_fastembed_is_unavailable():
    with mock.patch.dict(sys.modules, {"fastembed": None, "onnxruntime": None}):
        assert emb.FastEmbedBackend.available() is False


def test_149_control_importable_is_available():
    with mock.patch.dict(sys.modules, {
            "fastembed": types.ModuleType("fastembed"),
            "onnxruntime": types.ModuleType("onnxruntime")}):
        assert emb.FastEmbedBackend.available() is True


# ── 222: SentenceTransformers availability probe ────────────────────────

def test_222_missing_sentence_transformers_is_unavailable():
    with mock.patch.dict(sys.modules, {"sentence_transformers": None}):
        assert emb.SentenceTransformersBackend.available() is False


def test_222_control_importable_is_available():
    with mock.patch.dict(sys.modules, {
            "sentence_transformers": types.ModuleType("sentence_transformers")}):
        assert emb.SentenceTransformersBackend.available() is True


# ── 267: dim probe ──────────────────────────────────────────────────────

class _BadModel:
    def get_sentence_embedding_dimension(self):
        raise RuntimeError("shape?")


class _GoodModel:
    def get_sentence_embedding_dimension(self):
        return 768


def _st_backend():
    return emb.SentenceTransformersBackend.__new__(
        emb.SentenceTransformersBackend)


def test_267_dim_crash_degrades_to_none():
    b = _st_backend()
    b._model = _BadModel()
    b.model = "whatever"
    assert b.dim is None


def test_267_control_loaded_model_dim():
    b = _st_backend()
    b._model = _GoodModel()
    b.model = "whatever"
    assert b.dim == 768


def test_267_control_unloaded_known_model_uses_table():
    b = _st_backend()
    b._model = None
    b.model = "sentence-transformers/all-MiniLM-L6-v2"
    assert b.dim == 384


# ── 308: Ollama availability probe ──────────────────────────────────────

class _Resp:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_308_ollama_unreachable_is_unavailable():
    with mock.patch.object(urllib.request, "urlopen",
                           side_effect=OSError("no ollama")):
        assert emb.OllamaBackend.available() is False


def test_308_control_ollama_with_embed_model_is_available():
    payload = json.dumps({"models": [{"name": "nomic-embed-text"}]}).encode()
    with mock.patch.object(urllib.request, "urlopen",
                           return_value=_Resp(payload)):
        assert emb.OllamaBackend.available() is True

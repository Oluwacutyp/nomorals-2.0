"""Build-map #29: brain upgrade path. All offline — no servers spawned."""
from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


# ── KV-cache quantization (local_server) ────────────────────────────────────

def _manager(monkeypatch, **kwargs):
    monkeypatch.setenv("NM_PROFILE", kwargs.pop("profile", "termux"))
    from nomorals.llm.local_server import GGUFServerManager
    return GGUFServerManager(**kwargs)


def test_termux_always_gets_kv_quant(monkeypatch):
    mgr = _manager(monkeypatch, profile="termux")
    assert mgr.kv_cache_quant == "q8_0"
    args = mgr.build_args("/m/gguf", "/usr/bin/llama-server")
    assert "--ctk" in args and "--ctv" in args
    i = args.index("--ctk")
    assert args[i + 1] == "q8_0" and args[i + 3] == "q8_0"


def test_termux_flash_attn_off(monkeypatch):
    mgr = _manager(monkeypatch, profile="termux")
    assert mgr.flash_attn is False
    args = mgr.build_args("/m/gguf", "/usr/bin/llama-server")
    assert "--flash-attn" in args
    assert args[args.index("--flash-attn") + 1] == "off"


def test_termux_kv_quant_explicitly_disabled(monkeypatch):
    mgr = _manager(monkeypatch, profile="termux", kv_cache_quant="")
    assert mgr.kv_cache_quant is None
    assert "--ctk" not in mgr.build_args("/m/gguf", "llama-server")


def test_workstation_kv_quant_opt_in(monkeypatch):
    monkeypatch.delenv("LLAMA_KV_QUANT", raising=False)
    mgr = _manager(monkeypatch, profile="workstation")
    assert mgr.kv_cache_quant is None
    assert "--ctk" not in mgr.build_args("/m/gguf", "llama-server")
    monkeypatch.setenv("LLAMA_KV_QUANT", "q8_0")
    mgr2 = _manager(monkeypatch, profile="workstation")
    assert mgr2.kv_cache_quant == "q8_0"
    assert "--ctk" in mgr2.build_args("/m/gguf", "llama-server")


def test_flash_attn_off_on_arm(monkeypatch):
    monkeypatch.setattr("nomorals.llm.local_server._is_arm", lambda: True)
    mgr = _manager(monkeypatch, profile="workstation")
    assert mgr.flash_attn is False
    args = mgr.build_args("/m/gguf", "llama-server")
    assert args[args.index("--flash-attn") + 1] == "off"


def test_flash_attn_on_x86_workstation(monkeypatch):
    monkeypatch.setattr("nomorals.llm.local_server._is_arm", lambda: False)
    mgr = _manager(monkeypatch, profile="workstation")
    assert mgr.flash_attn is True
    args = mgr.build_args("/m/gguf", "llama-server")
    assert args[args.index("--flash-attn") + 1] == "on"


def test_explicit_args_win(monkeypatch):
    mgr = _manager(monkeypatch, profile="termux",
                   kv_cache_quant="q4_0", flash_attn=True)
    args = mgr.build_args("/m/gguf", "llama-server")
    assert args[args.index("--ctk") + 1] == "q4_0"
    assert args[args.index("--flash-attn") + 1] == "on"


# ── model catalog ──────────────────────────────────────────────────────────

def test_pick_brain_termux(monkeypatch):
    monkeypatch.setenv("NM_PROFILE", "termux")
    monkeypatch.delenv("DEVON_BRAIN", raising=False)
    from nomorals.llm.model_catalog import pick_brain
    assert pick_brain() == "codebeast-3.8b"


def test_pick_brain_workstation(monkeypatch):
    monkeypatch.setenv("NM_PROFILE", "workstation")
    monkeypatch.delenv("DEVON_BRAIN", raising=False)
    from nomorals.llm.model_catalog import pick_brain
    assert pick_brain() == "qwen3.5-4b"


def test_pick_brain_env_override(monkeypatch):
    monkeypatch.setenv("DEVON_BRAIN", "phi-4-mini")
    from nomorals.llm.model_catalog import pick_brain
    assert pick_brain("termux") == "phi-4-mini"


def test_pick_brain_multimodal_warns_without_mmproj(monkeypatch, caplog):
    monkeypatch.delenv("DEVON_BRAIN", raising=False)
    from nomorals.llm import model_catalog
    monkeypatch.setattr(model_catalog, "_mmproj_present", lambda cache_dir="models": False)
    with caplog.at_level("WARNING"):
        assert model_catalog.pick_brain(multimodal=True) == "qwen3-vl-4b"
    assert "#3899" in caplog.text


def test_catalog_entries_complete():
    from nomorals.llm.model_catalog import BRAIN_MODELS
    for model_id, spec in BRAIN_MODELS.items():
        for field in ("hf_repo", "quant", "size_gb", "context", "license", "notes"):
            assert field in spec, f"{model_id} missing {field}"


# ── benchmark harness ──────────────────────────────────────────────────────

class _FakeServer:
    def __init__(self, text):
        self.text = text

    def generate(self, prompt):
        return self.text


def test_benchmark_report_structure():
    from nomorals.llm.benchmark import TASKS, run_benchmark

    def factory(model):
        text = {"m-a": "return a + b", "m-b": "naira dollar blah"}.get(model, "hmm")
        return _FakeServer(text)

    tasks = [
        {"id": "code_fix", "prompt": "fix", "keywords": ["return a + b"]},
        {"id": "summarize", "prompt": "sum", "keywords": ["naira", "dollar"]},
    ]
    report = run_benchmark(factory, ["m-a", "m-b"], tasks)
    assert len(report.results) == 4
    by_task = {(r.model, r.task_id): r for r in report.results}
    assert by_task[("m-a", "code_fix")].quality == 1.0
    assert by_task[("m-a", "summarize")].quality == 0.0
    assert by_task[("m-b", "code_fix")].quality == 0.0
    assert by_task[("m-b", "summarize")].quality == 1.0
    summary = report.summary()
    assert summary["m-a"]["avg_quality"] == 0.5
    assert summary["m-b"]["avg_quality"] == 0.5
    assert report.winner() in ("m-a", "m-b")  # tie → tokens/sec decides
    d = report.to_dict()
    assert d["models"] == ["m-a", "m-b"] and len(d["results"]) == 4


def test_benchmark_never_raises_per_task():
    from nomorals.llm.benchmark import run_benchmark

    class _Boom:
        def generate(self, prompt):
            raise RuntimeError("gpu exploded")

    report = run_benchmark(lambda m: _Boom(), ["m-x"],
                           [{"id": "t", "prompt": "p", "keywords": ["k"]}])
    assert report.results[0].error
    assert report.results[0].quality == 0.0
    assert report.winner() == ""


def test_benchmark_save(tmp_path):
    from nomorals.llm.benchmark import run_benchmark
    report = run_benchmark(lambda m: _FakeServer("return a + b"),
                           ["m-a"], [{"id": "t", "prompt": "p",
                                      "keywords": ["return a + b"]}])
    path = report.save(tmp_path / "bench.json")
    loaded = json.loads(path.read_text())
    assert loaded["winner"] == "m-a"


def test_default_tasks_are_devons_own():
    from nomorals.llm.benchmark import TASKS
    ids = {t["id"] for t in TASKS}
    assert {"code_fix", "summarize", "nl_parse", "multilingual"} <= ids


# ── qwen3 embeddings ───────────────────────────────────────────────────────

class _EmbeddingHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        n = len(body.get("input", []))
        payload = {"data": [
            {"index": i, "embedding": [0.1 * (i + 1)] * 8} for i in range(n)]}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture()
def embedding_server():
    server = HTTPServer(("127.0.0.1", 0), _EmbeddingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_qwen3_embedder_happy_path(embedding_server):
    from nomorals.memory.embeddings import Qwen3Embedder
    emb = Qwen3Embedder(url=embedding_server)
    vectors = emb.embed_many(["hello", "world"])
    assert len(vectors) == 2 and len(vectors[0]) == 8
    assert emb.available()


def test_qwen3_embedder_falls_back_to_hashing(monkeypatch):
    monkeypatch.setenv("EMBEDDING_URL", "http://127.0.0.1:1")  # dead port
    from nomorals.memory.embeddings import Embedder
    emb = Embedder(provider="qwen3", dimensions=64)
    assert emb.is_semantic  # optimistic until the probe fails
    vectors = emb.embed_many(["hello world"])
    assert len(vectors) == 1 and len(vectors[0]) == 64
    assert emb.stats["fallbacks"] == 1
    assert not emb.is_semantic  # probe failed — honest now
    # Second call goes straight to hashing (no repeated doomed probe).
    emb.embed_many(["another"])
    assert emb.stats["fallbacks"] == 1


def test_qwen3_provider_uses_server(monkeypatch, embedding_server):
    monkeypatch.setenv("EMBEDDING_URL", embedding_server)
    from nomorals.memory.embeddings import Embedder
    emb = Embedder(provider="qwen3")
    vectors = emb.embed_many(["hello"])
    assert emb.dimensions == 8  # server's native dim adopted
    assert len(vectors[0]) == 8


def test_qwen3_env_overrides(monkeypatch):
    monkeypatch.setenv("EMBEDDING_URL", "http://example.test:9999")
    monkeypatch.setenv("EMBEDDING_MODEL", "custom-emb")
    from nomorals.memory.embeddings import Qwen3Embedder
    emb = Qwen3Embedder()
    assert emb.url == "http://example.test:9999"
    assert emb.model == "custom-emb"

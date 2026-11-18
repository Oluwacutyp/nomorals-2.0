"""Lane 65 — wire ``lora_for`` into the live local-server path.

Failing-first: on pristine 3068ff5e there is NO production path from a
character's ``lora_file`` to a serving llama-server. ``lora_for`` itself
resolves + validates the adapter, but nothing calls it outside tests,
no ``lora_speak`` exists, ``GGUFServerManager`` has no completion call,
no LoRA-server registry exists, and ``/charagent`` has no way to name an
adapter. These tests pin the whole path:

* ``specialists.lora_speak`` — resolve through ``lora_for`` (never
  bypassing its validation), complete through a serving LoRA server,
  honest ``ok: False`` naming the wall otherwise, never raises;
* ``llm/local_server.py`` — ``GGUFServerManager.complete`` (chat
  completions over the module's own ``HttpClient`` seam) +
  ``ensure_lora_server`` (one manager per adapter, derived port, RAM
  gate for the second copy of the base weights, refuses a foreign
  port-holder by bumping the port);
* ``Character.speak`` — LoRA first when an adapter is named, brain
  ``suggest`` as the fallback generator, deterministic fallback last;
  the outcome rides on ``char._last_lora_status`` for the caller;
* ``/charagent`` — ``create ... lora=<path>`` names an adapter,
  ``lora <name> [path|none]`` sets/shows/clears it, ``talk`` names the
  LoRA wall on fallback and stays clean when the adapter speaks.

No real model or binary is needed: servers are fakes or a stub HTTP
server on an ephemeral port; the adapter fixture is a real 10 MB GGUF
(magic + size, the ``gguf_check`` floor).
"""
import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.characters import Character, CharacterStore
from nomorals.characters import specialists as cs

# ── fixtures ─────────────────────────────────────────────────────────────

def _adapter(tmp_path, name="mara.gguf", size_mb=10):
    """A real-enough GGUF adapter: magic at offset 0, >= 10 MB."""
    path = tmp_path / name
    with open(path, "wb") as fh:
        fh.write(b"GGUF")
        fh.seek(size_mb * 1024 * 1024 - 1)
        fh.write(b"\x00")
    return str(path)


def _suggest(prompt: str) -> str:
    first = (prompt or "").splitlines()[0]
    if first.startswith("You are Mara"):
        return "MARA-LINE"
    return "BRAIN-LINE"


def _handler():
    from nomorals.agents.partner.runtime_games import RuntimeGamesMixin

    class _RT(RuntimeGamesMixin):
        def __init__(self, ctx):
            self.context = ctx

    return _RT(SimpleNamespace(suggest=_suggest))


def _scratch_store():
    root = Path(tempfile.mkdtemp())
    os.environ["NOMORALS_DATA_DIR"] = str(root)
    store = CharacterStore()
    assert store.dir == root / "characters"
    return root, store


def _make_char(store, **kw):
    c = Character(name="Mara", persona={"witty": 0.9}, **kw)
    store.save(c)
    return c


class _FakeProcess:
    def poll(self):
        return None  # alive


class _FakeManager:
    """A GGUFServerManager double: records construction, fakes start/complete."""

    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.host = kwargs.get("host", "127.0.0.1")
        self.port = kwargs.get("port", 8080)
        self.lora_files = kwargs.get("lora_files", "")
        self._process = None
        self.model_path = ""
        self.start_calls = []
        self.complete_text = "LORA-LINE"
        self.complete_exc = None
        _FakeManager.instances.append(self)

    @property
    def base_url(self):
        return f"http://{self.host}:{self.port}"

    def _healthy(self):
        return self._process is not None

    def start(self, model):
        from nomorals.llm.local_server import GGUFDiagnosis

        self.start_calls.append(model)
        self._process = _FakeProcess()  # ours: a live child, not "already serving"
        self.model_path = str(model)
        return GGUFDiagnosis(ok=True, binary="/bin/llama-server",
                             model_path=str(model), url=self.base_url)

    def stop(self):
        self._process = None

    def complete(self, prompt, **kw):
        if self.complete_exc is not None:
            raise self.complete_exc
        return self.complete_text


@pytest.fixture(autouse=True)
def _reset_lora_state(monkeypatch):
    """Each test starts with an empty LoRA-server registry + down-cache."""
    from nomorals.llm import local_server as ls

    if hasattr(ls, "stop_lora_servers"):
        ls.stop_lora_servers()
    if hasattr(cs, "reset_lora_down_cache"):
        cs.reset_lora_down_cache()
    _FakeManager.instances.clear()
    yield
    if hasattr(ls, "stop_lora_servers"):
        ls.stop_lora_servers()
    if hasattr(cs, "reset_lora_down_cache"):
        cs.reset_lora_down_cache()


# ── specialists.lora_speak ───────────────────────────────────────────────

def test_lora_speak_names_no_adapter_wall():
    char = Character(name="Mara")
    result = cs.lora_speak(char, "hello")
    assert result["ok"] is False
    assert "no lora" in result["reason"].lower()


def test_lora_speak_validates_adapter_via_lora_for(tmp_path):
    char = Character(name="Mara", lora_file=str(tmp_path / "missing.gguf"))
    result = cs.lora_speak(char, "hello")
    assert result["ok"] is False
    assert "missing" in result["reason"].lower()


def test_lora_speak_uses_live_server(monkeypatch, tmp_path):
    from nomorals.llm import local_server as ls

    adapter = _adapter(tmp_path)
    char = Character(name="Mara", lora_file=adapter)
    fake = _FakeManager(port=8096)
    monkeypatch.setattr(ls, "ensure_lora_server",
                        lambda *a, **k: (fake, ""))
    result = cs.lora_speak(char, "You are Mara: hello",
                           base_model="/models/base.gguf")
    assert result["ok"] is True
    assert result["text"] == "LORA-LINE"
    assert result["adapter"] == adapter
    assert "8096" in result["server"]


def test_lora_speak_empty_completion_is_failure(monkeypatch, tmp_path):
    from nomorals.llm import local_server as ls

    char = Character(name="Mara", lora_file=_adapter(tmp_path))
    fake = _FakeManager(port=8096)
    fake.complete_text = "   "
    monkeypatch.setattr(ls, "ensure_lora_server",
                        lambda *a, **k: (fake, ""))
    result = cs.lora_speak(char, "hello", base_model="/models/base.gguf")
    assert result["ok"] is False
    assert "no text" in result["reason"].lower()


def test_lora_speak_server_error_names_wall_never_raises(monkeypatch, tmp_path):
    from nomorals.llm import local_server as ls

    char = Character(name="Mara", lora_file=_adapter(tmp_path))
    fake = _FakeManager(port=8096)
    fake.complete_exc = RuntimeError("llama-server chat failed: refused")
    monkeypatch.setattr(ls, "ensure_lora_server",
                        lambda *a, **k: (fake, ""))
    result = cs.lora_speak(char, "hello", base_model="/models/base.gguf")
    assert result["ok"] is False
    assert "refused" in result["reason"]


def test_lora_speak_negative_memo_skips_repeat_boot(monkeypatch, tmp_path):
    from nomorals.llm import local_server as ls

    char = Character(name="Mara", lora_file=_adapter(tmp_path))
    calls = []

    def failing_ensure(*a, **k):
        calls.append(1)
        return None, "no llama.cpp binary found"

    monkeypatch.setattr(ls, "ensure_lora_server", failing_ensure)
    first = cs.lora_speak(char, "hello", base_model="/models/base.gguf")
    second = cs.lora_speak(char, "hello", base_model="/models/base.gguf")
    assert first["ok"] is False and second["ok"] is False
    assert "no llama.cpp binary" in second["reason"]
    assert len(calls) == 1  # the second call is served from the memo


def test_lora_speak_needs_a_base_model(tmp_path):
    char = Character(name="Mara", lora_file=_adapter(tmp_path))
    result = cs.lora_speak(char, "hello", base_model="")
    assert result["ok"] is False
    assert "base model" in result["reason"].lower()


# ── llm/local_server.py: ensure_lora_server + complete ───────────────────

def test_lora_port_derivation_is_stable_and_offset():
    from nomorals.llm.local_server import lora_server_port

    p1 = lora_server_port("/x/mara.gguf", base_port=8080)
    p2 = lora_server_port("/x/mara.gguf", base_port=8080)
    assert p1 == p2
    assert 8096 <= p1 <= 8111  # base + 16 .. base + 31, never the shared port


def test_ensure_lora_server_starts_manager_with_adapter(monkeypatch, tmp_path):
    from nomorals.llm import local_server as ls

    adapter = _adapter(tmp_path)
    base = _adapter(tmp_path, name="base.gguf", size_mb=10)
    monkeypatch.setattr(ls, "GGUFServerManager", _FakeManager)
    monkeypatch.setattr(ls, "find_gguf", lambda *a, **k: base)
    monkeypatch.setattr(ls, "port_in_use", lambda *a, **k: False)
    monkeypatch.setattr("nomorals.core.profile.memory_mb", lambda: 32768)

    mgr, reason = ls.ensure_lora_server(adapter, base_model=base)
    assert reason == ""
    assert isinstance(mgr, _FakeManager)
    assert mgr.lora_files == adapter
    assert mgr.start_calls == [base]
    # second ensure reuses the same manager — no second boot
    mgr2, _ = ls.ensure_lora_server(adapter, base_model=base)
    assert mgr2 is mgr
    assert len(mgr.start_calls) == 1


def test_ensure_lora_server_bumps_a_foreign_port(monkeypatch, tmp_path):
    from nomorals.llm import local_server as ls

    adapter = _adapter(tmp_path)
    base = _adapter(tmp_path, name="base.gguf", size_mb=10)
    derived = ls.lora_server_port(adapter, base_port=8080)
    monkeypatch.setattr(ls, "GGUFServerManager", _FakeManager)
    monkeypatch.setattr(ls, "find_gguf", lambda *a, **k: base)
    # the derived port is held by someone else; the next one is free
    monkeypatch.setattr(ls, "port_in_use",
                        lambda host, port: port == derived)
    monkeypatch.setattr("nomorals.core.profile.memory_mb", lambda: 32768)

    mgr, reason = ls.ensure_lora_server(adapter, base_model=base)
    assert reason == ""
    assert mgr.port == derived + 1
    assert mgr.start_calls == [base]


def test_ensure_lora_server_ram_gate_names_the_numbers(monkeypatch, tmp_path):
    from nomorals.llm import local_server as ls

    adapter = _adapter(tmp_path)
    base = _adapter(tmp_path, name="base.gguf", size_mb=10)
    monkeypatch.setattr(ls, "GGUFServerManager", _FakeManager)
    monkeypatch.setattr(ls, "find_gguf", lambda *a, **k: base)
    # 2 GB box, 10 MB base: the gate is about the FILE size vs RAM only
    # when the file is big — fake a 4 GB base on a 2 GB box.
    monkeypatch.setattr("nomorals.core.profile.memory_mb", lambda: 2048)
    monkeypatch.setattr(os.path, "getsize", lambda p: 4 * 1024 * 1024 * 1024)

    mgr, reason = ls.ensure_lora_server(adapter, base_model=base)
    assert mgr is None
    assert "ram" in reason.lower()
    assert _FakeManager.instances == []  # nothing was even constructed


def test_manager_complete_posts_chat_completions():
    """A REAL stub HTTP server: complete() speaks OpenAI chat on it."""
    from nomorals.llm.local_server import GGUFServerManager

    seen = {}

    class _H(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            seen["body"] = json.loads(self.rfile.read(
                int(self.headers["Content-Length"] or 0)))
            payload = json.dumps({"choices": [{"message": {
                "role": "assistant", "content": "STUB-LINE"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        mgr = GGUFServerManager(port=server.server_address[1])
        text = mgr.complete("You are Mara: hello")
        assert text == "STUB-LINE"
        assert seen["body"]["messages"][0]["content"] == "You are Mara: hello"
        assert seen["body"]["stream"] is False
    finally:
        server.shutdown()
        server.server_close()


def test_manager_complete_raises_on_bad_shape():
    from nomorals.llm.local_server import GGUFServerManager

    class _H(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.dumps({"choices": []}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        mgr = GGUFServerManager(port=server.server_address[1])
        with pytest.raises(RuntimeError):
            mgr.complete("hello")
    finally:
        server.shutdown()
        server.server_close()


def test_health_status_reads_a_live_loopback_server():
    """REGRESSION (lane 65, found by tracing): ``health_status`` polled
    loopback through the SSRF-guarded ``HttpClient``, so a server
    answering {"status": "ok"} still read "dead" — and ``_healthy()``
    could never be True, so ``start()`` could never observe a boot
    succeed. The poll is explicitly internal tooling on loopback now."""
    from nomorals.llm.local_server import health_status

    class _H(BaseHTTPRequestHandler):
        def do_GET(self):
            payload = json.dumps({"status": "ok"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        assert health_status("127.0.0.1", server.server_address[1]) == "ok"
    finally:
        server.shutdown()
        server.server_close()


# ── Character.speak: LoRA first, suggest fallback ────────────────────────

def test_speak_prefers_lora_server_line(monkeypatch, tmp_path):
    char = Character(name="Mara", lora_file=_adapter(tmp_path))
    calls = []

    def fake_suggest(prompt):
        calls.append(prompt)
        return "BRAIN-LINE"

    monkeypatch.setattr(
        cs, "lora_speak",
        lambda c, prompt, **k: {"ok": True, "text": "LORA-LINE",
                                "adapter": c.lora_file,
                                "server": "http://127.0.0.1:8096"})
    line = char.speak("hello there", fake_suggest)
    assert line == "LORA-LINE"
    assert calls == []  # the brain was never asked
    assert char._last_lora_status["ok"] is True


def test_speak_falls_back_to_suggest_when_lora_down(monkeypatch, tmp_path):
    char = Character(name="Mara", lora_file=_adapter(tmp_path))
    monkeypatch.setattr(
        cs, "lora_speak",
        lambda c, prompt, **k: {"ok": False,
                                "reason": "no llama.cpp binary found"})
    line = char.speak("hello there", _suggest)
    assert line == "MARA-LINE"
    assert char._last_lora_status["ok"] is False
    assert "llama.cpp" in char._last_lora_status["reason"]


def test_speak_without_adapter_unchanged():
    char = Character(name="Mara")
    line = char.speak("hello there", _suggest)
    assert line == "MARA-LINE"
    assert getattr(char, "_last_lora_status", {}) == {}


# ── /charagent naming surface ────────────────────────────────────────────

def test_charagent_create_accepts_lora_arg(tmp_path):
    _, store = _scratch_store()
    adapter = _adapter(tmp_path)
    rt = _handler()
    out = rt._control_charagent(f"create Mara lora={adapter}", "local:test")
    char = store.get_by_name("Mara")
    assert char is not None
    assert char.lora_file == adapter
    assert "lora" in out.lower()
    assert "validated" in out.lower()  # the adapter passed gguf_check


def test_charagent_lora_verb_sets_shows_clears(tmp_path):
    _, store = _scratch_store()
    adapter = _adapter(tmp_path)
    _make_char(store)
    rt = _handler()

    out = rt._control_charagent(f"lora Mara {adapter}", "local:test")
    assert store.get_by_name("Mara").lora_file == adapter
    assert "validated" in out.lower()

    out = rt._control_charagent("lora Mara", "local:test")
    assert "mara.gguf" in out
    assert "validated" in out.lower()

    out = rt._control_charagent("lora Mara none", "local:test")
    assert store.get_by_name("Mara").lora_file == ""
    assert "cleared" in out.lower()


def test_charagent_lora_verb_names_missing_file(tmp_path):
    _, store = _scratch_store()
    _make_char(store)
    rt = _handler()
    out = rt._control_charagent(
        f"lora Mara {tmp_path / 'missing.gguf'}", "local:test")
    assert "missing" in out.lower()
    # the name is still recorded (the file may land later); talk falls back
    assert store.get_by_name("Mara").lora_file.endswith("missing.gguf")


def test_charagent_talk_names_lora_wall_on_fallback(monkeypatch, tmp_path):
    _, store = _scratch_store()
    _make_char(store, lora_file=str(tmp_path / "missing.gguf"))
    rt = _handler()
    out = rt._control_charagent("talk Mara hello there", "local:test")
    assert out.startswith("🎭 Mara: MARA-LINE")
    assert "LoRA:" in out
    assert "missing" in out.lower()


def test_charagent_talk_lora_success_is_clean_reply(monkeypatch, tmp_path):
    _, store = _scratch_store()
    _make_char(store, lora_file=_adapter(tmp_path))
    rt = _handler()
    monkeypatch.setattr(
        cs, "lora_speak",
        lambda c, prompt, **k: {"ok": True, "text": "LORA-LINE",
                                "adapter": c.lora_file,
                                "server": "http://127.0.0.1:8096"})
    out = rt._control_charagent("talk Mara hello there", "local:test")
    assert out == "🎭 Mara: LORA-LINE"  # no LoRA note when the adapter speaks


# ── the provision docstring stops claiming no call site ──────────────────

def test_provision_docstring_no_longer_claims_no_lora_call_site():
    from nomorals.characters import provision

    assert "lora_for`` have no live call site" not in (
        provision.__doc__ or "")

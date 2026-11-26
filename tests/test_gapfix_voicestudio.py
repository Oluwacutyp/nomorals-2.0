"""VoiceStudio integration: local voice studio over its HTTP API.

The organ (nomorals/voice/voicestudio.py) drives a locally-running
VoiceStudio backend (AGPL, external process) — clone, design, dub,
transcribe, narrate. Tests run WITHOUT a live backend: the client's
transport is faked with canned /health, /openapi.json, /v1/audio/*
responses. The live contract (real backend on 127.0.0.1:3900) is
covered by the SURVEY's live-proof steps, not by this file.
"""

from __future__ import annotations

import io
import json
import struct
import wave
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse

import pytest

from nomorals.core.http import HttpResponse, RequestError

# The organ FIRST: on pristine this file collection-errors naming it.
from nomorals.voice.voicestudio import (
    VoiceStudioClient,
    VoiceStudioError,
    engine_license,
)

FAKE_SCHEMA = {
    "openapi": "3.0.0",
    "paths": {
        "/v1/voices/clone": {
            "post": {
                "operationId": "cloneVoice",
                "tags": ["voices"],
                "summary": "Clone a voice from a reference recording",
                "requestBody": {"content": {"multipart/form-data": {
                    "schema": {"type": "object", "properties": {
                        "name": {"type": "string"},
                        "language": {"type": "string"},
                        "description": {"type": "string"},
                        "reference": {"type": "string",
                                      "format": "binary"},
                    }}}}}},
        },
        "/v1/voices/design": {
            "post": {
                "operationId": "designVoice",
                "tags": ["voices"],
                "summary": "Design a voice from a description",
                "requestBody": {"content": {"application/json": {
                    "schema": {"type": "object", "properties": {
                        "description": {"type": "string"},
                        "name": {"type": "string"},
                        "language": {"type": "string"},
                        "sample_text": {"type": "string"},
                    }}}}}},
        },
        "/v1/dubbing/jobs": {
            "post": {
                "operationId": "createDubJob",
                "tags": ["dubbing"],
                "summary": "Dub a video",
                "requestBody": {"content": {"multipart/form-data": {
                    "schema": {"type": "object", "properties": {
                        "target_language": {"type": "string"},
                        "voice": {"type": "string"},
                        "video": {"type": "string", "format": "binary"},
                    }}}}}},
        },
        "/v1/dubbing/jobs/{job_id}": {
            "get": {
                "operationId": "getDubJob",
                "tags": ["dubbing"],
                "summary": "Dub job status",
            }
        },
    },
}

FAKE_SCHEMA_NO_NATIVE = {"openapi": "3.0.0", "paths": {
    "/v1/audio/speech": {"post": {"operationId": "createSpeech"}},
}}


def make_wav_bytes(seconds: float = 0.1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        frames = int(8000 * seconds)
        w.writeframes(struct.pack(f"<{frames}h", *([0] * frames)))
    return buf.getvalue()


def make_wav_file(path: Path, seconds: float = 0.1) -> Path:
    path.write_bytes(make_wav_bytes(seconds))
    return path


class FakeTransport:
    """Canned VoiceStudio backend. ``mode`` selects a failure shape."""

    def __init__(self, mode: str = "ok", *, schema: dict | None = None,
                 dub_out: str = "", dub_never_done: bool = False):
        self.mode = mode
        self.schema = FAKE_SCHEMA if schema is None else schema
        self.dub_out = dub_out
        self.dub_never_done = dub_never_done
        self.calls: list[tuple] = []
        self._dub_polls = 0
        self._schema_calls = 0

    def _resp(self, status: int, body: bytes,
              ctype: str = "application/json") -> HttpResponse:
        return HttpResponse(status=status, body=body,
                            headers={"Content-Type": ctype},
                            url="http://127.0.0.1:3900/x")

    def _json(self, payload) -> HttpResponse:
        return self._resp(200, json.dumps(payload).encode())

    def get(self, url: str, **kw) -> HttpResponse:
        self.calls.append(("GET", url))
        if self.mode == "down":
            raise RequestError(
                "request to http://127.0.0.1:3900 failed: "
                "connection refused", retryable=True)
        path = urlparse(url).path
        if path == "/health":
            return self._json({"status": "ok", "version": "0.5.7"})
        if path == "/.well-known/voicestudio-speech":
            return self._json({"transports": ["http", "ws", "mcp"]})
        if path == "/v1/audio/voices":
            return self._json({
                "voices": [
                    {"id": "alloy", "name": "Alloy",
                     "engine": "omnivoice", "language": "en"},
                    {"id": "nova", "name": "Nova",
                     "engine": "cosyvoice", "language": "en"},
                ],
                "engines": ["omnivoice", "cosyvoice"]})
        if path == "/openapi.json":
            self._schema_calls += 1
            return self._json(self.schema)
        if path.startswith("/v1/dubbing/jobs/"):
            self._dub_polls += 1
            if self.dub_never_done or self._dub_polls == 1:
                return self._json({"status": "running"})
            return self._json({"status": "done",
                               "output_path": self.dub_out})
        raise RequestError(f"fake: no route for GET {path}")

    def post_json(self, url: str, payload, **kw) -> HttpResponse:
        self.calls.append(("POST-JSON", url, payload))
        if self.mode == "auth":
            raise HTTPError(url, 401, "Unauthorized", {}, io.BytesIO(b""))
        if self.mode == "error-body":
            return self._json({"error": "model not loaded"})
        path = urlparse(url).path
        if path == "/v1/audio/speech":
            if self.mode == "non-audio":
                return self._resp(200, b"oops this is not audio at all",
                                  "text/plain")
            return self._resp(200, make_wav_bytes(), "audio/wav")
        if path == "/v1/voices/design":
            return self._json({"id": "vs-design-1",
                               "engine": "omnivoice"})
        raise RequestError(f"fake: no route for POST {path}")

    def post_multipart(self, url: str, fields=None, files=None,
                       **kw) -> HttpResponse:
        self.calls.append(("POST-MULTIPART", url, fields or {}, files or []))
        path = urlparse(url).path
        if path == "/v1/audio/transcriptions":
            fmt = (fields or {}).get("response_format", "verbose_json")
            if fmt == "text":
                return self._resp(200, b"hello world", "text/plain")
            return self._json({
                "text": "hello world",
                "segments": [{"start": 0.0, "end": 1.0,
                              "text": "hello world"}],
                "duration": 1.5})
        if path == "/v1/voices/clone":
            return self._json({"id": "vs-clone-1",
                               "engine": "omnivoice"})
        if path == "/v1/dubbing/jobs":
            return self._json({"job_id": "job-1"})
        raise RequestError(f"fake: no route for POST {path}")

    def request(self, method: str, url: str, **kw) -> HttpResponse:
        if method.upper() == "GET":
            return self.get(url, **kw)
        raise RequestError(f"fake: no route for {method}")

    def download(self, url: str, destination) -> Path:
        dest = Path(destination)
        dest.write_bytes(b"fake-video")
        return dest


def make_client(tmp_path, mode: str = "ok", **kw) -> VoiceStudioClient:
    transport = FakeTransport(mode, **kw)
    client = VoiceStudioClient(
        transport=transport,
        voices_dir=str(tmp_path / "voices"),
        out_dir=str(tmp_path / "out"))
    client._transport_fake = transport  # type: ignore[attr-defined]
    return client


# ── status / voices ───────────────────────────────────────────────
def test_status_ok(tmp_path):
    client = make_client(tmp_path)
    result = client.status()
    assert result["ok"] is True
    assert result["reachable"] is True
    assert result["version"] == "0.5.7"
    assert "mcp" in str(result.get("transports", ""))


def test_status_down_is_honest(tmp_path):
    client = make_client(tmp_path, "down")
    result = client.status()
    assert result["ok"] is False
    assert result["reachable"] is False
    assert result["kind"] == "unavailable"
    assert "docker run" in result["hint"]


def test_list_voices(tmp_path):
    client = make_client(tmp_path)
    result = client.list_voices()
    assert result["ok"] is True
    assert [v["id"] for v in result["voices"]] == ["alloy", "nova"]
    assert result["voices"][0]["license"] == "CC-BY-NC-4.0"
    assert result["voices"][1]["license"] == "Apache-2.0"
    assert result["engines"] == ["omnivoice", "cosyvoice"]


def test_engine_license_table():
    assert engine_license("omnivoice") == "CC-BY-NC-4.0"
    assert engine_license("k2-fsa/OmniVoice") == "CC-BY-NC-4.0"
    assert engine_license("something-new").startswith("unknown")


# ── speak ─────────────────────────────────────────────────────────
def test_speak_writes_wav(tmp_path):
    client = make_client(tmp_path)
    result = client.speak("Hello there", voice="alloy")
    assert result["ok"] is True
    assert result["format"] == "wav"
    assert result["voice"] == "alloy"
    out = Path(result["path"])
    assert out.exists()
    with wave.open(str(out), "rb") as w:
        assert w.getframerate() == 8000
    _, _, payload = client._transport_fake.calls[-1]
    assert payload["model"] == "tts-1"
    assert payload["input"] == "Hello there"
    assert payload["voice"] == "alloy"


def test_speak_empty_text_errors(tmp_path):
    client = make_client(tmp_path)
    with pytest.raises(VoiceStudioError):
        client.speak("   ")


def test_speak_error_json_body_is_not_audio(tmp_path):
    client = make_client(tmp_path, "error-body")
    with pytest.raises(VoiceStudioError) as excinfo:
        client.speak("hello")
    assert excinfo.value.kind == "engine"
    assert "model not loaded" in str(excinfo.value)
    assert not list(Path(client.out_dir).glob("*"))


def test_speak_non_audio_body_is_not_audio(tmp_path):
    client = make_client(tmp_path, "non-audio")
    with pytest.raises(VoiceStudioError) as excinfo:
        client.speak("hello")
    assert excinfo.value.kind == "bad_response"
    assert not list(Path(client.out_dir).glob("*"))


def test_speak_auth_401(tmp_path):
    client = make_client(tmp_path, "auth")
    with pytest.raises(VoiceStudioError) as excinfo:
        client.speak("hello")
    assert excinfo.value.kind == "auth"
    assert "NM_VOICESTUDIO_API_KEY" in excinfo.value.hint


# ── transcribe ────────────────────────────────────────────────────
def test_transcribe_verbose_json(tmp_path):
    client = make_client(tmp_path)
    audio = make_wav_file(tmp_path / "note.wav")
    result = client.transcribe(str(audio))
    assert result["ok"] is True
    assert result["text"] == "hello world"
    assert result["segments"][0]["start"] == 0.0
    assert result["duration"] == 1.5


def test_transcribe_text_format(tmp_path):
    client = make_client(tmp_path)
    audio = make_wav_file(tmp_path / "note.wav")
    result = client.transcribe(str(audio), response_format="text")
    assert result["text"] == "hello world"
    assert result["format"] == "text"


def test_transcribe_missing_file_errors(tmp_path):
    client = make_client(tmp_path)
    with pytest.raises(VoiceStudioError):
        client.transcribe(str(tmp_path / "nope.wav"))
    assert client._transport_fake.calls == []


# ── clone / design (discovered native ops) ────────────────────────
def test_clone_via_discovered_endpoint(tmp_path):
    client = make_client(tmp_path)
    ref = make_wav_file(tmp_path / "ref.wav")
    result = client.clone_voice("narrator", str(ref), language="en")
    assert result["ok"] is True
    assert result["profile_id"] == "vs-clone-1"
    assert result["license"] == "CC-BY-NC-4.0"
    recorded = client.recorded_profiles()
    assert len(recorded) == 1
    assert recorded[0]["kind"] == "clone"
    assert recorded[0]["reference"] == str(ref)
    kind, url, fields, files = client._transport_fake.calls[-1]
    assert kind == "POST-MULTIPART"
    assert url.endswith("/v1/voices/clone")
    assert fields["name"] == "narrator"
    assert files[0][0] == "reference"  # file field mapped from schema


def test_clone_no_operation_is_honest(tmp_path):
    client = make_client(tmp_path, schema=FAKE_SCHEMA_NO_NATIVE)
    ref = make_wav_file(tmp_path / "ref.wav")
    with pytest.raises(VoiceStudioError) as excinfo:
        client.clone_voice("narrator", str(ref))
    assert excinfo.value.kind == "unsupported"
    assert "clone in the app" in str(excinfo.value)


def test_clone_missing_reference_errors(tmp_path):
    client = make_client(tmp_path)
    with pytest.raises(VoiceStudioError):
        client.clone_voice("narrator", str(tmp_path / "nope.wav"))


def test_design_via_discovered_endpoint(tmp_path):
    client = make_client(tmp_path)
    result = client.design_voice("a warm radio host", name="host")
    assert result["ok"] is True
    assert result["profile_id"] == "vs-design-1"
    recorded = client.recorded_profiles()
    assert recorded[0]["kind"] == "design"
    assert recorded[0]["description"] == "a warm radio host"
    kind, url, payload = client._transport_fake.calls[-1]
    assert url.endswith("/v1/voices/design")
    assert payload["description"] == "a warm radio host"


def test_design_no_operation_is_honest(tmp_path):
    client = make_client(tmp_path, schema=FAKE_SCHEMA_NO_NATIVE)
    with pytest.raises(VoiceStudioError) as excinfo:
        client.design_voice("a warm radio host")
    assert excinfo.value.kind == "unsupported"


# ── dub ───────────────────────────────────────────────────────────
def test_dub_job_poll_to_done(tmp_path):
    dub_out = str(tmp_path / "dubbed.mp4")
    Path(dub_out).write_bytes(b"fake-video")
    client = make_client(tmp_path, dub_out=dub_out)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-source")
    result = client.dub_video(str(video), "es", voice="alloy",
                              poll_s=1.0)
    assert result["ok"] is True
    assert result["job_id"] == "job-1"
    assert result["status"] == "done"
    assert result["path"] == dub_out
    kind, url, fields, files = client._transport_fake.calls[1]
    assert url.endswith("/v1/dubbing/jobs")
    assert fields["target_language"] == "es"
    assert files[0][0] == "video"


def test_dub_timeout_returns_job_id(tmp_path):
    client = make_client(tmp_path, dub_never_done=True)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-source")
    result = client.dub_video(str(video), "es", timeout_s=0.01,
                              poll_s=1.0)
    assert result["ok"] is False
    assert result["job_id"] == "job-1"
    assert result["retryable"] is True


def test_dub_no_operation_is_honest(tmp_path):
    client = make_client(tmp_path, schema=FAKE_SCHEMA_NO_NATIVE)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-source")
    with pytest.raises(VoiceStudioError) as excinfo:
        client.dub_video(str(video), "es")
    assert excinfo.value.kind == "unsupported"


# ── narrate / store / discovery ───────────────────────────────────
def test_narrate_chunks_and_concats(tmp_path):
    client = make_client(tmp_path)
    text = (
        "The old lighthouse keeper climbed the iron stairs every "
        "evening, carrying a lamp that had burned for forty years. "
        "Below him the harbour lights came on one by one, and the "
        "fishing boats turned homeward through the cold grey water. "
        "He never once missed a night, not even when the winter gale "
        "tore the shutters from their hinges.")
    result = client.narrate(text, voice="alloy", chunk_chars=200)
    assert result["ok"] is True
    assert result["chunks"] >= 2
    assert result["seconds"] > 0
    out = Path(result["path"])
    assert out.exists()
    with wave.open(str(out), "rb") as w:
        assert w.getnframes() > 0


def test_chunk_text_never_splits_sentence():
    from nomorals.voice.voicestudio import _chunk_text
    chunks = _chunk_text("One here. Two here. Three here.", 20)
    assert chunks == ["One here. Two here.", "Three here."]


def test_recorded_profiles_roundtrip(tmp_path):
    client = make_client(tmp_path)
    assert client.recorded_profiles() == []
    client.record_profile("p1", "narrator", kind="clone",
                          engine="omnivoice")
    profiles = client.recorded_profiles()
    assert len(profiles) == 1
    assert profiles[0]["license"] == "CC-BY-NC-4.0"


def test_discover_caches_schema(tmp_path):
    client = make_client(tmp_path)
    assert client.resolve_operation("clone", "voice") is not None
    assert client.resolve_operation("design", "voice") is not None
    assert client._transport_fake._schema_calls == 1


def test_map_fields_drops_undeclared():
    from nomorals.voice.voicestudio import VoiceStudioClient
    mapped = VoiceStudioClient._map_fields(
        {"name": {"type": "string"}, "language": {"type": "string"}},
        {"name": "x", "surprise": "y"})
    assert mapped == {"name": "x"}


# ── tool surface ──────────────────────────────────────────────────
# One registry build for the whole file (it imports every tool
# module, ~17s) — the tool tests share it and only swap the client.
_TOOL_SPEC = None


def _tool_fn():
    global _TOOL_SPEC
    if _TOOL_SPEC is None:
        from types import SimpleNamespace

        from nomorals.tools.registry import ToolRegistry
        registry = ToolRegistry(SimpleNamespace())
        registry.register_builtins()
        spec = registry.get("voicestudio")
        assert spec is not None
        _TOOL_SPEC = (spec, registry)
    return _TOOL_SPEC


def test_tool_registered_810():
    spec, registry = _tool_fn()
    assert len(registry._tools) == 810
    assert "base_url" not in spec.parameters
    assert "url" not in spec.parameters


def test_tool_status_action(tmp_path, monkeypatch):
    from nomorals.tools import voicestudio as tool_module
    client = make_client(tmp_path)
    monkeypatch.setattr(tool_module, "_CLIENT", client)
    try:
        spec, _ = _tool_fn()
        result = spec.fn("status")
        assert result["ok"] is True
        assert result["version"] == "0.5.7"
    finally:
        monkeypatch.setattr(tool_module, "_CLIENT", None)


def test_tool_speak_action(tmp_path, monkeypatch):
    from nomorals.tools import voicestudio as tool_module
    client = make_client(tmp_path)
    monkeypatch.setattr(tool_module, "_CLIENT", client)
    try:
        spec, _ = _tool_fn()
        result = spec.fn("speak", text="hello", voice="alloy")
        assert result["ok"] is True
        assert Path(result["path"]).exists()
    finally:
        monkeypatch.setattr(tool_module, "_CLIENT", None)


def test_tool_down_returns_hint(tmp_path, monkeypatch):
    from nomorals.tools import voicestudio as tool_module
    client = make_client(tmp_path, "down")
    monkeypatch.setattr(tool_module, "_CLIENT", client)
    try:
        spec, _ = _tool_fn()
        result = spec.fn("status")
        assert result["ok"] is False
        assert "docker run" in result["hint"]
    finally:
        monkeypatch.setattr(tool_module, "_CLIENT", None)


def test_tool_unknown_action(tmp_path, monkeypatch):
    from nomorals.tools import voicestudio as tool_module
    client = make_client(tmp_path)
    monkeypatch.setattr(tool_module, "_CLIENT", client)
    try:
        spec, _ = _tool_fn()
        result = spec.fn("frobnicate")
        assert result["ok"] is False
        assert "status" in result["hint"]
    finally:
        monkeypatch.setattr(tool_module, "_CLIENT", None)

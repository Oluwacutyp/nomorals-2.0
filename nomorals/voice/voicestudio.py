"""VoiceStudio — Devon's client for a locally-running VoiceStudio backend.

VoiceStudio (debpalash/VoiceStudio, AGPL-3.0) is a fully-local voice
studio: voice cloning, voice design, video dubbing, dictation,
transcription and audiobooks in 646 languages. Devon talks to it as
an EXTERNAL TOOL over its local HTTP API — no VoiceStudio code is
vendored here (AGPL process boundary); this module is Devon's own
client, stdlib plus the house HttpClient only.

Stable surface (OpenAI-compatible, documented upstream):
  GET  /health
  GET  /openapi.json                      (discover native contracts)
  GET  /v1/audio/voices                  (profiles + engines)
  POST /v1/audio/speech                  (text -> audio)
  POST /v1/audio/transcriptions         (audio -> text)
  GET  /.well-known/voicestudio-speech  (transport discovery)
  MCP at /mcp (the brain can also speak MCP; this client uses REST)

Clone / design / dub / long-form jobs live behind NATIVE endpoints
that evolve independently — this client discovers them from the
running backend's /openapi.json at call time (never a hardcoded
guess) and fails honestly when the running version exposes none.

Licences, named per engine (upstream README + Model Manager): the
default OmniVoice weights are CC-BY-NC (noncommercial); other
engines carry their own licence, surfaced from the voices listing
when the backend reports one. Unknown engine -> "unknown — check
the Model Manager before client work".

The backend is loopback-only by design (127.0.0.1:3900). The base
URL comes from the environment (NM_VOICESTUDIO_URL) — never from a
tool parameter, so no caller can point this client at an arbitrary
host. Loopback calls need no key; a remote backend needs the bearer
(NM_VOICESTUDIO_API_KEY) and answers 401/403 without it.
"""

from __future__ import annotations

import json
import os
import re
import time
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..core.errors import NoMoralsError
from ..core.http import HttpClient, RequestError
from ..core.logging_setup import get_logger

__all__ = [
    "VoiceStudioClient",
    "VoiceStudioError",
    "default_client",
    "engine_license",
    "ENGINE_LICENSES",
]

_log = get_logger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:3900"
DEFAULT_MODEL = "tts-1"
DEFAULT_STT_MODEL = "whisper-1"

#: Licence per known engine, from the upstream README + Model
#: Manager. The default OmniVoice weights are noncommercial
#: (CC-BY-NC — the weights, not the code: k2-fsa/OmniVoice is
#: Apache-2.0 as code, CC-BY-NC as weights per its model card); the
#: bundled audio tokenizer adds a third layer (Boson Higgs Audio 2
#: Community License). CosyVoice 3 is Apache-2.0. Any engine not
#: listed here reports "unknown" and the caller must check the Model
#: Manager before client work.
ENGINE_LICENSES: dict[str, str] = {
    "omnivoice": "CC-BY-NC-4.0",
    "k2-fsa/omnivoice": "CC-BY-NC-4.0",
    "omni-voice": "CC-BY-NC-4.0",
    "cosyvoice": "Apache-2.0",
}

START_HINT = (
    "Start the VoiceStudio app, or run: docker run -d "
    "-p 127.0.0.1:3900:3900 -v omnivoice-data:/app/omnivoice_data "
    "ghcr.io/debpalash/voicestudio:stable — then retry."
)

#: Audio magic -> format. mp4/m4a carry "ftyp" at offset 4.
_AUDIO_MAGIC: tuple[tuple[bytes, str, int], ...] = (
    (b"RIFF", "wav", 0),
    (b"ID3", "mp3", 0),
    (b"\xff\xfb", "mp3", 0),
    (b"\xff\xf3", "mp3", 0),
    (b"\xff\xf2", "mp3", 0),
    (b"fLaC", "flac", 0),
    (b"OggS", "ogg", 0),
    (b"ftyp", "mp4", 4),
)

_AUDIO_MIME_BY_EXT = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".webm": "audio/webm",
    ".aac": "audio/aac",
}

_DUB_DONE = {"done", "completed", "complete", "succeeded", "success",
             "finished"}
_DUB_FAILED = {"failed", "error", "errored", "cancelled", "canceled"}


def engine_license(engine: str) -> str:
    """Licence for a named engine, or 'unknown' with the check named."""
    key = (engine or "").strip().lower()
    if key in ENGINE_LICENSES:
        return ENGINE_LICENSES[key]
    if "omnivoice" in key:
        return ENGINE_LICENSES["omnivoice"]
    return "unknown — check the Model Manager before client work"


def _default_voices_dir() -> str:
    """Same rule as the voice catalogue (catalogue.py): env override,
    else the user-level default."""
    override = os.environ.get("NOMORALS_VOICES_DIR", "").strip()
    if override:
        return os.path.expanduser(override)
    return os.path.join(os.path.expanduser("~"), ".config",
                        "nomorals", "voices")


def _default_out_dir() -> str:
    override = os.environ.get("NM_VOICESTUDIO_OUT_DIR", "").strip()
    if override:
        return os.path.expanduser(override)
    home = os.environ.get("NM_HOME", "").strip()
    base = Path(os.path.expanduser(home)) if home else Path.home() / ".nomorals"
    return str(base / "audio" / "voicestudio")


def _sniff_audio(data: bytes) -> str | None:
    for magic, fmt, off in _AUDIO_MAGIC:
        if data[off:off + len(magic)] == magic:
            return fmt
    return None


class VoiceStudioError(NoMoralsError):
    """A VoiceStudio failure that names its kind and the next step."""

    def __init__(self, message: str, *, kind: str = "error",
                 hint: str = "", retryable: bool = False):
        super().__init__(message)
        self.kind = kind
        self.hint = hint
        self.retryable = retryable

    def as_dict(self) -> dict[str, Any]:
        return {"ok": False, "error": str(self), "kind": self.kind,
                "hint": self.hint, "retryable": self.retryable}


class VoiceStudioClient:
    """HTTP client for a locally-running VoiceStudio backend.

    ``transport`` is injectable (anything with get/post_json/
    post_multipart/request like the house HttpClient) so tests run
    without a live backend. The default transport is the house
    HttpClient with loopback access enabled and proxying forced off:
    this backend lives on 127.0.0.1 by design and its traffic never
    leaves the machine.
    """

    def __init__(self, base_url: str | None = None,
                 api_key: str | None = None,
                 timeout: float = 120.0,
                 transport: Any = None,
                 voices_dir: str | None = None,
                 out_dir: str | None = None):
        self.base_url = (base_url
                         or os.environ.get("NM_VOICESTUDIO_URL", "")
                         .strip() or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = (api_key
                        or os.environ.get("NM_VOICESTUDIO_API_KEY", "")
                        .strip())
        self.timeout = timeout
        self.voices_dir = voices_dir or _default_voices_dir()
        self.out_dir = out_dir or _default_out_dir()
        self._schema: dict[str, Any] | None = None
        if transport is not None:
            self._transport = transport
        else:
            self._transport = HttpClient(
                timeout=timeout, allow_private_ips=True,
                force_direct=True)

    # ── plumbing ──────────────────────────────────────────────────
    def _headers(self) -> dict[str, str]:
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _classify(self, exc: Exception, where: str) -> VoiceStudioError:
        msg = str(exc)
        code = getattr(exc, "code", None) or getattr(exc, "status", None)
        lowered = msg.lower()
        if code in (401, 403) or "401" in lowered or "403" in lowered:
            return VoiceStudioError(
                f"VoiceStudio refused {where}: not authorized "
                f"(HTTP {code or '401/403'})",
                kind="auth",
                hint="Set NM_VOICESTUDIO_API_KEY for a remote backend; "
                     "loopback needs no key.",
                retryable=False)
        if code == 404:
            return VoiceStudioError(
                f"VoiceStudio has no {where} endpoint (HTTP 404)",
                kind="unsupported",
                hint="The running backend version may not expose this "
                     "operation — check its /openapi.json.",
                retryable=False)
        if (isinstance(exc, RequestError)
                or "timed out" in lowered or "timeout" in lowered
                or "connection refused" in lowered
                or "failed to establish" in lowered
                or "name or service not known" in lowered
                or type(exc).__name__ in ("TimeoutError", "TimeoutError_")):
            return VoiceStudioError(
                f"VoiceStudio is not reachable at {self.base_url} "
                f"({where}: {msg})",
                kind="unavailable", hint=START_HINT, retryable=True)
        return VoiceStudioError(
            f"VoiceStudio {where} failed: {msg}",
            kind="engine", retryable=True)

    def _get(self, path: str, *, timeout: float | None = None):
        try:
            return self._transport.get(
                self._url(path), headers=self._headers(),
                timeout=timeout if timeout is not None else self.timeout)
        except Exception as exc:  # noqa: BLE001 — classified below
            raise self._classify(exc, f"GET {path}") from exc

    def _post_json(self, path: str, payload: dict[str, Any], *,
                   timeout: float | None = None):
        try:
            return self._transport.post_json(
                self._url(path), payload, headers=self._headers(),
                timeout=timeout if timeout is not None else self.timeout)
        except Exception as exc:  # noqa: BLE001 — classified below
            raise self._classify(exc, f"POST {path}") from exc

    def _post_multipart(self, path: str, *,
                        fields: dict[str, str] | None = None,
                        files: list[tuple[str, str, str]] | None = None,
                        timeout: float | None = None):
        try:
            return self._transport.post_multipart(
                self._url(path), fields=fields or {},
                files=files or [], headers=self._headers(),
                timeout=timeout if timeout is not None else self.timeout)
        except Exception as exc:  # noqa: BLE001 — classified below
            raise self._classify(exc, f"POST {path}") from exc

    # ── stable surface ────────────────────────────────────────────
    def status(self) -> dict[str, Any]:
        """Backend health. Never raises for a down backend."""
        try:
            r = self._get("/health", timeout=10.0)
        except VoiceStudioError as exc:
            if exc.kind == "unavailable":
                return {"ok": False, "reachable": False,
                        "error": str(exc), "kind": "unavailable",
                        "hint": START_HINT, "base_url": self.base_url}
            raise
        info: dict[str, Any] = {"ok": True, "reachable": True,
                                 "base_url": self.base_url}
        try:
            body = r.json()
            if isinstance(body, dict):
                info["version"] = (body.get("version")
                                   or body.get("app_version") or "")
                info["status"] = body.get("status", "")
        except Exception:  # noqa: BLE001 — health body is best-effort
            info["status"] = r.text[:200]
        try:
            wk = self._get("/.well-known/voicestudio-speech",
                           timeout=10.0).json()
            if isinstance(wk, dict):
                info["transports"] = wk
        except Exception:  # noqa: BLE001 — discovery is best-effort
            pass
        return info

    def list_voices(self) -> dict[str, Any]:
        """Voice profiles + engines from the running backend."""
        r = self._get("/v1/audio/voices", timeout=10.0)
        try:
            body = r.json()
        except Exception as exc:  # noqa: BLE001 — JSON is the contract
            raise VoiceStudioError(
                "VoiceStudio /v1/audio/voices did not return JSON",
                kind="bad_response", retryable=False) from exc
        if isinstance(body, dict):
            raw = body.get("voices") or body.get("data") or []
            engines = body.get("engines") or []
        elif isinstance(body, list):
            raw, engines = body, []
        else:
            raw, engines = [], []
        voices = []
        for entry in raw if isinstance(raw, list) else []:
            if not isinstance(entry, dict):
                continue
            vid = (entry.get("id") or entry.get("voice_id")
                   or entry.get("name") or "")
            engine = (entry.get("engine") or entry.get("model") or "")
            voices.append({
                "id": str(vid),
                "name": str(entry.get("name") or vid),
                "engine": str(engine),
                "language": str(entry.get("language") or ""),
                "license": (entry.get("license")
                            or engine_license(str(engine))),
            })
        return {"ok": True, "voices": voices,
                "engines": engines if isinstance(engines, list) else []}

    def speak(self, text: str, voice: str = "",
              model: str = DEFAULT_MODEL,
              response_format: str = "wav",
              out_path: str | None = None) -> dict[str, Any]:
        """Text -> audio file via the backend's speech endpoint.

        The response bytes are probed before anything is written:
        an error body saved as audio is a bug, not a file (upstream
        warns the backend can do exactly that).
        """
        text = (text or "").strip()
        if not text:
            raise VoiceStudioError("speak needs non-empty text",
                                   kind="error", retryable=False)
        payload: dict[str, Any] = {"model": model, "input": text,
                                   "response_format": response_format}
        if voice:
            payload["voice"] = voice
        r = self._post_json("/v1/audio/speech", payload)
        headers = {k.lower(): v for k, v in (r.headers or {}).items()}
        ctype = headers.get("content-type", "").lower()
        if "application/json" in ctype:
            try:
                err = r.json()
                detail = (err.get("error") if isinstance(err, dict)
                          else err)
            except Exception:  # noqa: BLE001 — error body is best-effort
                detail = r.text[:300]
            raise VoiceStudioError(
                f"VoiceStudio refused to speak: {detail}",
                kind="engine", retryable=False)
        fmt = _sniff_audio(r.body)
        if fmt is None:
            # Raw PCM has no magic bytes; trust it only when the
            # backend said so and the format asked for was pcm.
            if response_format == "pcm" and ctype.startswith("audio/"):
                fmt = "pcm"
            else:
                raise VoiceStudioError(
                    "VoiceStudio returned bytes that are not audio "
                    f"({len(r.body)} bytes, Content-Type: "
                    f"{ctype or 'unknown'})",
                    kind="bad_response", retryable=False)
        out = Path(out_path) if out_path else self._out_path(fmt)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(r.body)
        return {"ok": True, "path": str(out), "bytes": len(r.body),
                "format": fmt, "voice": voice or "",
                "model": model}

    def _out_path(self, fmt: str) -> Path:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        return Path(self.out_dir) / f"voicestudio-{stamp}.{fmt}"

    def transcribe(self, audio_path: str,
                   model: str = DEFAULT_STT_MODEL,
                   response_format: str = "verbose_json"
                   ) -> dict[str, Any]:
        """Audio file -> transcript via the backend's STT endpoint."""
        path = Path(audio_path)
        if not path.exists():
            raise VoiceStudioError(
                f"transcribe: no such audio file: {audio_path}",
                kind="error", retryable=False)
        mime = _AUDIO_MIME_BY_EXT.get(path.suffix.lower(),
                                      "application/octet-stream")
        r = self._post_multipart(
            "/v1/audio/transcriptions",
            fields={"model": model, "response_format": response_format},
            files=[("file", str(path), mime)])
        if response_format in ("json", "verbose_json"):
            try:
                body = r.json()
            except Exception as exc:  # noqa: BLE001 — JSON is the contract
                raise VoiceStudioError(
                    "VoiceStudio transcription did not return JSON",
                    kind="bad_response", retryable=False) from exc
            if not isinstance(body, dict):
                raise VoiceStudioError(
                    "VoiceStudio transcription returned an unexpected shape",
                    kind="bad_response", retryable=False)
            return {"ok": True, "text": str(body.get("text", "")),
                    "segments": body.get("segments") or [],
                    "duration": body.get("duration"),
                    "format": response_format}
        return {"ok": True, "text": r.text,
                "format": response_format}

    def narrate(self, text: str, voice: str = "",
                model: str = DEFAULT_MODEL,
                out_path: str | None = None,
                chunk_chars: int = 1500) -> dict[str, Any]:
        """Long text -> one wav: chunked speak + PCM concat.

        Chunks never split a sentence; a single over-long sentence is
        hard-split. All chunks are wav, so the join is stdlib-only.
        """
        text = (text or "").strip()
        if not text:
            raise VoiceStudioError("narrate needs non-empty text",
                                   kind="error", retryable=False)
        chunks = _chunk_text(text, max(200, chunk_chars))
        out = Path(out_path) if out_path else self._out_path("wav")
        out.parent.mkdir(parents=True, exist_ok=True)
        total_frames = 0
        params = None
        with wave.open(str(out), "wb") as dest:
            for i, chunk in enumerate(chunks):
                part = self.speak(chunk, voice=voice, model=model,
                                  response_format="wav",
                                  out_path=str(out) + f".part{i}.wav")
                with wave.open(part["path"], "rb") as src:
                    if params is None:
                        params = (src.getnchannels(), src.getsampwidth(),
                                  src.getframerate())
                        dest.setnchannels(params[0])
                        dest.setsampwidth(params[1])
                        dest.setframerate(params[2])
                    elif (src.getnchannels(), src.getsampwidth(),
                          src.getframerate()) != params:
                        raise VoiceStudioError(
                            "VoiceStudio wav params changed mid-narration",
                            kind="bad_response", retryable=False)
                    frames = src.readframes(src.getnframes())
                    dest.writeframes(frames)
                    total_frames += src.getnframes()
                Path(part["path"]).unlink(missing_ok=True)
        seconds = (total_frames / params[2]) if params else 0.0
        return {"ok": True, "path": str(out), "chunks": len(chunks),
                "seconds": round(seconds, 2), "format": "wav",
                "voice": voice or "", "model": model}

    # ── discovery (native endpoints) ──────────────────────────────
    def discover(self, *, refresh: bool = False) -> dict[str, Any]:
        """The running backend's /openapi.json, cached per client."""
        if self._schema is None or refresh:
            r = self._get("/openapi.json", timeout=10.0)
            try:
                schema = r.json()
            except Exception as exc:  # noqa: BLE001 — JSON is the contract
                raise VoiceStudioError(
                    "VoiceStudio /openapi.json did not return JSON",
                    kind="bad_response", retryable=False) from exc
            if not isinstance(schema, dict):
                raise VoiceStudioError(
                    "VoiceStudio /openapi.json returned an unexpected shape",
                    kind="bad_response", retryable=False)
            self._schema = schema
        return self._schema

    def resolve_operation(self, *verbs: str,
                          need_path_param: bool = False
                          ) -> tuple[str, str, dict] | None:
        """Find a native operation by verb in id/tags/summary.

        Returns (method, path, operation) or None. POST wins over
        other methods when several match. ``need_path_param`` keeps
        only operations whose path carries a ``{param}`` (a job
        lookup, never a job creation).
        """
        schema = self.discover()
        paths = schema.get("paths") or {}
        candidates = []
        for path, ops in paths.items():
            if not isinstance(ops, dict):
                continue
            if need_path_param and "{" not in path:
                continue
            for method, op in ops.items():
                if not isinstance(op, dict):
                    continue
                hay = " ".join([
                    str(op.get("operationId", "")),
                    str(op.get("summary", "")),
                    " ".join(str(t) for t in op.get("tags", []) or []),
                    path,
                ]).lower()
                if any(v.lower() in hay for v in verbs):
                    candidates.append((method.upper(), path, op))
        if not candidates:
            return None
        # Rank by how many verbs matched (the operation that names
        # the job beats one that merely shares a word), then POST
        # first, then schema order. Without the score, "design"
        # resolves to the clone op — both are POST and both mention
        # "voice".
        def _rank(candidate: tuple[str, str, dict]) -> tuple[int, int]:
            _, path, op = candidate
            hay = " ".join([
                str(op.get("operationId", "")),
                str(op.get("summary", "")),
                " ".join(str(t) for t in op.get("tags", []) or []),
                path,
            ]).lower()
            hits = sum(1 for v in verbs if v.lower() in hay)
            return (-hits, 0 if candidate[0] == "POST" else 1)
        candidates.sort(key=_rank)
        return candidates[0]

    def _operation_props(self, op: dict) -> dict[str, Any]:
        """Declared requestBody properties (JSON or multipart)."""
        body = (op.get("requestBody") or {}).get("content") or {}
        for ctype in ("multipart/form-data", "application/json"):
            schema = (body.get(ctype) or {}).get("schema") or {}
            props = schema.get("properties") or {}
            if props:
                return props
        return {}

    @staticmethod
    def _map_fields(declared: dict[str, Any],
                    values: dict[str, Any]) -> dict[str, Any]:
        """Map semantic field names onto the operation's declared
        properties: exact, case-insensitive, then substring. Fields
        with no declared match are dropped (a strict validator would
        422 them) and the drop is logged."""
        if not declared:
            return dict(values)
        lowered = {k.lower(): k for k in declared}
        mapped: dict[str, Any] = {}
        for key, value in values.items():
            target = None
            if key in declared:
                target = key
            elif key.lower() in lowered:
                target = lowered[key.lower()]
            else:
                for name in declared:
                    if key.lower() in name.lower() or \
                            name.lower() in key.lower():
                        target = name
                        break
            if target is None:
                _log.warning("voicestudio: field %r not declared by "
                             "the backend operation; dropping it", key)
                continue
            mapped[target] = value
        return mapped

    @staticmethod
    def _file_field(declared: dict[str, Any]) -> str | None:
        for name, spec in declared.items():
            if isinstance(spec, dict) and spec.get("format") == "binary":
                return name
        return None

    def _fill_path(self, path: str, values: dict[str, Any]) -> str:
        def sub(match: re.Match) -> str:
            key = match.group(1)
            if key in values:
                return str(values[key])
            raise VoiceStudioError(
                f"VoiceStudio operation path needs {{{key}}} and the "
                "call did not supply it",
                kind="unsupported", retryable=False)
        return re.sub(r"\{([^}]+)\}", sub, path)

    def _call_native(self, op: tuple[str, str, dict], *,
                     json_body: dict[str, Any] | None = None,
                     form_fields: dict[str, str] | None = None,
                     files: list[tuple[str, str, str]] | None = None,
                     timeout: float | None = None):
        method, path, spec = op
        props = self._operation_props(spec)
        path_values = dict(json_body or {})
        path_values.update(form_fields or {})
        url_path = self._fill_path(path, path_values)
        if files:
            field = self._file_field(props)
            if field is None:
                raise VoiceStudioError(
                    "VoiceStudio operation declares no file upload "
                    f"field ({method} {path})",
                    kind="unsupported", retryable=False)
            mapped_files = [(field, fpath, mime)
                            for (_name, fpath, mime) in files]
            mapped_fields = self._map_fields(props, form_fields or {})
            return self._post_multipart(url_path, fields=mapped_fields,
                                        files=mapped_files,
                                        timeout=timeout)
        if json_body is not None:
            mapped = self._map_fields(props, json_body)
            if method == "POST":
                return self._post_json(url_path, mapped, timeout=timeout)
            try:
                return self._transport.request(
                    method, self._url(url_path),
                    data=json.dumps(mapped).encode("utf-8"),
                    headers={"Content-Type": "application/json",
                             **self._headers()},
                    timeout=timeout if timeout is not None
                    else self.timeout)
            except Exception as exc:  # noqa: BLE001 — classified below
                raise self._classify(
                    exc, f"{method} {url_path}") from exc
        try:
            return self._transport.request(
                method, self._url(url_path), headers=self._headers(),
                timeout=timeout if timeout is not None else self.timeout)
        except Exception as exc:  # noqa: BLE001 — classified below
            raise self._classify(exc, f"{method} {url_path}") from exc

    @staticmethod
    def _body_dict(response: Any) -> dict[str, Any]:
        try:
            body = response.json()
        except Exception as exc:  # noqa: BLE001 — JSON is the contract
            raise VoiceStudioError(
                "VoiceStudio operation did not return JSON",
                kind="bad_response", retryable=False) from exc
        if not isinstance(body, dict):
            raise VoiceStudioError(
                "VoiceStudio operation returned an unexpected shape",
                kind="bad_response", retryable=False)
        return body

    # ── clone / design / dub (native, discovered) ─────────────────
    def clone_voice(self, name: str, reference_path: str,
                    language: str = "",
                    description: str = "") -> dict[str, Any]:
        """Clone a voice from a reference recording.

        Only voices the owner is authorized to use — the reference
        path is recorded in the profile store, so every clone is
        auditable. The backend enforces its own consent flow in the
        app; the API path keeps the receipt.
        """
        name = (name or "").strip()
        if not name:
            raise VoiceStudioError("clone needs a voice name",
                                   kind="error", retryable=False)
        ref = Path(reference_path)
        if not ref.exists():
            raise VoiceStudioError(
                f"clone: no such reference audio: {reference_path}",
                kind="error", retryable=False)
        if ref.stat().st_size == 0:
            raise VoiceStudioError(
                f"clone: reference audio is empty: {reference_path}",
                kind="error", retryable=False)
        if (ref.suffix.lower() not in _AUDIO_MIME_BY_EXT
                and _sniff_audio(ref.read_bytes()[:16]) is None):
            raise VoiceStudioError(
                f"clone: reference does not look like audio: "
                f"{reference_path}",
                kind="error", retryable=False)
        op = self.resolve_operation("clone", "voice", "profile")
        if op is None:
            raise VoiceStudioError(
                "VoiceStudio exposes no voice-clone operation in its "
                "/openapi.json — this backend version cannot clone "
                "from the API (clone in the app instead)",
                kind="unsupported",
                hint="Open the VoiceStudio app, clone there, then "
                     "use the saved profile name with speak.",
                retryable=False)
        mime = _AUDIO_MIME_BY_EXT.get(ref.suffix.lower(),
                                      "application/octet-stream")
        fields: dict[str, str] = {"name": name}
        if language:
            fields["language"] = language
        if description:
            fields["description"] = description
        r = self._call_native(
            op, form_fields=fields,
            files=[("reference", str(ref), mime)])
        body = self._body_dict(r)
        profile_id = str(body.get("id") or body.get("voice_id")
                         or body.get("profile_id") or name)
        engine = str(body.get("engine") or body.get("model") or "")
        record = self.record_profile(
            profile_id, name, kind="clone", engine=engine,
            reference=str(ref), description=description)
        return {"ok": True, "profile_id": profile_id, "name": name,
                "engine": engine,
                "license": engine_license(engine),
                "recorded": record}

    def design_voice(self, description: str, name: str = "",
                     language: str = "",
                     sample_text: str = "") -> dict[str, Any]:
        """Design a voice from a description, via the backend."""
        description = (description or "").strip()
        if not description:
            raise VoiceStudioError("design needs a description",
                                   kind="error", retryable=False)
        op = self.resolve_operation("design", "voice")
        if op is None:
            raise VoiceStudioError(
                "VoiceStudio exposes no voice-design operation in its "
                "/openapi.json — this backend version cannot design "
                "from the API (design in the app instead)",
                kind="unsupported",
                hint="Open the VoiceStudio app, design there, then "
                     "use the saved profile name with speak.",
                retryable=False)
        body_in: dict[str, Any] = {"description": description}
        if name:
            body_in["name"] = name
        if language:
            body_in["language"] = language
        if sample_text:
            body_in["sample_text"] = sample_text
        r = self._call_native(op, json_body=body_in)
        body = self._body_dict(r)
        profile_id = str(body.get("id") or body.get("voice_id")
                         or body.get("profile_id") or name or description[:24])
        engine = str(body.get("engine") or body.get("model") or "")
        record = self.record_profile(
            profile_id, name or profile_id, kind="design",
            engine=engine, description=description)
        return {"ok": True, "profile_id": profile_id,
                "name": name or profile_id, "engine": engine,
                "license": engine_license(engine),
                "recorded": record}

    def dub_video(self, video_path: str, target_language: str,
                  voice: str = "", timeout_s: float = 600.0,
                  poll_s: float = 5.0) -> dict[str, Any]:
        """Dub a video into another language. Async underneath:
        creates the job, then polls until it lands or the timeout
        hits — a timeout returns the job id, never a fake file."""
        video = Path(video_path)
        if not video.exists():
            raise VoiceStudioError(
                f"dub: no such video: {video_path}",
                kind="error", retryable=False)
        if not (target_language or "").strip():
            raise VoiceStudioError("dub needs a target language",
                                   kind="error", retryable=False)
        op = self.resolve_operation("dub")
        if op is None:
            raise VoiceStudioError(
                "VoiceStudio exposes no dubbing operation in its "
                "/openapi.json — this backend version cannot dub "
                "from the API (dub in the app instead)",
                kind="unsupported",
                hint="Open the VoiceStudio app, dub there, then "
                     "collect the finished video.",
                retryable=False)
        mime = {".mp4": "video/mp4", ".mov": "video/quicktime",
                ".mkv": "video/x-matroska",
                ".webm": "video/webm"}.get(video.suffix.lower(),
                                           "video/mp4")
        fields: dict[str, str] = {
            "target_language": target_language}
        if voice:
            fields["voice"] = voice
        r = self._call_native(
            op, form_fields=fields,
            files=[("video", str(video), mime)],
            timeout=timeout_s)
        body = self._body_dict(r)
        job_id = str(body.get("job_id") or body.get("id") or "")
        if not job_id:
            # Synchronous dub: the finished file came back inline.
            out = _output_path(body)
            if out:
                return {"ok": True, "job_id": "", "status": "done",
                        "path": self._resolve_output(out)}
            return {"ok": True, "job_id": "", "status": "done",
                    "detail": body}
        return self.poll_dub(job_id, timeout_s=timeout_s, poll_s=poll_s)

    def poll_dub(self, job_id: str, timeout_s: float = 600.0,
                 poll_s: float = 5.0) -> dict[str, Any]:
        """Poll a dub job until it lands, fails, or the timeout hits."""
        if not (job_id or "").strip():
            raise VoiceStudioError("dub_status needs a job id",
                                   kind="error", retryable=False)
        op = self.resolve_operation("job", "status", "progress",
                                    need_path_param=True)
        deadline = time.monotonic() + max(1.0, timeout_s)
        last: dict[str, Any] = {"status": "submitted"}
        while True:
            if op is None:
                return {"ok": True, "job_id": job_id,
                        "status": "submitted",
                        "note": "No job-status endpoint discovered in "
                                "/openapi.json — check the finished "
                                "video in the VoiceStudio app."}
            try:
                path = self._fill_path(op[1], {"job_id": job_id,
                                               "id": job_id})
                r = self._transport.request(
                    op[0], self._url(path), headers=self._headers(),
                    timeout=30.0)
                last = self._body_dict(r)
            except VoiceStudioError:
                raise
            except Exception as exc:  # noqa: BLE001 — classified below
                raise self._classify(exc, "dub status") from exc
            status = str(last.get("status") or last.get("state")
                         or "").lower()
            if status in _DUB_DONE:
                out = _output_path(last)
                if out:
                    return {"ok": True, "job_id": job_id,
                            "status": status,
                            "path": self._resolve_output(out)}
                return {"ok": True, "job_id": job_id,
                        "status": status, "detail": last}
            if status in _DUB_FAILED:
                raise VoiceStudioError(
                    f"VoiceStudio dub job {job_id} failed: "
                    f"{last.get('error') or status}",
                    kind="engine", retryable=False)
            if time.monotonic() >= deadline:
                return {"ok": False, "job_id": job_id,
                        "status": status or "running",
                        "error": "dub still running after "
                                 f"{timeout_s:g}s — retry dub_status "
                                 "with this job id",
                        "kind": "timeout", "retryable": True}
            time.sleep(max(1.0, poll_s))

    # ── local profile store ───────────────────────────────────────
    def _resolve_output(self, value: str) -> str:
        """A dub output: a local path passes through; an http(s) URL
        is downloaded into the out dir (the file is the result)."""
        if value.startswith(("http://", "https://")):
            dest = Path(self.out_dir) / Path(value.split("?")[0]).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            download = getattr(self._transport, "download", None)
            if callable(download):
                return str(download(value, dest))
            return value
        return value

    def _profiles_path(self) -> Path:
        return Path(self.voices_dir) / "voicestudio_profiles.json"

    def record_profile(self, profile_id: str, name: str, *,
                       kind: str, engine: str = "",
                       reference: str = "",
                       description: str = "") -> dict[str, Any]:
        """Record a cloned/designed profile locally (audit trail)."""
        path = self._profiles_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            store = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(store, list):
                store = []
        except (OSError, ValueError):
            store = []
        record = {
            "profile_id": profile_id, "name": name, "kind": kind,
            "engine": engine, "license": engine_license(engine),
            "reference": reference, "description": description,
            "created_at": datetime.now(UTC).isoformat(),
        }
        store = [r for r in store if r.get("profile_id") != profile_id]
        store.append(record)
        path.write_text(json.dumps(store, indent=2), encoding="utf-8")
        return record

    def recorded_profiles(self) -> list[dict[str, Any]]:
        """Profiles this client has recorded (empty when none)."""
        try:
            store = json.loads(
                self._profiles_path().read_text(encoding="utf-8"))
            return store if isinstance(store, list) else []
        except (OSError, ValueError):
            return []


def _output_path(body: dict[str, Any]) -> str | None:
    for key in ("output_path", "output", "video", "file", "path"):
        value = body.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text into chunks that never split a sentence."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(sentence), max_chars):
                chunks.append(sentence[i:i + max_chars])
            continue
        if current and len(current) + 1 + len(sentence) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks or [text]


def default_client() -> VoiceStudioClient:
    """A client from the environment (base URL + key)."""
    return VoiceStudioClient()

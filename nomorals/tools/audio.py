"""Voice I/O: text-to-speech and speech-to-text.

Two planes, both optional — the system runs with no audio at all:

**TTS** tries local engines first (no account, no network):
``espeak-ng`` (Termux: ``pkg install espeak-ng``), ``pico2wave``, macOS ``say``,
the ``edge-tts`` Python package, then an OpenAI-compatible ``/audio/speech``
endpoint when one is configured.

**STT** mirrors that: an OpenAI-compatible ``/audio/transcriptions``
endpoint (whisper-1 et al.) or a local ``whisper.cpp`` binary.

Generated audio lands in ``<NM_HOME>/<audio.audio_dir>`` and the ``speak``
tool returns the path — the chat layer decides whether to send it as a
file (``gateway.send_file``) or just report the path.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
import wave
from pathlib import Path
from typing import Any

from ..core.errors import ToolError
from ..core.logging_setup import get_logger
from ..core.policy import Capability

__all__ = ["detect_tts_engines", "tts", "stt", "audio_info", "register"]

_log = get_logger(__name__)

_TTS_TIMEOUT = 120.0
_STT_TIMEOUT = 300.0

#: engine name -> how to invoke it. argv is a template; {out} and {voice} are
#: substituted, the text is always a final argv element (never a shell string).
_ESPEAK = "espeak-ng"
_PICO = "pico2wave"
_SAY = "say"


def detect_tts_engines() -> list[str]:
    """Which TTS engines are actually usable on this machine, in priority order."""
    found: list[str] = []
    if shutil.which(_ESPEAK):
        found.append("espeak_ng")
    if shutil.which(_PICO):
        found.append("pico2wave")
    if shutil.which(_SAY):
        found.append("say")
    try:
        import edge_tts  # noqa: F401

        found.append("edge_tts")
    except ImportError:  # noqa: E103 - optional TTS backend probe
        pass
    return found


def detect_stt(provider: str = "auto", *, base_url: str = "", api_key: str = "") -> str | None:
    """Resolve which STT provider will actually work; None = nothing available."""
    if provider and provider != "auto":
        if provider == "openai_compat":
            return "openai_compat" if (base_url or "") and (api_key or "") else None
        if provider == "whisper_cpp":
            return "whisper_cpp" if _whisper_cpp_binary() else None
        return None
    # auto: prefer a configured endpoint (higher quality), fall back to local
    if (base_url or "") and (api_key or ""):
        return "openai_compat"
    if _whisper_cpp_binary():
        return "whisper_cpp"
    return None


def _whisper_cpp_binary() -> str | None:
    env_bin = os.environ.get("NM_WHISPER_CPP_BIN", "").strip()
    if env_bin and Path(env_bin).exists():
        return env_bin
    for name in ("whisper-cpp", "whisper.cpp", "main"):
        path = shutil.which(name)
        if path:
            return path
    return None


def _whisper_cpp_model() -> str | None:
    model = os.environ.get("NM_WHISPER_CPP_MODEL", "").strip()
    if model and Path(model).exists():
        return model
    return None


# ── TTS ──────────────────────────────────────────────────────────────────────


def _tts_local(text: str, engine: str, voice: str, out: Path) -> None:
    if engine == "espeak_ng":
        argv = [_ESPEAK]
        if voice:
            argv += ["-v", voice]
        argv += ["-w", str(out), text]
    elif engine == "pico2wave":
        argv = [_PICO, "-w", str(out)]
        if voice:
            argv += ["-w", str(out)]  # pico2wave has no voice arg; keep single -w
        argv = [_PICO, "-w", str(out), text]
    elif engine == "say":
        argv = [_SAY, "-o", str(out), "aiff"]
        if voice:
            argv += ["-v", voice]
        argv.append(text)
    else:
        raise ToolError(f"unknown local TTS engine {engine!r}")
    proc = subprocess.run(argv, capture_output=True, timeout=_TTS_TIMEOUT, check=False)
    if proc.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        raise ToolError(
            f"{engine} failed (exit {proc.returncode}): "
            f"{(proc.stderr or proc.stdout or b'').decode('utf-8', 'replace')[:300]}"
        )


def _tts_edge(text: str, voice: str, out: Path) -> None:
    import asyncio

    import edge_tts

    voice_id = voice or None
    if voice_id:
        coro = edge_tts.Communicate(text, voice_id).save(str(out))
    else:
        coro = edge_tts.Communicate(text).save(str(out))
    asyncio.run(asyncio.wait_for(coro, timeout=_TTS_TIMEOUT))
    if not out.exists() or out.stat().st_size == 0:
        raise ToolError("edge_tts produced no audio")


def _tts_openai_compat(text: str, voice: str, out: Path, *, base_url: str, api_key: str, model: str) -> None:
    from ..core.http import HttpClient

    base = base_url.rstrip("/")
    if not base.endswith("/v1") and "/v1" not in base:
        base = base + "/v1"
    client = HttpClient(timeout=_TTS_TIMEOUT)
    response = client.post_json(
        f"{base}/audio/speech",
        {"model": model or "tts-1", "voice": voice or "alloy", "input": text[:4000]},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    if not response.ok:
        raise ToolError(f"TTS API {response.status}: {response.text[:300]}")
    out.write_bytes(response.body)
    if out.stat().st_size == 0:
        raise ToolError("TTS API returned an empty file")


def tts(
    text: str,
    *,
    engine: str = "auto",
    voice: str = "",
    out_dir: str | Path,
    base_url: str = "",
    api_key: str = "",
    model: str = "",
    timeout: float = _TTS_TIMEOUT,
) -> dict[str, Any]:
    """Synthesize speech to a file. Returns {path, engine, format, bytes, seconds}."""
    text = (text or "").strip()
    if not text:
        raise ToolError("tts needs text")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    engines = detect_tts_engines()
    if engine in ("", "auto"):
        engine = engines[0] if engines else ("openai_compat" if base_url and api_key else "")
        if not engine:
            raise ToolError(
                "no TTS engine available — Termux: pkg install espeak-ng "
                "(or set an OpenAI-compatible endpoint for API TTS)"
            )
    elif engine not in engines and engine != "openai_compat":
        available = ", ".join(engines) or "none"
        raise ToolError(f"TTS engine {engine!r} not available (found: {available})")

    ext = "mp3" if engine in {"edge_tts", "openai_compat"} else (
        "aiff" if engine == "say" else "wav")
    out = out_dir / f"tts-{int(time.time() * 1000)}.{ext}"
    started = time.perf_counter()
    try:
        if engine == "edge_tts":
            _tts_edge(text, voice, out)
        elif engine == "openai_compat":
            _tts_openai_compat(text, voice, out, base_url=base_url, api_key=api_key, model=model)
        else:
            _tts_local(text, engine, voice, out)
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"TTS timed out after {timeout}s") from exc
    return {
        "path": str(out),
        "engine": engine,
        "format": ext,
        "bytes": out.stat().st_size,
        "seconds": round(time.perf_counter() - started, 2),
    }


# ── STT ──────────────────────────────────────────────────────────────────────


def _to_wav(path: Path) -> Path:
    """whisper.cpp wants wav; convert with ffmpeg when the input isn't one."""
    if path.suffix.lower() in {".wav", ".wave"}:
        return path
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ToolError(
            f"{path.suffix} audio needs ffmpeg to convert to wav "
            "(Termux: pkg install ffmpeg) or send a .wav"
        )
    converted = path.with_suffix(".wav")
    proc = subprocess.run(
        [ffmpeg, "-y", "-i", str(path), "-ar", "16000", "-ac", "1", str(converted)],
        capture_output=True, timeout=_STT_TIMEOUT, check=False,
    )
    if proc.returncode != 0 or not converted.exists():
        raise ToolError(
            "ffmpeg conversion failed: "
            f"{proc.stderr.decode('utf-8', 'replace')[:300]}"
        )
    return converted


def _stt_whisper_cpp(path: Path, language: str) -> str:
    binary = _whisper_cpp_binary()
    model = _whisper_cpp_model()
    if not binary:
        raise ToolError("whisper.cpp binary not found (set NM_WHISPER_CPP_BIN)")
    if not model:
        raise ToolError("no whisper.cpp model (set NM_WHISPER_CPP_MODEL to a .bin path)")
    wav = _to_wav(path)
    argv = [binary, "-m", model, "-f", str(wav)]
    if language and language != "auto":
        argv += ["-l", language]
    proc = subprocess.run(argv, capture_output=True, timeout=_STT_TIMEOUT, check=False)
    if proc.returncode != 0:
        raise ToolError(
            f"whisper.cpp failed (exit {proc.returncode}): "
            f"{(proc.stderr or b'').decode('utf-8', 'replace')[:300]}"
        )
    return re.sub(r"\[[^]]*\]", "", proc.stdout.decode("utf-8", "replace")).strip()


def _stt_openai_compat(
    path: Path, language: str, *, base_url: str, api_key: str, model: str
) -> str:
    from ..core.http import HttpClient

    base = base_url.rstrip("/")
    if not base.endswith("/v1") and "/v1" not in base:
        base = base + "/v1"
    client = HttpClient(timeout=_STT_TIMEOUT)
    # Groq doesn't serve OpenAI's "whisper-1" — use their turbo model.
    default_model = ("whisper-large-v3-turbo" if "groq.com" in base
                     else "whisper-1")
    response = client.post_multipart(
        f"{base}/audio/transcriptions",
        fields={"model": model or default_model, **({"language": language} if language and language != "auto" else {})},
        files=[("file", str(path), "application/octet-stream")],
        headers={"Authorization": f"Bearer {api_key}"},
    )
    if not response.ok:
        raise ToolError(f"STT API {response.status}: {response.text[:300]}")
    data = response.json()
    return str(data.get("text") or "").strip()


def stt(
    path: str | Path,
    *,
    provider: str = "auto",
    language: str = "en",
    base_url: str = "",
    api_key: str = "",
    model: str = "",
) -> dict[str, Any]:
    """Transcribe an audio file. Returns {text, provider, seconds}."""
    source = Path(path).expanduser()
    if not source.exists():
        raise ToolError(f"no such audio file: {source}")
    resolved = detect_stt(provider, base_url=base_url, api_key=api_key)
    if not resolved:
        raise ToolError(
            "no STT backend available — configure an OpenAI-compatible endpoint "
            "(NM_AUDIO_STT_BASE_URL + NM_AUDIO_STT_API_KEY) or a whisper.cpp "
            "binary (NM_WHISPER_CPP_BIN + NM_WHISPER_CPP_MODEL)"
        )
    started = time.perf_counter()
    if resolved == "whisper_cpp":
        text = _stt_whisper_cpp(source, language)
    else:
        text = _stt_openai_compat(source, language, base_url=base_url,
                                  api_key=api_key, model=model)
    return {
        "text": text,
        "provider": resolved,
        "seconds": round(time.perf_counter() - started, 2),
        "file": str(source),
        "bytes": source.stat().st_size,
    }


# ── info ─────────────────────────────────────────────────────────────────────


def audio_info(path: str | Path) -> dict[str, Any]:
    """Format, size, and duration (wav exact; others estimated)."""
    source = Path(path).expanduser()
    if not source.exists():
        raise ToolError(f"no such file: {source}")
    size = source.stat().st_size
    suffix = source.suffix.lower()
    info: dict[str, Any] = {"path": str(source), "bytes": size, "format": suffix.lstrip(".") or "unknown"}
    if suffix in {".wav", ".wave"}:
        try:
            with wave.open(str(source), "rb") as handle:
                frames = handle.getnframes()
                rate = handle.getframerate() or 1
                info["duration"] = round(frames / rate, 2)
                info["channels"] = handle.getnchannels()
                info["sample_rate"] = rate
        except (wave.Error, EOFError):
            info["duration"] = None
    elif suffix == ".mp3":
        info["duration"] = round(size / 132300, 2)  # ~128 kbps estimate
        info["duration_note"] = "estimated (128kbps)"
    elif suffix in {".ogg", ".opus", ".m4a", ".aac", ".flac", ".aiff", ".aif"}:
        info["duration"] = None
    return info


# ── tool registration ────────────────────────────────────────────────────────


def register(registry: Any) -> None:
    """Attach the speak / transcribe / audio_info tools to a registry."""
    context = registry.context
    settings = getattr(context, "settings", None) if context is not None else None
    audio_settings = getattr(settings, "audio", None) if settings else None
    llm_settings = getattr(settings, "llm", None) if settings else None

    def audio_dir() -> Path:
        rel = getattr(audio_settings, "audio_dir", "audio") or "audio"
        if settings is not None and hasattr(settings, "resolve"):
            return Path(settings.resolve(rel))
        return Path(rel)

    def _api_pair() -> tuple[str, str]:
        # Explicit audio settings win, then the LLM's per-provider keys
        # (groq_api_key etc. — populated from NM_GROQ_API_KEY and friends),
        # then the raw provider env vars the LLM broker itself uses.
        # Base URL and key stay paired — never mix providers.
        base = getattr(audio_settings, "stt_base_url", "") or ""
        key = getattr(audio_settings, "stt_api_key", "") or ""
        if base and key:
            return base, key
        if llm_settings is not None:
            for prefix in ("groq", "openai", "openrouter"):
                pkey = getattr(llm_settings, f"{prefix}_api_key", "") or ""
                pbase = getattr(llm_settings, f"{prefix}_base_url", "") or ""
                if pkey and pbase:
                    return pbase, pkey
            base = getattr(llm_settings, "base_url", "") or ""
            key = getattr(llm_settings, "api_key", "") or ""
            if base and key:
                return base, key
        if os.environ.get("OPENAI_API_KEY", ""):
            return (os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                    os.environ["OPENAI_API_KEY"])
        if os.environ.get("GROQ_API_KEY", ""):
            return (os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
                    os.environ["GROQ_API_KEY"])
        if os.environ.get("NM_GROQ_API_KEY", ""):
            return ("https://api.groq.com/openai/v1",
                    os.environ["NM_GROQ_API_KEY"])
        return "", ""

    def api_base() -> str:
        return _api_pair()[0]

    def api_key() -> str:
        return _api_pair()[1]

    def _check_net() -> None:
        checker = getattr(context, "check", None)
        if checker is not None:
            decision = checker(Capability.NET_OUT)
            if not getattr(decision, "allowed", True):
                raise ToolError(f"network not granted: {getattr(decision, 'reason', '')}")

    @registry.register(
        "speak",
        description=(
            "text-to-speech: synthesize audio to a file "
            "(espeak-ng / pico2wave / say / edge-tts / OpenAI-compatible, auto-detected)"
        ),
        capability=Capability.EXEC_SHELL,
        parameters={
            "text": "str — what to say",
            "engine": "str (optional) — auto | espeak_ng | pico2wave | say | edge_tts | openai_compat",
            "voice": "str (optional) — engine voice name",
        },
    )
    def speak(text: str, *, engine: str = "auto", voice: str = "") -> dict[str, Any]:
        base, key = api_base(), api_key()
        if engine in ("auto", "") and base and key and not detect_tts_engines():
            engine = "openai_compat"
        if engine == "openai_compat":
            _check_net()
        return tts(
            text, engine=engine, voice=voice, out_dir=audio_dir(),
            base_url=base, api_key=key,
            model=getattr(audio_settings, "stt_model", "") or "tts-1",
        )

    @registry.register(
        "transcribe",
        description=(
            "speech-to-text: transcribe an audio file "
            "(OpenAI-compatible /audio/transcriptions or local whisper.cpp)"
        ),
        capability=Capability.EXEC_SHELL,
        parameters={
            "path": "str — audio file (workspace-relative or absolute)",
            "language": "str (optional, default en; 'auto' to detect)",
            "provider": "str (optional) — auto | openai_compat | whisper_cpp",
        },
    )
    def transcribe(path: str, *, language: str = "en", provider: str = "auto") -> dict[str, Any]:
        from .filesystem import safe_media_path

        # voice notes land in the chat media dir (outside the workspace
        # sandbox by design) — safe_media_path allows Devon's own
        # downloads without opening the sandbox.
        target = safe_media_path(context, path, must_exist=True)
        base, key = api_base(), api_key()
        if provider == "openai_compat" or (provider == "auto" and base and key):
            _check_net()
        # Groq doesn't serve OpenAI's whisper-1 — pick the model for
        # the endpoint (empty = let stt() choose the provider default).
        stt_model = getattr(audio_settings, "stt_model", "") or ""
        if not stt_model and "groq.com" in base:
            stt_model = "whisper-large-v3-turbo"
        return stt(
            target, provider=provider, language=language,
            base_url=base, api_key=key, model=stt_model,
        )

    @registry.register(
        "audio_info",
        description="audio file metadata: format, size, duration (wav exact, mp3 estimated)",
        capability=Capability.FS_READ,
        parameters={"path": "str — audio file path"},
    )
    def audio_info_tool(path: str) -> dict[str, Any]:
        from .filesystem import safe_path

        return audio_info(safe_path(context, path, must_exist=True))

    # ── neural TTS (the owner's universal_tts.py, integrated) ────────────────

    def voices_dir() -> str:
        return str(audio_dir() / "voices")

    @registry.register(
        "tts_say",
        description=(
            "Neural text-to-speech (Bark / XTTS v2 / Kokoro, lazy-loaded): "
            "synthesizes text with inline emotion/pause tags "
            "([happy] [whisper] [laughs] [pause:300]) into a WAV voice note. "
            "voice = a registered profile name; mood bridges the partner "
            "mood system into tags."
        ),
        capability=Capability.FS_WRITE,
        parameters={
            "text": "str — what to say (tags allowed)",
            "voice": "str (optional) — registered voice profile name",
            "backend": "str (optional) — auto | bark | xtts | kokoro",
            "mood": "str (optional) — happy | sad | angry | tired | …",
            "mood_level": "int 0-5 (optional, 5)",
            "out_name": "str (optional) — output file name",
        },
    )
    def tts_say(text: str, *, voice: str = "", backend: str = "",
                mood: str = "", mood_level: str = "", out_name: str = "") -> dict[str, Any]:
        from ..voice.tts import UniversalTTS, available_backends

        text = (text or "").strip()
        if not text:
            raise ToolError("tts_say needs text")
        backend_name = (backend or "").strip() or \
            getattr(audio_settings, "neural_tts", "") or "auto"
        try:
            level = int(mood_level) if str(mood_level).strip() else 5
        except ValueError:
            level = 5
        engine = UniversalTTS(backend=backend_name, voices_dir=voices_dir())
        out_path = ""
        if out_name:
            out_path = str(audio_dir() / f"{out_name}.wav")
        try:
            return engine.speak(
                text, voice_name=voice or None, out_path=out_path,
                mood=mood, mood_level=level)
        except RuntimeError as exc:
            have = available_backends()
            hint = f" available: {', '.join(have)}" if have else ""
            raise ToolError(f"tts backend unavailable: {exc}{hint}") from exc

    @registry.register(
        "tts_voices",
        description=(
            "Manage TTS voice profiles: list, add a reference clip for "
            "cloning, register a built-in preset, or remove. "
            "Clone operations are audit-logged."
        ),
        capability=Capability.FS_WRITE,
        parameters={
            "action": "str — list | add | preset | remove",
            "name": "str — profile name (add/preset/remove)",
            "path": "str — reference audio file (add)",
            "preset_id": "str — built-in voice id (preset)",
            "language": "str (optional, en)",
        },
    )
    def tts_voices(action: str, *, name: str = "", path: str = "",
                   preset_id: str = "",
                   language: str = "") -> dict[str, Any]:
        from ..voice.tts import VoiceLibrary

        action = (action or "list").strip().lower()
        lib = VoiceLibrary(voices_dir())
        if action == "list":
            return {"voices": lib.list()}
        if action == "add":
            if not name:
                raise ToolError("tts_voices add needs a name")
            sample = path
            if sample and not os.path.isabs(sample):
                try:
                    from .filesystem import safe_path

                    sample = str(safe_path(context, sample, must_exist=True))
                except ToolError:
                    raise ToolError(f"voice sample not found: {path!r}") from None
            if not sample:
                # profile without reference audio (preset-style)
                from ..voice.tts import VoiceProfile

                profile = VoiceProfile(
                    name=name, language=language or "en")
                lib.profiles[name] = profile
                lib._save_index()
                return {"ok": True, "profile": profile.to_dict()}
            profile = lib.upload_voice(
                name, sample,
                language=language or "en")
            profile.audit_clone("tts_voices")
            return {"ok": True, "profile": profile.to_dict()}
        if action == "preset":
            if not name or not preset_id:
                raise ToolError("tts_voices preset needs name + preset_id")
            profile = lib.register_preset(
                name, preset_id, language=language or "en")
            return {"ok": True, "profile": profile.to_dict()}
        if action == "remove":
            return {"ok": lib.remove(name or "")}
        raise ToolError(f"unknown tts_voices action {action!r}")

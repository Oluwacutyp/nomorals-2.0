"""VoiceStudio tool — drive a locally-running VoiceStudio backend.

VoiceStudio (debpalash/VoiceStudio, AGPL-3.0) is a fully-local voice
studio: clone, design, dub, transcribe, narrate. Devon talks to it
as an external tool over its local HTTP API (see
``nomorals.voice.voicestudio``); no VoiceStudio code is vendored
anywhere in this repo.

One tool, ``voicestudio``, with an ``action`` verb. When the backend
is not running, every action returns ``{"ok": False, ...}`` naming
how to start it — never a fake voice, never a silent fallback. The
base URL is environment-only (NM_VOICESTUDIO_URL); there is no
``base_url`` parameter, so no caller can point this tool at an
arbitrary host.
"""

from __future__ import annotations

from typing import Any

from ..core.policy import Capability
from ..voice.voicestudio import VoiceStudioClient, VoiceStudioError

__all__ = ["register"]

_CLIENT: VoiceStudioClient | None = None


def _get_client() -> VoiceStudioClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = VoiceStudioClient()
    return _CLIENT


def _set_client_for_tests(client: VoiceStudioClient | None) -> None:
    global _CLIENT
    _CLIENT = client


def register(registry: Any) -> None:
    """Attach the voicestudio tool to a registry."""

    @registry.register(
        "voicestudio",
        description=(
            "VoiceStudio — the local voice studio (clone, design, dub, "
            "transcribe, narrate) via a locally-running VoiceStudio "
            "backend (http://127.0.0.1:3900). Actions: status (backend "
            "health + transports), voices (saved profiles + engines), "
            "speak <text> [voice] [model] [response_format] (text to an "
            "audio file), transcribe <audio> (audio to text, "
            "verbose_json by default), clone <name> <audio> [language] "
            "(clone a voice from a reference recording — only voices "
            "the owner is authorized to use; the reference is recorded "
            "for audit), design <description> [name] [language] (design "
            "a voice from words), dub <video> <language> [voice] (dub a "
            "video into another language; waits for the job, returns "
            "the job id on timeout), dub_status <job_id> (resume a dub "
            "job), narrate <text> [voice] (long text to one wav). When "
            "the backend is down every action says so plainly with the "
            "start command — never a fake voice."
        ),
        capability=Capability.MODEL_CALL,
        parameters={
            "action": "str — status|voices|speak|transcribe|clone|"
                      "design|dub|dub_status|narrate (default status)",
            "text": "str — text for speak|narrate",
            "voice": "str — VoiceStudio voice profile id for speak|"
                     "narrate|dub",
            "model": "str — engine/model id (default tts-1)",
            "response_format": "str — wav|mp3|opus|aac|flac|pcm for "
                               "speak; json|text|verbose_json|srt|vtt "
                               "for transcribe",
            "audio": "str — audio file path for transcribe|clone",
            "video": "str — video file path for dub",
            "language": "str — language for clone|design|dub",
            "name": "str — voice name for clone|design",
            "description": "str — voice description for design",
            "sample_text": "str — sample text for design",
            "out": "str — output file path for speak|narrate",
            "job_id": "str — dub job id for dub_status",
            "timeout": "int — seconds to wait for a dub job "
                       "(0 = default 600)",
        },
    )
    def voicestudio(action: str = "status", text: str = "",
                    voice: str = "", model: str = "tts-1",
                    response_format: str = "", audio: str = "",
                    video: str = "", language: str = "",
                    name: str = "", description: str = "",
                    sample_text: str = "", out: str = "",
                    job_id: str = "", timeout: int = 0) -> dict[str, Any]:
        client = _get_client()
        try:
            return _dispatch(client, action, text=text, voice=voice,
                             model=model,
                             response_format=response_format,
                             audio=audio, video=video, language=language,
                             name=name, description=description,
                             sample_text=sample_text, out=out,
                             job_id=job_id, timeout=timeout)
        except VoiceStudioError as exc:
            return exc.as_dict()


def _dispatch(client: VoiceStudioClient, action: str, **kw: Any
              ) -> dict[str, Any]:
    action = (action or "status").strip().lower()
    if action == "status":
        return client.status()
    if action == "voices":
        result = client.list_voices()
        result["recorded"] = client.recorded_profiles()
        return result
    if action == "speak":
        return client.speak(
            kw["text"], voice=kw["voice"], model=kw["model"],
            response_format=kw["response_format"] or "wav",
            out_path=kw["out"] or None)
    if action == "transcribe":
        return client.transcribe(
            kw["audio"], model=kw["model"] or "whisper-1",
            response_format=kw["response_format"] or "verbose_json")
    if action == "clone":
        return client.clone_voice(kw["name"], kw["audio"],
                                  language=kw["language"],
                                  description=kw["description"])
    if action == "design":
        return client.design_voice(kw["description"], name=kw["name"],
                                   language=kw["language"],
                                   sample_text=kw["sample_text"])
    if action == "dub":
        return client.dub_video(
            kw["video"], kw["language"], voice=kw["voice"],
            timeout_s=float(kw["timeout"] or 600))
    if action == "dub_status":
        return client.poll_dub(kw["job_id"],
                               timeout_s=float(kw["timeout"] or 60))
    if action == "narrate":
        return client.narrate(kw["text"], voice=kw["voice"],
                              model=kw["model"],
                              out_path=kw["out"] or None)
    return {"ok": False,
            "error": f"unknown voicestudio action: {action!r}",
            "kind": "error",
            "hint": "Use one of: status, voices, speak, transcribe, "
                    "clone, design, dub, dub_status, narrate.",
            "retryable": False}

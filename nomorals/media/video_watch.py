"""Video watcher — Devon truly sees and hears a video.

The goal (devon-video-watcher): a video the owner sends or names gets a
genuine description/answer grounded in its frames and audio — not just
its caption. The organ composes three rungs that already exist in the
repo, and adds no fake one:

- frames: :func:`nomorals.media_edit.videos.extract_frames` (ffmpeg)
  + :func:`nomorals.media_edit.videos.video_probe` for duration/fps.
- seeing: :func:`nomorals.tools.vision.extract` — the vision model's
  structured reading (summary + key text + notable elements) of each
  sampled frame. Strict: it raises ModelError when no vision-capable
  provider is configured, so "cannot see" is honest, never an empty
  success.
- hearing: :func:`nomorals.media_edit.videos.extract_audio` (ffmpeg,
  wav) + :func:`nomorals.tools.audio.stt` — raises ToolError when no
  STT backend is configured.

This is NOT :mod:`nomorals.media.director.watchvideo` — that organ is
temporal QA on a render (cuts/pacing/motion against a plan). This one
understands an arbitrary video: what it shows, what it says, and what
it means for the owner's question.

Honest degradation, by rung: a silent video (no audio stream) is still
fully "watched" — silence is a fact, not a failure. A video whose
frames cannot be described (no vision model) but whose audio
transcribes is "partial" — heard, not seen — and the limit names the
wall. A video neither rung can read is "unavailable" with both reasons.
A bad path, an unreadable file, or a missing ffmpeg never raises: it
returns "unavailable" with the reason.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..core.errors import ModelError
from ..core.logging_setup import get_logger
from ..media_edit.videos import (
    MediaEditError,
    extract_audio,
    extract_frames,
    video_probe,
)

_log = get_logger(__name__)

__all__ = [
    "watch_video",
    "render_watch",
    "watch_capabilities",
    "DEFAULT_FRAMES",
    "MAX_FRAMES",
]

#: Frames sampled per video by default — enough to follow a short clip,
#: few enough that a vision call per frame stays cheap.
DEFAULT_FRAMES = 6

#: Hard cap on sampled frames — a watch is a skim, not a frame-by-frame
#: read; longer videos still get evenly spaced frames, not more of them.
MAX_FRAMES = 12

#: Width frames are downscaled to before the vision call (extract_frames
#: default is 640; named here so the watch's cost is visible in one place).
FRAME_WIDTH = 640

#: Transcript / summary ceilings in the rendered reply — the result dict
#: keeps the full text; only the chat rendering truncates.
TRANSCRIPT_PREVIEW_CHARS = 1500
SUMMARY_CHARS = 2000


def watch_video(
    path: str | os.PathLike[str],
    *,
    context: Any = None,
    question: str = "",
    frames: int = DEFAULT_FRAMES,
    describe_fn: Callable[[bytes, str], dict[str, Any]] | None = None,
    transcribe_fn: Callable[[str], dict[str, Any]] | None = None,
    sample_fn: Callable[..., Any] | None = None,
    audio_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Watch ``path`` and understand it: what it shows, what it says.

    ``describe_fn`` takes (frame_bytes, prompt) and returns a vision
    ``extract`` result (``summary`` / ``key_text`` / ``notable_elements`` /
    ``scene_type``); ``transcribe_fn`` takes an audio path and returns an
    ``stt`` result (``text`` / ``provider`` / ``seconds``). Both default
    to the real rungs; tests inject fakes. ``sample_fn`` / ``audio_fn``
    are the ffmpeg seams (same shape as the real calls, tests only).

    Never raises for a bad video: an unreadable file, a failed probe,
    a failed sample, or both understanding rungs down yields a dict
    with ``status`` ``"unavailable"`` and the reason in ``notes``.
    """
    source = str(path)
    count = max(1, min(int(frames or DEFAULT_FRAMES), MAX_FRAMES))
    question = (question or "").strip()
    limits: list[str] = []
    try:
        info = video_probe(source)
    except Exception as exc:  # noqa: BLE001 - bad video reports, never crashes
        return _unavailable(source, f"could not read the video: {exc}")
    duration = float(info.get("duration") or 0.0)
    fps = float(info.get("fps") or 0.0)
    streams = info.get("streams") or []
    has_audio = any(str(s.get("type") or "") == "audio" for s in streams)
    if duration <= 0:
        return _unavailable(source, "no duration to watch — empty or "
                                    "unreadable video")

    describe = describe_fn or (
        lambda data, prompt: _describe_frame(data, prompt, context))
    transcribe = transcribe_fn or _transcribe_audio
    frame_notes: list[dict[str, Any]] = []
    seen = False
    with tempfile.TemporaryDirectory(prefix="videowatch-") as tmp:
        try:
            sampled = (sample_fn or _sample_frames)(source, count, tmp)
        except Exception as exc:  # noqa: BLE001 - seeing may still fail alone
            sampled = []
            limits.append(f"could not sample frames: {exc}")
        for stamp, frame_path in sampled:
            try:
                data = Path(frame_path).read_bytes()
            except OSError as exc:
                frame_notes.append({"t": stamp, "error": str(exc)})
                continue
            prompt = _frame_prompt(stamp, duration, question)
            try:
                got = describe(data, prompt)
            except Exception as exc:  # noqa: BLE001 - one bad frame is a limit
                frame_notes.append({"t": stamp, "error": str(exc)})
                continue
            frame_notes.append({
                "t": round(stamp, 3),
                "summary": str(got.get("summary") or "").strip(),
                "key_text": str(got.get("key_text") or "").strip(),
                "elements": [str(e) for e in (got.get("notable_elements") or [])],
                "scene_type": str(got.get("scene_type") or "unknown"),
            })
            seen = True
        if sampled and not seen:
            first_err = next((f.get("error", "") for f in frame_notes
                              if f.get("error")), "")
            limits.append("could not see the frames"
                          + (f": {first_err}" if first_err else
                             " (no vision-capable model configured)"))
        transcript = ""
        transcript_provider = ""
        heard = False
        if not has_audio:
            limits.append("silent video — no audio track to hear")
            heard = True  # silence is a fact, not a failed rung
        else:
            try:
                audio_path = (audio_fn or _extract_audio_track)(source, tmp)
                got = transcribe(audio_path)
                transcript = str(got.get("text") or "").strip()
                transcript_provider = str(got.get("provider") or "")
                heard = bool(transcript)
                if not heard:
                    limits.append("audio transcribed to nothing — the "
                                  "track may be silent or unintelligible")
            except Exception as exc:  # noqa: BLE001 - hearing may fail alone
                limits.append(f"could not hear the audio: {exc}")

    if seen and heard:
        status = "watched"
    elif seen or heard:
        status = "partial"
    else:
        return _unavailable(source, "; ".join(limits) or
                            "neither seeing nor hearing worked",
                            duration=duration, fps=fps,
                            limits=limits, frames=frame_notes)
    return {
        "status": status,
        "summary": _compose_summary(frame_notes, transcript, question),
        "question": question,
        "frames": frame_notes,
        "transcript": transcript,
        "transcript_provider": transcript_provider,
        "duration": round(duration, 3),
        "fps": fps,
        "frames_sampled": len(sampled),
        "has_audio": has_audio,
        "limits": limits,
        "source": source,
        "backend": "ffmpeg",
    }


def _unavailable(source: str, notes: str, **extra: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "unavailable",
        "summary": "",
        "question": "",
        "frames": [],
        "transcript": "",
        "transcript_provider": "",
        "duration": 0.0,
        "fps": 0.0,
        "frames_sampled": 0,
        "has_audio": False,
        "limits": [notes],
        "notes": notes,
        "source": source,
        "backend": "ffmpeg",
    }
    result.update(extra)
    if "limits" not in extra:
        result["limits"] = [notes]
    return result


def _sample_frames(source: str, count: int,
                   tmp: str) -> list[tuple[float, str]]:
    """Evenly spaced frames across the video. Returns (seconds, path)."""
    info = video_probe(source)
    duration = float(info.get("duration") or 0.0)
    res = extract_frames(source, count=count, out_dir=tmp,
                         width=FRAME_WIDTH)
    files = sorted(res.get("frames") or [])
    if not files:
        raise MediaEditError(f"no frames sampled from {source}")
    step = duration / len(files) if duration > 0 else 0.0
    return [(i * step, f) for i, f in enumerate(files)]


def _extract_audio_track(source: str, tmp: str) -> str:
    res = extract_audio(source, out_dir=tmp, ext=".wav")
    return str(res["output"])


def _describe_frame(data: bytes, prompt: str,
                    context: Any) -> dict[str, Any]:
    """Default seeing rung: the vision model's structured extract."""
    from ..tools.vision import extract as vision_extract

    if context is None:
        raise ModelError(
            "cannot see the frames: no context (and so no vision-capable "
            "model provider) was given to the watcher")
    return vision_extract(context, data, prompt)


def _transcribe_audio(audio_path: str) -> dict[str, Any]:
    """Default hearing rung: the STT tool (endpoint or whisper.cpp)."""
    from ..tools.audio import stt

    return stt(audio_path, provider="auto")


def _frame_prompt(stamp: float, duration: float, question: str) -> str:
    base = (f"This is a frame at {stamp:.1f}s of a {duration:.1f}s video. "
            "Describe what it shows: the scene, the people or objects, "
            "any action, and any visible text.")
    if question:
        base += (f" Answer this about the frame, first and foremost: "
                 f"{question}")
    return base


def _compose_summary(frames: list[dict[str, Any]], transcript: str,
                     question: str) -> str:
    lines: list[str] = []
    if question:
        lines.append(f"About {question!r}:")
    for frame in frames:
        summary = str(frame.get("summary") or "").strip()
        if not summary:
            continue
        stamp = frame.get("t", 0.0)
        lines.append(f"[{float(stamp):.1f}s] {summary}")
    if transcript:
        shown = transcript if len(transcript) <= 600 else transcript[:597] + "…"
        lines.append(f"Heard: {shown!r}")
    text = "\n".join(lines).strip()
    return text[:SUMMARY_CHARS] if len(text) > SUMMARY_CHARS else text


def render_watch(result: dict[str, Any]) -> str:
    """Render a :func:`watch_video` result as a compact chat reply.

    Frame lines carry their timestamps; the transcript is an excerpt;
    every limit is named in plain words — the reply never hides what
    Devon could not see or hear.
    """
    status = str(result.get("status") or "unavailable")
    duration = result.get("duration") or 0.0
    head = f"🎬 {float(duration):.1f}s video — "
    if status == "watched":
        head += "watched it."
    elif status == "partial":
        head += "partly watched — see the limits below."
    else:
        notes = str(result.get("notes") or
                    "; ".join(result.get("limits") or []) or
                    "could not watch this video")
        return head + f"could not watch it: {notes}"
    lines = [head]
    question = str(result.get("question") or "").strip()
    if question:
        lines.append(f"❓ {question}")
    for frame in result.get("frames") or []:
        summary = str(frame.get("summary") or "").strip()
        if not summary:
            continue
        stamp = float(frame.get("t", 0.0))
        line = f"[{stamp:.1f}s] {summary}"
        key_text = str(frame.get("key_text") or "").strip()
        if key_text:
            shown = key_text if len(key_text) <= 120 else key_text[:117] + "…"
            line += f" · text: {shown!r}"
        lines.append(line)
    transcript = str(result.get("transcript") or "").strip()
    if transcript:
        shown = (transcript if len(transcript) <= TRANSCRIPT_PREVIEW_CHARS
                 else transcript[:TRANSCRIPT_PREVIEW_CHARS - 1] + "…")
        lines.append(f"🗣️ {shown!r}")
    for limit in result.get("limits") or []:
        lines.append(f"⚠️ {limit}")
    return "\n".join(line for line in lines if line.strip())


def _stt_endpoint(context: Any) -> tuple[str, str]:
    """Mirror the transcribe tool's endpoint ladder (tools/audio.py
    ``register()``: ``_api_pair``) — explicit audio settings, then the
    LLM's per-provider keys, then the raw provider env vars. Base URL
    and key stay paired — never mix providers."""
    settings = getattr(context, "settings", None) if context is not None else None
    audio_settings = getattr(settings, "audio", None) if settings else None
    llm_settings = getattr(settings, "llm", None) if settings else None
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


def watch_capabilities(context: Any = None) -> dict[str, Any]:
    """What the watcher can do on THIS machine, rung by rung.

    ffmpeg present? Vision model configured on the context? STT backend
    available? Each answer names its evidence — the /video watch info
    surface prints this verbatim, so a missing rung is a named wall,
    never a silent gap.
    """
    from ..media_edit.videos import ffmpeg_path
    from ..tools.audio import detect_stt

    try:
        ffmpeg = ffmpeg_path()
    except Exception as exc:  # noqa: BLE001 - absence is a reportable fact
        ffmpeg = ""
        ffmpeg_note = str(exc)
    else:
        ffmpeg_note = ""
    router = getattr(context, "router", None) if context is not None else None
    vision = router is not None
    try:
        base, key = _stt_endpoint(context)
        stt_provider = detect_stt("auto", base_url=base, api_key=key)
    except Exception:  # noqa: BLE001 - detection failure reads as unavailable
        stt_provider = None
    notes: list[str] = []
    if not ffmpeg:
        notes.append("ffmpeg is missing — no frames and no audio can be "
                     "pulled from a video on this machine "
                     f"({ffmpeg_note})" if ffmpeg_note else
                     "ffmpeg is missing — no frames and no audio can be "
                     "pulled from a video on this machine")
    if not vision:
        notes.append("no vision-capable model is configured on this "
                     "context — frames cannot be described, only heard")
    if not stt_provider:
        notes.append("no speech-to-text backend is configured "
                     "(NM_AUDIO_STT_BASE_URL + NM_AUDIO_STT_API_KEY, or "
                     "whisper.cpp via NM_WHISPER_CPP_BIN + "
                     "NM_WHISPER_CPP_MODEL) — audio cannot be transcribed")
    return {
        "ffmpeg": bool(ffmpeg),
        "ffmpeg_path": ffmpeg,
        "vision": vision,
        "stt": stt_provider,
        "ready": bool(ffmpeg and vision and stt_provider),
        "notes": notes,
    }

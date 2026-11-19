"""Code-video narration — the video speaks.

The Anthropic code-video style as the owner asked it: text in, polished
video out, NO video model — "text to speech is the bastard I'm talking
about". The code backend rendered real but SILENT MP4s; this rung speaks
the scene's own script and muxes the voice into the MP4.

The narration text is derived from the scene spec itself — the title
and the labels of the elements that appear in each build step — never
a canned line. Timing follows the explainer's own step windows
(:func:`nomorals.media.motion_studio.explainers._window`): a segment
starts when its step's window starts, and never overlaps the next
segment — when the spoken words need more room than the requested
duration, the render grows to fit the voice instead of clipping it.

The voice comes from Devon's existing TTS ladder
(:class:`nomorals.voice.tts.UniversalTTS`) — the same seam the
character voice rung and the audio pipeline's synthesize stage call.
No new TTS client, no placeholder tone: when no TTS backend is
installed on the host, the render reports ``narration: missing`` with
the reason and still delivers the video — a silent file is never
passed off as narrated.
"""

from __future__ import annotations

import os
import tempfile
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ...core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "NARRATION_SAMPLE_RATE",
    "scene_narration",
    "synthesize_narration",
]

#: The narration mix is one mono track at the Piper default rate — the
#: rate the ladder's CPU backends speak at. Segment wavs at other rates
#: are resampled onto it in the layout.
NARRATION_SAMPLE_RATE = 22050

#: Silence between two spoken segments, so one voice never runs into
#: the next; and the tail kept after the last word before the video ends.
_GAP_S = 0.15
_TAIL_S = 0.25

#: How many times the layout re-solves the step windows after the
#: duration grows to fit the voice. Converges on the first re-solve in
#: practice; the bound is a guard, not a loop.
_MAX_LAYOUT_PASSES = 3


# ---------------------------------------------------------------------------
# the script — derived from the scene spec
# ---------------------------------------------------------------------------

def _speakable(text: str) -> str:
    """Scene text as spoken words — arrows and dashes read aloud."""
    out = (text or "").replace("→", " to ").replace("—", ", ")
    return " ".join(out.split())


def _speakable_code(text: str) -> str:
    """Code as spoken words — operators read aloud, not spelled.

    Kept separate from :func:`_speakable` on purpose: equations keep
    their ``=``/``+`` verbatim (existing narration pins), while a
    trace line reads ``x = x + 1`` as ``x equals x plus 1``.
    """
    out = _speakable(text)
    out = out.replace("=", " equals ").replace("+", " plus ")
    out = out.replace("*", " times ").replace("/", " over ")
    return " ".join(out.split())


def _fmt_value(value: Any) -> str:
    """A chart value as spoken words — ``42.0`` reads as ``42``."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if num == int(num):
        return str(int(num))
    return str(num)


def _step_labels(spec: dict[str, Any]) -> dict[int, list[str]]:
    """Build-step index → the speakable labels that appear in it."""
    kind = spec.get("kind", "")
    by_step: dict[int, list[str]] = {}
    if kind == "diagram":
        elements = list(spec.get("nodes", [])) + list(spec.get("edges", []))
        for el in elements:
            label = _speakable(str(el.get("label", "") or ""))
            if label:
                by_step.setdefault(int(el.get("step", 0)), []).append(label)
    elif kind == "equation":
        for line in spec.get("lines", []):
            text = _speakable(str(line.get("text", "") or ""))
            if text:
                by_step.setdefault(int(line.get("step", 0)),
                                  []).append(text)
    elif kind == "code":
        for line in spec.get("lines", []):
            text = _speakable_code(str(line.get("text", "") or ""))
            values = _speakable_code(str(line.get("values", "") or ""))
            if values:
                text = f"{text}. Now: {values}" if text else values
            if text:
                by_step.setdefault(int(line.get("step", 0)),
                                  []).append(text)
    elif kind == "chart":
        for point in spec.get("series", []):
            label = _speakable(str(point.get("label", "") or ""))
            text = f"{label}, {_fmt_value(point.get('value'))}" \
                if label else _fmt_value(point.get("value"))
            by_step.setdefault(int(point.get("step", 0)),
                              []).append(text)
    return by_step


def scene_narration(scene: dict[str, Any], *, duration: float
                    ) -> list[dict[str, Any]]:
    """The scene's own script as timed segments.

    One segment per build step (the explainer's step windows over
    ``duration``); step 0 opens with the scene title. An HTML scene
    has no steps — one segment, its title, at t=0. A scene with no
    speakable text yields no segments (the caller reports that
    honestly instead of inventing words).

    Segments carry ``text`` (speakable), ``step``, ``start`` and ``end``
    (the nominal step window in seconds).
    """
    if not isinstance(scene, dict) or not scene:
        return []
    if "html" in scene:
        title = _speakable(str(scene.get("title", "") or ""))
        if not title:
            return []
        return [{"text": title, "step": 0, "start": 0.0,
                 "end": float(duration)}]
    from ..motion_studio.explainers import validate_design

    try:
        spec = validate_design(scene)
    except Exception:  # noqa: BLE001 - an invalid scene has no script
        return []
    n_steps = max(1, int(spec.get("steps", 1)))
    step_len = float(duration) / n_steps
    by_step = _step_labels(spec)
    title = _speakable(str(spec.get("title", "") or ""))
    segments: list[dict[str, Any]] = []
    for step in sorted(by_step):
        labels = by_step[step]
        parts: list[str] = []
        if step == 0 and title and title not in labels:
            parts.append(title)
        parts.extend(labels)
        text = ". ".join(parts)
        if not text.strip():
            continue
        segments.append({"text": text, "step": int(step),
                         "start": step * step_len,
                         "end": (step + 1) * step_len})
    return segments


# ---------------------------------------------------------------------------
# the voice — through the existing TTS ladder
# ---------------------------------------------------------------------------

def available_backends() -> list[str]:
    """TTS backends installed on this host (re-exported for patching)."""
    from ...voice.tts import available_backends as _avail

    return _avail()


def _default_tts(text: str, out_path: str) -> dict[str, Any]:
    """Speak ``text`` through UniversalTTS — the ladder's own seam.

    Returns ``{"ok": True, "path": ..., "backend": ...}`` or
    ``{"ok": False, "reason": ...}`` naming the wall. Never raises.
    """
    backends = available_backends()
    if not backends:
        return {"ok": False,
                "reason": ("no TTS backend installed on this box — "
                           "XTTS (and every other TTS backend) is "
                           "unavailable until one is installed — run "
                           "/models in Devon's chat to install a voice "
                           "backend")}
    try:
        from ...voice.tts import UniversalTTS

        engine = UniversalTTS(backend="auto", audience="private")
        result = engine.speak(text, out_path=out_path)
    except Exception as exc:  # noqa: BLE001 - a voice failure is a wall
        return {"ok": False, "reason": f"TTS render failed: {exc}"}
    if isinstance(result, dict) and result.get("ok") is False:
        return {"ok": False,
                "reason": str(result.get("reason") or result)}
    path = ""
    backend = ""
    if isinstance(result, dict):
        path = str(result.get("path") or result.get("out_path") or "")
        backend = str(result.get("backend") or "")
    if not path or not os.path.isfile(path):
        return {"ok": False,
                "reason": "TTS produced no audio file"}
    return {"ok": True, "path": path, "backend": backend or backends[0]}


def _wav_info(path: str) -> tuple[float, int]:
    """(duration_seconds, sample_rate) of a PCM wav file."""
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate()), w.getframerate()


def _read_wav_mono(path: str, target_sr: int) -> Any:
    """A wav as float32 mono samples at ``target_sr`` (numpy)."""
    import numpy as np

    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n_ch = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        data /= 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
        data /= 2147483648.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
                - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported wav sample width: {width}")
    if n_ch > 1:
        data = data.reshape(-1, n_ch).mean(axis=1)
    if sr != target_sr and len(data) > 1:
        src_t = np.arange(len(data), dtype=np.float64)
        dst_t = np.arange(int(round(len(data) * target_sr / sr)),
                          dtype=np.float64) * (sr / target_sr)
        data = np.interp(dst_t, src_t, data).astype(np.float32)
    return data


def synthesize_narration(segments: list[dict[str, Any]], *,
                         duration: float, out_dir: str,
                         name: str = "narration.wav",
                         tts_fn: Callable | None = None
                         ) -> dict[str, Any]:
    """Speak the segments and lay them on the video's timeline.

    Each segment is spoken through the TTS ladder (``tts_fn`` when
    given — ``(text, out_path) -> {"ok", "path", ...}`` — else the
    default UniversalTTS seam) and placed at its step window's start,
    never overlapping the previous segment: a segment starts at
    ``max(window start, previous end + gap)``. When the spoken words
    need more room than ``duration``, the timeline grows to fit the
    voice — the returned ``duration`` is the final video duration.

    Returns ``{"ok": True, "path", "backend", "duration", "segments":
    [{text, spoken_start, spoken_end}]}`` or ``{"ok": False, "reason"}``.
    """
    texts = [(s.get("text") or "").strip() for s in segments]
    if not any(texts):
        return {"ok": False,
                "reason": "nothing to narrate — the scene carries no "
                          "speakable text"}
    speak = tts_fn or _default_tts
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(prefix="vidnarrate-",
                                   dir=str(out_path)))
    try:
        spoken: list[dict[str, Any]] = []
        for i, seg in enumerate(segments):
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            piece = tmpdir / f"seg{i}.wav"
            try:
                res = speak(text, str(piece))
            except Exception as exc:  # noqa: BLE001 - voice wall
                return {"ok": False,
                        "reason": f"TTS render failed: {exc}"}
            if not isinstance(res, dict) or res.get("ok") is False:
                reason = ""
                if isinstance(res, dict):
                    reason = str(res.get("reason") or res)
                return {"ok": False,
                        "reason": reason or "TTS produced no audio"}
            path = str(res.get("path") or piece)
            if not os.path.isfile(path):
                return {"ok": False,
                        "reason": "TTS produced no audio file"}
            try:
                length, _sr = _wav_info(path)
            except Exception as exc:  # noqa: BLE001 - bad wav is a wall
                return {"ok": False,
                        "reason": f"TTS audio unreadable: {exc}"}
            spoken.append({"text": text, "step": int(seg.get("step", i)),
                           "path": path, "length": length,
                           "backend": str(res.get("backend") or "")})
        if not spoken:
            return {"ok": False,
                    "reason": "nothing to narrate — the scene carries "
                              "no speakable text"}
        n_steps = max(1, max(s["step"] for s in spoken) + 1)
        final_duration = float(duration)
        placed: list[dict[str, Any]] = []
        for _ in range(_MAX_LAYOUT_PASSES):
            step_len = final_duration / n_steps
            placed = []
            prev_end = 0.0
            for s in spoken:
                start = s["step"] * step_len
                if placed:
                    start = max(start, prev_end + _GAP_S)
                end = start + s["length"]
                placed.append({**s, "spoken_start": start,
                               "spoken_end": end})
                prev_end = end
            needed = prev_end + _TAIL_S
            if needed <= final_duration + 1e-6:
                break
            final_duration = needed
        # the mix — one mono track at the narration rate, silence
        # everywhere no voice is speaking.
        import numpy as np

        total = int(round(final_duration * NARRATION_SAMPLE_RATE))
        mix = np.zeros(total, dtype=np.float32)
        for s in placed:
            data = _read_wav_mono(s["path"], NARRATION_SAMPLE_RATE)
            at = int(round(s["spoken_start"] * NARRATION_SAMPLE_RATE))
            end = min(total, at + len(data))
            if end > at:
                mix[at:end] += data[:end - at]
        mix = np.clip(mix, -1.0, 1.0)
        final = out_path / name
        with wave.open(str(final), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(NARRATION_SAMPLE_RATE)
            w.writeframes((mix * 32767.0).astype(np.int16).tobytes())
        backend = next((s["backend"] for s in placed if s["backend"]),
                       "")
        return {"ok": True, "path": str(final), "backend": backend,
                "duration": final_duration,
                "segments": [{"text": s["text"],
                              "spoken_start": s["spoken_start"],
                              "spoken_end": s["spoken_end"]}
                             for s in placed]}
    finally:
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)

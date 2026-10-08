"""Lightweight TTS vocal track for /music — the Termux-friendly voice.

The full vocal chain (DiffSinger -> RVC, :mod:`nomorals.media.vocals`)
needs a real machine and registered voice models. This module is the
lightweight alternative wired into the default ``/music`` path:

1. Take the chorus hook lyrics from the composed :class:`Song`.
2. Synthesize them with the profile-appropriate TTS backend
   (:class:`nomorals.voice.tts.UniversalTTS`, audience="private" — the
   owner's own song, so XTTS cloning is allowed where installed).
3. Tile the hook across the chorus sections with small gaps and a short
   fade on each tile (no pitch-shifting DSP — the hook keeps its
   natural voice, like a vocal sample/chant over the beat).
4. Mix the vocal under the instrumental bed, peak-normalized.

A sung/spoken hook over the beat is infinitely better than silence.

Fail-closed: no TTS backend available -> no vocal track, honest note on
the song. Never fake audio, never raises.
"""

from __future__ import annotations

import logging
import os
import wave
from array import array
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

__all__ = [
    "hook_lyrics",
    "chorus_regions",
    "song_duration_beats",
    "render_hook_vocal",
    "add_vocal_track",
    "mix_vocal_track",
]

#: the synth's sample rate — vocal tiles are resampled to this
_SAMPLE_RATE = 22050
#: gap between repeated hook tiles, seconds
_TILE_GAP_S = 0.6
#: fade in/out per tile, seconds (avoids clicks at tile boundaries)
_TILE_FADE_S = 0.015
#: vocal level under the bed
_VOCAL_GAIN = 0.85
#: beats per bar (the composer works in 4/4)
_BEATS_PER_BAR = 4


def _profile_kind() -> str:
    try:
        from ..core.profile import detect_profile
        return str(detect_profile().kind).lower()
    except Exception:  # noqa: BLE001
        return ""


def hook_lyrics(song: Any) -> list[str]:
    """The hook: lyrics of the first chorus section.

    Falls back to the first section that has any lyrics. Never raises.
    """
    try:
        sections = list(getattr(song, "sections", None) or [])
        for s in sections:
            if str(getattr(s, "name", "")).lower() == "chorus":
                lines = [str(l) for l in (getattr(s, "lyrics", None) or [])]
                if lines:
                    return lines
        for s in sections:
            lines = [str(l) for l in (getattr(s, "lyrics", None) or [])]
            if lines:
                return lines
    except Exception:  # noqa: BLE001
        _log.debug("hook_lyrics failed", exc_info=True)
    return []


def song_duration_beats(song: Any) -> float:
    """Total song length in beats (4/4). Never raises."""
    try:
        total_bars = sum(int(getattr(s, "bars", 0) or 0)
                         for s in (getattr(song, "sections", None) or []))
        return float(total_bars * _BEATS_PER_BAR)
    except Exception:  # noqa: BLE001
        return 0.0


def chorus_regions(song: Any) -> list[tuple[float, float]]:
    """Absolute (start_beat, end_beat) of every chorus section.

    Falls back to the middle third of the song when there is no chorus
    (every style in the composer has one, but never assume). Never raises.
    """
    try:
        regions: list[tuple[float, float]] = []
        bar = 0
        sections = list(getattr(song, "sections", None) or [])
        for s in sections:
            bars = int(getattr(s, "bars", 0) or 0)
            if str(getattr(s, "name", "")).lower() == "chorus" and bars > 0:
                regions.append((bar * _BEATS_PER_BAR,
                                (bar + bars) * _BEATS_PER_BAR))
            bar += bars
        if not regions:
            total = song_duration_beats(song)
            if total > 0:
                regions = [(total / 3.0, 2.0 * total / 3.0)]
        return regions
    except Exception:  # noqa: BLE001
        _log.debug("chorus_regions failed", exc_info=True)
        return []


def _default_tts(text: str, out_path: str) -> dict[str, Any]:
    """Synthesize text -> wav via the profile-appropriate TTS backend.

    audience="private": this is the owner's own song, so XTTS cloning is
    allowed where installed; on Termux the engine falls through to
    whatever is actually usable (Chatterbox/Piper/Kokoro/system).
    Never raises — returns {"ok", "path"|"reason"}.
    """
    try:
        from ..voice.tts import UniversalTTS, available_backends
        if not available_backends():
            return {"ok": False,
                    "reason": "no TTS backend installed — vocal track skipped"}
        engine = UniversalTTS(backend="auto", audience="private")
        res = engine.speak(text, out_path=out_path)
        path = str(res.get("path", "") or "")
        if not path or not os.path.isfile(path):
            return {"ok": False,
                    "reason": "TTS produced no audio — vocal track skipped"}
        return {"ok": True, "path": path,
                "backend": str(res.get("backend", "")),
                "sample_rate": int(res.get("sample_rate", 0) or 0)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False,
                "reason": f"TTS vocal failed ({exc}) — vocal track skipped"}


def _read_wav_mono(path: str) -> tuple[array, int]:
    """WAV -> (mono float samples as array('d'), sample_rate)."""
    with wave.open(path, "rb") as wf:
        nch = wf.getnchannels()
        sr = wf.getframerate()
        width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    n = len(raw) // max(1, width)
    if width == 1:
        vals = [(b - 128) / 128.0 for b in raw[:n]]
    elif width == 2:
        import struct
        vals = [v / 32768.0
                for v in struct.unpack("<%dh" % n, raw[: n * 2])]
    elif width == 4:
        import struct
        vals = [v / 2147483648.0
                for v in struct.unpack("<%di" % n, raw[: n * 4])]
    else:
        vals = [0.0] * n
    if nch > 1 and n:
        frames = n // nch
        vals = [sum(vals[i * nch:(i + 1) * nch]) / nch
                for i in range(frames)]
    return array("d", vals), int(sr or _SAMPLE_RATE)


def _resample_linear(samples: array, src_sr: int, dst_sr: int) -> array:
    """Linear-interpolation resample. Never raises."""
    try:
        if src_sr == dst_sr or len(samples) < 2:
            return array("d", samples)
        ratio = dst_sr / src_sr
        n = max(1, int(round(len(samples) * ratio)))
        out = array("d", [0.0]) * n
        last = len(samples) - 1
        for i in range(n):
            pos = i / ratio
            lo = int(pos)
            hi = min(last, lo + 1)
            frac = pos - lo
            out[i] = samples[lo] * (1.0 - frac) + samples[hi] * frac
        return out
    except Exception:  # noqa: BLE001
        return array("d", samples)


def _tile_vocal(vocal: array, regions: list[tuple[float, float]],
                tempo: float, total_beats: float) -> array:
    """Tile the hook across chorus regions with gaps and edge fades.

    Returns a full-song-length mono buffer at _SAMPLE_RATE. Never raises.
    """
    try:
        beat_s = 60.0 / max(20.0, float(tempo or 100.0))
        total_n = int(total_beats * beat_s * _SAMPLE_RATE) + _SAMPLE_RATE
        out = array("d", [0.0]) * total_n
        if not vocal or not regions:
            return out
        vlen = len(vocal)
        gap_n = int(_TILE_GAP_S * _SAMPLE_RATE)
        fade_n = max(1, int(_TILE_FADE_S * _SAMPLE_RATE))
        # pre-faded tile
        tile = array("d", vocal)
        for i in range(min(fade_n, vlen)):
            f = i / fade_n
            tile[i] *= f
            tile[vlen - 1 - i] *= f
        for start_b, end_b in regions:
            start_n = int(start_b * beat_s * _SAMPLE_RATE)
            end_n = int(end_b * beat_s * _SAMPLE_RATE)
            pos = start_n
            while pos + vlen <= end_n and pos < total_n:
                lim = min(total_n, pos + vlen)
                for i in range(lim - pos):
                    out[pos + i] += tile[i]
                pos += vlen + gap_n
        return out
    except Exception:  # noqa: BLE001
        _log.debug("tile_vocal failed", exc_info=True)
        return array("d", [0.0])


def mix_vocal_track(bed_path: str, vocal_path: str,
                    regions: list[tuple[float, float]], tempo: float,
                    total_beats: float, out_path: str,
                    vocal_gain: float = _VOCAL_GAIN) -> str:
    """Mix the TTS hook under the instrumental bed -> out_path.

    Pure stdlib DSP (wave/array): resample vocal to the bed rate, tile
    across chorus regions, sum under the bed, peak-normalize with the
    same soft-clip curve as the builtin synth. Never raises.
    """
    bed, bed_sr = _read_wav_mono(bed_path)
    voc, voc_sr = _read_wav_mono(vocal_path)
    voc = _resample_linear(voc, voc_sr, _SAMPLE_RATE)
    tiled = _tile_vocal(voc, regions, tempo, total_beats)
    tiled = _resample_linear(tiled, _SAMPLE_RATE, bed_sr)
    n = max(len(bed), len(tiled))
    mix = array("d", [0.0]) * n
    for i in range(len(bed)):
        mix[i] += bed[i]
    for i in range(len(tiled)):
        mix[i] += tiled[i] * vocal_gain
    # peak-normalize + gentle soft clip (same curve as synth.mix_tracks)
    peak = 0.0
    for s in mix:
        a = abs(s)
        if a > peak:
            peak = a
    if peak > 0:
        norm = 0.89 / peak
        for i, s in enumerate(mix):
            v = s * norm
            # tanh approximation without importing math per-sample cost
            # concerns — math.tanh is fine here (one pass over the mix)
            import math
            mix[i] = math.tanh(v * 1.2) * 0.95
    import struct
    pcm = array("h", (max(-32768, min(32767, int(s * 32767)))
                      for s in mix))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(bed_sr)
        wf.writeframes(pcm.tobytes())
    return out_path


def render_hook_vocal(song: Any, workdir: str,
                      tts_fn: Any = None,
                      profile: str = "") -> dict[str, Any]:
    """Synthesize the chorus hook -> wav path. Never raises.

    Returns {"ok": True, "path", "backend", "note"} or
    {"ok": False, "reason"}. ``tts_fn`` is injectable for tests:
    ``tts_fn(text, out_path) -> {"ok", "path"|"reason"}``.
    """
    try:
        lines = hook_lyrics(song)
        if not lines:
            return {"ok": False, "reason": "song has no lyrics to sing"}
        # the hook: first few lines, spoken as one phrase
        hook_text = ". ".join(l.strip().rstrip(".") for l in lines[:4]
                              if l.strip())
        if not hook_text:
            return {"ok": False, "reason": "hook text is empty"}
        prof = (profile or "").strip().lower() or _profile_kind()
        base = Path(workdir)
        base.mkdir(parents=True, exist_ok=True)
        target = str(base / "hook_vocal.wav")
        fn = tts_fn or _default_tts
        res = fn(hook_text, target)
        if not isinstance(res, dict) or not res.get("ok"):
            reason = (res.get("reason") if isinstance(res, dict)
                      else "TTS failed")
            return {"ok": False, "reason": str(reason or "TTS failed")}
        path = str(res.get("path") or "")
        if not path or not os.path.isfile(path):
            return {"ok": False, "reason": "TTS produced no audio file"}
        backend = str(res.get("backend") or "tts")
        note = (f"hook vocal via {backend} (lightweight TTS; "
                f"profile {prof or 'unknown'})")
        return {"ok": True, "path": path, "backend": backend, "note": note,
                "text": hook_text}
    except Exception as exc:  # noqa: BLE001
        _log.debug("render_hook_vocal failed", exc_info=True)
        return {"ok": False, "reason": f"vocal render failed: {exc}"}


def add_vocal_track(song: Any, bed_path: str, workdir: str,
                    tts_fn: Any = None, profile: str = "",
                    vocal_gain: float = _VOCAL_GAIN) -> dict[str, Any]:
    """Add the TTS hook vocal under an instrumental bed.

    Returns {"ok": True, "path": final_wav, "note"} — or
    {"ok": False, "reason", "path": bed_path} when vocals are skipped
    honestly (no TTS backend, no lyrics). The caller keeps the bed
    either way. Never raises.
    """
    try:
        if not bed_path or not os.path.isfile(bed_path):
            return {"ok": False, "reason": "no bed audio to sing over",
                    "path": bed_path or ""}
        vr = render_hook_vocal(song, workdir, tts_fn=tts_fn, profile=profile)
        if not vr.get("ok"):
            return {"ok": False, "reason": str(vr.get("reason", "")),
                    "path": bed_path}
        regions = chorus_regions(song)
        total_beats = song_duration_beats(song)
        tempo = float(getattr(song, "tempo", 100) or 100)
        out_path = str(Path(workdir) / "song_with_vocals.wav")
        mix_vocal_track(bed_path, str(vr["path"]), regions, tempo,
                        total_beats, out_path, vocal_gain=vocal_gain)
        note = str(vr.get("note", "")) + " — mixed under the bed"
        return {"ok": True, "path": out_path, "note": note,
                "backend": str(vr.get("backend", ""))}
    except Exception as exc:  # noqa: BLE001
        _log.debug("add_vocal_track failed", exc_info=True)
        return {"ok": False, "reason": f"vocal track failed: {exc}",
                "path": bed_path or ""}

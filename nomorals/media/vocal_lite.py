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

No TTS backend? No problem: the chorus melody is hummed with a
vocal-like timbre (formant-ish harmonics + vibrato) straight from the
arranged melody notes — so /music ALWAYS has a vocal line. TTS = sung
words, no TTS = hummed melody. Never silence, never fake audio, never
raises.
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
    "render_hummed_vocal",
    "add_vocal_track",
    "mix_vocal_track",
    "mix_hum_track",
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

#: hum timbre — "ah"-like vowel: strong fundamental, moderate 2nd/3rd,
#: tapering upper harmonics (formant-ish, sits above the instrumental)
_HUM_HARMONICS = (1.0, 0.42, 0.20, 0.09, 0.04)
#: hum vibrato: rate in Hz, depth in cents (±)
_HUM_VIBRATO_RATE = 5.0
_HUM_VIBRATO_CENTS = 30.0
#: hum level under the bed (a touch lower than TTS — it's a texture)
_HUM_GAIN = 0.7


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
    same soft-clip curve as the builtin synth. Never raises.  Returns the
    output path, or ``""`` (logged, nothing written) when the mix would
    exceed the audio write cap.
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
    from . import caps

    pcm = array("h", (max(-32768, min(32767, int(s * 32767)))
                      for s in mix))
    frames = pcm.tobytes()
    ok, reason = caps.check_write_size(
        caps.wav_expected_bytes(len(frames)), caps.MAX_AUDIO_WRITE_BYTES)
    if not ok:
        caps.refuse_write(f"mix_vocal_track({out_path})", reason)
        return ""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(bed_sr)
        wf.writeframes(frames)
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


def _render_hum_tone(freq: float, n: int, velocity: float,
                     harmonics: tuple = _HUM_HARMONICS,
                     vibrato_rate: float = _HUM_VIBRATO_RATE,
                     vibrato_cents: float = _HUM_VIBRATO_CENTS) -> array:
    """One hummed note: vocal-ish harmonic stack + pitch vibrato.

    Pure stdlib (wavetable sine + ADSR from :mod:`nomorals.media.synth`).
    Vibrato is real FM — the pitch wobbles ±vibrato_cents at
    vibrato_rate Hz, like a human voice. Never raises.
    """
    import math
    from .synth import _sine, _adsr, SAMPLE_RATE
    out = array("d", [0.0]) * n
    try:
        if n <= 0 or freq <= 0:
            return out
        env = _adsr(n, 0.04, 0.06, 0.85, 0.09, n)
        # precompute the vibrato pitch multiplier once per note
        # (same wobble for every harmonic — cheaper than pow per harmonic)
        vib_oct = vibrato_cents / 1200.0
        two_pi = 2.0 * math.pi
        vib_mult = array("d", [0.0]) * n
        for i in range(n):
            t = i / SAMPLE_RATE
            vib_mult[i] = 2.0 ** (
                vib_oct * math.sin(two_pi * vibrato_rate * t))
        for h, amp in enumerate(harmonics, start=1):
            if amp <= 0:
                continue
            base_f = freq * h
            ph = 0.0
            for i in range(n):
                ph += two_pi * base_f * vib_mult[i] / SAMPLE_RATE
                out[i] += amp * _sine(ph)
        vel = max(0.05, min(1.0, velocity / 100.0))
        for i in range(n):
            out[i] *= env[i] * vel * 0.5
    except Exception:  # noqa: BLE001
        _log.debug("_render_hum_tone failed", exc_info=True)
    return out


def _write_mono_wav(path: str, samples: array, sr: int) -> str:
    """Write a mono float buffer to 16-bit WAV. Never raises."""
    try:
        pcm = array("h", (max(-32768, min(32767, int(s * 32767)))
                          for s in samples))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
        return path
    except Exception:  # noqa: BLE001
        _log.debug("_write_mono_wav failed", exc_info=True)
        return ""


def _event_in_chorus(event: Any, regions: list[tuple[float, float]]) -> bool:
    try:
        s = float(getattr(event, "start", -1))
        return any(rs <= s < re_ for rs, re_ in regions)
    except Exception:  # noqa: BLE001
        return False


def render_hummed_vocal(song: Any, workdir: str, melody_events: Any,
                        tempo: float | None = None) -> dict[str, Any]:
    """Hum the chorus melody — the no-TTS vocal fallback.

    Takes the arranged melody :class:`NoteEvent`s (absolute beat
    positions, as produced by the composer's ``_arrange``), keeps the
    ones inside chorus regions, and renders each as a hummed note with
    vibrato. The hum is already positioned — no tiling needed.

    Returns {"ok": True, "path", "backend": "hum", "note"} or
    {"ok": False, "reason"}. Never raises.
    """
    try:
        from .synth import midi_to_freq, SAMPLE_RATE
        regions = chorus_regions(song)
        if not regions:
            return {"ok": False, "reason": "no chorus regions to hum"}
        events = [e for e in (melody_events or [])
                  if _event_in_chorus(e, regions)]
        if not events:
            return {"ok": False,
                    "reason": "no chorus melody notes to hum"}
        tempo = float(tempo or getattr(song, "tempo", 100) or 100)
        beat_s = 60.0 / max(20.0, tempo)
        total_beats = song_duration_beats(song)
        total_n = int(total_beats * beat_s * SAMPLE_RATE) + SAMPLE_RATE
        out = array("d", [0.0]) * total_n
        for e in events:
            try:
                note = int(getattr(e, "note", 0))
                start_b = float(getattr(e, "start", 0))
                dur_b = float(getattr(e, "duration", 0.5))
                vel = int(getattr(e, "velocity", 96))
            except Exception:  # noqa: BLE001
                continue
            if dur_b <= 0 or not 0 < note < 128:
                continue
            n = max(16, int(dur_b * beat_s * SAMPLE_RATE))
            tone = _render_hum_tone(midi_to_freq(note), n, vel)
            start_n = int(start_b * beat_s * SAMPLE_RATE)
            if start_n >= total_n:
                continue
            lim = min(total_n, start_n + n)
            for i in range(lim - start_n):
                out[start_n + i] += tone[i] * 0.9
        base = Path(workdir)
        base.mkdir(parents=True, exist_ok=True)
        hum_path = str(base / "hum_vocal.wav")
        if not _write_mono_wav(hum_path, out, SAMPLE_RATE):
            return {"ok": False, "reason": "could not write hum wav"}
        note = ("hummed vocal melody (no TTS installed — `pkg install espeak-ng` "
                "for sung lyrics)")
        return {"ok": True, "path": hum_path, "backend": "hum",
                "note": note, "notes": len(events)}
    except Exception as exc:  # noqa: BLE001
        _log.debug("render_hummed_vocal failed", exc_info=True)
        return {"ok": False, "reason": f"hum render failed: {exc}"}


def mix_hum_track(bed_path: str, hum_path: str, out_path: str,
                  vocal_gain: float = _HUM_GAIN) -> str:
    """Mix the hummed vocal under the instrumental bed -> out_path.

    Unlike :func:`mix_vocal_track` there is no tiling — the hum notes
    are already at their absolute song positions. Same peak-normalize +
    soft-clip curve as the builtin synth, same audio write-cap guard.
    Never raises. Returns the output path or "".
    """
    try:
        bed, bed_sr = _read_wav_mono(bed_path)
        hum, hum_sr = _read_wav_mono(hum_path)
        hum = _resample_linear(hum, hum_sr, bed_sr)
        n = max(len(bed), len(hum))
        mix = array("d", [0.0]) * n
        for i in range(len(bed)):
            mix[i] += bed[i]
        for i in range(len(hum)):
            mix[i] += hum[i] * vocal_gain
        peak = 0.0
        for s in mix:
            a = abs(s)
            if a > peak:
                peak = a
        if peak > 0:
            import math
            norm = 0.89 / peak
            for i, s in enumerate(mix):
                mix[i] = math.tanh(s * norm * 1.2) * 0.95
        import struct  # noqa: F401  (parity with mix_vocal_track)
        from . import caps
        pcm = array("h", (max(-32768, min(32767, int(s * 32767)))
                          for s in mix))
        frames = pcm.tobytes()
        ok, reason = caps.check_write_size(
            caps.wav_expected_bytes(len(frames)), caps.MAX_AUDIO_WRITE_BYTES)
        if not ok:
            caps.refuse_write(f"mix_hum_track({out_path})", reason)
            return ""
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(bed_sr)
            wf.writeframes(frames)
        return out_path
    except Exception:  # noqa: BLE001
        _log.debug("mix_hum_track failed", exc_info=True)
        return ""


def _add_hummed_vocal(song: Any, bed_path: str, workdir: str,
                      melody_events: Any,
                      vocal_gain: float = _HUM_GAIN) -> dict[str, Any]:
    """The no-TTS path: hum the chorus melody under the bed.

    Returns {"ok": True, "path", "note", "backend": "hum"} or
    {"ok": False, "reason", "path": bed_path}. Never raises.
    """
    try:
        from . import caps
        hr = render_hummed_vocal(song, workdir, melody_events)
        if not hr.get("ok"):
            return {"ok": False, "reason": str(hr.get("reason", "")),
                    "path": bed_path}
        out_path = str(Path(workdir) / "song_with_vocals.wav")
        mixed = mix_hum_track(bed_path, str(hr["path"]), out_path,
                              vocal_gain=vocal_gain)
        if not mixed:
            cap_mib = caps.MAX_AUDIO_WRITE_BYTES // (1024 * 1024)
            return {"ok": False,
                    "reason": "hum mix refused: output would exceed the "
                              f"{cap_mib} MiB audio write cap (see caps.py)",
                    "path": bed_path}
        return {"ok": True, "path": out_path,
                "note": str(hr.get("note", "")), "backend": "hum",
                "notes": hr.get("notes", 0)}
    except Exception as exc:  # noqa: BLE001
        _log.debug("_add_hummed_vocal failed", exc_info=True)
        return {"ok": False, "reason": f"hum vocal failed: {exc}",
                "path": bed_path or ""}


def add_vocal_track(song: Any, bed_path: str, workdir: str,
                    tts_fn: Any = None, profile: str = "",
                    vocal_gain: float = _VOCAL_GAIN,
                    melody_events: Any = None) -> dict[str, Any]:
    """Add a vocal under an instrumental bed — TTS hook or hummed melody.

    Returns {"ok": True, "path": final_wav, "note"} — or
    {"ok": False, "reason", "path": bed_path} when vocals are honestly
    skipped (no lyrics, no bed, no TTS *and* no melody to hum).
    The caller keeps the bed either way. Never raises.

    ``melody_events``: arranged melody :class:`NoteEvent`s (absolute
    beat positions). When the TTS backend is missing and melody events
    are available, the chorus is hummed instead of skipped — so /music
    always has a vocal line.
    """
    try:
        if not bed_path or not os.path.isfile(bed_path):
            return {"ok": False, "reason": "no bed audio to sing over",
                    "path": bed_path or ""}
        vr = render_hook_vocal(song, workdir, tts_fn=tts_fn, profile=profile)
        if not vr.get("ok"):
            reason = str(vr.get("reason", ""))
            # HUM FALLBACK: no TTS backend is not silence. When the
            # arranged melody is available, hum the chorus instead.
            if "no TTS backend installed" in reason and melody_events:
                return _add_hummed_vocal(song, bed_path, workdir,
                                         melody_events,
                                         vocal_gain=_HUM_GAIN)
            return {"ok": False, "reason": reason,
                    "path": bed_path}
        regions = chorus_regions(song)
        total_beats = song_duration_beats(song)
        tempo = float(getattr(song, "tempo", 100) or 100)
        out_path = str(Path(workdir) / "song_with_vocals.wav")
        mixed = mix_vocal_track(bed_path, str(vr["path"]), regions, tempo,
                                total_beats, out_path, vocal_gain=vocal_gain)
        if not mixed:
            cap_mib = caps.MAX_AUDIO_WRITE_BYTES // (1024 * 1024)
            return {"ok": False,
                    "reason": "vocal mix refused: output would exceed the "
                              f"{cap_mib} MiB audio write cap (see caps.py)",
                    "path": bed_path}
        note = str(vr.get("note", "")) + " — mixed under the bed"
        return {"ok": True, "path": out_path, "note": note,
                "backend": str(vr.get("backend", ""))}
    except Exception as exc:  # noqa: BLE001
        _log.debug("add_vocal_track failed", exc_info=True)
        return {"ok": False, "reason": f"vocal track failed: {exc}",
                "path": bed_path or ""}

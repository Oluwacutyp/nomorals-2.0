"""Multi-speaker dialogue — works on ANY backend.

Dia does ``[S1]``/``[S2]`` natively (GPU-only). This module brings
dialogue to every backend: split the script by speaker, synthesize
each turn with that speaker's voice, stitch with natural pauses.

Format:
    [S1: Zara] Hey, have you heard the new track?
    [S2: Kilo] [laughs] Yeah, it's fire.

Speaker names map to catalogue voices. Unknown speakers get the
default voice with a pitch offset so they're distinguishable.
"""

import re
from array import array
from dataclasses import dataclass
from typing import Any


TURN_RE = re.compile(r"\[S(\d+)(?::\s*([^\]]+))?\]")


@dataclass
class Turn:
    speaker_id: str      # "1", "2", ...
    speaker_name: str    # "Zara", "" if unnamed
    text: str            # the line (may contain emotion tags)


def parse_dialogue(script: str) -> list[Turn]:
    """Split a dialogue script into speaker turns."""
    turns: list[Turn] = []
    # Find all speaker markers
    markers = list(TURN_RE.finditer(script))
    if not markers:
        return [Turn(speaker_id="1", speaker_name="", text=script.strip())]

    for i, m in enumerate(markers):
        start = m.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(script)
        text = script[start:end].strip()
        if not text:
            continue
        turns.append(Turn(
            speaker_id=m.group(1),
            speaker_name=(m.group(2) or "").strip(),
            text=text,
        ))
    return turns


def stitch_turns(turn_audios: list[tuple[array, int]],
                 pause_ms: int = 400) -> tuple[array, int]:
    """Stitch turn audios with natural inter-turn pauses.

    All inputs must share the sample rate (resample upstream).
    Returns (samples, sample_rate).
    """
    if not turn_audios:
        return array("h"), 24000
    sr = turn_audios[0][1]
    pause_samples = int(sr * pause_ms / 1000)
    out = array("h")
    for i, (samples, _) in enumerate(turn_audios):
        if i > 0:
            out.extend([0] * pause_samples)
        out.extend(samples)
    return out, sr


def dialogue_needs_backend_split(backend_name: str) -> bool:
    """True if this backend can't do multi-speaker natively."""
    # Dia handles [S1]/[S2] in-model. Everyone else gets split.
    return backend_name != "dia"


def _turn_pause_ms(text: str) -> int:
    """Natural inter-turn pause from the turn's ending punctuation."""
    t = (text or "").rstrip()
    if t.endswith(("...", "…")):
        return 700
    if t.endswith(("!", "?")):
        return 550
    if t.endswith(","):
        return 300
    if t.endswith((".", ":", ";")):
        return 450
    return 350


def estimate_duration(turn_audios: list[tuple[array, int]],
                      pause_ms: int = 400) -> float:
    """Seconds the stitched dialogue will run (before rendering)."""
    if not turn_audios:
        return 0.0
    sr = turn_audios[0][1]
    total = sum(len(s) for s, _ in turn_audios) / max(1, sr)
    total += pause_ms / 1000.0 * max(0, len(turn_audios) - 1)
    return round(total, 2)


def render_dialogue(script: str, tts: Any,
                    voice_map: dict[str, str] | None = None,
                    pause_ms: int | None = None,
                    resample_to: int = 0) -> dict[str, Any]:
    """Full dialogue pipeline: parse → per-turn synth → stitch.

    ``tts``: a UniversalTTS (or anything with ``speak(text,
    voice_name=...)`` → ``{"path"}``). ``voice_map``: speaker id or
    name → voice name (``{"1": "zara", "Zara": "zara"}``); unknown
    speakers fall back to the engine default with a pitch offset per
    speaker index so they stay distinguishable.

    ``pause_ms`` None → per-turn punctuation pauses. Returns
    ``{"ok", "path", "turns", "seconds", "sample_rate", "voices"}``.
    """
    import wave as _wave
    turns = parse_dialogue(script)
    turns = [t for t in turns if (t.text or "").strip()]
    if not turns:
        return {"ok": False, "reason": "no turns parsed"}
    voice_map = voice_map or {}
    rendered: list[tuple[array, int]] = []
    used_voices: list[str] = []
    sr0 = 0
    for idx, turn in enumerate(turns):
        vname = (voice_map.get(turn.speaker_id)
                 or voice_map.get(turn.speaker_name) or "")
        pitch_shift_st = 0.0
        if not vname:
            # pitch-offset fallback: ±2 semitones per speaker index
            pitch_shift_st = ((idx % 4) - 1.5) * 2.0
        result = tts.speak(turn.text, voice_name=vname or None)
        path = result.get("path", "")
        if not path:
            return {"ok": False,
                    "reason": f"turn {idx + 1} produced no audio"}
        with _wave.open(path, "rb") as w:
            sr = w.getframerate()
            samples = array("h", w.readframes(w.getnframes()))
        if pitch_shift_st:
            from .emotion_dsp import pitch_shift
            samples = pitch_shift(samples, pitch_shift_st, sr)
        if resample_to and sr != resample_to:
            from .tts import _resample_linear
            floats = [v / 32768.0 for v in samples]
            floats = _resample_linear(floats, sr, resample_to)
            samples = array(
                "h", [int(max(-32768, min(32767, v * 32767)))
                      for v in floats])
            sr = resample_to
        if not sr0:
            sr0 = sr
        elif sr != sr0:
            raise RuntimeError(
                f"sample-rate drift between turns ({sr0} vs {sr}) — "
                "pass resample_to to normalize")
        rendered.append((samples, sr))
        used_voices.append(vname or f"pitch-shifted default "
                           f"({pitch_shift_st:+.0f}st)")
    # stitch with per-turn punctuation pauses
    out = array("h")
    for i, (samples, _) in enumerate(rendered):
        if i > 0:
            pm = pause_ms if pause_ms is not None else _turn_pause_ms(
                turns[i - 1].text)
            out.extend([0] * int(sr0 * pm / 1000))
        out.extend(samples)
    import tempfile
    dest = tempfile.mktemp(prefix="dialogue_", suffix=".wav")
    with _wave.open(dest, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr0)
        w.writeframes(out.tobytes())
    return {"ok": True, "path": dest, "turns": len(turns),
            "seconds": round(len(out) / sr0, 2),
            "sample_rate": sr0, "voices": used_voices}

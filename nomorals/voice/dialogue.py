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

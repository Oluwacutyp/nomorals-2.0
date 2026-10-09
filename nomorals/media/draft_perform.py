"""Perform a song draft — beat and vocals shaped together from the draft.

The draft (:mod:`nomorals.media.song_draft`) is the score. This module turns
it into sound:

- :func:`draft_melody_events` — every lyric line's melody contour becomes
  real :class:`~nomorals.core.midi.NoteEvent`\\ s: the contour walks in
  semitones from a register set by the section's energy. Vocals follow the
  *drafted* melody, not a generic arranged one.
- :func:`perform_draft` — full pipeline: instrumental bed via
  :class:`~nomorals.media.music.MusicCreator`, draft lyrics injected,
  draft-driven vocal via :mod:`nomorals.media.vocal_lite`. The draft's
  energy arc sets vocal dynamics per section.

This is the "beat and vocals working together" requirement: both come from
the same draft instead of being generated independently and layered.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_log = logging.getLogger(__name__)

_KEY_ROOTS = {"C": 60, "C#": 61, "D": 62, "D#": 63, "E": 64, "F": 65,
              "F#": 66, "G": 67, "G#": 68, "A": 69, "A#": 70, "B": 71}

# contour move → semitone step (a walk, not jumps, unless marked)
_CONTOUR_STEPS = {"up": 2, "down": -2, "hold": 0, "leap": 7, "drop": -7}


def _base_midi(key: str, energy: float) -> int:
    """Register for a section: higher energy sings higher."""
    root = _KEY_ROOTS.get((key or "C").upper().replace("B", "B"), 60)
    # energy 0.0 → root, 1.0 → root + octave-ish lift into the chorus belt
    return root + int(energy * 7)


def draft_melody_events(draft: Any) -> list[Any]:
    """Build melody NoteEvents from the draft's per-line contours.

    Each line's contour walks in semitones; lines split their section's
    bars evenly. Higher-energy sections sing in a higher register with
    stronger velocity — the draft's energy arc becomes dynamics.
    """
    from ..core.midi import NoteEvent

    events: list[Any] = []
    beats_per_bar = 4
    bar = 0
    for sec in (draft.sections or []):
        sec_bars = max(1, int(sec.bars or 4))
        sec_start = bar * beats_per_bar
        lines = [ln for ln in (sec.lines or []) if (ln.text or "").strip()]
        base = _base_midi(getattr(draft, "key", "C"),
                          float(sec.energy or 0.5))
        velocity = 70 + int(float(sec.energy or 0.5) * 40)  # 70–110
        if lines:
            beats_per_line = (sec_bars * beats_per_bar) / len(lines)
            for li, ln in enumerate(lines):
                contour = list(ln.contour or ["hold"])
                n_notes = max(1, len(contour))
                note_dur = beats_per_line / n_notes
                pitch = base
                # small per-line variation so lines don't all start identical
                pitch += (li % 3 - 1) * 2
                for ni, move in enumerate(contour):
                    pitch += _CONTOUR_STEPS.get(str(move).lower(), 0)
                    # keep the voice in a singable band around the base
                    pitch = max(base - 7, min(base + 12, pitch))
                    start = sec_start + li * beats_per_line + ni * note_dur
                    events.append(NoteEvent(
                        note=int(pitch), start=round(start, 3),
                        duration=round(note_dur * 0.92, 3),
                        velocity=velocity))
        bar += sec_bars
    _log.info("draft melody: %d events from %d sections",
              len(events), len(draft.sections or []))
    return events


def _song_from_draft(draft: Any, context: Any, workdir: str) -> Any:
    """Build a MusicCreator Song carrying the draft's lyrics and timing.

    The bed arrangement comes from MusicCreator (real instrumentation);
    section names/bars/lyrics come from the draft so the vocal renderer
    lines everything up.
    """
    from .music import MusicCreator, Section

    creator = MusicCreator(context)
    song = creator.compose(
        draft.topic or draft.title, style=draft.style or "pop",
        title=draft.title, key=draft.key or "C",
        with_midi=False, with_audio=False, with_vocals=False,
        with_score=False, workdir=workdir)
    # override tempo with the draft's
    if getattr(draft, "tempo", 0):
        song.tempo = int(draft.tempo)
    # rebuild sections from the draft (name/bars/lyrics drive the vocal)
    new_sections = []
    for sec in (draft.sections or []):
        lyrics = [(ln.text or "") for ln in (sec.lines or [])
                  if (ln.text or "").strip()]
        new_sections.append(Section(name=sec.name or "verse",
                                    bars=max(1, int(sec.bars or 4)),
                                    lyrics=lyrics))
    if new_sections:
        song.sections = new_sections
    return song, creator


def perform_draft(draft: Any, context: Any = None,
                  workdir: str = "music") -> dict[str, Any]:
    """Perform a song draft end to end → {"ok", "path", "note", ...}.

    Bed from MusicCreator, vocals from the draft's own melody contours via
    vocal_lite. Never raises — returns {"ok": False, "reason"} on failure.
    """
    try:
        if draft is None or not getattr(draft, "sections", None):
            return {"ok": False, "reason": "no draft to perform"}
        if context is None:
            return {"ok": False,
                    "reason": "an LLM/render context is required"}

        song, creator = _song_from_draft(draft, context, workdir)

        # render the bed (instrumental only)
        try:
            bed_path = creator._render_audio(song, workdir, with_vocals=False)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": f"bed render failed: {exc}"}

        # draft-driven vocal: contours → melody events → sung vocal
        melody = draft_melody_events(draft)
        try:
            from .vocal_lite import add_vocal_track
            from ..tools.filesystem import safe_path
            base = safe_path(context, (workdir or "music").strip("/"))
            vr = add_vocal_track(song, str(bed_path), str(base),
                                 melody_events=melody)
        except Exception as exc:  # noqa: BLE001
            return {"ok": True, "path": str(bed_path),
                    "note": f"bed only — draft vocal failed: {exc}",
                    "bed_path": str(bed_path), "draft_title": draft.title}
        if vr.get("ok"):
            return {"ok": True, "path": str(vr["path"]),
                    "note": f"🎵 \"{draft.title}\" performed from draft — "
                            f"{vr.get('note', '')}",
                    "bed_path": str(bed_path),
                    "draft_title": draft.title,
                    "vocal_mode": vr.get("vocal_mode", "")}
        return {"ok": True, "path": str(bed_path),
                "note": f"bed only — {vr.get('reason', 'vocal skipped')}",
                "bed_path": str(bed_path), "draft_title": draft.title}
    except Exception as exc:  # noqa: BLE001
        _log.warning("perform_draft failed: %s", exc)
        return {"ok": False, "reason": str(exc)}

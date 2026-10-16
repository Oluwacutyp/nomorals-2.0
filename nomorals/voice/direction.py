"""Voice direction track — performance direction kept separate from dialogue.

A :class:`VoiceDirectionTrack` is a sidecar, never an edit of the
dialogue: the dialogue text stays exactly as written, and the track
carries the *performance* — prosody (pitch/rate/volume), emotion, and
timing (pauses, emphasis) — anchored to word ranges. TTS consumes the
pair ``(dialogue, track)``: :func:`segments` splits the dialogue into
(text, cue) pairs by word anchor, and :func:`render_for_tts` renders
each segment through the director with the cue's emotion, so the
backend hears the direction without the dialogue ever being rewritten
to carry it.

Anchor model: a cue covers ``word_start`` (inclusive) to ``word_end``
(exclusive) in the dialogue's whitespace-split words. ``word_start`` /
``word_end`` of ``None`` means "to the start / to the end". Cues must
not overlap; gaps between cues render with the track defaults.

Invariant: joining the segment texts with single spaces reproduces the
dialogue exactly (whitespace-normalized) — the track can never drop,
duplicate, or reorder a word.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.errors import ValidationError
from .director import PerformanceScript, direct, render_for

__all__ = [
    "DirectionCue",
    "VoiceDirectionTrack",
    "segments",
    "render_for_tts",
    "synthesize_with_direction",
]


@dataclass
class DirectionCue:
    """One directed span: word range + how to perform it."""

    word_start: int | None = None
    word_end: int | None = None
    emotion: str = ""
    prosody: dict[str, float] = field(default_factory=dict)
    pause_before_s: float = 0.0
    pause_after_s: float = 0.0
    emphasis: bool = False

    def __post_init__(self) -> None:
        if self.word_start is not None and self.word_start < 0:
            raise ValidationError("cue word_start must be >= 0")
        if self.word_end is not None and self.word_end < 0:
            raise ValidationError("cue word_end must be >= 0")
        if (self.word_start is not None and self.word_end is not None
                and not self.word_start < self.word_end):
            raise ValidationError(
                f"cue needs word_start < word_end, got "
                f"{self.word_start}..{self.word_end}")
        for key in ("pitch", "rate", "volume"):
            if key in self.prosody:
                val = float(self.prosody[key])
                if not (0.25 <= val <= 4.0):
                    raise ValidationError(
                        f"prosody {key} must be in [0.25, 4.0], got {val}")
                self.prosody[key] = val
        if self.pause_before_s < 0 or self.pause_after_s < 0:
            raise ValidationError("cue pauses must be >= 0")
        self.emotion = str(self.emotion or "").strip().lower()

    def to_dict(self) -> dict[str, Any]:
        return {"word_start": self.word_start, "word_end": self.word_end,
                "emotion": self.emotion, "prosody": dict(self.prosody),
                "pause_before_s": self.pause_before_s,
                "pause_after_s": self.pause_after_s,
                "emphasis": self.emphasis}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DirectionCue":
        return cls(
            word_start=data.get("word_start"),
            word_end=data.get("word_end"),
            emotion=str(data.get("emotion", "") or ""),
            prosody={k: float(v) for k, v in
                     dict(data.get("prosody") or {}).items()},
            pause_before_s=float(data.get("pause_before_s", 0.0)),
            pause_after_s=float(data.get("pause_after_s", 0.0)),
            emphasis=bool(data.get("emphasis", False)))


@dataclass
class VoiceDirectionTrack:
    """The whole performance: cues + defaults for the gaps between them."""

    cues: list[DirectionCue] = field(default_factory=list)
    default_emotion: str = "neutral"
    default_prosody: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cues = [c if isinstance(c, DirectionCue)
                     else DirectionCue.from_dict(dict(c))
                     for c in self.cues]
        # cues must not overlap (checked on resolved word ranges —
        # open-ended cues resolve against a concrete dialogue in
        # segments(), so here we check only fully-bounded pairs)
        bounded = sorted(
            (c for c in self.cues
             if c.word_start is not None and c.word_end is not None),
            key=lambda c: c.word_start or 0)
        for prev, cur in zip(bounded, bounded[1:]):
            if (cur.word_start or 0) < (prev.word_end or 0):
                raise ValidationError(
                    f"direction cues overlap: {prev.to_dict()} vs "
                    f"{cur.to_dict()}")
        self.default_emotion = str(self.default_emotion or "neutral").strip().lower()

    def to_dict(self) -> dict[str, Any]:
        return {"cues": [c.to_dict() for c in self.cues],
                "default_emotion": self.default_emotion,
                "default_prosody": dict(self.default_prosody)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoiceDirectionTrack":
        return cls(
            cues=[DirectionCue.from_dict(c)
                  for c in data.get("cues", [])],
            default_emotion=str(data.get("default_emotion", "neutral") or "neutral"),
            default_prosody={k: float(v) for k, v in
                             dict(data.get("default_prosody") or {}).items()})

    def cue_for_word(self, index: int) -> DirectionCue | None:
        """The cue covering word ``index``, or None (track defaults apply)."""
        for cue in self.cues:
            start = cue.word_start if cue.word_start is not None else 0
            end = cue.word_end  # None = open-ended, handled by caller count
            if index >= start and (end is None or index < end):
                return cue
        return None


def segments(dialogue: str, track: VoiceDirectionTrack) -> list[dict[str, Any]]:
    """Split ``dialogue`` into (text, cue) pairs by the track's anchors.

    Consecutive words sharing a cue (or sharing "no cue") merge into one
    segment. The dialogue is never modified — segments are slices of
    its whitespace-split words.
    """
    words = (dialogue or "").split()
    if not words:
        return []
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(words):
        cue = track.cue_for_word(i)
        j = i + 1
        while j < len(words) and track.cue_for_word(j) is cue:
            j += 1
        out.append({"text": " ".join(words[i:j]),
                    "word_start": i, "word_end": j,
                    "cue": cue.to_dict() if cue else None,
                    "emotion": (cue.emotion if cue and cue.emotion
                                else track.default_emotion),
                    "prosody": dict(cue.prosody) if cue
                               else dict(track.default_prosody),
                    "pause_before_s": cue.pause_before_s if cue else 0.0,
                    "pause_after_s": cue.pause_after_s if cue else 0.0,
                    "emphasis": bool(cue.emphasis) if cue else False})
        i = j
    return out


def render_for_tts(dialogue: str, track: VoiceDirectionTrack, *,
                   backend: str = "plain") -> list[dict[str, Any]]:
    """Render each segment through the director → backend-native text.

    Returns one entry per segment: ``{"text", "rendered", "extra",
    "emotion", "prosody", "pause_before_s", "pause_after_s", ...}`` where
    ``rendered`` is what the TTS backend should speak (director tags in
    that backend's vocabulary) and ``text`` is the untouched dialogue
    slice. ``extra`` carries the CosyVoice-style instruction when the
    backend uses one.
    """
    rendered: list[dict[str, Any]] = []
    for seg in segments(dialogue, track):
        script: PerformanceScript = direct(
            seg["text"], mood=seg["emotion"] or "neutral", intensity=3)
        text, extra = render_for(backend, script)
        rendered.append({**seg, "rendered": text, "extra": extra})
    return rendered


def synthesize_with_direction(dialogue: str, track: VoiceDirectionTrack, *,
                              tts_fn: Callable[..., Any],
                              backend: str = "plain",
                              **tts_kwargs: Any) -> Any:
    """Hand ``(dialogue, track)`` to a TTS callable, direction intact.

    ``tts_fn`` receives ``rendered_segments`` (the director output from
    :func:`render_for_tts`), ``dialogue`` (verbatim), and
    ``track`` (the track dict) — the backend decides how to realize
    prosody/emotion/timing, but it can never claim the direction was
    part of the dialogue: the two arrive as separate arguments. Any
    extra ``tts_kwargs`` pass through untouched.
    """
    if not callable(tts_fn):
        raise ValidationError("synthesize_with_direction needs a tts_fn callable")
    return tts_fn(rendered_segments=render_for_tts(dialogue, track, backend=backend),
                   dialogue=dialogue, track=track.to_dict(), **tts_kwargs)

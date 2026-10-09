"""Documentary style preset — calm, patient, legible.

Slow ken-burns drift on stills, dissolve transitions, lower-third
titles, minimal captions, and music that sits quietly under narration
instead of fighting it. Composes engine primitives only.
"""

from __future__ import annotations

from typing import Any

from ...edit_engine import (
    AudioLayer,
    AudioMix,
    Clip,
    EditSpec,
    Effect,
    TextLayer,
)
from ...edit_engine.text import as_words, estimate_word_timings


def _coerce_clips(clips: Any, scenes: Any) -> list[Clip]:
    if scenes is not None and not clips:
        from ...edit_engine.spec import _scene_to_clip
        return [_scene_to_clip(s) for s in scenes]
    return [c if isinstance(c, Clip) else Clip.from_dict(dict(c))
            for c in (clips or [])]


def _slow_drift(clips: list[Clip], *, zoom: float = 1.15) -> list[Clip]:
    """Clips without effects get a slow ken-burns drift (alternating)."""
    out: list[Clip] = []
    for i, c in enumerate(clips):
        if c.effects:
            out.append(c)
            continue
        direction = "in" if i % 2 == 0 else "out"
        out.append(Clip(path=c.path, start=c.start, end=c.end,
                        speed=c.speed, gain=c.gain,
                        effects=(Effect(name="ken_burns",
                                        params={"zoom": zoom,
                                                "direction": direction}),)))
    return out


def build_spec(*, clips: Any = (), scenes: Any = None,
               beats: Any = None, beat_audio: Any = None,
               voiceover: Any = None, music: Any = None,
               captions: Any = None,
               title: str | None = None,
               transition: Any = "dissolve",
               transition_duration: float = 0.8,
               caption_style: str = "minimal",
               caption_wpm: float = 150.0,
               seed: int = 0,
               width: int = 1080, height: int = 1920, fps: int = 30,
               out: Any = None, keep_clip_audio: bool = False,
               crf: int | None = None, preset: str | None = None,
               profile: str | None = None,
               **_ignored: Any) -> EditSpec:
    """Compose the documentary look → engine EditSpec."""
    from ...edit_engine.timeline import Transition
    clip_list = _slow_drift(_coerce_clips(clips, scenes))

    layers: list[TextLayer] = []
    if title:
        layers.append(TextLayer(text=title, position="bottom-left",
                                box=True, size=64, start=0.5, end=5.0))
    if captions:
        if isinstance(captions, str) and captions.strip():
            words = estimate_word_timings(captions, wpm=caption_wpm)
        elif isinstance(captions, TextLayer):
            layers.append(captions)
            words = []
        else:
            words = as_words(list(captions))
        if words:
            layers.append(TextLayer(
                words=[{"start": w.start, "end": w.end, "text": w.text}
                       for w in words],
                style=caption_style, position="bottom"))

    mix: AudioMix | None = None
    alayers: list[AudioLayer] = []
    if music:
        alayers.append(AudioLayer(path=str(music), loop=True, volume=0.45,
                                  fade_in=2.0, fade_out=2.0))
    if voiceover:
        alayers.append(AudioLayer(path=str(voiceover), volume=1.0))
    if alayers:
        mix = AudioMix(layers=alayers, ducking="none", target_lufs=-16.0)

    tr = (transition if isinstance(transition, Transition)
          else Transition(kind=transition, duration=transition_duration))
    return EditSpec(
        clips=clip_list, transition=tr, text_layers=layers, audio=mix,
        fps=fps, width=width, height=height, seed=seed, crf=crf,
        preset=preset, profile=profile,
        out=str(out) if out else None, keep_clip_audio=keep_clip_audio)

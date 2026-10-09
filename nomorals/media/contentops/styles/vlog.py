"""Vlog style preset — punchy, fast, personable.

Jump cuts, pop zooms on emphasis, quick flash cuts, bold centered
captions. Composes engine primitives only.
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


def _pop_zooms(clips: list[Clip], *, zmax: float = 1.25) -> list[Clip]:
    """Clips without effects get punch-zoom pop-ins (alternating depth)."""
    out: list[Clip] = []
    for i, c in enumerate(clips):
        if c.effects:
            out.append(c)
            continue
        z = zmax if i % 2 == 0 else 1.12
        out.append(Clip(path=c.path, start=c.start, end=c.end,
                        speed=c.speed, gain=c.gain,
                        effects=(Effect(name="punch_zoom",
                                        params={"zmax": z}),)))
    return out


def build_spec(*, clips: Any = (), scenes: Any = None,
               beats: Any = None, beat_audio: Any = None,
               voiceover: Any = None, music: Any = None,
               captions: Any = None,
               transition: Any = "cut",
               caption_style: str = "hormozi",
               caption_wpm: float = 160.0,
               seed: int = 0,
               width: int = 1080, height: int = 1920, fps: int = 30,
               out: Any = None, keep_clip_audio: bool = False,
               crf: int | None = None, preset: str | None = None,
               profile: str | None = None,
               **_ignored: Any) -> EditSpec:
    """Compose the vlog look → engine EditSpec."""
    clip_list = _pop_zooms(_coerce_clips(clips, scenes))

    layers: list[TextLayer] = []
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
                style=caption_style, position="center", size=84))

    mix: AudioMix | None = None
    alayers: list[AudioLayer] = []
    if music:
        alayers.append(AudioLayer(path=str(music), loop=True, volume=0.5))
    if voiceover:
        alayers.append(AudioLayer(path=str(voiceover), volume=1.0))
    if alayers:
        mix = AudioMix(layers=alayers,
                       ducking="sidechain" if len(alayers) > 1 else "none",
                       duck_key=len(alayers) - 1,
                       target_lufs=-14.0)

    return EditSpec(
        clips=clip_list, transition=transition, text_layers=layers,
        audio=mix, fps=fps, width=width, height=height, seed=seed,
        crf=crf, preset=preset, profile=profile,
        out=str(out) if out else None, keep_clip_audio=keep_clip_audio)

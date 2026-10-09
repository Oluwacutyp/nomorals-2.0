"""Phonk style preset — the beat-synced short-form edit grammar.

Composes engine primitives only: cut grid from beats, per-segment
motion effects, blink transitions, beat-snapped karaoke captions, and
a music bed ducked under the voiceover.

Content kwargs for :func:`build_spec`:

* ``clips`` / ``scenes`` — engine Clips or loose scene mappings
  ({image, duration, effect, zoom_direction})
* ``beats`` — explicit cut grid; ``beat_audio`` — detect from this file
* ``voiceover`` / ``music`` — audio paths
* ``captions`` — text, word list, TextLayer, or list thereof
* ``transition`` — boundary style (default ``"cut"``)
* ``seed``, ``width``, ``height``, ``fps``, ``out``, ``keep_clip_audio``,
  ``crf``, ``preset``, ``profile``
* ``caption_style`` — karaoke/hormozi/minimal/mrbeast (default karaoke)
* ``caption_sync`` — auto|beats|voiceover (default auto)
* ``effect_cycle`` — per-segment effects when clips carry none
  (default shake → punch_zoom → beat_pulse)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ...edit_engine import (
    AudioLayer,
    AudioMix,
    Clip,
    EditSpec,
    Effect,
    TextLayer,
)
from ...edit_engine.text import as_words, words_to_ass
from ....media_edit import captions as _captions
from ....media_edit.videos import MediaEditError
from ..beats import detect_beats_full
from ....core.logging_setup import get_logger

_log = get_logger(__name__)

DEFAULT_EFFECT_CYCLE = ("shake", "punch_zoom", "beat_pulse")


def resolve_beats(beats: list[float] | None,
                  beat_audio: str | Path | None) -> list[float] | None:
    """Explicit grid wins; else detect from ``beat_audio``; else None."""
    if beats:
        return [float(b) for b in beats]
    if beat_audio:
        info = detect_beats_full(str(beat_audio))
        _log.info("phonk: %d beats (%.1f bpm, %s engine)",
                  len(info.beats), info.bpm, info.backend)
        if not info.beats:
            raise MediaEditError(
                f"no beats detected in {beat_audio} — supply beats "
                "explicitly or pick a track with a clear beat")
        return list(info.beats)
    return None


def beat_sync_words(words: list[Any], beats: list[float] | None, *,
                    mode: str = "both") -> list[_captions.Word]:
    """Snap word timings to the nearest beat (the edit-grid caption look).

    ``mode``: "both" (snap start+end), "start" (starts only). Monotonicity
    enforced: no zero-length or overlapping words. ``beats`` empty →
    words unchanged.
    """
    ws = as_words(words)
    if mode not in ("both", "start"):
        raise MediaEditError(f"unknown sync mode {mode!r}; use both/start")
    if not beats:
        return ws
    grid = sorted(float(b) for b in beats)

    def snap(t: float) -> float:
        return min(grid, key=lambda g: abs(g - t))

    out: list[_captions.Word] = []
    for w in ws:
        s = snap(w.start)
        e = snap(w.end) if mode == "both" else w.end
        e = max(e, s + 0.08)
        if out:
            s = max(s, out[-1].end + 0.01)
            e = max(e, s + 0.08)
        out.append(_captions.Word(start=round(s, 3), end=round(e, 3),
                                  text=w.text))
    return out


def words_to_beat_ass(words: list[Any], beats: list[float] | None = None, *,
                      style: str = "karaoke",
                      width: int = 1080, height: int = 1920,
                      sync: bool = True) -> str:
    """Word timings → word-highlight .ass, beat-grid or voiceover timed."""
    ws = as_words(words)
    if sync and beats:
        ws = beat_sync_words(ws, beats)
    ass = words_to_ass(ws, style=style, width=width, height=height)
    return ass


def burn_beat_captions(video: str | Path,
                       words: list[Any],
                       beats: list[float] | None = None, *,
                       style: str = "karaoke",
                       out_dir: str | Path | None = None,
                       suffix: str = "captioned",
                       sync: bool = True) -> dict[str, Any]:
    """Burn beat/voiceover-timed word-highlight captions into ``video``.

    Keeps the decoupled-transcript pattern: ``{video}.words.json`` +
    ``{video}.srt`` sidecars next to the video.
    """
    p = Path(video)
    if not p.exists():
        raise MediaEditError(f"no such video: {video}")
    ws = as_words(words)
    if not ws:
        raise MediaEditError("no words — nothing to caption")
    ass_text = words_to_beat_ass(ws, beats, style=style, sync=sync)
    ass_path = p.with_name(f"{p.stem}.{style}.ass")
    ass_path.write_text(ass_text, encoding="utf-8")
    _captions.save_transcript(p, ws)
    srt_path = p.with_suffix(".srt")
    srt_path.write_text(_captions.words_to_srt(ws), encoding="utf-8")
    run = _captions.burn_captions(p, ass_path, out_dir=out_dir, suffix=suffix)
    run["words"] = len(ws)
    run["style"] = style
    run["synced_to_beats"] = bool(sync and beats)
    return run


def _coerce_clips(clips: Any, scenes: Any) -> list[Clip]:
    if scenes is not None and not clips:
        from ...edit_engine.spec import _scene_to_clip
        return [_scene_to_clip(s) for s in scenes]
    return [c if isinstance(c, Clip) else Clip.from_dict(dict(c))
            for c in (clips or [])]


def _apply_effect_cycle(clips: list[Clip],
                        cycle: tuple[str, ...]) -> list[Clip]:
    """Clips without effects get the phonk motion cycle."""
    out: list[Clip] = []
    k = 0
    for c in clips:
        if c.effects:
            out.append(c)
            continue
        name = cycle[k % len(cycle)]
        k += 1
        out.append(Clip(path=c.path, start=c.start, end=c.end,
                        speed=c.speed, gain=c.gain,
                        effects=(Effect(name=name),)))
    return out


def _captions_to_layers(captions: Any, beats: list[float] | None, *,
                        style: str, sync: str, wpm: float,
                        width: int, height: int) -> list[TextLayer]:
    """captions (text | words | TextLayer | list) → TextLayers."""
    if captions is None:
        return []
    if isinstance(captions, TextLayer):
        return [captions]
    if isinstance(captions, (list, tuple)) and captions and all(
            isinstance(c, TextLayer) for c in captions):
        return list(captions)
    # word-list or plain text → one karaoke layer
    if isinstance(captions, str):
        if not captions.strip():
            return []
        from ...edit_engine.text import estimate_word_timings
        words = estimate_word_timings(captions, wpm=wpm)
        estimated = True
    else:
        words = as_words(list(captions))
        estimated = False
    snap = (sync == "beats" or (sync == "auto" and estimated and beats))
    if sync not in ("auto", "beats", "voiceover"):
        raise MediaEditError(
            f"unknown caption sync {sync!r}; use auto/beats/voiceover")
    if snap and beats:
        words = beat_sync_words(words, beats)
    return [TextLayer(words=[{"start": w.start, "end": w.end, "text": w.text}
                             for w in words],
                      style=style)]


def _audio_to_mix(voiceover: Any, music: Any, *,
                  ducking: str = "sidechain",
                  duck_level: float = 0.30,
                  target_lufs: float = -14.0,
                  loop_music: bool = True) -> AudioMix | None:
    """voiceover/music paths → AudioMix (music ducked under voice)."""
    layers: list[AudioLayer] = []
    vo_idx = 0
    if music:
        layers.append(AudioLayer(path=str(music), loop=loop_music,
                                 volume=1.0))
        vo_idx = 1
    if voiceover:
        layers.append(AudioLayer(path=str(voiceover), volume=1.0))
    if not layers:
        return None
    return AudioMix(
        layers=layers,
        ducking=ducking if len(layers) > 1 else "none",
        duck_key=vo_idx if len(layers) > 1 else 0,
        duck_level=duck_level, target_lufs=target_lufs)


def build_spec(*, clips: Any = (), scenes: Any = None,
               beats: list[float] | None = None,
               beat_audio: Any = None,
               voiceover: Any = None, music: Any = None,
               captions: Any = None,
               transition: Any = "cut",
               caption_style: str = "karaoke",
               caption_sync: str = "auto",
               caption_wpm: float = 150.0,
               effect_cycle: tuple[str, ...] = DEFAULT_EFFECT_CYCLE,
               ducking: str = "sidechain",
               duck_level: float = 0.30,
               target_lufs: float = -14.0,
               loop_music: bool = True,
               seed: int = 0,
               width: int = 1080, height: int = 1920, fps: int = 30,
               out: Any = None, keep_clip_audio: bool = False,
               crf: int | None = None, preset: str | None = None,
               profile: str | None = None,
               **_ignored: Any) -> EditSpec:
    """Compose the phonk look → engine EditSpec."""
    beat_src = beat_audio
    grid = resolve_beats(
        list(beats) if beats else None,
        str(beat_src) if beat_src else None)
    clip_list = _apply_effect_cycle(_coerce_clips(clips, scenes),
                                    tuple(effect_cycle))
    layers = _captions_to_layers(captions, grid, style=caption_style,
                                 sync=caption_sync, wpm=caption_wpm,
                                 width=width, height=height)
    mix = _audio_to_mix(voiceover, music, ducking=ducking,
                        duck_level=duck_level, target_lufs=target_lufs,
                        loop_music=loop_music)
    return EditSpec(
        clips=clip_list, cut_points=grid, transition=transition,
        text_layers=layers, audio=mix, fps=fps, width=width,
        height=height, seed=seed, crf=crf, preset=preset,
        profile=profile, out=str(out) if out else None,
        keep_clip_audio=keep_clip_audio)

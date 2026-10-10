"""Caption pipeline: faster-whisper (word timestamps) → .ass → FFmpeg burn-in.

The #1 video feature by usage frequency. Complete path::

    words = transcribe_words("clip.mp4")          # faster-whisper, word timings
    out = burn_captions("clip.mp4", words, style="hormozi")

Transcription and rendering are DECOUPLED: the word transcript is stored
as ``{video}.words.json`` next to the video, so ``recaption()`` can fix
typos and re-render without re-running the model::

    words = load_transcript("clip.mp4")
    words[3] = Word(words[3].start, words[3].end, "Devon")  # fix typo
    out = recaption("clip.mp4", words)

An .srt sidecar is exported alongside every burn (the Gling pattern —
editors can re-cut from it).

Style presets are ported from Capite's 27-preset concept (only the four
highest-usage shapes are shipped): ``hormozi``, ``mrbeast``, ``karaoke``,
``minimal``.

Dependencies are optional and honest: no faster-whisper → pip-hint error
(no fake words); no libass in ffmpeg → burn refuses with the real reason
(but the .ass/.srt sidecars are still written).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .videos import (
    MediaEditError,
    burn_subtitles,
    has_libass,
    run_ffmpeg,
    video_probe,
)

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# words
# ---------------------------------------------------------------------------

@dataclass
class Word:
    """One transcribed word with timing (seconds)."""
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "text": self.text}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Word":
        return cls(start=float(data["start"]), end=float(data["end"]),
                   text=str(data["text"]))


def transcribe_words(video_path: str | os.PathLike[str], *,
                     language: str = "en", model: str = "",
                     progress_cb: Any = None) -> list[Word]:
    """Transcribe ``video_path`` → word-timed words via faster-whisper.

    Audio is extracted to 16 kHz mono wav first. Requires
    ``pip install faster-whisper`` — missing → clear error, never fake words.
    """
    p = Path(video_path)
    if not p.exists():
        raise MediaEditError(f"no such video: {video_path}")
    try:
        from ..voice.stt import FasterWhisperBackend
    except ImportError as exc:
        raise MediaEditError(
            "captioning needs faster-whisper: pip install faster-whisper"
        ) from exc
    backend = FasterWhisperBackend(model=model)  # raises with pip hint
    wav = _extract_wav(p)
    try:
        segments, _info = backend.model.transcribe(
            str(wav), language=language or None,
            vad_filter=True, condition_on_previous_text=False,
            word_timestamps=True)
        words: list[Word] = []
        for seg in segments:
            for w in getattr(seg, "words", None) or []:
                text = (w.word or "").strip()
                if not text:
                    continue
                words.append(Word(start=float(w.start), end=float(w.end),
                                  text=text))
        return words
    finally:
        try:
            wav.unlink(missing_ok=True)
        except OSError:
            pass


def _extract_wav(video: Path) -> Path:
    """Extract 16 kHz mono wav for STT (temp file, caller deletes)."""
    out = Path(tempfile.mkstemp(prefix="caption-", suffix=".wav")[1])
    info = video_probe(video)
    if not any(s.get("codec_type") == "audio"
               for s in info.get("streams", [])):
        raise MediaEditError(f"no audio track in {video} — nothing to caption")
    run_ffmpeg(["-i", str(video), "-vn", "-ar", "16000", "-ac", "1",
                "-c:a", "pcm_s16le", str(out)],
               duration=info.get("duration"))
    return out


# ---------------------------------------------------------------------------
# .ass generation
# ---------------------------------------------------------------------------

# ASS colours are &HAABBGGRR.
_STYLES: dict[str, dict[str, Any]] = {
    # Big bold white + black stroke, bottom-centre. The Hormozi look.
    "hormozi": {
        "Fontname": "Arial Black", "Fontsize": 68,
        "PrimaryColour": "&H00FFFFFF", "SecondaryColour": "&H000000FF",
        "OutlineColour": "&H00000000", "BackColour": "&H80000000",
        "Bold": -1, "Italic": 0, "BorderStyle": 1, "Outline": 4,
        "Shadow": 2, "Alignment": 2, "MarginL": 20, "MarginR": 20,
        "MarginV": 60,
    },
    # Yellow fill, thick black outline — the MrBeast thumbnail-caption look.
    "mrbeast": {
        "Fontname": "Arial Black", "Fontsize": 72,
        "PrimaryColour": "&H0000FFFF", "SecondaryColour": "&H000000FF",
        "OutlineColour": "&H00000000", "BackColour": "&H80000000",
        "Bold": -1, "Italic": 0, "BorderStyle": 1, "Outline": 5,
        "Shadow": 2, "Alignment": 2, "MarginL": 20, "MarginR": 20,
        "MarginV": 60,
    },
    # Word-by-word fill sweep ({\kf} tags). Primary = filled colour,
    # Secondary = unfilled colour — the fill sweeps secondary → primary.
    "karaoke": {
        "Fontname": "Arial", "Fontsize": 64,
        "PrimaryColour": "&H0000FFFF", "SecondaryColour": "&H00FFFFFF",
        "OutlineColour": "&H00000000", "BackColour": "&H80000000",
        "Bold": -1, "Italic": 0, "BorderStyle": 1, "Outline": 3,
        "Shadow": 1, "Alignment": 2, "MarginL": 20, "MarginR": 20,
        "MarginV": 60,
    },
    # Small, clean, no shadow — readable without shouting.
    "minimal": {
        "Fontname": "Arial", "Fontsize": 44,
        "PrimaryColour": "&H00FFFFFF", "SecondaryColour": "&H000000FF",
        "OutlineColour": "&H80000000", "BackColour": "&H80000000",
        "Bold": 0, "Italic": 0, "BorderStyle": 1, "Outline": 2,
        "Shadow": 0, "Alignment": 2, "MarginL": 20, "MarginR": 20,
        "MarginV": 40,
    },
    # Cyan-on-black neon glow — the cyberpunk/tech-creator look.
    "neon": {
        "Fontname": "Arial Black", "Fontsize": 66,
        "PrimaryColour": "&H00FFFF00", "SecondaryColour": "&H000000FF",
        "OutlineColour": "&H00000000", "BackColour": "&HC0000000",
        "Bold": -1, "Italic": 0, "BorderStyle": 1, "Outline": 3,
        "Shadow": 4, "Alignment": 2, "MarginL": 20, "MarginR": 20,
        "MarginV": 60,
    },
    # Clean top-centre podcast look — stays out of the speaker's face.
    "podcast": {
        "Fontname": "Arial", "Fontsize": 52,
        "PrimaryColour": "&H00FFFFFF", "SecondaryColour": "&H000000FF",
        "OutlineColour": "&H80000000", "BackColour": "&H99000000",
        "Bold": -1, "Italic": 0, "BorderStyle": 3, "Outline": 0,
        "Shadow": 0, "Alignment": 8, "MarginL": 30, "MarginR": 30,
        "MarginV": 40,
    },
    # MrBeast yellow + bounce: each caption pops in with a scale
    # transform (needs animate="pop"; pairs with any style).
    "beast-bounce": {
        "Fontname": "Arial Black", "Fontsize": 74,
        "PrimaryColour": "&H0000FFFF", "SecondaryColour": "&H000000FF",
        "OutlineColour": "&H00000000", "BackColour": "&H80000000",
        "Bold": -1, "Italic": 0, "BorderStyle": 1, "Outline": 5,
        "Shadow": 3, "Alignment": 2, "MarginL": 20, "MarginR": 20,
        "MarginV": 90,
    },
}

_STYLE_FIELDS = (
    "Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
    "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,"
    "BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding"
)

_MAX_WORDS_PER_EVENT = 4
_MAX_EVENT_SECS = 2.0
_SENTENCE_END = re.compile(r"[.!?…]+$")


def _ass_ts(secs: float) -> str:
    """ASS timestamp H:MM:SS.cc (centiseconds)."""
    if secs < 0:
        secs = 0.0
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    cs = int(round((secs - int(secs)) * 100))
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _group_events(words: list[Word]) -> list[list[Word]]:
    """Word-level timings → caption events: ≤4 words, ≤2 s, break on
    sentence punctuation. The duration cap is pre-checked so an event
    never exceeds it (unless a single word does)."""
    events: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        if cur and (len(cur) >= _MAX_WORDS_PER_EVENT
                    or (w.end - cur[0].start) >= _MAX_EVENT_SECS):
            events.append(cur)
            cur = []
        cur.append(w)
        if _SENTENCE_END.search(w.text):
            events.append(cur)
            cur = []
    if cur:
        events.append(cur)
    return events


def _event_text(event: list[Word], style: str) -> str:
    """Render one event's dialogue text. Karaoke gets per-word {\kf} fill
    tags (duration in centiseconds); other styles get plain spaced words."""
    if style != "karaoke":
        return " ".join(w.text for w in event)
    parts = []
    for w in event:
        dur_cs = max(1, int(round((w.end - w.start) * 100)))
        parts.append(f"{{\\kf{dur_cs}}}{w.text}")
    return " ".join(parts)


def words_to_ass(words: list[Word], *, style: str = "hormozi",
                 title: str = "captioned",
                 animate: str = "none") -> str:
    """Word list → .ass subtitle script content.

    ``animate="pop"`` adds a ``{\\t}`` scale-in transform to every event
    (the TikTok/Submagic bounce); ``"none"`` is the static default.
    """
    if style not in _STYLES:
        raise MediaEditError(
            f"unknown caption style {style!r}; use: {sorted(_STYLES)}")
    animate = (animate or "none").lower()
    if animate not in ("none", "pop"):
        raise MediaEditError(
            f"unknown caption animation {animate!r}; use none|pop")
    if not words:
        raise MediaEditError("no words to render — nothing to caption")
    st = _STYLES[style]
    style_line = (
        f"Style: Caption,{st['Fontname']},{st['Fontsize']},"
        f"{st['PrimaryColour']},{st['SecondaryColour']},"
        f"{st['OutlineColour']},{st['BackColour']},"
        f"{st['Bold']},{st['Italic']},0,0,100,100,0,0,"
        f"{st['BorderStyle']},{st['Outline']},{st['Shadow']},"
        f"{st['Alignment']},{st['MarginL']},{st['MarginR']},{st['MarginV']},1"
    )
    lines = [
        "[Script Info]",
        f"Title: {title}",
        "ScriptType: v4.00+",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        "PlayResX: 1280",
        "PlayResY: 720",
        "",
        "[V4+ Styles]",
        f"Format: {_STYLE_FIELDS}",
        style_line,
        "",
        "[Events]",
        ("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
         "Effect, Text"),
    ]
    events = _group_events(words)
    for i, event in enumerate(events):
        start = _ass_ts(event[0].start)
        # Never let an event bleed into the next one's start.
        end_t = event[-1].end
        if i + 1 < len(events):
            end_t = min(end_t, events[i + 1][0].start - 0.05)
        end = _ass_ts(max(end_t, event[0].start + 0.2))
        text = _event_text(event, style)
        if animate == "pop":
            # scale-in bounce: 60% → 112% → settle at 100% over 240 ms
            text = ("{\\t(0,120,\\fscx60\\fscy60)"
                    "\\t(120,240,\\fscx112\\fscy112)}" + text)
        lines.append(
            f"Dialogue: 0,{start},{end},Caption,,0,0,0,,{text}")
    return "\n".join(lines) + "\n"


def _srt_ts(secs: float) -> str:
    if secs < 0:
        secs = 0.0
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    ms = int(round((secs - int(secs)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def words_to_srt(words: list[Word]) -> str:
    """Word list → .srt sidecar (plain text, no karaoke tags)."""
    if not words:
        raise MediaEditError("no words to render — nothing to caption")
    out = []
    for i, event in enumerate(_group_events(words), start=1):
        start = _srt_ts(event[0].start)
        end = _srt_ts(max(event[-1].end, event[0].start + 0.2))
        text = " ".join(w.text for w in event)
        out.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# burn + decoupled transcripts
# ---------------------------------------------------------------------------

def transcript_path(video_path: str | os.PathLike[str]) -> Path:
    """``clip.mp4`` → ``clip.words.json`` next to the video."""
    p = Path(video_path)
    return p.with_name(p.stem + ".words.json")


def save_transcript(video_path: str | os.PathLike[str],
                    words: list[Word]) -> Path:
    """Store the word transcript next to the video (decouples STT from
    rendering — fix typos without re-running the model)."""
    tp = transcript_path(video_path)
    tp.write_text(json.dumps([w.to_dict() for w in words],
                             ensure_ascii=False, indent=1),
                  encoding="utf-8")
    return tp


def load_transcript(video_path: str | os.PathLike[str]) -> list[Word] | None:
    """Load a stored transcript, or None when there isn't one."""
    tp = transcript_path(video_path)
    if not tp.exists():
        return None
    try:
        data = json.loads(tp.read_text(encoding="utf-8"))
        return [Word.from_dict(d) for d in data]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        _log.warning("caption transcript %s unreadable: %s", tp, exc)
        return None


def burn_captions(video: str | os.PathLike[str],
                  words_or_ass: list[Word] | str | os.PathLike[str], *,
                  style: str = "hormozi",
                  out_dir: str | os.PathLike[str] | None = None,
                  suffix: str = "captioned",
                  write_sidecars: bool = True) -> dict[str, Any]:
    """Burn captions into ``video``. ``words_or_ass`` is a Word list or a
    path to an existing .ass file.

    Writes ``{video}.words.json`` (transcript) and ``{video}.srt`` sidecars
    next to the video when given words. Returns the burn_subtitles dict
    ({"input", "output", "bytes", "seconds"}).
    """
    p = Path(video)
    if isinstance(words_or_ass, (str, os.PathLike)):
        ass_path = Path(words_or_ass)
        if not ass_path.exists():
            raise MediaEditError(f"no such subtitle file: {ass_path}")
    else:
        words = list(words_or_ass)
        ass_content = words_to_ass(words, style=style, title=p.stem)
        ass_path = p.with_name(p.stem + f".{style}.ass")
        ass_path.write_text(ass_content, encoding="utf-8")
        if write_sidecars:
            save_transcript(p, words)
            srt_path = p.with_suffix(".srt")
            srt_path.write_text(words_to_srt(words), encoding="utf-8")
            _log.info("caption sidecars: %s, %s",
                      transcript_path(p).name, srt_path.name)
    if not has_libass():
        raise MediaEditError(
            "this ffmpeg build has no libass — cannot burn subtitles. "
            f"The .ass sidecar is at {ass_path}; install ffmpeg with libass "
            "to burn it in.")
    run = burn_subtitles(p, ass_path, out_dir=out_dir, suffix=suffix)
    return {"input": run["input"], "output": run["output"],
            "bytes": run["bytes"], "seconds": run["seconds"],
            "ass": str(ass_path)}


def caption_video(video: str | os.PathLike[str], *,
                  language: str = "en", style: str = "hormozi",
                  out_dir: str | os.PathLike[str] | None = None,
                  model: str = "") -> dict[str, Any]:
    """Full pipeline: transcribe → burn. One call, honest errors."""
    words = transcribe_words(video, language=language, model=model)
    if not words:
        raise MediaEditError(
            f"transcription produced no words for {video} — nothing to burn")
    return burn_captions(video, words, style=style, out_dir=out_dir)


def recaption(video: str | os.PathLike[str],
              edited: list[Word] | list[dict[str, Any]] | str | os.PathLike[str],
              *, style: str = "hormozi",
              out_dir: str | os.PathLike[str] | None = None,
              suffix: str = "recaptioned") -> dict[str, Any]:
    """Re-render captions from an EDITED transcript — no STT re-run.

    ``edited``: a Word list, a list of word dicts, or a path to a
    ``.words.json`` file. Fix typos, then burn.
    """
    if isinstance(edited, (str, os.PathLike)):
        ep = Path(edited)
        data = json.loads(ep.read_text(encoding="utf-8"))
        words = [Word.from_dict(d) for d in data]
    else:
        words = [w if isinstance(w, Word) else Word.from_dict(w)
                 for w in edited]
    if not words:
        raise MediaEditError("edited transcript is empty — nothing to burn")
    return burn_captions(video, words, style=style, out_dir=out_dir,
                         suffix=suffix)


def caption_styles() -> list[str]:
    """Available .ass style presets."""
    return sorted(_STYLES)

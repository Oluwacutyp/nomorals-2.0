"""Kinetic typography / lyric-video engine.

Word-timed text animation rendered frame-by-frame with PIL (no libass
dependency, so it works everywhere including termux). Timing comes from
real data — a timing track, TTS alignment, or beat-snapped estimates via
the edit engine's :func:`estimate_word_timings` — never invented.

Style presets are data (``STYLE_PRESETS``): colors, animation, background.
New looks = new dict entries, not new code.

    from nomorals.media.motion_studio.typography import render_lyrics
    render_lyrics("line one\\nline two", "lyric.mp4", audio="song.mp3",
                  preset="neon_pop")
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ._core import (
    MotionStudioError,
    ease,
    find_font,
    new_render_path,
    profile_defaults,
    probe_duration,
    record_ledger,
    render_sequence,
    text_size,
)
from ..contentops.beats import detect_beats
from ..edit_engine.text import estimate_word_timings  # edit_engine public API
from ...core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "STYLE_PRESETS", "list_presets", "Word", "to_words",
    "render_lyrics", "render_quote_card",
]


@dataclass
class Word:
    text: str
    start: float
    end: float


#: preset name → look. ``animation``: fade | slide_up | pop | karaoke.
#: ``bg``: gradient | blobs | cover | solid.
STYLE_PRESETS: dict[str, dict[str, object]] = {
    "neon_pop": {
        "label": "neon pop — big centered words, beat-punched",
        "text": "#FFFFFF", "highlight": "#00F0FF", "dim": "#5a5a72",
        "bg": "blobs", "bg_colors": ["#07070f", "#141428"],
        "animation": "pop", "size_rel": 0.105, "stroke": 2,
        "line_words": 5,
    },
    "karaoke": {
        "label": "classic karaoke — line highlight sweep",
        "text": "#F5F5F5", "highlight": "#FFE14D", "dim": "#6b6b80",
        "bg": "gradient", "bg_colors": ["#0b1020", "#1a2340"],
        "animation": "karaoke", "size_rel": 0.085, "stroke": 0,
        "line_words": 7,
    },
    "minimal": {
        "label": "quiet minimal — fade in/out, small caps feel",
        "text": "#EDEDED", "highlight": "#FFFFFF", "dim": "#555566",
        "bg": "solid", "bg_colors": ["#000000", "#000000"],
        "animation": "fade", "size_rel": 0.075, "stroke": 0,
        "line_words": 8,
    },
    "phonk_type": {
        "label": "phonk — hard slide-ups, high contrast",
        "text": "#FFFFFF", "highlight": "#FF3860", "dim": "#3d3d4d",
        "bg": "cover", "bg_colors": ["#0a0a0f", "#0a0a0f"],
        "animation": "slide_up", "size_rel": 0.115, "stroke": 3,
        "line_words": 4,
    },
    "lofi_type": {
        "label": "lofi — soft fade, warm paper tones",
        "text": "#F2E8D5", "highlight": "#E8B04B", "dim": "#7a6f5f",
        "bg": "gradient", "bg_colors": ["#141824", "#232a3d"],
        "animation": "fade", "size_rel": 0.08, "stroke": 0,
        "line_words": 7,
    },
    "bold_statement": {
        "label": "quote cards — huge words, one phrase per beat",
        "text": "#FFFFFF", "highlight": "#B537F2", "dim": "#444455",
        "bg": "blobs", "bg_colors": ["#080810", "#101018"],
        "animation": "pop", "size_rel": 0.13, "stroke": 2,
        "line_words": 4,
    },
}


def list_presets() -> list[dict[str, object]]:
    return [{"name": n, "label": v["label"]} for n, v in STYLE_PRESETS.items()]


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------

def to_words(lyrics: str | Sequence, *,
             audio: str | os.PathLike | None = None,
             duration: float | None = None,
             wpm: float = 150.0) -> list[Word]:
    """Normalize lyric input to a timed word list.

    Accepts: ``[(text, start, end), …]``, ``[(text, start), …]``,
    ``[{text, start, end}, …]``, or plain text (timings estimated, snapped
    to the audio's beat grid when ``audio`` is given).
    """
    if isinstance(lyrics, str):
        tokens = [t for t in lyrics.replace("\n", " ").split() if t]
        if not tokens:
            raise MotionStudioError("no lyric text — nothing to render")
        beats = None
        if audio:
            try:
                beats = detect_beats(audio)
            except Exception as exc:  # noqa: BLE001 - estimates still work
                _log.warning("beat snap failed: %s", exc)
        est = estimate_word_timings(" ".join(tokens), wpm=wpm, beats=beats)
        return [Word(w.text, w.start, w.end) for w in est]

    words: list[Word] = []
    for item in lyrics:
        if isinstance(item, Word):
            words.append(item)
        elif isinstance(item, dict):
            words.append(Word(str(item["text"]), float(item["start"]),
                              float(item.get("end", item["start"] + 0.4))))
        else:
            text, start = item[0], float(item[1])
            end = float(item[2]) if len(item) > 2 else start + 0.4
            words.append(Word(str(text), start, end))
    if not words:
        raise MotionStudioError("empty word list — nothing to render")
    return sorted(words, key=lambda w: w.start)


def _phrases(words: list[Word], per_line: int) -> list[list[Word]]:
    """Group words into display phrases, splitting on punctuation too."""
    phrases, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= per_line or w.text.rstrip().endswith((",", ".", "!", "?", "…", "—")):
            phrases.append(cur)
            cur = []
    if cur:
        phrases.append(cur)
    return phrases


#: Safe-area insets per format: (top, bottom) as fraction of height.
#: Keeps type clear of platform UI (progress bars, captions, notches).
SAFE_AREAS: dict[str, tuple[float, float]] = {
    "9:16": (0.10, 0.16),   # reels/tiktok: caption + progress bar zones
    "16:9": (0.06, 0.10),   # youtube: title + progress bar
    "1:1": (0.08, 0.12),    # feed square: UI chrome top/bottom
}


def safe_area(format: str) -> tuple[float, float]:
    """(top, bottom) safe insets for a format. Never raises."""
    return SAFE_AREAS.get(str(format), (0.08, 0.12))


def emphasis_words(words: list[Word]) -> set[int]:
    """Auto-detect emphasis words (the hook words that deserve the pop).

    Heuristic, deterministic: ALL-CAPS words, words ending in ``!``,
    and words repeated 3+ times across the lyric (the hook) — the same
    words a human editor would punch. Returns word indices.
    """
    out: set[int] = set()
    counts: dict[str, int] = {}
    for w in words:
        key = w.text.strip(" ,.!?…—\"'").lower()
        if key:
            counts[key] = counts.get(key, 0) + 1
    for i, w in enumerate(words):
        t = w.text.strip()
        if not t:
            continue
        core = t.strip(" ,.!?…—\"'")
        if core.isupper() and len(core) > 1:
            out.add(i)
        elif t.endswith("!"):
            out.add(i)
        elif counts.get(core.lower(), 0) >= 3:
            out.add(i)
    return out


# ---------------------------------------------------------------------------
# backgrounds
# ---------------------------------------------------------------------------

def _hex(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


class _Background:
    def __init__(self, w: int, h: int, kind: str,
                 colors: list[str], image: str | os.PathLike | None,
                 seed: int):
        self.w, self.h, self.kind = w, h, kind
        self.c0 = np.array(_hex(colors[0]), dtype=np.float32)
        self.c1 = np.array(_hex(colors[-1]), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        n = 5
        self.bx = self.rng.random(n) * w
        self.by = self.rng.random(n) * h
        self.br = (0.25 + self.rng.random(n) * 0.35) * min(w, h)
        ang = self.rng.random(n) * 2 * math.pi
        self.bvx = np.cos(ang) * 12
        self.bvy = np.sin(ang) * 9
        self.cover = None
        if image:
            try:
                im = Image.open(image).convert("RGB")
                s = max(w / im.width, h / im.height)
                self.cover = im.resize((int(im.width * s) + 1,
                                        int(im.height * s) + 1), Image.LANCZOS)
            except Exception:  # noqa: BLE001
                self.cover = None

    def frame(self, t: float) -> Image.Image:
        if self.kind == "solid":
            return Image.new("RGB", (self.w, self.h),
                             tuple(int(v) for v in self.c0))
        if self.kind == "gradient":
            k = 0.5 + 0.5 * math.sin(t * 0.35)
            ys = np.linspace(0, 1, self.h, dtype=np.float32)[:, None, None]
            arr = self.c0[None, None, :] * (1 - ys) + self.c1[None, None, :] * ys
            arr = arr * (0.92 + 0.08 * k)
            base = Image.fromarray(np.clip(
                np.repeat(arr, self.w, axis=1), 0, 255).astype(np.uint8))
        elif self.kind == "cover" and self.cover is not None:
            zoom = 1.0 + 0.06 * math.sin(t * 0.3)
            cw, ch = int(self.w * zoom), int(self.h * zoom)
            bg = self.cover.resize((cw, ch), Image.LANCZOS).filter(
                ImageFilter.GaussianBlur(22))
            base = bg.crop(((cw - self.w) // 2, (ch - self.h) // 2,
                            (cw + self.w) // 2, (ch + self.h) // 2))
            dark = Image.new("RGB", (self.w, self.h), (0, 0, 0))
            base = Image.blend(base, dark, 0.55)
        else:  # blobs
            arr = np.tile(self.c0[None, None, :], (self.h, self.w, 1))
            # blob positions are a pure function of t → deterministic drift
            bxs = (self.bx + self.bvx * t) % self.w
            bys = (self.by + self.bvy * t) % self.h
            ys, xs = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
            for bx, by, br in zip(bxs, bys, self.br):
                d2 = ((xs - bx) ** 2 + (ys - by) ** 2) / (br ** 2)
                m = np.exp(-d2 * 2.2)[..., None]
                arr = arr * (1 - m * 0.5) + self.c1 * (m * 0.5)
            base = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        return base


# ---------------------------------------------------------------------------
# renderer
# ---------------------------------------------------------------------------

class _LyricRenderer:
    def __init__(self, words: list[Word], preset: dict, w: int, h: int,
                 fps: float, image: str | os.PathLike | None, seed: int):
        self.words = words
        self.preset = preset
        self.w, self.h, self.fps = w, h, fps
        self.font_size = max(18, int(h * float(preset["size_rel"])))
        self.font = find_font(self.font_size)
        self.font_big = find_font(int(self.font_size * 1.06))
        self.bg = _Background(w, h, str(preset["bg"]),
                              [str(c) for c in preset["bg_colors"]],
                              image, seed)
        self.phrases = _phrases(words, int(preset["line_words"]))
        self.anim = str(preset["animation"])
        self.stroke = int(preset["stroke"])
        #: emphasis word indices — the hook words that pop harder
        self.emphasis = emphasis_words(words)
        #: safe-area insets guessed from aspect (vertical/horizontal/square)
        ar = (w / h) if h else 1.0
        fmt = "9:16" if ar < 0.8 else ("16:9" if ar > 1.2 else "1:1")
        self.safe_top, self.safe_bottom = safe_area(fmt)
        # phrase index per word
        self.word_phrase = {}
        for pi, ph in enumerate(self.phrases):
            for wd in ph:
                self.word_phrase[id(wd)] = pi

    def _phrase_at(self, t: float) -> int:
        for pi, ph in enumerate(self.phrases):
            if ph[0].start <= t <= ph[-1].end + 0.6:
                return pi
        # before/after: clamp
        if t < self.phrases[0][0].start:
            return 0
        return len(self.phrases) - 1

    def _draw_word(self, base: Image.Image, d: ImageDraw.ImageDraw,
                   word: Word, x: int, y: int, t: float, state: str) -> int:
        """Draw one word; returns its advance width."""
        preset = self.preset
        dt = t - word.start
        # fonts resolve from the CURRENT self.font_size (slot-scaled by frame())
        if state == "upcoming":
            alpha, scale, dy = 90, 1.0, 0
            col = _hex(str(preset["dim"]))
            font = find_font(self.font_size)
        elif state == "past":
            alpha, scale, dy = 110, 0.94, 0
            col = _hex(str(preset["text"]))
            font = find_font(self.font_size)
        else:  # active
            col = _hex(str(preset["highlight"]))
            font = find_font(int(self.font_size * 1.06))
            k = min(max(dt / 0.28, 0.0), 1.0)
            e = ease("ease_out", k)
            if self.anim == "pop":
                scale = 0.55 + 0.45 * e + 0.12 * math.sin(e * math.pi) * (1 - e)
                alpha = int(60 + 195 * e)
                dy = 0
            elif self.anim == "slide_up":
                scale = 1.0
                alpha = int(40 + 215 * e)
                dy = int(-26 * (1 - e))
            elif self.anim == "karaoke":
                scale = 1.06
                alpha = 255
                dy = 0
            else:  # fade
                scale = 1.0
                alpha = int(30 + 225 * e)
                dy = 0
            # emphasis words (the hook) pop 15% bigger when active
            try:
                wi = self.words.index(word)
            except ValueError:
                wi = -1
            if state == "active" and wi in self.emphasis:
                scale = scale * 1.15 + 0.05
                alpha = min(255, alpha + 30)
        size = max(10, int(self.font_size * scale))
        fnt = find_font(size) if scale != 1.0 or state == "active" else font
        tw, th = text_size(d, word.text, fnt)
        layer = Image.new("RGBA", (tw + 20, th + 20), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        ld.text((10, 10), word.text, font=fnt, fill=col + (alpha,),
                stroke_width=self.stroke,
                stroke_fill=(0, 0, 0, int(alpha * 0.7)))
        base.paste(layer, (x, y + dy - 10), layer)
        # karaoke sweep: highlight portion proportional to word progress
        if self.anim == "karaoke" and state == "active":
            span = max(word.end - word.start, 0.05)
            frac = min(max((t - word.start) / span, 0.0), 1.0)
            if 0 < frac < 1:
                sx = int(x + (tw + 20) * frac)
                d.rectangle([x, y - 10, sx, y + th + 10],
                            fill=_hex(str(preset["highlight"])) + (70,))
        return tw

    def _wrap(self, d: ImageDraw.ImageDraw, ph: list[Word],
              fnt) -> list[list[tuple[Word, int]]]:
        """Greedy word-wrap a phrase into lines that fit the canvas."""
        max_w = int(self.w * 0.92)
        lines: list[list[tuple[Word, int]]] = []
        cur: list[tuple[Word, int]] = []
        cur_w = 0
        for wd in ph:
            tw, _ = text_size(d, wd.text + " ", fnt)
            if cur and cur_w + tw > max_w:
                lines.append(cur)
                cur, cur_w = [], 0
            cur.append((wd, tw))
            cur_w += tw
        if cur:
            lines.append(cur)
        return lines

    def frame(self, i: int, t: float) -> np.ndarray:
        base = self.bg.frame(t).convert("RGBA")
        d = ImageDraw.Draw(base, "RGBA")
        pi = self._phrase_at(t)
        phrases = self.phrases
        # layout: previous (dimmed, small, top) / current (center) / next (faint, bottom)
        # current slot is clamped into the format's safe area (platform UI)
        cur_yrel = min(max(0.47, self.safe_top + 0.12),
                       1.0 - self.safe_bottom - 0.12)
        slots = []
        if pi > 0:
            slots.append((pi - 1, max(0.16, self.safe_top + 0.06), 0.62))
        slots.append((pi, cur_yrel, 1.0))
        if pi + 1 < len(phrases):
            slots.append((pi + 1, min(0.82, 1.0 - self.safe_bottom), 0.55))
        for pidx, yrel, wscale in slots:
            ph = phrases[pidx]
            slot_size = int(self.font_size * wscale)
            fnt = find_font(slot_size)
            lines = self._wrap(d, ph, fnt)
            line_h = int(slot_size * 1.35)
            block_h = line_h * len(lines)
            y = int(self.h * yrel - block_h / 2)
            real_size = self.font_size
            self.font_size = slot_size
            try:
                for line in lines:
                    total = sum(tw for _, tw in line)
                    x = int((self.w - total) / 2)
                    for wd, adv in line:
                        if pidx < pi:
                            state = "past"
                        elif pidx > pi:
                            state = "upcoming"
                        else:
                            state = ("active" if wd.start <= t <= wd.end + 0.15
                                     else ("past" if t > wd.end else "upcoming"))
                        self._draw_word(base, d, wd, x, y, t, state)
                        x += int(adv)
                    y += line_h
            finally:
                self.font_size = real_size
        return np.asarray(base.convert("RGB"))


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def render_lyrics(lyrics: str | Sequence,
                  out: str | os.PathLike | None = None, *,
                  audio: str | os.PathLike | None = None,
                  duration: float | None = None,
                  preset: str = "neon_pop",
                  image: str | os.PathLike | None = None,
                  size: tuple[int, int] | None = None,
                  fps: float | None = None,
                  wpm: float = 150.0,
                  seed: int = 21) -> str:
    """Render a kinetic lyric video. Returns the output path.

    ``lyrics``: plain text, or a timed word list (see :func:`to_words`).
    When ``audio`` is given it is muxed and the render lasts the audio
    (or ``duration``); otherwise ``duration`` is required.
    """
    spec = STYLE_PRESETS.get(preset)
    if spec is None:
        raise MotionStudioError(
            f"unknown lyric preset {preset!r} — pick from: "
            f"{', '.join(sorted(STYLE_PRESETS))}")
    words = to_words(lyrics, audio=audio, wpm=wpm)
    if audio and duration is None:
        duration = probe_duration(audio)
    if not duration or duration <= 0:
        # fall back: last word end + tail
        duration = words[-1].end + 2.0
    # keep words inside the render
    words = [w for w in words if w.start < duration]
    if not words:
        raise MotionStudioError("no words fall inside the render duration")

    defaults = profile_defaults()
    size = size or defaults["size"]
    fps = fps or float(defaults["fps"])
    renderer = _LyricRenderer(words, spec, size[0], size[1], fps, image, seed)
    n = max(1, int(round(duration * fps)))
    out_path = Path(out) if out else new_render_path(f"lyric-{preset}")
    render_sequence(n, size[0], size[1], fps, out_path, renderer.frame,
                    audio=audio, crf=int(defaults["crf"]),
                    preset=str(defaults["preset"]))
    record_ledger({"kind": "lyric_video", "path": str(out_path),
                   "preset": preset, "words": len(words),
                   "duration": duration})
    return str(out_path)


def render_quote_card(text: str, out: str | os.PathLike | None = None, *,
                      duration: float = 5.0, preset: str = "bold_statement",
                      size: tuple[int, int] | None = None,
                      fps: float | None = None, seed: int = 21) -> str:
    """One big statement, word-by-word pop — quote cards / hooks."""
    words = [Word(w, i * 0.45, i * 0.45 + 0.6)
             for i, w in enumerate(text.split())]
    return render_lyrics(words, out, duration=duration, preset=preset,
                         size=size, fps=fps, seed=seed)

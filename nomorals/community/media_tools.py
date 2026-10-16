"""In-group media tools: stickers, voice notes, video descriptors, transcripts.

Everything here is honest about what it does and does not do:

* **Sticker maker** — builds a real SVG sticker from text/emoji
  (deterministic colours, wrapped text, safe escaping). An SVG is a
  real artifact the chat layer can send as a sticker/photo; no
  generative model is claimed.
* **Audio → voice note** — a *descriptor*: validates the audio
  metadata the caller supplies (duration, format, size) and returns a
  voice-note spec the chat layer can attach. It does not transcode —
  it never claims to have re-encoded audio it never touched.
* **Video trim / upscale** — *descriptors* again: trim plans and
  upscale targets computed from supplied metadata. The numbers are
  real arithmetic; no pixels are processed here, and the plan says
  so. The chat layer (or a real ffmpeg step outside this package)
  executes the plan.
* **Voice-note transcription** — accepts a transcript supplied by the
  caller (platform-provided captions, a human typing it out, an
  external STT result passed in). It stores and formats it. When no
  transcript is supplied it says so plainly — it **never** invents
  words the speaker did not say. There is no fake STT here.

Isolation: stdlib + ``nomorals.core`` only. Media bytes are never
read here — callers pass metadata / text, and this module computes.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger

_log = get_logger("nomorals.community.media_tools")

__all__ = [
    "StickerSpec", "make_sticker", "render_sticker_svg", "save_sticker",
    "VoiceNoteSpec", "audio_to_voice_note",
    "TrimPlan", "trim_video", "UpscalePlan", "upscale_descriptor",
    "Transcript", "transcribe_voice_note", "format_transcript",
    "SUPPORTED_AUDIO_FORMATS", "SUPPORTED_VIDEO_FORMATS",
]

SUPPORTED_AUDIO_FORMATS = ("mp3", "ogg", "opus", "wav", "m4a", "aac")
SUPPORTED_VIDEO_FORMATS = ("mp4", "mov", "webm", "mkv")

#: Sticker canvas. Telegram stickers are 512×512; this SVG matches.
STICKER_SIZE = 512

#: Deterministic palette — the colour comes from the text hash, so the
#: same words always make the same sticker.
_STICKER_PALETTE = (
    ("#FFD93D", "#1A1A2E"), ("#6BCB77", "#1A1A2E"), ("#4D96FF", "#FFFFFF"),
    ("#FF6B6B", "#FFFFFF"), ("#9B5DE5", "#FFFFFF"), ("#F15BB5", "#FFFFFF"),
    ("#00BBF9", "#1A1A2E"), ("#00F5D4", "#1A1A2E"),
)

_WORD_RE = re.compile(r"\S+")


# ── stickers ───────────────────────────────────────────────────────────────


@dataclass
class StickerSpec:
    """A sticker: the words, the emoji, the colours, the SVG."""

    id: str = ""
    text: str = ""
    emoji: str = ""
    bg: str = ""
    fg: str = ""
    svg: str = ""
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StickerSpec":
        return cls(
            id=str(data.get("id", "")),
            text=str(data.get("text", "")),
            emoji=str(data.get("emoji", "")),
            bg=str(data.get("bg", "")),
            fg=str(data.get("fg", "")),
            svg=str(data.get("svg", "")),
            created_at=float(data.get("created_at") or 0.0),
        )


def _wrap_words(text: str, max_per_line: int = 12) -> list[str]:
    """Greedy word wrap for the sticker canvas. Pure text shaping."""
    words = _WORD_RE.findall(text)
    lines: list[str] = []
    cur = ""
    for w in words:
        if len(cur) + len(w) + (1 if cur else 0) > max_per_line:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines[:4]  # four lines max — stickers are glanced at, not read


def render_sticker_svg(text: str, emoji: str = "", bg: str = "",
                       fg: str = "") -> str:
    """Render the sticker SVG. Real output: valid SVG, escaped text."""
    text = (text or "").strip()
    emoji = (emoji or "").strip()
    if not bg or not fg:
        digest = hashlib.sha256(text.encode()).digest()
        bg, fg = _STICKER_PALETTE[digest[0] % len(_STICKER_PALETTE)]
    lines = _wrap_words(text) or ["…"]
    font_size = 64 if len(lines) <= 2 else 48
    # vertical centre: emoji row on top when present, text below it
    y_start = 300 if emoji else 256
    if emoji:
        y_start = 330
    text_rows = ""
    for i, line in enumerate(lines):
        y = y_start + i * (font_size + 8) - (len(lines) - 1) * (font_size + 8) // 2
        text_rows += (f'<text x="256" y="{y}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="{font_size}" '
                      f'font-weight="800" fill="{html.escape(fg)}">'
                      f"{html.escape(line)}</text>\n")
    emoji_row = ""
    if emoji:
        emoji_row = (f'<text x="256" y="180" text-anchor="middle" '
                     f'font-family="sans-serif" font-size="120">'
                     f"{html.escape(emoji)}</text>\n")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{STICKER_SIZE}" '
        f'height="{STICKER_SIZE}" viewBox="0 0 {STICKER_SIZE} {STICKER_SIZE}">\n'
        f'<rect width="{STICKER_SIZE}" height="{STICKER_SIZE}" rx="96" fill="{html.escape(bg)}"/>\n'
        f"{emoji_row}{text_rows}</svg>\n")


def make_sticker(text: str, emoji: str = "") -> StickerSpec:
    """Make a sticker from words (+ optional emoji). Never raises."""
    try:
        text = (text or "").strip()[:80] or "…"
        emoji = (emoji or "").strip()[:8]
        digest = hashlib.sha256(text.encode()).digest()
        bg, fg = _STICKER_PALETTE[digest[0] % len(_STICKER_PALETTE)]
        return StickerSpec(
            id="stk_" + new_short_id(), text=text, emoji=emoji,
            bg=bg, fg=fg,
            svg=render_sticker_svg(text, emoji, bg, fg),
            created_at=time.time())
    except Exception as exc:  # noqa: BLE001 — sticker making never raises
        _log.debug("make_sticker failed: %s", exc)
        return StickerSpec(id="stk_" + new_short_id(), text="…",
                           svg=render_sticker_svg("…"))


def save_sticker(spec: StickerSpec, path: Path | str) -> Path | None:
    """Write the sticker SVG to ``path``. Returns the path or None."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(spec.svg, encoding="utf-8")
        return p
    except OSError as exc:
        _log.warning("save_sticker failed: %s", exc)
        return None


# ── audio → voice note ─────────────────────────────────────────────────────


@dataclass
class VoiceNoteSpec:
    """Descriptor for an audio file as a voice note. Metadata only."""

    id: str = ""
    source_format: str = ""
    duration_s: float = 0.0
    size_bytes: int = 0
    voice_note: bool = False  # True when it fits voice-note shape
    issues: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audio_to_voice_note(duration_s: float = 0.0, fmt: str = "",
                        size_bytes: int = 0) -> VoiceNoteSpec:
    """Describe audio as a voice note. No transcoding is performed.

    ``duration_s`` / ``fmt`` / ``size_bytes`` are caller-supplied
    metadata (from the platform message or a real probe upstream).
    The result says whether the audio fits voice-note shape
    (≤ 5 min, known format) and what is wrong when it doesn't.
    """
    fmt = (fmt or "").lower().strip().lstrip(".")
    issues: list[str] = []
    if fmt and fmt not in SUPPORTED_AUDIO_FORMATS:
        issues.append(f"format {fmt!r} is not a known audio format "
                      f"(known: {', '.join(SUPPORTED_AUDIO_FORMATS)})")
    if duration_s < 0:
        issues.append("duration cannot be negative")
    if duration_s > 300:
        issues.append("longer than 5 minutes — split it before sending as a voice note")
    if size_bytes < 0:
        issues.append("size cannot be negative")
    spec = VoiceNoteSpec(
        id="vn_" + new_short_id(),
        source_format=fmt,
        duration_s=max(0.0, float(duration_s or 0.0)),
        size_bytes=max(0, int(size_bytes or 0)),
        voice_note=not issues,
        issues=issues,
        note=("Descriptor only: no audio was re-encoded. Send the original "
              "file with the voice-note flag, or transcode upstream first."))
    return spec


# ── video trim / upscale descriptors ───────────────────────────────────────


@dataclass
class TrimPlan:
    """A trim plan: computed ranges, not processed video."""

    id: str = ""
    source_duration_s: float = 0.0
    start_s: float = 0.0
    end_s: float = 0.0
    out_duration_s: float = 0.0
    clamped: bool = False
    issues: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def trim_video(duration_s: float, start_s: float, end_s: float) -> TrimPlan:
    """Compute a trim plan. Clamps to the real duration; refuses empties.

    Returns a plan the chat layer can execute (e.g. with ffmpeg) — this
    function itself touches no video bytes.
    """
    issues: list[str] = []
    try:
        duration = max(0.0, float(duration_s or 0.0))
        start = float(start_s or 0.0)
        end = float(end_s or 0.0)
    except (TypeError, ValueError):
        return TrimPlan(issues=["trim needs numeric duration/start/end"],
                        note="Descriptor only: no video was processed.")
    clamped = False
    if start < 0:
        start, clamped = 0.0, True
        issues.append("start clamped to 0")
    if end > duration:
        end, clamped = duration, True
        issues.append("end clamped to the video duration")
    if end <= start:
        issues.append("end must be after start — nothing to trim")
        return TrimPlan(id="trim_" + new_short_id(),
                        source_duration_s=duration, start_s=start, end_s=end,
                        out_duration_s=0.0, clamped=clamped, issues=issues,
                        note="Descriptor only: no video was processed.")
    return TrimPlan(
        id="trim_" + new_short_id(),
        source_duration_s=duration, start_s=start, end_s=end,
        out_duration_s=end - start, clamped=clamped, issues=issues,
        note=("Descriptor only: no video was processed. Execute with "
              f"ffmpeg -ss {start:g} -to {end:g} -i <input> -c copy <output>."))


@dataclass
class UpscalePlan:
    """An upscale target: computed geometry, not processed pixels."""

    id: str = ""
    source_w: int = 0
    source_h: int = 0
    target_w: int = 0
    target_h: int = 0
    scale: float = 1.0
    issues: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def upscale_descriptor(width: int, height: int,
                       target: str = "1080p") -> UpscalePlan:
    """Compute an upscale target from source geometry.

    ``target``: "720p" | "1080p" | "4k" (height-based). Aspect ratio is
    preserved; the plan never claims pixels were enhanced — upscaling
    stretches, it does not add detail, and the plan says so.
    """
    targets = {"720p": 720, "1080p": 1080, "4k": 2160}
    issues: list[str] = []
    try:
        w, h = int(width or 0), int(height or 0)
    except (TypeError, ValueError):
        return UpscalePlan(issues=["upscale needs numeric width/height"],
                           note="Descriptor only: no pixels were processed.")
    if w <= 0 or h <= 0:
        return UpscalePlan(issues=["width and height must be positive"],
                           note="Descriptor only: no pixels were processed.")
    target_h = targets.get(str(target).lower(), 1080)
    if str(target).lower() not in targets:
        issues.append(f"unknown target {target!r} — defaulted to 1080p")
    scale = target_h / h
    if scale < 1.0:
        issues.append("target is smaller than the source — that's a downscale, "
                      "not an upscale; geometry still computed")
    return UpscalePlan(
        id="up_" + new_short_id(), source_w=w, source_h=h,
        target_w=round(w * scale), target_h=target_h, scale=round(scale, 3),
        issues=issues,
        note=("Descriptor only: no pixels were processed. Upscaling "
              "stretches the existing picture — it cannot add detail "
              "that was never recorded."))


# ── voice-note transcription ────────────────────────────────────────────────


@dataclass
class Transcript:
    """A voice-note transcript — always supplied, never generated here."""

    id: str = ""
    voice_note_id: str = ""
    text: str = ""
    source: str = ""  # who supplied it: "platform-captions" | "human" | "external-stt" | …
    language: str = ""
    confidence: float | None = None  # only when the supplier reported one
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Transcript":
        return cls(
            id=str(data.get("id", "")),
            voice_note_id=str(data.get("voice_note_id", "")),
            text=str(data.get("text", "")),
            source=str(data.get("source", "")),
            language=str(data.get("language", "")),
            confidence=data.get("confidence"),
            created_at=float(data.get("created_at") or 0.0),
        )


def transcribe_voice_note(voice_note_id: str = "",
                          transcript: str | None = None,
                          *, source: str = "", language: str = "",
                          confidence: float | None = None) -> Transcript | None:
    """Accept a transcript for a voice note. Never fakes one.

    ``transcript`` must be supplied by the caller — platform captions,
    a human typing it out, or an external STT result passed in with its
    ``source`` named. When ``transcript`` is None/empty this returns
    None: no transcript exists, and inventing words for someone's voice
    message would be fabrication. ``confidence`` is stored only when
    the supplier reported one; it is never invented either.
    """
    text = (transcript or "").strip()
    if not text:
        return None
    conf: float | None = None
    if confidence is not None:
        try:
            conf = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            conf = None
    return Transcript(
        id="tx_" + new_short_id(),
        voice_note_id=(voice_note_id or "").strip(),
        text=text[:5000],
        source=(source or "supplied").strip()[:80] or "supplied",
        language=(language or "").strip()[:20],
        confidence=conf,
        created_at=time.time(),
    )


def format_transcript(tx: Transcript) -> str:
    """Render a transcript as chat text, with its provenance."""
    conf = f" (confidence {tx.confidence:.0%})" if tx.confidence is not None else ""
    lang = f" [{tx.language}]" if tx.language else ""
    return (f"🗣️ Transcript{lang} — via {tx.source}{conf}:\n"
            f"{tx.text}")

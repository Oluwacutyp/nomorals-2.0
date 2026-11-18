"""Video briefs — Summarize.tech (52-websites #39): a long video as a
sectioned brief.

Paste a YouTube link (or a local video path) and get the video back as
a brief: timestamped sections with headings, each section's summary and
key points, an overall TLDR, every section deep-linking to its moment,
and the brief naming which transcript it was built from.

The pipeline is transcript-driven, composing organs that already
exist — no parallel system:

- caption track first: yt-dlp subtitle fetch (manual captions over
  auto-captions), parsed from WebVTT into word-timed words;
- ASR fallback: the video is downloaded through the media pipeline and
  transcribed with :func:`nomorals.media_edit.captions.transcribe_words`
  (faster-whisper, word-timed);
- sectioning: ~5-minute windows aligned to the transcript's own
  timestamps (a short tail merges into the previous section);
- summarising: :mod:`nomorals.documents.summarize` — extractive by
  default (deterministic, every sentence in a summary is the
  transcript's own words), abstractive only when asked and a brain is
  available, falling back to extractive and saying so;
- translation: a foreign-language transcript is translated to English
  through the translator ladder (:mod:`nomorals.llm.translate`), the
  original title kept, the source language named.

Honest limits, stated where the brief is shown: quality is bounded by
transcript quality — auto-captions on accented or technical speech
degrade the brief, and the brief says which transcript it was built
from (caption track vs ASR) and which language. A video with neither
a caption track nor a transcribable audio track is refused honestly —
no section, fact or timestamp is ever invented. Word timings inside a
caption cue are spread evenly across the cue's own start/end: the cue
timings are the transcript's, the interpolation inside a cue is the
organ's, and section boundaries only ever cut at word starts.

The YouTube multi-search legs are NOT used here: transcript
acquisition rides the caption fetcher and the download/transcribe
rungs, never the search legs.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from ..documents.summarize import DocumentError, keyphrases, summarize_text, tldr
from ..media_edit.captions import Word, transcribe_words

_log = get_logger("nomorals.media.video_brief")

__all__ = [
    "BriefSection",
    "VideoBrief",
    "VideoBriefStore",
    "brief_video",
    "build_brief",
    "deep_link",
    "format_brief_reply",
    "format_ts",
    "parse_video_ref",
    "parse_vtt",
    "section_words",
]

#: Section window in seconds — the reference's published summaries show
#: ~5-minute section boundaries.
WINDOW_S = 300.0

#: A trailing window shorter than this merges into the previous section
#: instead of standing alone as a stub.
MIN_TAIL_S = 60.0

#: Sentences per section summary (extractive default).
SECTION_SENTENCES = 2

#: Key points per section.
SECTION_KEY_POINTS = 5

#: Words in the overall TLDR.
TLDR_WORDS = 60

BRIEF_FILENAME = "video_briefs.json"

_YT_ID_RE = re.compile(
    r"(?:v=|youtu\.be/|/shorts/|/embed/|/live/|youtube:)"
    r"([A-Za-z0-9_\-]{11})")
_YT_BARE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{11}$")
_VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")

# HH:MM:SS.mmm or MM:SS.mmm, both ends of a cue.
_TS_RE = re.compile(
    r"(?:(\d+):)?(\d+):(\d+)\.(\d+)\s*-->\s*"
    r"(?:(\d+):)?(\d+):(\d+)\.(\d+)")
_TAG_RE = re.compile(r"<[^>]+>")

_LANG_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "pt": "Portuguese", "it": "Italian", "yo": "Yoruba", "ha": "Hausa",
    "ig": "Igbo", "ar": "Arabic", "zh": "Chinese", "ja": "Japanese",
    "ko": "Korean", "hi": "Hindi", "ru": "Russian", "nl": "Dutch",
    "sv": "Swedish", "tr": "Turkish", "pl": "Polish", "uk": "Ukrainian",
}


def _default_home(home: str | Path | None) -> Path:
    if home:
        return Path(home)
    return Path("~/.nomorals").expanduser()


def _ts_to_seconds(hours: str | None, minutes: str, seconds: str,
                   millis: str) -> float:
    total = int(minutes) * 60 + int(seconds) + int(millis) / 1000.0
    if hours:
        total += int(hours) * 3600
    return total


def format_ts(seconds: float) -> str:
    """Seconds -> MM:SS (or H:MM:SS past the hour)."""
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def deep_link(video_id: str, start: float) -> str:
    """YouTube deep link to a section's moment."""
    return ("https://www.youtube.com/watch?v="
            f"{video_id}&t={int(start)}")


def parse_video_ref(ref: str) -> dict[str, Any]:
    """A pasted ref -> {kind, video_id, url, path}.

    ``kind`` is ``youtube`` (watch / youtu.be / shorts / embed / live /
    bare 11-char id), ``local`` (a path to a video file) or ``unknown``.
    Never raises.
    """
    text = (ref or "").strip()
    if not text:
        return {"kind": "unknown", "video_id": "", "url": "", "path": ""}
    match = _YT_ID_RE.search(text)
    if match:
        video_id = match.group(1)
        return {"kind": "youtube", "video_id": video_id,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "path": ""}
    if _YT_BARE_ID_RE.match(text):
        return {"kind": "youtube", "video_id": text,
                "url": f"https://www.youtube.com/watch?v={text}",
                "path": ""}
    low = text.lower()
    if text.startswith(("/", "./", "~/", "http://", "https://")) or \
            low.endswith(_VIDEO_EXTS):
        if low.startswith(("http://", "https://")):
            return {"kind": "unknown", "video_id": "", "url": text,
                    "path": ""}
        return {"kind": "local", "video_id": "", "url": "",
                "path": text}
    return {"kind": "unknown", "video_id": "", "url": text, "path": ""}


def _norm_words(words: Any) -> list[Word]:
    """Word list or dicts -> [Word]. Never raises."""
    out: list[Word] = []
    for item in words or []:
        try:
            if isinstance(item, Word):
                out.append(item)
            elif isinstance(item, dict):
                out.append(Word.from_dict(item))
        except (TypeError, ValueError, KeyError):
            continue
    return [w for w in out if w.text.strip()]


def parse_vtt(text: str) -> list[Word]:
    """WebVTT -> word-timed words.

    Cue timings are the transcript's own; words inside one cue are
    spread evenly across the cue's start/end (a cue's words share its
    timing, so even spacing is the honest interpolation). Tags are
    stripped, entities unescaped. Malformed input yields [] — never
    raises.
    """
    words: list[Word] = []
    if not text:
        return words
    lines = text.splitlines()
    i = 0
    total = len(lines)
    while i < total:
        line = lines[i].strip()
        i += 1
        if not line or line.upper().startswith(("WEBVTT", "NOTE")):
            continue
        match = _TS_RE.search(line)
        if not match:
            continue
        try:
            start = _ts_to_seconds(match.group(1), match.group(2),
                                   match.group(3), match.group(4))
            end = _ts_to_seconds(match.group(5), match.group(6),
                                 match.group(7), match.group(8))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        cue_lines: list[str] = []
        while i < total and lines[i].strip():
            cue_lines.append(lines[i].strip())
            i += 1
        raw = " ".join(cue_lines)
        raw = _TAG_RE.sub("", raw)
        raw = html.unescape(raw)
        tokens = [t for t in raw.split() if t.strip()]
        if not tokens:
            continue
        span = (end - start) / len(tokens)
        for index, token in enumerate(tokens):
            words.append(Word(start=start + index * span,
                              end=start + (index + 1) * span,
                              text=token))
    return words


def section_words(
        words: list[Word] | list[dict[str, Any]],
        window_s: float = WINDOW_S,
        min_tail_s: float = MIN_TAIL_S,
) -> list[tuple[float, float, list[Word]]]:
    """Group word-timed words into ~``window_s`` windows.

    A word belongs to the window its start falls in, so boundaries cut
    only at word starts. A trailing window shorter than ``min_tail_s``
    merges into the previous one. Returns [(start, end, words)] with
    ``end`` the last word's end in the window. Empty input -> [].
    """
    timed = _norm_words(words)
    if not timed:
        return []
    window = max(60.0, float(window_s or WINDOW_S))
    buckets: dict[int, list[Word]] = {}
    for word in timed:
        buckets.setdefault(int(word.start // window), []).append(word)
    out: list[tuple[float, float, list[Word]]] = []
    for key in sorted(buckets):
        group = buckets[key]
        out.append((key * window, max(w.end for w in group), group))
    if len(out) > 1:
        last_start, last_end, _ = out[-1]
        if last_end - last_start < max(0.0, float(min_tail_s or 0.0)):
            prev_start, _, prev_words = out[-2]
            _, _, last_words = out[-1]
            out[-2] = (prev_start, last_end, prev_words + last_words)
            out.pop()
    return out


# ── data ────────────────────────────────────────────────────────────────


@dataclass
class BriefSection:
    """One section of a brief: a time window, grounded in the transcript."""

    start: float = 0.0
    end: float = 0.0
    heading: str = ""
    summary: str = ""
    key_points: list[str] = field(default_factory=list)
    deep_link: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end,
                "heading": self.heading, "summary": self.summary,
                "key_points": list(self.key_points),
                "deep_link": self.deep_link}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BriefSection:
        return cls(
            start=float(data.get("start", 0.0) or 0.0),
            end=float(data.get("end", 0.0) or 0.0),
            heading=str(data.get("heading", "") or ""),
            summary=str(data.get("summary", "") or ""),
            key_points=[str(p) for p in (data.get("key_points") or [])],
            deep_link=str(data.get("deep_link", "") or ""))


@dataclass
class VideoBrief:
    """A video brief: sections + TLDR, every line grounded in the transcript."""

    ok: bool = False
    brief_id: str = ""
    video_ref: str = ""
    video_id: str = ""
    title: str = ""
    duration: float = 0.0
    language: str = "en"
    transcript_source: str = ""
    transcript_language: str = ""
    translated: bool = False
    sections: list[BriefSection] = field(default_factory=list)
    overall_tldr: str = ""
    summary_rung: str = "extractive"
    limits: list[str] = field(default_factory=list)
    reason: str = ""
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "brief_id": self.brief_id,
            "video_ref": self.video_ref, "video_id": self.video_id,
            "title": self.title, "duration": self.duration,
            "language": self.language,
            "transcript_source": self.transcript_source,
            "transcript_language": self.transcript_language,
            "translated": self.translated,
            "sections": [s.to_dict() for s in self.sections],
            "overall_tldr": self.overall_tldr,
            "summary_rung": self.summary_rung,
            "limits": list(self.limits), "reason": self.reason,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoBrief:
        return cls(
            ok=bool(data.get("ok", False)),
            brief_id=str(data.get("brief_id", "") or ""),
            video_ref=str(data.get("video_ref", "") or ""),
            video_id=str(data.get("video_id", "") or ""),
            title=str(data.get("title", "") or ""),
            duration=float(data.get("duration", 0.0) or 0.0),
            language=str(data.get("language", "en") or "en"),
            transcript_source=str(data.get("transcript_source", "") or ""),
            transcript_language=str(
                data.get("transcript_language", "") or ""),
            translated=bool(data.get("translated", False)),
            sections=[BriefSection.from_dict(s)
                      for s in (data.get("sections") or [])],
            overall_tldr=str(data.get("overall_tldr", "") or ""),
            summary_rung=str(data.get("summary_rung", "extractive")
                             or "extractive"),
            limits=[str(item) for item in (data.get("limits") or [])],
            reason=str(data.get("reason", "") or ""),
            created_at=float(data.get("created_at", 0.0) or 0.0))


# ── summarising ─────────────────────────────────────────────────────────


def _sentences_fallback(text: str) -> list[str]:
    """Plain sentence split for the fallback path (no scorer needed)."""
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _summarize_section(text: str, *, sentences: int = SECTION_SENTENCES,
                       abstractive_fn: Any = None) -> tuple[str, list[str], str]:
    """One section's summary + key points, from the section's own text.

    Returns (summary, key_points, rung). Extractive by default — the
    summary sentences ARE the transcript's sentences. With an
    ``abstractive_fn`` the summary is rewritten by the brain and the
    rung says so; a failed abstractive call falls back to extractive.
    """
    if abstractive_fn is not None:
        try:
            rewritten = str(abstractive_fn(text, max_words=120) or "").strip()
        except Exception:  # noqa: BLE001 — a dead rung sheds
            rewritten = ""
        if rewritten:
            try:
                points = keyphrases(text, top_n=SECTION_KEY_POINTS,
                                    max_words=3)
            except DocumentError:
                points = []
            return rewritten, points, "abstractive"
    try:
        picked = summarize_text(text, sentences=sentences, method="tf",
                                diversity=False, position_bias=False)
        summary = " ".join(picked)
    except DocumentError:
        summary = " ".join(_sentences_fallback(text)[:sentences])
    try:
        points = keyphrases(text, top_n=SECTION_KEY_POINTS, max_words=3)
    except DocumentError:
        points = []
    return summary, points, "extractive"


def _heading_for(text: str, index: int, start: float, end: float) -> str:
    """A section heading from the section's own keyphrases — never invented."""
    try:
        phrases = keyphrases(text, top_n=2, max_words=3)
    except DocumentError:
        phrases = []
    if phrases:
        head = phrases[0]
        return head[:1].upper() + head[1:] if head else head
    return (f"Part {index + 1} "
            f"({format_ts(start)}–{format_ts(end)})")


def _lang_name(code: str) -> str:
    return _LANG_NAMES.get((code or "").lower(), (code or "").upper())


def _translate_text(text: str, *, source: str, translate_fn: Any) -> str | None:
    """One text through the injected translator; None when it declines."""
    try:
        out = translate_fn(text, source=source, target="en")
    except Exception:  # noqa: BLE001 — a dead rung sheds
        return None
    if isinstance(out, str):
        return out.strip() or None
    translated = getattr(out, "translated", False)
    result = getattr(out, "text", "") or ""
    if translated and str(result).strip():
        return str(result).strip()
    return None


def build_brief(
        video_ref: str,
        words: list[Word] | list[dict[str, Any]],
        *,
        title: str = "",
        duration: float = 0.0,
        language: str = "en",
        transcript_source: str = "captions",
        window_s: float = WINDOW_S,
        min_tail_s: float = MIN_TAIL_S,
        sentences_per_section: int = SECTION_SENTENCES,
        abstractive: bool = False,
        abstractive_fn: Any = None,
        translate_fn: Any = None,
) -> VideoBrief:
    """Compose a brief from word-timed transcript words — pure, testable.

    Sections are ~``window_s`` windows aligned to the words' own
    timestamps; each section is summarised from its own words only, so
    a section's summary can never carry another section's facts. A
    foreign-language transcript is translated to English when a
    ``translate_fn`` is given (the original title is always kept); with
    no translator the brief stays in the transcript's language and says
    so. Never raises for empty input — it returns ``ok: False``.
    """
    ref = parse_video_ref(video_ref)
    video_id = ref["video_id"]
    brief_id = video_id or hashlib.sha1(
        video_ref.encode("utf-8")).hexdigest()[:12]
    timed = _norm_words(words)
    if not timed:
        return VideoBrief(
            ok=False, brief_id=brief_id, video_ref=video_ref,
            video_id=video_id, title=title,
            reason="no transcript words to brief — the video gave no "
                   "caption track and no transcribable audio",
            created_at=time.time())
    lang = (language or "en").lower()
    windows = section_words(timed, window_s=window_s, min_tail_s=min_tail_s)
    full_text = " ".join(w.text for w in timed)
    use_abstractive = bool(abstractive)
    abstractive_fn_resolved: Any = abstractive_fn
    if use_abstractive and abstractive_fn_resolved is None:
        try:
            from ..documents.summarize import summarize_abstractive
            abstractive_fn_resolved = summarize_abstractive
        except Exception:  # noqa: BLE001 — no brain, extractive it is
            abstractive_fn_resolved = None
    sections: list[BriefSection] = []
    rungs: set[str] = set()
    limits: list[str] = []
    for index, (start, end, group) in enumerate(windows):
        text = " ".join(w.text for w in group)
        summary, points, rung = _summarize_section(
            text, sentences=sentences_per_section,
            abstractive_fn=abstractive_fn_resolved
            if use_abstractive else None)
        rungs.add(rung)
        sections.append(BriefSection(
            start=start, end=end,
            heading=_heading_for(text, index, start, end),
            summary=summary, key_points=points,
            deep_link=deep_link(video_id, start) if video_id else ""))
    try:
        overall = tldr(full_text, words=TLDR_WORDS)
    except DocumentError:
        overall = sections[0].summary if sections else ""
    translated = False
    if lang and lang != "en":
        if translate_fn is not None:
            translated = True
            for section in sections:
                rendered = _translate_text(section.summary, source=lang,
                                           translate_fn=translate_fn)
                if rendered is not None:
                    section.summary = rendered
                else:
                    translated = False
                    limits.append(
                        f"section {format_ts(section.start)} could not "
                        f"be translated — it stays in {_lang_name(lang)}")
                rendered_points: list[str] = []
                for point in section.key_points:
                    rendered_point = _translate_text(
                        point, source=lang, translate_fn=translate_fn)
                    rendered_points.append(rendered_point or point)
                section.key_points = rendered_points
            rendered_tldr = _translate_text(overall, source=lang,
                                            translate_fn=translate_fn)
            if rendered_tldr is not None:
                overall = rendered_tldr
            else:
                translated = False
                limits.append("the TLDR could not be translated — it "
                              f"stays in {_lang_name(lang)}")
            if translated:
                limits.append(f"translated from {_lang_name(lang)} "
                              f"({lang}) into English; original title kept")
        else:
            limits.append(
                f"no translator available — this brief stays in "
                f"{_lang_name(lang)} ({lang})")
    return VideoBrief(
        ok=True, brief_id=brief_id, video_ref=video_ref,
        video_id=video_id, title=title or video_ref,
        duration=float(duration or max(w.end for w in timed)),
        language="en" if translated else lang,
        transcript_source=transcript_source,
        transcript_language=lang,
        translated=translated,
        sections=sections, overall_tldr=overall,
        summary_rung="abstractive" if rungs == {"abstractive"}
        else "extractive",
        limits=limits, created_at=time.time())


# ── acquisition ─────────────────────────────────────────────────────────


def _default_caption_fetcher(video_id: str, *,
                             languages: tuple[str, ...] = ("en",)
                             ) -> dict[str, Any]:
    """YouTube caption track via yt-dlp subtitle fetch (no video download).

    Manual captions win over auto-captions; the first listed language
    wins. Returns {words, language, manual, title, reason}. Never
    raises — a missing yt-dlp or a captionless video is a reason, not
    an exception.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        import yt_dlp
        has_module = True
    except ImportError:
        has_module = False
    if not has_module and not shutil.which("yt-dlp"):
        return {"words": None, "language": "",
                "reason": "yt-dlp is not installed — captions need yt-dlp"}
    try:
        with tempfile.TemporaryDirectory(prefix="videobrief-caps-") as tmp:
            tmpdir = Path(tmp)
            if has_module:
                import yt_dlp
                opts: dict[str, Any] = {
                    "quiet": True, "no_warnings": True,
                    "skip_download": True,
                    "writesubtitles": True, "writeautomaticsub": True,
                    "subtitleslangs": list(languages),
                    "subtitlesformat": "vtt/best",
                    "outtmpl": str(tmpdir / "%(id)s.%(ext)s"),
                }
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                title = str((info or {}).get("title", "") or "")
            else:
                langs = ",".join(languages)
                proc = subprocess.run(
                    ["yt-dlp", "--write-subs", "--write-auto-subs",
                     "--sub-langs", langs, "--sub-format", "vtt/best",
                     "--skip-download", "-o",
                     str(tmpdir / "%(id)s.%(ext)s"), url],
                    capture_output=True, text=True, encoding="utf-8",
                    errors="replace", timeout=120)
                title = ""
                if proc.returncode != 0:
                    return {"words": None, "language": "",
                            "reason": "yt-dlp subtitle fetch failed: "
                            f"{(proc.stderr or '').strip()[-200:]}"}
            for lang in languages:
                for manual in (True, False):
                    suffix = f".{lang}.vtt" if manual else f".{lang}.auto.vtt"
                    matches = sorted(tmpdir.glob(f"*{suffix}"))
                    if matches:
                        text = matches[0].read_text(
                            encoding="utf-8", errors="replace")
                        words = parse_vtt(text)
                        if words:
                            return {"words": words, "language": lang,
                                    "manual": manual, "title": title,
                                    "reason": ""}
            return {"words": None, "language": "",
                    "reason": "no captions on this video"}
    except Exception as exc:  # noqa: BLE001 — captions never raise
        return {"words": None, "language": "",
                "reason": f"caption fetch failed: {exc}"}


def _default_download_fn(ref: str, context: Any) -> str:
    """Download a URL through the media pipeline -> local path."""
    from ..media.video import VideoFinder
    out = VideoFinder(context).download(ref, audio_only=False)
    path = str(out.get("path") or out)
    if not path:
        raise ValueError("download returned no path")
    return path


def _default_translate_fn(context: Any) -> Any:
    """The translator ladder as a (text, *, source, target) -> str|None fn."""
    from ..llm.translate import translator_for
    translator = translator_for(context)

    def _translate(text: str, *, source: str, target: str) -> str | None:
        outcome = translator.translate(text, target=target, source=source)
        if getattr(outcome, "translated", False):
            return str(getattr(outcome, "text", "") or "").strip() or None
        return None

    return _translate


def brief_video(
        ref: str,
        *,
        home: str | Path | None = None,
        context: Any = None,
        caption_fetcher: Any = None,
        download_fn: Any = None,
        transcribe_fn: Any = None,
        window_s: float = WINDOW_S,
        abstractive: bool = False,
        abstractive_fn: Any = None,
        translate_fn: Any = None,
        store: Any = None,
) -> VideoBrief:
    """Brief a video: captions first, ASR fallback, honest refusal.

    ``caption_fetcher`` takes a video id and returns
    {words, language, manual, title, reason}; ``download_fn`` takes the
    ref and returns a local path; ``transcribe_fn`` takes a local path
    and returns word-timed words (or (words, language)). Defaults wire
    the real rungs. The brief is saved to the durable store when one is
    given (or built from ``home``). Never raises for a bad ref or a
    dead rung — it returns ``ok: False`` with the reason.
    """
    parsed = parse_video_ref(ref)
    if parsed["kind"] == "unknown":
        return VideoBrief(
            ok=False, video_ref=ref,
            reason=f"not a YouTube link or video file: {ref}",
            created_at=time.time())
    failures: list[str] = []
    title = ""
    language = "en"
    words: list[Word] = []
    source = ""
    caption_manual = True
    if parsed["kind"] == "youtube":
        fetcher = caption_fetcher or _default_caption_fetcher
        try:
            got = fetcher(parsed["video_id"])
        except Exception as exc:  # noqa: BLE001 — captions never raise
            got = {"words": None, "reason": f"caption fetch failed: {exc}"}
        fetched = _norm_words((got or {}).get("words"))
        if fetched:
            words = fetched
            source = "captions"
            language = str((got or {}).get("language") or "en")
            title = str((got or {}).get("title") or "")
            caption_manual = bool((got or {}).get("manual"))
        else:
            failures.append("captions: " + str(
                (got or {}).get("reason") or "no captions on this video"))
    if not words:
        # ASR fallback: download (YouTube) or transcribe the file as-is.
        path = ""
        try:
            if parsed["kind"] == "youtube":
                if download_fn is not None:
                    path = str(download_fn(ref))
                elif context is not None:
                    path = _default_download_fn(ref, context)
                else:
                    raise ValueError(
                        "no download route — pass a context or download_fn")
            else:
                path = str(parsed["path"])
            transcribe = transcribe_fn or transcribe_words
            got_words = transcribe(path)
            asr_lang = ""
            if isinstance(got_words, tuple):
                got_words, asr_lang = got_words
            words = _norm_words(got_words)
            if words:
                source = "asr"
                language = str(asr_lang or "")
            else:
                failures.append("transcription: the audio transcribed "
                                "to nothing")
        except Exception as exc:  # noqa: BLE001 — ASR never raises
            failures.append("download/transcription: " + str(exc))
    if not words:
        return VideoBrief(
            ok=False, video_ref=ref,
            reason="no transcript to brief from — "
                   + "; ".join(failures),
            created_at=time.time())
    resolved_translate = translate_fn
    if resolved_translate is None and context is not None:
        try:
            resolved_translate = _default_translate_fn(context)
        except Exception:  # noqa: BLE001 — no translator, brief stays put
            resolved_translate = None
    brief = build_brief(
        ref, words, title=title, language=language or "en",
        transcript_source=source, window_s=window_s,
        abstractive=abstractive, abstractive_fn=abstractive_fn,
        translate_fn=resolved_translate)
    if brief.ok and source == "captions":
        brief.limits.append(
            "built from the caption track"
            + ("" if caption_manual
               else " (auto-captions — quality may vary)"))
    if brief.ok and source == "asr":
        brief.limits.append("built from ASR transcription (faster-whisper)"
                            " — quality follows the audio")
    if brief.ok:
        target_store = store or VideoBriefStore(home)
        try:
            target_store.save(brief)
        except Exception:  # noqa: BLE001 — a dead store never kills a brief
            _log.warning("video_brief: store save failed", exc_info=True)
    return brief


# ── durable store ───────────────────────────────────────────────────────


class VideoBriefStore:
    """Durable video briefs: <home>/data/video_briefs.json.

    Atomic writes (temp file + replace). A brief holds the ref, the
    title, the transcript source and language, the sections with their
    summaries and deep links, and the TLDR — so a brief can be re-read
    and re-shown across restarts without re-briefing the video.
    """

    def __init__(self, path: str | Path | None = None,
                 home: str | Path | None = None) -> None:
        if path is not None:
            self.path = str(path)
        else:
            self.path = str(_default_home(home) / "data" / BRIEF_FILENAME)
        self._data: dict[str, Any] = {"briefs": {}}
        self._load()

    def _load(self) -> None:
        try:
            raw = Path(self.path).read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict) and isinstance(
                    data.get("briefs"), dict):
                self._data = data
        except (OSError, ValueError):
            self._data = {"briefs": {}}

    def _save(self) -> None:
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        tmp.replace(path)

    def save(self, brief: VideoBrief) -> None:
        self._data["briefs"][brief.brief_id] = brief.to_dict()
        self._save()

    def get(self, brief_id: str) -> dict[str, Any] | None:
        return self._data["briefs"].get(brief_id)

    def list(self) -> list[dict[str, Any]]:
        """Brief metadata, newest first."""
        out = [{
            "id": item.get("brief_id", ""), "title": item.get("title", ""),
            "video_ref": item.get("video_ref", ""),
            "created_at": item.get("created_at", 0.0),
            "sections": len(item.get("sections") or []),
            "duration": item.get("duration", 0.0),
            "transcript_source": item.get("transcript_source", ""),
        } for item in self._data["briefs"].values()]
        out.sort(key=lambda row: row["created_at"], reverse=True)
        return out


# ── rendering ───────────────────────────────────────────────────────────


def format_brief_reply(brief: VideoBrief) -> str:
    """A brief in the owner's standing format: title, TLDR, sections
    with timestamps and deep links, key points, and the transcript
    source line. An unavailable brief renders its reason, never a
    fabricated summary."""
    if not brief.ok:
        return ("no brief for that video — "
                + (brief.reason or "no transcript available"))
    lines = [f"🎬 {brief.title or brief.video_ref}"]
    source_word = ("caption track" if brief.transcript_source == "captions"
                   else "ASR transcription" if brief.transcript_source == "asr"
                   else brief.transcript_source or "unknown transcript")
    lang_word = _lang_name(brief.transcript_language)
    lines.append(f"{format_ts(brief.duration)} · {source_word}"
                 + (f" ({lang_word})" if brief.transcript_language else "")
                 + (" · translated to English" if brief.translated else ""))
    if brief.overall_tldr:
        lines.append(f"\nTLDR: {brief.overall_tldr}")
    for section in brief.sections:
        stamp = f"{format_ts(section.start)}–{format_ts(section.end)}"
        head = f"\n⏱ {stamp} — {section.heading}"
        if section.deep_link:
            head += f" ({section.deep_link})"
        lines.append(head)
        if section.summary:
            lines.append(section.summary)
        for point in section.key_points:
            lines.append(f"• {point}")
    for limit in brief.limits:
        lines.append(f"⚠️ {limit}")
    return "\n".join(lines)

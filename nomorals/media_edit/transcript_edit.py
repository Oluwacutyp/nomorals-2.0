"""Transcript-first video editing ("cut the umms") — the Descript pattern.

Send a video → Devon transcribes it (word-level, via captions.py) → you
say "cut the umms" / "remove silences" / "keep only the part about
pricing" → Devon cuts by transcript timestamps.

Primitives are :func:`videos.trim` and :func:`videos.concat` only —
no re-encoding hacks. Silence detection is plain ffmpeg
``silencedetect`` (stdlib subprocess, no librosa). The "smart" edit
(:func:`edit_by_transcript`) takes an explicit ``llm_fn`` — when it is
None we raise rather than guess with heuristics and call it smart.

Range validation is strict: an LLM returning garbage ranges must never
corrupt the video.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from ..core.logging_setup import get_logger
from .captions import (
    Word,
    load_transcript,
    save_transcript,
    transcribe_words,
)
from .videos import (
    MediaEditError,
    concat,
    ffmpeg_path,
    trim,
    video_probe,
)

_log = get_logger(__name__)

#: Filler words/phrases cut by "cut the umms". Checked as whole words —
#: "umbrella" must never match "um".
DEFAULT_FILLERS: tuple[str, ...] = (
    "um", "uh", "umm", "uhm", "er", "ah", "like", "you know",
)


# ---------------------------------------------------------------------------
# transcript wrapper
# ---------------------------------------------------------------------------

@dataclass
class TimestampedTranscript:
    """A word-timed transcript with phrase search → time ranges."""

    words: list[Word] = field(default_factory=list)

    @classmethod
    def from_words(cls, words: Sequence[Word | dict[str, Any]]) -> "TimestampedTranscript":
        out = [w if isinstance(w, Word) else Word.from_dict(w) for w in words]
        return cls(words=out)

    @classmethod
    def load(cls, video_path: str | os.PathLike[str]) -> "TimestampedTranscript | None":
        """Load the cached ``.words.json`` (see captions.py), or None."""
        words = load_transcript(video_path)
        return cls(words=words) if words else None

    @classmethod
    def transcribe(cls, video_path: str | os.PathLike[str], *,
                   language: str = "en", cache: bool = True) -> "TimestampedTranscript":
        """Transcribe and (by default) cache the words next to the video."""
        words = transcribe_words(video_path, language=language)
        if cache:
            save_transcript(video_path, words)
        return cls(words=words)

    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    def ranges_for(self, phrase: str, *,
                   threshold: float = 0.6) -> list[tuple[float, float]]:
        """Fuzzy-find ``phrase`` in the transcript → [(start, end), ...].

        Multi-word phrases are matched against sliding windows of words
        with difflib; single words use case-insensitive whole-word match.
        Returns the time span of each best-matching window per hit.
        """
        phrase = (phrase or "").strip().lower()
        if not phrase or not self.words:
            return []
        tokens = phrase.split()
        n = len(tokens)
        hits: list[tuple[float, float]] = []
        if n == 1:
            for w in self.words:
                if w.text.strip().lower().strip(".,!?;:\"'()") == tokens[0]:
                    hits.append((w.start, w.end))
            return hits
        # sliding window over word texts
        texts = [w.text.strip().lower().strip(".,!?;:\"'()") for w in self.words]
        joined = [" ".join(texts[i:i + n]) for i in range(len(texts) - n + 1)]
        target = " ".join(tokens)
        for i, window in enumerate(joined):
            if difflib.SequenceMatcher(None, window, target).ratio() >= threshold:
                hits.append((self.words[i].start, self.words[i + n - 1].end))
        return hits


# ---------------------------------------------------------------------------
# range math (pure — fully unit-testable)
# ---------------------------------------------------------------------------

def _merge_ranges(ranges: list[tuple[float, float]],
                  *, gap: float = 0.05) -> list[tuple[float, float]]:
    """Sort + merge overlapping or near-adjacent ranges."""
    rs = sorted((float(s), float(e)) for s, e in ranges if float(e) > float(s))
    out: list[tuple[float, float]] = []
    for s, e in rs:
        if out and s <= out[-1][1] + gap:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _complement(cuts: list[tuple[float, float]],
                start: float, end: float) -> list[tuple[float, float]]:
    """Ranges to KEEP = [start, end) minus the cut ranges."""
    cuts = _merge_ranges(cuts)
    keeps: list[tuple[float, float]] = []
    cur = start
    for s, e in cuts:
        if s > cur:
            keeps.append((cur, min(s, end)))
        cur = max(cur, e)
        if cur >= end:
            break
    if cur < end:
        keeps.append((cur, end))
    return [(s, e) for s, e in keeps if e - s > 1e-3]


def _pad(ranges: list[tuple[float, float]], pad: float,
         start: float, end: float) -> list[tuple[float, float]]:
    """Expand each range by ``pad`` seconds, clamped to [start, end]."""
    return _merge_ranges([(max(start, s - pad), min(end, e + pad))
                          for s, e in ranges])


def _duration(video: str | os.PathLike[str]) -> float:
    info = video_probe(video)
    dur = float(info.get("duration") or 0)
    if dur <= 0:
        raise MediaEditError(f"could not determine duration of {video}")
    return dur


def _assemble(video: str | os.PathLike[str],
              keeps: list[tuple[float, float]], *,
              out_dir: str | os.PathLike[str] | None,
              suffix: str) -> dict[str, Any]:
    """Trim each kept range, concat into one file. trim/concat only."""
    if not keeps:
        raise MediaEditError("nothing to keep — every range was cut")
    parts: list[str] = []
    for i, (s, e) in enumerate(keeps):
        res = trim(video, s, e, out_dir=out_dir, suffix=f"{suffix}-p{i}")
        parts.append(res["output"])
    if len(parts) == 1:
        return {"input": str(video), "output": parts[0], "segments": 1,
                "mode": "trim-only"}
    res = concat(parts, out_dir=out_dir, suffix=suffix)
    for p in parts:  # tidy intermediate trims
        try:
            Path(p).unlink(missing_ok=True)
        except OSError:
            pass
    return {"input": str(video), "output": res["output"],
            "segments": len(parts), "mode": "trim+concat"}


# ---------------------------------------------------------------------------
# filler removal
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z']+")


def find_fillers(words: Sequence[Word],
                 fillers: Sequence[str] = DEFAULT_FILLERS) -> list[tuple[float, float]]:
    """Locate filler words (whole-word, case-insensitive).

    Multi-word fillers like "you know" match consecutive words. "umbrella"
    does not match "um" — the check is on extracted word tokens.
    """
    filler_sets = [f.lower().split() for f in fillers if f.strip()]
    if not filler_sets or not words:
        return []
    # normalize word tokens
    toks = [_WORD_RE.findall(w.text.lower()) for w in words]
    hits: list[tuple[float, float]] = []
    for f in filler_sets:
        n = len(f)
        for i in range(len(words) - n + 1):
            window = [t for j in range(n) for t in toks[i + j]]
            if window == f:
                hits.append((words[i].start, words[i + n - 1].end))
    return _merge_ranges(hits)


def remove_fillers(video: str | os.PathLike[str],
                   words: Sequence[Word | dict[str, Any]] | None = None, *,
                   fillers: Sequence[str] = DEFAULT_FILLERS,
                   pad: float = 0.2,
                   out_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Cut filler words ("um", "uh", "you know"...) with ``pad`` seconds of
    breathing room. No fillers found → the original path is returned with
    a note; never a fake edit."""
    words = [w if isinstance(w, Word) else Word.from_dict(w)
             for w in (words or [])]
    if not words:
        loaded = load_transcript(video)
        if loaded:
            words = loaded
        else:
            words = transcribe_words(video)
            save_transcript(video, words)
    cuts = _pad(find_fillers(words, fillers), pad, 0.0, _duration(video))
    if not cuts:
        return {"input": str(video), "output": str(video), "cut": 0,
                "note": "no filler words found — returned the original"}
    dur = _duration(video)
    keeps = _complement(cuts, 0.0, dur)
    res = _assemble(video, keeps, out_dir=out_dir, suffix="nofillers")
    res["cut"] = len(cuts)
    res["note"] = f"cut {len(cuts)} filler region(s)"
    return res


# ---------------------------------------------------------------------------
# silence removal (ffmpeg silencedetect, stdlib subprocess)
# ---------------------------------------------------------------------------

_SIL_START = re.compile(r"silence_start:\s*([0-9.]+)")
_SIL_END = re.compile(r"silence_end:\s*([0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)")


def detect_silences(video: str | os.PathLike[str], *,
                    threshold_db: float = -40.0,
                    min_silence_s: float = 0.5,
                    timeout: float = 300.0) -> list[tuple[float, float]]:
    """Find silent spans via ffmpeg's silencedetect filter.

    Returns [(start, end), ...]. Never raises on parse issues — returns
    what it could parse. ffmpeg itself failing raises MediaEditError.
    """
    ff = ffmpeg_path()
    filt = f"silencedetect=noise={threshold_db}dB:d={min_silence_s}"
    try:
        proc = subprocess.run(
            [ff, "-hide_banner", "-nostats", "-i", str(video),
             "-af", filt, "-f", "null", "-"],
            capture_output=True, text=True, timeout=timeout,
            start_new_session=True)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MediaEditError(f"silence detection failed: {exc}") from exc
    spans: list[tuple[float, float]] = []
    cur_start: float | None = None
    for line in proc.stderr.splitlines():
        m = _SIL_START.search(line)
        if m:
            cur_start = float(m.group(1))
            continue
        m = _SIL_END.search(line)
        if m and cur_start is not None:
            spans.append((cur_start, float(m.group(1))))
            cur_start = None
    return spans


def remove_silences(video: str | os.PathLike[str], *,
                    threshold_db: float = -40.0,
                    min_silence_s: float = 0.5,
                    pad: float = 0.15,
                    out_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Cut silent spans (with ``pad`` kept on each side so speech isn't
    clipped). No silences → original returned with a note."""
    dur = _duration(video)
    cuts = _pad(detect_silences(video, threshold_db=threshold_db,
                                min_silence_s=min_silence_s),
                pad, 0.0, dur)
    if not cuts:
        return {"input": str(video), "output": str(video), "cut": 0,
                "note": "no silences found — returned the original"}
    keeps = _complement(cuts, 0.0, dur)
    res = _assemble(video, keeps, out_dir=out_dir, suffix="nosilence")
    res["cut"] = len(cuts)
    res["note"] = f"cut {len(cuts)} silent span(s)"
    return res


# ---------------------------------------------------------------------------
# LLM-driven freeform edit
# ---------------------------------------------------------------------------

def _validate_ranges(raw: Any, duration: float) -> list[tuple[float, float]]:
    """Hard validation of LLM-proposed keep ranges. Raises on garbage."""
    if not isinstance(raw, list) or not raw:
        raise MediaEditError(
            f"edit model returned no usable ranges (got {raw!r:.80})")
    ranges: list[tuple[float, float]] = []
    for item in raw:
        if (not isinstance(item, (list, tuple)) or len(item) != 2):
            raise MediaEditError(
                f"edit model returned a malformed range {item!r} — "
                "expected [start, end] pairs")
        try:
            s, e = float(item[0]), float(item[1])
        except (TypeError, ValueError):
            raise MediaEditError(
                f"edit model returned non-numeric range {item!r}")
        if not (0 <= s < e <= duration):
            raise MediaEditError(
                f"edit model returned out-of-bounds range [{s}, {e}] "
                f"(video is {duration:.1f}s) — refusing to cut")
        ranges.append((s, e))
    merged = _merge_ranges(ranges)
    if not merged:
        raise MediaEditError("edit model ranges collapsed to nothing")
    return merged


def edit_by_transcript(video: str | os.PathLike[str],
                       words: Sequence[Word | dict[str, Any]] | None,
                       instruction: str, *,
                       llm_fn: Callable[[str], str] | None = None,
                       out_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Map a natural-language edit instruction to time ranges via an LLM.

    ``llm_fn(prompt) -> str`` must return JSON ``{"keep": [[s, e], ...]}``.
    When ``llm_fn`` is None we raise — guessing with heuristics and
    calling it smart would be dishonest. Ranges are validated hard.
    """
    instruction = (instruction or "").strip()
    if not instruction:
        raise MediaEditError("edit_by_transcript needs an instruction")
    if llm_fn is None:
        raise MediaEditError(
            "edit_by_transcript needs llm_fn — a callable taking the prompt "
            "and returning JSON {\"keep\": [[start, end], ...]}. Refusing to "
            "guess the edit without it.")
    words = [w if isinstance(w, Word) else Word.from_dict(w)
             for w in (words or [])]
    if not words:
        raise MediaEditError("no transcript words — nothing to edit by")
    dur = _duration(video)
    lines = [f"[{w.start:7.2f} - {w.end:7.2f}] {w.text}" for w in words]
    prompt = (
        "You are a video editor. Below is a word-timed transcript of a "
        f"{dur:.1f}-second video. The user's edit instruction is:\n\n"
        f"INSTRUCTION: {instruction}\n\n"
        "Reply with ONLY a JSON object: {\"keep\": [[start, end], ...]} — "
        "the time ranges (seconds) to KEEP, in order. Ranges must satisfy "
        f"0 <= start < end <= {dur:.1f}. Everything not kept is cut.\n\n"
        "TRANSCRIPT:\n" + "\n".join(lines))
    raw = llm_fn(prompt)
    text = raw.strip()
    # tolerate ```json fences
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MediaEditError(
            f"edit model did not return JSON ({exc}); refusing to cut")
    keeps = _validate_ranges(data.get("keep") if isinstance(data, dict) else None,
                             dur)
    res = _assemble(video, keeps, out_dir=out_dir, suffix="edited")
    res["instruction"] = instruction
    return res


# ---------------------------------------------------------------------------
# topic keep ("keep only the part about X") — deterministic, no LLM
# ---------------------------------------------------------------------------

def keep_topic(video: str | os.PathLike[str],
               words: Sequence[Word | dict[str, Any]] | None,
               topic: str, *, context_s: float = 8.0,
               out_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Keep the parts mentioning ``topic`` (±``context_s`` seconds).

    Deterministic: fuzzy phrase search on the transcript, no LLM. Raises
    a clear error when the topic is never mentioned."""
    topic = (topic or "").strip()
    if not topic:
        raise MediaEditError("keep_topic needs a topic phrase")
    tt = TimestampedTranscript.from_words(words or [])
    if not tt.words:
        loaded = load_transcript(video)
        tt = TimestampedTranscript(words=loaded) if loaded else \
            TimestampedTranscript.transcribe(video)
    dur = _duration(video)
    hits = tt.ranges_for(topic)
    if not hits:
        raise MediaEditError(
            f"the topic {topic!r} was never mentioned in this video — "
            "nothing to keep")
    keeps = _pad(hits, context_s, 0.0, dur)
    res = _assemble(video, keeps, out_dir=out_dir, suffix="topic")
    res["topic"] = topic
    res["mentions"] = len(hits)
    return res


# ---------------------------------------------------------------------------
# NLE handoff XML (the Gling pattern — Devon doesn't replace pro editors)
# ---------------------------------------------------------------------------

def export_xml(words: Sequence[Word | dict[str, Any]],
               video_path: str | os.PathLike[str], *,
               fps: float = 30.0,
               out_path: str | os.PathLike[str] | None = None) -> Path:
    """Export an FCPXML timeline: the clip with transcript markers.

    Opens in Final Cut Pro / Premiere (via import) / Resolve. Words are
    grouped into sentence-ish markers so the timeline stays readable.
    """
    words = [w if isinstance(w, Word) else Word.from_dict(w) for w in words]
    if not words:
        raise MediaEditError("export_xml needs transcript words")
    src = Path(video_path)
    dur = _duration(video_path)
    out = Path(out_path) if out_path else src.with_suffix(".fcpxml")

    def tc(sec: float) -> str:
        # FCPXML time: rational frames/…s
        frames = int(round(sec * fps))
        return f"{frames}/{int(fps)}s"

    # group words into markers at sentence boundaries (or 6s chunks)
    markers: list[tuple[float, float, str]] = []
    cur: list[Word] = []
    for w in words:
        cur.append(w)
        if re.search(r"[.!?]$", w.text) or (w.end - cur[0].start) > 6:
            markers.append((cur[0].start, cur[-1].end,
                            " ".join(x.text for x in cur)))
            cur = []
    if cur:
        markers.append((cur[0].start, cur[-1].end,
                        " ".join(x.text for x in cur)))

    def esc(s: str) -> str:
        return (s.replace("&", "&amp;").replace("<", "&lt;")
                 .replace(">", "&gt;").replace('"', "&quot;"))

    marker_xml = "\n".join(
        f'        <marker start="{tc(s)}" duration="{tc(e - s)}" '
        f'value="{esc(t)}" completed="1"/>'
        for s, e, t in markers)
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<fcpxml version="1.9">
  <resources>
    <format id="r1" name="FFVideoFormat1080p30" frameDuration="1/30s"
            width="1920" height="1080"/>
    <asset id="r2" name="{esc(src.name)}" start="0s" duration="{tc(dur)}"
           hasVideo="1" hasAudio="1">
      <media-rep kind="original-media" src="file://{esc(str(src.resolve()))}"/>
    </asset>
  </resources>
  <library>
    <event name="transcript-edit">
      <project name="{esc(src.stem)}">
        <sequence format="r1" duration="{tc(dur)}">
          <spine>
            <clip name="{esc(src.name)}" start="0s" duration="{tc(dur)}">
              <video ref="r2"/>
{marker_xml}
            </clip>
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>
"""
    out.write_text(xml, encoding="utf-8")
    return out

"""Word-level caption timing for code videos — captions that follow
the spoken word.

The recheck (2026-11-18) proved there is NO WhisperX word-level
caption timing anywhere in videogen/motion_studio. The narration rung
(lane 79) gives the captions real speech to time against, and this
module times it:

* **measured** — the narration mix wav is transcribed by the repo's
  own faster-whisper word-timestamp seam
  (:func:`nomorals.media_edit.captions.transcribe_words` — the
  WhisperX pattern) and each script word is matched to a measured
  word, in order. The aligner model is ``base`` by default (the
  /models faster-whisper provision's base, ~140 MB) — ``large-v3-turbo``
  is the STT default for long-form transcription, far too heavy for
  aligning a few seconds of clean TTS; overridable with
  ``VIDEOGEN_CAPTION_MODEL``.
* **estimated** — when faster-whisper is not installed (or yields
  nothing), the words are split across the segment's own spoken
  window. The window itself is measured (the spoken audio length);
  only the within-window distribution is an estimate — and every
  such word is labelled ``"estimated"``, never ``"measured"``.
  A word the transcriber mis-hears is interpolated the same way and
  the record says ``"mixed"``.
* **missing** — no narration, no captions, with the reason named.

Provenance is per word, following the motion-studio §2.4 discipline
(:mod:`nomorals.media.motion_studio.typography`): "measured" is a
claim only a measurement earns; an estimate never gets it by default.
:func:`check_caption_timing` is the structural arbiter the QC rung
calls — words in order, no overlaps, inside their segment windows,
covering the script exactly.
"""

from __future__ import annotations

import os
from typing import Any

from ...core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "caption_narration",
    "check_caption_timing",
    "draw_captions",
    "captions_srt",
    "CAPTION_MODEL_DEFAULT",
]

#: The aligner model — the /models faster-whisper provision's base
#: (~140 MB). Clean TTS audio aligns well at base; large-v3-turbo
#: (~1.5 GB) is the STT default for long-form transcription and far
#: too heavy for a caption rung. Overridable per call and via env.
CAPTION_MODEL_DEFAULT = "base"

#: A measured word clamped this far outside its segment window is
#: still the same word (VAD/alignment drift); further out is not a
#: match for that word at all.
_MATCH_TOLERANCE_S = 0.35

#: Lookahead when matching script words to measured words in order —
#: a mis-heard word is skipped, never matched to a later word's time.
_MATCH_LOOKAHEAD = 5

#: Shortest window a word may carry (seconds) — below this the word
#: is unreadable on screen.
_MIN_WORD_S = 0.05

#: Overlap the structural check tolerates (rounding, seconds).
_OVERLAP_TOLERANCE_S = 0.001


def _norm(text: str) -> str:
    """A word as matchable text — lowercase, alphanumerics only."""
    return "".join(ch for ch in str(text or "").lower() if ch.isalnum())


def _script_words(text: str) -> list[str]:
    return [t for t in (text or "").split() if t]


def _default_transcriber(path: str, *, language: str = "en",
                         model: str = "") -> list[dict[str, Any]]:
    """Measured words via the repo's STT seam: :class:`STTTranscriber`
    (``media/edit_engine/text.py``) over an explicit
    ``FasterWhisperBackend``. Rung 1 transcribes the narration wav
    directly — the caption engine's ``transcribe_words`` expects a
    video file and refuses a wav ("no audio track"), while the
    narration audio is audio-only. Raises ImportError/RuntimeError
    when faster-whisper is not installed — the caller falls back
    honestly, never fakes words.
    """
    from ...voice.stt import FasterWhisperBackend
    from ..edit_engine.text import STTTranscriber

    backend = FasterWhisperBackend(model=model or CAPTION_MODEL_DEFAULT)
    return STTTranscriber(stt=backend,
                          language=language or "en").transcribe_words(path)


def _as_word(raw: Any) -> dict[str, Any] | None:
    """A transcriber word (dataclass or dict) as a plain dict."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        text = str(raw.get("text") or raw.get("word") or "").strip()
        if not text:
            return None
        try:
            return {"text": text, "start": float(raw["start"]),
                    "end": float(raw["end"])}
        except (KeyError, TypeError, ValueError):
            return None
    text = str(getattr(raw, "text", "") or "").strip()
    if not text:
        return None
    try:
        return {"text": text, "start": float(raw.start),
                "end": float(raw.end)}
    except (AttributeError, TypeError, ValueError):
        return None


def _estimate_segment(words: list[str], start: float,
                      end: float) -> list[dict[str, Any]]:
    """Words split across a spoken window, char-weighted.

    The window is measured (the spoken audio length); only the
    within-window distribution is an estimate — every word is
    labelled so.
    """
    if not words:
        return []
    span = max(end - start, _MIN_WORD_S * len(words))
    weights = [max(len(w), 1) for w in words]
    total = float(sum(weights))
    out: list[dict[str, Any]] = []
    t = float(start)
    for i, word in enumerate(words):
        share = span * weights[i] / total
        w_end = start + span if i == len(words) - 1 else t + share
        # full precision, never rounded: rounding a boundary to
        # 3 decimals can push it 0.0005s outside the spoken window
        # (a 5s/3-segment step is 1.66667s), and the validator
        # rightly flags that as outside — display code rounds.
        out.append({"text": word, "start": t,
                    "end": max(w_end, t + _MIN_WORD_S),
                    "timing_source": "estimated"})
        t = w_end
    return out


def _interpolate(words: list[dict[str, Any]], lo: float,
                 hi: float) -> list[dict[str, Any]]:
    """Estimated timings for words with no measurement, inside a gap.

    ``lo``/``hi`` are the neighbouring measured boundaries (or the
    segment window); the words split the gap char-weighted.
    """
    texts = [w["text"] for w in words]
    est = _estimate_segment(texts, lo, hi)
    for w, e in zip(words, est, strict=False):
        w["start"], w["end"] = e["start"], e["end"]
        w["timing_source"] = "estimated"
    return words


def caption_narration(narration: dict[str, Any], *,
                      transcriber: Any = None,
                      model: str = "",
                      language: str = "en") -> dict[str, Any]:
    """Time the narration's script word by word.

    ``narration`` is a :func:`synthesize_narration` result —
    ``path`` (the mix wav), ``duration``, ``segments`` (each ``text``
    + ``spoken_start``/``spoken_end``). ``transcriber`` is an
    injectable ``(path, *, language, model) -> [words]`` seam for
    tests; production uses the faster-whisper seam.

    Returns the caption record — ``status`` (``measured`` | ``mixed``
    | ``estimated`` | ``missing``), ``timing_source`` (same),
    ``words`` (each ``text``/``start``/``end``/``timing_source``/
    ``segment``), ``segments`` (the spoken windows), counts and the
    reason when nothing could be timed. A measured alignment that
    fails the structural check falls back to the estimate path with
    the reason named — a bad measurement never ships as captions.
    """
    segments = [s for s in (narration or {}).get("segments") or []
                if (s.get("text") or "").strip()]
    empty: dict[str, Any] = {"status": "missing", "timing_source": "",
                             "words": [], "segments": [],
                             "script": [], "words_total": 0,
                             "words_measured": 0, "words_estimated": 0,
                             "unmatched_measured": 0, "clamped": 0,
                             "model": "", "reason": ""}
    if not segments:
        return {**empty, "reason": ("no narration segments — the scene "
                                    "carries no speakable text")}
    spoken = [{"text": str(s.get("text") or ""),
               "spoken_start": float(s.get("spoken_start", 0.0)),
               "spoken_end": float(s.get("spoken_end", 0.0))}
              for s in segments]
    script = [s["text"] for s in spoken]
    script_words: list[tuple[int, str]] = []
    for i, s in enumerate(spoken):
        for w in _script_words(s["text"]):
            script_words.append((i, w))
    if not script_words:
        return {**empty, "reason": ("no narration segments — the scene "
                                    "carries no speakable text")}
    audio = str((narration or {}).get("path") or "")
    model_id = model or os.environ.get("VIDEOGEN_CAPTION_MODEL", "") \
        or CAPTION_MODEL_DEFAULT
    measured: list[dict[str, Any]] = []
    reason = ""
    # an injected transcriber is called even without a real audio file
    # (tests inject a fake); the production seam needs the real wav.
    attempt = transcriber is not None \
        or (audio and os.path.isfile(audio))
    if attempt:
        fn = transcriber or _default_transcriber
        try:
            raw = fn(audio, language=language, model=model_id)
            measured = [w for w in (_as_word(r) for r in (raw or []))
                        if w is not None]
            measured.sort(key=lambda w: w["start"])
            if not measured:
                reason = "transcriber yielded no word timings"
        except Exception as exc:  # noqa: BLE001 - fallback is honest
            reason = (f"no measured word timings — {exc}")
            _log.warning("caption alignment fell back to estimate: %s",
                         exc)
            measured = []
    else:
        reason = "no narration audio to align against"
    if measured:
        words, unmatched, clamped = _match(script_words, spoken, measured)
        status = ("measured" if all(
            w["timing_source"] == "measured" for w in words)
            else "mixed")
        record = {"status": status, "timing_source": status,
                  "words": words, "segments": spoken, "script": script,
                  "words_total": len(words),
                  "words_measured": sum(
                      1 for w in words
                      if w["timing_source"] == "measured"),
                  "words_estimated": sum(
                      1 for w in words
                      if w["timing_source"] == "estimated"),
                  "unmatched_measured": unmatched, "clamped": clamped,
                  "model": model_id, "reason": reason}
        ok, problems = check_caption_timing(
            record, duration=float((narration or {}).get("duration") or 0.0)
            or (spoken[-1]["spoken_end"] + 0.25))
        if ok:
            return record
        _log.warning("measured caption alignment failed validation "
                     "(%s) — falling back to estimate", "; ".join(problems))
        reason = ("measured alignment failed validation: "
                  + "; ".join(problems))
    # estimate path — no transcriber, no audio, or a failed measurement
    words = []
    for i, s in enumerate(spoken):
        for w in _estimate_segment(_script_words(s["text"]),
                                   s["spoken_start"], s["spoken_end"]):
            words.append({**w, "segment": i})
    return {"status": "estimated", "timing_source": "estimated",
            "words": words, "segments": spoken, "script": script,
            "words_total": len(words), "words_measured": 0,
            "words_estimated": len(words), "unmatched_measured": 0,
            "clamped": 0, "model": "",
            "reason": reason or ("faster-whisper is not installed on "
                                 "this host — captions are timed from "
                                 "the spoken segment windows, not "
                                 "measured per word (run /models to "
                                 "install faster-whisper)")}


def _match(script_words: list[tuple[int, str]],
           spoken: list[dict[str, Any]],
           measured: list[dict[str, Any]]
           ) -> tuple[list[dict[str, Any]], int, int]:
    """Match script words to measured words, in order.

    Returns (words, unmatched_measured, clamped). A script word with
    no measured match inside the lookahead is interpolated into the
    gap between its measured neighbours (or the segment window) and
    labelled estimated; a measured word matching no script word is
    skipped (a mis-transcription) and counted.
    """
    words: list[dict[str, Any]] = []
    j = 0
    unmatched = 0
    clamped = 0
    pending: list[dict[str, Any]] = []  # script words awaiting timing
    gap_start = 0.0  # the gap pending words fill starts here
    gap_seg = -1

    def flush(end: float) -> None:
        nonlocal pending
        if not pending:
            return
        _interpolate(pending, gap_start, end)
        words.extend(pending)
        pending = []

    for seg_idx, text in script_words:
        seg = spoken[seg_idx]
        if pending and seg_idx != gap_seg:
            # pending words belong to an earlier segment — fill the
            # rest of that segment's window after the last measured word
            flush(float(spoken[gap_seg]["spoken_end"]))
        want = _norm(text)
        hit: dict[str, Any] | None = None
        hit_k = -1
        for k in range(j, min(len(measured), j + _MATCH_LOOKAHEAD)):
            m = measured[k]
            if _norm(m["text"]) != want:
                continue
            if m["start"] < seg["spoken_start"] - _MATCH_TOLERANCE_S \
                    or m["end"] > seg["spoken_end"] + _MATCH_TOLERANCE_S:
                continue
            hit, hit_k = m, k
            break
        if hit is None:
            # gap starts after the last measured word in this
            # segment, or at the segment start when none placed
            if not pending and gap_seg != seg_idx:
                gap_start = float(seg["spoken_start"])
                gap_seg = seg_idx
            pending.append({"text": text, "start": 0.0, "end": 0.0,
                            "timing_source": "estimated",
                            "segment": seg_idx})
            continue
        # the measured words skipped over were mis-transcriptions
        unmatched += hit_k - j
        j = hit_k + 1
        if pending:
            flush(float(hit["start"]))
        start = max(hit["start"], seg["spoken_start"])
        end = min(hit["end"], seg["spoken_end"])
        if start != hit["start"] or end != hit["end"]:
            clamped += 1
        if end - start < _MIN_WORD_S:
            # clamped to nothing — interpolate it with its neighbours
            if not pending and gap_seg != seg_idx:
                gap_start = float(seg["spoken_start"])
                gap_seg = seg_idx
            pending.append({"text": text, "start": 0.0, "end": 0.0,
                            "timing_source": "estimated",
                            "segment": seg_idx})
            continue
        words.append({"text": text, "start": start,
                      "end": end, "timing_source": "measured",
                      "segment": seg_idx})
        gap_start = float(end)
        gap_seg = seg_idx
    if pending:
        flush(float(spoken[gap_seg]["spoken_end"]))
    words.sort(key=lambda w: (w["start"], w["end"]))
    unmatched += len(measured) - j
    return words, unmatched, clamped


def check_caption_timing(captions: dict[str, Any], *,
                         duration: float) -> tuple[bool, list[str]]:
    """(ok, reasons) — the structural arbiter for a caption record.

    Words must be in order with no overlaps, each inside its segment
    window and the video duration, and the word sequence must cover
    the script exactly. Pure — renders nothing, measures nothing.
    """
    reasons: list[str] = []
    words = list((captions or {}).get("words") or [])
    if not words:
        return False, ["no caption words"]
    segments = list((captions or {}).get("segments") or [])
    script = list((captions or {}).get("script") or [])
    if segments and script:
        want = [_norm(w) for s in script for w in _script_words(s)]
        got = [_norm(w.get("text")) for w in words]
        if got != want:
            return False, ["coverage: caption words do not match the "
                           "narration script in order"]
    prev_end = 0.0
    for i, w in enumerate(words):
        start = float(w.get("start", 0.0))
        end = float(w.get("end", 0.0))
        if not (w.get("text") or "").strip():
            reasons.append(f"word {i}: empty text")
        if start >= end:
            reasons.append(f"word {i} {w.get('text')!r}: start "
                           f"{start:.3f} >= end {end:.3f}")
        if start < -1e-6:
            reasons.append(f"word {i} {w.get('text')!r}: starts before 0")
        if end > float(duration) + 0.01:
            reasons.append(f"word {i} {w.get('text')!r}: ends after the "
                           f"video ({end:.3f} > {duration:.3f})")
        if start < prev_end - _OVERLAP_TOLERANCE_S:
            reasons.append(f"word {i} {w.get('text')!r}: overlaps the "
                           f"previous word")
        seg_idx = w.get("segment")
        if isinstance(seg_idx, int) and 0 <= seg_idx < len(segments):
            seg = segments[seg_idx]
            if start < float(seg["spoken_start"]) - 1e-6 \
                    or end > float(seg["spoken_end"]) + 1e-6:
                reasons.append(f"word {i} {w.get('text')!r}: outside "
                               f"its segment window")
        prev_end = max(prev_end, end)
    return (not reasons), reasons


# ---------------------------------------------------------------------------
# the overlay — words burned into the frame at time t
# ---------------------------------------------------------------------------

#: caption ink (motion-studio INK) and the current-word accent.
_CAPTION_INK = (242, 244, 250)
_CAPTION_ACCENT = (0, 240, 255)
_CAPTION_BOX = (10, 10, 14)


def _layout_lines(words: list[dict[str, Any]], width: int,
                  font: Any, draw: Any) -> list[list[dict[str, Any]]]:
    """Words grouped into centred lines, ≤92% of the frame width."""
    max_w = width * 0.92
    lines: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    cur_w = 0.0
    space_w = draw.textlength(" ", font=font)
    for word in words:
        ww = draw.textlength(word["text"], font=font)
        add = ww if not cur else ww + space_w
        if cur and cur_w + add > max_w:
            lines.append(cur)
            cur, cur_w = [word], ww
        else:
            cur.append(word)
            cur_w += add
    if cur:
        lines.append(cur)
    return lines


def draw_captions(img: Any, words: list[dict[str, Any]], t: float
                  ) -> Any:
    """Burn the active caption line into a PIL frame at time ``t``.

    The line holding the word spoken at ``t`` is drawn at the bottom
    (inside the safe area); words already spoken read in ink, the
    word being spoken reads in accent, unspoken words are not drawn
    yet. No word active at ``t`` → the frame is untouched. Returns
    the same image (drawn in place).
    """
    from PIL import ImageDraw

    from ..motion_studio._core import find_font

    if not words:
        return img
    active = [w for w in words
              if float(w["start"]) <= t < float(w["end"])]
    if not active:
        return img
    width, height = img.size
    m = min(width, height)
    font = find_font(max(14, int(m * 0.058)), "DejaVuSans-Bold.ttf")
    draw = ImageDraw.Draw(img)
    lines = _layout_lines(words, width, font, draw)
    line = next((ln for ln in lines if active[0] in ln), lines[0])
    _, top, _, bottom = draw.textbbox((0, 0), "Ag", font=font)
    line_h = int((bottom - top) * 1.5)
    y = height - line_h - int(height * 0.035)
    space_w = draw.textlength(" ", font=font)
    widths = [draw.textlength(w["text"], font=font) for w in line]
    total_w = sum(widths) + space_w * (len(line) - 1)
    x = (width - total_w) / 2
    draw.rectangle([x - 12, y - 6, x + total_w + 12, y + line_h + 6],
                   fill=_CAPTION_BOX)
    for w, ww in zip(line, widths, strict=False):
        if t >= float(w["start"]):
            current = float(w["start"]) <= t < float(w["end"])
            draw.text((x, y), w["text"], font=font,
                      fill=_CAPTION_ACCENT if current else _CAPTION_INK)
        x += ww + space_w
    return img


# ---------------------------------------------------------------------------
# the sidecar — an SRT beside the MP4, the Gling pattern
# ---------------------------------------------------------------------------

def _srt_ts(t: float) -> str:
    total_ms = max(0, int(round(float(t) * 1000)))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def captions_srt(words: list[dict[str, Any]], *, max_words: int = 6,
                 max_span: float = 2.0) -> str:
    """Timed words as an SRT document — cue per phrase.

    Groups consecutive words into cues (≤ ``max_words`` or
    ``max_span`` seconds); an empty word list yields an empty string.
    """
    cues: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    for w in words or []:
        if cur and (len(cur) >= max_words
                    or float(w["end"]) - float(cur[0]["start"]) > max_span):
            cues.append(cur)
            cur = []
        cur.append(w)
    if cur:
        cues.append(cur)
    out: list[str] = []
    for i, cue in enumerate(cues, start=1):
        out.append(str(i))
        out.append(f"{_srt_ts(cue[0]['start'])} --> "
                   f"{_srt_ts(cue[-1]['end'])}")
        out.append(" ".join(w["text"] for w in cue))
        out.append("")
    return "\n".join(out)

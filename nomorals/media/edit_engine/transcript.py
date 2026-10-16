"""Transcript → EditSpec compiler.

This is the ONE transcript-edit path: a transcript edit compiles to an
:class:`EditSpec` (clips + explicit segment plan) and renders through the
edit engine. :mod:`nomorals.media_edit.transcript_edit` keeps its
trim/concat helpers, but new transcript edits go through here — no
parallel engine, one declarative plan from words to render.

Pure: :func:`keeps_for` computes keep-ranges from words + ops with no
I/O; :func:`compile_transcript_edit` turns those ranges into an
:class:`EditSpec`. Range math is reused from
:mod:`nomorals.media_edit.transcript_edit` (filler detection, merge,
complement, pad) so the two surfaces can never disagree about what
"cut the umms" means.

Ops (applied in list order, each a dict)::

    {"op": "remove_fillers", "pad": 0.2}
    {"op": "remove_silences", "spans": [[s, e], ...], "pad": 0.15}
    {"op": "keep_topic", "topic": "pricing", "context_s": 8.0}
    {"op": "keep_ranges", "ranges": [[s, e], ...]}
    {"op": "cut_ranges", "ranges": [[s, e], ...]}

Invariant: the compiled spec's segments cover exactly the keep-ranges —
same count, same bounds, same total duration — and every range lies
inside ``[0, duration]``.
"""

from __future__ import annotations

from typing import Any, Sequence

from ...media_edit.captions import Word
from ...media_edit.transcript_edit import (
    TimestampedTranscript,
    _complement,
    _merge_ranges,
    _pad,
    find_fillers,
)
from ...media_edit.videos import MediaEditError
from .spec import EditSpec
from .timeline import Clip

__all__ = ["keeps_for", "compile_transcript_edit"]


def _words(words: Sequence[Word | dict[str, Any]]) -> list[Word]:
    return [w if isinstance(w, Word) else Word.from_dict(w)
            for w in (words or [])]


def _subtract(keeps: list[tuple[float, float]],
              cuts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """``keeps`` minus ``cuts`` (cuts may span several keeps)."""
    out: list[tuple[float, float]] = []
    for s, e in keeps:
        out.extend(_complement(cuts, s, e))
    return _merge_ranges(out)


def _intersect(keeps: list[tuple[float, float]],
               ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """``keeps`` ∩ ``ranges``."""
    out: list[tuple[float, float]] = []
    for s, e in keeps:
        for rs, re_ in ranges:
            lo, hi = max(s, rs), min(e, re_)
            if hi - lo > 1e-3:
                out.append((lo, hi))
    return _merge_ranges(out)


def _validate_ranges(ranges: Any, duration: float, *, field: str) -> list[tuple[float, float]]:
    if not isinstance(ranges, (list, tuple)):
        raise MediaEditError(f"{field} must be a list of [start, end] pairs")
    out: list[tuple[float, float]] = []
    for item in ranges:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise MediaEditError(
                f"{field} has a malformed range {item!r} — "
                "expected [start, end] pairs")
        try:
            s, e = float(item[0]), float(item[1])
        except (TypeError, ValueError):
            raise MediaEditError(f"{field} has a non-numeric range {item!r}")
        if not (0 <= s < e <= duration):
            raise MediaEditError(
                f"{field} range [{s}, {e}] outside [0, {duration}] — "
                "refusing to compile")
        out.append((s, e))
    return _merge_ranges(out)


def keeps_for(words: Sequence[Word | dict[str, Any]], *,
              duration: float,
              ops: Sequence[dict[str, Any]] | None = None) -> list[tuple[float, float]]:
    """Pure: transcript words + ops → sorted, non-overlapping keep-ranges.

    Starts from the whole ``[0, duration]`` span and applies each op in
    order. Raises :class:`MediaEditError` when the ops would keep
    nothing (cutting everything is never a valid edit).
    """
    duration = float(duration)
    if duration <= 0:
        raise MediaEditError("compile needs a positive duration")
    ws = _words(words)
    keeps: list[tuple[float, float]] = [(0.0, duration)]
    for op in (ops or []):
        if not isinstance(op, dict):
            raise MediaEditError(f"transcript op must be a dict, got {op!r}")
        kind = str(op.get("op", "")).strip()
        if kind == "remove_fillers":
            pad = float(op.get("pad", 0.2))
            cuts = _pad(find_fillers(ws), pad, 0.0, duration)
            keeps = _subtract(keeps, cuts)
        elif kind == "remove_silences":
            pad = float(op.get("pad", 0.15))
            spans = _validate_ranges(op.get("spans") or [], duration,
                                     field="remove_silences.spans")
            keeps = _subtract(keeps, _pad(spans, pad, 0.0, duration))
        elif kind == "keep_topic":
            topic = str(op.get("topic", "")).strip()
            if not topic:
                raise MediaEditError("keep_topic needs a topic phrase")
            context_s = float(op.get("context_s", 8.0))
            tt = TimestampedTranscript.from_words(ws)
            hits = tt.ranges_for(topic)
            if not hits:
                raise MediaEditError(
                    f"the topic {topic!r} was never mentioned — "
                    "nothing to keep")
            keeps = _intersect(keeps, _pad(hits, context_s, 0.0, duration))
        elif kind == "keep_ranges":
            ranges = _validate_ranges(op.get("ranges") or [], duration,
                                      field="keep_ranges.ranges")
            keeps = _intersect(keeps, ranges)
        elif kind == "cut_ranges":
            ranges = _validate_ranges(op.get("ranges") or [], duration,
                                      field="cut_ranges.ranges")
            keeps = _subtract(keeps, ranges)
        else:
            raise MediaEditError(
                f"unknown transcript op {kind!r} — use: remove_fillers, "
                "remove_silences, keep_topic, keep_ranges, cut_ranges")
        if not keeps:
            raise MediaEditError(
                f"transcript op {kind!r} kept nothing — refusing to "
                "compile an empty edit")
    return keeps


def compile_transcript_edit(source: str, words: Sequence[Word | dict[str, Any]], *,
                            duration: float,
                            ops: Sequence[dict[str, Any]] | None = None,
                            fps: int = 30, width: int = 1080,
                            height: int = 1920, seed: int = 0,
                            out: str | None = None) -> EditSpec:
    """Compile a transcript edit to an :class:`EditSpec`.

    One clip per keep-range (``Clip(path=source, start=s, end=e)``) and
    an explicit segment plan ``[(i, e - s), …]`` — the engine renders
    exactly the keeps, in order, with no re-encoding guesswork.
    """
    if not (source or "").strip():
        raise MediaEditError("compile_transcript_edit needs a source clip")
    keeps = keeps_for(words, duration=duration, ops=ops)
    clips = [Clip(path=str(source), start=s, end=e) for s, e in keeps]
    segments = [(i, round(e - s, 6)) for i, (s, e) in enumerate(keeps)]
    return EditSpec(clips=clips, segments=segments, fps=fps,
                    width=width, height=height, seed=seed, out=out)

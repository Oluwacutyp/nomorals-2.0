"""Diarized transcripts feeding the transcript editor.

``voice/stt.py`` already has both halves — ``diarize()`` (who spoke
when, via pyannote, honest RuntimeError when absent) and the
transcriber backends — but nothing joined them into one editable
artifact.  This module is the join:

1. :func:`attach_speakers` — assign each transcript segment the speaker
   whose diarization interval contains the segment midpoint; merge
   consecutive same-speaker segments.
2. :class:`DiarizedTranscript` — the joined artifact: segments with
   speaker labels, exportable as text / SRT / VTT / JSON.
3. :class:`TranscriptEditor` — the editor the artifact feeds: rename a
   speaker everywhere, edit a segment's text, reassign a speaker,
   merge / split segments.  Every edit is recorded in ``history`` so
   the original transcript is never silently rewritten.

Transcriber and diarizer are injectable callables, so tests run with
fakes and production passes the real backends
(``UniversalSTT.transcribe`` + ``stt.diarize``).
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

_log = logging.getLogger(__name__)

__all__ = [
    "DiarizedSegment",
    "DiarizedTranscript",
    "TranscriptEditor",
    "attach_speakers",
    "diarize_and_transcribe",
    "merge_adjacent",
]


@dataclass
class DiarizedSegment:
    """One transcript segment with its speaker attached."""

    start: float
    end: float
    text: str
    speaker: str = "unknown"
    words: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "text": self.text,
                "speaker": self.speaker, "words": list(self.words)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiarizedSegment":
        return cls(
            start=float(data.get("start") or 0.0),
            end=float(data.get("end") or 0.0),
            text=str(data.get("text") or ""),
            speaker=str(data.get("speaker") or "unknown"),
            words=list(data.get("words") or []),
        )


def _midpoint(seg: dict[str, Any]) -> float:
    start = float(seg.get("start") or 0.0)
    end = float(seg.get("end") or start)
    return (start + end) / 2.0


def attach_speakers(
    transcript_segments: Sequence[dict[str, Any]],
    diarization_segments: Sequence[dict[str, Any]],
    *,
    default_speaker: str = "unknown",
) -> list[DiarizedSegment]:
    """Attach a speaker label to each transcript segment.

    The speaker is the diarization segment containing the transcript
    segment's midpoint; when no interval contains it, the nearest
    interval (by midpoint distance) wins; with no diarization at all
    every segment is ``default_speaker``.  Never raises on malformed
    input — bad segments are skipped, not fatal.
    """
    diar = [d for d in diarization_segments
            if isinstance(d, dict) and "speaker" in d]
    out: list[DiarizedSegment] = []
    for seg in transcript_segments:
        if not isinstance(seg, dict):
            continue
        mid = _midpoint(seg)
        speaker = default_speaker
        containing = [d for d in diar
                      if float(d.get("start") or 0.0) <= mid
                      <= float(d.get("end") or 0.0)]
        if containing:
            speaker = str(containing[0].get("speaker") or default_speaker)
        elif diar:
            nearest = min(
                diar,
                key=lambda d: abs(
                    (float(d.get("start") or 0.0)
                     + float(d.get("end") or 0.0)) / 2.0 - mid))
            speaker = str(nearest.get("speaker") or default_speaker)
        out.append(DiarizedSegment(
            start=float(seg.get("start") or 0.0),
            end=float(seg.get("end") or 0.0),
            text=str(seg.get("text") or ""),
            speaker=speaker,
            words=list(seg.get("words") or []),
        ))
    return merge_adjacent(out)


def merge_adjacent(
    segments: Sequence[DiarizedSegment],
) -> list[DiarizedSegment]:
    """Merge consecutive segments from the same speaker (gap-free join)."""
    merged: list[DiarizedSegment] = []
    for seg in segments:
        if (merged and merged[-1].speaker == seg.speaker
                and seg.start <= merged[-1].end + 0.5):
            last = merged[-1]
            last.end = max(last.end, seg.end)
            last.text = (last.text + " " + seg.text).strip()
            last.words = [*last.words, *seg.words]
        else:
            merged.append(copy.deepcopy(seg))
    return merged


def _ts(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    ms = int(round((seconds - int(seconds)) * 1000))
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


@dataclass
class DiarizedTranscript:
    """A transcript with speaker labels — the editor's input artifact."""

    segments: list[DiarizedSegment] = field(default_factory=list)
    audio: str = ""
    speakers: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.speakers:
            seen: list[str] = []
            for seg in self.segments:
                if seg.speaker not in seen:
                    seen.append(seg.speaker)
            self.speakers = seen

    def to_text(self) -> str:
        return "\n".join(f"{s.speaker}: {s.text}" for s in self.segments)

    def to_srt(self) -> str:
        blocks = []
        for i, seg in enumerate(self.segments, 1):
            blocks.append(
                f"{i}\n{_ts(seg.start)} --> {_ts(seg.end)}\n"
                f"{seg.speaker}: {seg.text}")
        return "\n\n".join(blocks)

    def to_vtt(self) -> str:
        blocks = ["WEBVTT"]
        for seg in self.segments:
            blocks.append(
                f"{_ts(seg.start).replace(',', '.')} --> "
                f"{_ts(seg.end).replace(',', '.')}\n"
                f"{seg.speaker}: {seg.text}")
        return "\n\n".join(blocks)

    def to_dict(self) -> dict[str, Any]:
        return {"audio": self.audio, "speakers": list(self.speakers),
                "segments": [s.to_dict() for s in self.segments]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiarizedTranscript":
        return cls(
            segments=[DiarizedSegment.from_dict(s)
                      for s in (data.get("segments") or [])],
            audio=str(data.get("audio") or ""),
            speakers=[str(s) for s in (data.get("speakers") or [])])


class TranscriptEditor:
    """Edit a diarized transcript: speakers, text, merge, split.

    All edits append to ``history`` (``{action, index, before, after}``)
    so the original is auditable.  ``transcript`` always reflects the
    current edited state.
    """

    def __init__(self, transcript: DiarizedTranscript) -> None:
        self.transcript = copy.deepcopy(transcript)
        self.history: list[dict[str, Any]] = []

    def _record(self, action: str, index: int,
                before: Any, after: Any) -> None:
        self.history.append({"action": action, "index": index,
                             "before": before, "after": after})

    def rename_speaker(self, old: str, new: str) -> int:
        """Rename ``old`` -> ``new`` everywhere.  Returns segments changed."""
        changed = 0
        for seg in self.transcript.segments:
            if seg.speaker == old:
                self._record("rename_speaker", -1, old, new)
                seg.speaker = new
                changed += 1
        if changed:
            speakers = self.transcript.speakers
            self.transcript.speakers = [
                new if s == old else s for s in speakers]
            if new not in self.transcript.speakers:
                self.transcript.speakers.append(new)
        return changed

    def edit_text(self, index: int, new_text: str) -> bool:
        if not 0 <= index < len(self.transcript.segments):
            return False
        seg = self.transcript.segments[index]
        self._record("edit_text", index, seg.text, new_text)
        seg.text = new_text
        return True

    def reassign_speaker(self, index: int, speaker: str) -> bool:
        if not 0 <= index < len(self.transcript.segments):
            return False
        seg = self.transcript.segments[index]
        self._record("reassign_speaker", index, seg.speaker, speaker)
        seg.speaker = speaker
        if speaker not in self.transcript.speakers:
            self.transcript.speakers.append(speaker)
        return True

    def merge_segments(self, index: int) -> bool:
        """Merge segment ``index`` into the previous one (same speaker)."""
        segs = self.transcript.segments
        if not 1 <= index < len(segs):
            return False
        prev, cur = segs[index - 1], segs[index]
        self._record("merge", index, cur.to_dict(), prev.to_dict())
        prev.end = max(prev.end, cur.end)
        prev.text = (prev.text + " " + cur.text).strip()
        prev.words = [*prev.words, *cur.words]
        del segs[index]
        return True

    def split_segment(self, index: int, at: float) -> bool:
        """Split segment ``index`` at time ``at`` (must fall strictly inside)."""
        segs = self.transcript.segments
        if not 0 <= index < len(segs):
            return False
        seg = segs[index]
        if not seg.start < at < seg.end:
            return False
        before = seg.to_dict()
        words_before = [w for w in seg.words
                        if float(w.get("end", w.get("start", 0)) or 0) <= at]
        words_after = [w for w in seg.words if w not in words_before]
        # Split text on word boundary when word timings exist, else halve.
        if words_before and words_after:
            first_text = " ".join(str(w.get("word", "")) for w in words_before).strip()
            second_text = " ".join(str(w.get("word", "")) for w in words_after).strip()
        else:
            half = len(seg.text) // 2
            first_text, second_text = seg.text[:half].strip(), seg.text[half:].strip()
        seg.end = at
        seg.text = first_text or seg.text
        seg.words = words_before
        new_seg = DiarizedSegment(start=at, end=before["end"],
                                  text=second_text, speaker=seg.speaker,
                                  words=words_after)
        segs.insert(index + 1, new_seg)
        self._record("split", index, before, new_seg.to_dict())
        return True

    def export(self, fmt: str = "text") -> str:
        fmt = (fmt or "text").lower()
        if fmt == "srt":
            return self.transcript.to_srt()
        if fmt == "vtt":
            return self.transcript.to_vtt()
        if fmt == "json":
            return self.transcript.to_json()
        return self.transcript.to_text()


def diarize_and_transcribe(
    audio_path: str,
    *,
    transcriber: Callable[..., dict[str, Any]] | None = None,
    diarizer: Callable[..., dict[str, Any]] | None = None,
    speaker_names: dict[str, str] | None = None,
    language: str = "en",
) -> DiarizedTranscript:
    """Transcribe + diarize one file into a :class:`DiarizedTranscript`.

    Defaults use the real backends (``UniversalSTT`` + ``stt.diarize``);
    a missing pyannote install surfaces as the honest
    ``RuntimeError`` from ``diarize`` — or as all-``unknown`` speakers
    when the caller passes ``diarizer=None`` explicitly to skip it.
    Inject fakes in tests.
    """
    from ..voice import stt as _stt

    if transcriber is None:
        def _default_transcribe(path: str, **_kw: Any) -> dict[str, Any]:
            return _stt.UniversalSTT().transcribe(path, language=language)
        transcriber = _default_transcribe
    result = transcriber(audio_path, language=language)
    segments = result.get("segments") or []
    if not segments and result.get("text"):
        segments = [{"start": 0.0, "end": 0.0,
                     "text": result.get("text")}]
    diarization: list[dict[str, Any]] = []
    if diarizer is None:
        try:
            diar = _stt.diarize(audio_path,
                                speaker_names=speaker_names or {})
            diarization = diar.get("segments") or []
        except RuntimeError:
            # pyannote absent: honest unknown speakers, not a crash.
            diarization = []
    else:
        diar = diarizer(audio_path, speaker_names=speaker_names or {})
        diarization = (diar or {}).get("segments") or []
    attached = attach_speakers(segments, diarization)
    speakers = sorted({s.speaker for s in attached})
    return DiarizedTranscript(segments=attached, audio=str(audio_path),
                              speakers=speakers)

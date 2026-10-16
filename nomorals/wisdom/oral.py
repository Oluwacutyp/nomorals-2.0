"""Oral-tradition-native ingest — recordings → transcribe → diarize →
align → provenance → index. The moat.

Pipeline stages, with the human in the loop where it matters:

1. ``submit_draft`` — a transcript (or a recording reference + a
   transcriber callable) becomes a DRAFT with provenance + permission.
   Restriction is checked FIRST: initiated / do-not-digitize material
   without a permission record raises RestrictedOralError and STOPS.
2. ``verify`` — the OWNER verifies the draft (correct speakers, correct
   words, permission confirmed). Nothing ingests unverified.
3. ``ingest`` — verified drafts are written to the store as JSON with
   full provenance, ready for the corpus indexer.

Honesty rules:
- No transcriber supplied → :class:`MissingTranscriberError`, never a
  fabricated transcript. Transcription is an injectable callable
  (``audio_ref -> transcript text``), so the owner's STT engine plugs in.
- Diarization is a real heuristic over ``SPEAKER: text`` turns in the
  supplied transcript (plus a word-level alignment that evenly splits a
  turn across its duration, labeled approximate).
- Permission + restriction are schema fields on every draft; ingest
  re-checks them (a draft verified without permission still refuses).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .traditions import RestrictedCorpusError, check_ingest

__all__ = [
    "OralError", "MissingTranscriberError", "PermissionRecord",
    "Turn", "DraftTranscript", "OralIngest", "split_turns",
    "align_words",
]

_SPEAKER_LINE = re.compile(r"^([A-Za-z0-9 _-]{1,40}):\s*(.+)$")


class OralError(ValueError):
    """Oral ingest failure."""


class MissingTranscriberError(OralError):
    """No transcriber supplied — transcription is the owner's STT
    engine, not something this module fakes."""


@dataclass
class PermissionRecord:
    """Permission for oral material: who granted it, for what scope."""

    granted_by: str  # custodian / elder / community contact
    scope: str  # what the permission covers, in plain words
    date: str = ""
    contact_ref: str = ""  # pointer, never a raw phone/email in the record
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Turn:
    """One diarized speaker turn."""

    speaker: str
    text: str
    start_s: float = 0.0
    end_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def split_turns(raw_transcript: str) -> list[Turn]:
    """Diarize a transcript: ``SPEAKER: text`` lines → turns.

    Lines without a speaker prefix continue the previous turn; text
    with no speaker markers at all becomes a single ``unknown`` turn.
    """
    turns: list[Turn] = []
    for line in (raw_transcript or "").splitlines():
        line = line.strip()
        if not line:
            continue
        m = _SPEAKER_LINE.match(line)
        if m:
            turns.append(Turn(speaker=m.group(1).strip(),
                              text=m.group(2).strip()))
        elif turns:
            turns[-1].text += " " + line
        else:
            turns.append(Turn(speaker="unknown", text=line))
    return turns


def align_words(turns: list[Turn], *,
                total_duration_s: float = 0.0) -> list[dict[str, Any]]:
    """Word-level alignment (APPROXIMATE, labeled as such).

    Without a forced-aligner, the honest alignment is an even split of
    each turn across its duration (or across the total duration when
    per-turn timings are absent). Consumers must treat ``approximate:
    True`` as 'do not use for verbatim timing claims'.
    """
    out: list[dict[str, Any]] = []
    n_turns = len(turns)
    for i, turn in enumerate(turns):
        words = turn.text.split()
        if not words:
            continue
        if turn.end_s > turn.start_s:
            start, end = turn.start_s, turn.end_s
        elif total_duration_s > 0:
            start = total_duration_s * i / max(1, n_turns)
            end = total_duration_s * (i + 1) / max(1, n_turns)
        else:
            start, end = 0.0, 0.0
        span = end - start
        for j, word in enumerate(words):
            w_start = start + span * j / len(words)
            w_end = start + span * (j + 1) / len(words)
            out.append({"word": word, "speaker": turn.speaker,
                        "start_s": round(w_start, 3),
                        "end_s": round(w_end, 3),
                        "approximate": True})
    return out


@dataclass
class DraftTranscript:
    """One oral draft: transcript + provenance + verification state."""

    draft_id: str
    audio_ref: str  # pointer to the recording (path/URL), not the audio
    language: str
    tradition_pack_id: str
    restriction: str = "open"
    turns: list[Turn] = field(default_factory=list)
    permission: PermissionRecord | None = None
    status: str = "draft"  # draft | verified | ingested | rejected
    provenance: dict[str, Any] = field(default_factory=dict)
    verified_by: str = ""
    verify_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "draft_id": self.draft_id, "audio_ref": self.audio_ref,
            "language": self.language,
            "tradition_pack_id": self.tradition_pack_id,
            "restriction": self.restriction,
            "turns": [t.to_dict() for t in self.turns],
            "permission": (self.permission.to_dict()
                           if self.permission else None),
            "status": self.status, "provenance": self.provenance,
            "verified_by": self.verified_by,
            "verify_notes": self.verify_notes,
            "text": " ".join(t.text for t in self.turns),
        }


class OralIngest:
    """Draft → owner verifies → ingest. Restriction checked twice
    (submit + ingest); either check can STOP the pipeline."""

    def __init__(self, store_dir: Path | str) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._drafts: dict[str, DraftTranscript] = {}

    def transcribe(self, audio_ref: str, *,
                   transcriber: Callable[[str], str] | None = None) -> str:
        """Transcribe via the caller's STT engine. No transcriber →
        MissingTranscriberError (never a fabricated transcript)."""
        if transcriber is None:
            raise MissingTranscriberError(
                "transcribe() needs a transcriber callable "
                "(audio_ref -> transcript text) — the owner's STT "
                "engine plugs in here; this module never fakes audio")
        if not audio_ref:
            raise OralError("transcribe needs an audio_ref")
        return transcriber(audio_ref)

    def submit_draft(self, draft_id: str, *, audio_ref: str = "",
                     transcript: str = "", language: str = "",
                     tradition_pack_id: str = "",
                     restriction: str = "open",
                     permission: PermissionRecord | None = None,
                     contributor: str = "") -> DraftTranscript:
        """Create a DRAFT. Restriction gate runs FIRST (STOP on
        initiated/do-not-digitize/community-permission without a
        permission record)."""
        if not draft_id:
            raise OralError("submit_draft needs a draft_id")
        if draft_id in self._drafts:
            raise OralError(f"draft {draft_id!r} already exists")
        if restriction not in ("open", "public-scholarship",
                               "community-permission", "initiated",
                               "do-not-digitize"):
            raise OralError(f"unknown restriction {restriction!r}")
        if restriction in ("initiated", "do-not-digitize",
                           "community-permission") and permission is None:
            raise RestrictedCorpusError(
                f"STOP — oral draft {draft_id!r} is restricted "
                f"({restriction}); a permission record from the "
                "tradition's custodians is required before even a "
                "draft is accepted.")
        turns = split_turns(transcript)
        draft = DraftTranscript(
            draft_id=draft_id, audio_ref=audio_ref, language=language,
            tradition_pack_id=tradition_pack_id, restriction=restriction,
            turns=turns, permission=permission, status="draft",
            provenance={"contributor": contributor,
                        "restriction": restriction,
                        "permission_ref": (
                            permission.contact_ref if permission else ""),
                        "source": "oral recording (draft transcript)"})
        self._drafts[draft_id] = draft
        return draft

    def verify(self, draft_id: str, *, owner: str = "",
               approve: bool = True, notes: str = "") -> DraftTranscript:
        """Owner verification: the draft is correct + permission is real.

        ``approve=False`` rejects the draft (status=rejected, never
        ingestible afterwards without a fresh submit).
        """
        draft = self._get(draft_id)
        if draft.status != "draft":
            raise OralError(
                f"draft {draft_id!r} is {draft.status}, not verifiable")
        draft.verified_by = owner
        draft.verify_notes = notes
        draft.status = "verified" if approve else "rejected"
        return draft

    def ingest(self, draft_id: str) -> dict[str, Any]:
        """Ingest a VERIFIED draft: re-checks restriction + permission,
        then writes the draft + alignment to the store as JSON."""
        draft = self._get(draft_id)
        if draft.status != "verified":
            raise OralError(
                f"draft {draft_id!r} is {draft.status} — only verified "
                "drafts ingest (draft → owner verifies → ingest)")
        if draft.restriction in ("initiated", "do-not-digitize",
                                 "community-permission") \
                and draft.permission is None:
            raise RestrictedCorpusError(
                f"STOP — draft {draft_id!r} lost its permission record; "
                "ingest refused.")
        # Honest use of the traditions gate: the pack text descriptor is
        # the draft itself; check_ingest enforces the same STOP rule.
        try:
            from .traditions import PackText, pack_by_id
            pack = pack_by_id(draft.tradition_pack_id) \
                if draft.tradition_pack_id else None
            if pack is not None:
                check_ingest(
                    pack, PackText(
                        title=f"oral draft {draft_id}",
                        restriction=draft.restriction),
                    permission_ref=(draft.permission.contact_ref
                                    or draft.permission.granted_by
                                    or "permission-record-on-file"))
        except RestrictedCorpusError:
            raise
        except Exception:  # noqa: BLE001 - unknown pack ≠ ingest failure
            pass
        payload = draft.to_dict()
        payload["alignment"] = align_words(draft.turns)
        path = self.store_dir / f"{draft_id}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(path)
        draft.status = "ingested"
        return {"ok": True, "draft_id": draft_id, "path": str(path),
                "turns": len(draft.turns),
                "words": len(payload["text"].split()),
                "provenance": draft.provenance}

    def get(self, draft_id: str) -> DraftTranscript:
        return self._get(draft_id)

    def list_drafts(self, *, status: str = "") -> list[dict[str, Any]]:
        out = [d.to_dict() for d in self._drafts.values()
               if not status or d.status == status]
        return out

    def _get(self, draft_id: str) -> DraftTranscript:
        try:
            return self._drafts[draft_id]
        except KeyError:
            raise OralError(f"unknown draft {draft_id!r}") from None

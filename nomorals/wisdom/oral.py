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
    "align_words", "search_ingested", "parse_oral_flags",
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


def search_ingested(store_dir: Path | str, query: str, *,
                    top: int = 5) -> list[dict[str, Any]]:
    """Keyword search over ingested oral records on disk.

    Reads the store's ``*.json`` files (ingest wrote them; a file

    Reads the store's ``*.json`` files (ingest wrote them; a file
    exists only for an ingested draft). A turn scores by how many
    distinct query terms (2+ chars, lowercased) its text contains;
    zero-score turns are dropped. Ranked by score desc, then draft
    id, then turn order. No embeddings, no invented relevance.
    """
    terms = {t for t in re.findall(r"[a-z0-9\u00c0-\uffff]{2,}",
                                   (query or "").lower())}
    if not terms:
        return []
    hits: list[dict[str, Any]] = []
    store = Path(store_dir)
    if not store.is_dir():
        return []
    for path in sorted(store.glob("*.json")):
        if path.name.endswith(".draft.json"):
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(raw, dict) or "draft_id" not in raw:
            continue
        prov = raw.get("provenance", {}) or {}
        for turn in raw.get("turns", []):
            text = str(turn.get("text", ""))
            low = text.lower()
            score = sum(1 for t in terms if t in low)
            if not score:
                continue
            hits.append({
                "draft_id": str(raw.get("draft_id", path.stem)),
                "speaker": str(turn.get("speaker", "unknown")),
                "text": text,
                "language": str(raw.get("language", "")),
                "tradition_pack_id": str(raw.get("tradition_pack_id", "")),
                "contributor": str(prov.get("contributor", "")),
                "permission_ref": str(prov.get("permission_ref", "")),
                "score": score,
            })
    hits.sort(key=lambda h: (-h["score"], h["draft_id"]))
    return hits[:max(0, top)]


#: ``--key`` flags the oral surfaces accept (CLI + chat share this).
ORAL_FLAG_KEYS = {
    "file", "language", "lang", "pack", "restriction", "granted-by",
    "scope", "contact-ref", "audio-ref", "contributor", "notes",
    "status",
}
#: Boolean ``--flag`` forms (no value follows).
ORAL_BOOL_FLAGS = {"reject"}


def parse_oral_flags(words: list[str]) -> tuple[dict[str, Any], list[str]]:
    """Parse ``--key value`` / ``--flag`` pairs out of a word list.

    Returns (flags, positionals). An unknown ``--key`` raises
    ValueError with the key named — a mistyped flag is never silently
    dropped. Shared by the CLI and chat surfaces so both parse the
    same way.
    """
    flags: dict[str, Any] = {}
    pos: list[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        if w.startswith("--"):
            key = w[2:]
            if key in ORAL_BOOL_FLAGS:
                flags[key] = True
                i += 1
            elif key not in ORAL_FLAG_KEYS:
                raise ValueError(f"unknown oral flag --{key}")
            elif i + 1 < len(words) and not words[i + 1].startswith("--"):
                flags[key] = words[i + 1]
                i += 2
            else:
                flags[key] = ""
                i += 1
        else:
            pos.append(w)
            i += 1
    return flags, pos


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
    (submit + ingest); either check can STOP the pipeline.

    Drafts are DURABLE: ``submit_draft`` / ``verify`` write the draft
    to ``<store>/<draft_id>.draft.json`` (atomic tmp+replace), so a
    draft submitted by one process (``nm wisdom oral submit``) can be
    verified and ingested by a later one. ``ingest`` writes the final
    ``<draft_id>.json`` and removes the draft file. Files in the store
    exist only because ingest wrote them, so a stored JSON is an
    ingested record even though its recorded status is "verified"
    (the status is snapshotted before the ingest flips it).
    """

    def __init__(self, store_dir: Path | str) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._drafts: dict[str, DraftTranscript] = {}

    def _draft_path(self, draft_id: str) -> Path:
        return self.store_dir / f"{draft_id}.draft.json"

    def _save_draft(self, draft: DraftTranscript) -> None:
        path = self._draft_path(draft.draft_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(draft.to_dict(), indent=2,
                                  ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def _load_draft(self, draft_id: str) -> DraftTranscript | None:
        path = self._draft_path(draft_id)
        if not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        try:
            turns = [Turn(speaker=str(t.get("speaker", "unknown")),
                          text=str(t.get("text", "")),
                          start_s=float(t.get("start_s", 0.0)),
                          end_s=float(t.get("end_s", 0.0)))
                     for t in raw.get("turns", [])]
            perm = raw.get("permission")
            permission = (PermissionRecord(
                granted_by=str(perm.get("granted_by", "")),
                scope=str(perm.get("scope", "")),
                date=str(perm.get("date", "")),
                contact_ref=str(perm.get("contact_ref", "")),
                notes=str(perm.get("notes", ""))) if perm else None)
            draft = DraftTranscript(
                draft_id=str(raw.get("draft_id", draft_id)),
                audio_ref=str(raw.get("audio_ref", "")),
                language=str(raw.get("language", "")),
                tradition_pack_id=str(raw.get("tradition_pack_id", "")),
                restriction=str(raw.get("restriction", "open")),
                turns=turns, permission=permission,
                status=str(raw.get("status", "draft")),
                provenance=dict(raw.get("provenance", {})),
                verified_by=str(raw.get("verified_by", "")),
                verify_notes=str(raw.get("verify_notes", "")))
        except (AttributeError, TypeError, ValueError):
            return None
        return draft

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
        self._save_draft(draft)  # durable across processes
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
        self._save_draft(draft)  # durable across processes
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
        self._draft_path(draft_id).unlink(missing_ok=True)
        return {"ok": True, "draft_id": draft_id, "path": str(path),
                "turns": len(draft.turns),
                "words": len(payload["text"].split()),
                "provenance": draft.provenance}

    def get(self, draft_id: str) -> DraftTranscript:
        return self._get(draft_id)

    def list_drafts(self, *, status: str = "") -> list[dict[str, Any]]:
        """All drafts: in-memory + durable ``.draft.json`` files.

        In-memory wins on id collision (same draft, newer state).
        """
        merged: dict[str, dict[str, Any]] = {}
        for path in sorted(self.store_dir.glob("*.draft.json")):
            loaded = self._load_draft(path.name[:-len(".draft.json")])
            if loaded is not None:
                merged[loaded.draft_id] = loaded.to_dict()
        for d in self._drafts.values():
            merged[d.draft_id] = d.to_dict()
        out = [d for d in merged.values()
               if not status or d["status"] == status]
        return sorted(out, key=lambda d: d["draft_id"])

    def list_ingested(self) -> list[dict[str, Any]]:
        """Durable listing of ingested records (the store's ``*.json``).

        A file exists in the store only because ingest wrote it, so
        every entry here is an ingested record; the recorded status
        is "verified" (snapshotted before ingest flipped it — see the
        class docstring).
        """
        out: list[dict[str, Any]] = []
        for path in sorted(self.store_dir.glob("*.json")):
            if path.name.endswith(".draft.json"):
                continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(raw, dict) or "draft_id" not in raw:
                continue
            turns = raw.get("turns", [])
            out.append({
                "draft_id": str(raw.get("draft_id", path.stem)),
                "status": str(raw.get("status", "verified")),
                "language": str(raw.get("language", "")),
                "tradition_pack_id": str(raw.get("tradition_pack_id", "")),
                "restriction": str(raw.get("restriction", "open")),
                "turns": len(turns),
                "words": len(str(raw.get("text", "")).split()),
                "provenance": dict(raw.get("provenance", {})),
                "path": str(path),
            })
        return out

    def search(self, query: str, *, top: int = 5) -> list[dict[str, Any]]:
        """Keyword search over INGESTED oral records on disk.

        Honest keyword matching: a turn scores by how many distinct
        query terms (2+ chars) it contains; turns scoring zero are
        dropped. No embeddings, no invented relevance — the score is
        the term count.
        """
        return search_ingested(self.store_dir, query, top=top)

    def read_ingested(self, draft_id: str) -> dict[str, Any]:
        """The full ingested record for one draft (or OralError)."""
        path = self.store_dir / f"{draft_id}.json"
        if not path.is_file():
            raise OralError(f"no ingested oral draft {draft_id!r}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise OralError(
                f"ingested oral draft {draft_id!r} is unreadable: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise OralError(f"ingested oral draft {draft_id!r} is malformed")
        return raw

    def _get(self, draft_id: str) -> DraftTranscript:
        try:
            return self._drafts[draft_id]
        except KeyError:
            pass
        loaded = self._load_draft(draft_id)
        if loaded is None:
            raise OralError(f"unknown draft {draft_id!r}") from None
        self._drafts[draft_id] = loaded
        return loaded

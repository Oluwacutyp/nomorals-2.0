"""Personal marks: the owner's own saved places (docs 19+21 M9).

A *mark* is one spot the owner stood in and named: he sends
``/map mark <label>`` from whichever platform he's on, Devon
captures the location of that moment (the command message's own
location, or the location pin he drops next into the chat an
armed *pending mark* is waiting in), and the spot is stored under
his label for later — list it, view it on the basemap, route back
to it, ask Devon where it is.

Capture is explicit intent, always: a location share NEVER
creates a mark on its own.  A mark exists only because an armed
``/map mark`` completed — immediately from the command message's
own location meta, or by the pending flow when the owner's next
location lands in the armed chat.  Marking an existing label
MOVES the mark (same identity, new coordinates, new time) — a
personal tool, not a refusal form.

Storage follows the M7 overlay store's idiom
(:mod:`nomorals.maps.overlays`): when the context carries the
shared db, marks land in runtime-created ``map_marks`` /
``map_pending_marks`` tables (migrations are frozen); with no db
they live in ``<home>/maps/marks.json``, written atomically
(tmp + replace).  Reads never raise; a corrupt store degrades to
empty + a named reason.

Pending marks are durable for the same reason: the arm and the
completing pin arrive as separate messages, and the runtime may
restart between them — an in-memory pending would silently lose
the owner's intent.  Pendings live in the same store, keyed by
``platform|chat_key|sender`` so exactly the identity that armed
one can complete it, one at a time per key, with a ten-minute
TTL enforced lazily (P3 precedent — no timers): an expired
pending is dropped when next observed, and after expiry a
location share marks nothing.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.failures import record_failure
from ..core.logging_setup import get_logger
from .coords import validate_lat, validate_lon

_log = get_logger(__name__)

__all__ = [
    "MAX_LABEL_CHARS", "METHOD_MESSAGE", "METHOD_PENDING",
    "PENDING_TTL_S", "Mark", "MarkError", "MarkStore", "PendingMark",
    "normalize_label", "pending_key",
]

#: Capture methods (the Mark.method vocabulary).
METHOD_MESSAGE = "message_location"
METHOD_PENDING = "pending_completion"
_METHODS = (METHOD_MESSAGE, METHOD_PENDING)

#: How long an armed pending mark waits for the owner's location
#: pin.  Long enough to switch apps and drop a pin; short enough
#: that a forgotten arm never completes against a place he has
#: long since left.
PENDING_TTL_S = 600.0

#: A label is a personal name for a place ("hostel", "mama's
#: shop"); generous for real names, fatal for payloads.
MAX_LABEL_CHARS = 80

_DDL_MARKS = (
    "CREATE TABLE IF NOT EXISTS map_marks ("
    "mark_id TEXT PRIMARY KEY, name TEXT, payload TEXT)"
)
_DDL_PENDING = (
    "CREATE TABLE IF NOT EXISTS map_pending_marks ("
    "pending_key TEXT PRIMARY KEY, payload TEXT)"
)


class MarkError(ValueError):
    """A named mark defect: invalid label, bad coordinates."""


def normalize_label(label: Any) -> str:
    """The canonical label: whitespace collapsed, length capped.

    Labels are the owner's own words; casefolding for identity
    happens at lookup, the label itself is kept as written.
    Raises :class:`MarkError` for an empty or over-long label —
    a named refusal, never a silent truncation (two labels
    truncated to the same text would silently collide).
    """
    text = " ".join(str(label or "").split())
    if not text:
        raise MarkError("a mark needs a label — /map mark <label>")
    if len(text) > MAX_LABEL_CHARS:
        raise MarkError(
            f"mark label too long ({len(text)} > {MAX_LABEL_CHARS} "
            "characters) — give the place a shorter name")
    return text


def _label_key(label: str) -> str:
    return label.casefold()


def _coerce_accuracy(value: Any) -> float | None:
    """Accuracy is platform garnish: keep a sane non-negative
    number, drop anything else rather than refuse the mark."""
    if value is None:
        return None
    try:
        acc = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(acc) or acc < 0:
        return None
    return acc


@dataclass(frozen=True)
class Mark:
    """One saved place: where the owner stood, under his label."""

    mark_id: str
    label: str
    lat: float
    lon: float
    created_at: float
    updated_at: float
    platform: str = ""
    method: str = METHOD_MESSAGE
    accuracy: float | None = None
    place_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mark_id": self.mark_id,
            "label": self.label,
            "lat": self.lat,
            "lon": self.lon,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "platform": self.platform,
            "method": self.method,
            "accuracy": self.accuracy,
            "place_hint": self.place_hint,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Mark | None:
        """Parse a stored payload; malformed rows read as absent
        (a corrupt payload must never crash a read)."""
        if not isinstance(raw, dict):
            return None
        try:
            label = normalize_label(raw.get("label"))
            lat = validate_lat(raw.get("lat"))
            lon = validate_lon(raw.get("lon"))
            created = float(raw.get("created_at") or 0.0)
            updated = float(raw.get("updated_at") or created)
        except (MarkError, TypeError, ValueError):
            return None
        method = str(raw.get("method") or METHOD_MESSAGE)
        if method not in _METHODS:
            method = METHOD_MESSAGE
        return cls(
            mark_id=str(raw.get("mark_id") or ""),
            label=label, lat=lat, lon=lon,
            created_at=created, updated_at=updated,
            platform=str(raw.get("platform") or ""),
            method=method,
            accuracy=_coerce_accuracy(raw.get("accuracy")),
            place_hint=str(raw.get("place_hint") or ""),
        )


@dataclass(frozen=True)
class PendingMark:
    """An armed intent: this identity's next location in this
    chat becomes the mark named ``label``."""

    key: str
    label: str
    platform: str = ""
    chat_key: str = ""
    sender_id: str = ""
    armed_at: float = 0.0

    @property
    def expires_at(self) -> float:
        return self.armed_at + PENDING_TTL_S

    def expired(self, now: float) -> bool:
        return now - self.armed_at > PENDING_TTL_S

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "platform": self.platform,
            "chat_key": self.chat_key,
            "sender_id": self.sender_id,
            "armed_at": self.armed_at,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> PendingMark | None:
        if not isinstance(raw, dict):
            return None
        try:
            label = normalize_label(raw.get("label"))
            armed = float(raw.get("armed_at") or 0.0)
        except (MarkError, TypeError, ValueError):
            return None
        key = str(raw.get("key") or "")
        if not key or armed <= 0.0:
            # An unparseable arm time would make the pending
            # immortal; drop it instead (never immortal).
            return None
        return cls(
            key=key, label=label,
            platform=str(raw.get("platform") or ""),
            chat_key=str(raw.get("chat_key") or ""),
            sender_id=str(raw.get("sender_id") or ""),
            armed_at=armed,
        )


def pending_key(platform: str, chat_key: str, sender_id: str) -> str:
    """The pending identity: platform + chat + arming sender.

    ``chat_key`` already embeds the platform on shipped surfaces;
    the explicit platform segment keeps keys unambiguous when a
    bare chat id is supplied.
    """
    return "|".join((str(platform or ""), str(chat_key or ""),
                     str(sender_id or "")))


class MarkStore:
    """Where marks and pending marks live (see module docstring).

    ``db`` is the shared application db handle (the runtime
    context's); ``home`` is the Devon home directory for the
    file fallback.  Either may be empty — an unattached store
    reads empty and refuses writes with a named reason.
    """

    def __init__(self, *, db: Any = None,
                 home: str | Path = "") -> None:
        self._db = db
        self._tables_ready = False
        self._path = (Path(str(home)) / "maps" / "marks.json"
                      if db is None and home else None)
        self._degraded = ""

    # ── posture ─────────────────────────────────────────────────

    def degraded_reason(self) -> str:
        """Why the store is degraded ("" when healthy)."""
        if self._degraded:
            return self._degraded
        if self._db is None and self._path is None:
            return "no mark store attached (no db, no home)"
        return ""

    # ── storage backends (the overlay store's shapes) ───────────

    def _ensure_tables(self) -> None:
        if self._tables_ready:
            return
        self._db.execute(_DDL_MARKS)
        self._db.execute(_DDL_PENDING)
        self._tables_ready = True

    def _load_file(self) -> dict[str, Any]:
        if self._path is None or not self._path.exists():
            return {}
        try:
            data = json.loads(
                self._path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
            self._degraded = "mark store is not a JSON object"
            return {}
        except Exception as exc:  # noqa: BLE001 — reads fail open
            record_failure("maps.marks_load_file", exc)
            self._degraded = f"mark store unreadable: {exc}"
            return {}

    def _save_file(self, data: dict[str, Any]) -> bool:
        if self._path is None:
            return False
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False,
                                      indent=1), encoding="utf-8")
            tmp.replace(self._path)
            return True
        except Exception as exc:  # noqa: BLE001
            record_failure("maps.marks_save_file", exc)
            return False

    def _load_sections(self) -> tuple[dict[str, Mark],
                                      dict[str, PendingMark]]:
        """Both sections from whichever backing is attached.
        Never raises: defects degrade to empty."""
        marks: dict[str, Mark] = {}
        pendings: dict[str, PendingMark] = {}
        if self._db is not None:
            try:
                self._ensure_tables()
                for row in self._db.query(
                        "SELECT payload FROM map_marks", ()):
                    mark = Mark.from_dict(
                        json.loads(row.get("payload") or "{}"))
                    if mark is not None and mark.mark_id:
                        marks[mark.mark_id] = mark
                for row in self._db.query(
                        "SELECT payload FROM map_pending_marks", ()):
                    pend = PendingMark.from_dict(
                        json.loads(row.get("payload") or "{}"))
                    if pend is not None:
                        pendings[pend.key] = pend
            except Exception as exc:  # noqa: BLE001 - reads fail open
                record_failure("maps.marks_read", exc)
                self._degraded = f"mark store db unreadable: {exc}"
                return {}, {}
            return marks, pendings
        data = self._load_file()
        raw_marks = data.get("marks")
        if isinstance(raw_marks, dict):
            for payload in raw_marks.values():
                mark = Mark.from_dict(payload)
                if mark is not None and mark.mark_id:
                    marks[mark.mark_id] = mark
        raw_pending = data.get("pending")
        if isinstance(raw_pending, dict):
            for payload in raw_pending.values():
                pend = PendingMark.from_dict(payload)
                if pend is not None:
                    pendings[pend.key] = pend
        return marks, pendings

    def _write_file_sections(self, marks: dict[str, Mark],
                             pendings: dict[str, PendingMark]) -> bool:
        return self._save_file({
            "marks": {mid: m.to_dict() for mid, m in marks.items()},
            "pending": {k: p.to_dict() for k, p in pendings.items()},
        })

    def _persist_mark(self, mark: Mark) -> bool:
        if self._db is not None:
            try:
                self._ensure_tables()
                self._db.execute(
                    "INSERT OR REPLACE INTO map_marks "
                    "(mark_id, name, payload) VALUES (?, ?, ?)",
                    (mark.mark_id, mark.label,
                     json.dumps(mark.to_dict(), ensure_ascii=False)))
                return True
            except Exception as exc:  # noqa: BLE001
                record_failure("maps.marks_persist", exc)
                return False
        marks, pendings = self._load_sections()
        marks[mark.mark_id] = mark
        return self._write_file_sections(marks, pendings)

    def _remove_mark(self, mark_id: str) -> bool:
        if self._db is not None:
            try:
                self._ensure_tables()
                self._db.execute(
                    "DELETE FROM map_marks WHERE mark_id = ?",
                    (mark_id,))
                return True
            except Exception as exc:  # noqa: BLE001
                record_failure("maps.marks_remove", exc)
                return False
        marks, pendings = self._load_sections()
        marks.pop(mark_id, None)
        return self._write_file_sections(marks, pendings)

    def _persist_pending(self, pend: PendingMark) -> bool:
        if self._db is not None:
            try:
                self._ensure_tables()
                self._db.execute(
                    "INSERT OR REPLACE INTO map_pending_marks "
                    "(pending_key, payload) VALUES (?, ?)",
                    (pend.key,
                     json.dumps(pend.to_dict(), ensure_ascii=False)))
                return True
            except Exception as exc:  # noqa: BLE001
                record_failure("maps.marks_persist_pending", exc)
                return False
        marks, pendings = self._load_sections()
        pendings[pend.key] = pend
        return self._write_file_sections(marks, pendings)

    def _remove_pending(self, key: str) -> bool:
        if self._db is not None:
            try:
                self._ensure_tables()
                self._db.execute(
                    "DELETE FROM map_pending_marks "
                    "WHERE pending_key = ?", (key,))
                return True
            except Exception as exc:  # noqa: BLE001
                record_failure("maps.marks_remove_pending", exc)
                return False
        marks, pendings = self._load_sections()
        pendings.pop(key, None)
        return self._write_file_sections(marks, pendings)

    # ── marks ───────────────────────────────────────────────────

    def save(self, label: Any, lat: Any, lon: Any, *,
             platform: str = "", method: str = METHOD_MESSAGE,
             accuracy: Any = None, place_hint: str = "",
             now: float | None = None) -> tuple[Mark, bool]:
        """Save a mark; an existing label MOVES.

        Returns ``(mark, moved)`` — ``moved`` is True when an
        existing mark with this label (case-insensitive) was
        relocated: same mark id and creation time, new
        coordinates/time/platform/method.  Raises
        :class:`MarkError` on invalid input or a failed write.
        """
        text = normalize_label(label)
        try:
            flat = validate_lat(lat)
            flon = validate_lon(lon)
        except ValueError as exc:
            raise MarkError(str(exc)) from exc
        if method not in _METHODS:
            raise MarkError(f"unknown capture method {method!r}")
        stamp = time.time() if now is None else float(now)
        marks, _pendings = self._load_sections()
        key = _label_key(text)
        existing = next(
            (m for m in marks.values() if _label_key(m.label) == key),
            None)
        if existing is not None:
            mark = Mark(
                mark_id=existing.mark_id, label=existing.label,
                lat=flat, lon=flon,
                created_at=existing.created_at, updated_at=stamp,
                platform=str(platform or ""), method=method,
                accuracy=_coerce_accuracy(accuracy),
                place_hint=str(place_hint or ""))
            moved = True
        else:
            mark = Mark(
                mark_id="mk-" + uuid.uuid4().hex[:12], label=text,
                lat=flat, lon=flon,
                created_at=stamp, updated_at=stamp,
                platform=str(platform or ""), method=method,
                accuracy=_coerce_accuracy(accuracy),
                place_hint=str(place_hint or ""))
            moved = False
        if not self._persist_mark(mark):
            raise MarkError(
                "mark " + repr(mark.label) + " could not be stored"
                + (f" ({self.degraded_reason()})"
                   if self.degraded_reason() else ""))
        _log.info("mark saved label=%s moved=%s method=%s",
                  mark.label, moved, method)
        return mark, moved

    def get(self, label: Any) -> Mark | None:
        """One mark by label (case-insensitive).  Never raises."""
        try:
            key = _label_key(normalize_label(label))
        except MarkError:
            return None
        marks, _ = self._load_sections()
        exact = next(
            (m for m in marks.values() if m.label == str(label)), None)
        if exact is not None:
            return exact
        return next(
            (m for m in marks.values() if _label_key(m.label) == key),
            None)

    def list(self) -> tuple[Mark, ...]:
        """All marks, sorted by label.  Never raises."""
        marks, _ = self._load_sections()
        return tuple(sorted(
            marks.values(),
            key=lambda m: (m.label.casefold(), m.mark_id)))

    def delete(self, label: Any) -> bool:
        """Remove a mark by label; True when one was removed."""
        mark = self.get(label)
        if mark is None:
            return False
        return self._remove_mark(mark.mark_id)

    def set_hint(self, mark_id: str, hint: str) -> bool:
        """Fill a mark's place hint in place (the lazy gazetteer
        fill).  Garnish: never raises, False on any failure."""
        hint = str(hint or "").strip()
        if not mark_id or not hint:
            return False
        marks, _ = self._load_sections()
        mark = marks.get(mark_id)
        if mark is None or mark.place_hint == hint:
            return False
        updated = Mark(
            mark_id=mark.mark_id, label=mark.label, lat=mark.lat,
            lon=mark.lon, created_at=mark.created_at,
            updated_at=mark.updated_at, platform=mark.platform,
            method=mark.method, accuracy=mark.accuracy,
            place_hint=hint)
        try:
            return self._persist_mark(updated)
        except Exception as exc:  # noqa: BLE001 - garnish path
            record_failure("maps.marks.set_hint", exc)
            return False

    # ── pending marks ───────────────────────────────────────────

    def arm(self, key: str, label: Any, *, platform: str = "",
            chat_key: str = "", sender_id: str = "",
            now: float | None = None) -> PendingMark:
        """Arm (or re-arm) the pending mark at ``key`` — one
        pending per key; arming again replaces it."""
        text = normalize_label(label)
        if not str(key or ""):
            raise MarkError("a pending mark needs a chat identity")
        pend = PendingMark(
            key=str(key), label=text, platform=str(platform or ""),
            chat_key=str(chat_key or ""), sender_id=str(sender_id or ""),
            armed_at=time.time() if now is None else float(now))
        if not self._persist_pending(pend):
            raise MarkError(
                "pending mark for " + repr(text)
                + " could not be stored")
        return pend

    def pending(self, key: str, *,
                now: float | None = None) -> PendingMark | None:
        """The live pending at ``key``; an expired one is dropped
        (lazily) and reads as absent.  Never raises."""
        stamp = time.time() if now is None else float(now)
        _, pendings = self._load_sections()
        pend = pendings.get(str(key or ""))
        if pend is None:
            return None
        if pend.expired(stamp):
            self._remove_pending(pend.key)
            return None
        return pend

    def pendings(self, *, now: float | None = None) -> tuple[PendingMark, ...]:
        """All live pendings (expired filtered, not deleted —
        :meth:`lapsed` is the sweeper).  Never raises."""
        stamp = time.time() if now is None else float(now)
        _, pendings = self._load_sections()
        live = [p for p in pendings.values() if not p.expired(stamp)]
        return tuple(sorted(live, key=lambda p: (p.label.casefold(),
                                                 p.key)))

    def lapsed(self, *, now: float | None = None) -> tuple[PendingMark, ...]:
        """Drop every expired pending; returns what lapsed so a
        surface can say so honestly.  Never raises."""
        stamp = time.time() if now is None else float(now)
        _, pendings = self._load_sections()
        gone = tuple(sorted(
            (p for p in pendings.values() if p.expired(stamp)),
            key=lambda p: (p.label.casefold(), p.key)))
        for pend in gone:
            self._remove_pending(pend.key)
        return gone

    def clear(self, key: str) -> PendingMark | None:
        """Remove the pending at ``key`` (after completion or an
        explicit cancel); returns what was removed."""
        _, pendings = self._load_sections()
        pend = pendings.get(str(key or ""))
        if pend is None:
            return None
        self._remove_pending(pend.key)
        return pend

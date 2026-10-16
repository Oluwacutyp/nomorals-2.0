"""SQLite identity spine: one durable record per person.

Replaces the JSON-backed ``nomorals.social.relationships`` store as the
system of record for *who* people are, without touching that module:
``migrate.py`` copies its data here, and this store then owns it.

Design rules (same as the tracker it supersedes):

- Scores are derived, not assigned. ``closeness`` is recomputed from
  interaction counters with the tracker's exact formula — frequency +
  recency + owner initiation — capped at 1.0. The owner's actions are
  the ground truth, not labels.
- Never delete history. There is no delete method anywhere in this
  module; corrections append (a new note, a new link, a new obligation
  row). The only dedupe is exact-duplicate suppression (same note text,
  same link triple, same commitment id), which keeps re-runs and
  re-migration from duplicating history.
- Stdlib only. Sibling-relative imports only.

Schema
------
``persons`` — one row per person, keyed by ``person_id``.
``notes`` — append-only mined facts, one row per note.
``links`` — person <-> person links (``kind`` is free-form: e.g.
``"friend"``, ``"colleague"``, ``"family"``, ``"introduced-by"``).
``gifts`` — gifts given/received, keyed off the same ``person_id``.
``obligations`` — commitments owed to / involving a person, keyed off
the same ``person_id`` (``commitment_id`` ties to the commitments
ledger when one exists; ``None`` means a standalone obligation).
"""

from __future__ import annotations

import math
import sqlite3
import time
from pathlib import Path
from typing import Any, Literal

GiftDirection = Literal["given", "received"]

_PERSON_COLUMNS = (
    "person_id",
    "display_name",
    "closeness",
    "warmth",
    "inbound_count",
    "outbound_count",
    "owner_initiated_count",
    "first_seen",
    "last_inbound",
    "last_outbound",
    "last_topic",
    "created_at",
    "updated_at",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    person_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL DEFAULT '',
    closeness REAL NOT NULL DEFAULT 0.0,
    warmth REAL NOT NULL DEFAULT 0.5,
    inbound_count INTEGER NOT NULL DEFAULT 0,
    outbound_count INTEGER NOT NULL DEFAULT 0,
    owner_initiated_count INTEGER NOT NULL DEFAULT 0,
    first_seen REAL NOT NULL,
    last_inbound REAL NOT NULL DEFAULT 0.0,
    last_outbound REAL NOT NULL DEFAULT 0.0,
    last_topic TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL REFERENCES persons(person_id),
    note TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notes_person ON notes(person_id);
CREATE TABLE IF NOT EXISTS links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_person TEXT NOT NULL REFERENCES persons(person_id),
    to_person TEXT NOT NULL REFERENCES persons(person_id),
    kind TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_links_from ON links(from_person);
CREATE INDEX IF NOT EXISTS idx_links_to ON links(to_person);
CREATE TABLE IF NOT EXISTS gifts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL REFERENCES persons(person_id),
    description TEXT NOT NULL DEFAULT '',
    direction TEXT NOT NULL CHECK (direction IN ('given', 'received')),
    value_minor INTEGER,
    currency TEXT NOT NULL DEFAULT '',
    date TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_gifts_person ON gifts(person_id);
CREATE TABLE IF NOT EXISTS obligations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL REFERENCES persons(person_id),
    commitment_id TEXT,
    what TEXT NOT NULL DEFAULT '',
    deadline TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'open',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_obligations_person ON obligations(person_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_obligations_commitment
    ON obligations(person_id, commitment_id)
    WHERE commitment_id IS NOT NULL;
"""


def _recompute_closeness(
    *,
    inbound_count: int,
    outbound_count: int,
    owner_initiated_count: int,
    first_seen: float,
    last_inbound: float,
    last_outbound: float,
    now: float,
) -> float:
    """The tracker's exact formula: frequency + recency + owner initiation.

    Kept byte-for-byte in shape with
    ``RelationshipTracker._recompute_closeness`` so migrated scores and
    live scores agree: log-scaled frequency (caps ~100 messages), recency
    decaying over 90 days of silence, owner initiation dominating.
    """
    total = inbound_count + outbound_count
    freq = min(1.0, math.log10(1 + total) / 2.0)
    last = max(last_inbound, last_outbound, first_seen)
    days_quiet = (now - last) / 86400
    recency = max(0.0, 1.0 - days_quiet / 90.0)
    initiated = min(1.0, owner_initiated_count / 5.0)
    return round(min(1.0, 0.30 * freq + 0.20 * recency + 0.50 * initiated), 3)


class PersonStore:
    """Per-person identity spine, SQLite-backed."""

    def __init__(self, path: str | Path | None = ":memory:") -> None:
        self.path = ":memory:" if path is None else str(path)
        if self.path != ":memory:":
            parent = Path(self.path).expanduser().parent
            parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ── lifecycle ────────────────────────────────────────────────────

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "PersonStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ── internal helpers ─────────────────────────────────────────────

    def _row(self, person_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM persons WHERE person_id = ?", (person_id,)
        ).fetchone()

    def _notes_for(self, person_id: str) -> list[str]:
        return [
            r["note"]
            for r in self._conn.execute(
                "SELECT note FROM notes WHERE person_id = ? ORDER BY id",
                (person_id,),
            )
        ]

    def _to_person(self, row: sqlite3.Row) -> dict[str, Any]:
        person = {col: row[col] for col in _PERSON_COLUMNS}
        person["notes"] = self._notes_for(row["person_id"])
        return person

    def _insert_person(self, person_id: str, display_name: str, now: float) -> None:
        self._conn.execute(
            "INSERT INTO persons (person_id, display_name, first_seen,"
            " created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (person_id, display_name, now, now, now),
        )

    # ── persons ──────────────────────────────────────────────────────

    def ensure(
        self, person_id: str, display_name: str = "", *, at: float | None = None
    ) -> dict[str, Any]:
        """Get-or-create a person. Never overwrites an existing name with
        blank, and never overwrites a set name — names are only filled
        when empty, mirroring the tracker."""
        now = time.time() if at is None else at
        row = self._row(person_id)
        if row is None:
            self._insert_person(person_id, display_name, now)
            self._conn.commit()
            return self.get(person_id)  # type: ignore[return-value]
        if display_name and not row["display_name"]:
            self._conn.execute(
                "UPDATE persons SET display_name = ?, updated_at = ?"
                " WHERE person_id = ?",
                (display_name, now, person_id),
            )
            self._conn.commit()
        return self.get(person_id)  # type: ignore[return-value]

    def get(self, person_id: str) -> dict[str, Any] | None:
        """One person with their notes, or ``None`` if unknown."""
        row = self._row(person_id)
        if row is None:
            return None
        return self._to_person(row)

    def import_record(self, record: dict[str, Any]) -> bool:
        """Insert a full migrated record verbatim; skip if the person
        already exists (appending only notes not already present).

        Used by :func:`nomorals.people.migrate.migrate_from_relationships`
        so migration preserves the tracker's exact counters/scores
        instead of re-deriving them from replayed events. Returns
        ``True`` when a new person row was inserted.
        """
        person_id = record["person_id"]
        now = time.time()
        existing = self._row(person_id)
        if existing is None:
            self._conn.execute(
                "INSERT INTO persons (person_id, display_name, closeness,"
                " warmth, inbound_count, outbound_count,"
                " owner_initiated_count, first_seen, last_inbound,"
                " last_outbound, last_topic, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    person_id,
                    record.get("display_name", ""),
                    float(record.get("closeness", 0.0)),
                    float(record.get("warmth", 0.5)),
                    int(record.get("inbound_count", 0)),
                    int(record.get("outbound_count", 0)),
                    int(record.get("owner_initiated_count", 0)),
                    float(record.get("first_seen", now)),
                    float(record.get("last_inbound", 0.0)),
                    float(record.get("last_outbound", 0.0)),
                    record.get("last_topic", ""),
                    now,
                    now,
                ),
            )
            created = True
        else:
            created = False
        for note in record.get("notes") or []:
            self._append_note(person_id, note, now)
        self._conn.commit()
        return created

    # ── interaction ──────────────────────────────────────────────────

    def record_inbound(
        self,
        person_id: str,
        *,
        display_name: str = "",
        warm: bool | None = None,
        topic: str = "",
        at: float | None = None,
    ) -> dict[str, Any]:
        """They messaged. Bumps closeness by recency-weighted frequency."""
        now = time.time() if at is None else at
        self.ensure(person_id, display_name, at=now)
        row = self._row(person_id)
        assert row is not None
        warmth = row["warmth"]
        if warm is not None:
            warmth = 0.8 * warmth + 0.2 * (1.0 if warm else 0.0)
        closeness = _recompute_closeness(
            inbound_count=row["inbound_count"] + 1,
            outbound_count=row["outbound_count"],
            owner_initiated_count=row["owner_initiated_count"],
            first_seen=row["first_seen"],
            last_inbound=now,
            last_outbound=row["last_outbound"],
            now=now,
        )
        self._conn.execute(
            "UPDATE persons SET inbound_count = inbound_count + 1,"
            " last_inbound = ?, last_topic = CASE WHEN ? != '' THEN ?"
            " ELSE last_topic END, warmth = ?, closeness = ?,"
            " updated_at = ? WHERE person_id = ?",
            (now, topic, topic, warmth, closeness, now, person_id),
        )
        self._conn.commit()
        return self.get(person_id)  # type: ignore[return-value]

    def record_outbound(
        self,
        person_id: str,
        *,
        display_name: str = "",
        owner_initiated: bool = False,
        at: float | None = None,
    ) -> dict[str, Any]:
        """We messaged them. Owner-initiated weighs heaviest — the
        owner's own behavior is the ground truth of who matters."""
        now = time.time() if at is None else at
        self.ensure(person_id, display_name, at=now)
        row = self._row(person_id)
        assert row is not None
        closeness = _recompute_closeness(
            inbound_count=row["inbound_count"],
            outbound_count=row["outbound_count"] + 1,
            owner_initiated_count=row["owner_initiated_count"]
            + (1 if owner_initiated else 0),
            first_seen=row["first_seen"],
            last_inbound=row["last_inbound"],
            last_outbound=now,
            now=now,
        )
        self._conn.execute(
            "UPDATE persons SET outbound_count = outbound_count + 1,"
            " owner_initiated_count = owner_initiated_count + ?,"
            " last_outbound = ?, closeness = ?, updated_at = ?"
            " WHERE person_id = ?",
            (1 if owner_initiated else 0, now, closeness, now, person_id),
        )
        self._conn.commit()
        return self.get(person_id)  # type: ignore[return-value]

    # ── notes (append-only) ──────────────────────────────────────────

    def _append_note(self, person_id: str, note: str, now: float) -> int | None:
        note = (note or "").strip()
        if not note:
            return None
        existing = self._conn.execute(
            "SELECT id FROM notes WHERE person_id = ? AND note = ?",
            (person_id, note),
        ).fetchone()
        if existing is not None:
            return int(existing["id"])
        cur = self._conn.execute(
            "INSERT INTO notes (person_id, note, created_at) VALUES (?, ?, ?)",
            (person_id, note, now),
        )
        return int(cur.lastrowid)

    def add_note(self, person_id: str, note: str) -> int | None:
        """Append a mined fact. Exact duplicates are suppressed (same
        note already recorded); corrections append as new notes. Returns
        the note row id, or ``None`` for a blank note."""
        self.ensure(person_id)
        note_id = self._append_note(person_id, note, time.time())
        self._conn.commit()
        return note_id

    # ── links ────────────────────────────────────────────────────────

    def add_link(
        self, from_person: str, to_person: str, kind: str = ""
    ) -> int:
        """Link two people (``kind`` free-form). Both endpoints are
        ensured. Re-adding the identical triple returns the existing
        row id instead of duplicating history."""
        self.ensure(from_person)
        self.ensure(to_person)
        existing = self._conn.execute(
            "SELECT id FROM links WHERE from_person = ? AND to_person = ?"
            " AND kind = ?",
            (from_person, to_person, kind),
        ).fetchone()
        if existing is not None:
            return int(existing["id"])
        cur = self._conn.execute(
            "INSERT INTO links (from_person, to_person, kind, created_at)"
            " VALUES (?, ?, ?, ?)",
            (from_person, to_person, kind, time.time()),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def links_for(self, person_id: str) -> list[dict[str, Any]]:
        """All links touching this person, either direction."""
        return [
            dict(r)
            for r in self._conn.execute(
                "SELECT * FROM links WHERE from_person = ? OR to_person = ?"
                " ORDER BY id",
                (person_id, person_id),
            )
        ]

    # ── gifts ────────────────────────────────────────────────────────

    def add_gift(
        self,
        person_id: str,
        description: str,
        direction: GiftDirection,
        *,
        value_minor: int | None = None,
        currency: str = "",
        date: str = "",
    ) -> int:
        """Record a gift given to / received from this person, keyed off
        the same ``person_id`` as everything else."""
        if direction not in ("given", "received"):
            raise ValueError(
                f"direction must be 'given' or 'received', got {direction!r}"
            )
        self.ensure(person_id)
        cur = self._conn.execute(
            "INSERT INTO gifts (person_id, description, direction,"
            " value_minor, currency, date, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                person_id,
                description,
                direction,
                value_minor,
                currency,
                date,
                time.time(),
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def gifts_for(self, person_id: str) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self._conn.execute(
                "SELECT * FROM gifts WHERE person_id = ? ORDER BY id",
                (person_id,),
            )
        ]

    # ── obligations ──────────────────────────────────────────────────

    def link_obligation(
        self,
        person_id: str,
        commitment_id: str | None,
        what: str,
        *,
        deadline: str = "",
        status: str = "open",
    ) -> int:
        """Tie a commitment to a person. Re-linking the same
        ``commitment_id`` for the same person returns the existing row
        id (idempotent); a ``None`` commitment id always appends a new
        standalone obligation row."""
        self.ensure(person_id)
        if commitment_id is not None:
            existing = self._conn.execute(
                "SELECT id FROM obligations WHERE person_id = ?"
                " AND commitment_id = ?",
                (person_id, commitment_id),
            ).fetchone()
            if existing is not None:
                return int(existing["id"])
        cur = self._conn.execute(
            "INSERT INTO obligations (person_id, commitment_id, what,"
            " deadline, status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (person_id, commitment_id, what, deadline, status, time.time()),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def obligations_for(self, person_id: str) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self._conn.execute(
                "SELECT * FROM obligations WHERE person_id = ? ORDER BY id",
                (person_id,),
            )
        ]

    # ── queries ──────────────────────────────────────────────────────

    def top_contacts(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM persons ORDER BY closeness DESC, person_id"
            " LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._to_person(r) for r in rows]

    def needs_reconnect(
        self, days: int = 30, *, now: float | None = None
    ) -> list[dict[str, Any]]:
        """Close contacts gone quiet — candidates for proactive outreach.
        Same rule as the tracker: closeness >= 0.4 and no inbound or
        outbound traffic in the last ``days``."""
        now = time.time() if now is None else now
        cutoff = now - days * 86400
        rows = self._conn.execute(
            "SELECT * FROM persons WHERE closeness >= 0.4"
            " AND MAX(last_inbound, last_outbound) < ?"
            " ORDER BY closeness DESC",
            (cutoff,),
        ).fetchall()
        return [self._to_person(r) for r in rows]

    def search(self, query: str) -> list[dict[str, Any]]:
        """Case-insensitive substring match over id, display name, last
        topic, and notes. Closest contacts first."""
        like = "%" + query.replace("\\", "\\\\").replace("%", "\\%").replace(
            "_", "\\_"
        ) + "%"
        rows = self._conn.execute(
            "SELECT DISTINCT p.* FROM persons p"
            " LEFT JOIN notes n ON n.person_id = p.person_id"
            " WHERE p.person_id LIKE ? ESCAPE '\\'"
            " OR p.display_name LIKE ? ESCAPE '\\'"
            " OR p.last_topic LIKE ? ESCAPE '\\'"
            " OR n.note LIKE ? ESCAPE '\\'"
            " ORDER BY p.closeness DESC, p.person_id",
            (like, like, like, like),
        ).fetchall()
        return [self._to_person(r) for r in rows]

"""Structured CBT intervention records (L5) — docs 22+23, Stage B P7.

The health shell already ships the frame: the timeline (the person's
own logged words), mood-triggered crisis resources in ``drift``, the
previsit prep, and ``guard_coaching``'s no-diagnosis law for coach
output.  What doc 22 §6's exercise set needs on top of that frame is
the *content form*: thought records, behavioral activation, cognitive
restructuring, sleep and activity routines, relapse-prevention plans
and session prep as **structured intervention records** — a named
kind, a field schema, completeness that can be checked mechanically.

This module is that record organ, in the shell's own idiom (an
owner-scoped sqlite file beside the timeline's, the ``Database``
wrapper, tolerant reads):

* :data:`INTERVENTION_KINDS` is the schema: per kind, the required
  and optional fields and their shapes.  A record that does not meet
  its kind's schema is refused at write with every problem named —
  the store never holds a half-shaped record and calls it done.
* :func:`diagnostic_scan` reuses the shipped guard vocabulary
  (``health.coach.COACH_BANNED_PHRASES`` — the one list, not a copy)
  over a record's rendered text.  The scan is the skeptic's
  instrument in the CBT pilot cell: a record carrying diagnostic or
  prescriptive language is blocked, never delivered.
  ``guard_coaching`` itself is untouched — same behavior, same list.
* Records are the person's own exercise work, structured.  Nothing
  here interprets, scores a mind, or diagnoses: the module stores
  the exercise, checks its shape, and renders it back.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from .coach import COACH_BANNED_PHRASES
from .timeline import health_db_path

__all__ = [
    "INTERVENTION_KINDS",
    "InterventionError",
    "InterventionStore",
    "validate_intervention",
    "diagnostic_scan",
    "render_intervention",
    "default_interventions_path",
]


class InterventionError(ValueError):
    """An intervention record failed its kind's schema; the message
    names every problem found."""


#: Field shapes a schema may demand: ``text`` (a non-empty string),
#: ``list`` (a non-empty list of strings), ``score`` (a number in
#: 0–10 — the exercises' own mastery/pleasure/intensity scales),
#: ``number`` (any non-negative number).
_FIELD_TYPES = ("text", "list", "score", "number")

#: The exercise set (doc 22 §6) as structured records: kind →
#: {"required": {field: shape}, "optional": {field: shape}}.
INTERVENTION_KINDS: dict[str, dict[str, dict[str, str]]] = {
    "thought_record": {
        "required": {
            "situation": "text",
            "automatic_thought": "text",
            "emotion": "text",
            "evidence_for": "text",
            "evidence_against": "text",
            "balanced_thought": "text",
        },
        "optional": {
            "intensity_before": "score",
            "intensity_after": "score",
            "emotion_after": "text",
        },
    },
    "behavioral_activation": {
        "required": {
            "activity": "text",
            "scheduled_for": "text",
        },
        "optional": {
            "expected_pleasure": "score",
            "expected_mastery": "score",
            "outcome": "text",
        },
    },
    "cognitive_restructuring": {
        "required": {
            "trigger": "text",
            "automatic_thought": "text",
            "reframe": "text",
        },
        "optional": {
            "distortions": "list",
            "challenge_questions": "list",
        },
    },
    "sleep_routine": {
        "required": {
            "bedtime": "text",
            "wake_time": "text",
            "wind_down": "list",
        },
        "optional": {
            "caffeine_cutoff": "text",
            "screen_cutoff": "text",
        },
    },
    "activity_routine": {
        "required": {
            "activity": "text",
            "days": "list",
            "duration_minutes": "number",
        },
        "optional": {
            "notes": "text",
        },
    },
    "relapse_prevention": {
        "required": {
            "triggers": "list",
            "warning_signs": "list",
            "coping_steps": "list",
        },
        "optional": {
            "support_contacts": "list",
            "commitment": "text",
        },
    },
    "session_prep": {
        "required": {
            "topics": "list",
        },
        "optional": {
            "questions": "list",
            "since_last": "text",
            "goals_for_session": "list",
        },
    },
}


def _field_problem(field: str, shape: str, value: Any) -> str:
    """One field against its shape — '' when it holds."""
    if shape == "text":
        if not isinstance(value, str) or not value.strip():
            return f"field {field!r} must be a non-empty text"
    elif shape == "list":
        if not isinstance(value, (list, tuple)) or not value \
                or not all(isinstance(v, str) and v.strip()
                           for v in value):
            return (f"field {field!r} must be a non-empty list of "
                    "texts")
    elif shape == "score":
        if isinstance(value, bool) \
                or not isinstance(value, (int, float)) \
                or not 0 <= float(value) <= 10:
            return f"field {field!r} must be a number from 0 to 10"
    elif shape == "number":
        if isinstance(value, bool) \
                or not isinstance(value, (int, float)) \
                or float(value) < 0:
            return f"field {field!r} must be a non-negative number"
    else:  # a schema bug, named like any other problem
        return f"field {field!r} declares unknown shape {shape!r}"
    return ""


def validate_intervention(kind: str,
                          fields: dict[str, Any]) -> list[str]:
    """Every problem keeping ``fields`` from being a complete
    ``kind`` record, named.  An empty list accepts the record."""
    schema = INTERVENTION_KINDS.get(kind or "")
    if schema is None:
        return [f"unknown intervention kind {kind!r} — kinds are: "
                + ", ".join(sorted(INTERVENTION_KINDS))]
    if not isinstance(fields, dict):
        return ["intervention fields must be a record of named "
                "fields"]
    problems: list[str] = []
    known = {**schema["required"], **schema["optional"]}
    for name in fields:
        if name not in known:
            problems.append(
                f"field {name!r} is not part of a {kind} record")
    for name, shape in schema["required"].items():
        if name not in fields:
            problems.append(
                f"a {kind} record needs its {name!r} field")
        else:
            problem = _field_problem(name, shape, fields[name])
            if problem:
                problems.append(problem)
    for name, shape in schema["optional"].items():
        if name in fields:
            problem = _field_problem(name, shape, fields[name])
            if problem:
                problems.append(problem)
    return problems


def diagnostic_scan(text: str) -> list[str]:
    """The shipped guard vocabulary found in ``text`` — the banned
    diagnostic/prescriptive phrases from ``health.coach``, reused
    verbatim.  An empty list means the text stays coaching-shaped:
    it describes the person's own work and diagnoses nothing."""
    lowered = (text or "").lower()
    return [phrase for phrase in COACH_BANNED_PHRASES
            if phrase.lower() in lowered]


def render_intervention(kind: str, fields: dict[str, Any]) -> str:
    """One record as readable text, fields in schema order — the
    text the diagnostic scan judges and owner surfaces show."""
    schema = INTERVENTION_KINDS.get(kind or "")
    title = kind.replace("_", " ") if kind else "intervention"
    lines = [f"{title} — intervention record"]
    if schema is None:
        return "\n".join(lines)
    ordered = list(schema["required"]) + list(schema["optional"])
    for name in ordered:
        if name not in fields:
            continue
        value = fields[name]
        shown = "; ".join(str(v) for v in value) \
            if isinstance(value, (list, tuple)) else str(value)
        lines.append(f"{name.replace('_', ' ')}: {shown}")
    return "\n".join(lines)


def default_interventions_path() -> Path:
    """The records' home: beside the health timeline's database,
    owner-scoped like everything in the shell."""
    return health_db_path().parent / "interventions.db"


_DDL = """
CREATE TABLE IF NOT EXISTS intervention_records (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL DEFAULT '',
  fields_json TEXT NOT NULL DEFAULT '{}',
  contract_id TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'recorded',
  created_at REAL NOT NULL DEFAULT 0,
  updated_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_interventions_kind
  ON intervention_records(kind);
"""


class InterventionStore:
    """Intervention records on their own sqlite file (the timeline
    idiom: ``storage.db.Database`` over a caller-named path)."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        from ..storage.db import Database

        path = Path(db_path) if db_path \
            else default_interventions_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = Database(path)
        self.db.executescript(_DDL)

    def record(self, kind: str, fields: dict[str, Any], *,
               contract_id: str = "") -> dict[str, Any]:
        """Store one complete record.  An incomplete or mis-shaped
        record is refused with every problem named — the store is
        the schema's gate, not a bucket."""
        problems = validate_intervention(kind, fields)
        if problems:
            raise InterventionError(
                f"a {kind or 'intervention'} record cannot be "
                "stored: " + "; ".join(problems))
        now = time.time()
        row = {
            "id": f"iv-{uuid.uuid4().hex[:12]}",
            "kind": kind,
            "fields": dict(fields),
            "contract_id": contract_id,
            "status": "recorded",
            "created_at": now,
            "updated_at": now,
        }
        self.db.execute(
            "INSERT INTO intervention_records (id, kind, fields_json,"
            " contract_id, status, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (row["id"], row["kind"], json.dumps(row["fields"]),
             row["contract_id"], row["status"], row["created_at"],
             row["updated_at"]))
        return row

    @staticmethod
    def _row(row: Any) -> dict[str, Any]:
        out = dict(row)
        try:
            out["fields"] = json.loads(out.pop("fields_json")
                                       or "{}")
        except Exception:  # noqa: BLE001 - a torn record reads empty
            out["fields"] = {}
        return out

    def get(self, record_id: str) -> dict[str, Any] | None:
        row = self.db.query_one(
            "SELECT * FROM intervention_records WHERE id = ?",
            (record_id,))
        return self._row(row) if row else None

    def list(self, *, kind: str = "", limit: int = 50
             ) -> list[dict[str, Any]]:
        if kind:
            rows = self.db.query(
                "SELECT * FROM intervention_records WHERE kind = ?"
                " ORDER BY created_at DESC LIMIT ?",
                (kind, int(limit)))
        else:
            rows = self.db.query(
                "SELECT * FROM intervention_records"
                " ORDER BY created_at DESC LIMIT ?", (int(limit),))
        return [self._row(r) for r in rows]

    def set_status(self, record_id: str, status: str) -> None:
        self.db.execute(
            "UPDATE intervention_records SET status = ?,"
            " updated_at = ? WHERE id = ?",
            (status, time.time(), record_id))

    def validate_record(self, record_id: str) -> list[str]:
        """Re-validate a STORED record on read-back — the verifier's
        instrument: what the store holds now, checked against the
        schema now.  A missing record is itself the problem."""
        row = self.get(record_id)
        if row is None:
            return [f"no intervention record {record_id!r}"]
        return validate_intervention(row["kind"], row["fields"])

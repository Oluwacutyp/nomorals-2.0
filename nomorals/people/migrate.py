"""Migrate a JSON-backed ``RelationshipTracker`` into the SQLite spine.

Deliberately duck-typed: this module does **not** import
``nomorals.social.relationships``. The tracker is accepted as any object
exposing its records — via its internal ``_rels`` dict, via
``top_contacts()``, or as a plain dict — and each record as an object
with attributes or a dict with the same keys. That keeps this package
stdlib-only / sibling-relative (so the layering test stays green) and
lets migration work against any tracker-shaped source.

Idempotent: re-running adds nothing duplicate. Persons already in the
store keep their counters/scores untouched; only notes not already
present are appended. The return value is the number of persons newly
inserted by this call (``0`` on a clean re-run).
"""

from __future__ import annotations

from typing import Any

from .store import PersonStore

_FIELDS = (
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
    "notes",
)


def _field(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def _iter_records(tracker: Any) -> list[Any]:
    """All relationship records, by the most complete route available."""
    if isinstance(tracker, dict):
        rels = tracker.get("relationships")
        if isinstance(rels, dict):
            return list(rels.values())
        return list(tracker.values())
    # Fullest first: the internal dict holds EVERY contact; top_contacts
    # is capped by its limit argument, so it is only a fallback.
    internal = getattr(tracker, "_rels", None)
    if isinstance(internal, dict) and internal:
        return list(internal.values())
    top_contacts = getattr(tracker, "top_contacts", None)
    if callable(top_contacts):
        try:
            return list(top_contacts(limit=1_000_000))
        except TypeError:
            return list(top_contacts())
    get = getattr(tracker, "get", None)
    if callable(get):
        return []
    return []


def migrate_from_relationships(tracker: Any, store: PersonStore) -> int:
    """Copy every tracker contact into ``store``. Returns the number of
    persons newly inserted (0 when everything was already migrated)."""
    inserted = 0
    for record in _iter_records(tracker):
        person_id = _field(record, "person_id")
        if not person_id:
            continue
        notes = _field(record, "notes", []) or []
        if isinstance(notes, str):
            notes = [notes]
        snapshot = {
            field: _field(record, field)
            for field in _FIELDS
            if field != "notes"
        }
        snapshot["notes"] = [str(n) for n in notes]
        if store.import_record(snapshot):
            inserted += 1
    return inserted

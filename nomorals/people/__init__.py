"""SQLite identity spine: one durable record per person.

The system of record for *who* people are — closeness derived from the
owner's own actions, append-only notes/links/gifts/obligations, all
keyed off the same ``person_id``. Migrated from the JSON-backed
``nomorals.social.relationships`` tracker via
:func:`migrate_from_relationships`; that module is left untouched.

Stdlib only; sibling-relative imports only.
"""

from __future__ import annotations

from .migrate import migrate_from_relationships
from .store import PersonStore

__all__ = ["PersonStore", "migrate_from_relationships"]

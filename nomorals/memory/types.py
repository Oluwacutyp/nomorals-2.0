"""Typed memory objects — the Tana supertag pattern.

Every memory entity gets a TYPE with a schema. Applying a type adds
structured fields: a ``person`` has birthday/last_contact, a
``commitment`` has deadline/status. Precision over vibes — this is the
biggest hallucination reducer in the memory stack, because a query for
"the person I met at the conference" can filter to persons instead of
hoping the vector search lands right.

The existing ``~/memory/people/*.md`` pages are the seed: they parse
into ``person`` entities (read-only — the pages are never rewritten).
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.ids import new_id
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "ENTITY_TYPES",
    "TypedEntity",
    "entity_home",
    "load_entity",
    "parse_person_page",
    "save_entity",
]

# ── type schemas ─────────────────────────────────────────────────────────────
# Each type: fields → {required: bool, kind: "text"|"date"|"number"|"enum",
# options: [...] for enums, description: str}.

ENTITY_TYPES: dict[str, dict[str, Any]] = {
    "person": {
        "description": "A person the user knows.",
        "fields": {
            "name": {"required": True, "kind": "text",
                     "description": "Display name."},
            "birthday": {"required": False, "kind": "date",
                         "description": "Birthday, YYYY-MM-DD if known."},
            "last_contact": {"required": False, "kind": "date",
                             "description": "Last contact date."},
            "relationship": {"required": False, "kind": "text",
                             "description": "How they relate to the user."},
            "notes": {"required": False, "kind": "text",
                      "description": "Free-form notes."},
        },
    },
    "project": {
        "description": "A project the user works on.",
        "fields": {
            "name": {"required": True, "kind": "text",
                     "description": "Project name."},
            "status": {"required": False, "kind": "enum",
                       "options": ["active", "paused", "done", "idea"],
                       "description": "Lifecycle status."},
            "deadline": {"required": False, "kind": "date",
                         "description": "Deadline, YYYY-MM-DD if known."},
            "repo": {"required": False, "kind": "text",
                     "description": "Repo path or URL."},
        },
    },
    "place": {
        "description": "A place the user goes or cares about.",
        "fields": {
            "name": {"required": True, "kind": "text",
                     "description": "Place name."},
            "address": {"required": False, "kind": "text",
                        "description": "Address or area."},
            "notes": {"required": False, "kind": "text",
                      "description": "Free-form notes."},
        },
    },
    "commitment": {
        "description": "Something the user promised to do.",
        "fields": {
            "title": {"required": True, "kind": "text",
                      "description": "What was promised."},
            "deadline": {"required": False, "kind": "date",
                         "description": "Due date, YYYY-MM-DD if known."},
            "status": {"required": False, "kind": "enum",
                       "options": ["open", "done", "dropped"],
                       "description": "Commitment status."},
            "with_whom": {"required": False, "kind": "text",
                          "description": "Who it's owed to."},
        },
    },
    "preference": {
        "description": "A durable user preference.",
        "fields": {
            "topic": {"required": True, "kind": "text",
                      "description": "What the preference is about."},
            "value": {"required": True, "kind": "text",
                      "description": "The preferred value."},
            "strength": {"required": False, "kind": "number",
                         "description": "0-1, how strongly held."},
        },
    },
    "habit": {
        "description": "A recurring behavior.",
        "fields": {
            "name": {"required": True, "kind": "text",
                     "description": "Habit name."},
            "frequency": {"required": False, "kind": "text",
                          "description": "e.g. daily, weekly."},
            "streak": {"required": False, "kind": "number",
                       "description": "Current streak count."},
        },
    },
}


@dataclass
class TypedEntity:
    """One typed memory object. Validated against ENTITY_TYPES."""
    etype: str
    id: str
    fields: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Schema check: unknown fields warn (kept, not dropped);
        missing required fields raise."""
        schema = ENTITY_TYPES.get(self.etype)
        if schema is None:
            raise ValueError(
                f"unknown entity type {self.etype!r}; "
                f"use one of {sorted(ENTITY_TYPES)}")
        known = schema["fields"]
        for key in self.fields:
            if key not in known:
                warnings.warn(
                    f"entity {self.id}: unknown field {key!r} "
                    f"for type {self.etype!r} — kept, not validated")
                _log.warning("typed entity %s: unknown field %r for %s",
                             self.id, key, self.etype)
        for key, spec in known.items():
            if spec.get("required") and not self.fields.get(key):
                raise ValueError(
                    f"entity type {self.etype!r} requires field {key!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"etype": self.etype, "id": self.id, "fields": self.fields,
                "created_at": self.created_at, "updated_at": self.updated_at}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TypedEntity":
        return cls(etype=data["etype"], id=data["id"],
                   fields=dict(data.get("fields", {})),
                   created_at=float(data.get("created_at", time.time())),
                   updated_at=float(data.get("updated_at", time.time())))

    def index_text(self) -> str:
        """Flat text for BM25/vector indexing."""
        parts = [self.etype, self.id]
        parts += [f"{k}: {v}" for k, v in self.fields.items() if v]
        return "\n".join(parts)


# ── people-page seed (read-only) ─────────────────────────────────────────────

def parse_person_page(path: str | Path) -> TypedEntity:
    """Parse a ``~/memory/people/*.md`` page into a ``person`` entity.

    Read-only: the page is never modified. Frontmatter keys map to
    person fields (display_name→name, summary→notes); the body is kept
    as notes when no summary exists.
    """
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    front: dict[str, str] = {}
    body = raw
    if raw.startswith("---"):
        end = raw.find("---", 3)
        if end != -1:
            for line in raw[3:end].strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    front[k.strip()] = v.strip()
            body = raw[end + 3:].strip()
    name = front.get("display_name") or p.stem.replace("_", " ").title()
    notes = front.get("summary") or body[:2000]
    fields: dict[str, Any] = {"name": name, "notes": notes}
    if front.get("updated"):
        fields["last_contact"] = front["updated"]
    # The relationship line usually lives in the body ("she is his
    # girlfriend"); keep the raw body searchable via notes.
    return TypedEntity(etype="person", id=f"person_{p.stem}",
                       fields=fields)


# ── sidecar store ─────────────────────────────────────────────────────────────

def entity_home(settings: Any = None) -> Path:
    """Home-dir path for typed entities, honoring runtime settings."""
    if settings is not None:
        home = getattr(settings, "home_path", None)
        if home:
            return Path(home) / "memory" / "entities"
    return Path.home() / ".nomorals" / "memory" / "entities"


def save_entity(entity: TypedEntity, settings: Any = None) -> Path:
    """Persist an entity as JSON. Returns the file path."""
    entity.updated_at = time.time()
    entity.validate()
    dest = entity_home(settings) / entity.etype / f"{entity.id}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(entity.to_dict(), indent=2), encoding="utf-8")
    return dest


def load_entity(entity_id: str, etype: str | None = None,
                settings: Any = None) -> TypedEntity | None:
    """Load an entity by id (optionally constrained to a type)."""
    base = entity_home(settings)
    candidates = ([base / etype / f"{entity_id}.json"] if etype
                  else sorted(base.glob(f"*/{entity_id}.json")))
    for c in candidates:
        if c.is_file():
            return TypedEntity.from_dict(
                json.loads(c.read_text(encoding="utf-8")))
    return None


def new_entity(etype: str, fields: dict[str, Any]) -> TypedEntity:
    """Convenience constructor with a fresh id."""
    return TypedEntity(etype=etype, id=new_id(etype), fields=dict(fields))

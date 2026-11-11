"""Saved mock-screen projects (FakeDetail Phase 6,
census §30).

A project is a saved screen SPEC — the typed model plus
its display settings and media REFERENCES — never baked
pixels: reopening one re-renders from the spec, so an
edit to one message re-renders only that message's
consequences and the rest of the screen is provably the
same.

Persistence follows the house pattern (living_cards'
``CardStore``): a runtime-created table, ``INSERT OR
REPLACE`` behind a compare-and-swap revision, and an
in-memory fallback for db-less contexts (CLI, tests).
The table is the organ's own — ``mock_screen_projects``
— and every payload carries ``"mock_screen_project":
True``, so a mock project can never be confused with a
real chat record (census §30 risk).

Media: a project stores references, not bytes.  A
reference that is a runtime token (``attach:`` /
``last:``) is resolved at render time by the runtime,
exactly as in the compact form; a concrete path is
checked at render time by :func:`missing_media` — a
project whose media is gone fails honestly, naming the
message and the path, never a silent placeholder.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

from .schema import Screen

__all__ = [
    "MockProject",
    "ProjectError",
    "ProjectStore",
    "missing_media",
    "valid_project_name",
]

#: Project names: a readable label, no path characters,
#: no leading space or dash.
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _\-]{0,63}$")

_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS mock_screen_projects ("
    "name TEXT PRIMARY KEY, kind TEXT NOT NULL DEFAULT 'chat', "
    "revision INTEGER NOT NULL DEFAULT 1, "
    "payload TEXT NOT NULL DEFAULT '{}', "
    "updated_at REAL NOT NULL DEFAULT 0)")


class ProjectError(Exception):
    """A project cannot be saved, opened, or rendered as
    asked — the message names the project and the reason."""


def valid_project_name(name: str) -> bool:
    """A usable project name (a readable label, no path
    characters)."""
    return bool(_NAME_RE.match(str(name or "").strip()))


@dataclass
class MockProject:
    """One saved screen project: the spec, its kind, and
    its revision.  ``screen`` is the live model;
    ``payload`` is the stored JSON shape."""

    name: str
    screen: Screen
    kind: str = "chat"
    revision: int = 1
    updated_at: float = 0.0

    def to_dict(self) -> dict:
        return {"mock_screen_project": True,
                "name": self.name, "kind": self.kind,
                "revision": self.revision,
                "updated_at": self.updated_at,
                "screen": self.screen.to_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> MockProject:
        if not isinstance(data, dict) \
                or not data.get("mock_screen_project"):
            raise ProjectError(
                "that record is not a mock screen project")
        return cls(
            name=str(data.get("name") or ""),
            screen=Screen.from_dict(data.get("screen") or {}),
            kind=str(data.get("kind") or "chat"),
            revision=int(data.get("revision") or 1),
            updated_at=float(data.get("updated_at") or 0.0))


def missing_media(screen: Screen, exists=None) -> list:
    """Media references on the screen that cannot render:
    (message id, reference) for every concrete path that
    does not exist.  Runtime tokens (``attach:`` /
    ``last:``) are resolved at render time, so they are
    never flagged here; an empty reference is the explicit
    no-media case.  ``exists`` defaults to os.path.exists
    (injectable for tests)."""
    check = exists if exists is not None else os.path.exists
    missing: list = []
    for msg in screen.messages:
        ref = str(getattr(msg, "media", "") or "")
        if not ref:
            continue
        if ref.startswith(("attach:", "last:")):
            continue
        if not check(ref):
            missing.append((msg.id, ref))
    return missing


class ProjectStore:
    """Durable mock-screen projects (SQLite via the
    context db), with an in-memory fallback for db-less
    contexts.  ``save(project, expect_revision=N)`` is a
    compare-and-swap: the write lands only when the stored
    revision is N, so two writers can never silently
    overwrite each other's revision."""

    def __init__(self, db: Any = None) -> None:
        self._db = db
        self._lock = threading.Lock()
        self._mem: dict[str, MockProject] = {}
        self._table_ready = False
        if db is not None:
            try:
                db.execute(_TABLE_SQL)
                self._table_ready = True
            except Exception:  # noqa: BLE001 - degrade to memory
                self._db = None

    # ── writes ───────────────────────────────────────────

    def save(self, project: MockProject, *,
             expect_revision: int | None = None) -> bool:
        """Persist ``project``.  With ``expect_revision``,
        the write lands only if the stored revision
        matches (a fresh project expects None stored).
        Returns whether the write landed."""
        if not valid_project_name(project.name):
            return False
        project.updated_at = time.time()
        with self._lock:
            current = self._read(project.name)
            if expect_revision is not None:
                stored = current.revision if current is not None \
                    else None
                if stored != expect_revision:
                    return False
            if self._db is not None and self._table_ready:
                try:
                    self._db.execute(
                        "INSERT OR REPLACE INTO "
                        "mock_screen_projects "
                        "(name, kind, revision, payload, "
                        "updated_at) VALUES (?,?,?,?,?)",
                        (project.name, project.kind,
                         project.revision,
                         json.dumps(project.to_dict()),
                         project.updated_at))
                except Exception:  # noqa: BLE001
                    return False
            else:
                self._mem[project.name] = MockProject.from_dict(
                    project.to_dict())
            return True

    def delete(self, name: str) -> bool:
        """Delete a project.  True when one was removed."""
        with self._lock:
            if self._db is not None and self._table_ready:
                try:
                    cur = self._db.execute(
                        "DELETE FROM mock_screen_projects "
                        "WHERE name = ?", (name,))
                    return bool(getattr(cur, "rowcount", 0))
                except Exception:  # noqa: BLE001
                    return False
            return self._mem.pop(name, None) is not None

    # ── reads ────────────────────────────────────────────

    def _read(self, name: str) -> MockProject | None:
        if self._db is not None and self._table_ready:
            try:
                row = self._db.query_one(
                    "SELECT payload FROM mock_screen_projects "
                    "WHERE name = ?", (name,))
            except Exception:  # noqa: BLE001
                return None
            if not row:
                return None
            try:
                return MockProject.from_dict(
                    json.loads(row["payload"]))
            except (TypeError, ValueError, ProjectError):
                return None
        held = self._mem.get(name)
        return MockProject.from_dict(held.to_dict()) \
            if held is not None else None

    def load(self, name: str) -> MockProject | None:
        """The named project, or None when there is none."""
        with self._lock:
            return self._read(str(name or ""))

    def exists(self, name: str) -> bool:
        return self.load(name) is not None

    def list_names(self) -> list[str]:
        """Project names, newest first."""
        with self._lock:
            if self._db is not None and self._table_ready:
                try:
                    rows = self._db.query(
                        "SELECT name FROM mock_screen_projects "
                        "ORDER BY updated_at DESC")
                    return [str(r["name"]) for r in rows or []]
                except Exception:  # noqa: BLE001
                    return []
            return [p.name for p in sorted(
                self._mem.values(),
                key=lambda p: -p.updated_at)]

"""Persisted registry of executable skill packages.

Follows the ``nomorals/os/session.py`` pattern: a ``Database`` plus a
``CREATE TABLE IF NOT EXISTS`` DDL script executed eagerly at construction.

Multiple versions of one skill coexist (``PRIMARY KEY (name, version)``);
exactly one version per name carries the *active pin* and is what
``get(name)`` resolves.  Installing a new version never moves the pin —
you promote with :meth:`pin` and roll back the same way, which makes a
bad skill version a one-command revert instead of a reinstall.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..core.errors import NotFound
from ..core.logging_setup import get_logger
from .manifest import ManifestError, SkillManifest

if TYPE_CHECKING:  # pragma: no cover
    from ..storage.db import Database

_log = get_logger(__name__)

__all__ = ["SkillRegistry", "SKILL_PACKAGES_DDL"]

SKILL_PACKAGES_DDL = """
CREATE TABLE IF NOT EXISTS skill_packages (
    name        TEXT NOT NULL,
    version     TEXT NOT NULL,
    manifest    TEXT NOT NULL,
    enabled     INTEGER NOT NULL DEFAULT 1,
    active      INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    PRIMARY KEY (name, version)
);
CREATE INDEX IF NOT EXISTS idx_skill_packages_active
    ON skill_packages(name, active);
"""


@dataclass
class InstalledSkill:
    """One installed skill version as the registry sees it."""

    name: str
    version: str
    manifest: SkillManifest
    enabled: bool
    active: bool
    created_at: float
    updated_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "enabled": self.enabled,
            "active": self.active,
            "tools": list(self.manifest.tools),
            "owner": self.manifest.owner,
            "description": self.manifest.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class SkillRegistry:
    """Install, version, pin, enable/disable, and resolve skill packages."""

    def __init__(self, db: "Database") -> None:
        self.db = db
        db.executescript(SKILL_PACKAGES_DDL)

    # ── write ───────────────────────────────────────────────────────────
    def install(self, manifest: SkillManifest | dict[str, Any]) -> InstalledSkill:
        """Validate and store a skill version.  The first version installed
        for a name becomes the active pin; later versions do not move it —
        promote explicitly with :meth:`pin`."""
        if isinstance(manifest, dict):
            manifest = SkillManifest.from_dict(manifest)
        manifest.validate_or_raise()
        now = time.time()
        existing = self.db.query_one(
            "SELECT COUNT(*) AS n FROM skill_packages WHERE name=?",
            (manifest.name,))
        first_version = not existing or int(existing["n"]) == 0
        # Re-installing an existing version must not move the pin: capture
        # its current active state before INSERT OR REPLACE wipes the row.
        prev = self.db.query_one(
            "SELECT active FROM skill_packages WHERE name=? AND version=?",
            (manifest.name, manifest.version))
        keep_active = first_version or bool(prev and prev["active"])
        self.db.execute(
            "INSERT OR REPLACE INTO skill_packages "
            "(name, version, manifest, enabled, active, created_at, "
            "updated_at) VALUES (?,?,?,?,?,?,?)",
            (manifest.name, manifest.version,
             json.dumps(manifest.to_dict(), ensure_ascii=False),
             1, 1 if keep_active else 0, now, now),
        )
        _log.info("skill package installed: %s %s (active=%s)",
                  manifest.name, manifest.version, keep_active)
        return self.get(manifest.name, manifest.version)  # type: ignore[return-value]

    def pin(self, name: str, version: str) -> InstalledSkill:
        """Move the active pin for ``name`` to ``version``.  This is the
        rollback primitive: ``pin(name, older_version)`` reverts a bad
        release without reinstalling anything."""        row = self.db.query_one(
            "SELECT 1 FROM skill_packages WHERE name=? AND version=?",
            (name, version))
        if row is None:
            raise NotFound(f"no installed skill {name!r} version {version!r}")
        self.db.execute(
            "UPDATE skill_packages SET active=0, updated_at=? WHERE name=?",
            (time.time(), name))
        self.db.execute(
            "UPDATE skill_packages SET active=1, updated_at=? WHERE name=? "
            "AND version=?", (time.time(), name, version))
        _log.info("skill %s pinned to version %s", name, version)
        return self.get(name, version)  # type: ignore[return-value]

    def deactivate(self, name: str) -> bool:
        """Clear the active pin for ``name``.

        All installed versions stay in the registry but none is active —
        the skill will not resolve until explicitly pinned.  Used for
        distilled drafts, which must wait for canary validation plus an
        explicit pin before they can take effect (never auto-activated).
        Returns True when at least one version row was touched.
        """
        cur = self.db.execute(
            "UPDATE skill_packages SET active=0, updated_at=? WHERE name=?",
            (time.time(), name))
        return cur.rowcount > 0

    def enable(self, name: str) -> bool:
        """Enable all installed versions of ``name``.  Returns False when
        the name is not installed."""
        return self._set_enabled(name, True)

    def disable(self, name: str) -> bool:
        """Disable all installed versions of ``name``.  A disabled skill
        resolves but refuses to run — the runner reports it instead of
        executing.  Returns False when the name is not installed."""
        return self._set_enabled(name, False)

    def _set_enabled(self, name: str, enabled: bool) -> bool:
        cur = self.db.execute(
            "UPDATE skill_packages SET enabled=?, updated_at=? WHERE name=?",
            (1 if enabled else 0, time.time(), name))
        return cur.rowcount > 0

    # ── read ────────────────────────────────────────────────────────────
    def get(self, name: str,
            version: str | None = None) -> InstalledSkill | None:
        """Resolve a skill: the active pin when ``version`` is None, else
        the exact version.  Returns None when nothing matches."""
        if version is None:
            row = self.db.query_one(
                "SELECT * FROM skill_packages WHERE name=? AND active=1",
                (name,))
        else:
            row = self.db.query_one(
                "SELECT * FROM skill_packages WHERE name=? AND version=?",
                (name, version))
        return self._from_row(row) if row else None

    def is_enabled(self, name: str, version: str | None = None) -> bool:
        skill = self.get(name, version)
        return bool(skill and skill.enabled)

    def versions(self, name: str) -> list[str]:
        rows = self.db.query(
            "SELECT version, active FROM skill_packages WHERE name=? "
            "ORDER BY created_at", (name,))
        return [r["version"] for r in rows]

    def list(self) -> list[dict[str, Any]]:
        """One entry per skill name: active version, all versions, enabled
        flag, tool chain."""
        rows = self.db.query(
            "SELECT * FROM skill_packages ORDER BY name, created_at")
        by_name: dict[str, dict[str, Any]] = {}
        for row in rows:
            skill = self._from_row(row)
            entry = by_name.setdefault(skill.name, {
                "name": skill.name,
                "active_version": None,
                "versions": [],
                "enabled": True,
                "tools": [],
                "owner": skill.manifest.owner,
                "description": skill.manifest.description,
            })
            entry["versions"].append(skill.version)
            entry["enabled"] = entry["enabled"] and skill.enabled
            if skill.active:
                entry["active_version"] = skill.version
                entry["tools"] = list(skill.manifest.tools)
                entry["owner"] = skill.manifest.owner
                entry["description"] = skill.manifest.description
        return [by_name[name] for name in sorted(by_name)]

    # ── internals ───────────────────────────────────────────────────────
    @staticmethod
    def _from_row(row: dict[str, Any]) -> InstalledSkill:
        try:
            manifest = SkillManifest.from_dict(
                json.loads(row["manifest"] or "{}"))
        except ManifestError:
            # A manifest that no longer validates (schema drift) still
            # resolves — the runner will refuse it with reasons.
            manifest = SkillManifest(name=row["name"], version=row["version"],
                                     tools=[])
        return InstalledSkill(
            name=row["name"],
            version=row["version"],
            manifest=manifest,
            enabled=bool(row.get("enabled", 1)),
            active=bool(row.get("active", 0)),
            created_at=float(row.get("created_at", 0.0)),
            updated_at=float(row.get("updated_at", 0.0)),
        )

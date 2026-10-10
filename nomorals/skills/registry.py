"""Persisted registry of executable skill packages.

Follows the ``nomorals/os/session.py`` pattern: a ``Database`` plus a
``CREATE TABLE IF NOT EXISTS`` DDL script executed eagerly at construction.

Multiple versions of one skill coexist (``PRIMARY KEY (name, version)``);
exactly one version per name carries the *active pin* and is what
``get(name)`` resolves.  Installing a new version never moves the pin —
you promote with :meth:`pin` and roll back the same way, which makes a
bad skill version a one-command revert instead of a reinstall.

Rug-pull defense: every installed version stores a short sha256 of its
canonical manifest JSON.  Re-installing a version whose bytes changed
warns loudly (per the MCP rug-pull research: "no widely deployed
mechanism for detecting this change" — this is that mechanism), and
:meth:`verify_definitions` re-scans every stored manifest against its
hash on demand.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..core.errors import NotFound
from ..core.logging_setup import get_logger
from .manifest import ManifestError, SkillManifest, diff_manifests

if TYPE_CHECKING:  # pragma: no cover
    from ..storage.db import Database

_log = get_logger(__name__)

__all__ = ["SkillRegistry", "SKILL_PACKAGES_DDL", "InstalledSkill"]

SKILL_PACKAGES_DDL = """
CREATE TABLE IF NOT EXISTS skill_packages (
    name        TEXT NOT NULL,
    version     TEXT NOT NULL,
    manifest    TEXT NOT NULL,
    enabled     INTEGER NOT NULL DEFAULT 1,
    active      INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    manifest_hash TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (name, version)
);
CREATE INDEX IF NOT EXISTS idx_skill_packages_active
    ON skill_packages(name, active);
"""

#: Stopwords stripped before keyword matching in search/recommend.
_STOPWORDS = frozenset(
    "a an the and or of to in on for with is are was were be been it its "
    "this that these those as at by from into than then so such no not do "
    "does did can could should would will just me my we our you your he she "
    "they them his her their what when where which who whom how why please "
    "".split())


def _tokens(text: str) -> list[str]:
    return [t for t in
            "".join(c.lower() if c.isalnum() else " " for c in text).split()
            if t and t not in _STOPWORDS]


def _ensure_columns(db: "Database") -> None:
    """Guarded migration: add columns that older installs lack."""
    try:
        cols = {r["name"] for r in db.query("PRAGMA table_info(skill_packages)")}
    except Exception:  # noqa: BLE001 — table may not exist yet
        return
    if "manifest_hash" not in cols:
        try:
            db.execute("ALTER TABLE skill_packages ADD COLUMN "
                       "manifest_hash TEXT NOT NULL DEFAULT ''")
        except Exception as exc:  # noqa: BLE001 — never break construction
            _log.debug("skill_packages migration skipped: %s", exc)


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
            "tags": list(self.manifest.tags),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class SkillRegistry:
    """Install, version, pin, enable/disable, and resolve skill packages."""

    def __init__(self, db: "Database") -> None:
        self.db = db
        db.executescript(SKILL_PACKAGES_DDL)
        _ensure_columns(db)

    # ── write ───────────────────────────────────────────────────────────
    def install(self, manifest: SkillManifest | dict[str, Any]) -> InstalledSkill:
        """Validate and store a skill version.  The first version installed
        for a name becomes the active pin; later versions do not move it —
        promote explicitly with :meth:`pin`.

        Re-installing an existing version with *changed* manifest bytes
        logs a rug-pull warning (the definition changed under a pinned
        version) but still stores — the pin, not the bytes, is the trust
        boundary; use :meth:`verify_definitions` to audit.
        """
        if isinstance(manifest, dict):
            manifest = SkillManifest.from_dict(manifest)
        manifest.validate_or_raise()
        now = time.time()
        new_hash = manifest.manifest_hash()
        existing = self.db.query_one(
            "SELECT COUNT(*) AS n FROM skill_packages WHERE name=?",
            (manifest.name,))
        first_version = not existing or int(existing["n"]) == 0
        # Re-installing an existing version must not move the pin: capture
        # its current active state before INSERT OR REPLACE wipes the row.
        prev = self.db.query_one(
            "SELECT active, manifest_hash FROM skill_packages "
            "WHERE name=? AND version=?",
            (manifest.name, manifest.version))
        keep_active = first_version or bool(prev and prev["active"])
        if prev and prev.get("manifest_hash") and \
                prev["manifest_hash"] != new_hash:
            _log.warning(
                "skill %s v%s re-installed with CHANGED manifest bytes "
                "(hash %s → %s): possible rug-pull — review the diff before "
                "trusting this version", manifest.name, manifest.version,
                prev["manifest_hash"], new_hash)
        self.db.execute(
            "INSERT OR REPLACE INTO skill_packages "
            "(name, version, manifest, enabled, active, created_at, "
            "updated_at, manifest_hash) VALUES (?,?,?,?,?,?,?,?)",
            (manifest.name, manifest.version,
             json.dumps(manifest.to_dict(), ensure_ascii=False),
             1, 1 if keep_active else 0, now, now, new_hash),
        )
        _log.info("skill package installed: %s %s (active=%s)",
                  manifest.name, manifest.version, keep_active)
        return self.get(manifest.name, manifest.version)  # type: ignore[return-value]

    def uninstall(self, name: str,
                  version: str | None = None) -> int:
        """Remove installed versions.  With ``version``: that version only;
        without: every version of the name.  Returns rows removed.

        There was previously no way to remove a skill at all — only
        disable it.  Uninstalling the active pin leaves the name with no
        active version until another is pinned.
        """
        if version is not None:
            cur = self.db.execute(
                "DELETE FROM skill_packages WHERE name=? AND version=?",
                (name, version))
        else:
            cur = self.db.execute(
                "DELETE FROM skill_packages WHERE name=?", (name,))
        removed = cur.rowcount or 0
        if removed:
            _log.info("skill %s%s uninstalled (%d row(s))", name,
                      f" v{version}" if version else "", removed)
        return removed

    def pin(self, name: str, version: str) -> InstalledSkill:
        """Move the active pin for ``name`` to ``version``.  This is the
        rollback primitive: ``pin(name, older_version)`` reverts a bad
        release without reinstalling anything."""
        row = self.db.query_one(
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

    def list(self, tag: str | None = None,
             owner: str | None = None) -> list[dict[str, Any]]:
        """One entry per skill name: active version, all versions, enabled
        flag, tool chain.  Filter by tag or owner when given."""
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
                "tags": list(skill.manifest.tags),
            })
            entry["versions"].append(skill.version)
            entry["enabled"] = entry["enabled"] and skill.enabled
            if skill.active:
                entry["active_version"] = skill.version
                entry["tools"] = list(skill.manifest.tools)
                entry["owner"] = skill.manifest.owner
                entry["description"] = skill.manifest.description
                entry["tags"] = list(skill.manifest.tags)
        entries = [by_name[name] for name in sorted(by_name)]
        if tag:
            entries = [e for e in entries if tag in e["tags"]]
        if owner:
            entries = [e for e in entries if e["owner"] == owner]
        return entries

    def stats(self) -> dict[str, Any]:
        """Registry totals: skills, versions, active pins, disabled."""
        rows = self.db.query(
            "SELECT name, version, enabled, active FROM skill_packages")
        names = {r["name"] for r in rows}
        return {
            "skills": len(names),
            "versions": len(rows),
            "active_pins": sum(1 for r in rows if r["active"]),
            "disabled_versions": sum(1 for r in rows if not r["enabled"]),
        }

    # ── discovery: search & recommend ───────────────────────────────────
    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Lexical skill search over name/description/tools/tags/owner.

        The measured default for tool selection as retrieval: keyword
        search first (it reaches ~90% of embedding retrieval on real
        tasks at zero cost and zero dependencies); semantic search is a
        documented extension point, not a half-built dependency.
        Returns entries with a ``score`` key, best first.
        """
        terms = _tokens(query)
        if not terms:
            return []
        scored: list[tuple[float, dict[str, Any]]] = []
        for entry in self.list():
            score = 0.0
            name = entry["name"].lower()
            desc = (entry["description"] or "").lower()
            owner = (entry["owner"] or "").lower()
            tools = " ".join(entry["tools"]).lower()
            tags = " ".join(entry["tags"]).lower()
            for term in terms:
                if term in name:
                    score += 10.0
                if term in tags:
                    score += 5.0
                if term in desc:
                    score += 3.0
                if term in tools:
                    score += 2.0
                if term in owner:
                    score += 1.0
            if score > 0:
                hit = dict(entry)
                hit["score"] = round(score, 2)
                scored.append((score, hit))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [hit for _, hit in scored[:max(1, limit)]]

    def recommend(self, task_description: str, limit: int = 5,
                  quality: dict[str, float] | None = None
                  ) -> list[dict[str, Any]]:
        """Rank skills for a task description.

        Lexical match via :meth:`search`, then an optional reliability
        boost: pass ``quality`` as ``{skill_name: success_rate}`` (e.g.
        from ``SkillBench``) and reliable skills rank above flaky ones
        with similar textual fit.
        """
        hits = self.search(task_description, limit=max(1, limit) * 3)
        quality = quality or {}
        ranked: list[tuple[float, dict[str, Any]]] = []
        for hit in hits:
            boost = 1.0 + 0.5 * float(quality.get(hit["name"], 0.0) or 0.0)
            final = hit["score"] * boost
            ranked.append((final, hit))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        out = []
        for final, hit in ranked[:max(1, limit)]:
            hit = dict(hit)
            hit["rank_score"] = round(final, 2)
            out.append(hit)
        return out

    # ── portability & review ────────────────────────────────────────────
    def export_skill(self, name: str,
                     version: str | None = None) -> dict[str, Any]:
        """Export one installed version as a portable package dict.

        The Agent Skills standard's big idea is universal portability —
        a skill authored anywhere runs anywhere.  The export round-trips
        through :meth:`import_skill`.
        """
        skill = self.get(name, version)
        if skill is None:
            raise NotFound(f"no installed skill {name!r}"
                           + (f" version {version!r}" if version else ""))
        return {
            "format": "devon-skill-package/1",
            "exported_at": time.time(),
            "active": skill.active,
            "enabled": skill.enabled,
            "manifest": skill.manifest.to_dict(),
        }

    def import_skill(self, package: dict[str, Any]) -> InstalledSkill:
        """Install from an :meth:`export_skill` package dict."""
        if not isinstance(package, dict) or "manifest" not in package:
            raise ManifestError("not a skill package: missing 'manifest'")
        installed = self.install(package["manifest"])
        if package.get("enabled") is False:
            self.disable(installed.name)
        return installed

    def diff_versions(self, name: str, v1: str, v2: str) -> list[str]:
        """What changed between two installed versions of a skill.

        The review aid before promoting a pin — the GitOps diff habit
        applied to skill definitions.
        """
        old = self.get(name, v1)
        new = self.get(name, v2)
        missing = [v for v, s in ((v1, old), (v2, new)) if s is None]
        if missing:
            raise NotFound(f"no installed skill {name!r} version(s): "
                           + ", ".join(missing))
        assert old is not None and new is not None
        return diff_manifests(old.manifest, new.manifest)

    def verify_definitions(self) -> list[dict[str, Any]]:
        """Re-scan every stored manifest against its recorded hash.

        The rug-pull re-scan the research says nobody deploys: returns
        one entry per version whose bytes no longer match the hash
        recorded at install time (direct DB tampering or a migration
        that rewrote manifests).
        """
        bad: list[dict[str, Any]] = []
        rows = self.db.query(
            "SELECT name, version, manifest, manifest_hash, active "
            "FROM skill_packages")
        for row in rows:
            try:
                manifest = SkillManifest.from_dict(
                    json.loads(row["manifest"] or "{}"))
            except ManifestError:
                bad.append({"name": row["name"], "version": row["version"],
                            "reason": "manifest no longer parses",
                            "active": bool(row["active"])})
                continue
            recorded = row.get("manifest_hash") or ""
            if recorded and manifest.manifest_hash() != recorded:
                bad.append({"name": row["name"], "version": row["version"],
                            "reason": f"hash mismatch: recorded "
                                      f"{recorded}, now "
                                      f"{manifest.manifest_hash()}",
                            "active": bool(row["active"])})
        return bad

    # ── presentation ────────────────────────────────────────────────────
    @staticmethod
    def format_table(entries: list[dict[str, Any]] | None = None,
                     registry: "SkillRegistry | None" = None) -> str:
        """Render skill entries as a readable text table.

        Pass entries explicitly, or pass ``registry=`` to list live.
        """
        if entries is None:
            entries = registry.list() if registry is not None else []
        if not entries:
            return ("no skill packages installed — "
                    "nm skill install --manifest <file.json>")
        rows = []
        for e in entries:
            status = "●" if e["enabled"] and e["active_version"] else \
                "◌" if e["enabled"] else "✕"
            rows.append((
                status,
                e["name"],
                e["active_version"] or "—",
                ",".join(e["versions"]),
                ",".join(e["tools"][:4]) + ("…"
                                            if len(e["tools"]) > 4 else ""),
                (e["description"] or "")[:48],
            ))
        widths = [max(len(r[i]) for r in rows) for i in range(6)]
        widths[0] = max(widths[0], 1)
        header = ("  ", "skill", "active", "versions", "tools",
                  "description")
        lines = [" ".join(h.ljust(widths[i])
                          for i, h in enumerate(header)).rstrip(),
                 " ".join("─" * widths[i] for i in range(6))]
        for r in rows:
            lines.append(" ".join(str(c).ljust(widths[i])
                                  for i, c in enumerate(r)).rstrip())
        lines.append("")
        lines.append("● active pin   ◌ enabled, no pin   ✕ disabled")
        return "\n".join(lines)

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

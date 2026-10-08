"""Spec-first plan mode (#4 extension) — the plan IS a durable design document.

SMART/Kiro pattern: design docs are the durable artifact, and code is
*generated from the spec*, not from a throwaway prompt.  A ``CodePlan``
(approve → execute) can be promoted to a ``PlanSpec``; the spec is
stored and versioned, and code can be regenerated from it at any time
(``regenerate_from_spec``) with a diff of what changed.

Hercules steal: every generation prompt carries the backend-included
block by default — apps/tools ship with their data layer (DB schema,
storage, scheduled jobs) instead of arriving stateless.
"""

from __future__ import annotations

import difflib
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "BACKEND_INCLUDED_BLOCK",
    "PlanSpec",
    "SpecStore",
    "plan_to_spec",
    "diff_specs",
    "render_spec",
    "inject_backend_included",
    "should_backend_include",
]

#: Spec fields compared by the version diff.
_SPEC_TEXT_FIELDS = ("goals", "architecture", "data_model", "api")


# ── backend-included (Hercules steal) ────────────────────────────────────

BACKEND_INCLUDED_BLOCK = """\
BACKEND INCLUDED (default — do not skip): this is a real app, not a demo.
Ship the data layer with it:
- a SQLite schema (CREATE TABLE IF NOT EXISTS ...) for the entities the app owns;
- storage helpers (save/load/list/get) with a sensible default path under the app dir;
- a scheduled-job hook (cron-style registration) when the feature needs periodic work;
- migrations that never destroy existing data.
Never deliver a stateless app when persistence would help, and never stub
the data layer (no NotImplementedError, no TODO-as-delivery)."""


def inject_backend_included(system_prompt: str) -> str:
    """Append the backend-included block to a generation system prompt."""
    try:
        base = str(system_prompt or "")
        if "BACKEND INCLUDED" in base:
            return base
        return base + "\n\n" + BACKEND_INCLUDED_BLOCK
    except Exception:  # noqa: BLE001 — never raises
        return str(system_prompt or "")


def should_backend_include(filename: str) -> bool:
    """Heuristic: whole-file generations get the block; test/doc edits skip."""
    try:
        name = str(filename or "").lower()
        if not name:
            return False
        if "/tests/" in name or name.startswith("tests/") or name.startswith("test_"):
            return False
        if name.endswith((".md", ".txt", ".rst", ".json", ".yaml", ".yml", ".toml")):
            return False
        return True
    except Exception:  # noqa: BLE001 — never raises
        return False


# ── spec document ────────────────────────────────────────────────────────


@dataclass
class PlanSpec:
    """A durable design document.  The code is generated FROM this."""

    spec_id: str = ""
    version: int = 1
    goals: str = ""
    architecture: str = ""
    data_model: str = ""
    api: str = ""
    tasks: list[dict[str, Any]] = field(default_factory=list)
    parent_spec_id: str = ""
    change_note: str = ""
    created_at: float = field(default_factory=time.time)

    @property
    def key(self) -> str:
        return f"{self.spec_id}@v{self.version}"


class SpecStore:
    """Stored, versioned plan specs.

    Specs live in memory for the session AND are persisted to a JSON
    file (``NM_SPEC_STORE``, default ``data/code-specs.json``) so a spec
    written by one ``nm code`` invocation can be re-read, edited, and
    regenerated from by later ones.  Like :class:`plan_mode.PlanStore`,
    persistence is best-effort; nothing here ever raises.
    """

    _specs: dict[str, PlanSpec] = {}
    _loaded: bool = False
    _KEEP = 50

    @classmethod
    def _path(cls) -> Path:
        return Path(os.environ.get("NM_SPEC_STORE", "data/code-specs.json"))

    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._loaded:
            return
        cls._loaded = True
        try:
            raw = cls._path().read_text(encoding="utf-8")
        except OSError:
            return
        try:
            rows = json.loads(raw)
        except ValueError:
            _log.warning("spec store file is not valid JSON — starting fresh")
            return
        if not isinstance(rows, list):
            return
        for row in rows:
            if not isinstance(row, dict) or "spec_id" not in row:
                continue
            try:
                spec = PlanSpec(
                    spec_id=str(row["spec_id"]),
                    version=int(row.get("version", 1)),
                    goals=str(row.get("goals", "")),
                    architecture=str(row.get("architecture", "")),
                    data_model=str(row.get("data_model", "")),
                    api=str(row.get("api", "")),
                    tasks=list(row.get("tasks") or []),
                    parent_spec_id=str(row.get("parent_spec_id", "")),
                    change_note=str(row.get("change_note", "")),
                    created_at=float(row.get("created_at", time.time())),
                )
            except (TypeError, ValueError):
                continue
            cls._specs[spec.key] = spec

    @classmethod
    def _persist(cls) -> None:
        try:
            path = cls._path()
            path.parent.mkdir(parents=True, exist_ok=True)
            rows = [asdict(s) for s in
                    sorted(cls._specs.values(),
                           key=lambda s: s.created_at)[-cls._KEEP:]]
            path.write_text(json.dumps(rows, ensure_ascii=False),
                            encoding="utf-8")
        except OSError as exc:  # noqa: BLE001 — persistence is best-effort
            _log.warning("could not persist spec store: %s", exc)

    @classmethod
    def new(cls, goals: str, architecture: str = "",
            data_model: str = "", api: str = "",
            tasks: list[dict[str, Any]] | None = None) -> PlanSpec:
        """Create the first version of a spec.  Never raises."""
        try:
            spec = PlanSpec(
                spec_id="spec_" + uuid.uuid4().hex[:10],
                version=1,
                goals=str(goals or ""),
                architecture=str(architecture or ""),
                data_model=str(data_model or ""),
                api=str(api or ""),
                tasks=list(tasks or []),
            )
        except Exception:  # noqa: BLE001
            return PlanSpec()
        cls._ensure_loaded()
        cls._specs[spec.key] = spec
        cls._persist()
        return spec

    @classmethod
    def get(cls, key: str) -> PlanSpec | None:
        """Fetch by ``spec_id`` (latest version) or ``spec_id@vN``."""
        try:
            cls._ensure_loaded()
            key = str(key or "").strip()
            if not key:
                return None
            if key in cls._specs:
                return cls._specs[key]
            # latest version of a bare spec_id
            cands = [s for s in cls._specs.values()
                     if s.spec_id == key]
            if not cands:
                return None
            return max(cands, key=lambda s: s.version)
        except Exception:  # noqa: BLE001 — never raises
            return None

    @classmethod
    def bump_version(cls, key: str,
                     changes: dict[str, Any] | None = None,
                     note: str = "") -> PlanSpec | None:
        """Spec edit → new version (parent linkage preserved).

        ``changes`` may override any of goals/architecture/data_model/
        api/tasks.  Returns the new version, or None for an unknown key.
        Never raises.
        """
        try:
            old = cls.get(key)
            if old is None:
                return None
            changes = changes or {}
            data = asdict(old)
            for field_name in (*_SPEC_TEXT_FIELDS, "tasks"):
                if field_name in changes:
                    data[field_name] = changes[field_name]
            new = PlanSpec(
                spec_id=old.spec_id,
                version=old.version + 1,
                goals=str(data.get("goals", "")),
                architecture=str(data.get("architecture", "")),
                data_model=str(data.get("data_model", "")),
                api=str(data.get("api", "")),
                tasks=list(data.get("tasks") or []),
                parent_spec_id=old.key,
                change_note=str(note or ""),
            )
            cls._specs[new.key] = new
            cls._persist()
            return new
        except Exception:  # noqa: BLE001 — never raises
            return None

    @classmethod
    def history(cls, spec_id: str) -> list[PlanSpec]:
        """All versions of a spec, oldest first.  Never raises."""
        try:
            cls._ensure_loaded()
            return sorted(
                (s for s in cls._specs.values() if s.spec_id == spec_id),
                key=lambda s: s.version)
        except Exception:  # noqa: BLE001
            return []


def diff_specs(old: PlanSpec, new: PlanSpec) -> dict[str, str]:
    """Unified diff per changed spec field (for code-regen review).

    Returns ``{field: diff_text}`` for text fields and a task-list diff
    under ``"tasks"`` when the task set changed.  Never raises.
    """
    out: dict[str, str] = {}
    try:
        if old is None or new is None:
            return out
        for name in _SPEC_TEXT_FIELDS:
            a = str(getattr(old, name, "") or "")
            b = str(getattr(new, name, "") or "")
            if a != b:
                out[name] = "\n".join(difflib.unified_diff(
                    a.splitlines(), b.splitlines(),
                    fromfile=f"{old.key}:{name}", tofile=f"{new.key}:{name}",
                    lineterm=""))
        old_tasks = [json.dumps(t, sort_keys=True, default=str)
                     for t in (old.tasks or [])]
        new_tasks = [json.dumps(t, sort_keys=True, default=str)
                     for t in (new.tasks or [])]
        if old_tasks != new_tasks:
            out["tasks"] = "\n".join(difflib.unified_diff(
                old_tasks, new_tasks,
                fromfile=f"{old.key}:tasks", tofile=f"{new.key}:tasks",
                lineterm=""))
    except Exception:  # noqa: BLE001 — never raises
        return out
    return out


def render_spec(spec: PlanSpec) -> str:
    """Human-readable design document.  Never raises."""
    try:
        lines = [f"📐 SPEC {spec.key}", ""]
        if spec.parent_spec_id:
            lines.append(f"(revises {spec.parent_spec_id})")
            lines.append("")
        if spec.change_note:
            lines.append(f"CHANGE: {spec.change_note[:200]}")
            lines.append("")
        sections = [("GOALS", spec.goals), ("ARCHITECTURE", spec.architecture),
                    ("DATA MODEL", spec.data_model), ("API", spec.api)]
        for title, body in sections:
            lines.append(f"## {title}")
            lines.append(str(body or "(not specified)").strip() or "(not specified)")
            lines.append("")
        lines.append("## TASKS")
        if spec.tasks:
            for t in spec.tasks:
                path = t.get("path", "?") if isinstance(t, dict) else "?"
                what = (t.get("what") or t.get("why", "")) if isinstance(t, dict) else ""
                tag = "new" if isinstance(t, dict) and t.get("new_file") else "edit"
                lines.append(f"  [{tag}] {path} — {str(what)[:120]}")
        else:
            lines.append("  (no tasks)")
        return "\n".join(lines)
    except Exception:  # noqa: BLE001 — never raises
        return f"SPEC {getattr(spec, 'key', '?')}"


def plan_to_spec(plan: Any, goals: str = "") -> PlanSpec | None:
    """Promote a :class:`plan_mode.CodePlan` to a durable PlanSpec.

    The file list becomes the task list; ``goals`` defaults to the plan
    task.  Architecture/data-model/API start empty — the designer (or
    the model) fills them on the first edit.  Never raises.
    """
    try:
        if plan is None:
            return None
        files = getattr(plan, "files", None) or []
        tasks: list[dict[str, Any]] = []
        for f in files:
            if not isinstance(f, dict):
                continue
            if not f.get("path"):
                continue
            tasks.append({
                "path": str(f["path"]),
                "what": str(f.get("why", "")),
                "new_file": bool(f.get("new_file")),
            })
        return SpecStore.new(
            goals=str(goals or getattr(plan, "task", "") or ""),
            tasks=tasks,
        )
    except Exception:  # noqa: BLE001 — never raises
        return None

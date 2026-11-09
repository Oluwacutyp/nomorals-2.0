"""The career pipeline store — applications, stages, next actions.

The careers goal class (WS2) is measured by an *application
pipeline* (``application_pipeline`` in the class registry), and
until this module there was no pipeline to measure: careers was
the one owner-named class with no organ at all.  This store is
that organ's memory — the owner's own job hunt, personal-first:
target companies and roles, every application with its stage,
an append-only stage history (when it moved is a fact, never
rewritten), notes, and the next action with its due date.

Persistence follows the finance savings store
(:mod:`nomorals.finance.goals`) — the repo's shape model for a
personal tracker: one JSON file, read through the storeguard
family guard (unreadable bytes are quarantined and saves then
refuse rather than overwriting them), written atomically via a
tmp file + replace.  The store lives at ``<home>/careers/
applications.json`` (see :func:`careers_path`, the
``finance_paths`` pattern).

Stage vocabulary: no application-stage vocabulary existed in the
repo (``matching``'s "shortlist" is candidate ranking, a different
thing), so the funnel adopts the WS2 census's own naming —
saved → applied → screen → interview → offer — with the census's
terminal "closed" split into its three real outcomes (accepted /
rejected / withdrawn): a single "closed" would hide whether the
pipeline succeeded, and the provider's measured % turns on that
distinction.  Moves between non-terminal stages are free (real
pipelines skip and regress; the append-only history records what
actually happened); a terminal application does not move again —
retrying a closed application is a NEW application.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from ..core.storeguard import StoreUnreadableError, read_json_file

_log = get_logger("nomorals.careers")

__all__ = [
    "ACTIVE_STAGES",
    "FUNNEL_STAGES",
    "STAGES",
    "STAGE_PROGRESS",
    "STALE_DAYS",
    "TERMINAL_STAGES",
    "Application",
    "CareerError",
    "CareerStore",
    "careers_path",
]

#: The stage funnel, in order.  ``accepted`` continues the funnel
#: past ``offer``; ``rejected`` / ``withdrawn`` close it sideways.
STAGES: tuple[str, ...] = (
    "saved", "applied", "screen", "interview", "offer",
    "accepted", "rejected", "withdrawn",
)

#: Stages on the success path, in funnel order (for progress
#: weights and "reached at least" computations).
FUNNEL_STAGES: tuple[str, ...] = (
    "saved", "applied", "screen", "interview", "offer", "accepted",
)

#: Stages where the application is still live.
ACTIVE_STAGES: tuple[str, ...] = (
    "saved", "applied", "screen", "interview", "offer",
)

#: Stages that end the application; it never moves again.
TERMINAL_STAGES: tuple[str, ...] = ("accepted", "rejected", "withdrawn")

#: Measured pipeline progress per stage — the provider's weights.
#: A saved application is intent (10); submitting is the first
#: real step (30); each funnel stage is worth 20; an accepted
#: offer is the whole pipeline (100).  A rejected/withdrawn
#: application measures 0 for a goal linked to it — it closed
#: without succeeding, and the reading says so in its headline.
STAGE_PROGRESS: dict[str, float] = {
    "saved": 10.0,
    "applied": 30.0,
    "screen": 50.0,
    "interview": 70.0,
    "offer": 90.0,
    "accepted": 100.0,
    "rejected": 0.0,
    "withdrawn": 0.0,
}

#: An active application with no next action scheduled and no
#: activity for this many days is stale — the board calls it out
#: instead of listing it silently.
STALE_DAYS = 14


class CareerError(ValueError):
    """A pipeline operation refused in words (bad stage, closed
    application, unknown application).  A ValueError so surfaces
    handle it like the goal store's input errors."""


def careers_path(settings: Any = None) -> Path:
    """The pipeline store path, honoring the runtime settings home.

    Mirrors :func:`nomorals.finance.budgets.finance_paths`: the
    settings home when one is in force, else ``~/.nomorals``.
    """
    if settings is not None:
        home = getattr(settings, "home_path", None)
        if home is not None:
            return Path(home) / "careers" / "applications.json"
    return Path.home() / ".nomorals" / "careers" / "applications.json"


@dataclass
class Application:
    """One application in the pipeline.

    ``history`` is append-only: every stage the application has
    occupied, with when and an optional note.  The current stage
    is the last entry's stage (mirrored on ``stage`` for cheap
    reads); the history is the audit trail the provider's
    "reached at least applied" arithmetic reads.
    """

    id: str
    company: str
    role: str
    stage: str = "saved"
    source: str = ""          # where the opening was found
    link: str = ""            # the posting / application URL
    history: list[dict[str, Any]] = field(default_factory=list)
    notes: list[dict[str, Any]] = field(default_factory=list)
    next_action: str = ""
    next_action_due: float | None = None
    cv_artifact_id: str = ""  # an artifact-store id (the CV sent)
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "company": self.company,
            "role": self.role,
            "stage": self.stage,
            "source": self.source,
            "link": self.link,
            "history": list(self.history),
            "notes": list(self.notes),
            "next_action": self.next_action,
            "next_action_due": self.next_action_due,
            "cv_artifact_id": self.cv_artifact_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Application:
        history = [dict(h) for h in data.get("history") or []
                   if isinstance(h, dict)]
        stage = str(data.get("stage", "saved") or "saved")
        if stage not in STAGES:
            stage = "saved"
        return cls(
            id=str(data.get("id", "")),
            company=str(data.get("company", "")),
            role=str(data.get("role", "")),
            stage=stage,
            source=str(data.get("source", "") or ""),
            link=str(data.get("link", "") or ""),
            history=history,
            notes=[dict(n) for n in data.get("notes") or []
                   if isinstance(n, dict)],
            next_action=str(data.get("next_action", "") or ""),
            next_action_due=(float(data["next_action_due"])
                             if data.get("next_action_due") else None),
            cv_artifact_id=str(data.get("cv_artifact_id", "") or ""),
            created_at=float(data.get("created_at", 0) or 0),
            updated_at=float(data.get("updated_at", 0) or 0),
        )

    # ── derived facts ────────────────────────────────────────────
    def reached_rank(self) -> int:
        """Highest funnel rank this application has EVER reached.

        From the append-only history, not the current stage: an
        application rejected after interviewing still *reached*
        the interview stage, and an application that was applied
        then withdrawn was still submitted.
        """
        best = FUNNEL_STAGES.index(self.stage) \
            if self.stage in FUNNEL_STAGES else -1
        for entry in self.history:
            stage = entry.get("stage")
            if stage in FUNNEL_STAGES:
                best = max(best, FUNNEL_STAGES.index(stage))
        return best

    def reached(self, stage: str) -> bool:
        """Whether the application ever reached ``stage`` or beyond."""
        return self.reached_rank() >= FUNNEL_STAGES.index(stage)

    def stage_since(self) -> float:
        """When the current stage began (last history entry)."""
        if self.history:
            try:
                return float(self.history[-1].get("at") or 0.0)
            except (TypeError, ValueError):
                return self.updated_at
        return self.created_at

    @property
    def closed(self) -> bool:
        return self.stage in TERMINAL_STAGES


class CareerStore:
    """Persistent application registry.  JSON file, finance-shape."""

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            path = careers_path()
        self.path = Path(path)
        #: Set when the last _load found the file unreadable; while
        #: set, _save refuses rather than overwriting the quarantined
        #: bytes with an empty-image rewrite (data-loss family guard,
        #: the finance GoalStore precedent).
        self._unreadable: BaseException | None = None

    # ── persistence ──────────────────────────────────────────────
    def _load(self) -> dict[str, dict[str, Any]]:
        data, exc = read_json_file(
            self.path, site="careers.store._load", logger=_log)
        if exc is not None:
            self._unreadable = exc
            return {}
        if data is not None:
            # A real parse heals the flag.  A missing file does NOT:
            # its absence may be our own quarantine, so only a fresh
            # instance (or a restored file) restarts this store.
            self._unreadable = None
            apps = data.get("applications")
            return apps if isinstance(apps, dict) else {}
        return {}

    def _save(self, apps: dict[str, dict[str, Any]]) -> None:
        unreadable = getattr(self, "_unreadable", None)
        if unreadable is not None:
            raise StoreUnreadableError(
                f"career store {self.path} is unreadable "
                f"({unreadable}); original bytes quarantined to "
                f"{self.path.name}.corrupt — refusing to overwrite them")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"version": 1, "applications": apps}, indent=1),
            encoding="utf-8")
        tmp.replace(self.path)

    def _mutate(self, app: Application) -> Application:
        apps = self._load()
        apps[app.id] = app.to_dict()
        self._save(apps)
        return app

    def _require(self, app_id: str) -> Application:
        app = self.get(app_id)
        if app is None:
            raise CareerError(f"unknown application {app_id!r}")
        return app

    # ── CRUD ─────────────────────────────────────────────────────
    def add(
        self,
        company: str,
        role: str,
        *,
        source: str = "",
        link: str = "",
        stage: str = "saved",
        next_action: str = "",
        next_action_due: float | None = None,
        now: float | None = None,
    ) -> Application:
        """File a new application; its history opens at ``stage``."""
        company = (company or "").strip()
        role = (role or "").strip()
        if not company:
            raise CareerError("an application needs a company")
        if not role:
            raise CareerError("an application needs a role")
        if stage not in STAGES:
            raise CareerError(
                f"unknown stage {stage!r} (stages: {', '.join(STAGES)})")
        ts = time.time() if now is None else now
        app = Application(
            id="app_" + uuid.uuid4().hex[:12],
            company=company,
            role=role,
            stage=stage,
            source=(source or "").strip(),
            link=(link or "").strip(),
            history=[{"stage": stage, "at": ts, "note": "filed"}],
            next_action=(next_action or "").strip(),
            next_action_due=next_action_due,
            created_at=ts,
            updated_at=ts,
        )
        return self._mutate(app)

    def get(self, app_id: str) -> Application | None:
        raw = self._load().get(app_id or "")
        if not raw:
            return None
        try:
            return Application.from_dict(raw)
        except Exception:  # noqa: BLE001
            return None

    def list(self, *, include_closed: bool = True) -> list[Application]:
        """Every application, board order (filed order)."""
        out = []
        for raw in self._load().values():
            try:
                app = Application.from_dict(raw)
            except Exception:  # noqa: BLE001
                continue
            if app.closed and not include_closed:
                continue
            out.append(app)
        return sorted(out, key=lambda a: a.created_at)

    def remove(self, app_id: str) -> bool:
        apps = self._load()
        if app_id not in apps:
            return False
        del apps[app_id]
        self._save(apps)
        return True

    # ── pipeline movement ────────────────────────────────────────
    def move(
        self,
        app_id: str,
        stage: str,
        *,
        note: str = "",
        now: float | None = None,
    ) -> Application:
        """Move an application to ``stage``, appending to its history.

        An unknown stage is refused in words; a terminal application
        does not move (retrying a closed application is a new
        application).  Moving to the current stage is refused too —
        the history records real movement, not noise.
        """
        if stage not in STAGES:
            raise CareerError(
                f"unknown stage {stage!r} (stages: {', '.join(STAGES)})")
        app = self._require(app_id)
        if app.closed:
            raise CareerError(
                f"{app.company} — {app.role} is closed ({app.stage}); "
                "it can't move again — file a new application to retry")
        if stage == app.stage:
            raise CareerError(
                f"{app.company} — {app.role} is already at {stage}")
        ts = time.time() if now is None else now
        app.history.append(
            {"stage": stage, "at": ts, "note": (note or "").strip()})
        app.stage = stage
        app.updated_at = ts
        if app.closed:
            # A closed application has no next action.
            app.next_action = ""
            app.next_action_due = None
        return self._mutate(app)

    def set_next_action(
        self,
        app_id: str,
        text: str,
        due: float | None = None,
        *,
        now: float | None = None,
    ) -> Application:
        """Set (or clear, with empty text) the next action + due date."""
        app = self._require(app_id)
        if app.closed:
            raise CareerError(
                f"{app.company} — {app.role} is closed ({app.stage}); "
                "closed applications have no next action")
        app.next_action = (text or "").strip()
        app.next_action_due = due if app.next_action else None
        app.updated_at = time.time() if now is None else now
        return self._mutate(app)

    def add_note(
        self, app_id: str, text: str, *, now: float | None = None,
    ) -> Application:
        """Append a timestamped note (notes are append-only too)."""
        text = (text or "").strip()
        if not text:
            raise CareerError("a note needs text")
        app = self._require(app_id)
        ts = time.time() if now is None else now
        app.notes.append({"at": ts, "text": text})
        app.updated_at = ts
        return self._mutate(app)

    def attach_cv(
        self, app_id: str, artifact_id: str, *, now: float | None = None,
    ) -> Application:
        """Record which artifact (a CV / cover letter) went with this
        application.  The bytes live in the artifact store; the
        pipeline keeps the reference, so versions stay the artifact
        store's job and this store never copies documents."""
        app = self._require(app_id)
        app.cv_artifact_id = (artifact_id or "").strip()
        app.updated_at = time.time() if now is None else now
        return self._mutate(app)

    # ── derived views ────────────────────────────────────────────
    def funnel_counts(self) -> dict[str, int]:
        """Applications at each stage right now (all stages named)."""
        counts = {stage: 0 for stage in STAGES}
        for app in self.list():
            counts[app.stage] = counts.get(app.stage, 0) + 1
        return counts

    def submitted_since(self, since: float) -> int:
        """Applications filed at/after ``since`` that were actually
        submitted (their history reached ``applied`` or beyond)."""
        n = 0
        for app in self.list():
            if app.created_at >= since and app.reached("applied"):
                n += 1
        return n

    def by_company(self) -> dict[str, list[Application]]:
        """The per-employer records: applications grouped by company
        (normalised name), each with its full stage history — the
        careers class's memory shape beside the goal."""
        out: dict[str, list[Application]] = {}
        for app in self.list():
            out.setdefault(app.company.casefold(), []).append(app)
        return out

    def overdue_actions(self, *, now: float) -> list[Application]:
        """Active applications whose next action is past due."""
        out = []
        for app in self.list(include_closed=False):
            if (app.next_action_due is not None
                    and app.next_action_due < now):
                out.append(app)
        return out

    def due_soon_actions(
        self, *, now: float, within: float = 48 * 3600,
    ) -> list[Application]:
        """Active applications with a next action due within ``within``."""
        out = []
        for app in self.list(include_closed=False):
            if (app.next_action_due is not None
                    and now <= app.next_action_due <= now + within):
                out.append(app)
        return out

    def stale(self, *, now: float,
              days: int = STALE_DAYS) -> list[Application]:
        """Active applications sitting untouched: no next action
        scheduled (a future due date is a plan, not staleness) and
        no activity for ``days`` days."""
        cutoff = now - days * 86400
        out = []
        for app in self.list(include_closed=False):
            if app.next_action_due is not None:
                continue
            if app.updated_at < cutoff:
                out.append(app)
        return out

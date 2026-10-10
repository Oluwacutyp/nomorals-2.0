"""Gig application drafting and submission pipeline.

Takes opportunities from ``nomorals.agents.opportunities`` and runs the
full application loop: draft → user review → submit → status tracking.

The explicit-override rule (user directive 2026-10-07): if the user
explicitly says "submit" / "apply now" / "send it", the application is
submitted immediately.  No re-confirmation.  No hesitation.  Explicit
instruction = execute.

Status pipeline: drafted → reviewed → submitted → interview | rejected | accepted
                                       ↘ withdrawn

Usage (agent code)::

    from nomorals.agents.gig_applier import GigApplier
    applier = GigApplier()
    app = applier.draft(opportunity, profile)   # LLM draft, status=drafted
    applier.submit(app.id, explicit=True)        # user said "submit" → goes now

Chat surface: ``/money apply <gig_id>`` (draft), ``/money apply <gig_id> submit``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from ..llm.brain import brain_for
from ..core.logging_setup import get_logger
from .opportunities import Opportunity

_log = get_logger(__name__)

__all__ = [
    "APPLICATION_STATUSES",
    "Application",
    "ApplicationStore",
    "GigApplier",
    "EXPLICIT_SUBMIT_WORDS",
    "is_explicit_submit",
]

#: Valid application statuses, in pipeline order.
APPLICATION_STATUSES = (
    "drafted",
    "reviewed",
    "submitted",
    "interview",
    "accepted",
    "rejected",
    "withdrawn",
)

#: Words that count as an explicit submit instruction (user directive).
EXPLICIT_SUBMIT_WORDS = frozenset({
    "submit", "apply now", "send it", "send the application",
    "submit it", "go ahead", "do it",
})


def is_explicit_submit(text: str) -> bool:
    """True when the user's text is an explicit submit instruction."""
    t = (text or "").lower().strip()
    if not t:
        return False
    return any(w in t for w in EXPLICIT_SUBMIT_WORDS)


@dataclass
class Application:
    gig_id: str
    title: str
    url: str
    source: str = ""
    kind: str = "paid_task"
    payout_text: str = ""
    status: str = "drafted"
    draft_text: str = ""
    submitted_at: Optional[float] = None
    drafted_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    notes: str = ""

    def __post_init__(self) -> None:
        if self.status not in APPLICATION_STATUSES:
            self.status = "drafted"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Application":
        d = dict(d)
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})

    def transition(self, new_status: str) -> None:
        if new_status not in APPLICATION_STATUSES:
            raise ValueError(f"unknown status {new_status!r}")
        self.status = new_status
        self.updated_at = time.time()
        if new_status == "submitted":
            self.submitted_at = time.time()


class ApplicationStore:
    """JSONL-backed application storage (one line per application)."""

    def __init__(self, settings: Any = None) -> None:
        self.settings = settings
        self._data_dir: Optional[Path] = None

    @property
    def data_dir(self) -> Path:
        if self._data_dir is None:
            if self.settings is not None and hasattr(self.settings, "resolve"):
                self._data_dir = Path(self.settings.resolve("data/opportunities"))
            else:
                self._data_dir = Path.home() / ".config" / "nomorals" / "opportunities"
            self._data_dir.mkdir(parents=True, exist_ok=True)
        return self._data_dir

    def _path(self) -> Path:
        return self.data_dir / "applications.jsonl"

    def load_all(self) -> dict[str, Application]:
        apps: dict[str, Application] = {}
        p = self._path()
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    a = Application.from_dict(json.loads(line))
                    apps[a.gig_id] = a
                except Exception:  # noqa: BLE001 - skip corrupt lines
                    continue
        return apps

    def save(self, app: Application) -> None:
        apps = self.load_all()
        apps[app.gig_id] = app
        lines = [json.dumps(a.to_dict()) for a in apps.values()]
        self._path().write_text("\n".join(lines) + "\n", encoding="utf-8")

    def get(self, gig_id: str) -> Optional[Application]:
        return self.load_all().get(gig_id)

    def list_by_status(self, status: Optional[str] = None,
                       limit: int = 30) -> list[Application]:
        apps = sorted(self.load_all().values(),
                      key=lambda a: a.updated_at, reverse=True)
        if status:
            apps = [a for a in apps if a.status == status]
        return apps[:limit]


_DRAFT_PROMPT = """You are drafting a job/gig application for the user.

Opportunity:
- Title: {title}
- Source: {source}
- URL: {url}
- Payout: {payout}
- Kind: {kind}

User profile:
- Skills: {skills}
- Regions: {regions}

Write a concise, professional application (150-250 words).  Lead with the
most relevant skill match.  No placeholders like [Your Name] — write it
ready to send, in first person.  End with a clear call to action.
"""


class GigApplier:
    """Drafts, tracks, and submits gig applications."""

    def __init__(self, settings: Any = None,
                 llm_fn: Any = None) -> None:
        self.settings = settings
        # llm_fn(prompt) -> str — injected for tests; falls back to router.
        self.llm_fn = llm_fn
        self.store = ApplicationStore(settings=settings)

    def _llm(self, prompt: str, context: Any = None) -> str:
        if self.llm_fn is not None:
            return self.llm_fn(prompt)
        router = getattr(context, "router", None) if context else None
        if router is None:
            from ..core.config import get_settings
            settings = self.settings or get_settings()
            router = getattr(settings, "router", None)
        if router is None:
            raise RuntimeError("no LLM available for drafting — pass llm_fn")
        resp = brain_for(context).complete(prompt, task_kind="chat")
        return resp.text if hasattr(resp, "text") else str(resp)

    def draft(self, opp: Opportunity, profile: Any = None,
              context: Any = None) -> Application:
        """Draft an application via LLM.  Status → drafted."""
        skills = ", ".join(getattr(profile, "skills", []) or []) or "general"
        regions = ", ".join(getattr(profile, "regions", []) or []) or "global"
        prompt = _DRAFT_PROMPT.format(
            title=opp.title, source=opp.source, url=opp.url,
            payout=opp.payout_text or "not specified",
            kind=opp.kind, skills=skills, regions=regions,
        )
        draft_text = self._llm(prompt, context).strip()
        app = Application(
            gig_id=opp.id, title=opp.title, url=opp.url,
            source=opp.source, kind=opp.kind,
            payout_text=opp.payout_text, draft_text=draft_text,
        )
        self.store.save(app)
        _log.info("drafted application for %s", opp.id)
        return app

    def review(self, gig_id: str) -> Optional[Application]:
        """Mark as reviewed (user has seen the draft)."""
        app = self.store.get(gig_id)
        if app is None:
            return None
        if app.status == "drafted":
            app.transition("reviewed")
            self.store.save(app)
        return app

    def submit(self, gig_id: str, explicit: bool = False) -> Application:
        """Submit an application.

        When ``explicit`` is True (user said "submit"/"apply now"), the
        application goes immediately — no re-confirmation, per the
        explicit-override directive.  Otherwise the caller is expected
        to have obtained confirmation already.
        """
        app = self.store.get(gig_id)
        if app is None:
            raise KeyError(f"no application for gig {gig_id!r} — draft first")
        # The actual submission mechanism (browser automation / API) is
        # per-board.  For now we record the submit intent with timestamp;
        # board-specific submitters plug in here.
        app.transition("submitted")
        self.store.save(app)
        _log.info("submitted application for %s (explicit=%s)", gig_id, explicit)
        try:
            from ..cognition.representation import log_representation_action
            log_representation_action(
                "apply", f"gig application submitted: {gig_id}",
                settings=self.settings,
                metadata={"gig_id": gig_id, "explicit": explicit})
        except Exception:  # noqa: BLE001 — ledger never breaks the action
            pass
        return app

    def set_status(self, gig_id: str, status: str,
                   notes: str = "") -> Optional[Application]:
        app = self.store.get(gig_id)
        if app is None:
            return None
        app.transition(status)
        if notes:
            app.notes = notes
        self.store.save(app)
        return app

    def render(self, apps: list[Application], limit: int = 15) -> str:
        lines = [f"📝 applications ({len(apps)} shown)"]
        for a in apps[:limit]:
            when = time.strftime("%m-%d", time.localtime(a.updated_at))
            lines.append(f"[{a.status}] {a.title} ({when})\n    ↳ {a.url}")
        if len(apps) > limit:
            lines.append(f"…and {len(apps) - limit} more")
        return "\n".join(lines)

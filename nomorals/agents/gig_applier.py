"""Gig application drafting and submission pipeline.

Takes opportunities from ``nomorals.agents.opportunities`` and runs the
full application loop: draft → user review → submit → status tracking →
follow-up.

The explicit-override rule (user directive 2026-10-07): if the user
explicitly says "submit" / "apply now" / "send it", the application is
submitted immediately.  No re-confirmation.  No hesitation.  Explicit
instruction = execute.

Honesty rule for submission: most gig boards (Upwork, Outlier, Mindrift)
are human-gated and account-bound — there is no universal "click submit"
that works everywhere.  So ``submit()`` only reports ``submitted`` when a
registered board submitter actually completed a submission and returned
evidence.  When no automated path exists, the application moves to
``submit_attempted`` with a concrete manual checklist and the application
URL surfaced to the owner — the record says what really happened.

Status pipeline: drafted → reviewed → submit_attempted → submitted
                              ↘ needs_human ↗        → interview | rejected | accepted
                                                       ↘ withdrawn

Board-specific submitters plug in via :func:`register_submitter`::

    from nomorals.agents.gig_applier import register_submitter

    def upwork_submit(app, draft_text):
        ...  # drive the real submission
        return {"ok": True, "method": "browser",
                "evidence": "application id 12345", "detail": "..."}

    register_submitter("upwork", upwork_submit)

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
from typing import Any, Callable, Optional

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
    "register_submitter",
    "SUBMITTERS",
]

#: Valid application statuses, in pipeline order.  ``submit_attempted``
#: means a submission was tried but no automated path completed it —
#: honest, not "submitted".  ``needs_human`` means the board wants a
#: human step (login wall, extra questions) before it can go through.
APPLICATION_STATUSES = (
    "drafted",
    "reviewed",
    "submit_attempted",
    "needs_human",
    "submitted",
    "interview",
    "accepted",
    "rejected",
    "withdrawn",
)

#: Allowed status jumps.  Outcomes (interview/accepted/rejected) are only
#: reachable from ``submitted``; nothing moves backwards except back to
#: ``reviewed`` for another pass.  ``set_status`` from chat uses these too,
#: so tracking stays truthful.
_VALID_TRANSITIONS: dict[str, frozenset[str]] = {
    # drafted can go straight to submit: explicit "submit now" drafts and
    # submits in one go without a review pass.  A submit attempt can also
    # surface a human wall immediately (needs_human).
    "drafted": frozenset({"reviewed", "submit_attempted", "needs_human",
                          "submitted", "withdrawn"}),
    "reviewed": frozenset({"drafted", "submit_attempted", "needs_human",
                           "submitted", "withdrawn"}),
    "submit_attempted": frozenset({"submitted", "needs_human", "reviewed",
                                   "withdrawn"}),
    "needs_human": frozenset({"submitted", "reviewed", "withdrawn"}),
    "submitted": frozenset({"interview", "accepted", "rejected",
                             "withdrawn"}),
    "interview": frozenset({"accepted", "rejected", "withdrawn"}),
    "accepted": frozenset(),
    "rejected": frozenset(),
    "withdrawn": frozenset(),
}

#: Default follow-up delay after a submission with no news.
FOLLOW_UP_DELAY_SECONDS = 7 * 24 * 3600

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
    #: how the (attempted) submission happened — board, method,
    #: timestamp, human-readable note.  Never carries credentials.
    submission_evidence: dict[str, Any] = field(default_factory=dict)
    #: when to nudge about this application if nothing changed.
    follow_up_at: Optional[float] = None
    #: append-only trail of status changes: {ts, from, to, note}.
    history: list[dict[str, Any]] = field(default_factory=list)

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

    def record_event(self, kind: str, note: str = "") -> None:
        self.history.append({"ts": time.time(), "kind": kind,
                             "note": note[:500]})

    def transition(self, new_status: str) -> None:
        if new_status not in APPLICATION_STATUSES:
            raise ValueError(f"unknown status {new_status!r}")
        allowed = _VALID_TRANSITIONS.get(self.status, frozenset())
        if new_status != self.status and new_status not in allowed:
            raise ValueError(
                f"invalid transition {self.status!r} → {new_status!r} "
                f"(allowed: {sorted(allowed) or 'none'})")
        old = self.status
        self.status = new_status
        self.updated_at = time.time()
        if new_status == "submitted":
            self.submitted_at = time.time()
            self.follow_up_at = time.time() + FOLLOW_UP_DELAY_SECONDS
        if old != new_status:
            self.record_event("status", f"{old} → {new_status}")


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


#: Board-specific submitters, keyed by lowercased source name then kind.
#: A submitter is ``fn(app, draft_text) -> dict`` and must perform the
#: real submission, returning ``{"ok": bool, "method": str,
#: "evidence": str, "detail": str, "needs_human": bool}``.  ``evidence``
#: is a human-readable proof (confirmation id, screenshot path, posted
#: URL) — never credentials.  Empty by default: boards are human-gated
#: and account-bound, so each real submitter is a deliberate plug-in,
#: not a guess.
SUBMITTERS: dict[str, Callable[..., dict[str, Any]]] = {}


def register_submitter(name: str,
                       fn: Callable[..., dict[str, Any]]) -> None:
    """Plug in a real per-board submitter (see module docstring)."""
    key = (name or "").strip().lower()
    if not key:
        raise ValueError("submitter name is required")
    if not callable(fn):
        raise ValueError("submitter must be callable")
    SUBMITTERS[key] = fn


def _submitter_for(app: "Application") -> Optional[Callable[..., dict[str, Any]]]:
    """Find a registered submitter for this application, if any."""
    for key in ((app.source or "").lower(), (app.kind or "").lower()):
        if key and key in SUBMITTERS:
            return SUBMITTERS[key]
    return None


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


_REVISE_PROMPT = """You are revising a job/gig application draft. Be ruthless.

Opportunity: {title} ({source}) — {url}

Current draft:
---
{draft}
---

Feedback: {feedback}

Return ONLY the revised application text (150-250 words), ready to send,
first person, no placeholders, no commentary about the revision.
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
              context: Any = None, force: bool = False) -> Application:
        """Draft an application via LLM.  Status → drafted.

        Refuses to silently overwrite an in-flight application for the
        same gig — returns the existing one unless ``force=True``.
        """
        existing = self.store.get(opp.id)
        if (existing is not None and not force
                and existing.status not in ("withdrawn", "rejected")):
            _log.info("draft skipped for %s: %s already in flight",
                      opp.id, existing.status)
            return existing
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
        app.record_event("drafted", f"for {opp.source}")
        self.store.save(app)
        _log.info("drafted application for %s", opp.id)
        return app

    def revise(self, gig_id: str, feedback: str = "",
               context: Any = None) -> Application:
        """Second pass on a draft: critique then tighten.

        ``feedback`` is the owner's notes (or empty for a self-critique
        pass).  Status is unchanged; the revision is recorded in history.
        """
        app = self.store.get(gig_id)
        if app is None:
            raise KeyError(f"no application for gig {gig_id!r} — draft first")
        if app.status not in ("drafted", "reviewed"):
            raise ValueError(
                f"cannot revise a {app.status} application — "
                "draft a new one instead")
        prompt = _REVISE_PROMPT.format(
            title=app.title, source=app.source, url=app.url,
            draft=app.draft_text,
            feedback=feedback.strip() or
            "(no owner feedback — self-critique: tighten, cut filler, "
            "strengthen the skill match and the call to action)",
        )
        revised = self._llm(prompt, context).strip()
        if revised:
            app.draft_text = revised
            app.record_event("revised",
                             feedback[:200] or "self-critique pass")
            self.store.save(app)
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
        """Submit an application — honestly.

        When ``explicit`` is True (user said "submit"/"apply now"), the
        application goes immediately — no re-confirmation, per the
        explicit-override directive.  Otherwise the caller is expected
        to have obtained confirmation already.

        The honesty rule: ``submitted`` is only set when a registered
        board submitter actually completed the submission and returned
        evidence.  When no automated path exists for the board, the
        application moves to ``submit_attempted`` with a manual
        checklist and the application URL — the record says what really
        happened instead of pretending the submit went through.
        """
        app = self.store.get(gig_id)
        if app is None:
            raise KeyError(f"no application for gig {gig_id!r} — draft first")
        if app.status == "submitted":
            _log.info("submit skipped for %s: already submitted", gig_id)
            return app
        if app.status not in ("drafted", "reviewed", "submit_attempted",
                              "needs_human"):
            raise ValueError(
                f"cannot submit a {app.status} application")

        submitter = _submitter_for(app)
        if submitter is None:
            # No automated path for this board — honest manual handoff.
            app.transition("submit_attempted")
            app.submission_evidence = {
                "board": app.source,
                "method": "manual",
                "at": time.time(),
                "note": ("no automated submitter registered for this "
                         "board — owner completes the submission"),
                "next_steps": self._manual_steps(app),
            }
            app.record_event("submit_attempted", "no board submitter")
            self.store.save(app)
            _log.info("submit attempted (manual) for %s", gig_id)
            return app

        try:
            result = submitter(app, app.draft_text) or {}
        except Exception as exc:  # noqa: BLE001 - record, don't pretend
            _log.exception("board submitter failed for %s", gig_id)
            app.transition("submit_attempted")
            app.submission_evidence = {
                "board": app.source,
                "method": getattr(submitter, "__name__", "submitter"),
                "at": time.time(),
                "note": f"submitter raised: {exc}",
                "next_steps": self._manual_steps(app),
            }
            app.record_event("submit_attempted", f"submitter error: {exc}"[:200])
            self.store.save(app)
            return app

        evidence = {
            "board": app.source,
            "method": str(result.get("method") or
                          getattr(submitter, "__name__", "submitter")),
            "at": time.time(),
            "note": str(result.get("evidence") or result.get("detail") or ""),
        }
        app.submission_evidence = evidence
        if result.get("ok"):
            app.transition("submitted")
            app.record_event("submitted", evidence["note"][:200])
            _log.info("submitted application for %s (explicit=%s)",
                      gig_id, explicit)
        elif result.get("needs_human"):
            app.transition("needs_human")
            app.submission_evidence["next_steps"] = self._manual_steps(app)
            app.record_event("needs_human", evidence["note"][:200])
            _log.info("submit needs human for %s", gig_id)
        else:
            app.transition("submit_attempted")
            app.submission_evidence["next_steps"] = self._manual_steps(app)
            app.record_event("submit_attempted", evidence["note"][:200])
            _log.info("submit attempted (not completed) for %s", gig_id)
        self.store.save(app)
        try:
            from ..cognition.representation import log_representation_action
            log_representation_action(
                "apply", f"gig application {app.status}: {gig_id}",
                settings=self.settings,
                metadata={"gig_id": gig_id, "explicit": explicit,
                          "status": app.status})
        except Exception:  # noqa: BLE001 — ledger never breaks the action
            pass
        return app

    @staticmethod
    def _manual_steps(app: Application) -> list[str]:
        """Concrete checklist for completing the submission by hand."""
        return [
            f"open the application page: {app.url}",
            "paste the draft below into the application form",
            "answer any board-specific questions (availability, rate, portfolio links)",
            "confirm the submission on the site, then run: "
            f"/money applications  (and tell me the outcome so I can track it)",
        ]

    def withdraw(self, gig_id: str, notes: str = "") -> Optional[Application]:
        """Withdraw an application (terminal)."""
        app = self.store.get(gig_id)
        if app is None:
            return None
        app.transition("withdrawn")
        if notes:
            app.notes = notes
        app.record_event("withdrawn", notes[:200])
        self.store.save(app)
        return app

    # ── follow-ups: tracking doesn't end at submit ────────────────────────
    def follow_ups_due(self, now: float | None = None) -> list[Application]:
        """Applications waiting on news past their follow-up time.

        Covers ``submitted`` (with ``follow_up_at`` set at submit time),
        ``submit_attempted`` and ``needs_human`` (owner was handed a
        checklist — nudge if it was never completed).
        """
        now = time.time() if now is None else now
        due = []
        for app in self.store.load_all().values():
            if app.status not in ("submitted", "submit_attempted",
                                  "needs_human"):
                continue
            at = app.follow_up_at
            if at is None and app.status != "submitted":
                # manual-handoff states without a timer: use updated_at.
                at = app.updated_at + FOLLOW_UP_DELAY_SECONDS
            if at is not None and at <= now:
                due.append(app)
        return sorted(due, key=lambda a: a.follow_up_at or 0)

    def snooze_follow_up(self, gig_id: str,
                         days: float = 7) -> Optional[Application]:
        """Push the follow-up nudge out by ``days``."""
        app = self.store.get(gig_id)
        if app is None:
            return None
        app.follow_up_at = time.time() + max(0.5, days) * 86400
        app.record_event("follow_up_snoozed", f"+{days}d")
        self.store.save(app)
        return app

    def follow_up_draft(self, gig_id: str,
                        context: Any = None) -> str:
        """Draft a follow-up message for a submitted application.

        Uses the LLM when available, otherwise an offline template —
        follow-ups must work without a model.
        """
        app = self.store.get(gig_id)
        if app is None:
            raise KeyError(f"no application for gig {gig_id!r}")
        when = time.strftime(
            "%Y-%m-%d", time.localtime(app.submitted_at or app.updated_at))
        template = (
            f"Subject: Following up — {app.title}\n\n"
            "Hi,\n\n"
            f"I'm writing to follow up on my application for \"{app.title}\" "
            f"({app.source}), submitted on {when}.\n\n"
            "I'm still very interested and would welcome the chance to "
            "discuss how I can help. Happy to share work samples or jump "
            "on a quick call.\n\n"
            "Thanks for your time,\n"
        )
        prompt = (
            "Write a short, professional follow-up message (80-120 words) "
            f"for a job application titled \"{app.title}\" at {app.source}, "
            f"submitted on {when}. The current draft of the original "
            f"application was:\n---\n{app.draft_text[:1500]}\n---\n"
            "First person, ready to send, no placeholders."
        )
        try:
            text = self._llm(prompt, context).strip()
        except Exception:  # noqa: BLE001 - offline template fallback
            text = ""
        if not text:
            text = template
        app.record_event("follow_up_drafted", "")
        self.store.save(app)
        return text

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

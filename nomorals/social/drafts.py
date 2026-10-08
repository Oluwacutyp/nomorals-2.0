"""Agent-drafts with the explicit-override rule (build-map #43).

The user's standing directive (2026-10-07), encoded exactly:

- DEFAULT: Devon drafts a post → asks "post directly or send for review?"
- OVERRIDE: the user explicitly says post ("post at 6 for engagement",
  "post this", "send it", "publish now", "reply now") → JUST POST IT.
  No re-confirmation. No hesitation. No reluctance. No policy text.
  Explicit instruction = execute.

The override is implemented by minting a real policy confirmation token
for the publish capability — the user's own words are the authorization,
and the policy layer records it in the audit log like any other
confirmation. This is not a bypass of the policy system; it IS the
policy system, fed by the owner's explicit intent.

Never posts without EITHER explicit owner intent OR review approval
(button tap / /approve). Drafts are SQLite-backed and owner-scoped.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "Draft",
    "DraftQueue",
    "detect_explicit_post",
    "execute_post",
    "propose_post",
    "parse_post_time",
    "EXPLICIT_POST_PATTERNS",
]

# ── statuses ────────────────────────────────────────────────────────────────

DRAFT = "draft"
PENDING_REVIEW = "pending_review"
APPROVED = "approved"
SCHEDULED = "scheduled"
POSTED = "posted"
DISCARDED = "discarded"

_VALID_TRANSITIONS = {
    DRAFT: {PENDING_REVIEW, DISCARDED},
    PENDING_REVIEW: {APPROVED, SCHEDULED, DISCARDED, DRAFT},
    APPROVED: {POSTED, SCHEDULED, DISCARDED},
    SCHEDULED: {POSTED, DISCARDED, PENDING_REVIEW},
    POSTED: set(),
    DISCARDED: set(),
}


@dataclass
class Draft:
    """One post draft moving through draft → review → posted."""

    id: str
    content: str
    platforms: list[str] = field(default_factory=list)
    scheduled_time: float | None = None
    status: str = DRAFT
    created_at: float = field(default_factory=time.time)
    posted_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ── explicit-post intent detection ────────────────────────────────────────────
# The standing directive: explicit "post", "submit", "apply now", "send it",
# or "reply now" means execute without reconfirming. These patterns detect the
# *posting* subset. Bare nouns ("read that post") must NOT match — every
# pattern requires a verb-like command form.

EXPLICIT_POST_PATTERNS: list[tuple[str, str]] = [
    # (regex, action) — action is "post" or "schedule"
    (r"\bpost\s+this\b", "post"),
    (r"\bpost\s+it\b", "post"),
    (r"\bpost\s+that\b", "post"),
    (r"\bpost\s+now\b", "post"),
    (r"\bpublish\s+now\b", "post"),
    (r"\bsend\s+it\b", "post"),
    (r"\breply\s+now\b", "post"),
    (r"\bsubmit\b", "post"),
    (r"\bpost\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", "schedule"),
    (r"\bschedule\s+(this|it|that)\b", "schedule"),
    (r"\bpost\s+it\s+for\s+engagement\b", "post"),
    (r"\bpost\s+for\s+engagement\b", "post"),
]

_AT_RE = re.compile(
    r"\bpost\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", re.IGNORECASE)


def parse_post_time(text: str, *, now: datetime | None = None) -> float | None:
    """Parse "post at 6" / "post at 6:30pm" → unix timestamp.

    Bare "at 6" means 6pm when 6pm is still ahead today, else 6am
    tomorrow. Explicit am/pm always wins. Returns None when no time found.
    """
    now = now or datetime.now()
    m = _AT_RE.search(text or "")
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    ampm = (m.group(3) or "").lower()
    if ampm == "pm" and hour < 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    elif not ampm:
        # Bare hour: prefer the upcoming 6pm-style slot; if that's past,
        # fall to the morning slot tomorrow.
        evening = hour + 12 if hour < 12 else hour
        candidate = now.replace(hour=evening % 24, minute=minute,
                                second=0, microsecond=0)
        if candidate > now:
            return candidate.timestamp()
        morning = hour % 12
        candidate = (now + timedelta(days=1)).replace(
            hour=morning, minute=minute, second=0, microsecond=0)
        return candidate.timestamp()
    candidate = now.replace(hour=hour % 24, minute=minute,
                            second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate.timestamp()


def detect_explicit_post(text: str) -> dict[str, Any] | None:
    """Detect explicit-post intent in the owner's message.

    Returns ``{"action": "post"|"schedule", "at": ts|None}`` or None.
    Case-insensitive. The bare noun "post" ("read that post") never matches.
    """
    text = (text or "").strip()
    if not text:
        return None
    for pattern, action in EXPLICIT_POST_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if not m:
            continue
        at = parse_post_time(text) if action == "schedule" else None
        return {"action": action, "at": at, "matched": m.group(0)}
    return None


# ── draft queue ─────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS social_drafts (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    platforms TEXT NOT NULL DEFAULT '[]',
    scheduled_time REAL,
    status TEXT NOT NULL DEFAULT 'draft',
    created_at REAL NOT NULL,
    posted_at REAL,
    metadata TEXT NOT NULL DEFAULT '{}'
)
"""


def _db_path(explicit: str | Path | None) -> Path:
    if explicit:
        return Path(explicit)
    return Path.home() / ".nomorals" / "social_drafts.db"


class DraftQueue:
    """SQLite-backed draft lifecycle. Owner-scoped, never raises on read."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._path = _db_path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self._path))
        self._db.row_factory = sqlite3.Row
        self._db.execute(_SCHEMA)
        self._db.commit()

    # -- CRUD ---------------------------------------------------------------
    def create_draft(self, content: str,
                     platforms: list[str] | None = None,
                     *, metadata: dict[str, Any] | None = None) -> Draft:
        import json as _json
        from ..core.ids import new_id
        draft = Draft(id=new_id(), content=(content or "").strip(),
                      platforms=list(platforms or []),
                      metadata=dict(metadata or {}))
        if not draft.content:
            raise ValueError("draft content is empty")
        self._db.execute(
            "INSERT INTO social_drafts (id, content, platforms, status,"
            " created_at, metadata) VALUES (?,?,?,?,?,?)",
            (draft.id, draft.content, _json.dumps(draft.platforms),
             DRAFT, draft.created_at, _json.dumps(draft.metadata)))
        self._db.commit()
        return draft

    def get(self, draft_id: str) -> Draft | None:
        import json as _json
        row = self._db.execute(
            "SELECT * FROM social_drafts WHERE id = ?",
            (draft_id,)).fetchone()
        if row is None:
            return None
        return Draft(
            id=row["id"], content=row["content"],
            platforms=_json.loads(row["platforms"] or "[]"),
            scheduled_time=row["scheduled_time"], status=row["status"],
            created_at=row["created_at"], posted_at=row["posted_at"],
            metadata=_json.loads(row["metadata"] or "{}"))

    def _transition(self, draft_id: str, to: str) -> Draft:
        draft = self.get(draft_id)
        if draft is None:
            raise KeyError(f"unknown draft {draft_id!r}")
        allowed = _VALID_TRANSITIONS.get(draft.status, set())
        if to not in allowed:
            raise ValueError(
                f"cannot move draft {draft_id!r} from {draft.status!r} to {to!r}")
        self._db.execute(
            "UPDATE social_drafts SET status = ? WHERE id = ?",
            (to, draft_id))
        self._db.commit()
        draft.status = to
        return draft

    def propose(self, draft_id: str) -> Draft:
        """Draft → pending_review (Devon asks: post directly or review?)."""
        return self._transition(draft_id, PENDING_REVIEW)

    def approve(self, draft_id: str) -> Draft:
        """Review approved (button tap / /approve)."""
        return self._transition(draft_id, APPROVED)

    def discard(self, draft_id: str) -> Draft:
        return self._transition(draft_id, DISCARDED)

    def mark_posted(self, draft_id: str) -> Draft:
        draft = self.get(draft_id)
        if draft is None:
            raise KeyError(f"unknown draft {draft_id!r}")
        self._db.execute(
            "UPDATE social_drafts SET status = ?, posted_at = ? WHERE id = ?",
            (POSTED, time.time(), draft_id))
        self._db.commit()
        draft.status = POSTED
        draft.posted_at = time.time()
        return draft

    def schedule(self, draft_id: str, when_ts: float) -> Draft:
        draft = self._transition(draft_id, SCHEDULED)
        self._db.execute(
            "UPDATE social_drafts SET scheduled_time = ? WHERE id = ?",
            (when_ts, draft_id))
        self._db.commit()
        draft.scheduled_time = when_ts
        return draft

    def pending(self) -> list[Draft]:
        rows = self._db.execute(
            "SELECT * FROM social_drafts WHERE status IN ('draft',"
            " 'pending_review', 'approved', 'scheduled')"
            " ORDER BY created_at DESC").fetchall()
        return [self.get(r["id"]) for r in rows if self.get(r["id"])]

    def close(self) -> None:
        try:
            self._db.close()
        except Exception:  # noqa: BLE001
            pass


# ── review UI ─────────────────────────────────────────────────────────────────

def propose_post(draft: Draft) -> tuple[str, list[tuple[str, str]]]:
    """Build the review message: draft preview + the standing question.

    Returns (message_text, buttons) where buttons are (label, callback_data).
    Callback data is namespaced ``draft:<action>:<id>`` for the chat layer.
    """
    preview = draft.content[:280] + ("…" if len(draft.content) > 280 else "")
    platforms = ", ".join(draft.platforms) or "default platforms"
    text = (
        f"📝 Draft ready ({platforms}):\n\n{preview}\n\n"
        f"Post directly or send for review?"
    )
    buttons = [
        ("📮 Post now", f"draft:post:{draft.id}"),
        ("⏰ Schedule", f"draft:schedule:{draft.id}"),
        ("✏️ Edit", f"draft:edit:{draft.id}"),
        ("🗑 Discard", f"draft:discard:{draft.id}"),
    ]
    return text, buttons


# ── the override: execute without reconfirming ────────────────────────────────

def execute_post(
    manager: Any,
    content: str,
    *,
    platforms: list[str] | None = None,
    draft_id: str | None = None,
    queue: DraftQueue | None = None,
    actor: str = "user",
) -> Any:
    """Post IMMEDIATELY on the owner's explicit instruction.

    This is the override path: the user's explicit "post this" / "send it" /
    "post at 6" is the authorization. A real policy confirmation token is
    minted for the publish capability (capability-bound, single-use, audited)
    — no confirmation dialog, no "are you sure?", no policy text shown to
    the user. The user's word is the authorization.

    ``manager`` is a SocialManager; ``platforms`` selects targets.
    When ``draft_id`` + ``queue`` are given, the draft is marked posted.
    Returns the PublishOutcome.
    """
    from ..core.policy import Capability
    targets = list(platforms or [])
    capability = (Capability.SOCIAL_BULK if len(targets) > 1
                  else Capability.SOCIAL_POST)
    confirmation = ""
    policy = getattr(getattr(manager, "context", None), "policy", None)
    if policy is not None and hasattr(policy, "mint_confirmation"):
        try:
            confirmation = policy.mint_confirmation(capability) or ""
        except Exception:  # noqa: BLE001 — fall through without token
            _log.warning("could not mint confirmation for explicit post",
                         exc_info=True)
    outcome = manager.publish(content, platforms=platforms or None,
                              actor=actor, confirmation=confirmation)
    if draft_id and queue is not None:
        try:
            queue.mark_posted(draft_id)
        except Exception:  # noqa: BLE001 — posting already happened
            _log.warning("could not mark draft %s posted", draft_id,
                         exc_info=True)
    _log.info("explicit-post executed (%d platforms, no reconfirmation)",
              len(getattr(outcome, "results", []) or []))
    return outcome


def schedule_post(
    scheduler: Any,
    queue: DraftQueue,
    draft_id: str,
    when_ts: float,
    *,
    platforms: list[str] | None = None,
) -> Any:
    """Schedule a draft via the scheduler. Auto-posts at ``when_ts`` with
    zero further interaction. Registers the ``social.post_draft`` action
    handler if the scheduler supports it."""
    draft = queue.schedule(draft_id, when_ts)

    def _fire(parameters: dict[str, Any]) -> None:
        # Resolved lazily at fire time so the app wires its own manager.
        fire = getattr(scheduler, "_draft_post_fn", None)
        if callable(fire):
            fire(parameters.get("draft_id", draft_id))

    if hasattr(scheduler, "register_action"):
        try:
            scheduler.register_action("social.post_draft", _fire)
        except Exception:  # noqa: BLE001
            _log.debug("register_action failed", exc_info=True)
    task_id = f"post-draft-{draft_id}"
    if hasattr(scheduler, "schedule_once"):
        import asyncio
        import inspect
        coro = scheduler.schedule_once(task_id, when_ts, "social.post_draft",
                                       {"draft_id": draft_id,
                                        "platforms": platforms or []})
        if inspect.isawaitable(coro):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(coro)
                coro = None
        return coro
    raise RuntimeError("scheduler does not support schedule_once")


def handle_draft_callback(
    queue: DraftQueue,
    callback_data: str,
    *,
    manager: Any = None,
    scheduler: Any = None,
    ask_schedule_time: Callable[[str], float] | None = None,
) -> str:
    """Route a ``draft:<action>:<id>`` button tap. Returns a reply string."""
    parts = (callback_data or "").split(":")
    if len(parts) != 3 or parts[0] != "draft":
        return ""
    _, action, draft_id = parts
    draft = queue.get(draft_id)
    if draft is None:
        return "that draft is gone."
    if action == "post":
        if manager is None:
            return "can't post — no publisher wired."
        queue.approve(draft_id)
        outcome = execute_post(manager, draft.content,
                               platforms=draft.platforms or None,
                               draft_id=draft_id, queue=queue)
        posted = len(getattr(outcome, "posted", []) or [])
        return f"posted to {posted} platform(s)."
    if action == "schedule":
        if ask_schedule_time is None:
            return "when should I post it? (e.g. 'post at 6')"
        when = ask_schedule_time(draft_id)
        schedule_post(scheduler, queue, draft_id, when,
                      platforms=draft.platforms or None)
        return f"scheduled for {datetime.fromtimestamp(when):%I:%M %p}."
    if action == "discard":
        queue.discard(draft_id)
        return "draft discarded."
    if action == "edit":
        return f"send me the revised text for this draft:\n\n{draft.content[:200]}"
    return ""

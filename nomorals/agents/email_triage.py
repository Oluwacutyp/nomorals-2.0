"""AI email triage with follow-up tracking.

Build-map #11. Reads the owner's Gmail, classifies unread mail
(``urgent`` / ``action`` / ``fyi``), drafts voice-matched replies into
a review queue, and resurfaces threads that went quiet after the owner
replied (the 48-hour forgotten-thread ping).

Safety posture (non-negotiable):

* triage is **read-only** — fetching never marks messages read;
* drafts are **never auto-sent** — not by the scheduler, not by
  ``followup_watch``; sending needs an approved draft or the owner's
  explicit "send it" phrasing (see :func:`is_explicit_send`);
* ``GmailConnector.send_message`` itself is confirmation-gated, so the
  draft queue is a second lock on top of the connector's own.

Chat: ``/email triage|drafts|send <id>|followups``.  Natural language
"what did <vendor> say about <topic>?" routes through the Core Mind's
``email_query`` intent (see ``_email_intent`` in coremind.py).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger("nomorals.email_triage")

__all__ = [
    "EmailItem",
    "TriageReport",
    "Draft",
    "FollowUp",
    "DraftQueue",
    "classify_item",
    "triage",
    "draft_reply",
    "followup_watch",
    "is_explicit_send",
    "check_explicit_send",
    "answer_vendor_query",
    "ensure_email_triage_job",
    "control_email",
    "EMAIL_DAILY_TASK_ID",
]

# ── classification ───────────────────────────────────────────────────────────

CATEGORY_URGENT = "urgent"
CATEGORY_ACTION = "action"
CATEGORY_FYI = "fyi"

_URGENT_KEYWORDS = (
    "urgent", "asap", "immediately", "deadline", "due today", "due tomorrow",
    "payment failed", "card declined", "security alert", "unauthorized",
    "suspended", "action required", "final notice", "overdue", "expires today",
    "court", "lawsuit", "fraud",
)

_ACTION_KEYWORDS = (
    "please", "could you", "can you", "would you", "let me know",
    "confirm", "approve", "review", "sign", "rsvp", "?", "feedback",
    "meeting", "call", "invoice",
)

_FYI_SENDER_PATTERNS = (
    "noreply", "no-reply", "donotreply", "newsletter", "notifications",
    "alerts@", "news@", "marketing@", "promo", "deals@",
)


def classify_item(
    subject: str,
    sender: str,
    snippet: str,
    labels: list[str] | None = None,
) -> tuple[str, float, bool]:
    """Classify one message → ``(category, urgency 0..1, needs_reply)``.

    Rule-based, fully offline.  Urgency is a score, not a verdict: the
    category drives the queue, the score drives the sort order.
    """
    text = f"{subject or ''} {snippet or ''}".lower()
    sender_l = (sender or "").lower()
    labels = labels or []

    urgency = 0.10  # baseline: it arrived, it exists
    urgent_hits = sum(1 for k in _URGENT_KEYWORDS if k in text)
    if urgent_hits:
        urgency += 0.30 + 0.10 * min(urgent_hits - 1, 3)
        return CATEGORY_URGENT, min(urgency, 1.0), True

    # Directly addressed to the owner reads more urgent than bulk mail.
    action_hits = sum(1 for k in _ACTION_KEYWORDS if k in text)
    is_bulk = any(p in sender_l for p in _FYI_SENDER_PATTERNS)
    needs_reply = False
    if action_hits and not is_bulk:
        urgency += 0.15 + 0.05 * min(action_hits - 1, 4)
        needs_reply = True
        return CATEGORY_ACTION, min(urgency, 0.85), needs_reply

    if is_bulk:
        return CATEGORY_FYI, min(urgency, 0.25), False
    # A personal mail with no ask is FYI with a nudge of urgency.
    return CATEGORY_FYI, min(urgency + 0.10, 0.40), False


# ── data ─────────────────────────────────────────────────────────────────────

@dataclass
class EmailItem:
    id: str
    thread_id: str
    subject: str
    sender: str
    snippet: str
    date_ts: float = 0.0
    urgency: float = 0.0
    category: str = CATEGORY_FYI
    needs_reply: bool = False
    labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TriageReport:
    items: list[EmailItem] = field(default_factory=list)
    generated_ts: float = field(default_factory=time.time)

    @property
    def urgent(self) -> list[EmailItem]:
        return [i for i in self.items if i.category == CATEGORY_URGENT]

    @property
    def action(self) -> list[EmailItem]:
        return [i for i in self.items if i.category == CATEGORY_ACTION]

    @property
    def fyi(self) -> list[EmailItem]:
        return [i for i in self.items if i.category == CATEGORY_FYI]

    @property
    def drafts_wanted(self) -> list[EmailItem]:
        return [i for i in self.items if i.needs_reply]

    def summary(self) -> str:
        lines = [
            f"📧 triage: {len(self.items)} unread — "
            f"{len(self.urgent)} urgent, {len(self.action)} action, "
            f"{len(self.fyi)} fyi",
        ]
        for item in sorted(self.items, key=lambda i: -i.urgency)[:8]:
            flag = {"urgent": "🔴", "action": "🟡", "fyi": "⚪"}[item.category]
            lines.append(f"{flag} {item.sender} — {item.subject[:60]}")
        if len(self.items) > 8:
            lines.append(f"… and {len(self.items) - 8} more")
        return "\n".join(lines)


@dataclass
class Draft:
    id: str
    thread_id: str
    to: str
    subject: str
    body: str
    created_ts: float = field(default_factory=time.time)
    status: str = "pending"  # pending | approved | sent | discarded

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Draft":
        return cls(
            id=str(data.get("id", "")),
            thread_id=str(data.get("thread_id", "")),
            to=str(data.get("to", "")),
            subject=str(data.get("subject", "")),
            body=str(data.get("body", "")),
            created_ts=float(data.get("created_ts", 0.0) or 0.0),
            status=str(data.get("status", "pending") or "pending"),
        )


@dataclass
class FollowUp:
    thread_id: str
    subject: str
    with_name: str
    last_owner_action_ts: float
    hours_quiet: float
    nudge_draft: str = ""


# ── triage ───────────────────────────────────────────────────────────────────

def _header(headers: list[dict[str, Any]], name: str) -> str:
    name_l = name.lower()
    for h in headers or []:
        if str(h.get("name", "")).lower() == name_l:
            return str(h.get("value", ""))
    return ""


def _item_from_message(msg: dict[str, Any]) -> EmailItem:
    payload = msg.get("payload") or {}
    headers = payload.get("headers") or []
    subject = _header(headers, "subject")
    sender = _header(headers, "from")
    date_raw = _header(headers, "date")
    date_ts = 0.0
    if date_raw:
        try:
            from email.utils import parsedate_to_datetime

            date_ts = parsedate_to_datetime(date_raw).timestamp()
        except Exception:  # noqa: BLE001 - bad date header, keep 0.0
            pass
    labels = list(msg.get("labelIds") or [])
    category, urgency, needs_reply = classify_item(
        subject, sender, msg.get("snippet", ""), labels)
    return EmailItem(
        id=str(msg.get("id", "")),
        thread_id=str(msg.get("threadId", "")),
        subject=subject or "(no subject)",
        sender=sender or "(unknown)",
        snippet=str(msg.get("snippet", ""))[:200],
        date_ts=date_ts,
        urgency=urgency,
        category=category,
        needs_reply=needs_reply,
        labels=labels,
    )


def triage(gmail: Any, *, max_messages: int = 25) -> TriageReport:
    """Fetch unread mail and classify it.  Read-only: never marks read."""
    report = TriageReport()
    try:
        listing = gmail.list_messages(query="is:unread",
                                      max_results=max(1, max_messages))
    except Exception as exc:  # noqa: BLE001 - connector down, empty report
        _log.warning("email triage: list failed: %s", exc)
        return report
    for ref in (listing.get("messages") or [])[:max_messages]:
        try:
            # "metadata" is enough (headers + snippet); "full" would pull
            # bodies we don't need for classification.
            msg = gmail.get_message(ref.get("id", ""), format="metadata")
        except Exception as exc:  # noqa: BLE001 - one bad message, keep going
            _log.debug("email triage: skipping %s: %s", ref.get("id"), exc)
            continue
        try:
            report.items.append(_item_from_message(msg))
        except Exception as exc:  # noqa: BLE001
            _log.debug("email triage: parse failed: %s", exc)
    report.items.sort(key=lambda i: -i.urgency)
    return report


# ── voice-matched drafts ─────────────────────────────────────────────────────

def _voice_hints(persona: Any) -> dict[str, str]:
    """Read brevity/formality/tone hints out of a UserModel, defensively."""
    hints = {"brevity": "short", "formality": "casual", "signoff": ""}
    try:
        prefs = getattr(persona, "preferences", None) or []
        blob = " ".join(str(getattr(p, "value", "")) for p in prefs).lower()
    except Exception:  # noqa: BLE001
        return hints
    if any(w in blob for w in ("formal", "professional", "proper")):
        hints["formality"] = "formal"
    if any(w in blob for w in ("brief", "short", "concise", "terse",
                               "straight to the point")):
        hints["brevity"] = "terse"
    elif any(w in blob for w in ("detailed", "thorough", "elaborate")):
        hints["brevity"] = "full"
    if any(w in blob for w in ("warm", "friendly", "kind")):
        hints["signoff"] = "warm"
    identity = getattr(persona, "identity", None) or {}
    try:
        name_attr = identity.get("name") if isinstance(identity, dict) else None
        if name_attr is not None:
            hints["owner_name"] = str(getattr(name_attr, "value", "")).strip()
    except Exception:  # noqa: BLE001
        pass
    return hints


def _first_name(sender: str) -> str:
    # "Adaeze Okafor <ada@example.com>" -> "Adaeze"; bare address -> handle.
    m = re.match(r"\s*([^<>,]+?)\s*(<[^>]+>)?\s*$", sender or "")
    name = (m.group(1).strip() if m else "").strip("\"' ")
    if "@" in name and "<" not in (sender or ""):
        name = name.split("@")[0]
    parts = re.split(r"[.\s_]+", name)
    return parts[0].capitalize() if parts and parts[0] else "there"


def draft_reply(item: EmailItem, persona: Any = None,
               *, llm: Any = None) -> str:
    """Draft a reply in the owner's voice.

    Template-based and offline-first — the real deliverable.  When an LLM
    is handed in it may polish, but the template path must already read
    like the owner.  Always returned marked ``[draft]``.
    """
    hints = _voice_hints(persona)
    name = _first_name(item.sender)
    formal = hints["formality"] == "formal"
    terse = hints["brevity"] == "terse"

    subject_l = (item.subject or "").lower()
    snippet_l = (item.snippet or "").lower()

    if any(k in subject_l or k in snippet_l
           for k in ("invoice", "payment", "receipt", "bill")):
        body_core = ("Got it — I'll sort this out and confirm once it's done."
                     if not formal else
                     "Thank you — I will take care of this and confirm once completed.")
    elif "?" in (item.snippet or "") or any(
            k in snippet_l for k in ("could you", "can you", "please",
                                     "let me know", "confirm")):
        body_core = ("On it — I'll check and get back to you shortly."
                     if not formal else
                     "Certainly — I will look into this and follow up shortly.")
    elif any(k in subject_l for k in ("meeting", "call", "schedule")):
        body_core = ("Works for me — send over the details."
                     if not formal else
                     "That works for me — please send the details.")
    else:
        body_core = ("Noted, thanks."
                     if terse else
                     "Thanks for sending this over — noted.")

    greeting = f"Hi {name}," if not formal else f"Hello {name},"
    sign = "Best," if formal else "Cheers,"
    owner = hints.get("owner_name", "")
    draft = f"{greeting}\n\n{body_core}\n\n{sign}\n{owner}".rstrip()

    if llm is not None:
        try:
            polished = llm.complete(
                "Polish this email draft. Keep it short, keep the meaning, "
                f"match a {hints['formality']} tone. Return only the draft:\n\n"
                f"{draft}")
            if polished and polished.strip():
                draft = polished.strip()
        except Exception as exc:  # noqa: BLE001 - polish is optional
            _log.debug("draft polish failed, keeping template: %s", exc)

    return f"[draft]\n{draft}"


# ── draft queue (persistent) ─────────────────────────────────────────────────

def _store_path(context: Any = None) -> Path:
    try:
        settings = getattr(context, "settings", None)
        if settings is not None:
            return Path(settings.resolve("data/email_triage.json"))
    except Exception:  # noqa: BLE001
        pass
    return Path(os.path.expanduser("~/.devon/email_triage.json"))


class DraftQueue:
    """Persistent draft queue.  Drafts are never auto-sent."""

    def __init__(self, path: Path | None = None, context: Any = None) -> None:
        self.path = Path(path) if path else _store_path(context)
        self._drafts: dict[str, Draft] = {}
        self._thread_activity: dict[str, float] = {}
        self._pending_send: str = ""  # draft id awaiting an explicit "send it"
        self._load()

    # -- persistence ---------------------------------------------------------
    def _load(self) -> None:
        try:
            if not self.path.exists():
                return
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for raw in data.get("drafts", []):
                try:
                    d = Draft.from_dict(raw)
                    if d.id:
                        self._drafts[d.id] = d
                except Exception:  # noqa: BLE001 - one bad draft, skip
                    continue
            self._thread_activity = {
                str(k): float(v)
                for k, v in (data.get("thread_activity") or {}).items()
            }
            self._pending_send = str(data.get("pending_send") or "")
        except Exception as exc:  # noqa: BLE001 - corrupt store, start fresh
            _log.warning("draft queue load failed (%s); starting fresh", exc)

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({
                "drafts": [d.to_dict() for d in self._drafts.values()],
                "thread_activity": self._thread_activity,
                "pending_send": self._pending_send,
            }, indent=1), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            _log.warning("draft queue save failed: %s", exc)

    # -- drafts ---------------------------------------------------------------
    def add(self, *, thread_id: str, to: str, subject: str,
            body: str) -> Draft:
        draft = Draft(
            id="dr_" + uuid.uuid4().hex[:10],
            thread_id=thread_id or "",
            to=to or "",
            subject=subject or "",
            body=body or "",
        )
        self._drafts[draft.id] = draft
        self._save()
        return draft

    def get(self, draft_id: str) -> Draft | None:
        return self._drafts.get(draft_id)

    def pending(self) -> list[Draft]:
        return [d for d in self._drafts.values() if d.status == "pending"]

    def approve(self, draft_id: str) -> bool:
        d = self._drafts.get(draft_id)
        if d is None or d.status != "pending":
            return False
        d.status = "approved"
        self._save()
        return True

    def discard(self, draft_id: str) -> bool:
        d = self._drafts.get(draft_id)
        if d is None or d.status in ("sent", "discarded"):
            return False
        d.status = "discarded"
        self._save()
        return True

    def send(self, gmail: Any, draft_id: str, *,
             confirmed: bool = False) -> dict[str, Any]:
        """Send one draft.  Requires approved status or explicit ``confirmed``.

        Fail-closed: pending drafts without confirmation raise, they never
        send.  The connector's own ``confirm_or_checkpoint`` is the second
        lock — ``confirmed=True`` here means the owner already approved.
        """
        d = self._drafts.get(draft_id)
        if d is None:
            raise ValueError(f"no draft {draft_id!r}")
        if d.status == "sent":
            raise ValueError(f"draft {draft_id!r} already sent")
        if d.status == "discarded":
            raise ValueError(f"draft {draft_id!r} was discarded")
        if d.status != "approved" and not confirmed:
            raise PermissionError(
                f"draft {draft_id!r} is not approved — approve it first, "
                "or the owner says 'send it' (explicit override)")
        result = gmail.send_message(
            d.to, d.subject, d.body, confirmed=True)
        d.status = "sent"
        if d.thread_id:
            self._thread_activity[d.thread_id] = time.time()
        self._save()
        return result

    # -- thread activity ------------------------------------------------------
    def note_owner_action(self, thread_id: str, ts: float | None = None) -> None:
        if thread_id:
            self._thread_activity[str(thread_id)] = ts or time.time()
            self._save()

    def last_owner_action(self, thread_id: str) -> float | None:
        return self._thread_activity.get(str(thread_id))

    # -- explicit-override ----------------------------------------------------
    def mark_pending_send(self, draft_id: str) -> None:
        """Arm one draft for the explicit-override ("send it") path."""
        self._pending_send = draft_id or ""
        self._save()

    def take_pending_send(self) -> str:
        """Consume the armed draft id ("" when none)."""
        pending = self._pending_send
        self._pending_send = ""
        if pending:
            self._save()
        return pending


# ── follow-up watch ──────────────────────────────────────────────────────────

def followup_watch(gmail: Any, queue: DraftQueue, *,
                   hours: float = 48.0,
                   now: float | None = None) -> list[FollowUp]:
    """Find threads the owner touched that went quiet.

    A thread qualifies when the owner's last action (sent draft, or a
    reply we can see in the thread) is older than ``hours`` and nothing
    newer arrived from the other side.  Returns nudges with a fresh
    draft ready — the caller decides whether to surface them; nothing
    is ever sent here.
    """
    now = time.time() if now is None else now
    cutoff = now - hours * 3600.0
    followups: list[FollowUp] = []

    for thread_id, action_ts in list(queue._thread_activity.items()):
        if action_ts >= cutoff:
            continue
        try:
            thread = gmail.get_thread(thread_id)
        except Exception as exc:  # noqa: BLE001
            _log.debug("followup: thread %s unreadable: %s", thread_id, exc)
            continue
        messages = thread.get("messages") or []
        if not messages:
            continue
        # Newest message in the thread — if anything arrived after the
        # owner's action, the thread isn't quiet.
        newest_ts = 0.0
        other_party = ""
        for m in messages:
            ts = float(m.get("internalDate", 0) or 0) / 1000.0
            if ts > newest_ts:
                newest_ts = ts
                headers = (m.get("payload") or {}).get("headers") or []
                other_party = _header(headers, "from") or other_party
        if newest_ts > action_ts:
            # They replied after us — thread is alive; refresh the marker.
            queue.note_owner_action(thread_id, newest_ts)
            continue
        first = messages[0]
        headers = (first.get("payload") or {}).get("headers") or []
        subject = _header(headers, "subject") or "(no subject)"
        hours_quiet = (now - action_ts) / 3600.0
        nudge = (
            f"Hi {_first_name(other_party)},\n\n"
            "Just circling back on this — any update on your end?\n\n"
            "Cheers,"
        )
        followups.append(FollowUp(
            thread_id=thread_id,
            subject=subject,
            with_name=other_party,
            last_owner_action_ts=action_ts,
            hours_quiet=round(hours_quiet, 1),
            nudge_draft=f"[draft]\n{nudge}",
        ))
    followups.sort(key=lambda f: -f.hours_quiet)
    return followups


# ── explicit-override ────────────────────────────────────────────────────────

#: Exact phrases that mean "send it now, no re-confirmation" (user
#: directive 2026-10-07).  Matched against the whole message after
#: normalization — deliberately narrow: when in doubt, DON'T match.
_EXPLICIT_SEND_PHRASES = (
    "send it",
    "send all",
    "send them",
    "send the drafts",
    "reply now",
    "send now",
    "yes send",
    "yes, send it",
    "please send it",
    "go ahead and send",
    "post it",
)


def is_explicit_send(text: str) -> bool:
    """True when the owner explicitly said to send, no reconfirmation.

    Errs toward False: questions ("should I send it?"), hedges, and
    longer sentences never match.
    """
    if not text or not isinstance(text, str):
        return False
    t = text.strip().lower()
    # Strip trailing punctuation for the whole-message match.
    t = re.sub(r"[!.\s]+$", "", t)
    if t in _EXPLICIT_SEND_PHRASES:
        return True
    # Also accept the phrase as a standalone sentence inside a short
    # message ("looks good. send it"), but never inside a question.
    if "?" in text:
        return False
    if len(t) > 60:
        return False
    sentences = [s.strip() for s in re.split(r"[.!;\n]+", t) if s.strip()]
    return any(s in _EXPLICIT_SEND_PHRASES for s in sentences)


# ── NL vendor query ──────────────────────────────────────────────────────────

_RE_VENDOR_QUERY = re.compile(
    r"what did\s+(.+?)\s+say about\s+(.+?)\s*\??\s*$", re.I)


def answer_vendor_query(gmail: Any, text: str,
                        *, max_results: int = 5) -> str | None:
    """'what did <vendor> say about <topic>?' → Gmail search + synthesis.

    Returns None when the text isn't a vendor query (the caller falls
    through to normal handling).
    """
    m = _RE_VENDOR_QUERY.match((text or "").strip())
    if not m:
        return None
    vendor, topic = m.group(1).strip(), m.group(2).strip()
    if not vendor or not topic:
        return None
    if vendor.lower() in ("you", "u", "ya", "yall", "they", "he", "she",
                          "it", "we", "everyone", "anyone"):
        return None
    query = f"from:{vendor} {topic}"
    try:
        listing = gmail.list_messages(query=query, max_results=max_results)
    except Exception as exc:  # noqa: BLE001
        return f"couldn't search mail: {exc}"
    refs = listing.get("messages") or []
    if not refs:
        return f"nothing from {vendor} about {topic} in your mail."
    bits: list[str] = []
    for ref in refs[:max_results]:
        try:
            msg = gmail.get_message(ref.get("id", ""), format="metadata")
        except Exception:  # noqa: BLE001
            continue
        item = _item_from_message(msg)
        bits.append(f"• {item.subject} — {item.snippet[:140]}")
    if not bits:
        return f"found threads from {vendor} about {topic} but couldn't read them."
    return (f"Here's what {vendor} said about {topic} "
            f"({len(bits)} thread{'s' if len(bits) != 1 else ''}):\n"
            + "\n".join(bits))


# ── scheduler job ────────────────────────────────────────────────────────────

EMAIL_DAILY_TASK_ID = "email-triage-daily"
EMAIL_DAILY_CRON = "0 8 * * *"  # 08:00 daily, before the morning briefing
EMAIL_DAILY_ACTION = "email_triage.daily"


def _get_gmail(context: Any) -> Any | None:
    """Build a GmailConnector from the context's vault, or None."""
    try:
        from ..accounts.vault import CredentialVault
        from ..connectors.registry import create_connector

        passphrase = os.environ.get("NM_VAULT_PASSPHRASE", "")
        vault = CredentialVault(context.db, master_passphrase=passphrase)
        gmail = create_connector("gmail", vault)
        st = gmail.status()
        if not getattr(st, "connected", False):
            _log.info("email triage: gmail not connected (%s)",
                      getattr(st, "detail", ""))
            return None
        return gmail
    except Exception as exc:  # noqa: BLE001
        _log.warning("email triage: connector unavailable: %s", exc)
        return None


async def ensure_email_triage_job(scheduler: Any, context: Any) -> bool:
    """Register the daily triage run. Idempotent. Profile-agnostic."""
    try:
        existing = await scheduler.list_cron_jobs()
    except Exception as exc:  # noqa: BLE001
        _log.warning("email triage: could not list cron jobs: %s", exc)
        return False
    if any(getattr(j, "task_id", "") == EMAIL_DAILY_TASK_ID
           for j in existing):
        return True

    async def _daily_run(**params: Any) -> None:
        gmail = _get_gmail(context)
        if gmail is None:
            return
        queue = DraftQueue(context=context)
        report = triage(gmail)
        for item in report.drafts_wanted:
            try:
                persona = getattr(context, "persona", None)
                body = draft_reply(item, persona)
                queue.add(thread_id=item.thread_id, to=item.sender,
                          subject=f"Re: {item.subject}", body=body)
            except Exception as exc:  # noqa: BLE001
                _log.debug("triage draft failed for %s: %s", item.id, exc)
        # Drafts are queued for review — never sent here.
        _log.info("email triage daily: %d unread, %d drafts queued",
                  len(report.items), len(queue.pending()))

    scheduler.register_action(EMAIL_DAILY_ACTION, _daily_run)
    await scheduler.schedule_cron(
        EMAIL_DAILY_TASK_ID,
        EMAIL_DAILY_CRON,
        EMAIL_DAILY_ACTION,
        parameters={},
    )
    _log.info("email triage daily job scheduled (%s)", EMAIL_DAILY_CRON)
    return True


# ── explicit-override in the NL path ─────────────────────────────────────────

def check_explicit_send(text: str, context: Any) -> str | None:
    """Owner said "send it" in chat → send the armed draft, no reconfirm.

    Returns the reply, or None when there's nothing to do (not an
    explicit-send phrase, or no draft is armed).  Called from the
    runtime's incoming path, owner chats only.
    """
    if not is_explicit_send(text):
        return None
    queue = DraftQueue(context=context)
    draft_id = queue.take_pending_send()
    if not draft_id:
        return None
    draft = queue.get(draft_id)
    if draft is None or draft.status == "sent":
        return None
    gmail = _get_gmail(context)
    if gmail is None:
        return "📧 gmail isn't connected — can't send."
    try:
        # The owner already saw the exact payload and said "send it":
        # confirmed=True is the explicit override, not a bypass.
        queue.send(gmail, draft_id, confirmed=True)
    except Exception as exc:  # noqa: BLE001
        return f"📧 couldn't send: {exc}"
    return f"📧 sent to {draft.to} — {draft.subject[:60]}"


# ── chat command ─────────────────────────────────────────────────────────────

def control_email(arg: str, context: Any) -> str:
    """``/email triage|drafts|send <id>|followups`` — the triage surface."""
    tail = (arg or "").strip()
    parts = tail.split(None, 1)
    sub = (parts[0] if parts else "").lower()
    rest = parts[1] if len(parts) > 1 else ""

    gmail = _get_gmail(context)
    if gmail is None:
        return ("📧 gmail isn't connected — run "
                "`nm connectors connect --name gmail` first.")

    queue = DraftQueue(context=context)

    if sub in ("", "triage"):
        report = triage(gmail)
        persona = getattr(context, "persona", None)
        made = 0
        for item in report.drafts_wanted:
            if any(d.thread_id == item.thread_id and d.status == "pending"
                   for d in queue.pending()):
                continue  # already have a pending draft for this thread
            queue.add(thread_id=item.thread_id, to=item.sender,
                      subject=f"Re: {item.subject}",
                      body=draft_reply(item, persona))
            made += 1
        out = report.summary()
        if made:
            out += f"\n\n✏️ {made} draft{'s' if made != 1 else ''} queued — /email drafts to review"
        return out

    if sub == "drafts":
        pending = queue.pending()
        if not pending:
            return "📧 no pending drafts."
        lines = [f"📧 {len(pending)} pending draft{'s' if len(pending) != 1 else ''}:"]
        for d in pending[:10]:
            lines.append(f"`{d.id}` → {d.to} — {d.subject[:50]}")
        if len(pending) > 10:
            lines.append(f"… and {len(pending) - 10} more")
        lines.append("\n`/email send <id>` sends one (asks first, unless you say \"send it\")")
        return "\n".join(lines)

    if sub == "send":
        draft_id = rest.strip()
        if not draft_id:
            return "usage: /email send <draft-id> — /email drafts lists them"
        # Explicit-override lives in the natural-language path (the owner
        # saying "send it" in chat); the slash form always confirms first.
        d = queue.get(draft_id)
        if d is None:
            return f"no draft `{draft_id}` — /email drafts lists them"
        if d.status == "sent":
            return f"draft `{draft_id}` already sent."
        # Arm the explicit-override: the owner's next "send it" in this
        # chat sends exactly this draft, no re-confirmation.
        queue.mark_pending_send(draft_id)
        return (f"📧 ready to send:\nTo: {d.to}\nSubject: {d.subject}\n\n"
                f"{d.body}\n\nReply `send it` and it's gone — "
                "nothing sends without your word.")

    if sub == "followups":
        fus = followup_watch(gmail, queue)
        if not fus:
            return "📧 no forgotten threads — everything's answered."
        lines = [f"📧 {len(fus)} thread{'s' if len(fus) != 1 else ''} gone quiet:"]
        for fu in fus[:8]:
            lines.append(
                f"• {fu.with_name} — {fu.subject[:50]} "
                f"({fu.hours_quiet:.0f}h quiet)")
        lines.append("\nSay `send it` + the thread, or I'll keep the nudge drafts ready.")
        return "\n".join(lines)

    return ("usage: /email triage | drafts | send <id> | followups\n"
            "or just ask: \"what did <vendor> say about <topic>?\"")

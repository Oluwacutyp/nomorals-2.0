"""WhatsApp group digests.

A group the owner is in can produce hundreds of messages a day. The
digest answers "what needs me?" without the owner reading the scroll:

* **mentioned Nx** — messages that tag the owner (``@name`` / name
  mention), counted per group.
* **decisions needing you** — messages that ask a question or call for a
  decision (``?``, "should we", "decide", "vote", "confirm", "approve",
  "need you", "your call") and are not yet resolved.
* **resolved threads** — questions that got an answer in-thread
  ("resolved", "done", "sorted", "agreed", "✅" on a decision message).

Per-group policy (stored per group id, default ``only-when-mentioned``):

* ``always`` — digest every group, even with no mentions.
* ``only-when-mentioned`` — digest only when the owner was mentioned
  or a decision needs them.
* ``never`` — no digest for this group, ever.

Pure functions: messages are plain dicts
(``{"sender", "text", "ts", "reply_to"}``), so this is fully testable
without a WhatsApp session. The adapter supplies the messages; this
module decides what the owner sees.
"""

from __future__ import annotations

import contextlib
import re
import time
from typing import Any

__all__ = [
    "GroupPolicy",
    "GROUP_POLICIES",
    "DEFAULT_GROUP_POLICY",
    "classify_message",
    "build_digest",
    "should_digest",
    "read_group_digest",
    "build_group_digest_text",
    "owner_me_for_group",
    "digest_policy_for_group",
    "DIGEST_JOB",
    "DIGEST_SPEC",
    "DIGEST_TOOL",
    "ensure_group_digest_job",
    "run_group_digest_tick_once",
]

#: Per-group digest policies.
GROUP_POLICIES: tuple[str, ...] = ("always", "only-when-mentioned", "never")

#: Default when a group has no explicit policy.
DEFAULT_GROUP_POLICY = "only-when-mentioned"

#: Alias type for readability.
GroupPolicy = str

#: Durable scheduler job name / cadence / tool name (mirrors the
#: group-rhythm and roster-diff ticks: idempotent, best-effort).
DIGEST_JOB = "group digest tick"
DIGEST_SPEC = "every 6h"
DIGEST_TOOL = "__group_digest_tick__"

#: How far back a digest window reaches when no watermark exists.
DIGEST_WINDOW_S = 24 * 3600.0

#: Cap on log rows read per group per digest (the organ shows at most
#: max_items per section; the read cap bounds the query, not the text).
DIGEST_READ_LIMIT = 500

_DECISION_RES = (
    re.compile(r"\?+\s*$"),
    re.compile(r"\b(should we|shall we|decide|decision|vote|confirm|approve|need you|your call|what do you think|any objection)\b", re.I),
)
_RESOLVED_RES = (
    re.compile(r"\b(resolved|sorted|settled|agreed|done|closed|fixed|confirmed)\b", re.I),
    re.compile(r"✅|✔️|👍"),
)
_MENTION_RES = (
    re.compile(r"@(\w+)"),
)


def classify_message(text: str, *, me: str = "") -> dict[str, bool]:
    """Flags for one message: mentioned / decision / resolved.

    ``me`` is the owner's name/username (``@`` optional). A message is
    "mentioned" when it tags ``@me`` or contains the owner's name as a
    word. "decision" is a question-or-call-for-decision that is not
    already marked resolved.
    """
    t = text or ""
    mentioned = False
    if me:
        name = me.strip().lstrip("@").lower()
        if name:
            if f"@{name}" in t.lower():
                mentioned = True
            elif re.search(rf"(?<![a-z0-9_]){re.escape(name)}(?![a-z0-9_])", t.lower()):
                mentioned = True
    decision = any(rx.search(t) for rx in _DECISION_RES)
    resolved = any(rx.search(t) for rx in _RESOLVED_RES)
    if resolved:
        decision = False  # resolved threads stop needing the owner
    return {"mentioned": mentioned, "decision": decision, "resolved": resolved}


def should_digest(policy: str, *, mentions: int, decisions: int) -> bool:
    """Whether this group earns a digest under its policy."""
    p = (policy or DEFAULT_GROUP_POLICY).strip().lower()
    if p == "never":
        return False
    if p == "always":
        return True
    # only-when-mentioned (and any unknown value fails closed to it)
    return mentions > 0 or decisions > 0


def build_digest(group: str, messages: list[dict[str, Any]], *,
                 policy: str = DEFAULT_GROUP_POLICY,
                 me: str = "",
                 max_items: int = 10) -> dict[str, Any]:
    """Build one group's digest. Returns a report dict::

        {"group", "policy", "included", "mentions", "decisions",
         "resolved", "text"}

    ``text`` is "" when the group earns no digest under its policy —
    callers send nothing instead of an empty digest.
    """
    mentions: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    resolved_count = 0
    for msg in messages or []:
        flags = classify_message(str(msg.get("text", "")), me=me)
        if flags["mentioned"]:
            mentions.append(msg)
        if flags["decision"]:
            decisions.append(msg)
        if flags["resolved"]:
            resolved_count += 1
    included = should_digest(policy, mentions=len(mentions), decisions=len(decisions))
    if not included:
        return {"group": group, "policy": policy, "included": False,
                "mentions": len(mentions), "decisions": len(decisions),
                "resolved": resolved_count, "text": ""}
    lines = [f"💬 *{group}* — digest"]
    if mentions:
        lines.append(f"  👤 mentioned ×{len(mentions)}:")
        for m in mentions[:max_items]:
            lines.append(f"    • {m.get('sender', '?')}: {(str(m.get('text', ''))[:140])}")
    if decisions:
        lines.append(f"  ❓ decisions needing you ×{len(decisions)}:")
        for m in decisions[:max_items]:
            lines.append(f"    • {m.get('sender', '?')}: {(str(m.get('text', ''))[:140])}")
    if resolved_count:
        lines.append(f"  ✅ resolved threads ×{resolved_count}")
    if not mentions and not decisions:
        lines.append(f"  ({len(messages or [])} messages, nothing needs you)")
    return {"group": group, "policy": policy, "included": True,
            "mentions": len(mentions), "decisions": len(decisions),
            "resolved": resolved_count, "text": "\n".join(lines)}


# ── wiring: the log read, the tick, the job ──────────────────────────
# The organ above stays pure. Everything below is the live wiring:
# ConversationLog is the ledger (the brain persists every inbound
# message it reads, group chatter included — see the survey), the
# Notifier is the delivery organ, the group surface holds the
# per-group policy + watermark.

def _group_title(surface: Any, group_key: str) -> str:
    """The group's display title, or the key when nothing is recorded."""
    try:
        entry = surface.store.group(group_key)
        title = (entry.get("title") or entry.get("name") or "").strip()
        if title:
            return title
    except Exception:  # noqa: BLE001 - a title is cosmetic
        pass
    return str(group_key)


def owner_me_for_group(surface: Any, group_key: str) -> str:
    """The owner's name as this group writes it, for mention matching.

    No stored owner display name exists anywhere (identity is by
    sender/chat ID), so the name is derived from the member ledger:
    the member whose OwnerRegistry rung is owner, whose stored
    ``name``/``username`` is exactly what the group writes. ``""``
    when the surface has no registry or no owner member record —
    mention matching then finds nothing, decisions still detected.
    Never hardcoded.
    """
    registry = getattr(surface, "owner_registry", None)
    if registry is None:
        return ""
    platform = str(group_key).split(":", 1)[0]
    try:
        owner_ids = list(registry.identities(platform) or [])
    except Exception:  # noqa: BLE001 - no owner ids, no mentions
        return ""
    try:
        members = surface.store.group(group_key).get("members") or {}
    except Exception:  # noqa: BLE001 - no ledger, no mentions
        return ""
    for uid in owner_ids:
        member = members.get(uid) or {}
        for field in ("name", "username"):
            value = str(member.get(field) or "").strip()
            if value:
                return value
    return ""


def digest_policy_for_group(surface: Any, group_key: str) -> str:
    """This group's digest policy from its merged settings."""
    try:
        policy = str(surface.store.settings(group_key).get(
            "digest_policy") or DEFAULT_GROUP_POLICY)
    except Exception:  # noqa: BLE001 - unreadable settings fail closed
        return DEFAULT_GROUP_POLICY
    policy = policy.strip().lower()
    return policy if policy in GROUP_POLICIES else DEFAULT_GROUP_POLICY


def read_group_digest(log: Any, group_key: str, *,
                      policy: str = DEFAULT_GROUP_POLICY,
                      me: str = "", since: float | None = None,
                      now: float | None = None,
                      max_items: int = 10) -> dict[str, Any]:
    """One group's digest from the ConversationLog.

    Reads ``messages_after(group_key, since)`` — the brain's persisted
    inbound rows — maps them to the organ's plain dicts and runs
    ``build_digest``. ``since=None`` means the whole 24h window.
    ``group`` in the report is the chat key; callers re-title it.
    """
    now = now if now is not None else time.time()
    since = now - DIGEST_WINDOW_S if since is None else float(since)
    rows = log.messages_after(group_key, since,
                              limit=DIGEST_READ_LIMIT) or []
    # messages_after is newest-first; the digest shows the most
    # recent mentions/decisions first (build_digest slices the
    # first max_items per section).
    messages = [{
        "sender": str(row.get("name") or "?"),
        "text": str(row.get("content") or ""),
        "ts": float(row.get("created_at") or 0.0),
    } for row in rows]
    return build_digest(group_key, messages, policy=policy, me=me,
                        max_items=max_items)


def _digest_watermark(surface: Any, group_key: str) -> float:
    try:
        return float(surface.store.group(group_key).get(
            "digest_last_ts") or 0.0)
    except Exception:  # noqa: BLE001 - no watermark yet
        return 0.0


def _set_digest_watermark(surface: Any, group_key: str, ts: float) -> None:
    try:
        surface.store.group(group_key)["digest_last_ts"] = float(ts)
        surface.store.save()
    except Exception:  # noqa: BLE001 - a watermark never breaks a digest
        pass


def build_group_digest_text(surface: Any, log: Any, group_key: str, *,
                            now: float | None = None,
                            since: float | None = None
                            ) -> dict[str, Any]:
    """One group's digest, titled and ready to send.

    Reads the log, applies the group's stored policy, derives ``me``
    from the member ledger, and swaps the chat key for the group's
    title in the text. Returns the ``build_digest`` report plus
    ``"title"``. ``"text"`` is ``""`` when the group earns no digest.
    """
    now = now if now is not None else time.time()
    policy = digest_policy_for_group(surface, group_key)
    if since is None:
        # The command's window: the full last 24h. The tick passes
        # its watermarked since explicitly, so a manual /digest is
        # never shortened by the last scheduled run.
        since = now - DIGEST_WINDOW_S
    digest = read_group_digest(
        log, group_key, policy=policy,
        me=owner_me_for_group(surface, group_key),
        since=since, now=now)
    title = _group_title(surface, group_key)
    text = str(digest.get("text") or "")
    if text and title != group_key:
        text = text.replace(f"*{group_key}*", f"*{title}*", 1)
    digest["title"] = title
    digest["text"] = text
    return digest


def run_group_digest_tick_once(*, context: Any = None,
                               surface: Any = None,
                               log: Any = None,
                               notifier: Any = None,
                               now: float | None = None
                               ) -> dict[str, Any]:
    """One scheduled digest pass: every claimed WhatsApp group with
    new messages since its watermark gets its digest built — no
    message required. Routine and silent; one bad group never kills
    the pass.

    ``surface`` / ``log`` / ``notifier`` are injectable for tests;
    surface and log are resolved from the context otherwise
    (``default_surface`` from the context's workspace dir,
    ``ConversationLog(context.db)``) — the rhythm tick's shape. The
    notifier is NOT resolved here: agents/notifier sits a layer
    above this organ, so the scheduler tool constructs it and
    passes it in (the immune tick's tool does the same). With no
    notifier the tick still builds and watermarks, and returns
    every digest as a {"group", "title", "body"} payload in
    summary["posted"] for the caller to deliver. A second run with
    no new messages builds nothing: the watermark advanced,
    nothing earned.
    """
    now = now if now is not None else time.time()
    summary: dict[str, Any] = {"ok": True, "groups": 0, "posted": [],
                               "skipped": 0, "failed": []}
    if surface is None:
        ws = None
        if context is not None:
            ws = getattr(getattr(context, "settings", None),
                         "workspace_dir", ".") or "."
        try:
            from .group_surface import default_surface

            surface = default_surface(ws or ".")
        except Exception as exc:  # noqa: BLE001 - no surface, no tick
            return {"ok": False, "error": f"no group surface: {exc}"}
    if log is None:
        db = getattr(context, "db", None) if context is not None else None
        if db is None:
            return {"ok": False, "error": "no database for digest log"}
        try:
            from ...storage.conversations import ConversationLog

            log = ConversationLog(db)
        except Exception as exc:  # noqa: BLE001 - no log, no tick
            return {"ok": False, "error": f"no digest log: {exc}"}
    for key in sorted(surface.store.data.keys()):
        try:
            if surface.is_channel(key):
                summary["skipped"] += 1
                continue
        except Exception:  # noqa: BLE001 - unreadable: skip
            summary["skipped"] += 1
            continue
        try:
            if not surface.claim_state(key)["claimed"]:
                summary["skipped"] += 1
                continue
        except Exception:  # noqa: BLE001 - one bad group never kills the pass
            summary["skipped"] += 1
            continue
        platform = str(key).split(":", 1)[0]
        if platform != "whatsapp":
            summary["skipped"] += 1
            continue
        summary["groups"] += 1
        try:
            since = max(_digest_watermark(surface, key),
                        now - DIGEST_WINDOW_S)
            digest = build_group_digest_text(surface, log, key,
                                             now=now, since=since)
            _set_digest_watermark(surface, key, now)
            if not digest["included"]:
                summary["skipped"] += 1
                continue
            payload = {"group": str(key),
                       "title": f"digest — {digest['title']}",
                       "body": str(digest["text"] or "")}
            summary["posted"].append(payload)
            if notifier is not None:
                notifier.publish("digest", title=payload["title"],
                                 body=payload["body"])
        except Exception as exc:  # noqa: BLE001 - one bad group never kills the pass
            summary["failed"].append({"group": str(key),
                                     "error": str(exc)})
    return summary


def ensure_group_digest_job(scheduler: Any, *,
                            spec: str = DIGEST_SPEC) -> Any:
    """Register the durable group-digest tick job. Idempotent —
    mirrors ``ensure_group_roster_diff_job``: an existing job at the
    requested cadence is returned as-is, a stale one is replaced.
    Best-effort: never raises."""
    if scheduler is None:
        return None
    try:
        for job in (scheduler.list_jobs() or []):
            name = (job.get("name") if isinstance(job, dict)
                    else getattr(job, "name", None))
            if name == DIGEST_JOB:
                job_spec = (job.get("spec") if isinstance(job, dict)
                            else getattr(job, "spec", None))
                if job_spec == spec:
                    return job
                with contextlib.suppress(Exception):
                    job_id = (job.get("id") if isinstance(job, dict)
                              else getattr(job, "id", None))
                    scheduler.remove(job_id)
                break
        return scheduler.add(DIGEST_JOB, spec, "tool",
                             {"tool": DIGEST_TOOL, "args": {}},
                             routine=True)
    except Exception:  # noqa: BLE001 - scheduling is best-effort
        return None

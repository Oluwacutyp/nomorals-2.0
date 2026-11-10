"""The engagement loop (WS3 P8): replies on his posts, read back.

Until this module, nothing in Devon read back the responses to the
owner's own posts: the measurement spine (WS3 P1) re-read COUNTS,
and chat triage reads inbound MESSAGES, but a reply or comment on
something he published had no surface, no store and no way to be
answered through the draft queue. This module is that loop:

* a durable ``social engagement poll`` job (registered in
  :mod:`nomorals.social.jobs`, run through the
  ``social_engagement_poll`` tool) re-reads engagement on his
  posted rows from each capable adapter's
  :meth:`PlatformAdapter.read_engagement` and stores it, deduped
  on the platform's own item id;
* the /engage board renders the stored items triage-shaped —
  what NEEDS HIM first, oldest first — and its verbs draft
  replies through the same DraftQueue + #43 rule as every other
  outward post (a reply is a draft at ``pending_review`` until
  he approves it on /drafts; nothing here posts);
* per-post engagement volume (items received, replies sent) is
  fed — once, at a fair age, through the same content+platform
  brief join the metrics spine uses — into the briefs outcome
  loop, and durably recorded in ``engagement_outcomes`` either
  way. It is evidence for calibration, never an auto-anything
  input: no draft is promoted, no window re-ranked, no claim
  re-scored because engagement arrived.

Platform truth is STRUCTURAL, like the metrics spine's: an
adapter that does not override ``read_engagement`` cannot
produce engagement, and the poll NAMES its platform
``unavailable`` — never silently absent, never zero-filled.
Per the P8 survey of the shipped adapters:

* mastodon — reads replies via the status-context endpoint;
  replies thread natively (``in_reply_to_id``);
* x — reads a post's conversation via recent search; replies
  thread natively (``reply.in_reply_to_tweet_id``);
* linkedin — NO read-back (the connector's surface is
  post/get_post/get_me; no comments operation, no
  ``r_member_social`` scope) and NO comment write, so no
  threaded reply either;
* whatsapp-status — NO read-back of replies (the seam exposes
  ``send_status`` + status VIEWERS; a view is not a reply, and
  status replies arrive as inbound DMs — chat triage's ground,
  not this loop's); a fresh status is not a threaded reply;
* telegram (personal channel) — NO read-back through the
  shipped seams (the userbot ``history()`` conversion drops
  sender + reply linkage and the discussion-anchor mapping is
  not carried on ChatMessage, so per-post attribution would be
  a guess); the adapter posts INTO the channel, so a "reply"
  from it would not thread onto a comment either;
* bluesky / postiz — no engagement read ships; bluesky's reply
  wants a full strong-ref dict, not the id the publish path
  threads, so it is not reply-capable through this seam.

The personal-surface live gate governs the personal adapters'
poll path exactly as it governs their posts: the poll consults
:meth:`engagement_gate` BEFORE any read attempt — a platform
under the owner's sticky stop is never even asked. Reads are
observation, so the degraded-brain observe-only rule does not
block them; the stop does. And where the gate refuses, the poll
records the platform ``gated`` with the refusal's own words.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from ..core.ids import new_id
from ..core.logging_setup import get_logger
from .base import PlatformAdapter

__all__ = [
    "EngagementStore",
    "adapter_supports_engagement",
    "platform_truth",
    "poll_engagement",
    "draft_reply",
    "mark_reply_posted",
    "reconcile_replied",
    "board_items",
    "render_board",
    "render_item",
    "STATE_NEW", "STATE_TRIAGED", "STATE_REPLIED", "STATE_IGNORED",
    "STATE_NEEDS_OWNER", "STATE_DONE", "OPEN_STATES",
]

_log = get_logger(__name__)

# ── item lifecycle ───────────────────────────────────────────────────
STATE_NEW = "new"                # arrived, unseen — needs him
STATE_TRIAGED = "triaged"        # seen (shown, or a reply is drafted)
STATE_REPLIED = "replied"        # the reply draft posted (terminal)
STATE_IGNORED = "ignored"        # he dismissed it (terminal)
STATE_NEEDS_OWNER = "needs_owner"  # Devon cannot route it — his call
STATE_DONE = "done"              # he handled it himself (terminal)

#: States an item can still be acted on from.
OPEN_STATES = (STATE_NEW, STATE_TRIAGED, STATE_NEEDS_OWNER)
TERMINAL_STATES = (STATE_REPLIED, STATE_IGNORED, STATE_DONE)

#: Survey one-liners for the board's named-unavailable line: why a
#: platform's replies are not read here, in plain words.
UNAVAILABLE_REASONS = {
    "linkedin": "its connector has no comments read",
    "whatsapp-status": "status replies arrive as direct messages, "
                       "not a readable thread",
    "telegram": "channel comments cannot be attributed to a post "
                "through the shipped seams",
    "bluesky": "its adapter ships no engagement read",
    "postiz": "its adapter ships no engagement read",
}

#: How far back a posted row stays worth polling — the spine's
#: own 90-day horizon (social.metrics.WINDOW_DAYS).
WINDOW_DAYS = 90
#: The outcome feed waits for this post age — the spine's rule
#: (social.metrics.OUTCOME_MIN_AGE_S): an hour's engagement is
#: noise; a day's is a fair read to stamp once.
OUTCOME_MIN_AGE_S = 24 * 3600.0
#: Bounded work per run, freshest posts first.
DEFAULT_LIMIT = 200


_DDL = """
CREATE TABLE IF NOT EXISTS engagement_items (
    id TEXT PRIMARY KEY,
    platform TEXT NOT NULL,
    post_id TEXT NOT NULL,
    post_external_id TEXT NOT NULL DEFAULT '',
    account_id TEXT NOT NULL DEFAULT '',
    item_external_id TEXT NOT NULL,
    author TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL DEFAULT 'reply',
    arrived_at REAL NOT NULL DEFAULT 0,
    state TEXT NOT NULL DEFAULT 'new',
    detail TEXT NOT NULL DEFAULT '',
    reply_draft_id TEXT NOT NULL DEFAULT '',
    reply_post_external_id TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(platform, item_external_id)
);
CREATE INDEX IF NOT EXISTS idx_engagement_items_state
    ON engagement_items(state, arrived_at);
CREATE INDEX IF NOT EXISTS idx_engagement_items_post
    ON engagement_items(post_id);
CREATE TABLE IF NOT EXISTS engagement_outcomes (
    id TEXT PRIMARY KEY,
    post_id TEXT NOT NULL,
    platform TEXT NOT NULL DEFAULT '',
    verdict TEXT NOT NULL,
    items_received INTEGER NOT NULL DEFAULT 0,
    replies_sent INTEGER NOT NULL DEFAULT 0,
    detail TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    UNIQUE(post_id)
);
"""


class EngagementStore:
    """Engagement items + outcome records over the app Database.

    Runtime-created tables (the Batch B pattern — migrations.py
    is frozen and untouched). Dedup is the table's own UNIQUE
    (platform, item_external_id): the platform's id for a reply
    is the identity, so a re-poll can never duplicate an item.
    """

    def __init__(self, db: Any) -> None:
        self._db = db
        db.executescript(_DDL)

    # ── items ────────────────────────────────────────────────────
    def add_item(
        self, *, platform: str, post_id: str, post_external_id: str,
        account_id: str, item_external_id: str, author: str,
        text: str, kind: str, arrived_at: float,
    ) -> bool:
        """Insert one item; True when new, False when already held."""
        now = time.time()
        cur = self._db.execute(
            "INSERT OR IGNORE INTO engagement_items (id, platform,"
            " post_id, post_external_id, account_id,"
            " item_external_id, author, text, kind, arrived_at,"
            " state, created_at, updated_at) VALUES"
            " (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (new_id(), platform, post_id, post_external_id,
             account_id, item_external_id, author, text, kind,
             float(arrived_at or 0), STATE_NEW, now, now))
        return bool(getattr(cur, "rowcount", 0))

    def get(self, item_id: str) -> dict[str, Any] | None:
        rows = self._db.query(
            "SELECT * FROM engagement_items WHERE id = ?", (item_id,))
        return rows[0] if rows else None

    def item_by_draft(self, draft_id: str) -> dict[str, Any] | None:
        if not draft_id:
            return None
        rows = self._db.query(
            "SELECT * FROM engagement_items WHERE reply_draft_id = ?",
            (draft_id,))
        return rows[0] if rows else None

    def items(
        self, *, states: tuple[str, ...] | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Items oldest first — attention order, not arrival-in-DB
        order: the oldest unanswered reply outranks today's."""
        if states:
            marks = ", ".join("?" for _ in states)
            rows = self._db.query(
                f"SELECT * FROM engagement_items WHERE state IN"
                f" ({marks}) ORDER BY arrived_at ASC, created_at ASC"
                f" LIMIT ?",
                (*states, int(limit)))
        else:
            rows = self._db.query(
                "SELECT * FROM engagement_items ORDER BY"
                " arrived_at ASC, created_at ASC LIMIT ?",
                (int(limit),))
        return [dict(r) for r in rows]

    def set_state(
        self, item_id: str, state: str, *, detail: str | None = None,
        reply_draft_id: str | None = None,
        reply_post_external_id: str | None = None,
    ) -> dict[str, Any] | None:
        sets = ["state = ?", "updated_at = ?"]
        params: list[Any] = [state, time.time()]
        if detail is not None:
            sets.append("detail = ?")
            params.append(detail)
        if reply_draft_id is not None:
            sets.append("reply_draft_id = ?")
            params.append(reply_draft_id)
        if reply_post_external_id is not None:
            sets.append("reply_post_external_id = ?")
            params.append(reply_post_external_id)
        params.append(item_id)
        self._db.execute(
            f"UPDATE engagement_items SET {', '.join(sets)}"
            " WHERE id = ?", tuple(params))
        return self.get(item_id)

    def counts_for_post(self, post_id: str) -> dict[str, int]:
        rows = self._db.query(
            "SELECT COUNT(*) AS n FROM engagement_items"
            " WHERE post_id = ?", (post_id,))
        received = int(rows[0]["n"]) if rows else 0
        rows = self._db.query(
            "SELECT COUNT(*) AS n FROM engagement_items"
            " WHERE post_id = ? AND state = ?",
            (post_id, STATE_REPLIED))
        sent = int(rows[0]["n"]) if rows else 0
        return {"received": received, "replies_sent": sent}

    # ── outcomes ─────────────────────────────────────────────────
    def record_outcome(
        self, *, post_id: str, platform: str, verdict: str,
        items_received: int, replies_sent: int, detail: str = "",
    ) -> bool:
        """Record a post's engagement outcome ONCE (first wins)."""
        cur = self._db.execute(
            "INSERT OR IGNORE INTO engagement_outcomes (id,"
            " post_id, platform, verdict, items_received,"
            " replies_sent, detail, created_at) VALUES"
            " (?, ?, ?, ?, ?, ?, ?, ?)",
            (new_id(), post_id, platform, verdict,
             int(items_received), int(replies_sent), detail,
             time.time()))
        return bool(getattr(cur, "rowcount", 0))

    def outcome_for(self, post_id: str) -> dict[str, Any] | None:
        rows = self._db.query(
            "SELECT * FROM engagement_outcomes WHERE post_id = ?",
            (post_id,))
        return rows[0] if rows else None


# ── capability ───────────────────────────────────────────────────────

def adapter_supports_engagement(adapter: PlatformAdapter) -> bool:
    """True when the adapter really implements engagement reads.

    The base class answers ``[]`` for every platform, so — exactly
    like the metrics spine's probe — capability is whether an
    override exists anywhere below :class:`PlatformAdapter`.
    """
    return type(adapter).read_engagement is not PlatformAdapter.read_engagement


def platform_truth(
    adapters: dict[str, PlatformAdapter],
) -> dict[str, dict[str, bool]]:
    """Per-platform {read, reply} capability, structurally read."""
    return {
        name: {
            "read": adapter_supports_engagement(adapter),
            "reply": bool(getattr(adapter, "supports_reply", False)),
        }
        for name, adapter in sorted(adapters.items())
    }


# ── the poll ─────────────────────────────────────────────────────────

def _feed_engagement_outcome(
    brief_store: Any, row: dict[str, Any], received: int,
    replies_sent: int,
) -> str:
    """Feed one post's engagement volume into the briefs loop.

    The metrics spine's join discipline, verbatim in shape: exact
    content + platform identity (publishing writes no brief-draft
    id onto a post row, so content is the only deterministic
    link). Zero or several linkable candidates means NO feed —
    ``unlinked`` / ``ambiguous`` — never a guess.
    """
    candidates = brief_store.find_by_content(
        str(row.get("content") or ""),
        platform=str(row.get("platform") or ""))
    linkable = [
        d for d in candidates
        if d.brief_id and not d.published_at and not d.outcome_metrics
    ]
    if not linkable:
        return "already_recorded" if candidates else "unlinked"
    if len(linkable) > 1:
        return "ambiguous"
    payload = {
        "engagement_items": int(received),
        "engagement_replies": int(replies_sent),
        # The briefs contract takes one engagement figure; for this
        # feed it is the count of responses the post drew — a real
        # measured number, reported unscaled.
        "engagement_score": int(received),
    }
    return "fed" if brief_store.record_outcome(
        linkable[0].draft_id, payload) else "feed_failed"


def poll_engagement(
    manager: Any,
    *,
    now: float | None = None,
    limit: int = DEFAULT_LIMIT,
    window_days: int = WINDOW_DAYS,
    outcome_min_age_s: float = OUTCOME_MIN_AGE_S,
    brief_store: Any = None,
    feed_outcomes: bool = True,
    store: EngagementStore | None = None,
) -> dict[str, Any]:
    """One engagement poll over the owner's posted rows.

    ``manager`` is a :class:`SocialManager` with adapters
    registered. Returns the run's accounting — every considered
    row lands in exactly one read bucket, every failure names its
    platform, account handle and the post's external id. A
    platform's failure never blocks the others. Never raises for
    a per-row problem; a spine-level failure (no database)
    raises, because a scheduled run that could not run at all
    must record as failed.
    """
    started = time.perf_counter()
    moment = float(now if now is not None else time.time())
    store = store or EngagementStore(manager.db)
    report: dict[str, Any] = {
        "ok": True,
        "considered": 0,
        "read_ok": 0,
        "new_items": 0,
        "duplicates": 0,
        "unavailable": 0,
        "unavailable_platforms": [],
        "gated": 0,
        "gated_platforms": [],
        "failed": 0,
        "skipped_no_account": 0,
        "skipped_no_adapter": 0,
        "outcomes_fed": 0,
        "outcomes_unlinked": 0,
        "outcomes_ambiguous": 0,
        "outcomes_already_recorded": 0,
        "outcomes_failed": 0,
        "outcomes_pending_age": 0,
        "escalated": 0,
        "failures": [],
    }
    rows = manager.db.query(
        "SELECT * FROM social_posts WHERE status = 'posted' "
        "AND posted_at IS NOT NULL AND external_id != '' "
        "AND posted_at > ? AND posted_at <= ? "
        "ORDER BY posted_at DESC LIMIT ?",
        (moment - window_days * 86400.0, moment,
         max(1, int(limit))))
    unavailable: set[str] = set()
    gated: dict[str, str] = {}
    briefs: Any = brief_store
    briefs_state = "pending"  # pending | ready | broken
    for raw_row in rows:
        row = dict(raw_row)
        report["considered"] += 1
        post_id = str(row.get("id") or "")
        platform = str(row.get("platform") or "")
        external_id = str(row.get("external_id") or "")
        account_row = manager.accounts.get(str(row.get("account_id") or ""))
        if account_row is None:
            report["skipped_no_account"] += 1
            _log.warning(
                "engagement poll: post %s on %s has no connected "
                "account", external_id, platform)
            continue
        from .base import Account

        account = Account.from_row(account_row)
        adapter = manager.adapters.get(platform)
        if adapter is None:
            report["skipped_no_adapter"] += 1
            _log.warning(
                "engagement poll: no adapter registered for %s "
                "(post %s)", platform, external_id)
            continue
        # The personal-surface live gate, BEFORE any read attempt:
        # a gated platform is recorded with the gate's own words
        # and its adapter is never called.
        gate_fn = getattr(adapter, "engagement_gate", None)
        if callable(gate_fn):
            try:
                verdict = str(gate_fn() or "")
            except Exception as exc:  # noqa: BLE001 - verdict, not crash
                verdict = f"the live gate could not be read ({exc})"
            if verdict:
                report["gated"] += 1
                gated.setdefault(platform, verdict)
                continue
        if not adapter_supports_engagement(adapter):
            report["unavailable"] += 1
            unavailable.add(platform)
            continue
        try:
            found = adapter.read_engagement(account, external_id)
        except Exception as exc:  # noqa: BLE001 - a read failure is data
            report["failed"] += 1
            _log.warning(
                "engagement poll failed for %s post %s (@%s): %s: %s",
                platform, external_id, account.handle,
                type(exc).__name__, exc)
            if len(report["failures"]) < 10:
                report["failures"].append({
                    "platform": platform, "handle": account.handle,
                    "external_id": external_id,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            continue
        report["read_ok"] += 1
        for item in found or []:
            item_id = str(item.get("item_id") or "")
            if not item_id:
                continue
            inserted = store.add_item(
                platform=platform, post_id=post_id,
                post_external_id=external_id,
                account_id=str(row.get("account_id") or ""),
                item_external_id=item_id,
                author=str(item.get("author") or ""),
                text=str(item.get("text") or ""),
                kind=str(item.get("kind") or "reply"),
                arrived_at=float(item.get("created_at") or 0)
                or moment)
            if inserted:
                report["new_items"] += 1
            else:
                report["duplicates"] += 1
        # The outcome feed: once per post, at a fair age, and only
        # on a run whose read of THIS post succeeded (the counts
        # stamped are this run's fresh truth).
        if not feed_outcomes:
            continue
        if store.outcome_for(post_id) is not None:
            continue
        if moment - float(row.get("posted_at") or 0.0) < outcome_min_age_s:
            report["outcomes_pending_age"] += 1
            continue
        if briefs_state == "pending":
            try:
                if briefs is None:
                    from ..marketing.briefs import BriefStore

                    briefs = BriefStore()
                briefs_state = "ready"
            except Exception as exc:  # noqa: BLE001 - feeding is best-effort
                briefs_state = "broken"
                _log.warning(
                    "engagement poll: brief store unavailable: %s", exc)
        if briefs_state != "ready":
            report["outcomes_failed"] += 1
            continue
        counts = store.counts_for_post(post_id)
        try:
            verdict = _feed_engagement_outcome(
                briefs, row, counts["received"], counts["replies_sent"])
        except Exception as exc:  # noqa: BLE001 - never sink the run
            verdict = "feed_failed"
            _log.warning(
                "engagement poll: outcome feed failed for %s post "
                "%s: %s", platform, external_id, exc)
        if verdict == "feed_failed":
            report["outcomes_failed"] += 1
            continue  # unrecorded: the next cycle retries this post
        store.record_outcome(
            post_id=post_id, platform=platform, verdict=verdict,
            items_received=counts["received"],
            replies_sent=counts["replies_sent"])
        report[{
            "fed": "outcomes_fed",
            "unlinked": "outcomes_unlinked",
            "ambiguous": "outcomes_ambiguous",
            "already_recorded": "outcomes_already_recorded",
        }[verdict]] += 1
    # Escalation: an open item whose post row is GONE from the
    # ledger cannot be routed by Devon — it becomes the owner's
    # call, named, never silently dropped.
    for item in store.items(states=OPEN_STATES, limit=200):
        if manager.posts.get(str(item.get("post_id") or "")) is None:
            store.set_state(
                item["id"], STATE_NEEDS_OWNER,
                detail="the post this answered is gone from the "
                       "ledger — yours to judge")
            report["escalated"] += 1
    report["unavailable_platforms"] = sorted(unavailable)
    report["gated_platforms"] = sorted(gated)
    report["gated_reasons"] = gated
    report["seconds"] = round(time.perf_counter() - started, 3)
    report["text"] = (
        f"engagement poll: {report['new_items']} new item(s), "
        f"{report['duplicates']} already held, "
        f"{report['unavailable']} unavailable "
        f"({', '.join(report['unavailable_platforms']) or 'none'}), "
        f"{report['gated']} gated "
        f"({', '.join(report['gated_platforms']) or 'none'}), "
        f"{report['failed']} failed, "
        f"{report['outcomes_fed']} outcomes fed to briefs")
    return report


# ── reply drafting ───────────────────────────────────────────────────

def _template_reply(item: dict[str, Any]) -> str:
    """The deterministic fallback draft: courteous, fact-free.

    A starting point for his edit in review — it thanks, it
    acknowledges, it promises nothing that is not already true.
    """
    author = str(item.get("author") or "").strip()
    name = author.split()[0] if author else "there"
    text = str(item.get("text") or "").rstrip()
    if text.endswith("?"):
        return (f"Good question, {name} — thanks for asking it "
                "here. I'd rather answer it properly than dash off "
                "half an answer, so: noted, and I'll come back to "
                "it.")
    return (f"Thanks, {name} — appreciate you taking the time to "
            "reply. Glad it landed; more on this soon.")


def draft_reply(
    store: EngagementStore, queue: Any, item_id: str, *,
    llm_fn: Callable[[str], str] | None = None,
    adapters: dict[str, PlatformAdapter] | None = None,
) -> dict[str, Any]:
    """Draft ONE reply to an engagement item, through the queue.

    The draft lands at ``pending_review`` on /drafts — the #43
    rule owns every outward step from there, unchanged. The
    drafter rung is recorded in the draft's metadata (``brain``
    when the brain wrote it, ``template`` for the deterministic
    fallback), never disguised.

    Threading honesty: the executable ``reply_to`` link is set
    ONLY when the platform's adapter declares ``supports_reply``
    — then approving + posting the draft threads it onto the
    original post. Otherwise the draft is a plain post draft and
    its metadata (and the return value) SAY so.
    """
    item = store.get(item_id)
    if item is None:
        return {"ok": False, "error": "that engagement item is gone"}
    if item["state"] in TERMINAL_STATES:
        return {"ok": False,
                "error": f"that item is already {item['state']} — "
                         "nothing to draft"}
    platform = str(item.get("platform") or "")
    adapter = (adapters or {}).get(platform)
    reply_native = bool(
        adapter is not None and getattr(adapter, "supports_reply", False))
    text = ""
    rung = "template"
    if llm_fn is not None:
        limit = int(getattr(adapter, "max_chars", 0) or 0)
        prompt = (
            "Draft a short, warm reply to a response on the "
            f"owner's {platform} post. The response, from "
            f"{item.get('author') or 'a reader'}: "
            f"\"{item.get('text') or ''}\". Reply in the owner's "
            "voice: direct, no hype, no promises about things "
            "that have not happened. "
            + (f"Keep it under {limit} characters. "
               if limit else "")
            + "Return only the reply text.")
        try:
            out = llm_fn(prompt)
        except Exception:  # noqa: BLE001 - fallback is the design
            out = ""
        if out and str(out).strip():
            text = str(out).strip()
            rung = "brain"
    if not text:
        text = _template_reply(item)
        rung = "template"
    metadata: dict[str, Any] = {
        "origin": "engagement",
        "engagement_item": item["id"],
        "platform": platform,
        "in_reply_to": str(item.get("post_external_id") or ""),
        "author": str(item.get("author") or ""),
        "drafter": rung,
        "reply_native": reply_native,
    }
    if reply_native:
        metadata["reply_to"] = str(item.get("post_external_id") or "")
    else:
        metadata["reply_note"] = (
            f"posts as a standalone post on {platform} — "
            f"{platform} has no threaded-reply path through its "
            "adapter, so this answers in the open rather than "
            "pretending to thread")
    draft = queue.create_draft(
        text, platforms=[platform] if platform else [],
        metadata=metadata)
    queue.propose(draft.id)
    store.set_state(
        item["id"],
        STATE_TRIAGED if item["state"] == STATE_NEW else item["state"],
        reply_draft_id=draft.id)
    return {"ok": True, "draft_id": draft.id, "text": text,
            "reply_native": reply_native, "drafter": rung,
            "platform": platform}


def mark_reply_posted(
    store: EngagementStore, draft: Any, *, external_id: str = "",
) -> bool:
    """Flip a posted reply draft's engagement item to ``replied``.

    Called from the draft-post path (``execute_post``) once the
    draft is actually posted. Idempotent and narrow: only the
    item this draft was drafted for, only from an open state.
    """
    item_id = str((getattr(draft, "metadata", None) or {}).get(
        "engagement_item") or "")
    if not item_id:
        return False
    item = store.get(item_id)
    if item is None or item["state"] in TERMINAL_STATES:
        return False
    if str(item.get("reply_draft_id") or "") != str(
            getattr(draft, "id", "") or ""):
        return False
    store.set_state(item_id, STATE_REPLIED,
                    reply_post_external_id=str(external_id or ""))
    return True


def reconcile_replied(store: EngagementStore, queue: Any) -> int:
    """Flip items whose reply draft the queue shows as POSTED.

    The board/poll belt to execute_post's braces: whatever path
    posted the draft, the queue's own state is the truth, and an
    item whose draft posted reads ``replied`` here.
    """
    flipped = 0
    for item in store.items(states=OPEN_STATES, limit=200):
        draft_id = str(item.get("reply_draft_id") or "")
        if not draft_id:
            continue
        try:
            draft = queue.get(draft_id)
        except Exception:  # noqa: BLE001 - a read hiccup skips one
            continue
        if draft is not None and str(
                getattr(draft, "status", "")) == "posted":
            store.set_state(item["id"], STATE_REPLIED)
            flipped += 1
    return flipped


# ── the board ────────────────────────────────────────────────────────

def board_items(store: EngagementStore) -> list[dict[str, Any]]:
    """The board's ONE ordering — board, briefing and verbs all
    number from this list: what needs him first (``new``, then
    escalated ``needs_owner``), oldest first within a class, then
    what he has in hand (``triaged``), oldest first."""
    new = store.items(states=(STATE_NEW,))
    escalated = store.items(states=(STATE_NEEDS_OWNER,))
    in_hand = store.items(states=(STATE_TRIAGED,))
    return new + escalated + in_hand


def _age_label(ts: float, *, now: float | None = None) -> str:
    import datetime

    if not ts:
        return ""
    return datetime.datetime.fromtimestamp(float(ts)).strftime(
        "%b %d, %H:%M")


def _excerpt(text: str, limit: int = 90) -> str:
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _item_line(index: int, item: dict[str, Any]) -> str:
    author = str(item.get("author") or "someone")
    platform = str(item.get("platform") or "")
    kind = str(item.get("kind") or "reply")
    when = _age_label(float(item.get("arrived_at") or 0))
    head = f"{index}. {author} · {platform} · {kind}"
    if when:
        head += f" · {when}"
    if item["state"] == STATE_NEEDS_OWNER:
        head += " · ⚠️ needs you personally"
    line = head + f"\n   \"{_excerpt(str(item.get('text') or ''))}\""
    if item.get("reply_draft_id"):
        line += "\n   ↳ a reply draft is waiting on /drafts"
    return line


def render_board(
    store: EngagementStore, *,
    truth: dict[str, dict[str, bool]] | None = None,
) -> str:
    """The /engage board text: needs-him first, unavailable named."""
    items = board_items(store)
    lines = ["💬 **engagement — replies & comments on your posts**",
             ""]
    if not items:
        lines.append("nothing is waiting on you — no unanswered "
                     "replies on your posts right now.")
    else:
        fresh = [i for i in items if i["state"] == STATE_NEW]
        escalated = [i for i in items
                     if i["state"] == STATE_NEEDS_OWNER]
        in_hand = [i for i in items if i["state"] == STATE_TRIAGED]
        idx = 0
        if fresh:
            lines.append("**needs you** (oldest first)")
            for item in fresh:
                idx += 1
                lines.append(_item_line(idx, item))
            lines.append("")
        if escalated:
            lines.append("**escalated — Devon cannot route these**")
            for item in escalated:
                idx += 1
                lines.append(_item_line(idx, item))
            lines.append("")
        if in_hand:
            lines.append("**in hand** (seen; draft a reply, ignore, "
                         "or mark done)")
            for item in in_hand:
                idx += 1
                lines.append(_item_line(idx, item))
            lines.append("")
        lines.append("verbs: /engage show <n> · /engage reply <n> "
                     "(drafts a reply onto /drafts) · /engage "
                     "ignore <n> · /engage done <n>")
    if truth:
        watching = sorted(
            name for name, t in truth.items() if t.get("read"))
        blind = sorted(
            name for name, t in truth.items() if not t.get("read"))
        lines.append("")
        if watching:
            lines.append("watching: " + ", ".join(watching))
        if blind:
            named = []
            for name in blind:
                reason = UNAVAILABLE_REASONS.get(name)
                named.append(f"{name} ({reason})" if reason else name)
            lines.append(
                "named unavailable — no replies are read from: "
                + "; ".join(named))
    return "\n".join(lines)


def render_item(item: dict[str, Any], index: int | None = None) -> str:
    """One item's full card for /engage show <n>."""
    lines = [
        f"💬 {item.get('author') or 'someone'} · "
        f"{item.get('platform') or ''} · {item.get('kind') or 'reply'}",
        f"state: {item.get('state') or ''}"
        + (f" · arrived {_age_label(float(item.get('arrived_at') or 0))}"
           if item.get("arrived_at") else ""),
        "",
        str(item.get("text") or ""),
    ]
    if item.get("detail"):
        lines += ["", f"⚠️ {item['detail']}"]
    if item.get("reply_draft_id"):
        lines += ["", "↳ a reply draft for this is on /drafts — "
                      "it posts only on your approval there"]
    n = f" {index}" if index else " <n>"
    lines += ["", f"reply: /engage reply{n} — drafts it · ignore: "
                  f"/engage ignore{n} · done: /engage done{n}"]
    return "\n".join(lines)

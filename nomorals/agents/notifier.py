"""Notifier: durable alerts, delivered to the owner on every live channel.

Every subsystem that produces something the owner should see (arena build
reviews, research suggestions, news digests, finished directives, backup
results) funnels through here instead of each wiring its own send loop.

* **Durable first** — the row lands in ``notifications`` before any send, so
  a dead platform never loses an alert; undelivered rows can be re-sent.
* **Multi-channel** — WhatsApp and/or Telegram (or Discord/local) whichever
  are running in the session; the owner is on at least one.
* **Feature-gated** — the ``notifier`` flag (``/features notifier on|off``).
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import time
from typing import Any

from ..core.ids import new_id
from ..core.logging_setup import get_logger
from .failures import record_failure

_log = get_logger(__name__)

__all__ = ["Notifier", "notify", "notify_doc", "resolve_gateway",
           "DeliveryOutcome",
           "ATTEMPT_SENT", "ATTEMPT_FAILED", "ATTEMPT_SKIPPED",
           "STATE_DIGEST", "STATE_LOGGED"]


def resolve_gateway(context: Any) -> Any | None:
    """Find the live chat gateway for a context.

    The runtime stores it in ``context.extras["gateway"]`` (see
    ``PartnerRuntime``); older/CLI code may set ``context.gateway``
    directly.  Returns ``None`` when there is no live gateway (CLI
    contexts) — delivery then degrades to persist-only.
    """
    gw = getattr(context, "gateway", None)
    if gw is not None:
        return gw
    try:
        extras = getattr(context, "extras", None)
        if isinstance(extras, dict):
            return extras.get("gateway")
    except Exception as exc:  # noqa: BLE001 - gateway resolution is best-effort
        record_failure("notifier.resolve_gateway", exc)
        pass
    return None


#: proactive notification kinds -> PartnerSettings toggle each one obeys.
#: Only these kinds go through the proactive gate; every other kind keeps
#: the historical behavior (feature flag ``notifier`` only).
PROACTIVE_KIND_SETTINGS: dict[str, str] = {
    "briefing": "proactive_briefing",
    "watcher": "proactive_watchers",
}

#: delivery states recorded in ``notifications.delivery_state``
STATE_SENT = "sent"
STATE_FAILED = "failed"            # gateway live, but no owner channel reached
STATE_PENDING = "pending"          # stored; no live gateway (redeliver later)
STATE_HELD_QUIET = "held-quiet-hours"
STATE_DISABLED = "disabled"        # proactive master/kind switch off
STATE_MUTED = "muted"              # feature flag ``notifier`` off
STATE_DEDUPED = "deduped"
STATE_DEAD = "dead"                # dead-letter: delivery retries exhausted
#: alert-policy tiers (Pass 7): the row is the whole delivery — a
#: DIGEST row waits to be read batched on /notify, a LOGGED row is
#: silent housekeeping kept only as journal evidence.  Both are
#: intentionally unsent and never picked up by redelivery.
STATE_DIGEST = "digest"
STATE_LOGGED = "logged"

#: delivery states that the redelivery queue is allowed to pick up.
#: Intentionally-unsent states (disabled/deduped/muted/held-quiet-hours)
#: and the terminal ``dead`` state are never retried.
RETRYABLE_STATES = (STATE_PENDING, STATE_FAILED)

#: per-channel attempt states recorded in ``notification_deliveries`` —
#: one row per owner target per delivery pass, so ``/notify`` can show
#: *why* a send didn't land instead of a bare "failed".
ATTEMPT_SENT = "sent"
ATTEMPT_FAILED = "failed"
ATTEMPT_SKIPPED = "skipped"   # target resolved but not attempted
                              # (platform not running, unresolvable entry)


class DeliveryOutcome:
    """Result of one delivery pass over the resolved owner targets.

    ``attempts`` is one dict per target — ``platform``, ``chat_id``,
    ``source`` (raw config entry), ``note`` (how the platform was
    resolved), ``state`` (sent/failed/skipped), ``message_id`` on
    success, ``error`` on failure/skip, ``attempted_at``.  Truthy when
    at least one channel was reached, so existing ``if outcome:``
    call sites keep working.
    """

    def __init__(self, attempts: list[dict[str, Any]] | None = None) -> None:
        self.attempts: list[dict[str, Any]] = list(attempts or [])

    @property
    def delivered(self) -> int:
        return sum(1 for a in self.attempts
                   if a.get("state") == ATTEMPT_SENT)

    @property
    def first_channel(self) -> str:
        for a in self.attempts:
            if a.get("state") == ATTEMPT_SENT:
                return str(a.get("platform") or "")
        return ""

    def __bool__(self) -> bool:
        return self.delivered > 0

    def __len__(self) -> int:
        return len(self.attempts)

    def describe_failures(self) -> str:
        parts = []
        for a in self.attempts:
            if a.get("state") == ATTEMPT_SENT:
                continue
            where = (f"{a.get('platform') or '?'}:{a.get('chat_id') or '?'}"
                     if a.get("platform") else f"entry {a.get('source')!r}")
            reason = a.get("error") or a.get("state")
            parts.append(f"{where} {a.get('state')} ({reason})")
        return "; ".join(parts)

#: Delivery retry policy — exponential backoff with jitter, the standard
#: shape for transient delivery failures (AWS Architecture Blog,
#: "Exponential Backoff And Jitter").  ``attempt`` is 1-based:
#: 60s, 120s, 240s, … capped at 6h.  After
#: :data:`DELIVERY_MAX_ATTEMPTS` failed attempts the row is dead-lettered
#: (``delivery_state = "dead"``) instead of retried forever — a dead
#: channel must page once, not spin the queue for weeks.
DELIVERY_MAX_ATTEMPTS = 8
DELIVERY_BACKOFF_BASE_SECONDS = 60.0
DELIVERY_BACKOFF_CAP_SECONDS = 6 * 3600.0


def delivery_backoff(attempt: int) -> float:
    """Seconds to wait before redelivery attempt ``attempt`` (1-based)."""
    attempt = max(1, int(attempt))
    backoff = DELIVERY_BACKOFF_BASE_SECONDS * (2.0 ** (attempt - 1))
    backoff = min(backoff, DELIVERY_BACKOFF_CAP_SECONDS)
    # full jitter, capped: spreads clustered retries without the wait
    # drifting far from the schedule.
    return backoff + random.uniform(0.0, min(30.0, backoff * 0.25))


#: env kill-switch for the Termux system-notification fallback.
#: Default on — on Termux an Android notification is the last-resort
#: channel when no chat adapter is live in the session.
TERMUX_NOTIFY_ENV = "NM_TERMUX_NOTIFY"

_termux_notify_bin: str | bool | None = None  # cached shutil.which result

#: kinds exempt from Notifier-level quiet hours.  ``briefing`` is a
#: scheduled send the owner explicitly asked for at a chosen time (like an
#: alarm); ``watcher`` alerts carry their own per-watcher quiet-hours
#: logic in the alerting pipeline (watchers.py) — holding them twice
#: would delay urgent hits for no reason.
QUIET_HOURS_EXEMPT = ("briefing", "watcher")


def _partner(context: Any) -> Any | None:
    return getattr(getattr(context, "settings", None), "partner", None)


def _in_quiet_hours_now(context: Any, now: float | None = None) -> bool:
    """True when ``now`` falls in the partner's proactive quiet window."""
    partner = _partner(context)
    if partner is None:
        return False
    try:
        start = int(getattr(partner, "quiet_start", 22) or 0)
        end = int(getattr(partner, "quiet_end", 8) or 0)
    except (TypeError, ValueError):
        return False
    if start == end:
        return False
    # owner's timezone, same resolution order as the briefing
    tzname = None
    settings = getattr(context, "settings", None)
    for attr in ("timezone", "tz", "owner_timezone"):
        tzname = getattr(settings, attr, None) if settings else None
        if tzname:
            break
    if not tzname and partner is not None:
        for attr in ("timezone", "tz"):
            tzname = getattr(partner, attr, None)
            if tzname:
                break
    try:
        from zoneinfo import ZoneInfo
        from datetime import datetime
        tz = ZoneInfo(str(tzname)) if tzname else None
        hour = datetime.now(tz).hour if tz else datetime.now().hour
    except Exception as exc:  # noqa: BLE001 - tz data missing -> wall clock
        record_failure("notifier._in_quiet_hours_now.tz", exc)
        from datetime import datetime
        hour = (datetime.fromtimestamp(now).hour if now
                else datetime.now().hour)
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


def proactive_gate(context: Any, kind: str) -> str | None:
    """Decide whether a proactive ``kind`` may send right now.

    Returns ``None`` when sending is allowed, otherwise a
    ``delivery_state`` explaining the hold (``"disabled"`` /
    ``"held-quiet-hours"``).  ``critical`` sends bypass the gate in
    ``publish`` — this helper only encodes the non-critical policy.
    """
    setting = PROACTIVE_KIND_SETTINGS.get(kind)
    if setting is None:
        return None  # not a proactive kind — historical behavior
    partner = _partner(context)
    if partner is None:
        return None  # no settings -> defaults allow (offline tests, CLI)
    if not bool(getattr(partner, "proactive_enabled", True)):
        return STATE_DISABLED
    if not bool(getattr(partner, setting, True)):
        return STATE_DISABLED
    if kind not in QUIET_HOURS_EXEMPT and _in_quiet_hours_now(context):
        return STATE_HELD_QUIET
    return None


class Notifier:
    def __init__(self, context: Any, gateway: Any = None) -> None:
        self.context = context
        self.db = getattr(context, "db", None)
        self.settings = getattr(context, "settings", None)
        #: live gateway when one exists; resolved from the context when not
        #: passed explicitly (the runtime keeps it in context.extras).
        #: None in CLI contexts (delivery = persist only).
        self.gateway = gateway if gateway is not None else resolve_gateway(context)

    #: identical (kind, title) within this window collapses to one alert
    DEDUPE_WINDOW_SECONDS = 600.0

    # ── core ─────────────────────────────────────────────────────────────────
    def publish(
        self,
        kind: str,
        title: str = "",
        body: str = "",
        *,
        force: bool = False,
        critical: bool = False,
        channels: list[str] | None = None,
        proactive_check: bool = True,
        output_kind: str | None = None,
        output_meta: dict[str, Any] | None = None,
        alert_subject: str = "",
        alert_hints: Any = None,
        doc: Any = None,
    ) -> dict[str, Any]:
        """Persist an alert and best-effort deliver it. Never raises.

        ``critical`` bypasses the feature flag and the proactive gates:
        build failures and security-relevant events must get through even
        when the owner has the notifier muted.  It does NOT bypass dedupe
        — the same critical alert re-firing inside the dedupe window is
        spam, not signal.  ``force`` bypasses dedupe and the ``notifier``
        feature flag, but still respects the proactive master/kind
        switches — an explicit "don't push to me" always wins.

        ``channels`` optionally restricts delivery to a subset of the
        owner's platforms (e.g. ``["telegram"]``); ``None`` (default)
        delivers to every live owner channel, preserving the historical
        behavior for all existing callers.

        ``proactive_check`` (default on) applies the proactive gate to
        proactive kinds (``briefing``, ``watcher``): the master
        ``proactive_enabled`` switch, the per-kind toggle, and quiet
        hours.  A held send is still stored — with ``delivery_state``
        ``"disabled"`` / ``"held-quiet-hours"`` — so the owner can see
        what didn't go out and why.

        ``output_kind`` / ``output_meta`` describe the output being
        pushed (e.g. ``"scheduled"`` + ``{"job": "memory-consolidation"}``)
        so delivery attaches ADAPTIVE buttons about that output. A push
        has no triggering command, so without these it gets NO buttons
        (explicit empty) — never stale Retry/Expand/Next inherited
        from an old command.

        ``alert_hints`` (Pass 7, :class:`~.alert_policy.AlertHints` or
        a plain dict) classifies an *unsolicited* alert for the ONE
        paging policy (:mod:`.alert_policy`): the decision may keep
        the alert out of the owner's chat entirely — DIGEST (stored,
        read batched on ``/notify``) or LOG (stored silently) — or
        PAGE it as today.  A recurrence page bypasses dedupe (the
        repeat IS the signal); failure/actionable pages keep dedupe.
        Every decided page carries the policy's decision line in its
        body, stating what escalated and naming the subject.  No
        hints → historical behaviour, untouched.

        ``doc`` (Formatting Pass F2) is a
        :class:`~nomorals.social.chat.present.Doc` the producer built
        instead of strings: the stored title/body derive from it
        (title ← ``doc.title`` when no title is given; body ← its
        canonical dialect body), so dedupe, ``/notify``, the Termux
        fallback and redelivery treat it exactly like a string alert,
        while delivery renders it per target platform through the
        presentation layer — HTML + explicit parse mode on the
        Telegram family, the native dialect elsewhere — with the
        Doc's own actions as its buttons.  Since F7 the row also
        stores the Doc's serialized form (``doc_json``, migration
        88), so redelivery reconstructs the Doc and re-renders it
        per platform WITH its structure; rows written before F7
        (NULL ``doc_json``) redeliver through the string path,
        formatted by the gateway like any dialect body.  A non-Doc
        ``doc`` is a producer bug: logged and ignored, never a lost
        alert.
        """
        doc_json = ""
        if doc is not None:
            from ..social.chat.present import Doc as _Doc
            from ..social.chat.present import body_text as _doc_body

            if isinstance(doc, _Doc):
                if not title:
                    title = doc.title
                if not body:
                    body = _doc_body(doc)
            else:
                _log.warning("notifier: publish(doc=...) got a non-Doc "
                             "%s — ignoring it, delivering strings",
                             type(doc).__name__)
                doc = None
            if doc is not None:
                doc_json = self._serialize_doc(doc)
        decision = None
        if alert_hints is not None and self.db is not None:
            from .alert_policy import AlertHints, AlertPolicy, \
                TIER_DIGEST, TIER_LOG, TIER_PAGE

            hints = AlertHints.coerce(alert_hints)
            if hints is not None:
                decision = AlertPolicy.from_context(self.context).decide(
                    self.db, kind=kind, subject=alert_subject, hints=hints)
                if decision.tier in (TIER_LOG, TIER_DIGEST):
                    state = (STATE_LOGGED if decision.tier == TIER_LOG
                             else STATE_DIGEST)
                    result = self._store(kind, title, body, delivered=0,
                                         delivery_state=state,
                                         doc_json=doc_json)
                    result["attempts"] = []
                    result["tier"] = decision.tier
                    result["tier_reason"] = decision.reason
                    result["subject"] = decision.subject
                    return result
                # TIER_PAGE: deliver as today.  Only a recurrence
                # page forces past dedupe (the repeat IS the signal);
                # failure/actionable pages keep dedupe, so a
                # persisting failure re-pages on the dedupe cadence
                # rather than once per tick.
                if decision.force:
                    force = True
                _line = decision.body_line()
                body = (f"{body}\n{_line}" if body else _line)
                if doc is not None:
                    # The Doc render is what the owner reads — the
                    # escalation line rides its footer too.
                    from dataclasses import replace as _dc_replace

                    doc = _dc_replace(
                        doc, footer=(f"{doc.footer}\n{_line}"
                                     if doc.footer else _line))
                    # the stored form must be the Doc actually
                    # delivered (decision line included)
                    doc_json = self._serialize_doc(doc)
        if self.db is not None:
            # DEDUPE CHOKE POINT — every proactive send funnels through
            # here, including critical ones.  The same (kind, title) inside
            # the window collapses to one alert: a re-firing urgent alert
            # is spam, not signal.  ``force`` is the only escape hatch
            # (explicit owner/manual sends).
            if not force and self._is_duplicate(kind, title):
                # same alert already went out in the window — skip, don't re-spam
                return {"id": "", "kind": kind, "title": title,
                        "delivered": False, "deduped": True,
                        "delivery_state": STATE_DEDUPED}
            if critical:
                pass  # critical skips the feature flag + proactive gates
            else:
                from .features import feature_enabled

                if not force and not feature_enabled(self.context, "notifier"):
                    # still record it — the owner can read the queue with /notify
                    result = self._store(kind, title, body, delivered=0,
                                         delivery_state=STATE_MUTED,
                                         doc_json=doc_json)
                    result["attempts"] = []
                    return result
                if proactive_check:
                    gate = proactive_gate(self.context, kind)
                    if gate is not None:
                        result = self._store(kind, title, body, delivered=0,
                                             delivery_state=gate,
                                             doc_json=doc_json)
                        result["attempts"] = []
                        return result
        nid = new_id()
        if self.gateway is None:
            # no chat gateway at all — still try the Termux system
            # notification before parking the alert as pending.
            if self._termux_fallback(title, body):
                result = self._store(kind, title, body, delivered=1,
                                     delivery_state=STATE_SENT, channel="termux",
                                     nid=nid, doc_json=doc_json)
                attempts = [self._termux_attempt(title)]
                self._record_attempts(nid, attempts)
                result["attempts"] = attempts
                return result
            # next_retry_at = 0: the first redelivery attempt is due
            # immediately (a hot-starting platform should drain the queue
            # at once); backoff engages after an attempt actually fails.
            result = self._store(kind, title, body, delivered=0,
                                 delivery_state=STATE_PENDING, nid=nid,
                                 doc_json=doc_json)
            result["attempts"] = []
            return result
        outcome = self._deliver(kind, title, body, channels=channels,
                                output_kind=output_kind,
                                output_meta=output_meta, doc=doc)
        reached = outcome.delivered
        if reached:
            result = self._store(kind, title, body, delivered=1,
                                 delivery_state=STATE_SENT,
                                 channel=outcome.first_channel, nid=nid,
                                 doc_json=doc_json)
            self._record_attempts(nid, outcome.attempts)
            result["attempts"] = outcome.attempts
            return result
        # No owner chat channel is live in this session (the Termux
        # reality: Discord banned the token, the Telegram user session
        # isn't connected here, …).  Last resort before parking the
        # alert: an Android system notification via termux-notification —
        # it lands in the shade of the phone running the bot.
        if self._termux_fallback(title, body):
            result = self._store(kind, title, body, delivered=1,
                                 delivery_state=STATE_SENT, channel="termux",
                                 nid=nid, doc_json=doc_json)
            attempts = outcome.attempts + [self._termux_attempt(title)]
            self._record_attempts(nid, attempts)
            result["attempts"] = attempts
            return result
        if not reached:
            _log.warning("notifier: %r reached 0 chat channels%s", title,
                         (": " + outcome.describe_failures())
                         if outcome.attempts else " (no owner targets resolved)")
        state = STATE_SENT if reached else STATE_FAILED
        result = self._store(kind, title, body,
                             delivered=1 if reached else 0,
                             delivery_state=state, nid=nid,
                             doc_json=doc_json)
        self._record_attempts(nid, outcome.attempts)
        result["attempts"] = outcome.attempts
        return result

    def publish_doc(self, kind: str, doc: Any, **kw: Any) -> dict[str, Any]:
        """Publish a producer-built presentation Doc (Formatting F2).

        Sugar over ``publish(kind, "", "", doc=doc)``: the stored
        title/body derive from the Doc, delivery renders it per
        platform.  Every other ``publish`` keyword (``critical``,
        ``channels``, ``alert_hints``, …) passes through.
        """
        return self.publish(kind, "", "", doc=doc, **kw)

    @staticmethod
    def _termux_attempt(title: str) -> dict[str, Any]:
        return {"platform": "termux", "chat_id": "", "source": "",
                "note": "last-resort system notification",
                "state": ATTEMPT_SENT, "message_id": "", "error": "",
                "attempted_at": time.time()}

    def _is_duplicate(self, kind: str, title: str) -> bool:
        """True when the same (kind, title) alerted within the dedupe window.

        Alert-policy DIGEST/LOG rows never "went out", so they do not
        count as the alert having fired — a subject whose first
        instances logged silently must still be able to page when the
        policy escalates it inside the window.
        """
        try:
            cutoff = time.time() - self.DEDUPE_WINDOW_SECONDS
            row = self.db.query_one(
                "SELECT id FROM notifications WHERE kind = ? AND title = ? "
                "AND created_at > ? AND COALESCE(delivery_state, '') "
                "NOT IN (?, ?)",
                (kind, title, cutoff, STATE_DIGEST, STATE_LOGGED),
            )
            return row is not None
        except Exception as exc:  # noqa: BLE001 - dedupe is best-effort
            record_failure("notifier._is_duplicate", exc)
            return False

    @staticmethod
    def _serialize_doc(doc: Any) -> str:
        """The stored JSON form of a publish's Doc (Formatting F7).

        Empty string when there is no Doc or serialization fails —
        the row then behaves exactly like a string row (redelivery
        through the string path).  Never raises."""
        if doc is None:
            return ""
        try:
            from ..social.chat.present import doc_to_json

            return doc_to_json(doc)
        except Exception:  # noqa: BLE001 - storage is best-effort
            _log.debug("notifier: doc serialization failed", exc_info=True)
            return ""

    def _store(self, kind: str, title: str, body: str, delivered: int,
               delivery_state: str = "", channel: str = "",
               retry_count: int = 0, next_retry_at: float = 0.0,
               nid: str | None = None,
               doc_json: str = "",
               ) -> dict[str, Any]:
        nid = nid or new_id()
        if self.db is not None:
            try:
                with self.db.transaction():
                    self.db.execute(
                        "INSERT INTO notifications (id, kind, title, body, delivered, "
                        "delivery_state, channel, retry_count, next_retry_at, "
                        "doc_json, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (nid, kind, title, body, delivered, delivery_state,
                         channel, retry_count, next_retry_at,
                         doc_json or None, time.time()),
                    )
            except Exception:  # noqa: BLE001 - pre-F7: no doc_json column
                try:
                    with self.db.transaction():
                        self.db.execute(
                            "INSERT INTO notifications (id, kind, title, body, delivered, "
                            "delivery_state, channel, retry_count, next_retry_at, "
                            "created_at) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (nid, kind, title, body, delivered, delivery_state,
                             channel, retry_count, next_retry_at, time.time()),
                        )
                except Exception:  # noqa: BLE001 - pre-migration-83: no retry cols
                    try:
                        with self.db.transaction():
                            self.db.execute(
                                "INSERT INTO notifications (id, kind, title, body, delivered, "
                                "delivery_state, created_at) "
                                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (nid, kind, title, body, delivered, delivery_state,
                                 time.time()),
                            )
                    except Exception:  # noqa: BLE001 - pre-migration: no delivery_state
                        try:
                            # pre-migration DBs lack delivery_state — store anyway
                            with self.db.transaction():
                                self.db.execute(
                                    "INSERT INTO notifications (id, kind, title, body, delivered, "
                                    "created_at) VALUES (?, ?, ?, ?, ?, ?)",
                                    (nid, kind, title, body, delivered, time.time()),
                                )
                        except Exception as exc:  # noqa: BLE001
                            record_failure("notifier._store.terminal", exc)
                            pass
        return {"id": nid, "kind": kind, "title": title,
                "delivered": bool(delivered),
                "delivery_state": delivery_state,
                "channel": channel}

    def _deliver(self, kind: str, title: str, body: str,
                 channels: list[str] | None = None,
                 output_kind: str | None = None,
                 output_meta: dict[str, Any] | None = None,
                 doc: Any = None) -> DeliveryOutcome:
        """Send to the owner's resolved channels.  Never raises.

        Returns a :class:`DeliveryOutcome` — one attempt record per
        resolved owner target (sent / failed / skipped, with the
        platform's ``message_id`` on success and the error/reason on
        failure), so ``/notify`` can show *why* a send didn't land.

        Target resolution (:func:`resolve_owner_targets`) is the fix for
        the audit's 0-delivery root cause: ``owner_chats`` entries may be
        bare chat ids (``"123456789"``, exactly what
        ``docs/TERMUX_ENV_TEMPLATE.txt`` documents) or name a platform
        whose same-family sibling is the one actually running
        (``telegram`` vs ``telegram-bot``) — both used to resolve to zero
        targets and silently store ``"failed"``.

        ``channels`` restricts delivery to the named platforms
        (``["telegram"]`` matches the ``telegram``/``telegram-bot``
        family); ``None`` delivers to all resolved owner channels.

        Buttons: a push has no triggering command, so it NEVER inherits
        buttons from the last inbound command. When ``output_kind`` is
        given (or derivable — a ``"schedule"`` push carries its job in
        the title), delivery attaches ADAPTIVE buttons about that
        output; otherwise it sends explicit NO buttons (``[]``), so the
        adapter cannot fall through to stale origin buttons.

        ``doc`` (F2): when a presentation Doc rides along, each target
        receives the Doc rendered for its own platform (see
        :meth:`_deliver_doc_send`) and the Doc's actions — when it has
        any — replace the adaptive buttons.  The string ``text`` built
        below is still what the row stores and what redelivery sends.
        """
        from ..social.chat.base import ChatRef
        from ..social.chat.gateway import resolve_owner_targets

        # Adaptive buttons for this push — computed once, attached to
        # every channel send. [] = explicitly no buttons.
        buttons: list[list[tuple[str, str]]] = []
        try:
            from ..social.chat.adaptive_buttons import adaptive_buttons_for

            _okind = output_kind
            _ometa = output_meta
            if _okind is None and kind == "schedule":
                # Redelivery path: the stored row carries no
                # output_meta, but the title names the job
                # ("✅ scheduled: <job>"). Derive it.
                _okind = "scheduled"
                _ometa = {"job": title.split("scheduled:", 1)[-1].strip()
                          .lstrip("✅❌ ").strip()}
            if _okind is not None:
                _kb = adaptive_buttons_for(output_kind=_okind,
                                           payload=_ometa,
                                           text=body or "")
                buttons = _kb if _kb else []
        except Exception as exc:  # noqa: BLE001 - buttons never break delivery
            record_failure("notifier._deliver.buttons", exc)
            buttons = []

        if doc is not None:
            # A Doc's own actions describe THIS output — they win over
            # the output_kind-derived adaptive buttons.
            try:
                from ..social.chat.present import actions_rows

                _doc_rows = actions_rows(doc)
                if _doc_rows:
                    buttons = _doc_rows
            except Exception as exc:  # noqa: BLE001 - buttons never break delivery
                record_failure("notifier._deliver.doc_buttons", exc)
                pass
        #: per-platform Doc renders, computed once per _deliver pass
        _doc_renders: dict[str, Any] = {}

        # Formatting F7 — the document route for Doc overflow (survey
        # §D.7 ladder rung 3, made deliberate): when a Doc's
        # overflow_text is itself document-sized (LongOutput's own
        # route decision), the overflow travels as a file attachment
        # via gateway.send_file instead of an ever-longer expandable
        # quote, and the in-chat Doc notes the attachment.  Targets
        # where the file send fails keep the full inline Doc — the
        # attachment is an enhancement, never a delivery requirement.
        # The temp file lives for this _deliver pass only.
        _overflow: dict[str, Any] | None = None
        if doc is not None and getattr(doc, "overflow_text", "") \
                and self.gateway is not None \
                and hasattr(self.gateway, "send_file"):
            try:
                from ..social.chat.progress_throttle import LongOutput

                _route, _ = LongOutput().route(doc.overflow_text)
                if _route == "document":
                    import tempfile

                    _fd, _path = tempfile.mkstemp(
                        prefix="devon-overflow-", suffix=".txt")
                    with os.fdopen(_fd, "w", encoding="utf-8") as _fh:
                        _fh.write(doc.overflow_text)
                    _overflow = {"path": _path,
                                 "chars": len(doc.overflow_text)}
            except Exception as exc:  # noqa: BLE001 - inline overflow remains
                record_failure("notifier._deliver.overflow", exc)
                _overflow = None

        outcome = DeliveryOutcome()
        if self.gateway is None:
            return outcome
        from .render import banner
        # NO length cut here: the gateway chunks long messages losslessly
        # (paragraph → sentence → word boundary, labelled (i/n)). Cutting
        # at 3900 used to slice briefings mid-sentence and silently drop
        # the rest — the "briefings always get cut off" bug.
        text = banner(title, "🔔") + (f"\n{body}" if body else "")
        partner = getattr(self.settings, "partner", None) if self.settings else None
        raw = getattr(partner, "owner_chats", "") or ""
        status: dict[str, Any] = {}
        try:
            status = self.gateway.status() or {}
        except Exception:  # noqa: BLE001 - a broken status() must not kill delivery
            _log.warning("notifier: gateway.status() failed; treating all "
                         "channels as down")
        registry_lookup = getattr(self.gateway, "platforms_for_chat_id", None)
        try:
            targets = resolve_owner_targets(
                raw, status=status, registry_lookup=registry_lookup)
        except Exception:  # noqa: BLE001 - resolution must never break delivery
            _log.warning("notifier: owner target resolution failed",
                         exc_info=True)
            targets = []
        if not raw.strip():
            _log.warning("notifier: _deliver found no owner_chats configured — "
                         "nothing can be delivered to a chat channel")
        elif not targets:
            _log.warning("notifier: owner_chats=%r resolved to zero delivery "
                         "targets", raw[:80])
        want = {c.strip().lower() for c in (channels or []) if c.strip()}

        def _wanted(target: Any) -> bool:
            if not want:
                return True
            fams = {target.platform.strip().lower(),
                    target.platform.strip().lower().split("-")[0]}
            src_plat = target.source.partition(":")[0].strip().lower()
            if src_plat:
                fams.add(src_plat)
                fams.add(src_plat.split("-")[0])
            return bool(fams & want)

        for target in targets:
            if not _wanted(target):
                continue
            attempt: dict[str, Any] = {
                "platform": target.platform, "chat_id": target.chat_id,
                "source": target.source, "note": target.note,
                "state": ATTEMPT_SKIPPED, "message_id": "",
                "error": "", "attempted_at": time.time(),
            }
            outcome.attempts.append(attempt)
            # Per-chat proactive toggle / quiet mode (spec §7): a chat
            # that never wants to be messaged first never gets a
            # proactive send — recorded as skipped, never silent.
            from ..partner.chat_modes import proactive_allowed
            _chat_key = f"{target.platform}:{target.chat_id}"
            if not proactive_allowed(_chat_key):
                attempt["error"] = (
                    "proactive messages are off for this chat "
                    "(quiet mode / /chatmode feature proactive off)")
                continue
            if not target.platform:
                # Unresolvable config entry — recorded, not silent.
                attempt["error"] = (
                    f"could not resolve {target.source!r} to a chat platform; "
                    f"check NM_PARTNER_OWNER_CHATS (want 'platform:id' or a "
                    f"bare chat id)")
                continue
            info = status.get(target.platform, {})
            if not (isinstance(info, dict) and info.get("running_in_session")):
                attempt["error"] = (
                    f"platform {target.platform!r} not running in this "
                    f"session")
                continue
            try:
                if doc is not None:
                    # Presentation-layer delivery (F2): the Doc rendered
                    # for THIS target's platform, its parse mode
                    # explicit, its actions as the buttons.
                    result = self._deliver_doc_send(
                        target, doc, buttons, _doc_renders,
                        overflow=_overflow)
                else:
                    # Explicit buttons (adaptive, or [] for none) — the
                    # adapter must NOT derive buttons from the last inbound
                    # command for a push.
                    try:
                        result = self.gateway.send(
                            target.platform,
                            ChatRef(platform=target.platform,
                                    chat_id=target.chat_id),
                            text, buttons=buttons)
                    except TypeError as exc:
                        # A gateway whose send() predates the buttons kwarg
                        # (minimal doubles, older adapters): buttons are an
                        # enhancement, never a delivery requirement — retry
                        # plain rather than failing the push. Any other
                        # TypeError re-raises into the failure path below.
                        if "buttons" not in str(exc):
                            raise
                        _log.debug("notifier: gateway.send rejects buttons — "
                                   "retrying plain")
                        result = self.gateway.send(
                            target.platform,
                            ChatRef(platform=target.platform,
                                    chat_id=target.chat_id),
                            text)
            except Exception as exc:  # noqa: BLE001 - one dead channel skips
                attempt["state"] = ATTEMPT_FAILED
                attempt["error"] = f"send raised: {exc}"[:300]
                _log.warning("notifier: send to %s:%s raised: %s",
                             target.platform, target.chat_id, exc)
                continue
            if getattr(result, "ok", False):
                # ok=True + message_id is the platform's delivery
                # confirmation (Telegram Bot API returns the accepted
                # message id; adapters report their own equivalents).
                attempt["state"] = ATTEMPT_SENT
                attempt["message_id"] = str(
                    getattr(result, "message_id", "") or "")
            else:
                attempt["state"] = ATTEMPT_FAILED
                attempt["error"] = str(
                    getattr(result, "error", "") or "send failed")[:300]
                _log.warning("notifier: send to %s:%s failed: %s",
                             target.platform, target.chat_id,
                             attempt["error"])
        if _overflow is not None:
            try:
                os.unlink(_overflow["path"])
            except OSError as exc:
                record_failure("notifier._deliver.unlink", exc)
                pass
        return outcome

    def _deliver_doc_send(self, target: Any, doc: Any,
                          buttons: list[list[tuple[str, str]]],
                          renders: dict[str, Any],
                          overflow: dict[str, Any] | None = None) -> Any:
        """One Doc → one gateway send for one target (Formatting F2).

        The Doc is rendered for the target's platform through the
        presentation layer (cached per platform in ``renders``) and
        sent with its parse mode explicit on the Telegram family —
        the gateway then applies it to EVERY chunk with its balanced
        markup chunking (the F1 formatted path), so a long Doc
        arrives formatted throughout, never just on the last chunk.

        Degradation mirrors the string path's buttons philosophy —
        an enhancement is never a delivery requirement: a gateway
        whose ``send()`` predates a kwarg loses that kwarg, not the
        alert. Buttons drop first; dropping the parse mode swaps the
        HTML for the plain rendering (raw HTML without a parse mode
        would render as literal tags). Any other error propagates to
        the caller's per-target failure path.

        ``overflow`` (F7): the prepared document-route payload for a
        Doc whose overflow_text is document-sized.  The file goes
        FIRST (so the in-chat note never claims an attachment that
        didn't land); when it lands, this target receives the Doc
        copy whose overflow moved into the attachment, its footer
        noting the fact.  When it doesn't, the target receives the
        full Doc with the overflow inline, exactly as before F7.
        """
        from ..social.chat.base import ChatRef
        from ..social.chat.present import render as _render_doc

        platform = str(target.platform or "")
        chat = ChatRef(platform=target.platform, chat_id=target.chat_id)
        variant = ""
        if overflow is not None:
            try:
                from ..social.chat.progress_throttle import LongOutput

                file_res = self.gateway.send_file(
                    platform, chat, overflow["path"],
                    caption=LongOutput().caption(doc.overflow_text))
                if getattr(file_res, "ok", False):
                    from dataclasses import replace as _dc_replace

                    note = ("📄 full detail attached as a document "
                            f"({overflow['chars']:,} chars)")
                    doc = _dc_replace(
                        doc, overflow_text="",
                        footer=(f"{doc.footer}\n{note}" if doc.footer
                                else note))
                    variant = "|attached"
                else:
                    _log.debug("notifier: overflow document send failed "
                               "for %s — overflow stays inline", platform)
            except Exception:  # noqa: BLE001 - inline overflow remains
                _log.debug("notifier: overflow document send raised for "
                           "%s — overflow stays inline", platform,
                           exc_info=True)
        cache_key = platform + variant
        rendered = renders.get(cache_key)
        if rendered is None:
            rendered = _render_doc(doc, platform)
            renders[cache_key] = rendered
        text, parse_mode = rendered.text, rendered.parse_mode
        use_buttons = True
        for _attempt in range(3):
            kw: dict[str, Any] = {}
            if use_buttons:
                kw["buttons"] = buttons
            if parse_mode:
                kw["parse_mode"] = parse_mode
            try:
                return self.gateway.send(platform, chat, text, **kw)
            except TypeError as exc:
                msg = str(exc)
                if use_buttons and "buttons" in msg:
                    use_buttons = False
                    _log.debug("notifier: gateway.send rejects buttons "
                               "— retrying without (doc path)")
                    continue
                if parse_mode and "parse_mode" in msg:
                    plain = renders.get("__plain__" + variant)
                    if plain is None:
                        plain = _render_doc(doc, "local")
                        renders["__plain__" + variant] = plain
                    text, parse_mode = plain.text, ""
                    _log.debug("notifier: gateway.send rejects "
                               "parse_mode — retrying with the plain "
                               "rendering (doc path)")
                    continue
                raise
        # Reached only if a gateway rejects a kwarg its error message
        # doesn't name — the bare send either lands or fails honestly
        # into the caller's failure path.
        return self.gateway.send(platform, chat, text)

    def _record_attempts(self, nid: str,
                         attempts: list[dict[str, Any]]) -> None:
        """Persist one ``notification_deliveries`` row per attempt.

        Never raises; pre-migration DBs (no table yet) just skip.
        """
        if self.db is None or not nid or not attempts:
            return
        try:
            with self.db.transaction():
                for a in attempts:
                    self.db.execute(
                        "INSERT INTO notification_deliveries "
                        "(notification_id, platform, chat_id, state, "
                        "message_id, error, note, attempted_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (nid, str(a.get("platform") or ""),
                         str(a.get("chat_id") or ""),
                         str(a.get("state") or ""),
                         str(a.get("message_id") or ""),
                         str(a.get("error") or "")[:500],
                         str(a.get("note") or "")[:200],
                         float(a.get("attempted_at") or time.time())),
                    )
        except Exception as exc:  # noqa: BLE001 - tracking must not break sends
            _log.debug("notifier: could not record delivery attempts: %s", exc)

    def delivery_attempts(self, notification_id: str) -> list[dict[str, Any]]:
        """Per-channel attempt records for one notification, oldest first.

        Powers the honest ``/notify`` view.  Never raises; pre-migration
        DBs (no ``notification_deliveries`` table) return [].
        """
        if self.db is None or not notification_id:
            return []
        try:
            return self.db.query(
                "SELECT * FROM notification_deliveries "
                "WHERE notification_id = ? ORDER BY attempted_at ASC",
                (notification_id,),
            )
        except Exception as exc:  # noqa: BLE001
            record_failure("notifier.delivery_attempts", exc)
            return []

    def termux_fallback_available(self) -> bool:
        """True when the Termux system-notification fallback can fire.

        The three gates: not disabled via ``NM_TERMUX_NOTIFY=0``, running
        on Termux, and the ``termux-notification`` binary on PATH
        (termux-api installed).  Never raises.
        """
        global _termux_notify_bin
        try:
            if str(os.environ.get(TERMUX_NOTIFY_ENV, "1")).strip().lower() in {
                    "0", "false", "no", "off"}:
                return False
            from ..core.profile import is_termux
            if not is_termux():
                return False
            if _termux_notify_bin is None:
                _termux_notify_bin = shutil.which("termux-notification") or False
            return bool(_termux_notify_bin)
        except Exception as exc:  # noqa: BLE001 - probing must never raise
            record_failure("notifier.termux_fallback_available", exc)
            return False

    def _termux_fallback(self, title: str, body: str) -> bool:
        """Last-resort delivery on Termux: an Android system notification.

        Fires only when no chat channel is live in the session (the phone
        reality: the Discord token got banned, the Telegram user session
        isn't connected here, …).  ``termux-notification`` lands in the
        Android notification shade of the device running the bot — the
        phone in the owner's hand.

        Never raises: returns True only when the notification was issued.
        """
        try:
            if not self.termux_fallback_available():
                return False
            bin_path = _termux_notify_bin
            if not isinstance(bin_path, str) or not bin_path:
                return False  # shouldn't happen — availability just passed
            text = (body or "").strip() or title
            proc = subprocess.run(
                [bin_path,
                 "--title", str(title)[:90],
                 "--content", text[:400],
                 "--id", f"devon-{int(time.time() * 1000)}"],
                timeout=15, capture_output=True)
            if proc.returncode == 0:
                _log.info("notifier: delivered via termux-notification: %r",
                          str(title)[:60])
                return True
            _log.warning("notifier: termux-notification failed (rc=%s): %s",
                         proc.returncode, (proc.stderr or b"")[:160])
            return False
        except Exception:  # noqa: BLE001 - fallback must never break delivery
            _log.debug("notifier: termux fallback errored", exc_info=True)
            return False

    # ── read side ────────────────────────────────────────────────────────────
    def recent(self, limit: int = 10, kind: str = "") -> list[dict[str, Any]]:
        if self.db is None:
            return []
        try:
            if kind:
                rows = self.db.query(
                    "SELECT * FROM notifications WHERE kind = ? ORDER BY created_at DESC LIMIT ?",
                    (kind, limit),
                )
            else:
                rows = self.db.query(
                    "SELECT * FROM notifications ORDER BY created_at DESC LIMIT ?", (limit,)
                )
            return rows
        except Exception as exc:  # noqa: BLE001
            record_failure("notifier.recent", exc)
            return []

    def pending(self) -> list[dict[str, Any]]:
        """Undelivered alerts that are still being worked (retryable or
        held).  Intentionally-unsent rows (``disabled``, ``deduped``,
        ``muted``, the alert-policy tiers ``digest`` / ``logged``) and
        dead-lettered rows (``dead`` — retries exhausted) are excluded
        — redeliver must not resurrect what the owner turned off or the
        policy batched/silenced, and the dead queue has its own reader
        (:meth:`dead_letters`)."""
        if self.db is None:
            return []
        try:
            return self.db.query(
                "SELECT * FROM notifications WHERE delivered = 0 "
                "AND COALESCE(delivery_state, '') NOT IN "
                "('disabled', 'deduped', 'muted', 'dead', 'digest', "
                "'logged') "
                "ORDER BY created_at ASC LIMIT 20"
            )
        except Exception:  # noqa: BLE001
            try:  # pre-migration: no delivery_state column
                return self.db.query(
                    "SELECT * FROM notifications WHERE delivered = 0 "
                    "ORDER BY created_at ASC LIMIT 20"
                )
            except Exception as exc:  # noqa: BLE001
                record_failure("notifier.pending.fallback", exc)
                return []

    def digest_pending(self, limit: int = 20) -> list[dict[str, Any]]:
        """The bounded DIGEST buffer: alert-policy DIGEST-tier rows,
        newest first.  These were worth recording but not worth a
        page; ``/notify`` renders them batched.  Never raises."""
        if self.db is None:
            return []
        try:
            return self.db.query(
                "SELECT * FROM notifications WHERE delivery_state = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (STATE_DIGEST, max(1, int(limit))),
            )
        except Exception as exc:  # noqa: BLE001 - pre-policy DBs have no rows
            record_failure("notifier.digest_pending", exc)
            return []

    def digest_unfolded(self, limit: int = 20) -> list[dict[str, Any]]:
        """DIGEST-tier rows not yet folded into a briefing, oldest
        first (Formatting F7): the briefing's alert-digest section
        drains this queue FIFO, and :meth:`mark_digest_folded` —
        called only after the briefing carrying them is confirmed
        delivered — takes them out of the queue, so each digest item
        surfaces in exactly one briefing.  Pre-F7 databases (no
        ``digest_folded_at`` column) return []: folding is an F7
        behaviour and never guesses at an unmigrated store.  Never
        raises."""
        if self.db is None:
            return []
        try:
            return self.db.query(
                "SELECT * FROM notifications WHERE delivery_state = ? "
                "AND COALESCE(digest_folded_at, 0) = 0 "
                "ORDER BY created_at ASC LIMIT ?",
                (STATE_DIGEST, max(1, int(limit))),
            )
        except Exception as exc:  # noqa: BLE001 - pre-F7 DBs have no fold column
            record_failure("notifier.digest_unfolded", exc)
            return []

    def mark_digest_folded(self, ids: Any) -> int:
        """Mark DIGEST rows folded into a delivered briefing (F7).

        Only rows still in the digest state are touched — a row the
        owner already saw another way is never re-marked.  Returns
        the number of rows marked.  Never raises."""
        if self.db is None:
            return 0
        clean = [str(i) for i in (ids or []) if i]
        if not clean:
            return 0
        try:
            with self.db.transaction():
                marked = 0
                for nid in clean:
                    cur = self.db.execute(
                        "UPDATE notifications SET digest_folded_at = ? "
                        "WHERE id = ? AND delivery_state = ? "
                        "AND COALESCE(digest_folded_at, 0) = 0",
                        (time.time(), nid, STATE_DIGEST))
                    marked += int(getattr(cur, "rowcount", 0) or 0)
            return marked
        except Exception as exc:  # noqa: BLE001 - pre-F7 DBs have no fold column
            record_failure("notifier.mark_digest_folded", exc)
            return 0

    def alert_journal(self, limit: int = 50, kind: str = "",
                      subject: str = "") -> list[dict[str, Any]]:
        """Recent alert-policy tier decisions (the audit read).

        Every classified alert — paged, digested, or logged — leaves
        one journal row with its tier and reason; this is how the
        owner (or an audit) answers "why didn't that page me?".
        Never raises."""
        if self.db is None:
            return []
        try:
            from .alert_policy import AlertPolicy

            return AlertPolicy.journal(self.db, limit=limit, kind=kind,
                                       subject=subject)
        except Exception as exc:  # noqa: BLE001
            record_failure("notifier.alert_journal", exc)
            return []

    def dead_letters(self, limit: int = 10) -> list[dict[str, Any]]:
        """Dead-lettered notifications: delivery retries exhausted.

        The content is preserved — the owner can still read (and manually
        resend) what never reached a channel.  Never raises.
        """
        if self.db is None:
            return []
        try:
            return self.db.query(
                "SELECT * FROM notifications WHERE delivered = 0 "
                "AND delivery_state = ? ORDER BY created_at DESC LIMIT ?",
                (STATE_DEAD, max(1, int(limit))),
            )
        except Exception as exc:  # noqa: BLE001 - pre-migration: no delivery_state
            record_failure("notifier.dead_letters", exc)
            return []

    def queue_depth(self) -> dict[str, int]:
        """Current undelivered queue depth: ``retryable`` (pending/failed,
        still being retried with backoff), ``held`` (quiet hours),
        ``dead`` (retries exhausted).  Never raises."""
        out = {"retryable": 0, "held": 0, "dead": 0}
        if self.db is None:
            return out
        try:
            rows = self.db.query(
                "SELECT COALESCE(delivery_state, '') AS state, COUNT(*) AS n "
                "FROM notifications WHERE delivered = 0 GROUP BY state")
            for row in rows:
                state = str(row.get("state") or "")
                n = int(row.get("n") or 0)
                if state in RETRYABLE_STATES:
                    out["retryable"] += n
                elif state == STATE_HELD_QUIET:
                    out["held"] += n
                elif state == STATE_DEAD:
                    out["dead"] += n
        except Exception as exc:  # noqa: BLE001
            record_failure("notifier.queue_depth", exc)
            pass
        return out

    def _due_redeliveries(self, limit: int) -> list[dict[str, Any]]:
        """Rows the retry queue should attempt right now: retryable rows
        whose backoff has elapsed, plus quiet-hour holds (re-checked, not
        re-attempted)."""
        if self.db is None:
            return []
        now = time.time()
        try:
            return self.db.query(
                "SELECT * FROM notifications WHERE delivered = 0 "
                "AND delivery_state IN (?, ?) AND next_retry_at <= ? "
                "AND retry_count < ? "
                "UNION ALL "
                "SELECT * FROM notifications WHERE delivered = 0 "
                "AND delivery_state = ? AND next_retry_at <= ? "
                "ORDER BY created_at ASC LIMIT ?",
                (STATE_PENDING, STATE_FAILED, now, DELIVERY_MAX_ATTEMPTS,
                 STATE_HELD_QUIET, now, max(1, int(limit))),
            )
        except Exception:  # noqa: BLE001 - pre-migration 83: no backoff cols
            try:
                return [r for r in self.pending()
                        if (r.get("delivery_state") or "") in
                        (*RETRYABLE_STATES, STATE_HELD_QUIET)][:max(1, int(limit))]
            except Exception as exc:  # noqa: BLE001
                record_failure("notifier._due_redeliveries.fallback", exc)
                return []

    def _redeliver_row(self, row: dict[str, Any], now: float,
                       *, consume_attempt: bool) -> bool:
        """Attempt one redelivery.  Returns True when it landed.

        Every attempt burns one retry and reschedules with exponential
        backoff.  A quiet-hour hold is not itself an attempt — the row
        only reaches this method once the hold has ended (see
        :meth:`redeliver`) — but a send attempted after the hold is a
        real attempt and counts toward the dead-letter cap.
        """
        attempts = int(row.get("retry_count") or 0)
        if consume_attempt:
            attempts += 1
        outcome = DeliveryOutcome()
        if self.gateway is not None:
            # Formatting F7: a row stored with its Doc (doc_json,
            # migration 88) redelivers WITH its structure — the Doc
            # is reconstructed and rendered per platform again.
            # Pre-F7 rows (NULL) and unparseable forms keep the
            # historical string path, byte for byte.
            doc = None
            if row.get("doc_json"):
                try:
                    from ..social.chat.present import doc_from_json

                    doc = doc_from_json(row.get("doc_json"))
                except Exception as exc:  # noqa: BLE001 - fall back to string
                    record_failure("notifier._redeliver_row.doc", exc)
                    doc = None
            outcome = self._deliver(row["kind"], row.get("title", ""),
                                    row.get("body", ""), doc=doc)
        # record this pass's per-channel attempts against the row, so
        # /notify shows every retry's channel detail, not just the last.
        self._record_attempts(str(row.get("id") or ""), outcome.attempts)
        reached = outcome.delivered
        channel = outcome.first_channel
        if reached == 0 and self._termux_fallback(
                str(row.get("title", "")), str(row.get("body", ""))):
            reached, channel = 1, "termux"
            self._record_attempts(str(row.get("id") or ""),
                                  [self._termux_attempt(
                                      str(row.get("title", "")))])
        try:
            with self.db.transaction():
                if reached:
                    self.db.execute(
                        "UPDATE notifications SET delivered = 1, "
                        "delivery_state = ?, channel = ? WHERE id = ?",
                        (STATE_SENT, channel, row["id"]),
                    )
                elif attempts >= DELIVERY_MAX_ATTEMPTS:
                    # dead-letter: the channel is down for good (or gone —
                    # e.g. a banned token).  Page once in the log, keep
                    # the content readable via dead_letters().
                    _log.warning(
                        "notifier: dead-lettering %r (%s) after %d failed "
                        "delivery attempts — content preserved, no more retries",
                        str(row.get("title", ""))[:80], row.get("kind"),
                        attempts)
                    self.db.execute(
                        "UPDATE notifications SET retry_count = ?, "
                        "delivery_state = ? WHERE id = ?",
                        (attempts, STATE_DEAD, row["id"]),
                    )
                else:
                    self.db.execute(
                        "UPDATE notifications SET retry_count = ?, "
                        "next_retry_at = ?, delivery_state = ? WHERE id = ?",
                        (attempts, now + delivery_backoff(attempts + 1),
                         row.get("delivery_state") or STATE_FAILED, row["id"]),
                    )
        except Exception:  # noqa: BLE001 - pre-migration 83: no retry cols
            try:
                with self.db.transaction():
                    if reached:
                        self.db.execute(
                            "UPDATE notifications SET delivered = 1, "
                            "delivery_state = ? WHERE id = ?",
                            (STATE_SENT, row["id"]),
                        )
            except Exception as exc:  # noqa: BLE001
                record_failure("notifier._redeliver_row.state", exc)
                pass
        return bool(reached)

    def redeliver(self, *, limit: int = 20) -> int:
        """Re-send due pending alerts (call after a platform hot-starts, or
        let the scheduler tick drive it periodically).

        Only rows whose backoff has elapsed are attempted — a dead
        channel backs off (60s → 2m → 4m → … → 6h) instead of being
        hammered every tick.  Rows held for quiet hours are re-checked
        first — they only go out once quiet hours end, and the hold
        doesn't consume a retry attempt.  Rows that exhaust
        :data:`DELIVERY_MAX_ATTEMPTS` are dead-lettered
        (``delivery_state = "dead"``) instead of retried forever.
        Never raises.
        """
        sent = 0
        now = time.time()
        for row in self._due_redeliveries(limit=limit):
            state = (row.get("delivery_state") or "")
            try:
                if state == STATE_HELD_QUIET:
                    # The hold is a *clock* decision, not a per-kind
                    # gate decision: rows land here from any producer
                    # that defers during quiet hours (missions progress
                    # stores kind="mission"; the proactive gate stores
                    # its own kinds).  Re-check the clock itself.
                    # proactive_gate() cannot answer for kinds it does
                    # not govern (it returns None for them) and its two
                    # governed kinds are quiet-exempt by design — so
                    # consulting it alone sent held rows out on the
                    # first sweep, in the middle of quiet hours.
                    if _in_quiet_hours_now(self.context):
                        continue  # still quiet — keep holding
                    if proactive_gate(self.context, row.get("kind") or "") \
                            == STATE_DISABLED:
                        # The owner turned this kind (or proactive
                        # entirely) off while the row was held: honour
                        # the switch — park the row in the terminal
                        # "disabled" state (redelivery never picks it
                        # up again) instead of sending it.
                        try:
                            with self.db.transaction():
                                self.db.execute(
                                    "UPDATE notifications SET "
                                    "delivery_state = ? WHERE id = ?",
                                    (STATE_DISABLED, row["id"]),
                                )
                        except Exception:  # noqa: BLE001 - bookkeeping only
                            _log.debug("notifier: could not park held row "
                                       "as disabled", exc_info=True)
                        continue
                    # Quiet hours ended: deliver now.  The hold itself
                    # consumed no attempt, but a failed send here is a
                    # real attempt — it counts toward the dead-letter
                    # cap, else a dead channel retries every sweep
                    # forever.
                    if self._redeliver_row(row, now, consume_attempt=True):
                        sent += 1
                    continue
                if self._redeliver_row(row, now, consume_attempt=True):
                    sent += 1
            except Exception:  # noqa: BLE001 - one bad row must not stop the queue
                _log.debug("notifier: redelivery row errored", exc_info=True)
        return sent

    def delivery_summary(self, limit: int = 10,
                         kinds: tuple[str, ...] = ("briefing", "watcher")
                         ) -> list[dict[str, Any]]:
        """Recent proactive sends with their delivery states, newest first.

        Powers ``nm briefing status`` and the chat ``/notify`` view.
        """
        rows = self.recent(limit * 3)
        out = []
        for row in rows:
            if kinds and (row.get("kind") or "") not in kinds:
                continue
            out.append({
                "id": row.get("id", ""),
                "kind": row.get("kind", ""),
                "title": row.get("title", ""),
                "created_at": row.get("created_at", 0),
                "delivered": bool(row.get("delivered")),
                "channel": row.get("channel") or "",
                "delivery_state": (row.get("delivery_state") or
                                   ("sent" if row.get("delivered") else "pending")),
            })
            if len(out) >= limit:
                break
        return out


    def delivery_counts(self, hours: float = 24.0,
                        kinds: tuple[str, ...] | None = None
                        ) -> dict[str, int]:
        """How many proactive sends landed in each delivery state over the
        last ``hours`` — the numbers behind ``nm briefing status``.

        Counts every stored state the choke point records (sent / failed /
        pending / held-quiet-hours / disabled / muted).  Note: dedupe
        suppressions are deliberately NOT stored (a suppressed repeat is
        not an alert), so "deduped" never appears here — the dedupe
        window doing its job is visible as the *absence* of repeats.
        ``kinds`` restricts to e.g. ``("briefing", "watcher")``; ``None``
        counts everything.  Never raises; pre-migration DBs (no
        delivery_state column) fall back to delivered/undelivered counts.
        """
        if self.db is None:
            return {}
        cutoff = time.time() - max(1.0, float(hours or 24.0)) * 3600.0
        try:
            if kinds:
                placeholders = ",".join("?" for _ in kinds)
                rows = self.db.query(
                    "SELECT COALESCE(delivery_state, '') AS state, "
                    "COUNT(*) AS n FROM notifications "
                    "WHERE created_at > ? AND kind IN (%s) "
                    "GROUP BY state" % placeholders,
                    (cutoff, *kinds),
                )
            else:
                rows = self.db.query(
                    "SELECT COALESCE(delivery_state, '') AS state, "
                    "COUNT(*) AS n FROM notifications "
                    "WHERE created_at > ? GROUP BY state",
                    (cutoff,),
                )
            out: dict[str, int] = {}
            for row in rows:
                state = str(row.get("state") or "")
                if not state:
                    continue
                out[state] = out.get(state, 0) + int(row.get("n") or 0)
            return out
        except Exception:  # noqa: BLE001 — pre-migration: no delivery_state
            try:
                if kinds:
                    placeholders = ",".join("?" for _ in kinds)
                    rows = self.db.query(
                        "SELECT delivered, COUNT(*) AS n FROM notifications "
                        "WHERE created_at > ? AND kind IN (%s) "
                        "GROUP BY delivered" % placeholders,
                        (cutoff, *kinds),
                    )
                else:
                    rows = self.db.query(
                        "SELECT delivered, COUNT(*) AS n FROM notifications "
                        "WHERE created_at > ? GROUP BY delivered",
                        (cutoff,),
                    )
                out = {}
                for row in rows:
                    out["sent" if row.get("delivered") else "pending"] = int(
                        row.get("n") or 0)
                return out
            except Exception as exc:  # noqa: BLE001
                record_failure("notifier.delivery_counts.fallback", exc)
                return {}


def notify(context: Any, kind: str, title: str, body: str = "", **kw: Any) -> dict[str, Any]:
    """One-call helper: ``notify(context, "news", "Headlines", digest)``.

    Pass ``critical=True`` to bypass the feature gate.  Note: critical
    no longer bypasses dedupe — the same (kind, title) inside the dedupe
    window collapses to one alert; ``force=True`` is the explicit
    escape hatch for that too.
    """
    return Notifier(context).publish(kind, title, body, **kw)


def notify_doc(context: Any, kind: str, doc: Any, **kw: Any) -> dict[str, Any]:
    """One-call helper for presentation Docs (Formatting F2):
    ``notify_doc(context, "finance", doc)`` — the stored title/body
    derive from the Doc and delivery renders it per platform.  See
    :meth:`Notifier.publish` for the ``doc`` contract."""
    return Notifier(context).publish(kind, "", "", doc=doc, **kw)

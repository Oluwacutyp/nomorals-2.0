"""Alert discipline — ONE paging policy for unsolicited owner alerts.

The owner's rule (2026-10-22, from live immune-gate spam): Devon's
systems keep working exactly as they do internally — enforcement,
throttling, self-healing all unchanged — but the owner is *paged*
only for alerts worth his attention.  Two immune-gate notice pairs in
one morning (marginal threshold crossings, the process never named,
self-resolved by the throttle, the closure admitting the underlying
issue was not resolved) carried zero owner-actionable content in four
pages.  This module is the fix, applied to every unsolicited alert
path that opts in, not just the immune gate.

Every classified alert passes through :meth:`AlertPolicy.decide` and
lands in one of three tiers:

* **PAGE** — delivered to the owner immediately.  Reserved for what
  needs him: an owner-actionable event (a decision or an act), a
  failure (to resolve, of containment, of the alerting machinery
  itself), or **recurrence** — the same subject firing again beyond
  the configured threshold inside the window.  Recurrence is what
  turns housekeeping into a real thing: one marginal enforcement is
  the system doing its job; the fifth this morning is a runaway the
  throttle is not fixing.
* **DIGEST** — noteworthy but not actionable: stored durably and
  surfaced batched on the notification status surface (``/notify``),
  never pushed.
* **LOG** — silent.  Self-resolving housekeeping: a first-instance
  event the owning system is already handling.  Still journaled
  (below), never delivered, never queued for redelivery.

Rules that hold for every wired emitter:

* an incident's **closure inherits its opening's tier** — a quietly
  opened incident closes quietly; a paged incident's closure pages
  too (the owner was told it started; he is told it ended).
* every PAGE names its **subject** (process / job / source) — the
  caller composes it into the alert, and the policy appends its own
  decision line (tier reason + subject) to paged bodies, so what
  escalated is stated plainly.
* every decision is **journaled** (kind, subject, tier, reason,
  timestamp) in the durable ``alert_events`` journal — the policy
  itself is auditable, and recurrence counting reads the same
  journal, so silent events still count toward escalation.

Callers classify by passing :class:`AlertHints` (or a plain dict with
the same fields) to ``Notifier.publish(..., alert_hints=...)``.
Alerts published *without* hints keep the historical behaviour
exactly (delivered as before): the seam tiers classified alerts, it
does not silently re-tier the whole product.  Thresholds are config:
``alert_recurrence_threshold`` / ``alert_recurrence_window_s`` on the
partner settings when present, else the defaults here.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "TIER_PAGE", "TIER_DIGEST", "TIER_LOG",
    "AlertHints", "AlertDecision", "AlertPolicy",
    "DEFAULT_RECURRENCE_THRESHOLD", "DEFAULT_RECURRENCE_WINDOW_S",
    "JOURNAL_KEY",
]

#: the three paging tiers
TIER_PAGE = "page"
TIER_DIGEST = "digest"
TIER_LOG = "log"

#: defaults — a subject must recur this many times inside the window
#: before housekeeping escalates to a page.
DEFAULT_RECURRENCE_THRESHOLD = 3
DEFAULT_RECURRENCE_WINDOW_S = 6 * 3600.0

#: the durable decision journal (kv_store, alongside the ops-alert
#: cooldowns — same substrate, no second store).
JOURNAL_KEY = "alert_policy:journal:v1"
#: bounded: the journal is an audit trail for recent paging decisions,
#: not an archive.  Older rows prune on write.
JOURNAL_MAX_ROWS = 500


@dataclass
class AlertHints:
    """The caller's classification of one unsolicited alert.

    ``subject`` names what the alert is about (a process population,
    a job, a source) — required for recurrence counting and for the
    page to be actionable.  Exactly one primary classification should
    be set; precedence in :meth:`AlertPolicy.decide` is failure →
    actionable → closure → recurrence → self-resolving → notice.
    """

    subject: str = ""
    #: the owning system is already handling this (throttle, self-
    #: heal, retry) and it is expected to resolve without the owner.
    self_resolving: bool = False
    #: this event closes an incident opened earlier under the same
    #: ``incident_key`` — its tier is inherited from the opening.
    resolved: bool = False
    #: failure to resolve, containment failure, or the alerting
    #: machinery itself failing.  Always pages.
    failed: bool = False
    #: a decision or an act is needed from the owner.  Always pages.
    actionable: bool = False
    #: noteworthy milestone / standing condition — batched, not paged.
    notice: bool = False
    #: pairs an opening event with its closure.
    incident_key: str = ""

    @classmethod
    def coerce(cls, value: Any) -> "AlertHints | None":
        """Accept an AlertHints, a plain dict, or None (unclassified)."""
        if value is None or isinstance(value, AlertHints):
            return value
        if isinstance(value, dict):
            known = {f for f in cls.__dataclass_fields__}
            return cls(**{k: v for k, v in value.items() if k in known})
        return None


@dataclass
class AlertDecision:
    tier: str
    reason: str
    subject: str
    #: occurrences of this (kind, subject) inside the window,
    #: including this one (openings/events only; closures report the
    #: opening's count context as 0).
    occurrences: int = 0
    #: True when the tier is PAGE *because* something escalated
    #: (failure / actionable / recurrence) rather than by default.
    escalated: bool = False
    #: True only for a recurrence page: the publish must bypass the
    #: notifier's dedupe, because dedupe would collapse exactly the
    #: repeat that is the signal.  Failure/actionable pages keep
    #: dedupe — a persisting failure re-pages on the dedupe cadence,
    #: not once per tick.
    force: bool = False

    def body_line(self) -> str:
        """The policy's own line, appended to paged alert bodies."""
        return (f"alert policy: {self.tier} — {self.reason} "
                f"(subject: {self.subject})")


def _partner(context: Any) -> Any | None:
    return getattr(getattr(context, "settings", None), "partner", None)


class AlertPolicy:
    """Tier decisions + the durable decision journal."""

    def __init__(self, *, recurrence_threshold: int =
                 DEFAULT_RECURRENCE_THRESHOLD,
                 recurrence_window_s: float =
                 DEFAULT_RECURRENCE_WINDOW_S) -> None:
        try:
            self.recurrence_threshold = max(
                2, int(recurrence_threshold))
        except (TypeError, ValueError):
            self.recurrence_threshold = DEFAULT_RECURRENCE_THRESHOLD
        try:
            self.recurrence_window_s = max(
                60.0, float(recurrence_window_s))
        except (TypeError, ValueError):
            self.recurrence_window_s = DEFAULT_RECURRENCE_WINDOW_S

    @classmethod
    def from_context(cls, context: Any) -> "AlertPolicy":
        """Config from the partner settings when present (duck-typed
        ``alert_recurrence_threshold`` / ``alert_recurrence_window_s``),
        else the module defaults."""
        partner = _partner(context)
        threshold = getattr(partner, "alert_recurrence_threshold", None)
        window = getattr(partner, "alert_recurrence_window_s", None)
        return cls(
            recurrence_threshold=(
                DEFAULT_RECURRENCE_THRESHOLD if threshold is None
                else threshold),
            recurrence_window_s=(
                DEFAULT_RECURRENCE_WINDOW_S if window is None else window))

    # ── the journal ──────────────────────────────────────────────────

    @staticmethod
    def _load_journal(db: Any) -> list[dict[str, Any]]:
        try:
            from ..storage.kv import KVStore

            raw = KVStore(db).get_raw(JOURNAL_KEY)
            rows = json.loads(raw) if raw else []
            return rows if isinstance(rows, list) else []
        except Exception:  # noqa: BLE001 - journal is best-effort
            return []

    @staticmethod
    def _save_journal(db: Any, rows: list[dict[str, Any]],
                      now: float, window_s: float) -> None:
        try:
            from ..storage.kv import KVStore

            keep_after = now - max(window_s * 4.0, 24 * 3600.0)
            rows = [r for r in rows
                    if float(r.get("ts") or 0) >= keep_after]
            rows = rows[-JOURNAL_MAX_ROWS:]
            KVStore(db).set_raw(JOURNAL_KEY, json.dumps(rows), "json")
        except Exception:  # noqa: BLE001 - journal is best-effort
            _log.debug("alert policy: journal write failed", exc_info=True)

    @staticmethod
    def journal(db: Any, *, limit: int = 50, kind: str = "",
                subject: str = "") -> list[dict[str, Any]]:
        """Recent tier decisions, newest first — the audit read."""
        if db is None:
            return []
        rows = AlertPolicy._load_journal(db)
        out = []
        for row in reversed(rows):
            if kind and row.get("kind") != kind:
                continue
            if subject and row.get("subject") != subject:
                continue
            out.append(row)
            if len(out) >= max(1, int(limit)):
                break
        return out

    # ── the decision ─────────────────────────────────────────────────

    def decide(self, db: Any, *, kind: str, subject: str = "",
               hints: AlertHints, now: float | None = None
               ) -> AlertDecision:
        """Classify one alert, journal the decision, return it.

        ``db`` may be None (no durable journal): decisions then fall
        back to the per-event rules with no recurrence memory — a
        first self-resolving event still logs, a failed/actionable
        event still pages.
        """
        now = time.time() if now is None else now
        subj = str(subject or hints.subject or kind)
        rows = self._load_journal(db) if db is not None else []
        role = ("close" if hints.resolved
                else "open" if hints.incident_key else "event")
        occurrences = 0
        force = False
        if hints.failed:
            tier, escalated = TIER_PAGE, True
            reason = ("failure — the event did not resolve, containment "
                      "failed, or the alerting machinery itself failed")
        elif hints.actionable:
            tier, escalated = TIER_PAGE, True
            reason = "owner action or decision is needed"
        elif hints.resolved:
            opening = None
            for row in reversed(rows):
                if row.get("kind") != kind or row.get("role") != "open":
                    continue
                if hints.incident_key and \
                        row.get("incident_key") != hints.incident_key:
                    continue
                if not hints.incident_key and row.get("subject") != subj:
                    continue
                opening = row
                break
            if opening is not None:
                tier = str(opening.get("tier") or TIER_DIGEST)
                escalated = tier == TIER_PAGE
                reason = (f"closure — inherits the {tier} tier of its "
                          f"opening ({opening.get('reason', '')})")
            else:
                tier, escalated = TIER_DIGEST, False
                reason = "closure with no recorded opening"
        else:
            cutoff = now - self.recurrence_window_s
            occurrences = 1 + sum(
                1 for row in rows
                if row.get("kind") == kind
                and row.get("subject") == subj
                and row.get("role") != "close"
                and float(row.get("ts") or 0) >= cutoff)
            if occurrences >= self.recurrence_threshold:
                already_paged = any(
                    row.get("kind") == kind
                    and row.get("subject") == subj
                    and row.get("tier") == TIER_PAGE
                    and str(row.get("reason") or "").startswith(
                        "recurrence")
                    and float(row.get("ts") or 0) >= cutoff
                    for row in rows)
                if already_paged:
                    # The escalation already reached the owner in this
                    # window; a flapping subject must not page once
                    # per occurrence forever.  Further occurrences
                    # journal silently until the subject goes quiet
                    # for a full window.
                    tier, escalated = TIER_LOG, False
                    reason = ("recurrence continues — already paged "
                              "in this window")
                else:
                    tier, escalated = TIER_PAGE, True
                    reason = (f"recurrence — occurrence {occurrences} "
                              f"of this subject within "
                              f"{int(self.recurrence_window_s // 3600)}h; "
                              "housekeeping has become a pattern")
                    force = True
            elif hints.self_resolving:
                tier, escalated = TIER_LOG, False
                reason = ("self-resolving housekeeping — the owning "
                          "system is already handling it")
            elif hints.notice:
                tier, escalated = TIER_DIGEST, False
                reason = "noteworthy, but no owner action is needed"
            else:
                tier, escalated = TIER_DIGEST, False
                reason = ("classified alert with no owner-action "
                          "signal — batched, not paged")
        if db is not None:
            rows.append({
                "ts": now, "kind": kind, "subject": subj,
                "incident_key": hints.incident_key, "role": role,
                "tier": tier, "reason": reason,
                "occurrences": occurrences,
            })
            self._save_journal(db, rows, now, self.recurrence_window_s)
        return AlertDecision(tier=tier, reason=reason, subject=subj,
                             occurrences=occurrences, escalated=escalated,
                             force=force)

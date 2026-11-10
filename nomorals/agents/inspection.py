"""Unified notification inspection (L5) — doc 15 P6.

The read-side home for "why did I get this?" — one render family
shared by the chat /notify verbs, the brain's tools and the
``nm notify`` CLI, so the three surfaces can never drift apart.
Everything here READS through the shipped accessors (the candidate
journal, the decision journal, candidate coverage, the
notification→candidate reverse link); P6 changes no decision or
release semantics — P2/P3 are closed.

The chain a why-render walks is the one P3 journaled: the producer
and the fact it asserted → the decision rule and verdict → the
routing → every release-time freshness verdict → the delivery ack.
Where no decision record exists for a notification (rows stored
directly, bypassing publish), the render says so plainly and shows
only what the row itself proves — it never invents a chain.
"""

from __future__ import annotations

import time
from typing import Any

from .alert_policy import AlertPolicy

__all__ = [
    "decision_entries",
    "candidate_row_for",
    "why_for_notification",
    "render_decisions",
    "render_why",
    "render_why_missing",
    "render_coverage",
]


_VERDICT_WORDS = {
    "page": "paged you",
    "digest": "held for the morning digest",
    "log": "logged only — no push",
    "suppress": "suppressed — never sent",
    "deduped": "deduped — you were already told",
}

_DELIVERY_WORDS = {
    "sent": "delivered",
    "failed": "delivery FAILED — it will retry",
    "pending": "queued for delivery",
    "held-quiet-hours": "held — quiet hours",
    "held-brain-degraded": "held — brain degraded (observe-only)",
    "disabled": "not sent — notifications disabled",
    "muted": "not sent — muted",
    "digest": "in the digest — read on /notify, folded into the "
              "morning briefing",
    "logged": "logged only",
    "suppressed": "never sent — suppressed by the decision",
    "deduped": "not sent again — deduped",
    "superseded": "never sent — superseded by a newer record before "
                  "release",
}


def _ago(ts: float, *, now: float | None = None) -> str:
    now = time.time() if now is None else now
    seconds = max(0.0, now - float(ts or 0))
    if seconds < 45:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m ago" if mins else f"{hours}h ago"
    return f"{int(seconds // 86400)}d ago"


def _age_words(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0))
    if seconds < 90:
        return f"{int(seconds)}s old"
    if seconds < 5400:
        return f"{int(seconds // 60)}m old"
    if seconds < 172800:
        return f"{seconds / 3600:.1f}h old"
    return f"{seconds / 86400:.1f}d old"


def decision_entries(db: Any, *, limit: int = 25,
                     verdict: str = "") -> list[dict[str, Any]]:
    """Recent decisions, newest first — suppressed and held rows
    included; they are the point of the view."""
    return AlertPolicy.decision_journal(db, limit=limit,
                                        verdict=verdict)


def candidate_row_for(db: Any, event_id: str) -> dict[str, Any] | None:
    """The full candidate-journal row behind a decision entry."""
    if not event_id:
        return None
    for row in AlertPolicy.candidate_journal(db, limit=500):
        if row.get("event_id") == event_id:
            return row
    return None


def why_for_notification(db: Any, notification_id: str
                         ) -> dict[str, Any] | None:
    """The candidate row a stored notification came from — or None
    when the row was stored without the decision machinery."""
    return AlertPolicy.candidate_for_notification(db, notification_id)


def render_decisions(entries: list[dict[str, Any]], *,
                     total_hint: int | None = None) -> str:
    """The decisions board: every recent publish decision with its
    verdict and reason — nothing the gate did is invisible."""
    lines = ["notification decisions — newest first "
             "(suppressed + held included; if it was decided, "
             "it is here)"]
    if not entries:
        lines.append("no decisions recorded yet")
        return "\n".join(lines)
    for idx, entry in enumerate(entries, 1):
        when = time.strftime("%H:%M", time.localtime(entry["ts"]))
        verdict = _VERDICT_WORDS.get(entry.get("verdict", ""),
                                     entry.get("verdict", ""))
        what = entry.get("subject") or entry.get("kind") or "alert"
        klass = entry.get("producer_class") or ""
        head = f"{idx}. {when} {verdict} — {what}"
        if klass:
            head += f" ({klass})"
        lines.append(head)
        reason = (entry.get("reason") or "").strip()
        if reason:
            if len(reason) > 130:
                reason = reason[:127] + "..."
            lines.append(f"   {reason}")
    lines.append("/notify why decision <n> walks one decision's "
                 "whole chain · /notify why <n> starts from the "
                 "/notify board")
    return "\n".join(lines)


def _asserted_lines(candidate: dict[str, Any]) -> list[str]:
    parts = []
    subject = candidate.get("subject") or ""
    outcome = candidate.get("outcome") or ""
    if subject and outcome:
        parts.append(f"{subject} — {outcome}")
    elif subject:
        parts.append(subject)
    elif outcome:
        parts.append(f"outcome: {outcome}")
    prev = candidate.get("prev_revision") or ""
    current = candidate.get("current_revision") or ""
    if prev or current:
        parts.append(f"revision {prev or '—'} → {current or '—'}")
    change = candidate.get("change_evidence") or ""
    if change:
        parts.append(f"change: {change}")
    refs = [str(r) for r in (candidate.get("result_refs") or [])][:3]
    if refs:
        parts.append("evidence: " + ", ".join(refs))
    impact = candidate.get("impact") or ""
    if impact:
        parts.append(f"impact: {impact}")
    return parts


def render_why(row: dict[str, Any], *,
               now: float | None = None) -> str:
    """The full chain for one candidate-journal row, in owner
    language: producer → fact asserted → decision → routing →
    release checks → delivery ack. No internal ids."""
    now = time.time() if now is None else now
    candidate = row.get("candidate") or {}
    decision = row.get("decision") or {}
    title = candidate.get("title") or row.get("kind") or "alert"
    lines = [f"why this alert — {title}"]
    producer = candidate.get("producer") or row.get("producer") or ""
    klass = (candidate.get("producer_class")
             or row.get("producer_class") or "")
    origin = " · ".join(p for p in (klass, producer) if p)
    if origin:
        lines.append(f"from: {origin}")
    asserted = _asserted_lines(candidate)
    if asserted:
        lines.append("it asserted: " + " · ".join(asserted))
    else:
        lines.append("it asserted: a plain notice — no structured "
                     "fact (subject, outcome or revision) was claimed")
    gaps = candidate.get("evidence_gaps") or row.get("evidence_gaps") \
        or []
    if candidate.get("complete"):
        lines.append("evidence: complete")
    elif gaps:
        lines.append("evidence gaps: " + ", ".join(str(g) for g in gaps))
    tier = decision.get("tier") or ""
    if tier:
        verdict = _VERDICT_WORDS.get(tier, tier)
        reason = (decision.get("reason") or "").strip()
        line = f"decision: {verdict}"
        if reason:
            line += f" — {reason}"
        lines.append(line)
        material = decision.get("material_gaps") or []
        if material:
            lines.append("held back by missing: "
                         + ", ".join(str(g) for g in material))
        if decision.get("dedupe_mode") == "legacy-fallback":
            lines.append("dedupe: legacy title-based — no semantic "
                         "identity was available for this one")
        routing = decision.get("routing") or {}
        if routing:
            channels = routing.get("channels")
            where = (", ".join(str(c) for c in channels)
                     if channels else "all owner channels")
            audience = routing.get("audience") or "owner"
            lines.append(f"routing: {audience} · {where}")
    for release in row.get("releases") or []:
        verdict = release.get("verdict") or ""
        age = _age_words(release.get("age_s") or 0)
        line = (f"release check ({release.get('path', 'release')}): "
                f"{verdict} — the fact was {age} at release")
        if release.get("superseded_by"):
            line += ("; a newer record for the same subject had "
                     "already landed")
        detail = (release.get("detail") or "").strip()
        if detail:
            line += f" · {detail}"
        ack = release.get("ack") or ""
        if ack:
            line += f" · delivery after release: {ack}"
        lines.append(line)
    delivery = row.get("delivery") or {}
    state = delivery.get("state") or ""
    if state:
        words = _DELIVERY_WORDS.get(state, state)
        channel = delivery.get("channel") or ""
        line = f"delivery: {words}"
        if channel:
            line += f" ({channel})"
        lines.append(line)
    elif tier == "suppress":
        lines.append("delivery: never sent — suppressed by the "
                     "decision before anything was stored")
    lines.append(f"decided {_ago(row.get('ts') or 0, now=now)}")
    return "\n".join(lines)


def render_why_missing(notification: dict[str, Any]) -> str:
    """The honest answer for a notification with NO decision record:
    it was stored directly (the progress/quiet-hours path) or before
    the decision journal existed. States what the row itself proves
    and nothing more."""
    title = notification.get("title") or notification.get("kind") \
        or "alert"
    lines = [f"why this alert — {title}"]
    lines.append(
        "no decision record exists for this one: it was stored "
        "directly, without passing the publish decision (the "
        "progress-notes path stores rows this way), so there is no "
        "producer chain, decision or recheck to show — only what "
        "the row itself proves:")
    state = notification.get("delivery_state") or ""
    if state:
        lines.append("delivery: " + _DELIVERY_WORDS.get(state, state))
    kind = notification.get("kind") or ""
    if kind:
        lines.append(f"kind: {kind}")
    return "\n".join(lines)


def render_coverage(coverage: dict[str, Any]) -> str:
    """Candidate coverage in owner terms: how much of what the
    system tried to say carried complete evidence, and what the
    gate decided about all of it."""
    lines = ["notification coverage — what the system tried to say, "
             "and what the gate did with it"]
    total = coverage.get("candidates", 0)
    lines.append(f"{total} candidates recorded · "
                 f"{coverage.get('complete', 0)} with complete "
                 f"evidence · {coverage.get('with_gaps', 0)} with "
                 "gaps")
    by_class = coverage.get("by_class") or {}
    if by_class:
        parts = [f"{name} {count}" for name, count in
                 sorted(by_class.items(), key=lambda kv: -kv[1])]
        lines.append("by class: " + " · ".join(parts))
    decisions = coverage.get("decisions") or {}
    if decisions:
        parts = []
        for tier in ("page", "digest", "log", "suppress", "deduped"):
            if decisions.get(tier):
                parts.append(f"{_VERDICT_WORDS.get(tier, tier)} "
                             f"×{decisions[tier]}")
        for tier, count in sorted(decisions.items()):
            if tier not in ("page", "digest", "log", "suppress",
                            "deduped"):
                parts.append(f"{tier} ×{count}")
        lines.append("decisions: " + " · ".join(parts))
    gaps = coverage.get("gap_fields") or {}
    if gaps:
        parts = [f"{name} ({count})" for name, count in
                 sorted(gaps.items(), key=lambda kv: -kv[1])[:6]]
        lines.append("most common evidence gaps: " + ", ".join(parts))
    if coverage.get("legacy_fallback"):
        lines.append(f"legacy fallback (no semantic identity): "
                     f"{coverage['legacy_fallback']}")
    return "\n".join(lines)

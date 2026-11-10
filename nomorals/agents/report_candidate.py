"""ReportCandidate — the structured notification candidate (Doc 15).

Doc 15 §3: *every* producer creates a structured ReportCandidate, and
plain-string legacy callers enter through an adapter that resolves
source/subject from what the call carries and **marks missing
evidence** — they must not bypass the reportability decision merely
by omitting AlertHints.

Pass 1 landed the schema, the legacy adapter, and the journal in
OBSERVE mode: :meth:`Notifier.publish` builds a candidate for every
call and records it in the candidate journal (see
:mod:`.alert_policy`), while delivery behaviour is byte-identical to
before — no gate, no re-tiering, no dedupe change.  Pass 2 (this
schema's consumer) makes the decision mandatory on top of it: the
gate in :mod:`.alert_policy` reads the candidate's semantic shape —
:func:`semantic_identity` and :func:`is_unchanged_tick` below — and
every publish leaves a decision record in the candidate journal.
Nothing in this module decides any alert's fate; it defines the
identity the decision deduplicates by.

Field map against the doc's required candidate fields:

* event id, producer, source operation/job/mission/task/attempt and
  stable subject identity → ``event_id``, ``producer``,
  ``producer_class`` (the :mod:`.notify_producers` census class),
  ``source_kind``, ``source_id``, ``attempt``, ``subject``.
* owner, audience, destination and requested channel policy →
  ``owner``, ``audience``, ``channel_policy`` (the notifier's only
  destination is the owner's channels; ``channels=`` restricts it).
* previous/current semantic revision or canonical change evidence;
  event time and observation freshness → ``prev_revision``,
  ``current_revision``, ``change_evidence``, ``event_time``,
  ``observed_at``.  For legacy calls the observation *is* the
  publish, so ``observed_at == event_time`` — the anchor exists but
  is weak.  Pass 3's reckoning with it, stated plainly: the legacy
  anchor can prove *when the system last spoke* about a subject,
  never when the world last changed — so the release recheck does
  not trust the anchor as freshness evidence at all.  It re-reads
  the candidate journal's own later records for the same subject
  (a different revision or outcome recorded since *is* fresher
  truth), and where a held item's subject has no later record —
  or the item carries no subject at all — the release is journaled
  with its age and an ``unverifiable`` verdict rather than a
  pretended freshness pass (see :mod:`.alert_policy`
  :meth:`AlertPolicy.superseding_record` and the notifier's
  release path).
* actual outcome, useful result/artifact refs, impact and
  uncertainty → ``outcome`` (the :mod:`..core.result_contract`
  vocabulary), ``result_refs``, ``impact``, ``uncertainty``.
* owner action needed, deadline/urgency, supported next actions →
  ``action_needed``, ``deadline``, ``urgency``, ``next_actions``.
* solicited response / requested subscription-milestone / proactive
  result / diagnostic / routine telemetry → ``report_class``.  Every
  ``publish()`` is by construction an *unsolicited* push (solicited
  answers travel the chat path, never the notifier), so the adapter
  records ``"proactive"`` by construction rather than guessing a
  finer class the legacy call never declared.
* evidence refs and sensitivity → ``evidence_refs``,
  ``sensitivity`` (``"owner"`` by construction: the notifier sends
  to owner channels only; nothing here carries credentials or raw
  logs).

``evidence_gaps`` is the missing-evidence mark: the names of the
fields above the adapter could not resolve from the call.  A gapped
candidate is a visible, journaled state — never silently upgraded,
and (from Pass 2) never able to bypass the decision by omission.

Legacy resolution sources, in order: the ``alert_subject`` argument,
:class:`AlertHints` (subject/incident/outcome classification), the
``critical`` flag (urgency), ``channels`` (channel policy), and the
producer's ``output_meta`` side-channel, whose documented evidence
keys the adapter reads: ``job``, ``subject``, ``source_id``,
``attempt``, ``prev_revision``, ``current_revision``,
``change_evidence``, ``outcome``, ``impact``, ``uncertainty``,
``deadline``, ``urgency``, ``action_needed``, ``next_actions``,
``result_refs``, ``evidence_refs``.  A producer that starts passing
those keys graduates its candidates toward complete with no call
-shape change.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any

from ..core.ids import new_id
from .notify_producers import producer_class_for_module

__all__ = [
    "ReportCandidate", "build_legacy_candidate", "caller_module",
    "semantic_identity", "is_unchanged_tick",
    "parse_deadline", "deadline_epoch",
    "REPORT_CLASS_PROACTIVE", "URGENCY_CRITICAL",
]

#: report classes (doc §3) — the adapter can only honestly assert
#: the proactive one; the finer classes arrive with native producers.
REPORT_CLASS_PROACTIVE = "proactive"

URGENCY_CRITICAL = "critical"

#: outcome values that imply a useful result should be referenced
_RESULT_OUTCOMES = ("succeeded", "verified", "partial")


@dataclass(frozen=True)
class ReportCandidate:
    """One producer event, structured for the reportability decision."""

    event_id: str
    kind: str
    title: str
    body_present: bool
    producer: str          # dotted module that called publish
    producer_class: str    # census class ('' when unregistered)
    source_kind: str       # operation | job | mission | task
    source_id: str
    attempt: str
    subject: str
    event_time: float
    observed_at: float
    owner: str = "owner"
    audience: str = "owner"
    channel_policy: str = "all-owner-channels"
    prev_revision: str = ""
    current_revision: str = ""
    change_evidence: str = ""
    outcome: str = ""      # core.result_contract vocabulary
    result_refs: tuple[str, ...] = ()
    impact: str = ""
    uncertainty: str = ""
    action_needed: bool | None = None
    deadline: str = ""
    urgency: str = ""
    next_actions: tuple[str, ...] = ()
    report_class: str = REPORT_CLASS_PROACTIVE
    evidence_refs: tuple[str, ...] = ()
    sensitivity: str = "owner"
    evidence_gaps: tuple[str, ...] = ()
    legacy: bool = True

    @property
    def complete(self) -> bool:
        """True when no required evidence is missing."""
        return not self.evidence_gaps

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "kind": self.kind,
            "title": self.title,
            "body_present": self.body_present,
            "producer": self.producer,
            "producer_class": self.producer_class,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "attempt": self.attempt,
            "subject": self.subject,
            "event_time": self.event_time,
            "observed_at": self.observed_at,
            "owner": self.owner,
            "audience": self.audience,
            "channel_policy": self.channel_policy,
            "prev_revision": self.prev_revision,
            "current_revision": self.current_revision,
            "change_evidence": self.change_evidence,
            "outcome": self.outcome,
            "result_refs": list(self.result_refs),
            "impact": self.impact,
            "uncertainty": self.uncertainty,
            "action_needed": self.action_needed,
            "deadline": self.deadline,
            "urgency": self.urgency,
            "next_actions": list(self.next_actions),
            "report_class": self.report_class,
            "evidence_refs": list(self.evidence_refs),
            "sensitivity": self.sensitivity,
            "evidence_gaps": list(self.evidence_gaps),
            "legacy": self.legacy,
        }


def semantic_identity(candidate: ReportCandidate, *,
                        role: str = "event") -> str:
    """The dedupe identity of one candidate (Doc 15 §5, Pass 2).

    ``subject (or source) + attempt + semantic transition
    (prev_revision → current_revision) + outcome + audience +
    logical role`` — the identity a re-publication must match to be
    the *same* event.  Title, body, render timestamps, formatting
    and generated phrasing are excluded by construction: the same
    event reworded by a model keeps its identity, and two different
    operations that happen to share a title never do.

    One refinement: a candidate asserting *only* change evidence
    (no revision, no outcome) has nothing canonical to distinguish
    one asserted change from the next, so its identity carries the
    evidence itself (whitespace/case-normalized).  A producer that
    needs reword-proof dedupe for successive changes graduates to
    revisions — the revision, when present, is the identity and the
    evidence text is prose again.

    Returns ``""`` when the candidate carries no semantic identity
    at all — no subject/source, or nothing that distinguishes one
    state of the subject from another (no revision, no outcome, no
    change evidence).  The decision then falls back to the legacy
    ``(kind, title)`` dedupe and journals the fallback as such.
    """
    subject = candidate.subject or candidate.source_id
    if not subject:
        return ""
    if not (candidate.current_revision or candidate.outcome
            or candidate.change_evidence):
        return ""
    transition = ""
    if candidate.prev_revision or candidate.current_revision:
        transition = (f"{candidate.prev_revision}->"
                      f"{candidate.current_revision}")
    evidence = ""
    if not transition and not candidate.outcome \
            and candidate.change_evidence:
        evidence = "ev:" + " ".join(
            candidate.change_evidence.split()).casefold()
    parts = [subject, candidate.attempt, transition, candidate.outcome,
             evidence, candidate.audience, role]
    return "|".join(p for p in parts if p)


def is_unchanged_tick(candidate: ReportCandidate) -> bool:
    """True when the candidate asserts a standing state, not an event.

    A tick is a poll/watch observation that found no change: it
    names a subject at a revision and declares no transition into
    it, no outcome, and no new evidence — either explicitly (the
    producer reported ``prev_revision == current_revision``) or by
    shape (a current revision with no predecessor and nothing else
    asserted).  An owner-action need is a fact, not a tick, and an
    outcome is an event, not a tick; either one lifts the candidate
    out of this rule.  Urgency never does: a critical classification
    cannot turn an unchanged event into news (doc §4).
    """
    if candidate.action_needed is True:
        return False
    if candidate.outcome:
        return False
    if candidate.prev_revision and candidate.current_revision:
        return candidate.prev_revision == candidate.current_revision
    if candidate.current_revision and not candidate.prev_revision:
        return not candidate.change_evidence
    return False


def parse_deadline(value: Any) -> float | None:
    """Best-effort epoch seconds for a deadline value, or None.

    Accepts an epoch number (or a numeric string) and an ISO-8601
    string carrying an explicit zone.  A *naive* ISO string — no
    zone — returns None: whose clock it names is unknowable, and
    guessing one would manufacture verdicts (Pass 3's
    RULE-RECHECK-MOOT-DEADLINE simply does not apply to it).
    Unparseable values return None too.  Callers that need to tell
    "no deadline" from "unparseable deadline" compare against the
    raw value, which stays on the candidate untouched.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        from datetime import datetime

        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.timestamp()


def deadline_epoch(candidate: ReportCandidate) -> float | None:
    """The candidate's deadline as epoch seconds (see
    :func:`parse_deadline` for what honestly parses), or None."""
    return parse_deadline(candidate.deadline)


def caller_module() -> str:
    """The dotted module that called ``Notifier.publish``.

    Walks the stack past the notifier's own plumbing (``publish``
    itself, ``publish_doc``, the ``notify`` helpers, and this module)
    so helper-routed producers resolve to *their* module.  Returns
    ``""`` when the stack cannot be read — the candidate then marks
    the producer unresolved instead of guessing.
    """
    skip = {"nomorals.agents.notifier", "nomorals.agents.report_candidate"}
    try:
        frame = sys._getframe(1)
        while frame is not None:
            name = str(frame.f_globals.get("__name__") or "")
            if name and name not in skip:
                return name
            frame = frame.f_back
    except (ValueError, AttributeError):
        return ""
    return ""


def _str_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value if str(v))
    if value:
        return (str(value),)
    return ()


def build_legacy_candidate(
    *,
    kind: str,
    title: str = "",
    body: str = "",
    force: bool = False,
    critical: bool = False,
    channels: list[str] | None = None,
    alert_subject: str = "",
    alert_hints: Any = None,
    output_kind: str | None = None,
    output_meta: dict[str, Any] | None = None,
    producer: str = "",
    now: float | None = None,
) -> ReportCandidate:
    """Adapt one legacy ``publish()`` call into a ReportCandidate.

    Resolves every field the call genuinely carries and lists the
    rest in ``evidence_gaps`` — deterministic, and never raises on
    odd producer input (a malformed hint/meta degrades to gaps, not
    to a lost alert: the candidate is journal-side only in Pass 1).
    """
    from .alert_policy import AlertHints

    now = time.time() if now is None else now
    meta: dict[str, Any] = output_meta if isinstance(output_meta, dict) else {}
    hints = AlertHints.coerce(alert_hints)

    def meta_str(key: str) -> str:
        value = meta.get(key)
        return str(value) if value not in (None, "") else ""

    gaps: list[str] = []

    subject = (str(alert_subject or "")
               or (hints.subject if hints is not None else "")
               or meta_str("subject"))
    if not subject:
        gaps.append("subject")

    source_id = meta_str("source_id")
    if not source_id and subject:
        source_id = subject
    if not source_id and meta_str("job"):
        source_id = f"job:{meta_str('job')}"
    if not source_id and output_kind:
        source_id = str(output_kind)
    if not source_id:
        gaps.append("source_id")

    producer_class = producer_class_for_module(producer)
    source_kind = {"mission": "mission", "job": "job"}.get(
        producer_class, "operation")

    attempt = meta_str("attempt")
    if not attempt:
        gaps.append("attempt")

    prev_revision = meta_str("prev_revision")
    current_revision = meta_str("current_revision")
    if not prev_revision and not current_revision:
        gaps.append("revision")

    change_evidence = meta_str("change_evidence")
    if not change_evidence and hints is not None and hints.incident_key:
        change_evidence = (
            f"incident {hints.incident_key} "
            f"{'closed' if hints.resolved else 'opened'}")
    if not change_evidence:
        gaps.append("change_evidence")

    outcome = meta_str("outcome")
    if not outcome and hints is not None and hints.failed:
        outcome = "failed"
    if not outcome and hints is not None and hints.resolved:
        outcome = "succeeded"
    if not outcome:
        gaps.append("outcome")

    impact = meta_str("impact")
    if not impact:
        gaps.append("impact")
    uncertainty = meta_str("uncertainty")
    if not uncertainty:
        gaps.append("uncertainty")

    action_needed: bool | None = None
    if isinstance(meta.get("action_needed"), bool):
        action_needed = bool(meta["action_needed"])
    elif hints is not None and (hints.actionable or hints.failed):
        action_needed = True
    elif hints is not None:
        action_needed = False
    if action_needed is None:
        gaps.append("action_needed")

    deadline = meta_str("deadline")
    if action_needed is True and not deadline:
        gaps.append("deadline")

    urgency = URGENCY_CRITICAL if critical else meta_str("urgency")
    if not urgency:
        gaps.append("urgency")

    next_actions = _str_list(meta.get("next_actions"))
    if action_needed is True and not next_actions:
        gaps.append("next_actions")

    result_refs = _str_list(meta.get("result_refs"))
    if outcome in _RESULT_OUTCOMES and not result_refs:
        gaps.append("result_refs")

    evidence_refs = list(_str_list(meta.get("evidence_refs")))
    if hints is not None and hints.incident_key:
        ref = f"incident:{hints.incident_key}"
        if ref not in evidence_refs:
            evidence_refs.append(ref)
    if not evidence_refs:
        gaps.append("evidence_refs")

    if channels:
        channel_policy = "channels:" + ",".join(
            str(c) for c in channels if str(c))
    else:
        channel_policy = "all-owner-channels"

    return ReportCandidate(
        event_id=new_id(),
        kind=str(kind or ""),
        title=str(title or ""),
        body_present=bool(str(title or "").strip()
                          or str(body or "").strip()),
        producer=str(producer or ""),
        producer_class=producer_class,
        source_kind=source_kind,
        source_id=source_id,
        attempt=attempt,
        subject=subject,
        event_time=now,
        observed_at=now,
        channel_policy=channel_policy,
        prev_revision=prev_revision,
        current_revision=current_revision,
        change_evidence=change_evidence,
        outcome=outcome,
        result_refs=result_refs,
        impact=impact,
        uncertainty=uncertainty,
        action_needed=action_needed,
        deadline=deadline,
        urgency=urgency,
        next_actions=next_actions,
        evidence_refs=tuple(evidence_refs),
        evidence_gaps=tuple(gaps),
    )

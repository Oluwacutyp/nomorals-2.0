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
Thresholds are config: ``alert_recurrence_threshold`` /
``alert_recurrence_window_s`` on the partner settings when present,
else the defaults here.

Doc 15 Pass 2 — the decision is now MANDATORY.  Every publish is
adapted to a :class:`~.report_candidate.ReportCandidate` and decided
exactly once: hinted publishes through :meth:`AlertPolicy.decide`
(the classification above, byte-for-byte the historical semantics),
unhinted publishes through :meth:`AlertPolicy.decide_candidate`,
which applies the reportability gate to the candidate's own
evidence:

* **RULE-TICK** — a candidate asserting a standing state with no
  transition, outcome or new evidence is an unchanged tick and is
  SUPPRESSED (journaled, never stored, never pushed): a timer
  firing is not news.
* **RULE-DEDUP-SEMANTIC** — a candidate whose semantic identity
  (:func:`~.report_candidate.semantic_identity`) matches an
  already-paged identity is a replay of the same event and is
  deduped on that identity, whatever its title says this time.
* **RULE-HOLD-MATERIAL-GAPS** — a candidate that asserts a
  reportable fact (a revision transition, an outcome, change
  evidence, an owner-action need) but lacks evidence *material* to
  that fact — the subject itself; ``result_refs`` for a result
  outcome; ``next_actions`` for an action need; ``attempt`` for
  mission/job facts; ``impact`` for trading/finance facts — is HELD
  to the DIGEST tier with the gaps named, never paged as if
  complete and never silently dropped.
* **RULE-LEGACY-PASSTHROUGH** — a bare legacy publish asserting no
  structured fact at all keeps its historical delivery.  The
  decision still runs and is journaled (with its evidence gaps on
  record for the migration census), and legacy ``(kind, title)``
  dedupe still applies — the pass-7 pin: genuinely actionable
  one-shot paths (ops findings, trial waits, research results) page
  as before until their producers graduate to structured evidence.

Every decision — hinted or not — is recorded on the candidate's row
in the candidate journal (:data:`CANDIDATE_JOURNAL_KEY`), which is
therefore the 100%-coverage decision journal of record; the legacy
journal above stays the hinted-decision ledger the recurrence
counting reads.
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
    "TIER_SUPPRESS", "TIER_DEDUPED",
    "AlertHints", "AlertDecision", "AlertPolicy",
    "DEFAULT_RECURRENCE_THRESHOLD", "DEFAULT_RECURRENCE_WINDOW_S",
    "JOURNAL_KEY", "CANDIDATE_JOURNAL_KEY", "CANDIDATE_JOURNAL_MAX_ROWS",
    "MATERIAL_GAP_ORDER",
]

#: the three paging tiers
TIER_PAGE = "page"
TIER_DIGEST = "digest"
TIER_LOG = "log"

#: Pass 2 verdicts beside the tiers: an unchanged tick is SUPPRESSED
#: (doc §3's decision vocabulary — journaled, no notification row,
#: no push), and a semantic replay is DEDUPED on its identity.
TIER_SUPPRESS = "suppress"
TIER_DEDUPED = "deduped"

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

#: the candidate journal (Doc 15, Pass 1): one row per publish —
#: hinted or not — carrying the ReportCandidate the legacy adapter
#: built and its missing-evidence marks.  Same KV substrate and the
#: same prune discipline as the decision journal, under its own key
#: so the policy's recurrence counting (which reads the decision
#: journal) is untouched.  This is what makes coverage measurable:
#: before Pass 1, an unhinted publish left no record of *why* at all.
CANDIDATE_JOURNAL_KEY = "alert_policy:candidates:v1"
CANDIDATE_JOURNAL_MAX_ROWS = 500


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
    #: Pass 2: the ReportCandidate's event id, when the decision was
    #: taken on a candidate (the mandatory path).
    candidate_id: str = ""
    #: Pass 2: the candidate's semantic dedupe identity ("" when it
    #: carries none — the legacy (kind, title) dedupe then governs,
    #: recorded in ``dedupe_mode``).
    identity: str = ""
    #: Pass 2: "semantic" when ``identity`` governed dedupe,
    #: "legacy-fallback" when the (kind, title) fallback did.
    dedupe_mode: str = ""
    #: Pass 2: True when this page is a semantic replay of an
    #: already-paged identity — the publish returns deduped.
    deduped: bool = False
    #: Pass 2: the material evidence gaps that held this candidate
    #: to the digest tier (RULE-HOLD-MATERIAL-GAPS), in canonical
    #: order; empty for every other verdict.
    material_gaps: tuple[str, ...] = ()
    #: Pass 2: True when the decision line rides the delivered body:
    #: every hinted page (the historical behaviour) and every
    #: evidence-asserting candidate page.  A bare legacy pass-through
    #: page keeps its historical body byte-for-byte.
    annotate: bool = False

    @property
    def suppressed(self) -> bool:
        """True when the verdict is RULE-TICK suppression."""
        return self.tier == TIER_SUPPRESS

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
    def _load_journal(db: Any, key: str = JOURNAL_KEY) -> list[dict[str, Any]]:
        try:
            from ..storage.kv import KVStore

            raw = KVStore(db).get_raw(key)
            rows = json.loads(raw) if raw else []
            return rows if isinstance(rows, list) else []
        except Exception:  # noqa: BLE001 - journal is best-effort
            return []

    @staticmethod
    def _save_journal(db: Any, rows: list[dict[str, Any]],
                      now: float, window_s: float,
                      key: str = JOURNAL_KEY,
                      max_rows: int = JOURNAL_MAX_ROWS) -> None:
        try:
            from ..storage.kv import KVStore

            keep_after = now - max(window_s * 4.0, 24 * 3600.0)
            rows = [r for r in rows
                    if float(r.get("ts") or 0) >= keep_after]
            rows = rows[-max_rows:]
            KVStore(db).set_raw(key, json.dumps(rows), "json")
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

    # ── the candidate journal (Doc 15 P1; decisions from P2) ────

    @staticmethod
    def record_candidate(db: Any, candidate: dict[str, Any],
                         now: float | None = None,
                         decision: "AlertDecision | None" = None) -> None:
        """Journal one ReportCandidate and, from Pass 2, the
        mandatory decision taken on it — the candidate journal is
        the 100%-coverage decision journal of record: every publish
        (hinted or not) leaves exactly one row carrying the
        candidate identity, the verdict and the reason in the
        decision's own terms.  Best-effort like the decision
        journal: a journal failure never reaches the publish path."""
        if db is None:
            return
        now = time.time() if now is None else now
        rows = AlertPolicy._load_journal(db, CANDIDATE_JOURNAL_KEY)
        row: dict[str, Any] = {
            "ts": now,
            "kind": str(candidate.get("kind") or ""),
            "subject": str(candidate.get("subject") or ""),
            "event_id": str(candidate.get("event_id") or ""),
            "producer": str(candidate.get("producer") or ""),
            "producer_class": str(candidate.get("producer_class") or ""),
            "complete": not candidate.get("evidence_gaps"),
            "evidence_gaps": list(candidate.get("evidence_gaps") or []),
            "candidate": candidate,
        }
        if decision is not None:
            row["decision"] = {
                "tier": (TIER_DEDUPED if decision.deduped
                         else decision.tier),
                "reason": decision.reason,
                "subject": decision.subject,
                "identity": decision.identity,
                "dedupe_mode": decision.dedupe_mode,
                "material_gaps": list(decision.material_gaps),
                "escalated": decision.escalated,
            }
        rows.append(row)
        AlertPolicy._save_journal(
            db, rows, now, DEFAULT_RECURRENCE_WINDOW_S,
            key=CANDIDATE_JOURNAL_KEY,
            max_rows=CANDIDATE_JOURNAL_MAX_ROWS)

    @staticmethod
    def candidate_journal(db: Any, *, limit: int = 50, kind: str = "",
                          producer_class: str = ""
                          ) -> list[dict[str, Any]]:
        """Recent candidate records, newest first — the audit read
        beside :meth:`journal` (which stays decision-only)."""
        if db is None:
            return []
        rows = AlertPolicy._load_journal(db, CANDIDATE_JOURNAL_KEY)
        out = []
        for row in reversed(rows):
            if kind and row.get("kind") != kind:
                continue
            if producer_class and \
                    row.get("producer_class") != producer_class:
                continue
            out.append(row)
            if len(out) >= max(1, int(limit)):
                break
        return out

    @staticmethod
    def candidate_coverage(db: Any) -> dict[str, Any]:
        """Coverage over the retained candidate journal: how many
        publishes left a candidate, how many were complete, and
        which evidence fields the legacy population is missing —
        the measurement Pass 2's migration works from."""
        rows = (AlertPolicy._load_journal(db, CANDIDATE_JOURNAL_KEY)
                if db is not None else [])
        by_class: dict[str, int] = {}
        gap_fields: dict[str, int] = {}
        complete = 0
        for row in rows:
            cls = str(row.get("producer_class") or "(unregistered)")
            by_class[cls] = by_class.get(cls, 0) + 1
            if row.get("complete"):
                complete += 1
            for gap in row.get("evidence_gaps") or []:
                gap_fields[str(gap)] = gap_fields.get(str(gap), 0) + 1
        decisions: dict[str, int] = {}
        legacy_fallback = 0
        for row in rows:
            dec = row.get("decision")
            if not isinstance(dec, dict):
                continue
            verdict = str(dec.get("tier") or "")
            if verdict:
                decisions[verdict] = decisions.get(verdict, 0) + 1
            if dec.get("dedupe_mode") == "legacy-fallback":
                legacy_fallback += 1
        return {
            "candidates": len(rows),
            "complete": complete,
            "with_gaps": len(rows) - complete,
            "by_class": by_class,
            "gap_fields": gap_fields,
            #: Pass 2: the mandatory decision's verdict counts over
            #: the same retained rows, and how many publishes relied
            #: on the legacy (kind, title) dedupe fallback.
            "decisions": decisions,
            "suppressed": decisions.get(TIER_SUPPRESS, 0),
            "legacy_fallback": legacy_fallback,
        }

    @staticmethod
    def decision_journal(db: Any, *, limit: int = 50, kind: str = "",
                         verdict: str = "") -> list[dict[str, Any]]:
        """Recent mandatory decisions, newest first — one entry per
        publish, hinted or not, each carrying the candidate identity
        (``event_id``), the verdict and the reason.  This is the
        100%-coverage read beside :meth:`journal` (which stays the
        hinted-decision ledger the recurrence counting reads)."""
        if db is None:
            return []
        rows = AlertPolicy._load_journal(db, CANDIDATE_JOURNAL_KEY)
        out = []
        for row in reversed(rows):
            dec = row.get("decision")
            if not isinstance(dec, dict):
                continue
            if kind and row.get("kind") != kind:
                continue
            if verdict and dec.get("tier") != verdict:
                continue
            out.append({
                "ts": row.get("ts"),
                "event_id": row.get("event_id"),
                "kind": row.get("kind"),
                "subject": dec.get("subject") or row.get("subject"),
                "producer": row.get("producer"),
                "producer_class": row.get("producer_class"),
                "verdict": dec.get("tier"),
                "reason": dec.get("reason"),
                "identity": dec.get("identity"),
                "dedupe_mode": dec.get("dedupe_mode"),
                "material_gaps": dec.get("material_gaps") or [],
            })
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
                             force=force,
                             annotate=(tier == TIER_PAGE))

    # ── the mandatory candidate decision (Doc 15, Pass 2) ────────

    def _already_paged(self, db: Any, identity: str) -> bool:
        """True when a retained decision record shows this semantic
        identity already paged.  Reads the candidate journal (the
        decision journal of record); its bounded retention (500
        rows) is the dedupe memory's honest bound — the same class
        of bound as the legacy 600s window, far larger."""
        if db is None or not identity:
            return False
        rows = self._load_journal(db, CANDIDATE_JOURNAL_KEY)
        for row in rows:
            dec = row.get("decision")
            if isinstance(dec, dict) \
                    and dec.get("identity") == identity \
                    and dec.get("tier") == TIER_PAGE:
                return True
        return False

    def decide_candidate(self, db: Any, candidate: Any, *,
                         hints: "AlertHints | None" = None,
                         now: float | None = None) -> AlertDecision:
        """The ONE mandatory decision for a publish (Pass 2).

        ``candidate`` is a :class:`~.report_candidate.ReportCandidate`
        (duck-typed).  Hinted publishes delegate to :meth:`decide` —
        the historical classification, untouched — and gain only the
        semantic-replay check on top of a PAGE verdict.  Unhinted
        publishes are decided on the candidate's own evidence by the
        rules named in this module's docstring.  This method does
        NOT journal: the caller records the decision on the
        candidate's journal row (one row per publish), except that
        the hinted delegation journals in the legacy decision
        journal exactly as it always has (recurrence substrate).
        """
        from .report_candidate import is_unchanged_tick, semantic_identity

        now = time.time() if now is None else now
        role = ("close" if hints is not None and hints.resolved
                else "open" if hints is not None and hints.incident_key
                else "event")
        identity = semantic_identity(candidate, role=role)
        dedupe_mode = "semantic" if identity else "legacy-fallback"
        subj = (candidate.subject or candidate.source_id
                or candidate.kind)

        if hints is not None:
            decision = self.decide(
                db, kind=candidate.kind, subject=candidate.subject,
                hints=hints, now=now)
            decision.candidate_id = candidate.event_id
            decision.identity = identity
            decision.dedupe_mode = dedupe_mode
            if decision.tier == TIER_PAGE and identity \
                    and self._already_paged(db, identity):
                decision.deduped = True
                decision.reason = (
                    "semantic duplicate — this transition was "
                    f"already reported (identity: {identity}); "
                    "the original classification stands: "
                    + decision.reason)
            return decision

        # ── unhinted: the reportability gate on the evidence ─────
        if is_unchanged_tick(candidate):
            return AlertDecision(
                tier=TIER_SUPPRESS,
                reason=("unchanged tick — subject "
                        f"{subj} reports revision "
                        f"{candidate.current_revision or '—'} with no "
                        "transition, no outcome and no new evidence; "
                        "a timer firing is not news"),
                subject=subj, candidate_id=candidate.event_id,
                identity=identity, dedupe_mode=dedupe_mode)
        if identity and self._already_paged(db, identity):
            return AlertDecision(
                tier=TIER_PAGE, deduped=True,
                reason=("semantic duplicate — this transition was "
                        f"already reported (identity: {identity})"),
                subject=subj, candidate_id=candidate.event_id,
                identity=identity, dedupe_mode=dedupe_mode)
        fact = _asserted_fact(candidate)
        if fact:
            material = _material_gaps(candidate)
            if material:
                return AlertDecision(
                    tier=TIER_DIGEST,
                    reason=(f"held — the candidate asserts {fact} but "
                            "material evidence is missing: "
                            + ", ".join(material)
                            + "; held to the digest, never paged as "
                              "if complete"),
                    subject=subj, candidate_id=candidate.event_id,
                    identity=identity, dedupe_mode=dedupe_mode,
                    material_gaps=tuple(material))
            return AlertDecision(
                tier=TIER_PAGE, escalated=True,
                reason=f"reportable fact — {fact}",
                subject=subj, candidate_id=candidate.event_id,
                identity=identity, dedupe_mode=dedupe_mode,
                annotate=True)
        return AlertDecision(
            tier=TIER_PAGE,
            reason=("legacy notice — no structured evidence asserted; "
                    "the decision ran and the gaps are on record "
                    "(RULE-LEGACY-PASSTHROUGH)"),
            subject=subj, candidate_id=candidate.event_id,
            identity=identity, dedupe_mode=dedupe_mode)


#: canonical order for naming material gaps (stable reasons/journals)
MATERIAL_GAP_ORDER = ("subject", "attempt", "result_refs",
                      "next_actions", "impact")

#: outcomes that report a useful result — the result must be
#: referenced for the page to be complete (doc §3).
_RESULT_OUTCOMES = ("succeeded", "verified", "partial")

#: producer classes whose facts are per-attempt by nature — an
#: attempt-less mission/job fact cannot be told apart from another
#: attempt's, so ``attempt`` is material evidence for them.
_ATTEMPT_CLASSES = ("mission", "job")

#: producer classes whose pages are only actionable with their
#: impact stated — a market/money alert without impact is noise.
_IMPACT_CLASSES = ("trading", "finance")


def _asserted_fact(candidate: Any) -> str:
    """The reportable fact a candidate asserts, in words, or "".

    A fact is exactly one of the doc §3 gate's currencies: a
    semantic transition, an outcome, canonical change evidence, or
    an owner-action need.  Text alone is not a fact — it carries no
    identity the decision can verify, dedupe or re-check.
    """
    if candidate.prev_revision and candidate.current_revision \
            and candidate.prev_revision != candidate.current_revision:
        return (f"a revision transition "
                f"({candidate.prev_revision} → "
                f"{candidate.current_revision})")
    if candidate.outcome:
        return f"an outcome ({candidate.outcome})"
    if candidate.change_evidence:
        return "change evidence"
    if candidate.action_needed is True:
        return "an owner-action need"
    return ""


def _material_gaps(candidate: Any) -> list[str]:
    """The evidence gaps material to the fact this candidate
    asserts (RULE-HOLD-MATERIAL-GAPS), in canonical order.

    Only gaps the adapter actually marked are considered — a field
    the producer supplied is never re-litigated here.
    """
    gaps = set(candidate.evidence_gaps or ())
    material: list[str] = []
    if "subject" in gaps:
        material.append("subject")
    if candidate.producer_class in _ATTEMPT_CLASSES \
            and "attempt" in gaps:
        material.append("attempt")
    if candidate.outcome in _RESULT_OUTCOMES \
            and "result_refs" in gaps:
        material.append("result_refs")
    if candidate.action_needed is True and "next_actions" in gaps:
        material.append("next_actions")
    if candidate.producer_class in _IMPACT_CLASSES \
            and "impact" in gaps:
        material.append("impact")
    return [g for g in MATERIAL_GAP_ORDER if g in material]

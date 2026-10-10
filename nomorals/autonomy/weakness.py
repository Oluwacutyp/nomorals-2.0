"""Weakness detection → research → proposal → sandbox build/test.

The self-healing loop. The system watches itself for:

* **Tool failures** — tools that error repeatedly (tracked from the tool
  loop's audit trail).
* **Knowledge gaps** — questions the brain couldn't answer (the research
  organ's ``knowledge_gaps`` table).
* **Error patterns** — exceptions clustering in logs (fuzzy signature
  clustering, not exact-string matching).
* **Capability gaps** — the owner asks for something no tool covers.
* **Retry loops** — a tool retried N times in one turn without progress
  (Microsoft's 2025 agent failure taxonomy).
* **Cascading failures** — several tools failing inside one window,
  suggesting a shared root cause.
* **Silent degradation** — quality metrics drifting down without hard
  errors.

For each weakness:
1. **Detect** — score it (burn rate: fast/acute vs slow/chronic).
   Above threshold → open.
2. **Research** — the weakness is translated into a ``directive.gap``
   organ event for the research organ (the one directive kind the
   research organ's tick actually drains). The research organ
   investigates on its tick.
3. **Propose** — when research lands, draft a fix proposal (what's wrong,
   what the research found, what to change). Proposals go to the owner
   for approval — the system never rewrites itself silently.
4. **Sandbox test** — approved proposals emit a ``weakness.approved`` bus
   event and a scheduler organ event so a sandbox build mission can be
   queued without anyone tripping a command. Building and merging still
   need the owner's explicit go-ahead. That's the safety line.
5. **Recover** — success streaks auto-resolve ``tool_failure`` cases, so
   fixed tools stop haunting the list.
6. **Remember** — verified fixes are stored in the fixed-weakness memory.
   A recurring signature first checks the memory and one-step resolves
   instead of re-diagnosing from scratch (the Regenesis pattern: index
   failures by structure, never store a fix until a verifier confirms).

Detection, research, and proposals are automatic; recovery is automatic;
building and merging need the owner. Nothing here modifies production
code autonomously.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.events import Event, global_bus
from ..core.logging_setup import get_logger
from .. import organs as _organs

_log = get_logger(__name__)

#: Failure count in the window that opens a weakness case.
FAILURE_THRESHOLD = 3
#: Seconds of history considered for failure clustering.
FAILURE_WINDOW = 86400  # 24h
#: Consecutive successes that auto-resolve a tool_failure case.
RECOVERY_STREAK = 5
#: Retry attempts in one turn that open a retry_loop case.
RETRY_LOOP_THRESHOLD = 4
#: Distinct tools failing in the cascade window that open a case.
CASCADE_TOOL_THRESHOLD = 3
#: Seconds for cascade correlation.
CASCADE_WINDOW = 600  # 10 min
#: Days without new evidence before a case auto-expires.
STALE_CASE_DAYS = 30.0

#: All known weakness kinds.
WEAKNESS_KINDS = (
    "tool_failure",
    "knowledge_gap",
    "error_pattern",
    "capability_gap",
    "retry_loop",
    "cascading_failure",
    "silent_degradation",
)


@dataclass
class Weakness:
    """A detected weakness case."""
    id: str = ""
    kind: str = ""  # see WEAKNESS_KINDS
    subject: str = ""  # tool name, question, error signature, requested capability
    evidence: dict[str, Any] = field(default_factory=dict)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    occurrences: int = 1
    status: str = "open"  # open | researching | proposed | approved | resolved | dismissed
    proposal: str = ""
    severity: str = "watch"  # watch | warning | critical (burn-rate)


def ensure_schema(db: Any) -> None:
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS weaknesses (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            subject TEXT NOT NULL,
            evidence TEXT NOT NULL DEFAULT '{}',
            first_seen REAL NOT NULL DEFAULT 0,
            last_seen REAL NOT NULL DEFAULT 0,
            occurrences INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL DEFAULT 'open',
            proposal TEXT NOT NULL DEFAULT ''
        )
        """
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_weaknesses_status "
        "ON weaknesses(status, last_seen)"
    )
    # Tool health streaks for auto-recovery.
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS tool_health (
            tool TEXT PRIMARY KEY,
            ok_streak INTEGER NOT NULL DEFAULT 0,
            fail_streak INTEGER NOT NULL DEFAULT 0,
            last_ts REAL NOT NULL DEFAULT 0
        )
        """
    )
    # Per-sighting timestamps — powers burn-rate severity + acceleration.
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS weakness_sightings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            weakness_id TEXT NOT NULL,
            ts REAL NOT NULL DEFAULT 0
        )
        """
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_sightings_wid_ts "
        "ON weakness_sightings(weakness_id, ts)"
    )
    # Fixed-weakness memory: verified fixes, indexed by signature.
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS weakness_fixes (
            signature TEXT PRIMARY KEY,
            kind TEXT NOT NULL DEFAULT '',
            subject TEXT NOT NULL DEFAULT '',
            fix TEXT NOT NULL DEFAULT '',
            verified_ts REAL NOT NULL DEFAULT 0,
            resolved_wid TEXT NOT NULL DEFAULT ''
        )
        """
    )
    # Silent-degradation metric samples.
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS weakness_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT NOT NULL,
            metric TEXT NOT NULL,
            value REAL NOT NULL,
            ts REAL NOT NULL DEFAULT 0
        )
        """
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_wmetrics_subj_ts "
        "ON weakness_metrics(subject, metric, ts)"
    )


def _new_id() -> str:
    from ..core.ids import new_id
    return new_id("weak")


def _ledger(db: Any, kind: str, ref_id: str, summary: str,
            *, ok: bool = True,
            metadata: dict[str, Any] | None = None) -> None:
    try:
        from ..agents.autonomy_ledger import record_ledger
        record_ledger(db, "weakness", kind, ref_id, summary, ok=ok,
                      metadata=metadata)
    except Exception:  # noqa: BLE001 - telemetry is fail-open
        _log.debug("weakness ledger write failed", exc_info=True)


# ── fuzzy error signatures ───────────────────────────────────────────

_HEX_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
_NUM_RE = re.compile(r"\b\d[\d_\.,]*\b")
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
_PATH_RE = re.compile(r"(?:/[\w.\-]+)+|(?:[A-Za-z]:\\(?:[\w.\-]+\\)*)")
_QUOTED_RE = re.compile(r"""(['"])(?:(?!\1).){1,80}\1""")


def error_signature(error: str, location: str = "") -> str:
    """Normalize an error into a clusterable signature.

    Strips volatile parts (ids, numbers, paths, quoted values) so the
    same root cause through different entry points clusters together —
    the "index by structure, not vocabulary" pattern. Returns a short
    stable signature like ``valueerror@tools/search.py:88#<hash>``.
    """
    text = str(error or "")
    text = _UUID_RE.sub("<uuid>", text)
    text = _HEX_RE.sub("<hex>", text)
    text = _PATH_RE.sub("<path>", text)
    text = _QUOTED_RE.sub("<str>", text)
    text = _NUM_RE.sub("<n>", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    # Exception type is the strongest signal — keep it verbatim.
    m = re.match(r"^([a-z_][\w\.]*?(?:error|exception|failed|failure|timeout))",
                 text)
    exc = m.group(1) if m else ""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    loc = str(location or "").strip().replace("/", ".")[:60]
    parts = [p for p in (exc, loc) if p]
    head = "@".join(parts) if parts else "error"
    return f"{head}#{digest}"


def _cluster_subject(kind: str, subject: str) -> str:
    """Cluster key for a sighting — fuzzy for errors, exact otherwise."""
    if kind == "error_pattern":
        return error_signature(subject)
    return str(subject or "")


# ── fixed-weakness memory ────────────────────────────────────────────

def record_fix(db: Any, weakness_id: str, fix: str) -> bool:
    """Store a verified fix in the fixed-weakness memory.

    Only call after the fix is confirmed working (independent verifier,
    owner confirmation, or auto-recovery streak) — never store an
    unverified fix.
    """
    ensure_schema(db)
    row = db.execute(
        "SELECT kind, subject FROM weaknesses WHERE id = ?",
        (weakness_id,)).fetchone()
    if not row:
        return False
    kind, subject = row[0], row[1]
    sig = error_signature(subject) if kind == "error_pattern" else subject
    key = f"{kind}:{sig}"
    db.execute(
        "INSERT INTO weakness_fixes "
        "(signature, kind, subject, fix, verified_ts, resolved_wid) "
        "VALUES (?, ?, ?, ?, ?, ?) "
        "ON CONFLICT (signature) DO UPDATE SET fix = excluded.fix, "
        "verified_ts = excluded.verified_ts, "
        "resolved_wid = excluded.resolved_wid",
        (key, kind, subject, str(fix or "")[:2000],
         time.time(), weakness_id),
    )
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        pass
    _log.info("fix memorized for %s", key)
    _ledger(db, "fix_memorized", weakness_id,
            f"verified fix stored for {kind} {subject}",
            metadata={"signature": key})
    return True


def recall_fix(db: Any, kind: str, subject: str) -> dict[str, Any] | None:
    """Look up a verified fix for a recurring weakness.

    Returns the memorized fix dict, or None. A hit means one-step
    resolve instead of a full re-diagnosis.
    """
    ensure_schema(db)
    sig = error_signature(subject) if kind == "error_pattern" else subject
    key = f"{kind}:{sig}"
    row = db.execute(
        "SELECT fix, verified_ts, subject FROM weakness_fixes "
        "WHERE signature = ?", (key,)).fetchone()
    if not row:
        return None
    return {"fix": row[0], "verified_ts": row[1], "subject": row[2],
            "signature": key}


# ── detection ────────────────────────────────────────────────────────

def report_weakness(db: Any, kind: str, subject: str,
                    evidence: dict[str, Any] | None = None,
                    fuzzy: bool = True) -> str:
    """Report a weakness sighting. Clusters by (kind, subject).

    ``error_pattern`` subjects are fuzzy-clustered via
    :func:`error_signature` (set ``fuzzy=False`` for synthetic subjects
    like ``ledger:scheduler`` that are already canonical). When
    occurrences cross the threshold, emits ``weakness.detected`` and
    kicks off research — unless the fixed-weakness memory already knows
    the answer, in which case the case one-step resolves with the
    memorized fix.

    Returns the weakness id.
    """
    ensure_schema(db)
    now = time.time()
    ev = json.dumps(evidence or {})
    clustered = _cluster_subject(kind, subject) if fuzzy else str(subject)
    row = db.execute(
        "SELECT id, occurrences, status FROM weaknesses "
        "WHERE kind = ? AND subject = ? AND status NOT IN "
        "('resolved', 'dismissed')",
        (kind, clustered),
    ).fetchone()
    if row:
        wid, occ, status = row[0], int(row[1]), row[2]
        db.execute(
            "UPDATE weaknesses SET occurrences = occurrences + 1, "
            "last_seen = ?, evidence = ? WHERE id = ?",
            (now, ev, wid),
        )
        new_occ = occ + 1
    else:
        wid = _new_id()
        db.execute(
            "INSERT INTO weaknesses (id, kind, subject, evidence, "
            "first_seen, last_seen, occurrences) "
            "VALUES (?, ?, ?, ?, ?, ?, 1)",
            (wid, kind, clustered, ev, now, now),
        )
        new_occ = 1
        status = "open"
    db.execute(
        "INSERT INTO weakness_sightings (weakness_id, ts) VALUES (?, ?)",
        (wid, now),
    )
    # Bounded growth on sightings.
    db.execute(
        "DELETE FROM weakness_sightings WHERE ts < ?",
        (now - 30 * 86400,))
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        pass

    if new_occ >= FAILURE_THRESHOLD and status == "open":
        # Fixed-weakness memory first: known problem → one-step resolve.
        fix = recall_fix(db, kind, clustered)
        if fix and fix.get("fix"):
            resolve_weakness(
                db, wid,
                f"one-step resolve from fixed-weakness memory: "
                f"{fix['fix'][:200]}")
            _log.info("weakness %s resolved from memory (no re-diagnosis)",
                      wid)
            return wid
        db.execute(
            "UPDATE weaknesses SET status = 'researching' WHERE id = ?",
            (wid,),
        )
        try:
            db.commit()
        except Exception:  # noqa: BLE001
            pass
        sev = severity(db, wid)
        _log.info("weakness detected: %s %s (%d occurrences, %s)",
                  kind, clustered, new_occ, sev["level"])
        _ledger(db, "detected", wid,
                f"weakness detected: {kind} {clustered} "
                f"({new_occ} occurrences, severity={sev['level']})",
                metadata={"kind": kind, "subject": clustered,
                          "occurrences": new_occ,
                          "severity": sev["level"]})
        investigate_weakness(db, wid)
    return wid


def report_retry_loop(db: Any, tool: str, attempts: int,
                      last_error: str = "") -> str | None:
    """Report a retry loop: one tool retried without progress in a turn.

    Returns the weakness id when a case opens, else None.
    """
    if attempts < RETRY_LOOP_THRESHOLD:
        return None
    return report_weakness(
        db, "retry_loop", str(tool or "?"),
        {"attempts": attempts, "last_error": last_error[:300]})


def report_cascade(db: Any, tools: list[str],
                   window: float = CASCADE_WINDOW) -> str | None:
    """Report a cascading failure: several tools failing in one window.

    ``tools`` — distinct tool names that errored recently. Opens a
    single ``cascading_failure`` case instead of N independent ones.
    """
    uniq = sorted({str(t or "?") for t in tools})
    if len(uniq) < CASCADE_TOOL_THRESHOLD:
        return None
    return report_weakness(
        db, "cascading_failure", "+".join(uniq),
        {"tools": uniq, "window_s": window})


def record_degradation_sample(db: Any, subject: str, metric: str,
                              value: float,
                              ts: float | None = None) -> dict[str, Any]:
    """Feed one quality-metric sample into silent-degradation detection.

    Compares the recent median against the older baseline; a sustained
    drop opens a ``silent_degradation`` case. ``metric`` should be
    "higher is better" (e.g. success rate, answer score).
    """
    ensure_schema(db)
    now = ts if ts is not None else time.time()
    db.execute(
        "INSERT INTO weakness_metrics (subject, metric, value, ts) "
        "VALUES (?, ?, ?, ?)",
        (str(subject or "?"), str(metric or "?"), float(value), now),
    )
    db.execute(
        "DELETE FROM weakness_metrics WHERE ts < ?", (now - 14 * 86400,))
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        pass
    out: dict[str, Any] = {"subject": subject, "metric": metric}
    try:
        rows = db.execute(
            "SELECT value, ts FROM weakness_metrics "
            "WHERE subject = ? AND metric = ? ORDER BY ts",
            (str(subject or "?"), str(metric or "?"))).fetchall()
        if len(rows) >= 10:
            vals = [float(r[0]) for r in rows]
            mid = len(vals) // 2
            old = sorted(vals[:mid])[len(vals[:mid]) // 2]
            new = sorted(vals[mid:])[len(vals[mid:]) // 2]
            out["baseline_median"] = round(old, 3)
            out["recent_median"] = round(new, 3)
            if old > 0 and (old - new) / old >= 0.2:
                wid = report_weakness(
                    db, "silent_degradation", str(subject or "?"),
                    {"metric": metric, "baseline": old, "recent": new,
                     "drop_pct": round(100 * (old - new) / old, 1)})
                out["weakness_id"] = wid
    except Exception:  # noqa: BLE001
        _log.debug("degradation check failed", exc_info=True)
    return out


# ── burn-rate severity ───────────────────────────────────────────────

def severity(db: Any, weakness_id: str,
             ts: float | None = None) -> dict[str, Any]:
    """Burn-rate severity for a weakness case.

    Borrows the SRE multi-window idea: a *fast* rate (last hour) vs a
    *slow* baseline (last 24h). Acute spikes score critical; chronic
    bleed scores warning; quiet cases stay on watch. Raw counts alone
    are either noisy or blind — burn rate is what matters.
    """
    ensure_schema(db)
    now = ts if ts is not None else time.time()
    try:
        fast = db.execute(
            "SELECT COUNT(*) FROM weakness_sightings "
            "WHERE weakness_id = ? AND ts >= ?",
            (weakness_id, now - 3600)).fetchone()[0] or 0
        slow_n = db.execute(
            "SELECT COUNT(*) FROM weakness_sightings "
            "WHERE weakness_id = ? AND ts >= ?",
            (weakness_id, now - 86400)).fetchone()[0] or 0
    except Exception:  # noqa: BLE001
        return {"level": "watch", "fast_1h": 0, "slow_per_h": 0.0,
                "burn": 0.0}
    slow_per_h = slow_n / 24.0
    burn = fast / max(slow_per_h, 0.05)
    if fast >= 3 and burn >= 3.0:
        level = "critical"
    elif burn >= 1.5 or fast >= 2:
        level = "warning"
    else:
        level = "watch"
    return {"level": level, "fast_1h": int(fast),
            "slow_per_h": round(slow_per_h, 2),
            "burn": round(burn, 2)}


def acceleration(db: Any, kind: str, subject: str) -> dict[str, Any]:
    """Is this weakness failing *faster* than before?

    Compares the last-24h sighting rate against the trailing 7-day
    average. A rising ratio is an escalation flag even when the raw
    count hasn't crossed the threshold yet.
    """
    ensure_schema(db)
    now = time.time()
    clustered = _cluster_subject(kind, subject)
    try:
        row = db.execute(
            "SELECT id FROM weaknesses WHERE kind = ? AND subject = ? "
            "AND status NOT IN ('resolved', 'dismissed')",
            (kind, clustered)).fetchone()
        if not row:
            return {"accelerating": False, "ratio": 0.0}
        wid = row[0]
        recent = db.execute(
            "SELECT COUNT(*) FROM weakness_sightings "
            "WHERE weakness_id = ? AND ts >= ?",
            (wid, now - 86400)).fetchone()[0] or 0
        older = db.execute(
            "SELECT COUNT(*) FROM weakness_sightings "
            "WHERE weakness_id = ? AND ts >= ? AND ts < ?",
            (wid, now - 7 * 86400, now - 86400)).fetchone()[0] or 0
    except Exception:  # noqa: BLE001
        return {"accelerating": False, "ratio": 0.0}
    older_per_day = older / 6.0
    ratio = recent / max(older_per_day, 0.2)
    return {"accelerating": bool(ratio >= 2.0 and recent >= 2),
            "ratio": round(ratio, 2),
            "recent_24h": int(recent),
            "prior_per_day": round(older_per_day, 2)}


# ── investigation / lifecycle (unchanged API) ────────────────────────

def investigate_weakness(db: Any, weakness_id: str) -> bool:
    """Hand a weakness to the research organ for investigation.

    The research organ's tick only drains ``directive.watch`` and
    ``directive.gap`` organ events — so the weakness is translated into
    a ``directive.gap`` (a question it genuinely investigates) rather
    than a custom kind that would be consumed and ignored. Returns True
    when the handoff was queued.
    """
    ensure_schema(db)
    row = db.execute(
        "SELECT kind, subject, evidence, occurrences FROM weaknesses "
        "WHERE id = ?", (weakness_id,),
    ).fetchone()
    if not row:
        return False
    kind, subject, evidence_raw, occurrences = row
    try:
        evidence = json.loads(evidence_raw or "{}")
    except (ValueError, TypeError):
        evidence = {}
    question = _investigation_question(kind, subject, evidence,
                                       int(occurrences or 0))
    try:
        _organs.emit(
            db, src="autonomy.weakness", dst="research",
            kind="directive.gap",
            payload={
                "question": question,
                "context": json.dumps({
                    "weakness_id": weakness_id, "kind": kind,
                    "subject": subject, "evidence": evidence,
                }),
                "origin": "weakness.investigation",
            })
        _ledger(db, "investigating", weakness_id,
                f"investigating {kind} {subject} via research organ",
                metadata={"question": question[:300]})
        _log.info("weakness %s → research (directive.gap)", weakness_id)
        try:
            global_bus.publish(Event(
                topic="weakness.detected",
                data={"id": weakness_id, "kind": kind, "subject": subject,
                      "severity": severity(db, weakness_id)["level"]},
                source="nomorals.autonomy.weakness",
            ))
        except Exception:  # noqa: BLE001
            pass
        return True
    except Exception:  # noqa: BLE001
        _log.warning("weakness→research handoff failed", exc_info=True)
        return False


def _investigation_question(kind: str, subject: str,
                            evidence: dict[str, Any],
                            occurrences: int) -> str:
    """Frame the weakness as a researchable question."""
    if kind == "tool_failure":
        return (
            f"Why is the tool '{subject}' failing? It failed {occurrences} "
            f"times in 24h. Known error evidence: "
            f"{json.dumps(evidence)[:400]}. Find the likely root cause and "
            "the minimal fix (prompt change, input validation, retry/backoff "
            "strategy, or replacement approach)."
        )
    if kind == "knowledge_gap":
        return (
            f"The system could not answer: '{subject}'. Research this topic "
            "and produce a concise, sourced brief the knowledge base can "
            "ingest."
        )
    if kind == "error_pattern":
        return (
            f"Exceptions are clustering around: {subject} "
            f"({occurrences} occurrences). Evidence: "
            f"{json.dumps(evidence)[:400]}. Identify the root cause pattern "
            "and the minimal hardening fix."
        )
    if kind == "retry_loop":
        return (
            f"The tool '{subject}' is stuck in retry loops "
            f"({occurrences} episodes). Evidence: "
            f"{json.dumps(evidence)[:400]}. Find why retries never succeed "
            "— bad input shaping, missing precondition check, or a prompt "
            "that keeps producing the same failing call — and propose the "
            "minimal circuit-breaker or input fix."
        )
    if kind == "cascading_failure":
        return (
            f"Cascading failure across: {subject} ({occurrences} episodes). "
            f"Evidence: {json.dumps(evidence)[:400]}. Identify the shared "
            "root cause (network, auth, rate limit, upstream outage) and "
            "the bulkhead/circuit-breaker that would contain it."
        )
    if kind == "silent_degradation":
        return (
            f"Silent quality degradation on '{subject}' "
            f"({occurrences} detections). Evidence: "
            f"{json.dumps(evidence)[:400]}. Find what drifted — data, "
            "model, prompt, or upstream — and how to restore the baseline."
        )
    # capability_gap
    return (
        f"The owner asked for '{subject}' and no tool covers it. Research "
        "how this capability could be built: existing APIs, libraries, and "
        "the smallest viable implementation approach."
    )


def idle_tick(db: Any) -> dict[str, Any]:
    """The idle-cycle weakness step. Called by the IdleCoordinator.

    Ensures every open/thresholded case is routed to research, expires
    stale cases (bounded growth), and returns a status report (never
    raises).
    """
    report: dict[str, Any] = {"routed": 0, "open": 0, "expired": 0}
    try:
        ensure_schema(db)
        report["expired"] = expire_stale(db)
        rows = db.execute(
            "SELECT id, status FROM weaknesses "
            "WHERE status NOT IN ('resolved', 'dismissed')"
        ).fetchall()
        report["open"] = len(rows)
        for wid, status in rows:
            if status in ("open",):
                # Thresholded but never routed (e.g. crashed before).
                occurrences = db.execute(
                    "SELECT occurrences FROM weaknesses WHERE id = ?",
                    (wid,)).fetchone()
                if occurrences and int(occurrences[0]) >= FAILURE_THRESHOLD:
                    db.execute(
                        "UPDATE weaknesses SET status = 'researching' "
                        "WHERE id = ?", (wid,))
                    try:
                        db.commit()
                    except Exception:  # noqa: BLE001
                        pass
                    if investigate_weakness(db, wid):
                        report["routed"] += 1
            elif status == "researching":
                # Still researching — the research organ owns it now.
                pass
    except Exception as exc:  # noqa: BLE001
        _log.warning("weakness idle_tick failed: %s", exc, exc_info=True)
        report["error"] = str(exc)[:200]
    return report


def expire_stale(db: Any,
                 stale_days: float = STALE_CASE_DAYS) -> int:
    """Auto-dismiss cases with no new evidence for ``stale_days``.

    Bounded growth: a bot running for months shouldn't carry a forever
    list of one-off hiccups. Dismissed-by-expiry cases can still be
    re-opened by fresh sightings. Never raises.
    """
    ensure_schema(db)
    cutoff = time.time() - stale_days * 86400
    try:
        cur = db.execute(
            "UPDATE weaknesses SET status = 'dismissed' "
            "WHERE status NOT IN ('resolved', 'dismissed') "
            "AND last_seen < ?",
            (cutoff,))
        try:
            db.commit()
        except Exception:  # noqa: BLE001
            pass
        n = cur.rowcount or 0
        if n:
            _log.info("expired %d stale weakness cases", n)
            _ledger(db, "expired", "stale",
                    f"auto-dismissed {n} stale weakness cases")
        return n
    except Exception as exc:  # noqa: BLE001
        _log.debug("expire_stale failed: %s", exc)
        return 0


def record_proposal(db: Any, weakness_id: str, proposal: str) -> None:
    """Record a fix proposal for a weakness. Owner approves via chat."""
    ensure_schema(db)
    db.execute(
        "UPDATE weaknesses SET status = 'proposed', proposal = ? "
        "WHERE id = ?",
        (proposal, weakness_id),
    )
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        pass
    _log.info("weakness proposal recorded: %s", weakness_id)
    _ledger(db, "proposed", weakness_id,
            f"fix proposal recorded for {weakness_id}",
            metadata={"proposal": proposal[:500]})
    try:
        global_bus.publish(Event(
            topic="weakness.proposed",
            data={"id": weakness_id, "proposal": proposal[:500]},
            source="nomorals.autonomy.weakness",
        ))
    except Exception:  # noqa: BLE001
        pass


def approve_weakness(db: Any, weakness_id: str) -> bool:
    """Owner approves a proposal → status approved, sandbox build queued.

    Approval doesn't build anything itself — it queues a sandbox-build
    request through the scheduler organ event and a ``weakness.approved``
    bus event. The improvement/scheduler systems pick those up without
    manual commands. The sandbox build itself still runs outside the
    live tree and the merge needs the owner.
    """
    ensure_schema(db)
    row = db.execute(
        "SELECT kind, subject, proposal FROM weaknesses "
        "WHERE id = ? AND status = 'proposed'",
        (weakness_id,),
    ).fetchone()
    if not row:
        return False
    kind, subject, proposal = row
    db.execute(
        "UPDATE weaknesses SET status = 'approved' WHERE id = ?",
        (weakness_id,),
    )
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        pass
    _log.info("weakness approved: %s", weakness_id)
    _ledger(db, "approved", weakness_id,
            f"proposal approved for {kind} {subject} — sandbox build queued",
            metadata={"kind": kind, "subject": subject})
    payload = {"weakness_id": weakness_id, "kind": kind,
               "subject": subject, "proposal": proposal,
               "sandbox": True}
    try:
        _organs.emit(db, src="autonomy.weakness", dst="scheduler",
                     kind="weakness.sandbox_build", payload=payload)
    except Exception:  # noqa: BLE001
        _log.warning("weakness→scheduler handoff failed", exc_info=True)
    try:
        global_bus.publish(Event(
            topic="weakness.approved",
            data=payload,
            source="nomorals.autonomy.weakness",
        ))
    except Exception:  # noqa: BLE001
        pass
    return True


def resolve_weakness(db: Any, weakness_id: str, how: str = "",
                     fix: str = "") -> bool:
    """Mark a weakness resolved (fixed, recovered, or obsolete).

    Pass ``fix`` with the verified fix text to also memorize it in the
    fixed-weakness memory — the next recurrence one-step resolves.
    """
    ensure_schema(db)
    cur = db.execute(
        "UPDATE weaknesses SET status = 'resolved' "
        "WHERE id = ? AND status NOT IN ('resolved', 'dismissed')",
        (weakness_id,),
    )
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        pass
    if cur.rowcount:
        if fix:
            try:
                record_fix(db, weakness_id, fix)
            except Exception:  # noqa: BLE001
                _log.debug("record_fix failed", exc_info=True)
        _ledger(db, "resolved", weakness_id,
                f"weakness resolved: {how or 'no reason given'}")
        try:
            global_bus.publish(Event(
                topic="weakness.resolved",
                data={"id": weakness_id, "how": how[:200]},
                source="nomorals.autonomy.weakness",
            ))
        except Exception:  # noqa: BLE001
            pass
    return bool(cur.rowcount)


def dismiss_weakness(db: Any, weakness_id: str) -> bool:
    """Owner dismisses a weakness without a fix."""
    ensure_schema(db)
    cur = db.execute(
        "UPDATE weaknesses SET status = 'dismissed' WHERE id = ?",
        (weakness_id,),
    )
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        pass
    if cur.rowcount:
        _ledger(db, "dismissed", weakness_id,
                f"weakness dismissed by owner: {weakness_id}")
    return bool(cur.rowcount)


def record_tool_result(db: Any, tool: str, ok: bool,
                       error: str | None = None) -> dict[str, Any]:
    """Feed one tool-loop outcome into detection AND recovery.

    Failures cluster into weakness cases (via :func:`report_weakness`);
    successes build an ok-streak that auto-resolves a ``tool_failure``
    case once the tool is healthy again. This closes the loop: fixed
    tools stop haunting the weakness list without manual dismissal.
    """
    ensure_schema(db)
    now = time.time()
    tool = str(tool or "?")
    db.execute(
        "INSERT INTO tool_health (tool, ok_streak, fail_streak, last_ts) "
        "VALUES (?, 0, 0, ?) "
        "ON CONFLICT (tool) DO UPDATE SET last_ts = excluded.last_ts",
        (tool, now),
    )
    out: dict[str, Any] = {"tool": tool}
    if ok:
        db.execute(
            "UPDATE tool_health SET ok_streak = ok_streak + 1, "
            "fail_streak = 0 WHERE tool = ?",
            (tool,),
        )
        streak = db.execute(
            "SELECT ok_streak FROM tool_health WHERE tool = ?",
            (tool,)).fetchone()
        ok_streak = int(streak[0]) if streak else 0
        out["ok_streak"] = ok_streak
        if ok_streak >= RECOVERY_STREAK:
            row = db.execute(
                "SELECT id FROM weaknesses WHERE kind = 'tool_failure' "
                "AND subject = ? AND status NOT IN "
                "('resolved', 'dismissed')",
                (tool,)).fetchone()
            if row:
                resolve_weakness(db, row[0],
                                 f"auto-resolved: {ok_streak} consecutive "
                                 "successes")
                out["auto_resolved"] = row[0]
    else:
        db.execute(
            "UPDATE tool_health SET fail_streak = fail_streak + 1, "
            "ok_streak = 0 WHERE tool = ?",
            (tool,),
        )
        wid = report_weakness(db, "tool_failure", tool,
                              {"error": (error or "")[:300],
                               "ts": now})
        out["weakness_id"] = wid
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        pass
    return out


def tool_health(db: Any, tool: str) -> dict[str, Any]:
    """Current health streaks for one tool."""
    ensure_schema(db)
    row = db.execute(
        "SELECT ok_streak, fail_streak, last_ts FROM tool_health "
        "WHERE tool = ?", (str(tool or "?",),)).fetchone()
    if not row:
        return {"tool": tool, "ok_streak": 0, "fail_streak": 0,
                "last_ts": 0.0}
    return {"tool": tool, "ok_streak": int(row[0]),
            "fail_streak": int(row[1]), "last_ts": float(row[2] or 0.0)}


def open_weaknesses(db: Any) -> list[dict[str, Any]]:
    """All weaknesses needing attention (with burn-rate severity)."""
    ensure_schema(db)
    rows = db.execute(
        "SELECT id, kind, subject, evidence, first_seen, last_seen, "
        "occurrences, status, proposal FROM weaknesses "
        "WHERE status NOT IN ('resolved', 'dismissed') "
        "ORDER BY occurrences DESC, last_seen DESC"
    ).fetchall()
    out = []
    for r in rows:
        try:
            sev = severity(db, r[0])["level"]
        except Exception:  # noqa: BLE001
            sev = "watch"
        out.append({
            "id": r[0], "kind": r[1], "subject": r[2],
            "evidence": json.loads(r[3] or "{}"),
            "first_seen": r[4], "last_seen": r[5],
            "occurrences": r[6], "status": r[7], "proposal": r[8],
            "severity": sev,
        })
    sev_rank = {"critical": 0, "warning": 1, "watch": 2}
    out.sort(key=lambda w: (sev_rank.get(w["severity"], 3),
                            -w["occurrences"]))
    return out


def scan_tool_failures(db: Any, tool_audit: list[dict[str, Any]]) -> int:
    """Feed tool-loop audit records into weakness detection.

    ``tool_audit``: list of {tool, ok, error, ts}. Returns new cases opened.
    Success records also feed recovery streaks.
    """
    opened = 0
    cutoff = time.time() - FAILURE_WINDOW
    failures: dict[str, int] = {}
    for rec in tool_audit:
        if rec.get("ts", 0) < cutoff:
            continue
        tool = str(rec.get("tool", "?"))
        if rec.get("ok"):
            record_tool_result(db, tool, True)
        else:
            failures[tool] = failures.get(tool, 0) + 1
    for tool, count in failures.items():
        if count >= FAILURE_THRESHOLD:
            report_weakness(db, "tool_failure", tool,
                            {"failures_in_24h": count})
            opened += 1
    return opened


def scan_ledger_failures(db: Any, window_hours: float = 24.0,
                         min_runs: int = 10,
                         rate_threshold: float = 0.5) -> list[str]:
    """Cross-system trigger: ledger failure rates → weakness cases.

    Any autonomous system whose recent failure rate crosses the threshold
    gets a weakness case automatically. This is systems watching each
    other — the scheduler's repeated failures, for example, become a
    self-investigation without anyone tripping a command.
    """
    from ..agents.autonomy_ledger import ledger_failure_rate
    opened: list[str] = []
    systems = ("scheduler", "mission", "trigger", "cognition", "pulse",
               "autonomy", "watcher", "idle", "presence", "weakness",
               "improvement")
    for system in systems:
        stats = ledger_failure_rate(db, system, window_hours=window_hours)
        if stats["runs"] >= min_runs and \
                stats["failure_rate"] >= rate_threshold:
            wid = report_weakness(
                db, "error_pattern", f"ledger:{system}",
                {"system": system,
                 "failure_rate": stats["failure_rate"],
                 "runs": stats["runs"], "failures": stats["failures"],
                 "window_hours": window_hours},
                fuzzy=False)
            opened.append(wid)
    return opened


# ── owner-facing digest ──────────────────────────────────────────────

_SEV_ICON = {"critical": "🔴", "warning": "🟡", "watch": "⚪"}
_STATUS_ICON = {
    "open": "🆕", "researching": "🔍", "proposed": "📝",
    "approved": "✅", "resolved": "✔️", "dismissed": "🚫",
}


def weakness_digest(db: Any, limit: int = 10,
                    theme: str = "plain") -> str:
    """Human-readable weakness report for the owner.

    ``theme``: ``"plain"`` (chat-friendly) or ``"rich"`` (boxed, iconed).
    Lists open cases by severity with their status and what happens next.
    """
    items = open_weaknesses(db)[:max(1, int(limit))]
    if not items:
        if theme == "rich":
            return "╭─ 🛡️ weakness watch ─╮\n│ all clear — no open cases │\n╰─────────────────────╯"
        return "🛡️ weakness watch: all clear — no open cases."
    lines: list[str] = []
    if theme == "rich":
        lines.append("╭─ 🛡️ weakness watch ──────────────────╮")
    else:
        lines.append("🛡️ weakness watch")
    for w in items:
        sev = w.get("severity", "watch")
        icon = _SEV_ICON.get(sev, "⚪") if theme == "rich" \
            else _SEV_ICON.get(sev, "")
        st = _STATUS_ICON.get(w["status"], "") if theme == "rich" else ""
        nxt = {
            "open": "gathering evidence",
            "researching": "research organ investigating",
            "proposed": "awaiting your approval",
            "approved": "sandbox build queued",
        }.get(w["status"], w["status"])
        if theme == "rich":
            lines.append(
                f"│ {icon} [{sev:^8}] {w['kind']}: {w['subject'][:38]}")
            lines.append(
                f"│   {st} {w['status']} · ×{w['occurrences']} · {nxt}")
        else:
            lines.append(
                f"{icon} [{sev}] {w['kind']}: {w['subject']} "
                f"(×{w['occurrences']}, {w['status']} — {nxt})")
    if theme == "rich":
        lines.append("╰──────────────────────────────────────╯")
    else:
        lines.append(f"{len(items)} open case(s). Approve proposals with "
                     "the weakness id.")
    return "\n".join(lines)

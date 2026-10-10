"""Always-on research loop (Wave E): scheduled swarm -> digest -> upgrade queue.

Wave C built the research organs as on-demand tools: ``ResearchSwarm``
(fan out N specialist angles in parallel), ``ResearchPipeline`` in
``research_digest`` (findings -> versioned KG claims with conflict
detection -> digests -> upgrade tickets), and ``UpgradePipeline`` /
``UpgradeQueue`` in ``upgrade_queue`` (owner approve/deny -> evolution
agent implements). This module is the scheduled heart that keeps them
beating on their own:

* ``ensure_research_job`` — one durable scheduler job (``"research loop"``,
  ``every NM_RESEARCH_LOOP_HOURS``, default 6h), idempotent by name like
  the briefing/persona jobs. The scheduler calls the ``research_loop``
  tool's ``tick`` action.
* ``ResearchLoop.tick`` — one cycle:
    1. Re-check the owner's gates (the ``research`` feature flag, the
       proactive master switch ``partner.proactive_enabled``, quiet
       hours). A gated-out tick records a skipped run and defers — it
       never burns network/model budget against the owner's wishes.
    2. Pick topics: owner seeds (kv ``research_loop.topics``) plus KG
       follow-ups (contradicted claims that need fresh evidence), with
       anti-repeat (no topic researched in the last 3 runs).
    3. Run each topic through ``ResearchPipeline.run`` (swarm ->
       claims -> KG promotion -> digest -> upgrade tickets).
    4. File every qualifying ticket into the upgrade queue via
       ``UpgradePipeline.propose_from_ticket`` (source="research_loop").
       **The loop never approves or applies anything** — each proposal
       waits for the owner in the existing approve/deny flow.
    5. Notify the owner with the digest brief through the ``Notifier``
       (dedupe + feature-flag + delivery-state tracking) and persist
       the run in ``research_loop_runs``.
* ``status`` — job health, last/next run, findings + proposals counts,
  pending proposals: everything ``nm status`` needs to show.

Reuse, not duplication: the durable scheduling primitives live in
``scheduler.py``; the pipeline, tickets, and approve flow are used
unmodified; gates are the existing ``features`` flag and
``notifier.proactive_gate``/quiet-hours state — no parallel settings
system.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..storage.kv import KVStore
from .features import feature_enabled
from .notifier import Notifier, _in_quiet_hours_now

_log = get_logger(__name__)

__all__ = [
    "RESEARCH_LOOP_JOB",
    "ResearchLoop",
    "ensure_research_job",
    "loop_gates",
    "owner_topics",
    "set_owner_topics",
    "pick_topics",
    "research_hours",
    "research_max_proposals",
    "research_dedupe_days",
    "research_ignore_days",
    "usefulness_signals",
    "signals_summary",
    "status",
    "register",
]

#: Name of the durable scheduler job. Idempotent by name.
RESEARCH_LOOP_JOB = "research loop"

#: kv_store key holding the owner's topic list (JSON).
_TOPICS_KEY = "research_loop.topics"

#: Never re-research a topic that appeared in the last this many runs.
_ANTI_REPEAT_RUNS = 3

#: Bounded per tick so one cycle can't eat the network budget.
_MAX_TOPICS_PER_TICK = 2

#: Default interval between cycles (hours). Env: NM_RESEARCH_LOOP_HOURS.
_DEFAULT_HOURS = 6.0

#: Default max new upgrade-queue proposals filed per cycle.
#: Env: NM_RESEARCH_LOOP_MAX_PROPOSALS.
_DEFAULT_MAX_PROPOSALS = 5

#: How far back (days) the dedupe check looks for an already-filed
#: proposal of the same finding. Env: NM_RESEARCH_LOOP_DEDUPE_DAYS.
_DEFAULT_DEDUPE_DAYS = 14

#: A filed proposal still undecided after this many days counts as
#: "ignored" in the usefulness signals (the owner saw it and passed).
#: Env: NM_RESEARCH_LOOP_IGNORE_DAYS.
_DEFAULT_IGNORE_DAYS = 7

#: Title token overlap that marks two proposals as the same finding.
_TITLE_DUP_JACCARD = 0.65
#: Same files targeted + this token overlap also marks a duplicate.
_TITLE_FILE_DUP_JACCARD = 0.40

#: Fallback seeds when the owner set no topics. Product-relevant, evergreen,
#: and deliberately phrased to surface actionable upgrade findings.
_DEFAULT_TOPICS: tuple[str, ...] = (
    "autonomous coding agent self-improvement loop techniques",
    "AI agent long-term memory architectures knowledge graph",
    "retrieval augmented generation evaluation benchmarks 2026",
    "agentic workflow orchestration patterns reliability",
)


# ── interval config ──────────────────────────────────────────────────────────

def research_hours() -> float:
    """Cycle interval in hours. Env ``NM_RESEARCH_LOOP_HOURS`` (default 6),
    clamped to [1, 48]. Read at schedule time so the owner can retune the
    job without editing config."""
    raw = os.environ.get("NM_RESEARCH_LOOP_HOURS", "")
    try:
        hours = float(raw) if raw.strip() else _DEFAULT_HOURS
    except (TypeError, ValueError):
        _log.warning("NM_RESEARCH_LOOP_HOURS=%r not a number — using %.0f",
                     raw, _DEFAULT_HOURS)
        hours = _DEFAULT_HOURS
    return min(48.0, max(1.0, hours))


def research_max_proposals() -> int:
    """Max new upgrade-queue proposals filed per cycle. Env
    ``NM_RESEARCH_LOOP_MAX_PROPOSALS`` (default 5), clamped to [1, 50].
    Read per cycle so the owner can retune noise without a restart."""
    raw = os.environ.get("NM_RESEARCH_LOOP_MAX_PROPOSALS", "")
    try:
        n = int(raw) if raw.strip() else _DEFAULT_MAX_PROPOSALS
    except (TypeError, ValueError):
        _log.warning("NM_RESEARCH_LOOP_MAX_PROPOSALS=%r not a number — "
                     "using %d", raw, _DEFAULT_MAX_PROPOSALS)
        n = _DEFAULT_MAX_PROPOSALS
    return min(50, max(1, n))


def research_dedupe_days() -> int:
    """Dedupe lookback window (days) for cross-cycle proposal dedup.
    Env ``NM_RESEARCH_LOOP_DEDUPE_DAYS`` (default 14), clamped [1, 90]."""
    raw = os.environ.get("NM_RESEARCH_LOOP_DEDUPE_DAYS", "")
    try:
        n = int(raw) if raw.strip() else _DEFAULT_DEDUPE_DAYS
    except (TypeError, ValueError):
        _log.warning("NM_RESEARCH_LOOP_DEDUPE_DAYS=%r not a number — "
                     "using %d", raw, _DEFAULT_DEDUPE_DAYS)
        n = _DEFAULT_DEDUPE_DAYS
    return min(90, max(1, n))


def research_ignore_days() -> int:
    """Days a filed proposal may sit undecided before it counts as
    ``ignored`` in the usefulness signals. Env
    ``NM_RESEARCH_LOOP_IGNORE_DAYS`` (default 7), clamped [1, 365]."""
    raw = os.environ.get("NM_RESEARCH_LOOP_IGNORE_DAYS", "")
    try:
        n = int(raw) if raw.strip() else _DEFAULT_IGNORE_DAYS
    except (TypeError, ValueError):
        _log.warning("NM_RESEARCH_LOOP_IGNORE_DAYS=%r not a number — "
                     "using %d", raw, _DEFAULT_IGNORE_DAYS)
        n = _DEFAULT_IGNORE_DAYS
    return min(365, max(1, n))


# ── owner gates (existing systems only) ──────────────────────────────────────

def loop_gates(context: Any) -> tuple[bool, str]:
    """Re-check the owner's standing switches before a cycle runs.

    Returns (ok, reason): ``ok`` when the loop may burn network budget,
    otherwise the human-readable hold reason. The checks reuse exactly
    the existing owner controls — the ``research`` feature flag, the
    proactive master switch, and the partner quiet-hours window — so the
    loop never needs its own enable/disable UI.
    """
    if not feature_enabled(context, "research"):
        return False, "feature 'research' off (/features research on)"
    partner = getattr(getattr(context, "settings", None), "partner", None)
    if partner is not None and not bool(
            getattr(partner, "proactive_enabled", True)):
        return False, "proactive master switch off (NM_PARTNER_PROACTIVE_ENABLED=0)"
    try:
        if _in_quiet_hours_now(context):
            return False, "quiet hours — deferred to the next cycle"
    except Exception as exc:  # noqa: BLE001 - gate failure must not kill the loop
        _log.warning("quiet-hours check failed: %s", exc)
    return True, ""


# ── topic management ─────────────────────────────────────────────────────────

def _kv_get(context: Any, key: str) -> Any:
    db = getattr(context, "db", None)
    if db is None:
        return None
    try:
        return KVStore(db).get(key)
    except Exception:  # noqa: BLE001 - kv is best-effort
        return None


def _kv_put(context: Any, key: str, value: Any) -> bool:
    db = getattr(context, "db", None)
    if db is None:
        return False
    try:
        KVStore(db).set(key, value)
        return True
    except Exception as exc:  # noqa: BLE001
        _log.warning("kv put failed (%s): %s", key, exc)
        return False


def owner_topics(context: Any) -> list[str]:
    """The owner's topic list (kv), or the built-in seeds when unset."""
    stored = _kv_get(context, _TOPICS_KEY)
    if isinstance(stored, list):
        topics = [str(t).strip() for t in stored if str(t).strip()]
        if topics:
            return topics[:12]
    return list(_DEFAULT_TOPICS)


def set_owner_topics(context: Any, topics: list[str]) -> list[str]:
    """Replace the owner's topic list (max 12). Returns the stored list."""
    clean = [str(t).strip() for t in (topics or []) if str(t).strip()][:12]
    if not clean:
        raise ValueError("need at least one non-empty topic")
    if not _kv_put(context, _TOPICS_KEY, clean):
        raise RuntimeError("could not persist topics (no database on context)")
    return clean


def _contradicted_topics(context: Any, limit: int = 4) -> list[str]:
    """KG follow-ups: claims marked ``contradicted`` need fresh evidence.
    Returns their originating queries, newest first."""
    db = getattr(context, "db", None)
    if db is None:
        return []
    try:
        rows = db.query(
            "SELECT properties FROM kg_nodes WHERE type='claim' "
            "ORDER BY created_at DESC LIMIT 200")
    except Exception:  # noqa: BLE001 - KG tables may not exist yet
        return []
    out: list[str] = []
    for row in rows or []:
        try:
            props = json.loads(row.get("properties") or "{}")
        except (TypeError, ValueError):
            continue
        if str(props.get("status", "")) != "contradicted":
            continue
        query = str(props.get("query", "") or "").strip()
        if query and query not in out:
            out.append(query)
        if len(out) >= limit:
            break
    return out


def _recent_topics(context: Any, runs: int = _ANTI_REPEAT_RUNS) -> set[str]:
    """Topics covered by the last ``runs`` cycles (anti-repeat)."""
    db = getattr(context, "db", None)
    if db is None:
        return set()
    try:
        rows = db.query(
            "SELECT topics FROM research_loop_runs "
            "ORDER BY started_at DESC LIMIT ?", (runs,))
    except Exception:  # noqa: BLE001 - migration not applied yet
        return set()
    seen: set[str] = set()
    for row in rows or []:
        try:
            for t in json.loads(row.get("topics") or "[]"):
                seen.add(str(t).strip().lower())
        except (TypeError, ValueError):
            continue
    return seen


def pick_topics(context: Any, limit: int = _MAX_TOPICS_PER_TICK) -> list[str]:
    """Rotate topics: owner seeds first, then KG contradicted-follow-ups,
    skipping anything covered in the last ``_ANTI_REPEAT_RUNS`` cycles."""
    recent = _recent_topics(context)
    candidates: list[str] = []
    for topic in owner_topics(context) + _contradicted_topics(context):
        key = topic.strip().lower()
        if key and key not in recent and key not in candidates:
            candidates.append(key)
        if len(candidates) >= limit:
            break
    return candidates


# ── cross-cycle proposal dedupe ────────────────────────────────────────────

def _norm_title(title: str) -> str:
    """Lowercased, whitespace-collapsed title for exact-match dedup."""
    return " ".join(str(title or "").lower().split())


def _title_tokens(title: str) -> set[str]:
    """Alphanumeric tokens (len>=3) — the similarity vocabulary."""
    import re

    return set(re.findall(r"[a-z0-9]{3,}", str(title or "").lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _fingerprints_duplicate(a: tuple, b: tuple) -> bool:
    """Two (norm_title, tokens, frozenset(files)) fingerprints describe the
    same finding when the titles are identical, the titles are near-copies,
    or the tickets target the same files with clearly related titles."""
    if a[0] and a[0] == b[0]:
        return True
    if _jaccard(a[1], b[1]) >= _TITLE_DUP_JACCARD:
        return True
    if a[2] and b[2] and (a[2] & b[2]) and \
            _jaccard(a[1], b[1]) >= _TITLE_FILE_DUP_JACCARD:
        return True
    return False


def _ticket_fingerprint(title: str, files: list[str]) -> tuple:
    tokens = _title_tokens(title)
    return (_norm_title(title), tokens,
            frozenset(str(f).strip() for f in (files or []) if f))


def _recent_research_proposals(context: Any, days: int) -> list[tuple]:
    """Fingerprints of proposals filed by this loop in the last ``days``
    days, newest first. The ``source`` marker is stashed inside
    ``patch_plan`` at write time (no dedicated column); the comparison is
    done in Python so it never depends on SQLite JSON functions."""
    db = getattr(context, "db", None)
    if db is None:
        return []
    cutoff = time.time() - float(days) * 86400.0
    try:
        rows = db.query(
            "SELECT title, files, patch_plan FROM upgrade_proposals "
            "WHERE created_at >= ? ORDER BY created_at DESC LIMIT 500",
            (cutoff,))
    except Exception:  # noqa: BLE001 - table may not exist yet
        return []
    out: list[tuple] = []
    for row in rows or []:
        try:
            plan = json.loads(row.get("patch_plan") or "{}")
        except (TypeError, ValueError):
            continue
        if not isinstance(plan, dict) or plan.get("source") != "research_loop":
            continue
        try:
            files = json.loads(row.get("files") or "[]")
        except (TypeError, ValueError):
            files = []
        out.append(_ticket_fingerprint(str(row.get("title") or ""),
                                      list(files or [])))
    return out


def is_duplicate_proposal(context: Any, title: str, files: list[str],
                          days: int | None = None,
                          extra: list[tuple] | None = None) -> bool:
    """True when a proposal with this title/files was already filed by the
    loop in the last ``days`` days (default ``research_dedupe_days()``).
    Pass ``extra`` fingerprints to also check ones seen earlier in the
    current cycle (the caller appends newly filed ones as it goes)."""
    fps = list(_recent_research_proposals(
        context, research_dedupe_days() if days is None else days))
    if extra:
        fps.extend(extra)
    mine = _ticket_fingerprint(title, files)
    return any(_fingerprints_duplicate(mine, fp) for fp in fps)


# ── usefulness signals (raw counts only — no synthesized score) ──────────────

def usefulness_signals(context: Any) -> dict[str, int]:
    """Honest raw counts over every proposal this loop ever filed
    (``patch_plan.source == "research_loop"``):

    * ``proposed`` — total filed.
    * ``approved`` — owner said yes (approved, implemented, or failed).
    * ``denied`` — owner said no.
    * ``ignored`` — still ``proposed`` after ``research_ignore_days()``
      days with no owner decision.
    * ``pending`` — still ``proposed`` but inside the ignore window.
    * ``applied`` — reached implementation (implemented + failed).
    * ``tests_passed`` / ``tests_failed`` — implementation outcomes from
      ``record_implemented`` results (``ok`` true vs false).

    All zeros when there is no data — never a guess, never a score."""
    sig = {"proposed": 0, "approved": 0, "denied": 0, "ignored": 0,
           "pending": 0, "applied": 0, "tests_passed": 0, "tests_failed": 0}
    db = getattr(context, "db", None)
    if db is None:
        return sig
    try:
        rows = db.query(
            "SELECT status, created_at, applied_result, patch_plan "
            "FROM upgrade_proposals")
    except Exception:  # noqa: BLE001 - table may not exist yet
        return sig
    cutoff = time.time() - float(research_ignore_days()) * 86400.0
    for row in rows or []:
        try:
            plan = json.loads(row.get("patch_plan") or "{}")
        except (TypeError, ValueError):
            continue
        if not isinstance(plan, dict) or plan.get("source") != "research_loop":
            continue
        sig["proposed"] += 1
        st = str(row.get("status") or "")
        if st == "denied":
            sig["denied"] += 1
        elif st in ("approved", "implemented", "failed"):
            sig["approved"] += 1
            if st == "implemented":
                sig["applied"] += 1
                sig["tests_passed"] += 1
            elif st == "failed":
                sig["applied"] += 1
                sig["tests_failed"] += 1
        elif st == "proposed":
            if float(row.get("created_at") or 0.0) < cutoff:
                sig["ignored"] += 1
            else:
                sig["pending"] += 1
    return sig


def signals_summary(signals: dict[str, int] | None) -> str:
    """One-line owner-facing summary, e.g.
    ``12 proposed · 3 approved · 2 denied · 7 ignored · applied 3/3 tests
    passed``. Honest zeros when there is no data."""
    s = signals or {}
    p = int(s.get("proposed", 0) or 0)
    a = int(s.get("approved", 0) or 0)
    d = int(s.get("denied", 0) or 0)
    ig = int(s.get("ignored", 0) or 0)
    ap = int(s.get("applied", 0) or 0)
    tp = int(s.get("tests_passed", 0) or 0)
    return (f"{p} proposed · {a} approved · {d} denied · {ig} ignored "
            f"· applied {tp}/{ap} tests passed")


# ── run history ──────────────────────────────────────────────────────────────

def record_run(context: Any, record: dict[str, Any]) -> str:
    """Persist one cycle's outcome. Returns the run id."""
    db = getattr(context, "db", None)
    rid = str(record.get("id") or new_id("rlrun"))
    if db is None:
        return rid
    try:
        db.execute(
            "INSERT INTO research_loop_runs "
            "(id, started_at, finished_at, ok, skipped_reason, topics, "
            "findings_count, claims_count, proposals_created, proposal_ids, "
            "notified, error, proposals_deduped, proposals_capped, signals) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (rid,
             float(record.get("started_at", 0.0) or 0.0),
             float(record.get("finished_at", 0.0) or 0.0),
             1 if record.get("ok") else 0,
             str(record.get("skipped_reason", "") or ""),
             json.dumps(list(record.get("topics") or [])),
             int(record.get("findings_count", 0) or 0),
             int(record.get("claims_count", 0) or 0),
             int(record.get("proposals_created", 0) or 0),
             json.dumps(list(record.get("proposal_ids") or [])),
             1 if record.get("notified") else 0,
             str(record.get("error", "") or ""),
             int(record.get("proposals_deduped", 0) or 0),
             int(record.get("proposals_capped", 0) or 0),
             json.dumps(record.get("signals") or {})),
        )
    except Exception as exc:  # noqa: BLE001 - history is bookkeeping, not the mission
        _log.warning("could not record research loop run: %s", exc)
    return rid


def last_run(context: Any) -> dict[str, Any] | None:
    """The most recent recorded cycle (None when never run)."""
    db = getattr(context, "db", None)
    if db is None:
        return None
    try:
        row = db.query_one(
            "SELECT * FROM research_loop_runs ORDER BY started_at DESC LIMIT 1")
    except Exception:  # noqa: BLE001 - migration not applied yet
        return None
    if not row:
        return None
    out = dict(row)
    for key in ("topics", "proposal_ids"):
        try:
            out[key] = json.loads(out.get(key) or "[]")
        except (TypeError, ValueError):
            out[key] = []
    try:
        out["signals"] = json.loads(out.get("signals") or "{}")
    except (TypeError, ValueError):
        out["signals"] = {}
    return out


# ── the loop ─────────────────────────────────────────────────────────────────

class ResearchLoop:
    """One scheduled research cycle."""

    def __init__(self, context: Any, *,
                 max_topics: int = _MAX_TOPICS_PER_TICK,
                 min_confidence: float = 0.5,
                 specialists: list[str] | None = None,
                 max_proposals: int | None = None) -> None:
        self.context = context
        self.max_topics = max(1, int(max_topics))
        self.min_confidence = min_confidence
        self.specialists = specialists
        # Per-cycle proposal cap (rate-limit the noise). None = the env
        # default NM_RESEARCH_LOOP_MAX_PROPOSALS.
        self.max_proposals = (max(1, int(max_proposals))
                              if max_proposals
                              else max(1, research_max_proposals()))

    def tick(self) -> dict[str, Any]:
        """Run one cycle: gates -> topics -> pipeline -> upgrade queue ->
        owner notify -> record. Never raises."""
        started = time.time()
        record: dict[str, Any] = {
            "id": new_id("rlrun"), "started_at": started,
            "ok": False, "skipped_reason": "", "topics": [],
            "findings_count": 0, "claims_count": 0,
            "proposals_created": 0, "proposal_ids": [],
            "proposals_deduped": 0, "proposals_capped": 0,
            "signals": {},
            "notified": False, "error": "",
        }
        try:
            ok, reason = loop_gates(self.context)
            if not ok:
                record["skipped_reason"] = reason
                _log.info("research loop tick deferred: %s", reason)
                return self._finish(record)
            topics = pick_topics(self.context, self.max_topics)
            if not topics:
                record["skipped_reason"] = "no fresh topics (all covered recently)"
                return self._finish(record)
            record["topics"] = topics
            return self._finish(self._run_cycle(record, topics))
        except Exception as exc:  # noqa: BLE001 - a cycle must never crash the scheduler
            _log.exception("research loop tick failed")
            record["error"] = f"{type(exc).__name__}: {exc}"
            return self._finish(record)

    def _run_cycle(self, record: dict[str, Any],
                   topics: list[str]) -> dict[str, Any]:
        from .research_digest import ResearchPipeline
        from .upgrade_queue import UpgradePipeline

        briefs: list[str] = []
        proposal_ids: list[str] = []
        topics_done: list[str] = []
        # Cross-cycle dedupe fingerprints: proposals this loop already
        # filed (plus ones filed earlier in this cycle) so the same
        # finding never spawns a second ticket.
        fingerprints = _recent_research_proposals(self.context,
                                                 research_dedupe_days())
        deduped = 0
        capped = 0
        for topic in topics:
            try:
                result = ResearchPipeline.run(
                    topic, self.context,
                    specialists=self.specialists,
                    min_confidence=self.min_confidence,
                    promote_claims=True)
            except Exception as exc:  # noqa: BLE001 - one dead topic, not a dead cycle
                _log.warning("research loop topic %r failed: %s", topic, exc)
                continue
            topics_done.append(topic)
            report = result.get("report") or {}
            record["findings_count"] += len(report.get("findings") or [])
            record["claims_count"] += len(result.get("claims") or [])
            pipeline = UpgradePipeline(self.context)
            for ticket in result.get("tickets") or []:
                title = str(ticket.get("title") or "")
                files = [str(f) for f in
                         (ticket.get("suggested_files") or []) if f]
                fp = _ticket_fingerprint(title, files)
                if any(_fingerprints_duplicate(fp, seen)
                       for seen in fingerprints):
                    deduped += 1
                    _log.info("research loop deduped ticket %r "
                              "(already proposed recently)", title[:80])
                    continue
                if len(proposal_ids) >= self.max_proposals:
                    capped += 1
                    _log.info("research loop dropped ticket %r: per-cycle "
                              "proposal cap (%d) reached", title[:80],
                              self.max_proposals)
                    continue
                try:
                    pid = pipeline.propose_from_ticket(
                        ticket, source="research_loop")
                except Exception as exc:  # noqa: BLE001 - the ticket stays in the digest
                    _log.warning("research loop proposal failed: %s", exc)
                    continue
                if pid:
                    proposal_ids.append(pid)
                    fingerprints.append(fp)
            brief = str(result.get("brief") or "").strip()
            if brief:
                briefs.append(brief)
        record["topics"] = topics_done
        record["proposals_created"] = len(proposal_ids)
        record["proposal_ids"] = proposal_ids
        record["proposals_deduped"] = deduped
        record["proposals_capped"] = capped
        record["ok"] = bool(topics_done)
        if briefs:
            record["notified"] = self._notify(briefs, record)
        return record

    def _notify(self, briefs: list[str], record: dict[str, Any]) -> bool:
        """Send the cycle's digest briefs to the owner — exactly ONE digest
        publish per cycle (the notification cap), never one per finding.
        A one-line usefulness-signals summary is appended so the owner sees
        what the loop's proposals have earned so far. The notifier applies
        dedupe, the notifier feature flag, and delivery-state tracking; it
        returns delivery info (never raises)."""
        try:
            title = ("research loop: %d finding(s), %d proposal(s)"
                     % (record["findings_count"],
                        record["proposals_created"]))
            body = "\n\n".join(briefs)[:3000]
            summary = signals_summary(record.get("signals") or
                                      usefulness_signals(self.context))
            body = f"{body}\n\nsignals: {summary}"[:3200]
            res = Notifier(self.context).publish("research", title, body)
            return bool(isinstance(res, dict)
                        and res.get("delivered") not in (False, 0)
                        and not res.get("deduped"))
        except Exception as exc:  # noqa: BLE001 - notify is best-effort
            _log.warning("research loop notify failed: %s", exc)
            return False

    def _finish(self, record: dict[str, Any]) -> dict[str, Any]:
        record["finished_at"] = time.time()
        # Snapshot the usefulness signals into the run record so the
        # owner can see what the loop's track record looked like when
        # this cycle finished.
        try:
            record["signals"] = usefulness_signals(self.context)
        except Exception as exc:  # noqa: BLE001 - bookkeeping, not the mission
            _log.warning("research loop signals snapshot failed: %s", exc)
            record["signals"] = {}
        rid = record_run(self.context, record)
        out = {
            "ok": record["ok"],
            "run_id": rid,
            "skipped_reason": record["skipped_reason"],
            "topics": record["topics"],
            "findings_count": record["findings_count"],
            "claims_count": record["claims_count"],
            "proposals_created": record["proposals_created"],
            "proposal_ids": record["proposal_ids"],
            "proposals_deduped": record.get("proposals_deduped", 0),
            "proposals_capped": record.get("proposals_capped", 0),
            "signals": record.get("signals", {}),
            "notified": record["notified"],
            "error": record["error"],
            "seconds": round(record["finished_at"] - record["started_at"], 2),
        }
        return out


# ── durable job ──────────────────────────────────────────────────────────────

def ensure_research_job(context: Any) -> dict[str, Any]:
    """Register the single durable ``research loop`` job (idempotent).

    Runs ``every NM_RESEARCH_LOOP_HOURS`` (default 6h) via the
    ``research_loop`` tool's ``tick`` action. Safe to call on every boot:
    if the owner changed the interval (or env), the stale job is
    replaced; the loop itself re-checks the feature flag, the proactive
    master switch, and quiet hours before burning budget.
    """
    from .scheduler import Scheduler

    sched = Scheduler(context)
    hours = research_hours()
    want = f"every {int(hours * 3600)}s"
    try:
        have = [j for j in sched.list_jobs()
                if j.get("name") == RESEARCH_LOOP_JOB]
    except Exception:  # noqa: BLE001 — scheduler table may not exist yet
        have = []
    if have and have[0].get("spec") == want:
        return {"name": RESEARCH_LOOP_JOB, "already_scheduled": True,
                "job_id": have[0].get("id"), "interval_hours": hours}
    for j in have:  # stale interval -> replace
        try:
            sched.remove(j["id"])
        except Exception:  # noqa: BLE001
            pass
    job = sched.add(RESEARCH_LOOP_JOB, f"every {hours:g}h", "tool",
                    {"tool": "research_loop", "args": {"action": "tick"}})
    _log.info("scheduled research loop job: every %.2gh", hours)
    return {"name": RESEARCH_LOOP_JOB, "scheduled": True,
            "job_id": job.get("id"), "interval_hours": hours}


# ── status ───────────────────────────────────────────────────────────────────

def status(context: Any) -> dict[str, Any]:
    """Everything ``nm status`` needs: job health, gates, last/next run,
    cycle totals, and pending proposals in the owner's queue."""
    from .scheduler import Scheduler
    from .upgrade_queue import UpgradeQueue

    out: dict[str, Any] = {
        "job": {"scheduled": False, "enabled": False, "next_run": None,
                "next_run_iso": None, "interval_hours": research_hours()},
        "gates": {"feature_research": feature_enabled(context, "research"),
                  "proactive_master": True, "quiet_hours": False},
        "last_run": None,
        "pending_proposals": 0,
    }
    partner = getattr(getattr(context, "settings", None), "partner", None)
    if partner is not None:
        out["gates"]["proactive_master"] = bool(
            getattr(partner, "proactive_enabled", True))
    try:
        out["gates"]["quiet_hours"] = bool(_in_quiet_hours_now(context))
    except Exception:  # noqa: BLE001
        pass
    try:
        sched = Scheduler(context)
        jobs = [j for j in sched.list_jobs()
                if j.get("name") == RESEARCH_LOOP_JOB]
        if jobs:
            j = jobs[0]
            out["job"].update({
                "scheduled": True, "enabled": bool(j.get("enabled")),
                "next_run": j.get("next_run"),
                "next_run_iso": j.get("next_run_iso"),
                "interval_hours": research_hours(),
            })
    except Exception:  # noqa: BLE001
        pass
    out["last_run"] = last_run(context)
    try:
        out["pending_proposals"] = len(
            UpgradeQueue(context).list(status="proposed"))
    except Exception:  # noqa: BLE001
        pass
    out["topics"] = owner_topics(context)
    out["signals"] = usefulness_signals(context)
    return out


# ── tool registration ────────────────────────────────────────────────────────

def run_topic(context: Any, topic: str) -> dict[str, Any]:
    """Run one ad-hoc research cycle on a single topic (the tool's
    ``action=run`` and the ``nm research-loop run`` CLI share this).

    Honors the loop gates first: when the gates say no, the run is
    recorded as deferred with the skip reason instead of burning budget.
    """
    q = (topic or "").strip()
    if not q:
        raise ValueError("run_topic needs a topic")
    loop = ResearchLoop(context, max_topics=1)
    record: dict[str, Any] = {
        "id": new_id("rlrun"), "started_at": time.time(),
        "ok": False, "skipped_reason": "", "topics": [q],
        "findings_count": 0, "claims_count": 0,
        "proposals_created": 0, "proposal_ids": [],
        "proposals_deduped": 0, "proposals_capped": 0,
        "signals": {},
        "notified": False, "error": "",
    }
    ok, reason = loop_gates(context)
    if not ok:
        record["skipped_reason"] = reason
        record["finished_at"] = time.time()
        record_run(context, record)
        return {"ok": True, "deferred": True,
                "skipped_reason": reason, "topic": q}
    outcome = loop._run_cycle(record, [q])
    outcome["finished_at"] = time.time()
    try:
        outcome["signals"] = usefulness_signals(context)
    except Exception:  # noqa: BLE001 - bookkeeping only
        outcome["signals"] = {}
    rid = record_run(context, outcome)
    out = dict(outcome)
    out["run_id"] = rid
    return {"ok": True, **out}


def register(registry: Any) -> None:
    from ..core.policy import Capability

    context = registry.context

    @registry.register(
        "research_loop",
        description=(
            "always-on research loop (Wave E): run the scheduled research "
            "cycle on demand, check its status, or manage its topics and "
            "durable job. A tick runs swarm -> digest -> upgrade-queue "
            "proposals (owner approve/deny gate — nothing is auto-applied). "
            "Actions: tick | run | status | ensure | enable | disable | "
            "topics | set_topics."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "action": "str — tick | run | status | ensure | enable | disable "
                      "| topics | set_topics (default status)",
            "topic": "str (optional) — single topic for action=run",
            "topics": "list (optional) — topic list for action=set_topics",
            "max_topics": "int (optional, default 2) — topics per cycle",
        },
    )
    def research_loop(
        action: str = "status",
        topic: str = "",
        topics: Any = None,
        max_topics: int = 0,
        **_: Any,
    ) -> dict[str, Any]:
        verb = (action or "status").strip().lower()
        if verb == "status":
            return {"ok": True, **status(context)}
        if verb == "topics":
            return {"ok": True, "topics": owner_topics(context)}
        if verb == "set_topics":
            if isinstance(topics, str):
                topics = [t.strip() for t in topics.split(",") if t.strip()]
            return {"ok": True, "topics": set_owner_topics(context,
                                                           list(topics or []))}
        if verb == "ensure":
            return {"ok": True, **ensure_research_job(context)}
        if verb in ("enable", "disable"):
            from .scheduler import Scheduler

            sched = Scheduler(context)
            jobs = [j for j in sched.list_jobs()
                    if j.get("name") == RESEARCH_LOOP_JOB]
            if not jobs:
                info = ensure_research_job(context)
                jobs = [j for j in sched.list_jobs()
                        if j.get("name") == RESEARCH_LOOP_JOB]
                if not jobs:
                    return {"ok": False, "error": "job registration failed",
                            **info}
            row = sched.set_enabled(jobs[0]["id"], verb == "enable")
            return {"ok": True, "enabled": verb == "enable", "job": row}
        if verb == "tick":
            loop = ResearchLoop(context,
                                max_topics=max_topics or _MAX_TOPICS_PER_TICK)
            return {"ok": True, **loop.tick()}
        if verb == "run":
            return run_topic(context, topic)
        raise ValueError(f"unknown action: {action!r}")

"""PASS #9: agency rotation / deliberation / follow-through core.

Failing-first tests for verified bugs:
1. A BLOCKED overdue-critical initiative must not starve the whole
   rotation cycle via urgency preemption (planner picked every pick,
   work-key dedupe burns the rest, all other systems starve).
2. The maker-checker must not rubber-stamp: wisdom / impossible /
   digestion / planner / research(no-db) all returned True without an
   independent re-read.
3. Deliberation must record its selections durably (it recorded
   "deferred" but not "selected" — the choice evaporated).
"""
import hashlib
import os
import tempfile
import time

from nomorals.agency.checker import Checker
from nomorals.agency.deliberation import DeliberationLoop
from nomorals.agency.initiatives import Initiative, InitiativeStore
from nomorals.agency.knowledge import KnowledgeStore
from nomorals.agency.systems import ActiveSystems
from nomorals.storage.db import Database


def _systems():
    tmp = tempfile.mkdtemp()
    db = Database(os.path.join(tmp, "t.db"))
    store = InitiativeStore(db)
    return ActiveSystems(db=db, store=store), store, db


def _standing(store, name, now, priority=0.5):
    return store.create(Initiative(
        outcome=f"{name}: standing watch", kind="standing",
        objective=name, evidence="e", next_action="run",
        next_review_at=now + 3600, dedupe_key=f"system:{name}",
        priority=priority))


# ── Bug 1: blocked overdue-critical initiative starves the cycle ──

def test_blocked_overdue_critical_does_not_starve_cycle():
    systems, store, _ = _systems()
    now = time.time()
    # Production state: every system has its standing initiative, so the
    # planner is runnable and urgency preemption actually bites.
    systems.seed_initiatives(now=now)
    store.create(Initiative(
        outcome="leads: hot lead", kind="evidence_opportunity",
        objective="leads", evidence="x", next_action="work",
        next_review_at=now - 7200, dedupe_key="leads:hot:1",
        priority=0.9, blocker="DM send needs owner-approved draft"))
    res = systems.run_cycle(budget_s=600.0, now=now, scan_first=False)
    # The blocked initiative cannot be worked — it must not hold the
    # whole cycle hostage via urgency preemption.
    assert res["systems_run"] > 1, \
        f"cycle starved: only {res['ran']} ran"


def test_unblocked_overdue_critical_still_triggers_urgency():
    systems, store, _ = _systems()
    now = time.time()
    systems.seed_initiatives(now=now)
    store.create(Initiative(
        outcome="leads: hot lead", kind="evidence_opportunity",
        objective="leads", evidence="x", next_action="work",
        next_review_at=now - 7200, dedupe_key="leads:hot:2",
        priority=0.9))
    before = systems.scheduler.preemptions
    systems.run_cycle(budget_s=600.0, now=now, scan_first=False)
    assert systems.scheduler.preemptions > before, \
        "unblocked overdue-critical work must still preempt"


# ── Bug 2: maker-checker rubber stamps ──

def _queue_table(db):
    db.execute("""CREATE TABLE IF NOT EXISTS wisdom_ingest_queue (
        id INTEGER PRIMARY KEY AUTOINCREMENT, url TEXT NOT NULL,
        title TEXT NOT NULL, tradition TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT 'queued', queued_at REAL NOT NULL,
        done_at REAL NOT NULL DEFAULT 0, error TEXT NOT NULL DEFAULT '')""")


def _qslug(url):
    return "q-" + hashlib.sha256(url.encode()).hexdigest()[:10]


def test_checker_wisdom_rejects_unrecorded_queue_ingest():
    _, _, db = _systems()
    _queue_table(db)
    url = "https://example.com/text"
    db.execute(
        "INSERT INTO wisdom_ingest_queue (url, title, queued_at, status)"
        " VALUES (?, ?, ?, 'queued')", (url, "T", time.time()))
    check = Checker().verify(
        "wisdom", {"ingested": [_qslug(url)]}, adapters={}, db=db)
    assert not check.passed, \
        "checker passed a wisdom ingest with no completed queue row"


def test_checker_wisdom_accepts_recorded_queue_ingest():
    _, _, db = _systems()
    _queue_table(db)
    url = "https://example.com/text2"
    db.execute(
        "INSERT INTO wisdom_ingest_queue (url, title, queued_at, status,"
        " done_at) VALUES (?, ?, ?, 'done', ?)",
        (url, "T", time.time(), time.time()))
    check = Checker().verify(
        "wisdom", {"ingested": [_qslug(url)]}, adapters={}, db=db)
    assert check.passed, f"checker rejected a recorded ingest: {check.reason}"


def test_checker_wisdom_rejects_empty_claim():
    _, _, db = _systems()
    check = Checker().verify("wisdom", {"ingested": []}, adapters={}, db=db)
    assert not check.passed


class _ImpossibleAdapter:
    def __init__(self, items):
        self._items = items

    def queue_list(self, status=None):
        return self._items


def test_checker_impossible_rejects_claim_with_no_attempt():
    adapter = _ImpossibleAdapter([{"claim_id": "c1", "attempts": 0,
                                   "status": "open"}])
    check = Checker().verify(
        "impossible", {"claim_id": "c1"}, adapters={"impossible": adapter},
        db=None)
    assert not check.passed, \
        "checker passed an impossible claim with no recorded attempt"


def test_checker_impossible_accepts_attempted_claim():
    adapter = _ImpossibleAdapter([{"claim_id": "c1", "attempts": 2,
                                   "status": "open"}])
    check = Checker().verify(
        "impossible", {"claim_id": "c1"}, adapters={"impossible": adapter},
        db=None)
    assert check.passed, f"checker rejected an attempted claim: {check.reason}"


def test_checker_digestion_rejects_inflated_count():
    _, _, db = _systems()
    ks = KnowledgeStore(db)
    for i in range(5):
        ks.deposit("digestion", f"fact {i}", source="t")
    check = Checker().verify(
        "digestion", {"knowledge_count": 100}, adapters={}, db=db)
    assert not check.passed, \
        "checker passed a digestion claim above the real knowledge count"


def test_checker_digestion_accepts_honest_count():
    _, _, db = _systems()
    ks = KnowledgeStore(db)
    for i in range(5):
        ks.deposit("digestion", f"fact {i}", source="t")
    check = Checker().verify(
        "digestion", {"knowledge_count": 5}, adapters={}, db=db)
    assert check.passed, f"checker rejected an honest count: {check.reason}"


class _PlannerAdapter:
    def __init__(self, pending):
        self._pending = pending

    def pending_graphs(self):
        return self._pending


def test_checker_planner_rejects_still_pending_graph():
    adapter = _PlannerAdapter([{"id": "g1"}])
    check = Checker().verify(
        "planner", {"graph_id": "g1"}, adapters={"planner": adapter},
        db=None)
    assert not check.passed, \
        "checker passed a graph that is still pending (never re-read)"


def test_checker_planner_accepts_completed_graph():
    adapter = _PlannerAdapter([{"id": "g2"}])
    check = Checker().verify(
        "planner", {"graph_id": "g1"}, adapters={"planner": adapter},
        db=None)
    assert check.passed, f"checker rejected a completed graph: {check.reason}"


class _ResearchOrgan:
    def __init__(self, gaps):
        self._gaps = gaps

    def open_gaps(self, limit=20):
        return self._gaps[:limit]


def test_checker_research_fails_closed_without_db_or_organ():
    check = Checker().verify(
        "research", {"gap_id": "gap-1"}, adapters={}, db=None)
    assert not check.passed, \
        "checker passed a research claim with no independent read available"


def test_checker_research_rejects_still_open_gap_via_organ():
    adapter = _ResearchOrgan([{"id": "gap-1"}])
    check = Checker().verify(
        "research", {"gap_id": "gap-1"}, adapters={"research": adapter},
        db=None)
    assert not check.passed, \
        "checker passed a gap that is still open on re-read"


def test_checker_research_accepts_closed_gap_via_organ():
    adapter = _ResearchOrgan([{"id": "gap-9"}])
    check = Checker().verify(
        "research", {"gap_id": "gap-1"}, adapters={"research": adapter},
        db=None)
    assert check.passed, f"checker rejected a closed gap: {check.reason}"


# ── Bug 3: deliberation records selections durably ──

def test_deliberation_records_selected_events():
    _, store, db = _systems()
    now = time.time()
    it = store.create(Initiative(
        outcome="critical overdue", kind="standing", objective="digestion",
        evidence="e", next_action="fix it", priority=1.0,
        next_review_at=now - 7200, dedupe_key="crit:1"))
    chosen = DeliberationLoop(store).choose(now=now, capacity=1)
    assert chosen and chosen[0].id == it.id
    rows = db.query(
        "SELECT kind FROM agency_events WHERE initiative_id = ?",
        (it.id,))
    kinds = {r["kind"] for r in rows}
    assert "selected" in kinds, \
        f"deliberation recorded no 'selected' event (kinds={kinds})"

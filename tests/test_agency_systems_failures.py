"""Tail-1 pins for nomorals/agency/systems.py — 2 converts / 9 exclusions.

Exclusions pin the documented fail-soft behavior AND its silence:
every exclusion test asserts the agency failures ledger records
nothing (clear -> trigger -> failure_count() == 0), mirroring the
#213 legal pins. The ledger is imported read-only here; no record
call is added to the target for exclusions.

Converts (606, 654 — fallback organ ticks): on a failed fallback
tick the trigger now blocks with the exception and the turn is
useless, mirroring the per-trigger failure paths (592, 643) and
_work_trigger's contract. The success paths are pinned unchanged
(noted + useful). The two failure tests FAIL on base (useful=True
+ "noted", empty blocker) — that is the batch1-pristine proof.

No except handlers in this file at all, so the scanner counts 0
sites here (bare except-pass in a pins file would corrupt the
rescan, #169 lesson).
"""
import os
import tempfile
import time
from types import SimpleNamespace

from nomorals.agency import failures as agency_failures
from nomorals.agency.initiatives import Initiative, InitiativeStore
from nomorals.agency.systems import ActiveSystems
from nomorals.storage.db import Database

NOW = time.time()


def _systems(**kw):
    tmp = tempfile.mkdtemp()
    db = Database(os.path.join(tmp, "t.db"))
    store = InitiativeStore(db)
    return ActiveSystems(db=db, store=store, **kw), store, db


def _boom(msg="boom"):
    def _raise(*a, **k):
        raise RuntimeError(msg)
    return _raise


def _silent(trigger):
    """Run trigger() with a cleared ledger; assert nothing recorded."""
    agency_failures.clear_failures()
    result = trigger()
    assert agency_failures.failure_count() == 0, \
        "exclusion must not record to the agency failures ledger"
    return result


def _gap_table(db):
    db.execute("CREATE TABLE knowledge_gaps (id TEXT, status TEXT)")


def _queue_table(db):
    db.execute("CREATE TABLE wisdom_ingest_queue "
               "(id INTEGER PRIMARY KEY, status TEXT)")


def _trigger_obj(store, obj, key):
    return store.create(Initiative(
        outcome=f"{obj}: t", kind="evidence_opportunity", objective=obj,
        evidence="x", next_action="work", next_review_at=NOW - 10,
        dedupe_key=key))


# ── 222: _organ_pending ──────────────────────────────────────────
def test_222_broken_db_returns_zero_and_silent():
    s, _, _ = _systems()
    assert _silent(lambda: s._organ_pending("research")) == 0


def test_222_organ_raising_returns_zero_and_silent():
    s, _, _ = _systems(
        research_organ=SimpleNamespace(open_gaps=_boom()))
    assert _silent(lambda: s._organ_pending("research")) == 0


def test_222_healthy_db_counts_open_gaps():
    s, _, db = _systems()
    _gap_table(db)
    db.execute("INSERT INTO knowledge_gaps VALUES ('g1', 'open')")
    assert s._organ_pending("research") == 1


# ── 242: _adapter_pending ────────────────────────────────────────
def test_242_adapter_raising_returns_zero_and_silent():
    s, _, _ = _systems(
        adapters={"deals": SimpleNamespace(check_alerts=_boom())})
    assert _silent(lambda: s._adapter_pending("deals")) == 0


def test_242_healthy_adapter_counts():
    s, _, _ = _systems(
        adapters={"deals": SimpleNamespace(check_alerts=lambda: ["a", "b"])})
    assert s._adapter_pending("deals") == 2


# ── 270/280: _detector_source ────────────────────────────────────
def test_270_research_broken_db_empty_and_silent():
    s, _, _ = _systems()
    assert _silent(lambda: s._detector_source("research")) == {
        "gaps": [], "watches": []}


def test_270_research_healthy_lists_gaps():
    s, _, db = _systems()
    _gap_table(db)
    db.execute("INSERT INTO knowledge_gaps VALUES ('g9', 'open')")
    assert s._detector_source("research")["gaps"] == [{"id": "g9"}]


def test_280_wisdom_broken_db_zero_and_silent():
    s, _, _ = _systems()
    assert _silent(lambda: s._detector_source("wisdom")) == {"queued": 0}


def test_280_wisdom_healthy_counts_queued():
    s, _, db = _systems()
    _queue_table(db)
    db.execute("INSERT INTO wisdom_ingest_queue (status) VALUES ('queued')")
    assert s._detector_source("wisdom") == {"queued": 1}


# ── 298: _detector_places ────────────────────────────────────────
def test_298_adapter_raising_empty_and_silent():
    s, _, _ = _systems(
        adapters={"weather": SimpleNamespace(places=_boom())})
    assert _silent(lambda: s._detector_places("weather")) == []


def test_298_healthy_adapter_places():
    s, _, _ = _systems(
        adapters={"weather": SimpleNamespace(places=lambda: ["Lagos"])})
    assert s._detector_places("weather") == ["Lagos"]


# ── 381: _tick_planner graphs fallback ───────────────────────────
def test_381_graphs_raising_falls_through_no_execute_and_silent():
    calls = []
    s, _, _ = _systems(adapters={"planner": SimpleNamespace(
        pending_graphs=_boom(),
        execute_graph=lambda g: calls.append(g))})
    res = _silent(lambda: s._tick_planner(NOW, 60.0))
    assert calls == [] and res is not None


def test_381_healthy_graphs_execute():
    calls = []
    s, _, _ = _systems(adapters={"planner": SimpleNamespace(
        pending_graphs=lambda: [{"id": "g1"}],
        execute_graph=lambda g: calls.append(g) or {
            "completed": ["x"], "failed": []})})
    res = s._tick_planner(NOW, 60.0)
    assert calls == [{"id": "g1"}] and res.useful is True


# ── 409: _tick_impossible queue fallback ─────────────────────────
def test_409_queue_raising_falls_to_standing_watch_and_silent():
    s, store, _ = _systems(impossible=SimpleNamespace(queue_list=_boom()))
    it = store.create(Initiative(
        outcome="impossible: standing watch", kind="standing",
        objective="impossible", evidence="e", next_action="run",
        next_review_at=NOW + 3600, dedupe_key="system:impossible"))
    res = _silent(lambda: s._tick_impossible(NOW))
    assert res.work_key == f"impossible:{it.id}"


# ── 349/365: generic organ tick failure = honest empty turn ─────
def test_349_research_tick_raising_empty_turn_and_silent():
    s, _, _ = _systems(research_organ=SimpleNamespace(tick=_boom()))
    before = s.knowledge.count()

    def _run():
        res = s.tick("research", budget_s=60.0, now=NOW)
        assert res.useful is False
        assert s.knowledge.count() == before
        return res

    _silent(_run)


def test_365_wisdom_tick_raising_empty_turn_and_silent():
    s, _, _ = _systems(wisdom_organ=SimpleNamespace(tick=_boom()))
    before = s.knowledge.count()

    def _run():
        res = s.tick("wisdom", budget_s=60.0, now=NOW)
        assert res.useful is False
        assert s.knowledge.count() == before
        return res

    _silent(_run)


def test_349_research_tick_success_still_useful():
    s, _, _ = _systems(research_organ=SimpleNamespace(
        tick=lambda: SimpleNamespace(gaps_resolved=2, findings=1)))
    res = s.tick("research", budget_s=60.0, now=NOW)
    assert res.useful is True


def test_365_wisdom_tick_success_still_useful():
    s, _, _ = _systems(wisdom_organ=SimpleNamespace(
        tick=lambda **k: {"ingested": ["t1"]}))
    res = s.tick("wisdom", budget_s=60.0, now=NOW)
    assert res.useful is True


# ── 606/654: fallback tick converts ─────────────────────────────
def test_606_research_fallback_tick_failure_blocks():
    s, store, _ = _systems(research_organ=SimpleNamespace(tick=_boom()))
    it = _trigger_obj(store, "research", "research:tail1:1")
    res = s._work_trigger(it, NOW)
    back = store.get(it.id)
    assert res.useful is False
    assert "fallback tick failed" in (back.blocker or "")


def test_606_research_fallback_tick_success_noted_useful():
    s, store, _ = _systems(research_organ=SimpleNamespace(
        tick=lambda: SimpleNamespace(gaps_resolved=1, findings=0)))
    it = _trigger_obj(store, "research", "research:tail1:2")
    res = s._work_trigger(it, NOW)
    back = store.get(it.id)
    assert res.useful is True and "noted" in (back.evidence or "")


def test_654_wisdom_fallback_tick_failure_blocks():
    s, store, _ = _systems(wisdom_organ=SimpleNamespace(tick=_boom()))
    it = _trigger_obj(store, "wisdom", "wisdom:tail1:1")
    res = s._work_trigger(it, NOW)
    back = store.get(it.id)
    assert res.useful is False
    assert "fallback tick failed" in (back.blocker or "")


def test_654_wisdom_fallback_tick_success_noted_useful():
    s, store, _ = _systems(wisdom_organ=SimpleNamespace(
        tick=lambda **k: {"ingested": 3}))
    it = _trigger_obj(store, "wisdom", "wisdom:tail1:2")
    res = s._work_trigger(it, NOW)
    back = store.get(it.id)
    assert res.useful is True and "noted" in (back.evidence or "")

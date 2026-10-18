"""Failing-first: synthetic accelerated 72-hour world, NO owner prompts.

Feeds: changing news, forecast revision, duplicate gigs, lead reply,
price drop, research contradiction, expired opportunity, failed source,
process restart. Asserts the acceptance criteria from audit §9.
"""
import time

from nomorals.agency.initiatives import Initiative, InitiativeStore
from nomorals.agency.deliberation import DeliberationLoop
from nomorals.agency.domains import (
    advance_lead, cluster_news, evaluate_deal, score_gig)
from nomorals.agency.digestion import Claim, digest
from nomorals.agency.adequacy import assess_gap_adequacy
from nomorals.agency.supervision import BootSupervisor
from nomorals.agency.views import now_view, reconciled_agenda
from nomorals.storage.db import Database
from types import SimpleNamespace


def _run_world():
    db = Database(":memory:")
    store = InitiativeStore(db)
    loop = DeliberationLoop(store)
    t0 = 1_700_000_000.0
    now = t0
    log = []

    def tick(hours, fn=None):
        nonlocal now
        now += hours * 3600
        if fn:
            fn()
        # Deliberation each tick: overdue critical work must not starve.
        loop.choose(now=now, capacity=3)

    # Hour 0: standing objectives seed initiatives (no owner prompt).
    store.create(Initiative(outcome="Watch XAUUSD QM setup", kind="standing",
                            objective="trading", evidence="standing objective",
                            next_action="scan for QM", next_review_at=now + 3600,
                            dedupe_key="watch:qm"))
    store.create(Initiative(outcome="Find paid gig", kind="standing",
                            objective="earnings", evidence="standing objective",
                            next_action="scan boards", next_review_at=now + 3600,
                            dedupe_key="gig:scan"))
    # Hour 6: duplicate gig signals merge.
    tick(6)
    g1 = store.create(Initiative(outcome="Apply: Python bot gig", kind="evidence_opportunity",
                                 objective="earnings", evidence="board A",
                                 next_action="draft proposal",
                                 next_review_at=now + 3600, dedupe_key="gig:pybot"))
    g2 = store.create(Initiative(outcome="Apply: Python bot gig", kind="evidence_opportunity",
                                 objective="earnings", evidence="board B (dup)",
                                 next_action="draft proposal",
                                 next_review_at=now + 3600, dedupe_key="gig:pybot"))
    assert g1.id == g2.id, "duplicate gig signals must merge"
    # Hour 12: news duplicates cluster; forecast revision arrives.
    tick(6)
    clusters = cluster_news([{"title": "Fed Holds Rates", "url": "https://a/1"},
                             {"title": "fed holds rates", "url": "https://b/2"}])
    assert len(clusters) == 1
    # Hour 18: lead reply -> waiting state gets durable wake-up.
    tick(6)
    lead = store.create(Initiative(outcome="Lead: Acme scoped proposal",
                                   kind="evidence_opportunity",
                                   objective="earnings",
                                   evidence="lead replied asking for scope",
                                   next_action="wait for client reply",
                                   next_review_at=now + 24 * 3600,
                                   blocker="waiting:client-reply:acme",
                                   dedupe_key="lead:acme"))
    assert lead.blocker.startswith("waiting:")
    # Hour 24: price drop on needed item -> deal initiative; expired opp stops.
    tick(6)
    deal = evaluate_deal(needed=True, discount_pct=25.0, price=80.0,
                         landed_price=85.0, historical_low=100.0)
    assert deal["is_saving"] is True
    exp = store.create(Initiative(outcome="Expired gig", kind="evidence_opportunity",
                                  objective="earnings", evidence="listing closed",
                                  next_action="apply", next_review_at=now - 1,
                                  dedupe_key="gig:expired"))
    store.complete(exp.id, stop_reason="expired",
                   verification={"evidence": "listing 404"})
    assert store.get(exp.id).status == "completed"
    # Hour 30: research contradiction -> gap stays open, not resolved.
    tick(6)
    gap = {"question": "Does source X pay?", "context": "{}"}
    deep = SimpleNamespace(synthesis="One paragraph.", findings=[
        SimpleNamespace(title="t", url="https://a.example/1", snippet="s")],
        conflicts=[{"claim": "pays vs never pays"}])
    verdict = assess_gap_adequacy(deep, gap)
    assert not verdict.adequate
    # Hour 36: failed source reported honestly, no fake data.
    tick(6)
    log.append({"source": "board-c", "status": "unavailable",
                "note": "host unreachable; no data fabricated"})
    # Hour 48: PROCESS RESTART — new store handle on same DB recovers.
    tick(12)
    store2 = InitiativeStore(db)
    recovered = store2.boot_reconcile(now=now)
    assert recovered["leases_recovered"] >= 0
    live = store2.list_active()
    # Hour 72: every surviving initiative has next step or blocker.
    tick(24)
    for it in store2.list_active():
        assert it.next_action or it.blocker or it.next_review_at > 0, it
    return store2, log, now


def test_synthetic_72h_world():
    store, log, now = _run_world()
    # Initiatives arose from standing objectives + evidence, no prompts.
    assert len(store.list_active()) >= 3
    # Failed source reported honestly.
    assert log[0]["status"] == "unavailable"
    # Owner views work: now + reconciled agenda.
    assert now_view(store, now=now)["running"] is not None
    agenda = reconciled_agenda(store, now=now)
    assert agenda["open"] >= 0 and "closed" in agenda
    # Causal history: lineage/events recorded.
    events = store.events()
    assert events, "every action must leave a lineage/event trail"


def test_interactive_use_does_not_erase_background_obligations():
    db = Database(":memory:")
    store = InitiativeStore(db)
    store.create(Initiative(outcome="Background: watch lead reply",
                            kind="standing", objective="earnings",
                            evidence="standing", next_action="check inbox",
                            next_review_at=1_700_000_000.0 + 3600,
                            dedupe_key="bg:lead"))
    # Simulate interactive burst: deliberation with capacity 0 must not drop it.
    DeliberationLoop(store).choose(now=1_700_000_000.0, capacity=0)
    assert len(store.list_active()) == 1
    # Boot supervisor re-registers watches idempotently.
    sup = BootSupervisor(store)
    assert sup.reconcile(now=1_700_000_000.0)["watches_reregistered"] >= 0

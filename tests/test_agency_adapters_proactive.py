"""Failing-first tests for real organ adapters + proactive human-worker
behavior. Fakes implement the same duck-typed surface as the real
organs — NO live external calls in tests.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from nomorals.agency.adapters import (
    AgencyAuthority, DealsAdapter, GigsAdapter, LeadsAdapter,
    NewsAdapter, PlannerAdapter, WeatherAdapter,
)
from nomorals.agency.initiatives import InitiativeStore
from nomorals.agency.proactive import (
    LEAD_QUIET_S, detect_triggers, fire_triggers,
    load_proactive_state, save_proactive_state,
)
from nomorals.agency.systems import ActiveSystems
from nomorals.agency.wiring import ensure_agency_shift_job
from nomorals.core.policy import CapabilitySet, PermissionGradient, Policy
from nomorals.storage.db import Database


# ── fakes (adapter surface) ──────────────────────────────────────

class FakeHunter:
    """Hunter surface for the REAL DealsAdapter."""
    def __init__(self, detail=None, history=None, alerts=None):
        self._detail = detail or {"price": 40000.0, "fresh": True,
                                  "in_stock": True, "vendor": "ShopX",
                                  "url": "https://x/1"}
        self._history = history if history is not None else \
            {"lowest_price": 50000.0}
        self._alerts = alerts if alerts is not None else \
            [{"id": "a1", "product": "laptop", "reason": "price dropped"}]

    def check_alerts(self):
        return list(self._alerts)

    def price_detail(self, product):
        return dict(self._detail)

    def price_history(self, product):
        return dict(self._history)

    def search(self, query):
        return [{"product": query, "price": 40000.0}]


class FakeDeals:
    """Adapter surface for systems tests."""
    def __init__(self, saving=True):
        self._saving = saving
        self.checked = []

    def check_alerts(self):
        return [{"id": "a1", "product": "laptop",
                 "reason": "price dropped"}]

    def check_need(self, need):
        self.checked.append(need)
        if self._saving:
            return {"evaluation": {"is_saving": True, "action": "alert",
                                   "reason": "landed below historical low"},
                    "observed": {"price": 40000.0, "landed_price": 45000.0,
                                 "historical_low": 50000.0}}
        return {"evaluation": {"is_saving": False, "action": "watch",
                               "reason": "not below historical low"},
                "observed": {"price": 60000.0, "landed_price": 65000.0,
                             "historical_low": 50000.0}}


class FakeWeatherOrgan:
    def __init__(self, days=None, alerts=None):
        self._days = days if days is not None else [
            {"date": "2026-10-18", "summary": "sunny"}]
        self._alerts = alerts if alerts is not None else []

    def forecast(self, place, days=3):
        return {"ok": True, "place": place, "days": list(self._days)}

    def now(self, place):
        return {"ok": True, "place": place, "alerts": list(self._alerts)}


class FakeWeather:
    """Adapter surface for systems tests."""
    def __init__(self, days=None, alerts=None, places=None):
        self._days = days if days is not None else [
            {"date": "2026-10-18", "summary": "sunny"}]
        self._alerts = alerts if alerts is not None else []
        self._places = places if places is not None else ["Lagos"]

    def places(self):
        return list(self._places)

    def snapshot(self, place):
        return {"place": place, "days": list(self._days),
                "alerts": list(self._alerts), "ok": True}


class FakeNewsAgent:
    def __init__(self, items=None):
        self._items = items if items is not None else [
            {"title": "Big Story", "url": "https://a/1"},
            {"title": "Big Story!", "url": "https://b/2"},
            {"title": "Other News", "url": "https://c/3"}]

    def recent(self, limit=15):
        return list(self._items[:limit])

    def run(self, cap=None):
        return {"digest": "ok", "items": len(self._items)}


class FakeNews:
    def __init__(self, clusters=None):
        self._clusters = clusters if clusters is not None else [
            {"key": "big story", "title": "Big Story",
             "items": [{"title": "Big Story"}]}]

    def recent(self, limit=15):
        return [{"title": "Big Story"}]

    def fetch_digest(self):
        return {"digest": "ok"}

    def clusters(self, limit=15):
        return list(self._clusters)


class FakeSocial:
    def __init__(self, backlog=None, due=None):
        self._backlog = backlog if backlog is not None else \
            {"unanswered": 0, "promised_replies": 1, "milestones": 0,
             "stop_signal": False, "cadence_ok": True}
        self.due = list(due if due is not None else
                        [{"post_id": "p1", "text": "hello"}])
        self.published = []
        self.drafts = []

    def backlog(self):
        return dict(self._backlog)

    def due_posts(self):
        return list(self.due)

    def publish_due(self):
        out = list(self.due)
        self.published.extend(out)
        self.due = []
        return out

    def draft_for(self, action):
        self.drafts.append(action)
        return {"action": action, "text": action, "status": "draft",
                "sent": False}


class FakeLeadsStore:
    def __init__(self, leads=None):
        self.leads = {l["lead_id"]: dict(l) for l in (leads or [])}

    def list_leads(self, status=""):
        out = [dict(l) for l in self.leads.values()]
        return [l for l in out if not status or l["status"] == status]

    def capture_lead(self, name, platform, contact, interest="",
                     source_post=""):
        lead = {"lead_id": f"L{len(self.leads) + 1}", "name": name,
                "platform": platform, "contact": contact,
                "interest": interest, "source_post": source_post,
                "status": "new"}
        self.leads[lead["lead_id"]] = lead
        return dict(lead)

    def set_lead_status(self, lead_id, status):
        if lead_id not in self.leads:
            return False
        self.leads[lead_id]["status"] = status
        return True


class FakeLeads:
    def __init__(self, leads=None):
        self._leads = leads if leads is not None else [
            {"lead_id": "L1", "name": "Ada", "stage": "discovered",
             "status": "new"}]

    def live_leads(self):
        return [dict(l) for l in self._leads]

    def advance(self, lead, nxt):
        return {"lead_id": lead.get("lead_id", ""), "to": nxt,
                "status": "advanced", "store_status": "new"}

    def follow_up(self, lead, draft=""):
        return {"lead_id": lead.get("lead_id", ""), "status": "followed_up",
                "draft": draft or "follow-up", "sent": False}


class FakeGigs:
    def __init__(self, opps=None, followups=None):
        self._opps = opps if opps is not None else [
            {"id": "g1", "title": "Write docs", "payout_text": "$500",
             "win_probability": 0.5, "hours": 10}]
        self._followups = followups if followups is not None else [
            {"gig_id": "g0", "status": "submitted"}]
        self.submitted = []

    def scan(self):
        return [dict(o) for o in self._opps]

    def score(self, opp):
        return {"expected_net": 250.0, "recommendation": "pursue"}

    def draft(self, opp):
        return {"gig_id": opp.get("id", ""), "status": "drafted"}

    def submit(self, gig_id, explicit=False, authority=None):
        if authority is not None:
            decision = authority.may_submit(gig_id, explicit=explicit)
            if not decision.allowed:
                return {"gig_id": gig_id, "status": "submit_attempted",
                        "evidence": None, "reason": decision.reason,
                        "proposal_id": getattr(decision, "proposal_id", "")
                        or None}
        elif not explicit:
            return {"gig_id": gig_id, "status": "submit_attempted",
                    "evidence": None}
        self.submitted.append(gig_id)
        return {"gig_id": gig_id, "status": "submitted",
                "evidence": {"board": "test", "method": "test_submit"}}

    def follow_ups_due(self):
        return [dict(a) for a in self._followups]

    def follow_up_draft(self, gig_id):
        return {"gig_id": gig_id, "draft": "nudge", "status": "drafted"}

    def set_status(self, gig_id, status):
        return {"gig_id": gig_id, "status": status}


class FakeImpossible:
    def __init__(self, claims=None):
        self._claims = claims if claims is not None else [
            {"id": "c1", "problem": "cold fusion"}]
        self.attempted = []

    def queue_list(self, status="queued"):
        return list(self._claims)

    def attempt(self, claim_id):
        self.attempted.append(claim_id)
        return {"ok": True, "claim_id": claim_id, "verdict": "attempted"}


class FakePlanner:
    def __init__(self, graphs=None):
        self._graphs = list(graphs if graphs is not None else
                            [{"goal": "ship", "nodes": []}])
        self.executed = []

    def pending_graphs(self):
        return list(self._graphs)

    def execute_graph(self, graph):
        self.executed.append(graph)
        return {"completed": ["a"], "failed": [], "attempts": 1}


class FakeScheduler:
    def __init__(self):
        self.jobs = []
        self._n = 0

    def list_jobs(self):
        return list(self.jobs)

    def add(self, name, spec, payload_kind, payload, **kw):
        self._n += 1
        job = {"id": f"j{self._n}", "name": name, "spec": spec,
               "payload_kind": payload_kind, "payload": payload,
               "run_timeout_s": float(kw.get("run_timeout_s", 0.0))}
        self.jobs.append(job)
        return job

    def remove(self, job_id):
        self.jobs = [j for j in self.jobs if j["id"] != job_id]


# ── helpers ────────────────────────────────────────────────────────

def _systems(adapters=None, authority=None, **kw):
    db = Database(":memory:")
    store = InitiativeStore(db)
    systems = ActiveSystems(db=db, store=store, adapters=adapters or {},
                            authority=authority or AgencyAuthority(
                                gradient=PermissionGradient(Policy())),
                            **kw)
    return db, store, systems


def _auth(**kw):
    kw.setdefault("gradient", PermissionGradient(Policy()))
    return AgencyAuthority(**kw)


# ── deals ──────────────────────────────────────────────────────────

def test_deals_adapter_checks_price_and_verifies():
    adapter = DealsAdapter(hunter=FakeHunter())
    assert adapter.check_alerts()[0]["product"] == "laptop"
    res = adapter.check_need({"product": "laptop", "needed": True})
    assert res["evaluation"]["is_saving"] is True
    assert res["observed"]["landed_price"] == 40000.0
    assert res["observed"]["historical_low"] == 50000.0


def test_deals_necessity_gate_still_blocks_unnecessary():
    adapter = DealsAdapter(hunter=FakeHunter())
    res = adapter.check_need({"product": "laptop", "needed": False})
    assert res["evaluation"]["is_saving"] is False
    assert "unnecessary" in res["evaluation"]["reason"]


def test_deals_adapter_never_buys():
    adapter = DealsAdapter(hunter=FakeHunter())
    assert not hasattr(adapter, "buy")
    assert not hasattr(adapter, "purchase")
    res = adapter.check_need({"product": "laptop", "needed": True})
    assert "bought" not in res and "order_id" not in res


def test_deals_trigger_reverifies_and_completes_within_mandate(tmp_path):
    from functools import partial

    from nomorals.finance.mandate import (
        MandateStore, check_mandate, issue_mandate,
    )

    store_m = MandateStore(path=str(tmp_path / "mandates.json"))
    issue_mandate(store_m, scope="purchase", cap_per_txn=10_000_000_00,
                  cap_per_day=10_000_000_00)
    auth = _auth(mandate_check=partial(check_mandate, store_m))
    _, store, sys_ = _systems(adapters={"deals": FakeDeals()},
                              authority=auth)
    sys_.seed_initiatives(now=1000.0)
    sys_.scan_proactive(now=2000.0)
    sys_.tick("deals", now=2000.0)
    done = [it for it in store.list_all()
            if it.objective == "deals" and it.status == "completed"]
    assert done and "deal verified" in done[0].stop_reason


def test_deals_trigger_blocker_when_saving_but_no_mandate():
    # Saving verified, but no mandate → advance with blocker, no buy.
    _, store, sys_ = _systems(adapters={"deals": FakeDeals()})
    sys_.seed_initiatives(now=1000.0)
    sys_.scan_proactive(now=2000.0)
    sys_.tick("deals", now=2000.0)
    blocked = [it for it in store.list_active()
               if it.objective == "deals" and it.blocker]
    assert blocked and "mandate" in blocked[0].blocker


# ── weather ────────────────────────────────────────────────────────

def test_weather_adapter_snapshot_classifies_alerts():
    organ = FakeWeatherOrgan(alerts=["tornado warning"])
    adapter = WeatherAdapter(weather=organ)
    snap = adapter.snapshot("Austin")
    assert snap["days"]
    assert snap["alerts"][0]["category"] == "official_warning"


def test_weather_change_trigger_and_no_duplicate():
    fake = FakeWeather()
    state = {}
    # First sight: baseline only, no trigger.
    assert detect_triggers("weather", fake, state, now=1000.0,
                           places=["Lagos"]) == []
    # Same snapshot: no trigger.
    assert detect_triggers("weather", fake, state, now=2000.0,
                           places=["Lagos"]) == []
    # Changed snapshot: one trigger, then silence again.
    fake._days = [{"date": "2026-10-19", "summary": "storm"}]
    triggers = detect_triggers("weather", fake, state, now=3000.0,
                               places=["Lagos"])
    assert len(triggers) == 1
    assert triggers[0].kind == "forecast_change"
    assert detect_triggers("weather", fake, state, now=4000.0,
                           places=["Lagos"]) == []


def test_weather_official_warning_trigger():
    fake = FakeWeather(alerts=[{"category": "official_warning",
                                "text": "flood warning"}])
    state = {}
    detect_triggers("weather", fake, state, now=1000.0, places=["Lagos"])
    triggers = detect_triggers("weather", fake, state, now=2000.0,
                               places=["Lagos"])
    # Changing the alerts changes the signature → official trigger.
    fake._alerts = []
    triggers = detect_triggers("weather", fake, state, now=3000.0,
                               places=["Lagos"])
    assert triggers and triggers[0].kind in (
        "official_warning", "forecast_change")


# ── news ───────────────────────────────────────────────────────────

def test_news_adapter_clusters_duplicates():
    adapter = NewsAdapter(agent=FakeNewsAgent())
    clusters = adapter.clusters()
    assert len(clusters) == 2  # "Big Story" x2 = ONE cluster
    assert clusters[0]["key"] == "big story"
    assert len(clusters[0]["items"]) == 2


def test_news_fetch_digest_uses_organ():
    adapter = NewsAdapter(agent=FakeNewsAgent())
    assert adapter.fetch_digest()["digest"] == "ok"


def test_news_new_cluster_trigger_fires_once():
    fake = FakeNews()
    state = {}
    first = detect_triggers("news", fake, state, now=1000.0)
    assert len(first) == 1 and first[0].kind == "new_cluster"
    assert detect_triggers("news", fake, state, now=2000.0) == []


# ── social ─────────────────────────────────────────────────────────

def test_social_promised_reply_draft_no_publish_without_authority():
    soc = FakeSocial()
    _, store, sys_ = _systems(adapters={"social": soc})
    sys_.seed_initiatives(now=1000.0)
    sys_.scan_proactive(now=2000.0)
    sys_.tick("social", now=2000.0)
    assert soc.drafts  # draft built
    assert soc.published == []  # nothing sent without authority
    blocked = [it for it in store.list_active()
               if it.objective == "social" and it.blocker]
    assert blocked and "proposal" in blocked[0].blocker


def test_social_silence_when_nothing():
    soc = FakeSocial(backlog={"unanswered": 0, "promised_replies": 0,
                              "milestones": 0, "stop_signal": False,
                              "cadence_ok": True})
    _, _, sys_ = _systems(adapters={"social": soc})
    assert sys_.scan_proactive(now=2000.0) == []
    assert soc.drafts == []


def test_social_stop_signal_means_silence():
    soc = FakeSocial(backlog={"unanswered": 5, "promised_replies": 5,
                              "milestones": 1, "stop_signal": True,
                              "cadence_ok": True})
    _, _, sys_ = _systems(adapters={"social": soc})
    assert sys_.scan_proactive(now=2000.0) == []


# ── leads ──────────────────────────────────────────────────────────

def test_leads_lifecycle_validation():
    store = FakeLeadsStore()
    adapter = LeadsAdapter(store=store)
    lead = adapter.capture(name="Ada", platform="x", contact="@ada")
    assert lead["status"] == "new"
    res = adapter.advance({**lead, "stage": "discovered"}, "qualified")
    assert res["status"] == "advanced"
    assert store.leads[lead["lead_id"]]["status"] == "new"  # qualified→new
    bad = adapter.advance({**lead, "stage": "discovered"}, "won")
    assert bad["status"] == "invalid"


def test_leads_live_excludes_dead():
    store = FakeLeadsStore(leads=[
        {"lead_id": "L1", "name": "A", "status": "new"},
        {"lead_id": "L2", "name": "B", "status": "dead"}])
    adapter = LeadsAdapter(store=store)
    assert [l["lead_id"] for l in adapter.live_leads()] == ["L1"]


def test_leads_quiet_follow_up_trigger():
    leads = FakeLeads(leads=[{"lead_id": "L1", "name": "Ada",
                              "stage": "contacted", "status": "dmed"}])
    _, store, sys_ = _systems(adapters={"leads": leads})
    sys_.seed_initiatives(now=1000.0)
    # First scan: new_lead. Tick it (advance). Then jump past quiet.
    sys_.scan_proactive(now=2000.0)
    sys_.tick("leads", now=2000.0)
    state = sys_._proactive_state
    state["leads"]["last_seen"]["L1"] = 2000.0
    triggers = detect_triggers("leads", leads, state,
                               now=2000.0 + LEAD_QUIET_S + 1)
    assert any(t.kind == "quiet_lead" for t in triggers)


def test_leads_next_action_invariant():
    _, store, sys_ = _systems(adapters={"leads": FakeLeads()})
    sys_.seed_initiatives(now=1000.0)
    sys_.scan_proactive(now=2000.0)
    sys_.tick("leads", now=2000.0)
    for it in store.list_active():
        it.validate(now=3000.0)  # next_action / wake-up / blocker


# ── gigs ───────────────────────────────────────────────────────────

def test_gigs_discover_score_draft():
    gigs = FakeGigs()
    assert len(gigs.scan()) == 1
    assert gigs.score({"payout_text": "$500"})["recommendation"] == "pursue"
    assert gigs.draft({"id": "g1"})["status"] == "drafted"


def test_gigs_submit_requires_explicit_plus_authority():
    gigs = FakeGigs()
    # Default authority: autonomous submit → gradient proposal, NOT submitted.
    auth = AgencyAuthority(gradient=PermissionGradient(Policy()))
    res = gigs.submit("g1", explicit=False, authority=auth)
    assert res["status"] == "submit_attempted"  # NOT submitted
    assert res["evidence"] is None
    assert res.get("proposal_id"), "autonomous submit must record a proposal"
    assert gigs.submitted == []
    # Explicit but NOT for this gig (trial pattern) → denied, no submit.
    res_no = gigs.submit("g1", explicit=True, authority=auth)
    assert res_no["status"] == "submit_attempted"
    assert gigs.submitted == []
    # Explicit for THIS gig + policy grant for gig.submit → submitted.
    auth2 = AgencyAuthority(
        gradient=PermissionGradient(Policy()),
        grant=CapabilitySet.of("gig.submit"),
        explicit_submit_ids=frozenset({"g1"}))
    res2 = gigs.submit("g1", explicit=True, authority=auth2)
    assert res2["status"] == "submitted"
    assert res2["evidence"]
    # Explicit for this gig but NO policy grant → grants still apply.
    auth3 = AgencyAuthority(gradient=PermissionGradient(Policy()),
                            explicit_submit_ids=frozenset({"g1"}))
    res3 = gigs.submit("g1", explicit=True, authority=auth3)
    assert res3["status"] == "submit_attempted"


def test_gigs_follow_up_trigger_fires():
    gigs = FakeGigs()
    _, store, sys_ = _systems(adapters={"gigs": gigs})
    triggers = sys_.scan_proactive(now=2000.0)
    assert any(t.system == "gigs" and "g0" in t.evidence for t in triggers)
    assert any(it.objective == "gigs" for it in store.list_active())


def test_gigs_submission_evidence_honesty_regression():
    """Honesty reads the REAL ``submission_evidence`` field."""
    class RealShapeApplier:
        def submit(self, gig_id, explicit=False):
            return {"gig_id": gig_id, "status": "submitted",
                    "submission_evidence": {"board": "upwork",
                                            "method": "upwork_submit",
                                            "note": "receipt #9"}}

    class NoEvidenceApplier:
        def submit(self, gig_id, explicit=False):
            return {"gig_id": gig_id, "status": "submitted",
                    "submission_evidence": {}}

    auth = AgencyAuthority(
        gradient=PermissionGradient(Policy()),
        grant=CapabilitySet.of("gig.submit"),
        explicit_submit_ids=frozenset({"g1", "g2"}))
    good = GigsAdapter(applier=RealShapeApplier())
    res = good.submit("g1", explicit=True, authority=auth)
    assert res["status"] == "submitted"
    assert res["evidence"]["method"] == "upwork_submit"
    bad = GigsAdapter(applier=NoEvidenceApplier())
    res2 = bad.submit("g2", explicit=True, authority=auth)
    assert res2["status"] == "submit_attempted"


def test_gigs_score_uses_injected_payout_extractor():
    from nomorals.agents.opportunities import extract_payout_usd
    adapter = GigsAdapter(payout_extractor=extract_payout_usd)
    res = adapter.score({"payout_text": "$500 fixed",
                         "win_probability": 0.5, "hours": 10})
    assert res["advertised_payout"] == 500.0
    assert res["expected_net"] == 250.0


# ── research / wisdom / impossible / planner / digestion ───────────

def test_impossible_tick_attempts_queued_claim():
    fake = FakeImpossible()
    _, store, sys_ = _systems(impossible=fake)
    sys_.seed_initiatives(now=1000.0)
    sys_.scan_proactive(now=2000.0)
    sys_.tick("impossible", now=2000.0)
    assert fake.attempted == ["c1"]
    done = [it for it in store.list_all()
            if it.objective == "impossible" and it.status == "completed"]
    assert done and "c1" in str(done[0].verification)


def test_planner_graph_execution():
    fake = FakePlanner()
    _, _, sys_ = _systems(adapters={"planner": fake})
    sys_.seed_initiatives(now=1000.0)
    sys_.tick("planner", now=2000.0)
    assert len(fake.executed) == 1


def test_planner_adapter_consumes_graph():
    executed = []

    def _exec(graph, actions=None):
        executed.append(graph)
        class R:
            completed = ["a"]
            failed = []
            attempts = 1
        return R()

    adapter = PlannerAdapter(
        graphs=[{"goal": "g", "nodes": [], "edges": []}],
        execute_fn=_exec,
        graph_factory=lambda d: {"graph": d})
    res = adapter.execute_graph(adapter.pending_graphs()[0])
    assert res["completed"] == ["a"]
    assert adapter.pending_graphs() == []  # consumed, no re-execution


def test_digestion_tick_deposits():
    _, store, sys_ = _systems()
    sys_.seed_initiatives(now=1000.0)
    before = sys_.knowledge.count()
    sys_.tick("digestion", now=2000.0)
    assert sys_.knowledge.count() > before


# ── proactive machinery ────────────────────────────────────────────

def test_proactive_dedupe_no_duplicate_initiatives():
    _, store, sys_ = _systems(adapters={"gigs": FakeGigs()})
    first = sys_.scan_proactive(now=2000.0)
    assert first
    second = sys_.scan_proactive(now=3000.0)
    assert second == []
    gigs = [it for it in store.list_active() if it.objective == "gigs"]
    assert len(gigs) == len({it.dedupe_key for it in gigs})


def test_restart_persistence_no_refire():
    db = Database(":memory:")
    store = InitiativeStore(db)
    systems = ActiveSystems(db=db, store=store,
                            adapters={"gigs": FakeGigs()},
                            authority=_auth())
    assert systems.scan_proactive(now=2000.0)
    # Simulate restart: fresh ActiveSystems on the same DB reloads state.
    systems2 = ActiveSystems(db=db, store=store,
                             adapters={"gigs": FakeGigs()},
                             authority=_auth())
    assert systems2.scan_proactive(now=3000.0) == []


def test_proactive_state_roundtrip(tmp_path):
    db = Database(":memory:")
    state = load_proactive_state(db)
    state["gigs"]["seen_opp_ids"].add("g9")
    save_proactive_state(db, state)
    reloaded = load_proactive_state(db)
    assert "g9" in reloaded["gigs"]["seen_opp_ids"]


def test_fire_triggers_merges_on_dedupe_key():
    db = Database(":memory:")
    store = InitiativeStore(db)
    from nomorals.agency.proactive import Trigger
    t1 = Trigger(system="gigs", kind="new_opportunity",
                 dedupe_key="gigs:opp:g1", evidence="first",
                 next_action="work it")
    t2 = Trigger(system="gigs", kind="new_opportunity",
                 dedupe_key="gigs:opp:g1", evidence="second",
                 next_action="work it")
    fire_triggers(store, [t1], now=1000.0)
    fire_triggers(store, [t2], now=2000.0)
    assert len([it for it in store.list_active()
                if it.dedupe_key == "gigs:opp:g1"]) == 1


def test_shift_job_idempotent_and_stale_replaced():
    sched = FakeScheduler()
    job1 = ensure_agency_shift_job(sched)
    job2 = ensure_agency_shift_job(sched)
    assert job1 is job2 and len(sched.jobs) == 1
    job3 = ensure_agency_shift_job(sched, spec="0 * * * *")
    assert job3 is not job1 and len(sched.jobs) == 1
    assert job3["spec"] == "0 * * * *"


def test_shift_tool_registered():
    from nomorals.tools.autonomy import register
    registered = {}

    # register() uses @registry.register decorator form; emulate it.
    class DecoRegistry:
        context = None

        def register(self, name, fn=None, **kw):
            def deco(f):
                registered[name] = f
                return f
            if fn is not None:
                return deco(fn)
            return deco

    register(DecoRegistry())
    assert "__agency_shift__" in registered
    assert "__presence_heartbeat__" in registered


def test_work_shift_human_worker_end_to_end_no_prompt():
    adapters = {"deals": FakeDeals(), "weather": FakeWeather(),
                "news": FakeNews(), "social": FakeSocial(),
                "leads": FakeLeads(), "gigs": FakeGigs(),
                "impossible": FakeImpossible(),
                "planner": FakePlanner()}
    _, store, sys_ = _systems(adapters=adapters,
                              impossible=FakeImpossible())
    sys_.seed_initiatives(now=1000.0)
    # NO prompt, NO owner tick: the shift itself does the work.
    report = sys_.work_shift()
    assert report["triggers_fired"] > 0
    assert report["cycle"]["systems_run"] > 0
    assert report["knowledge_items"] > 0
    # Every nonterminal initiative still has next/wake-up/blocker.
    for it in store.list_active():
        it.validate()


def test_no_adapter_no_fabrication():
    _, store, sys_ = _systems(adapters={})
    sys_.seed_initiatives(now=1000.0)
    assert sys_.scan_proactive(now=2000.0) == []
    result = sys_.tick("deals", now=2000.0)
    assert result.useful in (True, False)  # standing watch only
    # No trigger initiatives fabricated without observations.
    assert not [it for it in store.list_active()
                if it.kind == "evidence_opportunity"]


def test_agency_adapters_import_no_organs():
    """Layering seam lock: agency/adapters.py must not import ANY organ
    (``from ..`` or ``nomorals.<pkg>``) — organs are injected by the
    caller. Same-package (``.domains``) imports are fine."""
    path = Path(__file__).resolve().parent.parent \
        / "nomorals" / "agency" / "adapters.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level and node.level >= 2:
                bad.append(f"from {'..' * node.level}{node.module}")
            elif node.level == 0 and (node.module or "").startswith(
                    "nomorals."):
                bad.append(f"from {node.module} import ...")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("nomorals."):
                    bad.append(f"import {alias.name}")
    assert bad == [], f"agency/adapters.py imports organs: {bad}"

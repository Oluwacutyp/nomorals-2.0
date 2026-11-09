"""Batch D — autonomy connects & finishes (audit F09–F12).

Each test was failing-first against the real code path on the
pre-batch tree (probe log: batchd-verify/probes-pristine.log, 0/14):

* F09 — the scheduled agency shift read ``context.gateway`` (the
  runtime binds the gateway in ``context.extras``), built its
  opportunity hunter with ``search_fn=None``, and never supplied
  the research/wisdom/impossible organs. One composition root
  (``agency/wiring.compose_agency``) now serves the scheduled shift
  and the interactive boot alike.
* F10 — a gig initiative completed as "scored + drafted" while the
  adapter returned ``status=no_applier`` (no draft exists), and the
  planner reported useful work for deliberation bookkeeping and for
  graph executions whose every node failed.
* F11 — knowledge normalization deleted every non-Latin character,
  so Arabic/Chinese deposits normalized to "" and were refused.
* F12 — freshness renewal, digestion fairness, exploration
  allocation, critical-capacity, future-wait selection, watch
  re-registration truthfulness, and adequacy relevance.
"""
from types import SimpleNamespace
from unittest import mock

import pytest

from nomorals.storage.db import Database


def _db(path=":memory:"):
    db = Database(path)
    db.migrate()
    return db


class FakeRegistry:
    def __init__(self, context):
        self.context = context
        self.fns = {}

    def register(self, name, fn=None, **kw):
        if fn is None:
            def deco(f):
                self.fns[name] = f
                return f
            return deco
        self.fns[name] = fn
        return fn


class _Sentinel(Exception):
    pass


class FakeGateway:
    def __init__(self):
        self.sent = []

    def send(self, platform, contact, body):
        self.sent.append((platform, contact, body))
        return SimpleNamespace(ok=True)


class FakeTools:
    """Transport seam: the context's web_search tool."""

    def __init__(self):
        self.calls = []

    def call(self, name, **kw):
        self.calls.append((name, kw))
        return SimpleNamespace(
            ok=True, error=None,
            unwrap=lambda: {"results": [{
                "url": "https://gigs.example/arabic-tutor",
                "title": "Arabic tutor gig $200",
                "snippet": "pay $200 for tutoring"}]})


def _ctx(tmp_path, gateway=None, tools=None):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    settings = SimpleNamespace(
        workspace_dir=str(ws), resolve=lambda rel: ws / rel)
    return SimpleNamespace(
        settings=settings,
        extras={"gateway": gateway} if gateway is not None else {},
        tools=tools if tools is not None else FakeTools(),
        db=_db(str(ws / "db.sqlite")))


def _capture_shift_kwargs(context):
    import nomorals.agency.adapters as adapters_mod
    from nomorals.tools import autonomy as ta

    captured = {}

    def capturing_build(**kw):
        captured.update(kw)
        raise _Sentinel()

    reg = FakeRegistry(context)
    ta.register(reg)
    with mock.patch.object(adapters_mod, "build_adapters",
                           capturing_build), pytest.raises(_Sentinel):
        reg.fns["__agency_shift__"]()
    return captured


# ── F09: scheduled composition root ────────────────────────────────

def test_f09_gateway_resolved_from_extras_into_shift_adapters(tmp_path):
    gw = FakeGateway()
    captured = _capture_shift_kwargs(_ctx(tmp_path, gateway=gw))
    assert captured.get("gateway") is gw
    assert captured.get("dm_sender") is not None


def test_f09_scheduled_hunter_carries_live_search(tmp_path):
    captured = _capture_shift_kwargs(
        _ctx(tmp_path, gateway=FakeGateway()))
    hunter = captured.get("opp_hunter")
    assert hunter is not None
    assert hunter.search_fn is not None


def test_f09_shift_systems_receive_real_organs(tmp_path):
    import nomorals.agency.systems as systems_mod
    from nomorals.tools import autonomy as ta

    seen = {}

    class FakeSystems:
        def __init__(self, **kw):
            seen.update(kw)

        def work_shift(self):
            return {}

    reg = FakeRegistry(_ctx(tmp_path, gateway=FakeGateway()))
    ta.register(reg)
    with mock.patch.object(systems_mod, "ActiveSystems", FakeSystems), \
            mock.patch("nomorals.agency.wiring.run_scheduled_expansion",
                       lambda *a, **k: {}):
        reg.fns["__agency_shift__"]()
    assert seen.get("research_organ") is not None
    assert seen.get("wisdom_organ") is not None
    assert seen.get("impossible") is not None


def test_f09_live_search_result_reaches_gig_pipeline(tmp_path,
                                                      monkeypatch):
    from nomorals.agency.wiring import compose_agency

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    tools = FakeTools()
    comp = compose_agency(_ctx(tmp_path, gateway=FakeGateway(),
                               tools=tools))
    assert comp["readiness"]["search"].startswith("ready")
    assert comp["readiness"]["gateway"] == "ready"
    opps = comp["adapters"]["gigs"].scan()
    urls = [o.get("url", "") for o in opps]
    assert "https://gigs.example/arabic-tutor" in urls, urls
    assert tools.calls, "the hunter never exercised the search seam"


def test_f09_lead_reaches_fake_gateway_via_composed_adapter(tmp_path,
                                                            monkeypatch):
    from nomorals.agency.wiring import compose_agency

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    gw = FakeGateway()
    comp = compose_agency(_ctx(tmp_path, gateway=gw))
    lead = {"lead_id": "lead-1", "name": "Ada", "platform": "telegram",
            "contact": "@ada", "status": "new"}
    out = comp["adapters"]["leads"].send_dm(
        lead, draft="hi Ada — following up", authority=None)
    assert out["sent"] is True
    assert gw.sent == [("telegram", "@ada", "hi Ada — following up")]


def test_f09_registered_shift_runs_end_to_end(tmp_path, monkeypatch):
    from nomorals.tools import autonomy as ta

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    tools = FakeTools()
    reg = FakeRegistry(_ctx(tmp_path, gateway=FakeGateway(),
                            tools=tools))
    ta.register(reg)
    with mock.patch("nomorals.agency.wiring.run_scheduled_expansion",
                    lambda *a, **k: {"stubbed": "transport seam"}), \
            mock.patch("nomorals.agents.weather.fetch_forecast",
                       lambda *a, **k: None), \
            mock.patch("nomorals.agents.weather.geocode",
                       lambda *a, **k: None):
        out = reg.fns["__agency_shift__"]()
    assert out["ok"] is True
    assert out["composition"]["gateway"] == "ready"
    assert out["composition"]["search"].startswith("ready")
    # The gig pipeline actually scanned through the live search seam
    # during the shift (not merely "adapter constructed").
    assert any(name == "web_search" for name, _ in tools.calls)


# ── F10: evidence-bearing lifecycle ────────────────────────────────

def _gig_systems(applier):
    from nomorals.agency.adapters import GigsAdapter
    from nomorals.agency.systems import ActiveSystems

    opp = {"id": "g1", "title": "Tutor gig", "payout_usd": 100.0,
           "win_probability": 0.5, "costs": 0, "hours": 1,
           "scam_indicators": []}
    adapter = GigsAdapter(hunter=SimpleNamespace(scan=lambda: [opp]),
                          applier=applier, opp_factory=lambda d: d)
    systems = ActiveSystems(db=_db(), adapters={"gigs": adapter})
    return systems, opp


def _gig_trigger(systems, opp):
    from nomorals.agency.initiatives import Initiative

    it = systems.store.create(Initiative(
        outcome="gig opp", kind="evidence_opportunity", objective="gigs",
        evidence="opp found", next_action="score and draft",
        next_review_at=0, dedupe_key="gig:opp:g1"))
    systems._trigger_payloads["gig:opp:g1"] = {"opp": opp}
    return it


def test_f10_no_applier_draft_does_not_complete():
    systems, opp = _gig_systems(applier=None)
    it = _gig_trigger(systems, opp)
    res = systems.tick("gigs", now=2000.0)
    after = systems.store.get(it.id)
    assert after.status != "completed"
    assert "draft not created" in (after.blocker or "")
    assert res.useful is False


def test_f10_checker_rejects_no_artifact_draft_claim():
    from nomorals.agency.adapters import GigsAdapter
    from nomorals.agency.checker import Checker

    _, opp = _gig_systems(applier=None)
    adapter = GigsAdapter(hunter=SimpleNamespace(scan=lambda: [opp]),
                          applier=None)
    claimed = {"opp": opp,
               "score": adapter.score(opp),
               "draft": {"gig_id": "g1", "status": "no_applier"}}
    check = Checker().verify("gigs", claimed,
                             adapters={"gigs": adapter}, db=None)
    assert not check.passed
    assert "no draft artifact" in check.reason


def test_f10_real_draft_still_completes():
    class FakeApplier:
        def draft(self, opp):
            return SimpleNamespace(to_dict=lambda: {
                "gig_id": opp["id"] if isinstance(opp, dict) else opp.id,
                "status": "drafted", "draft_text": "application body"})

    systems, opp = _gig_systems(applier=FakeApplier())
    it = _gig_trigger(systems, opp)
    res = systems.tick("gigs", now=2000.0)
    after = systems.store.get(it.id)
    assert after.status == "completed", after.blocker
    assert res.useful is True


def test_f10_planner_deliberation_only_is_not_useful():
    from nomorals.agency.initiatives import Initiative
    from nomorals.agency.systems import ActiveSystems

    systems = ActiveSystems(db=_db(), adapters={})
    systems.store.create(Initiative(
        outcome="some work", kind="standing", objective="research",
        evidence="e", next_action="do it", next_review_at=0))
    res = systems._tick_planner(2000.0, 60.0)
    assert res.useful is False


def test_f10_planner_all_failed_graph_is_not_useful():
    from nomorals.agency.systems import ActiveSystems

    adapter = SimpleNamespace(
        pending_graphs=lambda: [{"id": "g1"}],
        execute_graph=lambda g: {"completed": [], "failed": ["boom"],
                                 "attempts": 1})
    systems = ActiveSystems(db=_db(), adapters={"planner": adapter})
    res = systems._tick_planner(2000.0, 60.0)
    assert res.useful is False


def test_f10_planner_completed_graph_is_useful():
    from nomorals.agency.systems import ActiveSystems

    adapter = SimpleNamespace(
        pending_graphs=lambda: [{"id": "g1"}],
        execute_graph=lambda g: {"completed": ["n1"], "failed": [],
                                 "attempts": 1})
    systems = ActiveSystems(db=_db(), adapters={"planner": adapter})
    res = systems._tick_planner(2000.0, 60.0)
    assert res.useful is True


# ── F11: Unicode knowledge ingestion ───────────────────────────────

_SAMPLES = {
    "arabic": "المعرفة العربية تُحفظ كما هي تماما",
    "chinese": "知识摄取必须保留中文原文内容",
    "devanagari": "देवनागरी पाठ ज्ञान में संग्रहीत होता है",
    "cyrillic": "Кириллический текст сохраняется полностью",
    "diacritics": "Café résumé naïve — les accents comptent",
    "mixed": "Mixed English و عربي و 中文 together",
    "emoji": "Launch day 🚀 was a great success",
}


def test_f11_non_latin_deposit_retrieval_dedupe_expiry():
    from nomorals.agency.knowledge import KnowledgeStore

    ks = KnowledgeStore(_db())
    for name, text in _SAMPLES.items():
        assert ks.deposit("probe", text, source="probe",
                          now=100.0), f"{name} deposit rejected"
        hits = ks.search(text, now=101.0)
        assert hits, f"{name} not retrievable"
        assert hits[0]["text"] == text, f"{name} text mangled"
        # Dedupe: the same fact again is corroboration, not a copy.
        assert ks.deposit("probe", text, source="probe",
                          now=102.0) is False
    assert ks.count() == len(_SAMPLES)
    # Expiry: a TTL'd entry is served before, never after, its expiry.
    assert ks.deposit("probe", "ephemeral multilingual معلومة عابرة",
                      ttl_s=10, now=100.0)
    assert ks.search("ephemeral multilingual", now=105.0)
    assert not ks.search("ephemeral multilingual", now=120.0)
    assert ks.prune_expired(now=120.0) >= 1


# ── F12: freshness / fairness / registration / adequacy ────────────

def test_f12_reobservation_renews_expired_knowledge():
    from nomorals.agency.knowledge import KnowledgeStore

    ks = KnowledgeStore(_db())
    assert ks.deposit("probe", "freshness renewal fact alpha",
                      ttl_s=10, now=100.0)
    assert ks.search("freshness renewal", now=105.0)
    assert not ks.search("freshness renewal", now=120.0)
    # Re-observed at t=120 (duplicate sighting, no fresh TTL given):
    # the entry's own lifetime re-applies from the new sighting.
    assert ks.deposit("probe", "freshness renewal fact alpha",
                      now=120.0) is False
    assert ks.search("freshness renewal", now=125.0)
    assert not ks.search("freshness renewal", now=135.0)


def test_f12_digestion_fairness_newer_evidence_processed():
    from nomorals.agency.initiatives import Initiative
    from nomorals.agency.systems import ActiveSystems

    db = _db()
    systems = ActiveSystems(db=db)
    systems.store.create(Initiative(
        outcome="a", kind="evidence_opportunity", objective="research",
        evidence="alpha fact one", next_action="review",
        next_review_at=0, created_at=100.0))
    systems.store.create(Initiative(
        outcome="b", kind="evidence_opportunity", objective="research",
        evidence="beta fact two", next_action="review",
        next_review_at=0, created_at=200.0))
    first = systems.tick("digestion", now=2000.0)
    assert first.useful is True
    second = systems.tick("digestion", now=3000.0)
    assert second.useful is True
    texts = [r["text"] for r in db.query(
        "SELECT text FROM agency_knowledge WHERE system='digestion'")]
    assert "beta fact two" in texts


def test_f12_exploration_gets_a_slot():
    from nomorals.agency.deliberation import DeliberationLoop
    from nomorals.agency.initiatives import Initiative, InitiativeStore

    store = InitiativeStore(_db())
    store.create(Initiative(
        outcome="explore x", kind="exploration", objective="research",
        evidence="e", next_action="read", next_review_at=0))
    chosen = DeliberationLoop(store).choose(now=2000.0, capacity=3)
    assert len(chosen) == 1


def test_f12_exploration_share_still_bounded_in_large_batch():
    from nomorals.agency.deliberation import DeliberationLoop
    from nomorals.agency.initiatives import Initiative, InitiativeStore

    store = InitiativeStore(_db())
    for i in range(5):
        store.create(Initiative(
            outcome=f"work {i}", kind="explicit_goal",
            objective="research", evidence="e", next_action="do",
            next_review_at=0, dedupe_key=f"w:{i}", priority=0.9))
    for i in range(5):
        store.create(Initiative(
            outcome=f"explore {i}", kind="exploration",
            objective="research", evidence="e", next_action="read",
            next_review_at=0, dedupe_key=f"x:{i}", priority=0.1))
    chosen = DeliberationLoop(store).choose(now=2000.0, capacity=10)
    explore = [c for c in chosen if c.kind == "exploration"]
    # 5 exploit + the reserved/bounded 2 exploration; the remaining
    # capacity stays unfilled rather than breaking the share bound.
    assert len(explore) == 2  # 20% of 10 — the share is a bound, too
    assert len(chosen) == 7


def test_f12_critical_selection_respects_capacity():
    from nomorals.agency.deliberation import DeliberationLoop
    from nomorals.agency.initiatives import Initiative, InitiativeStore

    store = InitiativeStore(_db())
    for i in range(5):
        store.create(Initiative(
            outcome=f"crit {i}", kind="standing", objective="research",
            evidence="e", next_action="fix", priority=1.0,
            next_review_at=100.0, dedupe_key=f"crit:{i}"))
    chosen = DeliberationLoop(store).choose(now=10000.0, capacity=1)
    assert len(chosen) == 1


def test_f12_future_wait_not_selected_early():
    from nomorals.agency.deliberation import DeliberationLoop
    from nomorals.agency.initiatives import Initiative, InitiativeStore

    store = InitiativeStore(_db())
    store.create(Initiative(
        outcome="wait for later", kind="standing", objective="research",
        evidence="e", next_action="", next_review_at=9999.0))
    chosen = DeliberationLoop(store).choose(now=2000.0, capacity=3)
    assert chosen == []


def test_f12_watch_reregistration_reaches_facade():
    from nomorals.agency.initiatives import InitiativeStore
    from nomorals.agency.supervision import BootSupervisor

    calls = []

    class Facade:
        def register_watch(self, watch):
            calls.append(watch)
            return True

    sup = BootSupervisor(InitiativeStore(_db()),
                         scheduler_facade=Facade())
    assert sup.reregister_watches([{"id": "w1", "topic": "gold"}]) == 1
    assert len(calls) == 1
    # Idempotent: a second boot does not re-register with the facade.
    assert sup.reregister_watches([{"id": "w1", "topic": "gold"}]) == 0
    assert len(calls) == 1


def test_f12_watch_facade_rejection_is_not_counted():
    from nomorals.agency.initiatives import InitiativeStore
    from nomorals.agency.supervision import BootSupervisor

    class Rejecting:
        def register_watch(self, watch):
            return False

    sup = BootSupervisor(InitiativeStore(_db()),
                         scheduler_facade=Rejecting())
    assert sup.reregister_watches([{"id": "w1"}]) == 0
    # Not marked registered: a later boot retries the registration.
    assert sup.reregister_watches([{"id": "w1"}]) == 0
    assert "w1" not in sup._registered_watches


def test_f12_adequacy_rejects_irrelevant_two_domain_answer():
    from nomorals.agency.adequacy import assess_gap_adequacy

    gap = {"question": "How does the MT5 tick subscription API "
                       "authenticate?",
           "context": "{}"}
    findings = [
        SimpleNamespace(title="Soup", url="https://a.example/soup",
                        snippet="tomato soup recipe with basil and cream"),
        SimpleNamespace(title="Pasta", url="https://b.example/pasta",
                        snippet="pasta cooking times and sauce pairings"),
    ]
    deep = SimpleNamespace(
        synthesis="Tomato soup needs basil; pasta pairs with rich "
                  "sauces.",
        findings=findings, conflicts=[])
    verdict = assess_gap_adequacy(deep, gap)
    assert not verdict.adequate
    assert any("relevance" in r or "does not address" in r
               for r in verdict.reasons)


def test_f12_adequacy_subquestion_needs_real_coverage():
    from nomorals.agency.adequacy import assess_gap_adequacy

    # One incidental term ("data") must not "cover" a four-term
    # subquestion.
    gap = {"question": "tick data feed?",
           "context": '{"subquestions": '
                      '["tick data latency budget"]}'}
    findings = [
        SimpleNamespace(title="t", url="https://a.example/x",
                        snippet="data"),
        SimpleNamespace(title="t", url="https://b.example/y",
                        snippet="data"),
    ]
    deep = SimpleNamespace(synthesis="data data data", findings=findings,
                           conflicts=[])
    verdict = assess_gap_adequacy(deep, gap)
    assert not verdict.adequate
    assert "tick data latency budget" in verdict.unresolved_subquestions

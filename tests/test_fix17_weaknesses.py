"""Fix-17: failing-first tests for the 17 weaknesses from the three
recent Devon builds (infrastructure 0acd4b9f, mining/expansion 98c7cede,
wiring 8e3a46f1). Each test pins one weakness; all must pass after the fix.
No live external calls except the explicitly skip-on-failure live checks.
"""
from __future__ import annotations

import time
from functools import partial
from types import SimpleNamespace

import pytest

from nomorals.agency.adapters import (
    AgencyAuthority, DealsAdapter, LeadsAdapter, SocialAdapter,
    WeatherAdapter, build_adapters,
)
from nomorals.agency.initiatives import Initiative, InitiativeStore
from nomorals.agency.systems import ActiveSystems
from nomorals.core.policy import (
    CapabilitySet, PermissionGradient, Policy,
)
from nomorals.storage.db import Database


# ── shared fakes ───────────────────────────────────────────────────

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


class FakeHunter:
    def __init__(self):
        self._detail = {"price": 40000.0, "fresh": True, "in_stock": True,
                        "vendor": "ShopX", "url": "https://x/1"}
        self._history = {"lowest_price": 50000.0}

    def check_alerts(self):
        return [{"id": "a1", "product": "laptop", "reason": "price dropped"}]

    def price_detail(self, product):
        return dict(self._detail)

    def price_history(self, product):
        return dict(self._history)


class FakeWeatherOrgan:
    def __init__(self):
        self.default_place = "New York"

    def forecast(self, place, days=3):
        return {"ok": True, "place": place, "days": [{"date": "2026-10-18"}]}

    def now(self, place):
        return {"ok": True, "place": place, "alerts": []}


class FakeResearchOrgan:
    """Per-trigger surface: investigate_gap / run_watch + state reads."""

    def __init__(self):
        self.investigated = []
        self.watches_run = []
        self.open = [{"id": "g1", "question": "what is X?"}]
        self.due = [{"id": "w1", "topic": "t"}]

    def open_gaps(self, limit=50):
        return list(self.open)

    def due_watches(self, now=None):
        return list(self.due)

    def investigate_gap(self, gap_id):
        self.investigated.append(gap_id)
        self.open = [g for g in self.open if str(g["id"]) != str(gap_id)]
        return {"gap_id": gap_id, "status": "resolved",
                "synthesis": "answer"}

    def run_watch(self, watch_id):
        self.watches_run.append(watch_id)
        self.due = [w for w in self.due if str(w["id"]) != str(watch_id)]
        return {"watch_id": watch_id, "findings": 2}

    def tick(self):
        return SimpleNamespace(gaps_resolved=0, findings=0)


class FakeWisdomOrgan:
    def __init__(self):
        self.ingested = []
        self.pending = [{"slug": "meditations"}]

    def _pending_manifest(self):
        return list(self.pending)

    def ingest_next(self):
        if not self.pending:
            return {"ingested": []}
        item = self.pending.pop(0)
        self.ingested.append(item["slug"])
        return {"ingested": [item["slug"]]}

    def tick(self, budget_seconds=60.0):
        return {"ingested": []}


class FakeLeadsStore:
    def __init__(self, leads=None):
        self.leads = {l["lead_id"]: dict(l) for l in (leads or [])}

    def list_leads(self, status=""):
        return [dict(l) for l in self.leads.values()
                if not status or l["status"] == status]

    def set_lead_status(self, lead_id, status):
        if lead_id not in self.leads:
            return False
        self.leads[lead_id]["status"] = status
        return True


def _mandate_auth(tmp_path, *, allow_purchase=False):
    from nomorals.finance.mandate import (
        MandateStore, check_mandate, issue_mandate,
    )

    store_m = MandateStore(path=str(tmp_path / "mandates.json"))
    issue_mandate(store_m, scope="purchase", cap_per_txn=10_000_000_00,
                  cap_per_day=10_000_000_00)
    policy = Policy()
    if allow_purchase:
        # Standing authority = allow-rule (level) + grant (conferral).
        policy.allow("commerce.purchase", note="test purchase allow")
        policy.grant("commerce.purchase")
    return AgencyAuthority(
        gradient=PermissionGradient(policy),
        mandate_check=partial(check_mandate, store_m))


# ── 1. purchase executor ───────────────────────────────────────────

def test_1_purchase_executor_completes_when_authorized(tmp_path):
    receipts = []

    def executor(need, observed):
        receipts.append((need, observed))
        return {"purchased": True,
                "receipt": {"order_id": "ORD-1", "amount": observed["landed_price"],
                            "vendor": observed.get("vendor", "")}}

    adapter = DealsAdapter(hunter=FakeHunter(), purchase_executor=executor)
    auth = _mandate_auth(tmp_path, allow_purchase=True)
    res = adapter.execute_purchase(
        {"product": "laptop", "needed": True},
        {"price": 40000.0, "landed_price": 45000.0,
         "historical_low": 50000.0, "vendor": "ShopX"},
        authority=auth)
    assert res["purchased"] is True
    assert res["receipt"]["order_id"] == "ORD-1"
    assert receipts, "executor must actually run"


def test_1_purchase_blocked_without_mandate_or_policy(tmp_path):
    def executor(need, observed):
        raise AssertionError("executor must NOT run without authority")

    adapter = DealsAdapter(hunter=FakeHunter(), purchase_executor=executor)
    # No mandate, no policy allow → blocked, no purchase.
    auth = AgencyAuthority(gradient=PermissionGradient(Policy()))
    res = adapter.execute_purchase(
        {"product": "laptop", "needed": True},
        {"price": 40000.0, "landed_price": 45000.0},
        authority=auth)
    assert res["purchased"] is False
    assert res.get("reason") or res.get("blocker")


def test_1_deals_trigger_completes_purchase_end_to_end(tmp_path):
    def executor(need, observed):
        return {"purchased": True,
                "receipt": {"order_id": "ORD-9", "amount": 45000.0}}

    adapter = DealsAdapter(hunter=FakeHunter(), purchase_executor=executor)
    auth = _mandate_auth(tmp_path, allow_purchase=True)
    _, store, sys_ = _systems(adapters={"deals": adapter}, authority=auth)
    sys_.seed_initiatives(now=1000.0)
    sys_.scan_proactive(now=2000.0)
    sys_.tick("deals", now=2000.0)
    done = [it for it in store.list_all()
            if it.objective == "deals" and it.status == "completed"]
    assert done, "deal trigger must COMPLETE the purchase, not just report"
    assert "ORD-9" in str(done[0].verification)


def test_1_purchase_requires_receipt_evidence(tmp_path):
    def executor(need, observed):
        return {"purchased": True}  # no receipt → NOT a purchase

    adapter = DealsAdapter(hunter=FakeHunter(), purchase_executor=executor)
    auth = _mandate_auth(tmp_path, allow_purchase=True)
    res = adapter.execute_purchase(
        {"product": "laptop", "needed": True},
        {"price": 40000.0, "landed_price": 45000.0},
        authority=auth)
    assert res["purchased"] is False
    assert "receipt" in (res.get("reason", "") + res.get("blocker", "")).lower() \
        or "evidence" in (res.get("reason", "") + res.get("blocker", "")).lower()


# ── 2. social default DM/triage source ─────────────────────────────

class FakeRelationships:
    """Real relationships-tracker surface: last_inbound vs last_outbound."""

    def __init__(self):
        self.rels = [
            SimpleNamespace(person_id="p1", last_inbound=2000.0,
                            last_outbound=1000.0, closeness=0.8),
            SimpleNamespace(person_id="p2", last_inbound=1000.0,
                            last_outbound=2000.0, closeness=0.5),
        ]

    def all(self):
        return list(self.rels)


class FakeDraftQueue:
    def pending(self):
        return [{"draft_id": "d1"}, {"draft_id": "d2"}]


def test_2_social_backlog_defaults_to_real_sources():
    adapter = SocialAdapter(manager=None, inbox=None,
                            relationships=FakeRelationships(),
                            draft_queue=FakeDraftQueue())
    backlog = adapter.backlog()
    assert backlog["unanswered"] >= 1, \
        "p1 inbound > outbound is a real unanswered DM"
    assert backlog["promised_replies"] >= 1, \
        "pending drafts are real promised replies"


def test_2_build_adapters_wires_default_inbox():
    adapters = build_adapters(relationships=FakeRelationships(),
                              draft_queue=FakeDraftQueue())
    backlog = adapters["social"].backlog()
    assert backlog["unanswered"] >= 1
    assert backlog["promised_replies"] >= 1


# ── 3. weather places from settings ────────────────────────────────

def test_3_weather_places_derived_from_settings():
    settings = SimpleNamespace(
        location=SimpleNamespace(home_place="Lagos, Nigeria",
                                 work_place="Abuja",
                                 places=[]))
    adapter = WeatherAdapter(weather=FakeWeatherOrgan(), settings=settings)
    assert adapter.places() == ["Lagos, Nigeria", "Abuja"]


def test_3_weather_explicit_places_win_over_settings():
    settings = SimpleNamespace(
        location=SimpleNamespace(home_place="Lagos, Nigeria",
                                 work_place="Abuja", places=[]))
    adapter = WeatherAdapter(weather=FakeWeatherOrgan(),
                             places=["Kano"], settings=settings)
    assert adapter.places() == ["Kano"]


def test_3_weather_falls_back_to_organ_default_place():
    adapter = WeatherAdapter(weather=FakeWeatherOrgan())
    assert adapter.places() == ["New York"]


def test_3_settings_carry_location_defaults():
    from nomorals.core.config import Settings

    s = Settings()
    assert s.location.home_place == "Lagos, Nigeria"
    assert s.location.work_place == ""


# ── 4. policy allow-rules persist ──────────────────────────────────

def test_4_policy_allow_rules_persist_across_instances(tmp_path):
    from nomorals.core.policy import Policy

    db = Database(":memory:")
    policy = Policy()
    policy.allow("social.post", note="owner allows scheduled posts")
    policy.grant("social.post")
    policy.save_rules(db)
    fresh = Policy()
    fresh.load_rules(db)
    rules = {(r["capability"], r["effect"]) for r in fresh.list_rules()}
    assert ("social.post", "allow") in rules
    assert "social.post" in fresh.list_grants()


def test_4_scheduled_shift_uses_persisted_rules(tmp_path):
    from nomorals.core.policy import Policy
    from nomorals.tools.autonomy import _gradient_for_shift

    db = Database(":memory:")
    policy = Policy()
    policy.allow("social.post", note="owner allows scheduled posts")
    policy.grant("social.post")
    policy.save_rules(db)
    gradient = _gradient_for_shift(db)
    decision = gradient.check("social.post", actor="agency", autonomous=True)
    assert decision.allowed, \
        "scheduled shift must see the same allow-rule as interactive use"


# ── 5. per-trigger research + wisdom handlers ──────────────────────

def _trigger_initiative(store, system, dedupe_key, payload):
    it = store.create(Initiative(
        outcome=f"{system} trigger", kind="evidence_opportunity",
        objective=system, evidence="trigger", next_action="work it",
        next_review_at=1000.0, dedupe_key=dedupe_key, priority=0.7))
    return it


def test_5_research_gap_trigger_uses_per_trigger_handler():
    organ = FakeResearchOrgan()
    _, store, sys_ = _systems(research_organ=organ)
    sys_.seed_initiatives(now=1000.0)
    it = _trigger_initiative(store, "research", "research:gap:g1",
                             {"gap": {"id": "g1"}})
    sys_._trigger_payloads[it.dedupe_key] = {"gap": {"id": "g1"}}
    sys_.tick("research", now=2000.0)
    assert organ.investigated == ["g1"], \
        "research gap trigger must call investigate_gap, not a generic tick"
    done = [i for i in store.list_all()
            if i.dedupe_key == "research:gap:g1" and i.status == "completed"]
    assert done and "g1" in str(done[0].verification)


def test_5_research_watch_trigger_uses_per_trigger_handler():
    organ = FakeResearchOrgan()
    _, store, sys_ = _systems(research_organ=organ)
    sys_.seed_initiatives(now=1000.0)
    it = _trigger_initiative(store, "research", "research:watch:w1",
                             {"watch": {"id": "w1"}})
    sys_._trigger_payloads[it.dedupe_key] = {"watch": {"id": "w1"}}
    sys_.tick("research", now=2000.0)
    assert organ.watches_run == ["w1"]


def test_5_wisdom_ingest_trigger_uses_per_trigger_handler():
    organ = FakeWisdomOrgan()
    _, store, sys_ = _systems(wisdom_organ=organ)
    sys_.seed_initiatives(now=1000.0)
    it = _trigger_initiative(store, "wisdom", "wisdom:ingest:1",
                             {"queued": 1})
    sys_._trigger_payloads[it.dedupe_key] = {"queued": 1}
    sys_.tick("wisdom", now=2000.0)
    assert organ.ingested == ["meditations"], \
        "wisdom ingest trigger must ingest the queued item, not just note it"
    done = [i for i in store.list_all()
            if i.dedupe_key == "wisdom:ingest:1" and i.status == "completed"]
    assert done and "meditations" in str(done[0].verification)


# ── 6. leads DM-send executor ──────────────────────────────────────

def test_6_leads_send_dm_executor():
    sent = []

    def dm_sender(platform, contact, body):
        sent.append((platform, contact, body))
        return True

    store = FakeLeadsStore([{"lead_id": "L1", "name": "Ada",
                             "platform": "whatsapp", "contact": "+2341",
                             "status": "new", "stage": "qualified"}])
    adapter = LeadsAdapter(store=store, dm_sender=dm_sender)
    policy = Policy()
    policy.allow("social.dm", note="owner allows lead DMs")
    policy.grant("social.dm")
    auth = AgencyAuthority(gradient=PermissionGradient(policy))
    # explicit=True: the owner approved THIS draft (trial pattern) —
    # the confirmable gate is bypassed, deny/grant still apply.
    res = adapter.send_dm({"lead_id": "L1", "name": "Ada",
                           "platform": "whatsapp", "contact": "+2341"},
                          draft="Hi Ada, following up", authority=auth,
                          explicit=True)
    assert res["sent"] is True
    assert sent and sent[0][1] == "+2341"
    assert store.leads["L1"]["status"] == "dmed"


def test_6_leads_send_dm_denied_without_authority():
    def dm_sender(platform, contact, body):
        raise AssertionError("must not send without authority")

    store = FakeLeadsStore([{"lead_id": "L1", "name": "Ada",
                             "platform": "whatsapp", "contact": "+2341",
                             "status": "new"}])
    adapter = LeadsAdapter(store=store, dm_sender=dm_sender)
    auth = AgencyAuthority(gradient=PermissionGradient(Policy()))
    # No explicit approval, no grant: the confirmable gate denies and
    # the draft stays held — never sent.
    res = adapter.send_dm({"lead_id": "L1", "platform": "whatsapp",
                           "contact": "+2341"},
                          draft="Hi", authority=auth)
    assert res["sent"] is False
    assert res.get("reason")


def test_6_leads_contact_trigger_sends_when_authorized():
    sent = []

    def dm_sender(platform, contact, body):
        sent.append((platform, contact, body))
        return True

    class LeadsWithSend:
        def __init__(self):
            self._adapter = LeadsAdapter(
                store=FakeLeadsStore([{"lead_id": "L1", "name": "Ada",
                                       "platform": "whatsapp",
                                       "contact": "+2341", "status": "new",
                                       "stage": "qualified"}]),
                dm_sender=dm_sender)

        def live_leads(self):
            # Reflect the store, so the checker's independent re-read
            # sees the dmed status the send wrote.
            return [{"lead_id": lid, "name": l.get("name", ""),
                     "platform": l.get("platform", ""),
                     "contact": l.get("contact", ""),
                     "status": l.get("status", ""),
                     "stage": l.get("stage", "")}
                    for lid, l in
                    self._adapter.store.leads.items()]

        def advance(self, lead, nxt):
            return {"lead_id": "L1", "to": nxt, "status": "advanced",
                    "store_status": "dmed"}

        def send_dm(self, lead, draft="", authority=None,
                    explicit=False):
            return self._adapter.send_dm(lead, draft=draft,
                                         authority=authority,
                                         explicit=explicit)

    policy = Policy()
    policy.allow("social.dm", note="owner allows lead DMs")
    policy.grant("social.dm")
    auth = AgencyAuthority(gradient=PermissionGradient(policy))
    _, store, sys_ = _systems(adapters={"leads": LeadsWithSend()},
                              authority=auth)
    sys_.seed_initiatives(now=1000.0)
    sys_.scan_proactive(now=2000.0)
    # The trigger carries the owner-approved draft: the trial pattern
    # — explicit approval for THIS draft.
    for it in store.list_all():
        if it.objective == "leads" and it.kind == "evidence_opportunity":
            sys_._trigger_payloads[it.dedupe_key] = {
                "lead": {"lead_id": "L1", "name": "Ada",
                         "platform": "whatsapp", "contact": "+2341",
                         "status": "new", "stage": "qualified"},
                "approved_draft": "Hi Ada — following up on your enquiry."}
    sys_.tick("leads", now=2000.0)
    assert sent, "contact step must actually send the approved draft"
    done = [i for i in store.list_all()
            if i.objective == "leads" and i.status == "completed"]
    assert done and "DM sent" in str(done[0].stop_reason)


# ── 7. maker-checker ───────────────────────────────────────────────

class CheckerDeals:
    """Worker claims a saving on first check; independent re-check says no."""

    def __init__(self):
        self.calls = 0

    def check_alerts(self):
        return [{"id": "a1", "product": "laptop", "reason": "drop"}]

    def check_need(self, need):
        self.calls += 1
        if self.calls == 1:
            return {"evaluation": {"is_saving": True, "action": "alert",
                                   "reason": "landed below historical low"},
                    "observed": {"price": 40000.0, "landed_price": 45000.0,
                                 "historical_low": 50000.0}}
        return {"evaluation": {"is_saving": False, "action": "watch",
                               "reason": "re-check: not below historical low"},
                "observed": {"price": 60000.0, "landed_price": 65000.0,
                             "historical_low": 50000.0}}


def test_7_maker_checker_rejects_worker_claim(tmp_path):
    # Mandate present so the worker's claim reaches the checker gate
    # (without a mandate the handler blocks before completing).
    _, store, sys_ = _systems(adapters={"deals": CheckerDeals()},
                              authority=_mandate_auth(tmp_path))
    sys_.seed_initiatives(now=1000.0)
    sys_.scan_proactive(now=2000.0)
    sys_.tick("deals", now=2000.0)
    done = [i for i in store.list_all()
            if i.objective == "deals" and i.status == "completed"]
    assert not done, "worker claim must NOT self-verify into completion"
    blocked = [i for i in store.list_active()
               if i.objective == "deals" and i.blocker]
    assert blocked and "checker" in blocked[0].blocker.lower()


def test_7_maker_checker_passes_when_recheck_agrees(tmp_path):
    def executor(need, observed):
        return {"purchased": True, "receipt": {"order_id": "ORD-1"}}

    adapter = DealsAdapter(hunter=FakeHunter(), purchase_executor=executor)
    auth = _mandate_auth(tmp_path, allow_purchase=True)
    _, store, sys_ = _systems(adapters={"deals": adapter}, authority=auth)
    sys_.seed_initiatives(now=1000.0)
    sys_.scan_proactive(now=2000.0)
    sys_.tick("deals", now=2000.0)
    done = [i for i in store.list_all()
            if i.objective == "deals" and i.status == "completed"]
    assert done, "agreeing re-check must let the completion through"


# ── 8. film sources verified-live ──────────────────────────────────

def test_8_film_chain_has_verified_live_source():
    from nomorals.agency.sources_registry import collect
    from nomorals.media.film_sources import FILM_CHAIN, ArchiveOrgFilmSource

    assert any(isinstance(s, ArchiveOrgFilmSource) for s in FILM_CHAIN), \
        "film chain needs a verified-live source"
    specs, _ = collect()
    film = [s for s in specs if s.category == "film"]
    assert any(s.key == "film:archiveorg" and s.kind == "json"
               for s in film)


def test_8_archiveorg_film_search_live():
    """Live check: skip (not fail) when the network is unavailable."""
    from nomorals.media.film_sources import ArchiveOrgFilmSource

    try:
        results = ArchiveOrgFilmSource().search("Avatar", limit=3)
    except Exception as exc:  # noqa: BLE001 - offline env
        pytest.skip(f"no network for live film check: {exc}")
    assert results, "archive.org film search must return real candidates"
    assert all(r.title and r.url for r in results)


# ── 9. Boomplay browser-backed search ──────────────────────────────

_BOOMPLAY_MUSIC_HTML = """
<html><body>
<div class="songList">
<button data-id="13977" data-data="13977@+#5@+#10039@+#https://x/img.jpg@+#Outro Remember Me@+#Burna Boy@+#tok@+#03:42@+#OMS@+#0@+#tok2@+#Album@+#https://x/cover.jpg@+#https://x/artist.jpg@+#1777640@+#1777431@+#0">
<button class="songWrap"><div class="songText"><strong class="songName">Outro Remember Me</strong><p class="singerName">Burna Boy</p></div></button>
</button>
</div>
</body></html>
"""


def test_9_boomplay_browser_search_parses_song_entries():
    from nomorals.media.sources import BoomplaySource

    src = BoomplaySource()
    src._browser_fetch = lambda url: _BOOMPLAY_MUSIC_HTML
    results = src.search("Burna Boy", limit=5)
    assert results, "browser-backed search must return candidates"
    assert results[0].title == "Outro Remember Me"
    assert results[0].artist == "Burna Boy"
    assert results[0].url == "https://www.boomplay.com/songs/13977"


def test_9_boomplay_back_in_active_chain():
    from nomorals.media import resolver as resolver_mod

    assert "boomplay-search" in resolver_mod.SourceResolver._CHAIN_NAMES \
        if hasattr(resolver_mod.SourceResolver, "_CHAIN_NAMES") else True
    # The resolver's text chain must include the boomplay strategy.
    import inspect

    src = inspect.getsource(resolver_mod.SourceResolver._resolve_text)
    assert "boomplay" in src.lower(), \
        "boomplay must be back in the active resolve chain"


def test_9_boomplay_search_live():
    """Live check: skip (not fail) when the network is unavailable."""
    from nomorals.media.sources import BoomplaySource

    try:
        results = BoomplaySource().search("Burna Boy", limit=3)
    except Exception as exc:  # noqa: BLE001 - offline env
        pytest.skip(f"no network for live Boomplay check: {exc}")
    assert results, "browser-backed Boomplay search must return candidates"
    assert all(r.title and r.url.startswith("https://www.boomplay.com/songs/")
               for r in results)


# ── 10. SourceHealth runs on schedule ──────────────────────────────

def test_10_expansion_run_cycle_calls_sourcehealth():
    from nomorals.agency.expansion import ExpansionSystems
    from nomorals.agency.sourcehealth import (
        FetchResult, SourceHealth, SourceSpec,
    )
    from nomorals.agency.sources_registry import collect as collect_sources

    db = Database(":memory:")
    health = SourceHealth(db)
    fetched = []

    def fetcher(url, headers=None):
        fetched.append(url)
        return FetchResult(200, b'{"results": [{"id": 1}]}', {})

    specs = [SourceSpec(key="t:one", category="t", name="one",
                        url="https://example.com/x", kind="json")]
    systems = ExpansionSystems(
        db=db, source_health=health,
        registry_collect=lambda: specs, registry_fetcher=fetcher)
    systems.seed_initiatives(now=1000.0)
    out = systems.run_cycle(budget_s=60.0, now=2000.0)
    assert fetched, "expansion cycle must actually run SourceHealth"
    assert health.record("t:one") is not None
    assert health.record("t:one").status == "verified-live"
    assert "sourcehealth" in out["ran"]


def test_10_scheduled_expansion_helper_runs_sourcehealth():
    from nomorals.agency.sourcehealth import FetchResult, SourceSpec
    from nomorals.agency.wiring import run_scheduled_expansion

    db = Database(":memory:")
    fetched = []

    def fetcher(url, headers=None):
        fetched.append(url)
        return FetchResult(200, b'{"results": [{"id": 1}]}', {})

    out = run_scheduled_expansion(
        db, registry_collect=lambda: [SourceSpec(
            key="t:one", category="t", name="one",
            url="https://example.com/x", kind="json")],
        registry_fetcher=fetcher)
    assert fetched
    assert out["systems_run"] >= 1


# ── 11. expansion gated actions go through standing authority ──────

class GatedOrgan:
    gated_capability = "media.publish"

    def __init__(self):
        self.ran = []

    def tick(self):
        self.ran.append("tick")
        return {"done": 1}


class RecordingAuthority:
    """Trial-pattern authority: explicit go-ahead for THAT action only."""

    def __init__(self, explicit_ids=frozenset()):
        self.explicit_ids = frozenset(explicit_ids)
        self.checked = []

    def check(self, capability, *, explicit_override=False, draft="",
              context=None):
        self.checked.append((capability, explicit_override, context))
        allowed = bool(explicit_override and context
                       and context.get("action_id") in self.explicit_ids)
        return SimpleNamespace(
            allowed=allowed,
            reason="explicit go-ahead required" if not allowed else "ok",
            proposal_id="" if allowed else "prp_1",
            gradient="act_silent" if allowed else "propose")


def test_11_expansion_gated_tick_blocked_without_go_ahead():
    from nomorals.agency.expansion import ExpansionSystems

    organ = GatedOrgan()
    auth = RecordingAuthority()
    systems = ExpansionSystems(db=Database(":memory:"),
                               organs={"media": organ}, authority=auth)
    systems.seed_initiatives(now=1000.0)
    systems.tick("media", now=2000.0)
    assert organ.ran == [], "gated organ must NOT run without authority"
    assert auth.checked, "expansion must consult standing authority"
    assert auth.checked[0][0] == "media.publish"


def test_11_expansion_gated_tick_runs_with_trial_go_ahead():
    from nomorals.agency.expansion import ExpansionSystems

    organ = GatedOrgan()
    auth = RecordingAuthority(explicit_ids={"media:media.publish"})
    systems = ExpansionSystems(
        db=Database(":memory:"), organs={"media": organ}, authority=auth,
        explicit_go_aheads={"media:media.publish"})
    systems.seed_initiatives(now=1000.0)
    systems.tick("media", now=2000.0)
    assert organ.ran == ["tick"]


# ── 13. process isolation can truly kill ───────────────────────────

def _sleep_forever():
    import time as _t

    _t.sleep(3600)
    return "never"


def _add(a, b):
    return a + b


def _mutate_global():
    import nomorals.core.process_isolation as _m

    _m._PROBE_MARKER = "child-wrote-this"
    return "mutated"


def test_13_process_isolation_kills_hung_call():
    from nomorals.core.process_isolation import run_isolated

    started = time.time()
    res = run_isolated(_sleep_forever, timeout_s=1.0)
    elapsed = time.time() - started
    assert res.status == "timeout-killed"
    assert elapsed < 10, f"kill must be prompt, took {elapsed:.1f}s"


def test_13_process_isolation_returns_results():
    from nomorals.core.process_isolation import run_isolated

    res = run_isolated(_add, 2, 3, timeout_s=10.0)
    assert res.status == "ok"
    assert res.value == 5


def test_13_process_isolation_child_mutations_do_not_leak():
    import nomorals.core.process_isolation as mod

    assert not hasattr(mod, "_PROBE_MARKER")
    res = mod.run_isolated(_mutate_global, timeout_s=10.0)
    assert res.status == "ok"
    assert not hasattr(mod, "_PROBE_MARKER"), \
        "child process mutations must not leak into the parent"


def test_13_tool_loop_kills_isolated_tool():
    from nomorals.agents.partner.tool_loop import ToolCallingLoop, ToolCall
    from nomorals.tools.registry import ToolRegistry

    reg = ToolRegistry(enforce=False)
    reg.register("slow_tool", _sleep_forever, description="hangs")
    loop = ToolCallingLoop(llm=None, registry=reg, tool_timeout_s=2.0,
                           isolated_tools={"slow_tool"})
    out = loop._execute(ToolCall(name="slow_tool", args={}), actor="owner")
    assert "killed" in out.lower(), out
    assert "unknown" not in out.lower(), \
        "a killed process has a KNOWN outcome (dead), not unknown"


# ── 14. typed idempotent replay ────────────────────────────────────

def test_14_replay_restores_actual_types():
    from nomorals.tools.registry import ToolRegistry

    calls = []

    def effect():
        calls.append(1)
        return {"t": (1, 2), "s": {3, 4}, "b": b"xy",
                "d": {("a", "b"): 1}, "f": frozenset([9])}

    reg = ToolRegistry(enforce=False)
    reg.register("effect", effect, description="effect")
    first = reg.call("effect", actor="alice", idempotency_key="k1")
    assert first.ok
    second = reg.call("effect", actor="alice", idempotency_key="k1")
    assert second.ok
    assert len(calls) == 1, "replay must not re-execute"
    val = second.value
    assert isinstance(val["t"], tuple) and val["t"] == (1, 2)
    assert isinstance(val["s"], set) and val["s"] == {3, 4}
    assert isinstance(val["b"], bytes) and val["b"] == b"xy"
    assert isinstance(val["d"], dict) and val["d"] == {("a", "b"): 1}
    assert isinstance(val["f"], frozenset) and val["f"] == frozenset([9])


def test_14_replay_restores_types_across_restart():
    from nomorals.tools.registry import ToolRegistry

    db = Database(":memory:")
    db.migrate()
    from types import SimpleNamespace

    def make():
        reg = ToolRegistry(context=SimpleNamespace(db=db), enforce=False)
        reg.register("effect", lambda: {"t": (1, 2), "s": {3}},
                     description="effect")
        return reg

    first = make().call("effect", actor="alice", idempotency_key="k1")
    assert first.ok
    second = make().call("effect", actor="alice", idempotency_key="k1")
    assert second.ok
    assert isinstance(second.value["t"], tuple)
    assert isinstance(second.value["s"], set)


# ── 15. format_reply_detailed fallback ──────────────────────────────

def test_15_unstructured_tail_never_silently_discarded():
    from nomorals.partner.style import (
        StyleMeasurement, StyleProfile, format_reply_detailed,
    )

    profile = StyleProfile(
        audience="owner", platform="telegram", max_chars=120,
        target_chars=80, emoji_budget=0, allow_sections=False,
        allow_lists=False, allow_markup=False,
        measured=StyleMeasurement())
    sentences = [f"Sentence number {i} carries real content here."
                 for i in range(20)]
    text = " ".join(sentences)
    preview, overflow = format_reply_detailed(text, profile)
    assert overflow, "unstructured tail must land in overflow, not vanish"
    combined = preview + " " + overflow
    for s in sentences:
        assert s in combined, f"lost sentence: {s}"


def test_15_reformatted_head_does_not_duplicate_overflow():
    """LCP failure case: format_reply strips the '## ' header marker, so
    the preview is NOT a prefix of the original. The old LCP fallback
    returned the ENTIRE original as overflow (everything duplicated).
    The fallback must subtract what the preview already contains."""
    from nomorals.partner.style import (
        StyleMeasurement, StyleProfile, format_reply_detailed,
    )

    profile = StyleProfile(
        audience="owner", platform="telegram", max_chars=120,
        target_chars=80, emoji_budget=0, allow_sections=False,
        allow_lists=False, allow_markup=False,
        measured=StyleMeasurement())
    sentences = [f"Sentence number {i} carries real content here."
                 for i in range(20)]
    text = "## Report\n\n" + " ".join(sentences)
    preview, overflow = format_reply_detailed(text, profile)
    assert overflow, "tail must land in overflow"
    # Sentences already fully present in the preview must NOT be
    # duplicated into the overflow.
    assert sentences[0] not in overflow, \
        "overflow must not duplicate content the preview already has"
    assert sentences[-1] in preview + " " + overflow


def test_15_fallback_is_deterministic_not_lcp():
    from nomorals.partner.style import (
        StyleMeasurement, StyleProfile, format_reply_detailed,
    )

    profile = StyleProfile(
        audience="owner", platform="telegram", max_chars=120,
        target_chars=80, emoji_budget=0, allow_sections=False,
        allow_lists=False, allow_markup=False,
        measured=StyleMeasurement())
    text = "Alpha beta gamma. " * 40
    p1, o1 = format_reply_detailed(text, profile)
    p2, o2 = format_reply_detailed(text, profile)
    assert (p1, o1) == (p2, o2)


# ── 16. layering ───────────────────────────────────────────────────

def test_16_layering_green():
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_layering.py", "-q"],
        capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-1000:]


# ── 17. play search fix collection ─────────────────────────────────

def test_17_ytdlp_available_importable():
    from nomorals.media.resolver import ytdlp_available

    assert isinstance(ytdlp_available(), bool)

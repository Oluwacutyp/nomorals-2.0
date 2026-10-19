"""PASS #3: agency / autonomy layer — verified bugs, failing-first.

Each test below failed against the pre-fix code (verified) and passes
after the fix. No speculative refactors: every case is grounded in a
concrete stall/duplication/hang observed in the code.
"""
import os
import tempfile
import time

from nomorals.agency.initiatives import Initiative, InitiativeStore
from nomorals.agency.systems import ActiveSystems
from nomorals.storage.db import Database


def _systems():
    tmp = tempfile.mkdtemp()
    db = Database(os.path.join(tmp, "t.db"))
    store = InitiativeStore(db)
    return ActiveSystems(db=db, store=store), store


class _Leads:
    """Fake leads adapter: advance() persists the stage change."""

    def __init__(self):
        self.advances = []
        self.sends = []
        self.stage = {}

    def live_leads(self):
        return []

    def advance(self, lead, stage):
        lid = str(lead.get("lead_id", ""))
        self.advances.append((lid, stage))
        self.stage[lid] = stage
        return {"status": "advanced"}

    def send_dm(self, lead, draft="", authority=None, explicit=False):
        self.sends.append((str(lead.get("lead_id", "")), draft, explicit))
        if explicit:
            return {"sent": True, "evidence": {"channel": "test"}}
        return {"sent": False, "reason": "needs owner approval"}


def _seed_leads_trigger(systems, store, now, lead):
    systems.seed_initiatives(now=now)
    it = store.create(Initiative(
        outcome="leads: new lead", kind="evidence_opportunity",
        objective="leads", evidence="lead L1", next_action="work",
        next_review_at=now + 3600, dedupe_key="leads:new:L1", priority=0.5))
    systems._trigger_payloads["leads:new:L1"] = {"lead": dict(lead)}
    return it


# ── Bug 1: backed-off triggers are re-worked before their wake-up ──

def test_backed_off_trigger_not_reworked_before_wakeup():
    systems, store = _systems()
    now = time.time()
    adapter = _Leads()
    systems.adapters["leads"] = adapter
    it = _seed_leads_trigger(
        systems, store, now, {"lead_id": "L1", "stage": "discovered"})

    r1 = systems.tick("leads", now=now)
    assert r1.useful
    it1 = store.get(it.id)
    assert it1.blocker  # waiting on DM approval now
    assert it1.next_review_at > now
    assert adapter.advances == [("L1", "qualified")]

    # 30 minutes later: the wake-up has NOT arrived. The trigger must
    # be left alone — the code itself asked to be woken in an hour.
    systems.tick("leads", now=now + 1800)
    assert adapter.advances == [("L1", "qualified")], \
        "blocked trigger was re-worked before its wake-up"


def test_fresh_trigger_worked_immediately():
    # Pin: the wake-up filter must not delay never-worked triggers.
    systems, store = _systems()
    now = time.time()
    adapter = _Leads()
    systems.adapters["leads"] = adapter
    _seed_leads_trigger(
        systems, store, now, {"lead_id": "L1", "stage": "discovered"})
    systems.tick("leads", now=now)
    assert adapter.advances == [("L1", "qualified")]


# ── Bug 2: lead advanced twice across the DM-wait retry ──

def test_lead_not_advanced_twice_across_dm_wait_retry():
    systems, store = _systems()
    now = time.time()
    adapter = _Leads()
    systems.adapters["leads"] = adapter
    it = _seed_leads_trigger(
        systems, store, now, {"lead_id": "L1", "stage": "discovered"})

    systems.tick("leads", now=now)  # advances, then blocks on DM approval
    assert adapter.advances == [("L1", "qualified")]
    assert adapter.sends  # DM attempt was made and held

    # Owner approves the draft an hour later; the wake-up arrives.
    systems._trigger_payloads["leads:new:L1"] = {
        "lead": {"lead_id": "L1", "stage": "qualified"},
        "approved_draft": "Hi Ann — following up (owner approved).",
    }
    systems.tick("leads", now=now + 3700)

    assert adapter.advances == [("L1", "qualified")], \
        "lead was advanced a second time while waiting on DM approval"
    assert store.get(it.id).status == "completed"
    # The approved draft went through the explicit path exactly once.
    assert adapter.sends[-1][2] is True



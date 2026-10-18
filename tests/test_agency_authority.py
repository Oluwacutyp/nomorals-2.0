"""Authority wiring tests: the agency layer uses the EXISTING
standing-authority systems (core/policy.py + finance/mandate.py +
trial pattern) — it does not build a parallel one.

Verdict: PARTIAL. The trial system is SIGNUP-SCOPED ONLY (one account
per service, disposable persona, explicit confirmation for THAT
signup). General autonomous-work authority already exists in
core/policy.py (Policy + PermissionGradient) and finance/mandate.py
(PaymentMandate) — AgencyAuthority is the mapping onto those, not a
parallel system.
"""
from __future__ import annotations

from functools import partial

from nomorals.agency.adapters import AgencyAuthority
from nomorals.agency.initiatives import InitiativeStore
from nomorals.agency.systems import ActiveSystems
from nomorals.core.policy import (
    CapabilitySet, PermissionGradient, Policy,
)
from nomorals.finance.mandate import (
    MandateStore, check_mandate, issue_mandate,
)
from nomorals.storage.db import Database


def _auth(**kw):
    """AgencyAuthority with the REAL gradient injected (tests are
    outside the layering boundary, so they may import core.policy)."""
    kw.setdefault("gradient", PermissionGradient(Policy()))
    return AgencyAuthority(**kw)


def test_bare_authority_without_injected_gradient_fails_closed():
    auth = AgencyAuthority()
    assert auth.may_post().allowed is False
    assert auth.may_dm().allowed is False
    assert auth.may_submit("g1", explicit=True).allowed is False
    assert auth.can_spend(1.0) is False


def test_default_deny_and_proposal_recorded():
    auth = _auth()
    decision = auth.may_post()
    assert decision.allowed is False
    assert decision.needs_proposal is True
    assert decision.proposal_id
    assert auth.gradient.get_proposal(decision.proposal_id) is not None


def test_allow_rule_plus_grant_act_silent():
    policy = Policy()
    policy.allow("social.post")
    auth = AgencyAuthority(gradient=PermissionGradient(policy),
                           grant=CapabilitySet.of("social.post"))
    decision = auth.may_post()
    assert decision.allowed is True
    assert decision.gradient == "act_silent"


def test_deny_rule_beats_grant():
    policy = Policy()
    policy.allow("social.post")
    policy.deny("social.post")
    auth = AgencyAuthority(gradient=PermissionGradient(policy),
                           grant=CapabilitySet.of("social.post"))
    assert auth.may_post().allowed is False


def test_dm_needs_approval_not_silent():
    auth = _auth()
    decision = auth.may_dm()
    assert decision.allowed is False
    assert decision.gradient == "act_with_approval"


def test_explicit_override_still_needs_grant():
    # Trial pattern: explicit go-ahead for THAT gig, but grants apply.
    auth = AgencyAuthority(gradient=PermissionGradient(Policy()),
                           explicit_submit_ids=frozenset({"g1"}))
    assert auth.may_submit("g1", explicit=True).allowed is False


def test_explicit_override_obeys_deny():
    policy = Policy()
    policy.allow("gig.submit")
    policy.deny("gig.submit")
    auth = AgencyAuthority(gradient=PermissionGradient(policy),
                           grant=CapabilitySet.of("gig.submit"),
                           explicit_submit_ids=frozenset({"g1"}))
    assert auth.may_submit("g1", explicit=True).allowed is False


def test_explicit_submit_with_grant_allowed():
    policy = Policy()
    policy.allow("gig.submit")
    auth = AgencyAuthority(gradient=PermissionGradient(policy),
                           grant=CapabilitySet.of("gig.submit"),
                           explicit_submit_ids=frozenset({"g1"}))
    decision = auth.may_submit("g1", explicit=True)
    assert decision.allowed is True
    assert decision.explicit_override is True


def test_decisions_are_audit_verifiable():
    auth = _auth()
    decision = auth.may_post()
    assert auth.gradient.policy.verify_decision(decision) is True


def test_no_mandate_no_spend():
    auth = _auth()
    assert auth.can_spend(1.0) is False
    assert "mandate" in auth.spend_reason(1.0)


def test_mandate_gates_within_and_above_caps(tmp_path):
    store = MandateStore(path=str(tmp_path / "mandates.json"))
    issue_mandate(store, scope="purchase", cap_per_txn=5_000_000,
                  cap_per_day=50_000_000)
    auth = _auth(mandate_check=partial(check_mandate, store))
    assert auth.can_spend(42_000.0) is True   # ₦42,000 < ₦50,000/txn cap
    assert auth.can_spend(60_000.0) is False  # exceeds per-txn cap


def test_revoked_mandate_kills_spending(tmp_path):
    store = MandateStore(path=str(tmp_path / "mandates.json"))
    mandate = issue_mandate(store, scope="purchase",
                            cap_per_txn=5_000_000, cap_per_day=50_000_000)
    auth = _auth(mandate_check=partial(check_mandate, store))
    assert auth.can_spend(1_000.0) is True
    store.revoke(mandate.id)
    assert auth.can_spend(1_000.0) is False


def test_trial_is_signup_scoped_not_general_authority():
    """The trial system grants authority for signups only — it has no
    may_post / may_submit / can_spend surface at all."""
    from pathlib import Path
    flow = Path(__file__).resolve().parent.parent \
        / "nomorals" / "agents" / "trial" / "flow.py"
    text = flow.read_text(encoding="utf-8")
    assert "def may_post" not in text
    assert "def may_submit" not in text
    assert "def can_spend" not in text
    assert "confirm" in text.lower()  # confirmation gate for THAT signup


def test_social_trigger_end_to_end_authority():
    class FakeSocial:
        def __init__(self):
            self.due = [{"post_id": "p1", "text": "hello"}]
            self.published = []

        def backlog(self):
            return {"unanswered": 0, "promised_replies": 1,
                    "milestones": 0, "stop_signal": False,
                    "cadence_ok": True}

        def due_posts(self):
            return list(self.due)

        def publish_due(self):
            out = list(self.due)
            self.published.extend(out)
            self.due = []
            return out

        def draft_for(self, action):
            return f"draft: {action}"

    # Default: proposal, no publish.
    soc = FakeSocial()
    db = Database(":memory:")
    store = InitiativeStore(db)
    systems = ActiveSystems(db=db, store=store, adapters={"social": soc},
                            authority=_auth())
    systems.seed_initiatives(now=1000.0)
    systems.scan_proactive(now=2000.0)
    systems.tick("social", now=2000.0)
    assert soc.published == []
    blocked = [it for it in store.list_active()
               if it.objective == "social" and it.blocker]
    assert blocked and "proposal" in blocked[0].blocker

    # Act-silent: publishes.
    soc2 = FakeSocial()
    policy = Policy()
    policy.allow("social.post")
    auth2 = AgencyAuthority(
        gradient=PermissionGradient(policy),
        grant=CapabilitySet.of("social.post"))
    db2 = Database(":memory:")
    store2 = InitiativeStore(db2)
    systems2 = ActiveSystems(db=db2, store=store2, adapters={"social": soc2},
                             authority=auth2)
    systems2.seed_initiatives(now=1000.0)
    systems2.scan_proactive(now=2000.0)
    systems2.tick("social", now=2000.0)
    assert len(soc2.published) == 1

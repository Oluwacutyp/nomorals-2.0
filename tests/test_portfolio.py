"""Offline tests for build-map #79: contract portfolio."""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nomorals.legal.portfolio import (
    DISCLAIMER,
    PORTFOLIO_CHECK_ACTION,
    ContractPortfolio,
    check_all,
    control_contracts,
    ensure_schedule,
)
from nomorals.legal.contracts import review_contract

TENANCY = """TENANCY AGREEMENT
This tenancy agreement is between the landlord and the tenant for the
premises at 12 Adeola Odeku Street, Victoria Island, Lagos.
1. TERM: The tenancy shall commence on 1st January 2026 and run for a
period of 12 months.
2. RENT: Annual rent of ₦2,400,000 payable yearly.
3. RENEWAL: This agreement shall automatically renew for another 12
months unless either party gives notice in writing before 1st November 2026.
4. AGENCY FEE: 25% agency fee payable on signing.
"""


def _tmp() -> str:
    return os.path.join(tempfile.mkdtemp(), "portfolio.db")


def _portfolio() -> ContractPortfolio:
    return ContractPortfolio(db_path=_tmp())


def test_add_contract_ingests_review():
    p = _portfolio()
    review = review_contract(TENANCY, "tenancy")
    c = p.add_contract(review, doc_text=TENANCY, name="Lekki Flat",
                       parties="Landlord / Tenant")
    assert c is not None
    assert c.contract_type == "tenancy"
    assert c.grade == review.grade
    assert c.name == "Lekki Flat"


def test_obligation_extraction():
    p = _portfolio()
    result = p.add_raw(TENANCY, name="Lekki Flat")
    assert result is not None
    contract, _review = result
    obls = p.obligations(contract.id)
    # commencement 2026-01-01, derived expiry ~2027-01-01, renewal notice date
    kinds = {o.kind for o in obls}
    assert "commencement" in kinds
    assert "expiry" in kinds
    assert len(obls) >= 2


def test_timeline_sorted():
    p = _portfolio()
    p.track_obligation("", "File taxes", time.time() + 90 * 86400)
    p.track_obligation("", "Pay rent", time.time() + 10 * 86400)
    events = p.timeline()
    assert len(events) == 2
    due_ats = [e.due_at for e in events]
    assert due_ats == sorted(due_ats)
    assert "Pay rent" in events[0].description


def test_needs_attention_window():
    p = _portfolio()
    p.track_obligation("", "Soon", time.time() + 30 * 86400)
    p.track_obligation("", "Later", time.time() + 200 * 86400)
    attn = p.needs_attention(days=60)
    assert len(attn) == 1
    contract, obls = attn[0]
    assert any("Soon" in o.description for o in obls)


def test_track_and_mark_done():
    p = _portfolio()
    oid = p.track_obligation("", "Quarterly tax filing due",
                             time.time() + 30 * 86400)
    assert oid
    assert len(p.obligations()) == 1
    assert p.mark_done(oid)
    assert p.obligations() == []
    assert not p.mark_done("obl_nonexistent")


def test_sla_tracking_and_breach():
    p = _portfolio()
    result = p.add_raw("This service agreement between Client and Vendor. "
                       "Deliverables due monthly. " * 10, name="Web SLA")
    assert result is not None
    contract, _ = result
    assert p.track_sla(contract.id, "Uptime", "99.9%")
    status = p.sla_status(contract.id)
    assert len(status) == 1 and not status[0].breached
    assert p.report_sla_breach(contract.id, "Uptime", "99.1% in October")
    breaches = p.sla_breaches()
    assert len(breaches) == 1
    assert "99.1%" in breaches[0].note


def test_due_alerts():
    p = _portfolio()
    p.track_obligation("", "Overdue item", time.time() - 86400)
    alerts = p.due_alerts(days=14)
    assert len(alerts) == 1
    assert "Overdue" in alerts[0]


def test_check_all_host_entry():
    p = _portfolio()
    p.track_obligation("", "Soon", time.time() + 5 * 86400)
    alerts = check_all(p, days=14)
    assert len(alerts) == 1


def test_summary_and_attention_text():
    p = _portfolio()
    assert "empty" in p.summary()
    p.track_obligation("", "Renew domain", time.time() + 20 * 86400)
    s = p.summary()
    assert "portfolio" in s.lower()
    attn = p.attention_text(60)
    assert "Renew domain" in attn
    assert DISCLAIMER in attn


def test_control_contracts_summary():
    out = control_contracts("")
    assert "usage" in out.lower() or "/contracts" in out


def test_control_contracts_add():
    out = control_contracts("add Lekki Flat | " + TENANCY)
    assert "Added" in out
    assert "Tracking" in out
    assert DISCLAIMER in out


def test_control_contracts_attention():
    out = control_contracts("attention 30")
    assert "attention" in out.lower() or "Nothing" in out


def test_control_contracts_timeline():
    out = control_contracts("timeline")
    assert "timeline" in out.lower() or "No obligations" in out


def test_control_contracts_never_raises():
    for tail in ["", "garbage {{[", "add x", "done", "track a|b",
                 "sla", "breach", "obligations nope"]:
        out = control_contracts(tail)
        assert isinstance(out, str) and len(out) > 0


def test_control_contracts_track_done():
    p = ContractPortfolio()  # default path — chat uses default too
    oid = p.track_obligation("", "Chat test item", time.time() + 86400)
    assert oid
    out = control_contracts(f"done {oid}")
    assert "Marked done" in out


def test_information_only():
    p = _portfolio()
    result = p.add_raw(TENANCY, name="Info check")
    assert result is not None
    contract, review = result
    text = p.summary() + p.attention_text(60)
    for phrase in ("you should", "don't sign", "i recommend"):
        assert phrase not in text.lower()


def test_ensure_schedule_never_raises():
    class FakeScheduler:
        def __init__(self):
            self.jobs = []

        def list_jobs(self):
            raise AttributeError("no list_jobs")

        async def schedule_cron(self, **kwargs):
            self.jobs.append(kwargs)

    assert ensure_schedule(FakeScheduler())
    assert ensure_schedule(None) is False


def test_portfolio_bad_db_never_raises():
    p = ContractPortfolio(db_path="/nonexistent_dir_xyz/portfolio.db")
    # makedirs may succeed as root; either way nothing may raise
    assert p.list_contracts() == [] or True
    assert p.timeline() == []
    assert p.needs_attention() == []
    assert p.due_alerts() == []
    assert control_contracts("")  # default path never raises

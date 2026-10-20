"""MORE_TO_ADD gaps 7-10 + expose-dormant registry.

Invariants:
* asset reminders fire on interval (km OR time, whichever first);
* will sealed content never leaks plaintext (marker only; reveal needs
  the owner-injected unlock fn and hash-verifies);
* subscription auditor never cancels without confirmation — there is NO
  cancel-act function in the module, only reminders + draft text;
* public face rejects private data and keeps unapproved pages draft;
* expose-dormant registry entries all exist on disk.
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pytest

from nomorals.assets.maintenance import Asset, AssetLedger, ServiceRecord
from nomorals.finance.subscriptions import (
    detect_subscriptions,
    draft_cancel_message,
    renewal_reminders,
)
from nomorals.legacy.will import SEALED_MARKER, Beneficiary, DigitalAsset, Executor, SealedError, Will
from nomorals.meta.expose_dormant import REGISTRY, SKIPPED, find, verify_all
from nomorals.publicface.publish import PublicPage, list_public, publish, scan_private_data

REPO_ROOT = Path(__file__).resolve().parent.parent


# --- 7. vehicle / asset maintenance -----------------------------------

def test_asset_reminder_fires_on_km_interval():
    ledger = AssetLedger()
    car = ledger.add_asset(Asset(name="Camry", kind="vehicle", interval_km=5000,
                                 last_service_km=10000, odometer_km=14000))
    assert ledger.due_reminders(date(2026, 10, 16)) == []  # below 90% soon-window
    car.odometer_km = 15000
    reminders = ledger.due_reminders(date(2026, 10, 16))
    assert len(reminders) == 1 and reminders[0].reason == "km" and reminders[0].overdue


def test_asset_reminder_fires_on_time_interval():
    ledger = AssetLedger()
    ledger.add_asset(Asset(name="Gen", kind="generator", interval_days=90,
                           last_service_date=date(2026, 7, 1)))
    assert ledger.due_reminders(date(2026, 8, 1)) == []  # outside 14-day soon-window
    reminders = ledger.due_reminders(date(2026, 9, 29))
    assert len(reminders) == 1 and reminders[0].reason == "time" and reminders[0].overdue


def test_asset_ledger_costs_auditable():
    ledger = AssetLedger()
    asset = ledger.add_asset(Asset(name="AC", kind="ac", interval_days=180,
                                   last_service_date=date(2026, 1, 1)))
    ledger.log_service(ServiceRecord(asset_id=asset.id, service_date=date(2026, 6, 1),
                                     description="gas refill", cost_minor=4500000,
                                     vendor="CoolAir", receipt_ref="R-1"))
    assert ledger.total_cost(asset.id) == 4500000
    exported = ledger.export_ledger()
    assert len(exported) == 1 and exported[0]["receipt_ref"] == "R-1"
    assert exported[0]["cost_minor"] == 4500000
    with pytest.raises(ValueError):
        ServiceRecord(asset_id=asset.id, service_date=date(2026, 6, 2),
                      description="bad", cost_minor=-1)


# --- 8. legacy / digital will ------------------------------------------

def _xor_seal(key: str):
    def seal(plaintext: str) -> str:
        return "".join(chr(ord(c) ^ ord(key[i % len(key)])) for i, c in enumerate(plaintext))
    return seal


def test_will_sealed_never_leaks_plaintext():
    will = Will(owner_name="Owner", executor=Executor(name="Exec"),
                beneficiaries=[Beneficiary(name="Kid")],
                digital_assets=[DigitalAsset(label="wallet", kind="wallet")])
    secret = "my secret instructions: give the house to Kid"
    will.seal_content(secret, _xor_seal("k"), is_owner=True)
    summary = will.summary(is_owner=True)
    assert summary["sealed_content"] == SEALED_MARKER
    assert secret not in repr(summary) and secret not in str(will.summary(is_owner=True))
    assert will.sealed_preview(is_owner=True) == SEALED_MARKER
    # reveal works with the matching owner-injected unlock fn
    assert will.reveal(_xor_seal("k"), is_owner=True) == secret
    # wrong key fails loudly, never returns garbage
    with pytest.raises(SealedError):
        will.reveal(_xor_seal("WRONG"), is_owner=True)


def test_will_owner_only_and_no_fake_seal():
    will = Will(owner_name="Owner", executor=Executor(name="Exec"))
    with pytest.raises(PermissionError):
        will.summary(is_owner=False)
    with pytest.raises(PermissionError):
        will.seal_content("x", _xor_seal("k"), is_owner=False)
    with pytest.raises(PermissionError):
        will.reveal(_xor_seal("k"), is_owner=False)
    # no seal fn -> refuse, never store plaintext or fake-encrypt
    with pytest.raises(ValueError):
        will.seal_content("x", None, is_owner=True)
    with pytest.raises(SealedError):
        will.reveal(None, is_owner=True)


# --- 9. subscription auditor --------------------------------------------

def _txns():
    return [
        {"merchant": "Netflix", "amount_minor": 650000, "date": date(2026, 7, 1)},
        {"merchant": "Netflix", "amount_minor": 650000, "date": date(2026, 8, 1)},
        {"merchant": "Netflix", "amount_minor": 650000, "date": date(2026, 9, 1)},
        {"merchant": "Mama Put", "amount_minor": 300000, "date": date(2026, 9, 2)},
    ]


def test_subscriptions_detected_from_supplied_transactions_only():
    assert detect_subscriptions([]) == []
    # as_of pinned inside the fixture's era (the sibling tests do the
    # same): estimated_next_renewal is the next FUTURE occurrence as of
    # as_of, so an unpinned call would roll past 2026-10-02 as the
    # wall clock advances.
    subs = detect_subscriptions(_txns(), as_of=date(2026, 9, 28))
    assert [s.merchant for s in subs] == ["Netflix"]
    sub = subs[0]
    assert sub.occurrences == 3 and sub.cost_minor == 650000
    assert sub.estimated_next_renewal == date(2026, 10, 2)  # Sep 1 + median gap 31d
    # one-off merchant is not a subscription
    assert all(s.merchant != "Mama Put" for s in subs)


def test_subscriptions_unused_flagged_only_on_missing_usage():
    subs = detect_subscriptions(_txns(), usage_records={"Netflix": date(2026, 10, 15)},
                                as_of=date(2026, 10, 16))
    assert subs[0].suspected_unused is False
    subs2 = detect_subscriptions(_txns(), usage_records={}, as_of=date(2026, 10, 16))
    assert subs2[0].suspected_unused is True
    assert "no usage records" in subs2[0].unused_reason


def test_subscription_never_cancels_without_confirmation():
    import nomorals.finance.subscriptions as mod
    # No cancel-act function exists anywhere in the module.
    for name in dir(mod):
        assert not name.startswith("cancel"), f"cancel-act function present: {name}"
    assert not hasattr(mod, "cancel_subscription")
    assert not hasattr(mod, "cancel")
    subs = detect_subscriptions(_txns(), as_of=date(2026, 9, 28))
    reminders = renewal_reminders(subs, as_of=date(2026, 9, 28), days_ahead=7)
    assert len(reminders) == 1
    assert reminders[0].merchant == "Netflix"
    # draft text only — calling it changes no state, cancels nothing
    draft = draft_cancel_message(subs[0])
    assert "cancel my subscription" in draft.lower()
    assert renewal_reminders(subs, as_of=date(2026, 9, 28), days_ahead=7)[0].merchant == "Netflix"


# --- 10. public face -----------------------------------------------------

def test_publicface_unapproved_stays_draft():
    page = PublicPage(title="JAMB Physics 101", content="Forces and motion.", pack="jamb-tutor")
    result = publish(page, owner_approved=False)
    assert result.published is False and page.status == "draft"
    assert result.link is None
    assert list_public([page]) == []


def test_publicface_rejects_private_data():
    assert scan_private_data("my vault password is hunter2") != []
    assert scan_private_data("api_key = 'abcd1234efgh5678'") != []
    assert scan_private_data("-----BEGIN PRIVATE KEY-----") != []
    assert scan_private_data("Clean public lesson content.") == []
    page = PublicPage(title="Yoruba Songs", content="token = 'secret-thing-here'", pack="yoruba-singing")
    result = publish(page, owner_approved=True)
    assert result.published is False and page.status == "draft"
    assert result.violations != []


def test_publicface_approved_clean_publishes_with_link():
    page = PublicPage(title="Oral Corpus Tale", content="Once upon a time...", pack="oral-corpus")
    result = publish(page, owner_approved=True)
    assert result.published is True and page.status == "public"
    assert result.link is not None and result.link.url_path == f"/p/{page.slug}"
    assert len(list_public([page])) == 1


# --- expose dormant -------------------------------------------------------

def test_expose_registry_entries_all_exist_on_disk():
    ok, missing = verify_all()
    assert missing == [], f"registry entries missing on disk: {missing}"
    assert len(ok) == len(REGISTRY) and len(ok) >= 30
    for entry in REGISTRY:
        assert (REPO_ROOT / entry.path).is_file(), entry.path
        assert entry.module.startswith("nomorals.")
        assert entry.invoke, entry.name
    assert SKIPPED == [], f"skipped entries: {SKIPPED}"


def test_expose_registry_covers_named_organs():
    for keyword in ("coach", "contracts", "Socratic", "FSRS", "Ekiti", "WAEC",
                    "email", "calendar", "payments", "fingerprint", "goals",
                    "stable", "mesh", "hub", "biometric", "MCP", "context"):
        assert find(keyword), f"no registry entry for {keyword}"
    assert find("REVIEW_STYLES") or find("contracts")

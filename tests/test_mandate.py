"""Agent payment mandate object — money architecture (#69).

Structural, not advisory: every money-moving dispatch checks the active
mandate before executing. All offline.
"""
import time

import pytest

from nomorals.finance.ledger import Ledger
from nomorals.finance.mandate import (
    SCOPE_TRANSFER,
    MandateError,
    MandateStore,
    check_mandate,
    daily_transfer_spend,
    issue_mandate,
    require_mandate,
)


@pytest.fixture()
def store(tmp_path):
    return MandateStore(tmp_path / "mandates.json")


@pytest.fixture()
def ledger(tmp_path):
    return Ledger(tmp_path / "ledger.jsonl")


def _issue(store, **kw):
    kw.setdefault("principal", "owner")
    kw.setdefault("scope", SCOPE_TRANSFER)
    kw.setdefault("cap_per_txn", 5_000_000)      # ₦50,000
    kw.setdefault("cap_per_day", 20_000_000)     # ₦200,000
    kw.setdefault("ttl_days", 30)
    return store.issue(**kw)


# ── issue ─────────────────────────────────────────────────────────────

def test_issue_creates_mandate(store):
    m = _issue(store)
    assert m.id.startswith("mand_")
    assert m.principal == "owner"
    assert m.scope == SCOPE_TRANSFER
    assert not m.revoked
    assert not m.expired
    assert store.get(m.id) is not None


def test_issue_rejects_bad_caps(store):
    with pytest.raises(MandateError):
        store.issue(principal="owner", scope="transfer",
                    cap_per_txn=0, cap_per_day=100)
    with pytest.raises(MandateError):
        # per-txn cannot exceed daily
        store.issue(principal="owner", scope="transfer",
                    cap_per_txn=200, cap_per_day=100)


def test_issue_rejects_raw_secret_as_credential_ref(store):
    with pytest.raises(MandateError):
        store.issue(principal="owner", scope="transfer",
                    cap_per_txn=100, cap_per_day=1000,
                    credential_ref="sk_live_abc123def456")


def test_credential_ref_is_reference_not_secret(store):
    m = _issue(store, credential_ref="paystack:primary")
    assert m.credential_ref == "paystack:primary"
    raw = store.path.read_text()
    assert "sk_live" not in raw and "sk_test" not in raw


# ── check ─────────────────────────────────────────────────────────────

def test_check_ok_within_caps(store, ledger):
    m = _issue(store)
    result = check_mandate(store, "owner", SCOPE_TRANSFER, 1_000_000,
                           ledger=ledger)
    assert result.ok
    assert result.mandate.id == m.id


def test_check_no_mandate_blocks(store, ledger):
    result = check_mandate(store, "owner", SCOPE_TRANSFER, 1_000_000,
                           ledger=ledger)
    assert not result.ok
    assert "no payment mandate" in result.reason


def test_require_mandate_raises_without_mandate(store):
    with pytest.raises(MandateError) as ctx:
        require_mandate(store, "owner", SCOPE_TRANSFER, 1_000_000)
    assert "no payment mandate" in str(ctx.value)


def test_check_per_txn_cap(store, ledger):
    _issue(store, cap_per_txn=5_000_000)
    result = check_mandate(store, "owner", SCOPE_TRANSFER, 6_000_000,
                           ledger=ledger)
    assert not result.ok
    assert "per-transaction cap" in result.reason


def test_check_daily_cap(store, ledger):
    _issue(store, cap_per_txn=20_000_000, cap_per_day=20_000_000)
    # ₦150,000 already spent today
    ledger.log(15_000_000, category="transfer", note="transfer to Ada",
               kind="spend", source="paystack")
    result = check_mandate(store, "owner", SCOPE_TRANSFER, 6_000_000,
                           ledger=ledger)
    assert not result.ok
    assert "daily cap" in result.reason
    assert result.daily_spent_kobo == 15_000_000


def test_check_daily_cap_allows_within(store, ledger):
    _issue(store, cap_per_txn=20_000_000, cap_per_day=20_000_000)
    ledger.log(15_000_000, category="transfer", note="transfer to Ada",
               kind="spend", source="paystack")
    result = check_mandate(store, "owner", SCOPE_TRANSFER, 4_000_000,
                           ledger=ledger)
    assert result.ok
    assert result.daily_remaining_kobo == 1_000_000


def test_daily_spend_counts_only_transfers(store, ledger):
    _issue(store)
    ledger.log(9_000_000, category="food", note="lunch", kind="spend")
    assert daily_transfer_spend(ledger) == 0
    ledger.log(9_000_000, category="transfer", note="transfer to Ada",
               kind="spend")
    assert daily_transfer_spend(ledger) == 9_000_000


# ── expiry & revocation ───────────────────────────────────────────────

def test_expired_mandate_blocks(store, ledger):
    _issue(store, ttl_days=-1)  # already expired
    result = check_mandate(store, "owner", SCOPE_TRANSFER, 1_000_000,
                           ledger=ledger)
    assert not result.ok
    assert "expired" in result.reason


def test_revoke_kills_instantly(store, ledger):
    m = _issue(store)
    assert check_mandate(store, "owner", SCOPE_TRANSFER, 1_000_000,
                         ledger=ledger).ok
    assert store.revoke(m.id) is True
    result = check_mandate(store, "owner", SCOPE_TRANSFER, 1_000_000,
                           ledger=ledger)
    assert not result.ok
    assert "revoked" in result.reason


def test_revoke_all_stops_all_spending(store, ledger):
    _issue(store, scope="transfer")
    _issue(store, scope="all")
    assert store.revoke_all("owner") == 2
    result = check_mandate(store, "owner", SCOPE_TRANSFER, 1_000_000,
                           ledger=ledger)
    assert not result.ok


def test_revoke_unknown_returns_false(store):
    assert store.revoke("mand_nonexistent") is False


def test_scope_all_covers_transfer(store, ledger):
    _issue(store, scope="all")
    result = check_mandate(store, "owner", SCOPE_TRANSFER, 1_000_000,
                           ledger=ledger)
    assert result.ok


def test_wrong_scope_does_not_cover(store, ledger):
    _issue(store, scope="other_scope")
    result = check_mandate(store, "owner", SCOPE_TRANSFER, 1_000_000,
                           ledger=ledger)
    assert not result.ok


def test_active_prefers_newest(store):
    m1 = _issue(store, cap_per_txn=1_000_000)
    time.sleep(0.01)
    m2 = _issue(store, cap_per_txn=2_000_000)
    active = store.active("owner", SCOPE_TRANSFER)
    assert active.id == m2.id
    assert m1.id != m2.id


# ── dispatch integration: send.py (#49) ───────────────────────────────

def test_send_money_blocked_without_mandate(tmp_path):
    from nomorals.finance.send import RecipientStore, send_money
    ledger = Ledger(tmp_path / "l.jsonl")
    recips = RecipientStore(tmp_path / "r.json")
    recips.add("Mama", account_number="0123456789", bank_code="058")
    empty = MandateStore(tmp_path / "m.json")
    result = send_money("Mama", "5k", ledger=ledger, recipients=recips,
                        mandate_store=empty)
    assert result["ok"] is False
    assert "mandate" in result["error"].lower()


def test_send_money_proceeds_with_mandate(tmp_path):
    from nomorals.finance.send import RecipientStore, send_money
    ledger = Ledger(tmp_path / "l.jsonl")
    recips = RecipientStore(tmp_path / "r.json")
    recips.add("Mama", account_number="0123456789", bank_code="058")
    ms = MandateStore(tmp_path / "m.json")
    ms.issue(principal="owner", scope="transfer", cap_per_txn=10**12,
             cap_per_day=10**13, ttl_days=30)
    result = send_money("Mama", "5k", ledger=ledger, recipients=recips,
                        mandate_store=ms, override_warning=True)
    assert result.get("needs") == "biometric"


def test_confirm_send_blocked_after_revoke(tmp_path):
    from nomorals.finance.send import (
        RecipientStore, confirm_send, send_money,
    )
    ledger = Ledger(tmp_path / "l.jsonl")
    recips = RecipientStore(tmp_path / "r.json")
    recips.add("Mama", account_number="0123456789", bank_code="058")
    ms = MandateStore(tmp_path / "m.json")
    m = ms.issue(principal="owner", scope="transfer", cap_per_txn=10**12,
                 cap_per_day=10**13, ttl_days=30)
    staged = send_money("Mama", "5k", ledger=ledger, recipients=recips,
                        mandate_store=ms, override_warning=True)
    assert staged.get("needs") == "biometric"
    # Revoke between staging and confirmation — movement-time check catches it
    ms.revoke(m.id)
    result = confirm_send(staged["staged_id"], "tok_test", ledger=ledger,
                          mandate_store=ms)
    assert result["ok"] is False
    assert "mandate" in result["error"].lower()


# ── dispatch integration: paystack (#65) ──────────────────────────────

def test_paystack_transfer_blocked_without_mandate():
    from nomorals.connectors.base import ConnectorError
    from nomorals.connectors.paystack import PaystackConnector
    from nomorals.accounts.vault import CredentialVault
    from nomorals.storage.db import Database
    import tempfile, os
    v = CredentialVault(Database(":memory:"), master_passphrase="test")
    conn = PaystackConnector(v)
    ms = MandateStore(os.path.join(tempfile.mkdtemp(), "m.json"))
    with pytest.raises(ConnectorError) as ctx:
        conn.initiate_transfer(250_000, "RCP_abc", confirmed=True,
                               mandate_store=ms)
    assert "mandate" in str(ctx.value).lower()


def test_issue_mandate_convenience(tmp_path):
    m = issue_mandate(MandateStore(tmp_path / "m.json"),
                      principal="owner", scope="transfer",
                      cap_per_txn=1_000_000, cap_per_day=5_000_000)
    assert m.id.startswith("mand_")


def test_mandate_list(tmp_path):
    ms = MandateStore(tmp_path / "m.json")
    _issue(ms)
    _issue(ms)
    assert len(ms.list("owner")) == 2
    assert len(ms.list("someone_else")) == 0

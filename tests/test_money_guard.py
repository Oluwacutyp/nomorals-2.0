"""Conversational money primitive + pre-transaction warnings (build-map #49)."""
import time

import pytest

from nomorals.finance.guard import (
    Warning,
    check_outgoing,
    known_recipients,
    median_outgoing,
)
from nomorals.finance.ledger import Ledger, format_naira
from nomorals.finance.send import (
    RecipientStore,
    confirm_send,
    confirm_send_otp,
    parse_send_request,
    resolve_recipient,
    send_money,
)


@pytest.fixture()
def ledger(tmp_path):
    return Ledger(tmp_path / "ledger.jsonl")


@pytest.fixture()
def recipients(tmp_path):
    store = RecipientStore(tmp_path / "recipients.json")
    store.add("Mama", account_number="0123456789", bank_code="058",
              bank_name="GTBank")
    return store


@pytest.fixture()
def mandates(tmp_path):
    """Generous test mandate — #69 requires one for any money movement."""
    from nomorals.finance.mandate import MandateStore
    store = MandateStore(tmp_path / "mandates.json")
    store.issue(principal="owner", scope="transfer",
                cap_per_txn=100_000_000_000, cap_per_day=1_000_000_000_000,
                ttl_days=30)
    return store


def _seed_history(ledger, n=5, amount_kobo=500_000, to="Mama"):
    """5 × ₦5,000 transfers to Mama — a normal baseline."""
    base = time.time() - 86400 * 10
    for i in range(n):
        ledger.log(amount_kobo, category="transfer",
                   note=f"transfer to {to}", kind="spend",
                   ts=base + i * 86400)


# ── guard: anomaly detection ──────────────────────────────────────────────

def test_normal_transaction_no_warning(ledger, recipients):
    _seed_history(ledger)
    assert check_outgoing("Mama", 500_000, ledger) is None


def test_large_amount_warns(ledger):
    _seed_history(ledger)  # median ₦5,000
    w = check_outgoing("Mama", 5_000_000, ledger)  # ₦50,000 = 10× median
    assert isinstance(w, Warning)
    assert any("far above" in r for r in w.reasons)


def test_new_recipient_warns(ledger):
    _seed_history(ledger)
    w = check_outgoing("Stranger", 500_000, ledger)
    assert w is not None
    assert any("new recipient" in r for r in w.reasons)


def test_unusual_time_warns(ledger):
    _seed_history(ledger)
    # 3am local
    import datetime
    dt = datetime.datetime(2026, 10, 8, 3, 0)
    w = check_outgoing("Mama", 500_000, ledger, now=dt.timestamp())
    assert w is not None
    assert any("unusual time" in r for r in w.reasons)


def test_scam_pattern_warns(ledger):
    _seed_history(ledger)
    w = check_outgoing("Stranger", 5_000_000, ledger)  # ₦50k round → new
    assert w is not None
    assert any("scam pattern" in r for r in w.reasons)


def test_thin_history_no_amount_warning(ledger):
    # Fresh ledger — no baseline, so no amount signal (but new-recipient
    # still fires since there are no known recipients).
    w = check_outgoing("Mama", 100_000_000, ledger)
    assert w is None or not any("far above" in r for r in w.reasons)


def test_warning_message_format(ledger):
    _seed_history(ledger)
    w = check_outgoing("Stranger", 5_000_000, ledger)
    assert w.message.startswith("⚠️")
    assert "₦50,000" in w.message
    assert "Stranger" in w.message
    assert "Sure?" in w.message


def test_guard_never_raises():
    assert check_outgoing("x", 100, None) is None  # broken ledger


def test_known_recipients_extraction(ledger):
    _seed_history(ledger, to="Mama")
    assert "mama" in known_recipients(ledger)


def test_median_outgoing(ledger):
    _seed_history(ledger, n=5, amount_kobo=500_000)
    assert median_outgoing(ledger) == 500_000


# ── send primitive ────────────────────────────────────────────────────────

def test_parse_send_request():
    req = parse_send_request("send 5k to Mama")
    assert req is not None
    assert req["amount_kobo"] == 500_000
    assert req["to"].lower() == "mama"


def test_parse_send_request_no_match():
    assert parse_send_request("what's the weather?") is None


def test_unknown_recipient_asks(ledger, recipients, tmp_path):
    store = RecipientStore(tmp_path / "empty.json")
    result = send_money("Nobody", 500_000, ledger=ledger, recipients=store)
    assert result["ok"] is False
    assert result["needs"] == "recipient"
    assert "account number" in result["ask"].lower()


def test_known_recipient_stages_for_biometric(ledger, recipients, tmp_path, mandates):
    _seed_history(ledger)
    result = send_money("Mama", "5k", ledger=ledger, recipients=recipients,
                        mandate_store=mandates)
    assert result["needs"] == "biometric"
    assert "fingerprint" in result["prompt"].lower()
    assert result["staged_id"]


def test_warning_blocks_until_override(ledger, recipients, mandates):
    _seed_history(ledger)
    result = send_money("Mama", 5_000_000, ledger=ledger, recipients=recipients,
                        mandate_store=mandates)
    assert result["ok"] is False
    assert "warning" in result
    assert result["warning"]["message"].startswith("⚠️")
    # Explicit override → proceeds to biometric staging
    result2 = send_money("Mama", 5_000_000, ledger=ledger,
                         recipients=recipients, override_warning=True,
                         mandate_store=mandates)
    assert result2["needs"] == "biometric"


def test_confirm_send_requires_biometric_token(ledger, recipients, mandates):
    _seed_history(ledger)
    staged = send_money("Mama", "5k", ledger=ledger, recipients=recipients,
                        mandate_store=mandates)
    assert staged.get("needs") == "biometric"
    result = confirm_send(staged["staged_id"], None, ledger=ledger)
    assert result["ok"] is False
    assert "biometric" in result["error"].lower()


def test_confirm_send_fails_closed_without_connector(ledger, recipients, mandates):
    _seed_history(ledger)
    staged = send_money("Mama", "5k", ledger=ledger, recipients=recipients,
                        mandate_store=mandates)
    assert staged.get("needs") == "biometric"
    result = confirm_send(staged["staged_id"], "tok_test", ledger=ledger,
                          mandate_store=mandates,
                          paystack=None)
    assert result["ok"] is False
    assert "did not move" in result["error"]


def test_confirm_send_executes_via_paystack(ledger, recipients, mandates):
    _seed_history(ledger)
    staged = send_money("Mama", "5k", ledger=ledger, recipients=recipients,
                        mandate_store=mandates)
    assert staged.get("needs") == "biometric"

    class FakePaystack:
        def create_transfer_recipient(self, account_number, bank_code, **kw):
            assert account_number == "0123456789"
            return {"recipient_code": "RCP_test"}

        def initiate_transfer(self, amount_kobo, recipient_code, **kw):
            assert amount_kobo == 500_000
            assert recipient_code == "RCP_test"
            assert kw.get("confirmed") is True
            assert kw.get("biometric_token") == "tok_test"
            return {"status": "success", "reference": "TRF_test123"}

    result = confirm_send(staged["staged_id"], "tok_test", ledger=ledger,
                          mandate_store=mandates,
                          paystack=FakePaystack(), recipients=recipients)
    assert result["ok"] is True
    assert result["reference"] == "TRF_test123"
    # Ledger has the transfer
    txns = ledger.transactions(category="transfer")
    assert any(t.note == "transfer to Mama" for t in txns)


def test_confirm_send_otp_flow(ledger, recipients, mandates):
    _seed_history(ledger)
    staged = send_money("Mama", "5k", ledger=ledger, recipients=recipients,
                        mandate_store=mandates)
    assert staged.get("needs") == "biometric"

    class FakePaystack:
        def create_transfer_recipient(self, *a, **kw):
            return {"recipient_code": "RCP_test"}

        def initiate_transfer(self, *a, **kw):
            return {"status": "otp", "transfer_code": "TRF_otp1",
                    "reference": "ref_otp1"}

        def finalize_transfer(self, transfer_code, otp):
            assert transfer_code == "TRF_otp1"
            assert otp == "123456"
            return {"status": "success", "reference": "ref_otp1"}

    result = confirm_send(staged["staged_id"], "tok_test", ledger=ledger,
                          mandate_store=mandates,
                          paystack=FakePaystack(), recipients=recipients)
    assert result["ok"] is False
    assert result["needs"] == "otp"
    done = confirm_send_otp(staged["staged_id"], result["transfer_code"],
                            "123456", ledger=ledger,
                            paystack=FakePaystack())
    assert done["ok"] is True
    assert done["reference"] == "ref_otp1"


def test_audit_trail_logged(ledger, recipients, mandates):
    _seed_history(ledger)
    send_money("Mama", 5_000_000, ledger=ledger, recipients=recipients,
             mandate_store=mandates)
    audits = ledger.transactions(category="transfer_audit")
    assert any("warning_shown" in t.note for t in audits)


def test_recipient_store_validation(tmp_path):
    store = RecipientStore(tmp_path / "r.json")
    with pytest.raises(ValueError):
        store.add("", account_number="123", bank_code="058")
    with pytest.raises(ValueError):
        store.add("X", account_number="abc", bank_code="058")
    rec = store.add("Mama", account_number="0123456789", bank_code="058")
    assert store.get("mama")["account_number"] == "0123456789"
    assert store.get("MAMA") is not None  # case-insensitive

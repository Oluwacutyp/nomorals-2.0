"""Tests for nomorals.voice.money — multilingual voice money commands.

All offline. The Ekiti/Pidgin/Hausa/Yoruba examples use the documented
phrase patterns from DIALECT_NOTES — pattern coverage, not native-speaker
fluency claims. No audio, no network, no real money movement.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nomorals.voice import money as vm
from nomorals.voice.money import (
    MoneyIntent,
    TransferStaging,
    detect_language,
    handle_voice_money,
    parse_voice_money,
    transcribe_money_voice,
    voice_money_precheck,
)


def _ctx(tmp_path):
    return SimpleNamespace(settings=SimpleNamespace(home_path=str(tmp_path)))


# ── parsing: kinds, amounts, languages ───────────────────────────────────────

def test_parse_english_expense():
    i = parse_voice_money("I spent 5k on data")
    assert i.kind == "log_expense"
    assert i.amount_kobo == 500_000
    assert i.category == "data"
    assert i.language == "en"
    assert i.raw_transcript == "I spent 5k on data"


def test_parse_pidgin_expense():
    i = parse_voice_money("I don spend 2k for fuel")
    assert i.kind == "log_expense"
    assert i.amount_kobo == 200_000
    assert i.category == "transport"
    assert i.language == "pidgin"


def test_parse_ekiti_expense():
    # Subject-position "mi" (Ekiti) vs "mo" (Standard) — the documented
    # dialect marker. Undiacritized, as STT output would be.
    i = parse_voice_money("Mi na 5k fun data")
    assert i.kind == "log_expense"
    assert i.amount_kobo == 500_000
    assert i.language == "ekiti"


def test_parse_standard_yoruba_expense():
    i = parse_voice_money("Mo na 5k fun data")
    assert i.kind == "log_expense"
    assert i.language == "yoruba"


def test_possessive_mi_is_not_ekiti():
    # "owo mi" (my money) is the Standard possessive — must not alone
    # trigger the Ekiti label.
    lang, _ = detect_language("owo mi")
    assert lang == "yoruba"


def test_parse_codeswitched_transfer():
    i = parse_voice_money("Abeg send 5k to Mama jare")
    assert i.kind == "transfer"
    assert i.language == "mixed"
    assert i.amount_kobo == 500_000
    assert i.recipient == "Mama"


def test_parse_hausa_transfer():
    i = parse_voice_money("Aika 3k zuwa Mama")
    assert i.kind == "transfer"
    assert i.language == "hausa"
    assert i.amount_kobo == 300_000
    assert i.recipient == "Mama"


def test_parse_ekiti_transfer():
    i = parse_voice_money("Mi fe ran 5k si Mama")
    assert i.kind == "transfer"
    assert i.language == "ekiti"
    assert i.recipient == "Mama"


def test_fun_data_is_category_not_recipient():
    # "fún data" must not produce a person-recipient.
    i = parse_voice_money("Mi na 5k fun data")
    assert i.kind == "log_expense"
    assert i.recipient is None


def test_pidgin_na_particle_does_not_reroute_transfer():
    # Trailing "na" (Pidgin emphasis) must not turn a transfer into
    # an expense.
    i = parse_voice_money("Abeg send 5k to Mama na")
    assert i.kind == "transfer"
    assert i.recipient == "Mama"


def test_parse_balance_english():
    i = parse_voice_money("How much have I spent this month")
    assert i.kind == "check_balance"
    assert i.amount_kobo is None


def test_parse_balance_pidgin():
    i = parse_voice_money("How much I don spend this month")
    assert i.kind == "check_balance"


def test_parse_unknown():
    i = parse_voice_money("What is the weather like today")
    assert i.kind == "unknown"
    assert i.amount_kobo is None
    assert i.note == "What is the weather like today"


@pytest.mark.parametrize("text", ["", "   ", "!!!", "k", "send", "12345"])
def test_never_raises_on_weird_input(text):
    i = parse_voice_money(text)
    assert isinstance(i, MoneyIntent)
    # "12345" has an amount but no money verb → unknown, never a guess.
    assert i.kind == "unknown"


def test_amount_variants():
    assert parse_voice_money("I spent ₦1,500 on food").amount_kobo == 150_000
    assert parse_voice_money("I spent 2.5m on rent").amount_kobo == 250_000_000
    assert parse_voice_money("I spent 2000 naira on transport").amount_kobo == 200_000


def test_send_me_recipient():
    i = parse_voice_money("Abeg send me 5k")
    assert i.kind == "transfer"
    assert i.recipient == "me"


# ── routing: ledger writes, balance, biometric-gated transfers ───────────────

def test_handle_log_expense_writes_ledger(tmp_path):
    ctx = _ctx(tmp_path)
    reply = handle_voice_money("I spent 5k on data", ctx)
    assert "₦5,000" in reply
    from nomorals.finance.ledger import Ledger
    from nomorals.finance.budgets import finance_paths
    ledger_path, _ = finance_paths(ctx.settings)
    txns = Ledger(ledger_path).transactions()
    assert len(txns) == 1
    assert txns[0].amount_kobo == 500_000
    assert txns[0].category == "data"
    assert txns[0].source == "voice"


def test_handle_log_expense_ekiti_reply(tmp_path):
    ctx = _ctx(tmp_path)
    reply = handle_voice_money("Mi na 5k fun data", ctx)
    # Ekiti reply uses the "mi" subject form (documented dialect marker).
    assert reply.startswith("Mi ti kọ")


def test_handle_balance(tmp_path):
    from nomorals.finance.ledger import Ledger
    from nomorals.finance.budgets import finance_paths
    ctx = _ctx(tmp_path)
    ledger_path, _ = finance_paths(ctx.settings)
    Ledger(ledger_path).log(500_000, note="data", kind="spend")
    reply = handle_voice_money("How much have I spent this month", ctx)
    assert "₦5,000" in reply


def test_transfer_denied_without_biometric(tmp_path):
    ctx = _ctx(tmp_path)
    with patch.object(vm, "approve_with_biometric", return_value=None):
        reply = handle_voice_money("Abeg send 5k to Mama jare", ctx)
    assert "no go move" in reply.lower() or "not confirm" in reply.lower()
    import json
    path = tmp_path / "finance" / "staged_transfers.jsonl"
    recs = [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(recs) == 1
    assert recs[0]["status"] == "denied"
    assert recs[0]["biometric_token"] is None
    assert recs[0]["recipient"] == "Mama"
    assert recs[0]["amount_kobo"] == 500_000
    # Denied records are not pending action.
    assert TransferStaging(path).pending() == []


def test_transfer_staged_with_biometric(tmp_path):
    ctx = _ctx(tmp_path)
    with patch.object(vm, "approve_with_biometric", return_value="tok_abc"):
        reply = handle_voice_money("Abeg send 5k to Mama jare", ctx)
    assert "₦5,000" in reply and "Mama" in reply
    staging = TransferStaging(
        tmp_path / "finance" / "staged_transfers.jsonl")
    recs = staging.pending()
    assert len(recs) == 1
    assert recs[0].status == "biometric_confirmed"
    assert recs[0].biometric_token == "tok_abc"


def test_transfer_needs_details(tmp_path):
    ctx = _ctx(tmp_path)
    reply = handle_voice_money("send money to Mama", ctx)
    assert "Mama" in reply or "how much" in reply.lower()


def test_handle_unknown_replies_helpfully(tmp_path):
    ctx = _ctx(tmp_path)
    reply = handle_voice_money("tell me a story", ctx)
    assert "5k" in reply  # points at an example money phrase


# ── bridge pre-check ─────────────────────────────────────────────────────────

def test_precheck_falls_through_on_non_money(tmp_path):
    ctx = _ctx(tmp_path)
    assert voice_money_precheck("send me that file", ctx) is None
    assert voice_money_precheck("what is the weather", ctx) is None
    assert voice_money_precheck("", ctx) is None


def test_precheck_handles_confident_money(tmp_path):
    ctx = _ctx(tmp_path)
    reply = voice_money_precheck("I spent 5k on data", ctx)
    assert reply is not None
    assert "₦5,000" in reply


def test_precheck_transfer_without_amount_falls_through(tmp_path):
    # No amount to act on — let the brain ask for it instead of guessing.
    ctx = _ctx(tmp_path)
    assert voice_money_precheck("send money to Mama", ctx) is None


# ── STT entry ────────────────────────────────────────────────────────────────

class _FakeSTT:
    def __init__(self, text="", fail=False):
        self.text = text
        self.fail = fail
        self.seen_language = "unset"

    def transcribe(self, path, language="en"):
        self.seen_language = language
        if self.fail:
            raise RuntimeError("mic dead")
        return {"text": self.text, "language": "yo", "backend": "fake"}


def test_transcribe_money_voice_autodetect():
    stt = _FakeSTT("Mi na 5k fun data")
    intent = transcribe_money_voice("/tmp/nonexistent.wav", stt)
    # Auto-detect: language=None, not forced "en" — better for code-switching.
    assert stt.seen_language is None
    assert intent.kind == "log_expense"
    assert intent.language == "ekiti"


def test_transcribe_money_voice_failure_is_unknown():
    intent = transcribe_money_voice("/tmp/x.wav", _FakeSTT(fail=True))
    assert intent.kind == "unknown"


def test_transcribe_money_voice_callable():
    intent = transcribe_money_voice(
        "/tmp/x.wav", lambda path, language=None: "I don spend 2k for fuel")
    assert intent.kind == "log_expense"
    assert intent.language == "pidgin"


# ── staging store ────────────────────────────────────────────────────────────

def test_staging_mark_unknown_id(tmp_path):
    s = TransferStaging(tmp_path / "staged.jsonl")
    assert s.mark("nope", "denied") is False
    assert s.pending() == []

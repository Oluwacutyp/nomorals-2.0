"""Wiring tests: MORE_TO_ADD gap 9 — the subscription auditor reaches production.

The domain module (``nomorals/finance/subscriptions.py``) was unit-pinned
by ``tests/test_more_gaps710.py`` §9 but had zero production importers:
no chat command, no tool, no usage-record store, and its
``_txn_fields`` docstring promised Ledger.Transaction-shaped objects
it could not actually read (``amount_kobo``/``ts``). These tests pin
the wiring: durable usage records, the Ledger adapter, the audit
shared by the chat command and the tool, and the command registry.
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from nomorals.finance.commands import control_subscriptions
from nomorals.finance.ledger import Ledger, Transaction
from nomorals.finance.subscriptions import (
    UsageStore,
    audit,
    detect_subscriptions,
    render_audit,
    render_reminders,
)

NETFLIX = 650_000  # ₦6,500 in kobo


def _ctx(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(settings=SimpleNamespace(home_path=tmp_path))


def _ledger_with_netflix(tmp_path: Path) -> Ledger:
    """Three monthly Netflix charges (-85/-55/-25 days) + one one-off."""
    led = Ledger(tmp_path / "finance" / "ledger.jsonl")
    now = time.time()
    for days_ago in (85, 55, 25):
        led.log(NETFLIX, note="Netflix", kind="spend",
                ts=now - days_ago * 86400)
    led.log(300_000, note="Mama Put", kind="spend", ts=now - 2 * 86400)
    return led


# ── UsageStore ────────────────────────────────────────────────────────────

def test_usage_store_round_trip(tmp_path: Path) -> None:
    store = UsageStore(tmp_path / "usage.json")
    store.record("Netflix", date(2026, 10, 15))
    assert UsageStore(tmp_path / "usage.json").last_used("netflix") == date(2026, 10, 15)
    assert UsageStore(tmp_path / "usage.json").last_used("NETFLIX ") == date(2026, 10, 15)


def test_usage_store_tolerant_load(tmp_path: Path) -> None:
    bad = tmp_path / "usage.json"
    bad.write_text("{not json", encoding="utf-8")
    assert UsageStore(bad).all() == {}
    bad.write_text(json.dumps({"netflix": "2026-10-15", "junk": "not-a-date",
                               "also-bad": 42}), encoding="utf-8")
    assert UsageStore(bad).all() == {"netflix": date(2026, 10, 15)}


# ── Ledger adapter ────────────────────────────────────────────────────────

def test_txn_fields_accepts_ledger_transaction() -> None:
    now = time.time()
    txns = [
        Transaction(ts=now - 60 * 86400, amount_kobo=NETFLIX,
                    category="entertainment", note="Netflix"),
        Transaction(ts=now - 30 * 86400, amount_kobo=NETFLIX,
                    category="entertainment", note="Netflix"),
    ]
    subs = detect_subscriptions(txns, as_of=date.today())
    assert [s.merchant for s in subs] == ["Netflix"]
    assert subs[0].cost_minor == NETFLIX


def test_audit_detects_from_ledger(tmp_path: Path) -> None:
    led = _ledger_with_netflix(tmp_path)
    subs, reminders = audit(led.transactions(kind="spend"))
    assert [s.merchant for s in subs] == ["Netflix"]
    sub = subs[0]
    assert sub.occurrences == 3 and sub.cost_minor == NETFLIX
    assert sub.estimated_next_renewal is not None
    assert 0 <= (sub.estimated_next_renewal - date.today()).days <= 7
    assert len(reminders) == 1
    assert reminders[0].merchant == "Netflix"
    assert "cancel my subscription" in reminders[0].draft_message.lower()


def test_audit_unused_flag_needs_usage(tmp_path: Path) -> None:
    led = _ledger_with_netflix(tmp_path)
    subs, _ = audit(led.transactions(kind="spend"))
    assert subs[0].suspected_unused is True
    assert "no usage records" in subs[0].unused_reason
    store = UsageStore(tmp_path / "usage.json")
    store.record("Netflix", date.today())
    subs, _ = audit(led.transactions(kind="spend"),
                    usage_records=store.all())
    assert subs[0].suspected_unused is False


# ── Chat handler ──────────────────────────────────────────────────────────

def test_handler_empty_ledger_is_honest(tmp_path: Path) -> None:
    text = control_subscriptions("", _ctx(tmp_path))
    assert "no recurring subscriptions" in text.lower()
    assert "365" in text  # the window is named, never hidden


def test_handler_lists_detected(tmp_path: Path) -> None:
    _ledger_with_netflix(tmp_path)
    text = control_subscriptions("", _ctx(tmp_path))
    assert "Netflix" in text
    assert "6,500" in text
    assert "suspected unused" in text.lower()
    assert "monthly" in text.lower()


def test_handler_used_clears_flag(tmp_path: Path) -> None:
    _ledger_with_netflix(tmp_path)
    ctx = _ctx(tmp_path)
    assert "recorded" in control_subscriptions("used Netflix", ctx).lower()
    assert "suspected unused" not in control_subscriptions("", ctx).lower()


def test_handler_cancel_is_draft_only(tmp_path: Path) -> None:
    _ledger_with_netflix(tmp_path)
    ctx = _ctx(tmp_path)
    text = control_subscriptions("cancel Netflix", ctx)
    assert "cancel my subscription" in text.lower()
    assert "Netflix" in text
    assert "no subscription" in control_subscriptions("cancel Nobody", ctx).lower()
    assert "usage:" in control_subscriptions("used Netflix 2026-13-40", ctx).lower()


def test_handler_upcoming_window(tmp_path: Path) -> None:
    _ledger_with_netflix(tmp_path)
    text = control_subscriptions("upcoming", _ctx(tmp_path))
    assert "Netflix" in text
    assert "nothing renewing" in control_subscriptions("upcoming 0", _ctx(tmp_path)).lower()
    assert render_reminders([]).lower().startswith("nothing renewing")


# ── Registry + tool ───────────────────────────────────────────────────────

def test_registry_membership() -> None:
    from nomorals.social.chat.control import CONTROL_COMMANDS, PUBLIC_COMMANDS

    assert "subscriptions" in CONTROL_COMMANDS
    assert "subscriptions" not in PUBLIC_COMMANDS  # owner-only like /spending


def test_tool_registered_and_lists(tmp_path: Path) -> None:
    from nomorals.finance import tools as tools_mod

    class _CaptureRegistry:
        def __init__(self) -> None:
            self.fns: dict = {}

        def register(self, name, **kwargs):
            def deco(fn):
                self.fns[name] = fn
                return fn
            return deco

    cap = _CaptureRegistry()
    tools_mod.register(cap)
    assert "finance_subscriptions" in cap.fns
    _ledger_with_netflix(tmp_path)
    out = cap.fns["finance_subscriptions"](_ctx(tmp_path), action="list")
    assert out["ok"] is True
    assert out["count"] == 1
    assert "Netflix" in out["text"]
    assert out["charges"][0]["merchant"] == "Netflix"
    assert render_audit([]).lower().startswith("no recurring subscriptions")

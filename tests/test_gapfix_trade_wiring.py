"""Wiring tests: audit gap R1-1 — /trade, /recurring, /import reach production.

The handlers (``nomorals/finance/commands.py``: ``control_trade``,
``control_recurring``, ``control_import``) were complete but had zero
dispatch branches and zero ``control.py`` entries: all three answered
"unknown command". These tests pin the wiring: registry entries, dispatch
routing, the owner gate, and the real handlers against a temp-home Ledger.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from nomorals.finance.commands import (
    control_import,
    control_recurring,
    control_trade,
)
from nomorals.finance.ledger import Ledger
from nomorals.social.chat.base import ChatKind, ChatRef
from nomorals.social.chat.control import (
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    LIST_GROUPS,
    LIST_ONELINERS,
    PUBLIC_COMMANDS,
)

KINDS = ("trade", "recurring", "import")
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


# ── Registry ──────────────────────────────────────────────────────────────

def test_kinds_registered_and_owner_only() -> None:
    for kind in KINDS:
        assert kind in CONTROL_COMMANDS, f"/{kind} not registered"
        assert kind not in PUBLIC_COMMANDS, f"/{kind} must stay owner-only"
        page = COMMAND_DETAILS.get(kind)
        assert page is not None, f"/{kind} has no COMMAND_DETAILS page"
        for field in ("what", "usage", "example"):
            assert str(page.get(field, "")).strip(), \
                f"/{kind} DETAILS.{field} is empty"
        assert "run /" + kind + "." not in page["what"], \
            f"/{kind} has a placeholder help page"
        assert str(LIST_ONELINERS.get(kind, "")).strip(), \
            f"/{kind} has no LIST_ONELINERS entry"
        slots = [names for _t, names in LIST_GROUPS if kind in names]
        assert len(slots) == 1, f"/{kind} must sit in exactly one list group"


# ── Dispatch ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import load_settings
    from nomorals.social.chat.base import (
        ChatAdapter,
        SendResult,
    )
    from nomorals.social.chat.gateway import ChatGateway

    class FakeAdapter(ChatAdapter):
        def __init__(self, name: str) -> None:
            super().__init__(media_dir="/tmp/nm-test-media")
            self.name = name

        def run(self, handler) -> None:
            pass

        def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
                 buttons=None, **kwargs) -> SendResult:
            return SendResult(ok=True, platform=self.name,
                              message_id="m1")

    tmp = tmp_path_factory.mktemp("gapfix-trade")
    settings = load_settings(overrides={
        "home": str(tmp),
        "partner.platforms": "telegram",
        "partner.owner_chats": "5478650254",
        "partner.owner_sender_ids": "5478650254",
    })
    context = build_context(settings, with_executor=False, with_tools=False)
    adapter = FakeAdapter("telegram")
    gateway = ChatGateway({"telegram": adapter}, db=context.db)
    rt = PartnerRuntime(context, gateway=gateway)
    yield rt
    rt.stop()


def test_dispatch_trade_reaches_handler(runtime) -> None:
    with mock.patch("nomorals.finance.commands.control_trade",
                    return_value="TRADE-SENTINEL") as fn:
        out = runtime.handle_control("/trade stats", "telegram:1",
                                     message=None)
    assert out == "TRADE-SENTINEL"
    fn.assert_called_once()
    assert fn.call_args[0][0] == "stats"


def test_dispatch_recurring_reaches_handler(runtime) -> None:
    with mock.patch("nomorals.finance.commands.control_recurring",
                    return_value="RECURRING-SENTINEL") as fn:
        out = runtime.handle_control("/recurring upcoming", "telegram:1",
                                     message=None)
    assert out == "RECURRING-SENTINEL"
    fn.assert_called_once()
    assert fn.call_args[0][0] == "upcoming"


def test_dispatch_import_reaches_handler(runtime) -> None:
    with mock.patch("nomorals.finance.commands.control_import",
                    return_value="IMPORT-SENTINEL") as fn:
        out = runtime.handle_control("/import /tmp/x.csv", "telegram:1",
                                     message=None)
    assert out == "IMPORT-SENTINEL"
    fn.assert_called_once()
    assert fn.call_args[0][0] == "/tmp/x.csv"


def test_dispatch_denies_stranger(runtime) -> None:
    from nomorals.social.chat.base import ChatMessage

    chat = ChatRef(platform="telegram", chat_id="999",
                   kind=ChatKind.DM, title="")
    msg = ChatMessage(chat=chat, incoming=True, text="/trade stats",
                      sender="Stranger", sender_id="999999",
                      message_id="m-stranger", ts=time.time())
    out = runtime.handle_control("/trade stats", chat.key, message=msg)
    assert "just for the owner" in out


# ── /recurring handler ────────────────────────────────────────────────────

def test_recurring_empty_ledger_is_honest(tmp_path: Path) -> None:
    text = control_recurring("", _ctx(tmp_path))
    assert "no recurring charges detected yet" in text


def test_recurring_lists_detected(tmp_path: Path) -> None:
    _ledger_with_netflix(tmp_path)
    text = control_recurring("", _ctx(tmp_path))
    assert "Netflix" in text
    assert "monthly" in text.lower()


def test_recurring_upcoming_names_due(tmp_path: Path) -> None:
    _ledger_with_netflix(tmp_path)
    text = control_recurring("upcoming", _ctx(tmp_path))
    assert "Netflix" in text


def test_recurring_bad_verb_is_usage(tmp_path: Path) -> None:
    assert "usage:" in control_recurring("bogus", _ctx(tmp_path)).lower()


def test_recurring_sync_watches_bills(tmp_path: Path) -> None:
    from nomorals.finance.alerts import AlertStore as RealAlertStore

    _ledger_with_netflix(tmp_path)
    store_path = tmp_path / "alerts.json"
    with mock.patch("nomorals.finance.alerts.AlertStore",
                    lambda path=None: RealAlertStore(store_path)):
        text = control_recurring("sync", _ctx(tmp_path))
    assert "watching 1 recurring charge(s)" in text
    assert "1 new bill alert(s)" in text


# ── /trade handler ────────────────────────────────────────────────────────

class _FakeDesk:
    """Stands in for TradingDesk: the handler's verb routing runs for real."""

    def __init__(self) -> None:
        self.policy = SimpleNamespace(max_risk_pct_per_trade=1.0)
        self.positions: list[Any] = []

    def render_stats(self) -> str:
        return "paper stats here"

    def paper_positions(self) -> list[Any]:
        return list(self.positions)

    def portfolio_heat(self) -> dict[str, Any]:
        return {"within_limits": True, "heat_pct": 2.0,
                "heat_amount": 200.0, "equity": 10000.0,
                "open_positions": len(self.positions)}

    def kelly_fraction(self) -> dict[str, Any]:
        return {"ok": True, "kelly_pct": 10.0,
                "quarter_kelly_pct": 2.5,
                "note": "quarter-Kelly is the sane default"}

    def equity(self) -> float:
        return 10000.0

    def contract_specs(self, instrument: str) -> dict[str, Any]:
        return {"volume_min": "0.01", "volume_max": "100.0",
                "volume_step": "0.01"}

    def _contract_value(self, instrument: str) -> float:
        return 100.0

    def paper_open(self, instrument: str, side: str, volume: float, *,
                   stop_loss=None, take_profit=None) -> SimpleNamespace:
        pos = SimpleNamespace(id="P1", side=side, volume=volume,
                              entry=2600.0)
        self.positions.append(pos)
        return pos

    def paper_close(self, position_id: str) -> dict[str, Any]:
        return {"position_id": position_id, "pnl": 12.5,
                "r_multiple": 1.5, "reason": "manual close"}


def _trade_ctx(tmp_path: Path, desk: Any) -> SimpleNamespace:
    ctx = _ctx(tmp_path)
    return ctx


def test_trade_bare_is_usage(tmp_path: Path) -> None:
    with mock.patch("nomorals.finance.commands._trade_desk",
                    return_value=_FakeDesk()):
        text = control_trade("", _ctx(tmp_path))
    assert "usage:" in text.lower()
    assert "/trade stats" in text


def test_trade_stats_positions_heat_kelly(tmp_path: Path) -> None:
    desk = _FakeDesk()
    with mock.patch("nomorals.finance.commands._trade_desk",
                    return_value=desk):
        assert control_trade("stats", _ctx(tmp_path)) == "paper stats here"
        assert "no open paper positions" in \
            control_trade("positions", _ctx(tmp_path))
        heat = control_trade("heat", _ctx(tmp_path))
        assert "portfolio heat: 2.0%" in heat
        assert "within the 6% rule" in heat
        kelly = control_trade("kelly", _ctx(tmp_path))
        assert "Kelly: 10.0% full / 2.5% quarter-Kelly" in kelly


def test_trade_positions_lists_open(tmp_path: Path) -> None:
    desk = _FakeDesk()
    desk.positions.append(SimpleNamespace(
        id="P7", side="buy", volume=0.5, instrument="XAUUSD",
        entry=2600.0, stop_loss=2590.0, take_profit=2700.0,
        risk_amount=100.0))
    with mock.patch("nomorals.finance.commands._trade_desk",
                    return_value=desk):
        text = control_trade("positions", _ctx(tmp_path))
    assert "P7" in text and "XAUUSD" in text


def test_trade_size_open_close(tmp_path: Path) -> None:
    desk = _FakeDesk()
    with mock.patch("nomorals.finance.commands._trade_desk",
                    return_value=desk):
        sized = control_trade("size XAUUSD 2600 2590", _ctx(tmp_path))
        assert "size for XAUUSD: 0.1 lots" in sized
        opened = control_trade("open XAUUSD buy 0.5 --sl 2590 --tp 2700",
                               _ctx(tmp_path))
        assert "paper opened: P1 buy 0.5 XAUUSD @ 2600.0" in opened
        closed = control_trade("close P1", _ctx(tmp_path))
        assert "closed P1: +12.50 (+1.50R)" in closed


def test_trade_bad_verb_is_usage(tmp_path: Path) -> None:
    with mock.patch("nomorals.finance.commands._trade_desk",
                    return_value=_FakeDesk()):
        assert "usage:" in control_trade("bogus", _ctx(tmp_path)).lower()


def test_trade_unreachable_desk_is_honest(tmp_path: Path) -> None:
    with mock.patch("nomorals.finance.commands._trade_desk",
                    side_effect=RuntimeError("vault locked")):
        text = control_trade("stats", _ctx(tmp_path))
    assert "couldn't reach the trading desk" in text
    assert "vault locked" in text


def test_trade_desk_refusal_is_named(tmp_path: Path) -> None:
    from nomorals.finance.trading_desk import DeskError

    class _RefusingDesk(_FakeDesk):
        def render_stats(self) -> str:
            raise DeskError("no journal yet")

    with mock.patch("nomorals.finance.commands._trade_desk",
                    return_value=_RefusingDesk()):
        text = control_trade("stats", _ctx(tmp_path))
    assert "desk refused: no journal yet" in text


# ── /import handler ───────────────────────────────────────────────────────

_CSV = ("date,amount,description\n"
        "2026-11-01,-6500,Netflix\n"
        "2026-11-02,-3000,Mama Put\n")


def test_import_empty_tail_is_usage(tmp_path: Path) -> None:
    assert "usage:" in control_import("", _ctx(tmp_path)).lower()


def test_import_csv_then_duplicates_skipped(tmp_path: Path) -> None:
    csv_path = tmp_path / "statement.csv"
    csv_path.write_text(_CSV, encoding="utf-8")
    text = control_import(str(csv_path), _ctx(tmp_path))
    assert "imported 2 transaction(s)" in text
    again = control_import(str(csv_path), _ctx(tmp_path))
    assert "imported 0 transaction(s)" in again
    assert "skipped 2 duplicate(s)" in again


def test_import_missing_file_is_honest(tmp_path: Path) -> None:
    text = control_import(str(tmp_path / "nope.csv"), _ctx(tmp_path))
    assert "import failed:" in text

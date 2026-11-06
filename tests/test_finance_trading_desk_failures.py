"""Tail-114 pins: nomorals/finance/trading_desk.py census sites (433, 525, 621).

All three are held as exclusions:

433 — ``daily_pnl`` live-snapshot guard: CONVERTED by the Pass 3
     targeted fix, along exactly the lines this note prescribed.
     ``daily_pnl`` now returns None when the live snapshot fails
     (the unrealized component is unknowable; a journal-only total
     understated the day's loss), and ``_check_risk`` treats None as
     "cannot verify" and refuses — the same fail-closed shape as the
     position-count gate ("snapshot failure != free pass"). The
     display paths render None as "unknown" instead of a number.
     The pin below now asserts the None + the non-recording half;
     the working-snapshot controls are unchanged.
525 — ``paper_open``: ``_contract_value`` raising ``DeskError`` ->
     contract_value stays 0.0 and risk_amount falls back to unit 1.0
     ("paper can still track in price terms", in-code comment).
     Paper-only path; the position records the 0.0 it used.
621 — ``_is_demo_account``: connector raising -> False. Docstring
     prints "Best-effort". In two of three consumers False is the
     fail-CLOSED direction (live_open and the settings gate lose the
     demo exemption and refuse); in ``unlock_live`` the flag it sets
     is still fenced by the connector's own confirmation gate and
     ``settings.trading.live_enabled``.
"""

from __future__ import annotations

from pathlib import Path

from nomorals.finance.trading_desk import DeskError, TradingDesk

NOW = 1_700_000_000.0


class _RaisingConnector:
    def get_snapshot(self):
        raise RuntimeError("snapshot down")

    def get_account_details(self):
        raise RuntimeError("details down")


class _GoodConnector:
    def __init__(self, details=None, positions=None):
        self.details = details if details is not None else {"account": "demo"}
        self._positions = positions if positions is not None else []

    def get_snapshot(self):
        return {"positions": self._positions}

    def get_account_details(self):
        return self.details


def _desk(tmp_path: Path, connector, mode: str, name: str) -> TradingDesk:
    return TradingDesk(connector, mode=mode,
                       journal_path=tmp_path / f"{name}.jsonl",
                       state_path=tmp_path / f"{name}.json",
                       now=lambda: NOW)


# ── 433: daily_pnl live snapshot swallow ─────────────────────────────

def test_433_account_crash_returns_none_not_journal_only_total(tmp_path):
    # Converted (Pass 3): a crashed live snapshot makes the day P&L
    # unknown (None) — never the journal-only total, which understated
    # the loss feeding the daily-loss kill switch.
    desk = _desk(tmp_path, _RaisingConnector(), "live", "a")
    desk._journal("paper_close", pnl=5.0)
    before = desk.journal(limit=100)
    assert desk.daily_pnl() is None
    # non-recording: the read wrote nothing back to the journal
    assert desk.journal(limit=100) == before


def test_433_control_positions_are_added_when_snapshot_works(tmp_path):
    conn = _GoodConnector(positions=[{"profit": 2.5}, {"profit": -1.0}])
    desk = _desk(tmp_path, conn, "live", "b")
    desk._journal("paper_close", pnl=5.0)
    assert desk.daily_pnl() == 6.5


def test_433_empty_desk_returns_zero(tmp_path):
    desk = _desk(tmp_path, _GoodConnector(), "live", "c")
    assert desk.daily_pnl() == 0.0


# ── 525: paper_open contract-value fallback ──────────────────────────

def test_525_contract_value_failure_tracks_in_price_terms(tmp_path):
    desk = _desk(tmp_path, _GoodConnector(), "paper", "d")

    def _boom(instrument):
        raise DeskError("no contract conditions")

    desk._contract_value = _boom
    pos = desk.paper_open("XAUUSD", "buy", 0.5, entry=2000.0,
                          stop_loss=1990.0, take_profit=2030.0)
    assert pos.contract_value == 0.0
    # unit falls back to 1.0: risk = stop distance * 1.0 * volume
    assert pos.risk_amount == round(10.0 * 1.0 * 0.5, 2)


def test_525_control_contract_value_scales_risk(tmp_path):
    desk = _desk(tmp_path, _GoodConnector(), "paper", "e")
    desk._contract_value = lambda instrument: 100.0
    pos = desk.paper_open("XAUUSD", "buy", 0.5, entry=2000.0,
                          stop_loss=1990.0, take_profit=2030.0)
    assert pos.contract_value == 100.0
    assert pos.risk_amount == round(10.0 * 100.0 * 0.5, 2)


# ── 621: _is_demo_account best-effort probe ──────────────────────────

def test_621_connector_crash_returns_false(tmp_path):
    desk = _desk(tmp_path, _RaisingConnector(), "live", "f")
    assert desk._is_demo_account() is False


def test_621_demo_details_return_true(tmp_path):
    conn = _GoodConnector(details={"account": "Exness demo account"})
    desk = _desk(tmp_path, conn, "live", "g")
    assert desk._is_demo_account() is True


def test_621_non_demo_details_return_false(tmp_path):
    conn = _GoodConnector(details={"account": "real mt5"})
    desk = _desk(tmp_path, conn, "live", "h")
    assert desk._is_demo_account() is False

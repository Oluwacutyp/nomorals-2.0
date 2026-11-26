"""Pump desk board — ranked live candidates, open-call progress,
mint watchlist (gapfix126).

Offline: every test seeds dossiers/calls/observations into a temp
desk dir and reads them back through PumpDesk.board_view /
open_view, the pumpdesk tool, the /pump chat handler, and the
``nm pump`` CLI.  No network anywhere: the board's live mcap is
the observations ledger (SOL), never USD.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from nomorals.agents.pumpdesk import chain
from nomorals.agents.pumpdesk.board import (
    parse_board_filters,
    valid_mint,
)
from nomorals.agents.pumpdesk.calls import PumpCall
from nomorals.agents.pumpdesk.desk import PumpDesk


def _mint(seed: int) -> str:
    return chain.b58encode(bytes([(seed + i) % 256 for i in range(32)]))


MINT_CAND = _mint(11)
MINT_GATED = _mint(51)
MINT_COLD = _mint(91)
MINT_THIN = _mint(131)
MINT_OPEN = _mint(171)
MINT_NEAR = _mint(181)
MINT_RISK = _mint(191)
MINT_DOWN = _mint(201)
MINT_NOREAD = _mint(211)


def _dossier(mint: str, symbol: str, *, trades: int = 40,
             buyers: int = 20, velocity: float = 3.0,
             per_trade: float = 0.5, ratio: float = 2.5,
             curve_pct: float = 0.85, serial_rugger: bool = False,
             rug_ratio: float = 0.0, known_outcomes: int = 5) -> dict:
    return {
        "mint": mint, "symbol": symbol, "name": f"{symbol} Token",
        "creator": _mint(7), "created_at": time.time() - 3600,
        "first_seen": time.time() - 3600,
        "trades": {
            "trades": trades, "unique_buyers": buyers,
            "velocity_sol_per_min": velocity,
            "sol_per_trade": per_trade, "buy_sell_ratio": ratio,
        },
        "dev": {"serial_rugger": serial_rugger, "launches": 6,
                "known_outcomes": known_outcomes,
                "rug_ratio": rug_ratio},
        "curve": {
            "curve_pct": curve_pct, "curve_pct_basis": "test",
            "market_cap_sol": 120.0, "price_sol": 0.00012,
            "real_sol": 40.0, "virtual_sol": 70.0,
            "reserve_divergence": 0.0,
            "state": {"virtual_sol_reserves": 70_000_000_000,
                      "real_sol_reserves": 40_000_000_000},
            "graduated": False, "complete_flag": False,
        },
        "authorities": {"mint_authority_live": False,
                        "freeze_authority_live": False,
                        "supply": 10 ** 15, "decimals": 6},
        "holders": {"top1_share": 0.05, "top10_share": 0.2},
        "attention": {}, "smart": {},
    }


def _seed_board_desk(tmp_path) -> PumpDesk:
    desk = PumpDesk(context=None, desk_dir=str(tmp_path))
    desk.store.save_dossier(_dossier(MINT_CAND, "CAND"))
    desk.store.save_dossier(_dossier(MINT_GATED, "GATED",
                                     serial_rugger=True, rug_ratio=0.8))
    desk.store.save_dossier(_dossier(MINT_COLD, "COLD",
                                     velocity=0.1, per_trade=0.05,
                                     ratio=1.0, curve_pct=0.1))
    desk.store.save_dossier(_dossier(MINT_THIN, "THIN",
                                     trades=3, buyers=2))
    return desk


def _seed_open_desk(tmp_path, now: float) -> PumpDesk:
    desk = PumpDesk(context=None, desk_dir=str(tmp_path))
    pub = now - 3600

    def _call(mint, symbol, *, expires_in, scored=None):
        return PumpCall(
            mint=mint, symbol=symbol, call_class="dossier",
            published_at=pub, expires_at=now + expires_in,
            entry_mc_sol=100.0, target_mc_sol=200.0,
            invalidation_mc_sol=55.0, composite=0.84,
            scored=scored, call_id=f"call-{symbol}")

    def _obs(mint, mc_sol, at):
        desk.store.record_observation(mint, at, 1_000_000_000,
                                      int(mc_sol * 1e9), 0.5, "test")

    desk.calls.log.add(_call(MINT_OPEN, "OPEN", expires_in=5 * 3600))
    _obs(MINT_OPEN, 120.0, pub + 600)
    _obs(MINT_OPEN, 150.0, pub + 1800)
    desk.calls.log.add(_call(MINT_NEAR, "NEAR", expires_in=4 * 3600))
    _obs(MINT_NEAR, 190.0, pub + 1800)
    desk.calls.log.add(_call(MINT_RISK, "RISK", expires_in=3 * 3600))
    _obs(MINT_RISK, 60.0, pub + 1800)
    desk.calls.log.add(_call(MINT_DOWN, "DOWN", expires_in=2 * 3600))
    _obs(MINT_DOWN, 90.0, pub + 1800)
    desk.calls.log.add(_call(MINT_NOREAD, "NOREAD", expires_in=3600))
    expired = _call(MINT_CAND, "GONE", expires_in=-10)
    desk.calls.log.add(expired)
    graded = _call(MINT_GATED, "DONE", expires_in=5 * 3600,
                   scored=True)
    desk.calls.log.add(graded)
    return desk


# ── board ranking + filters ───────────────────────────────────────

def test_board_ranks_candidate_first_then_no_call_then_thin(tmp_path):
    desk = _seed_board_desk(tmp_path)
    view = desk.board_view()
    syms = [r["symbol"] for r in view["rows"]]
    assert syms == ["CAND", "COLD", "GATED", "THIN"]
    assert view["total"] == 4
    assert view["rows"][0]["verdict"] == "CALL_CANDIDATE"
    assert view["rows"][3]["verdict"] == "INSUFFICIENT_DATA"


def test_board_only_candidates(tmp_path):
    desk = _seed_board_desk(tmp_path)
    view = desk.board_view(only_candidates=True)
    assert [r["symbol"] for r in view["rows"]] == ["CAND"]


def test_board_min_composite(tmp_path):
    desk = _seed_board_desk(tmp_path)
    view = desk.board_view(min_composite=0.6)
    assert [r["symbol"] for r in view["rows"]] == ["CAND"]


def test_board_max_rug(tmp_path):
    desk = _seed_board_desk(tmp_path)
    view = desk.board_view(max_rug=0.5)
    assert [r["symbol"] for r in view["rows"]] == ["CAND", "COLD",
                                                   "THIN"]
    assert all(r["rug_risk"] <= 0.5 for r in view["rows"])


def test_board_min_momentum(tmp_path):
    desk = _seed_board_desk(tmp_path)
    view = desk.board_view(min_momentum=0.5)
    assert [r["symbol"] for r in view["rows"]] == ["CAND", "GATED"]


def test_board_limit(tmp_path):
    desk = _seed_board_desk(tmp_path)
    view = desk.board_view(limit=2)
    assert len(view["rows"]) == 2
    assert view["total"] == 4


def test_board_watched_pins_first_and_only_watched(tmp_path):
    desk = _seed_board_desk(tmp_path)
    desk.store.watch_mint(MINT_THIN, note="test")
    view = desk.board_view()
    assert view["rows"][0]["symbol"] == "THIN"
    assert view["rows"][0]["watched"] is True
    only = desk.board_view(only_watched=True)
    assert [r["symbol"] for r in only["rows"]] == ["THIN"]


def test_board_rows_are_sol_only(tmp_path):
    desk = _seed_board_desk(tmp_path)
    view = desk.board_view()
    blob = json.dumps(view["rows"]).lower()
    assert "usd" not in blob
    assert "market_cap_usd" not in blob


def test_board_empty_desk(tmp_path):
    desk = PumpDesk(context=None, desk_dir=str(tmp_path))
    view = desk.board_view()
    assert view["rows"] == []
    assert "no dossiers yet" in view["rendered"]


def test_board_render_names_verdict_and_evidence(tmp_path):
    desk = _seed_board_desk(tmp_path)
    rendered = desk.board_view()["rendered"]
    assert "pump board" in rendered
    assert "CALL_CANDIDATE" in rendered
    assert "SOL" in rendered


# ── filter parsing + mint validation ──────────────────────────────

def test_parse_board_filters():
    filters, bad = parse_board_filters(
        "min=0.6 rug=0.3 mom=0.45 candidates watched limit=5")
    assert filters["min_composite"] == 0.6
    assert filters["max_rug"] == 0.3
    assert filters["min_momentum"] == 0.45
    assert filters["only_candidates"] is True
    assert filters["only_watched"] is True
    assert filters["limit"] == 5
    assert bad == []


def test_parse_board_filters_bad_tokens_reported():
    filters, bad = parse_board_filters("min=2 bogus limit=999")
    assert filters["min_composite"] == 0.0  # out of range, ignored
    assert filters["limit"] == 50  # clamped to MAX_LIMIT
    assert "bogus" in bad and "min=2" in bad


def test_valid_mint():
    assert valid_mint(MINT_CAND) is True
    assert valid_mint("not-a-mint") is False
    assert valid_mint("") is False


# ── open calls ────────────────────────────────────────────────────

def test_open_rows_flags_and_progress(tmp_path):
    now = time.time()
    desk = _seed_open_desk(tmp_path, now)
    rows = desk.open_view()["calls"]
    by_sym = {r["symbol"]: r for r in rows}
    assert set(by_sym) == {"OPEN", "NEAR", "RISK", "DOWN", "NOREAD"}
    assert by_sym["OPEN"]["progress"] == pytest.approx(0.5)
    assert by_sym["OPEN"]["flag"] == "on-track"
    assert by_sym["OPEN"]["current_mc_sol"] == pytest.approx(150.0)
    assert by_sym["OPEN"]["peak_mc_sol"] == pytest.approx(150.0)
    assert by_sym["NEAR"]["flag"] == "near-target"
    assert by_sym["NEAR"]["progress"] == pytest.approx(0.9)
    assert by_sym["RISK"]["flag"] == "near-invalidation"
    assert by_sym["DOWN"]["flag"] == "underwater"
    assert by_sym["NOREAD"]["flag"] == "no-read"
    assert by_sym["NOREAD"]["current_mc_sol"] is None


def test_open_rows_sorted_by_expiry(tmp_path):
    now = time.time()
    desk = _seed_open_desk(tmp_path, now)
    rows = desk.open_view()["calls"]
    assert [r["symbol"] for r in rows] == ["NOREAD", "DOWN", "RISK",
                                           "NEAR", "OPEN"]
    assert rows[0]["seconds_to_expiry"] < rows[-1]["seconds_to_expiry"]


def test_open_empty(tmp_path):
    desk = PumpDesk(context=None, desk_dir=str(tmp_path))
    view = desk.open_view()
    assert view["calls"] == []
    assert "no open calls" in view["rendered"]


# ── mint watchlist ────────────────────────────────────────────────

def test_mint_watchlist_round_trip(tmp_path):
    desk = PumpDesk(context=None, desk_dir=str(tmp_path))
    desk.store.watch_mint(MINT_CAND, note="test")
    assert MINT_CAND in desk.store.watched_mints()
    assert desk.store.unwatch(MINT_CAND) is True
    assert MINT_CAND not in desk.store.watched_mints()
    assert desk.store.unwatch(MINT_CAND) is False


def test_unwatch_wallet_still_works(tmp_path):
    desk = PumpDesk(context=None, desk_dir=str(tmp_path))
    wallet = _mint(3)
    desk.store.watch_dev(wallet, note="test")
    assert desk.store.unwatch(wallet) is True


# ── tool, chat, CLI ───────────────────────────────────────────────

def test_tool_board_open_watch_mint(monkeypatch, tmp_path):
    monkeypatch.setenv("NOMORALS_PUMPDESK_DIR", str(tmp_path))
    from nomorals.tools.registry import ToolRegistry
    desk = _seed_board_desk(tmp_path)
    reg = ToolRegistry(SimpleNamespace())
    reg.register_builtins()
    assert len(reg._tools) == 808  # no new tool — board rides pumpdesk
    out = reg.call("pumpdesk", action="board").value
    assert [r["symbol"] for r in out["rows"]] == ["CAND", "COLD",
                                                 "GATED", "THIN"]
    out = reg.call("pumpdesk", action="board", only_candidates=True).value
    assert [r["symbol"] for r in out["rows"]] == ["CAND"]
    out = reg.call("pumpdesk", action="open").value
    assert out["calls"] == []
    out = reg.call("pumpdesk", action="watch_mint", mint="nope").value
    assert "error" in out
    out = reg.call("pumpdesk", action="watch_mint", mint=MINT_CAND).value
    assert out["watched"] == MINT_CAND
    out = reg.call("pumpdesk", action="unwatch", mint=MINT_CAND).value
    assert out["removed"] is True
    assert desk.store.watched_mints() == set()


def test_chat_board_open_watch_mint(monkeypatch, tmp_path):
    monkeypatch.setenv("NOMORALS_PUMPDESK_DIR", str(tmp_path))
    from nomorals.agents.partner.runtime_games import RuntimeGamesMixin
    fake = SimpleNamespace(context=None)
    desk = _seed_board_desk(tmp_path)
    out = RuntimeGamesMixin._control_pump(fake, "board", "test")
    assert "pump board" in out and "CAND" in out
    out = RuntimeGamesMixin._control_pump(fake, "board min=0.8", "test")
    assert "CAND" in out and "COLD" not in out
    out = RuntimeGamesMixin._control_pump(fake, "board bogus", "test")
    assert "ignored filter token" in out
    out = RuntimeGamesMixin._control_pump(fake, "open", "test")
    assert "no open calls" in out
    out = RuntimeGamesMixin._control_pump(
        fake, f"watch mint {MINT_CAND} my pick", "test")
    assert "pinned first" in out
    assert MINT_CAND in desk.store.watched_mints()
    out = RuntimeGamesMixin._control_pump(fake, "watch mint nope", "test")
    assert "not a valid token mint" in out
    out = RuntimeGamesMixin._control_pump(
        fake, f"unwatch {MINT_CAND}", "test")
    assert "stopped watching" in out


def test_cli_board_open_watch_mint(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("NOMORALS_PUMPDESK_DIR", str(tmp_path))
    from nomorals.cmdline.commands.pump import _cmd_pump
    _seed_board_desk(tmp_path)
    args = SimpleNamespace(pump_action="board", min_composite=0.0,
                           max_rug=1.0, min_momentum=0.0,
                           candidates=False, watched=False, limit=10)
    assert _cmd_pump(args, None) == 0
    assert "pump board" in capsys.readouterr().out
    args = SimpleNamespace(pump_action="open")
    assert _cmd_pump(args, None) == 0
    assert "no open calls" in capsys.readouterr().out
    args = SimpleNamespace(pump_action="watch-mint", mint="nope",
                           note="")
    assert _cmd_pump(args, None) == 2
    args = SimpleNamespace(pump_action="watch-mint", mint=MINT_CAND,
                           note="")
    assert _cmd_pump(args, None) == 0
    assert "pinned first" in capsys.readouterr().out
    args = SimpleNamespace(pump_action="unwatch", target=MINT_CAND)
    assert _cmd_pump(args, None) == 0
    assert "stopped watching" in capsys.readouterr().out


def test_cli_parser_accepts_new_subcommands():
    from nomorals.cmdline.parser import _parser
    args = _parser().parse_args(
        ["pump", "board", "--min-composite", "0.6", "--candidates"])
    assert args.pump_action == "board"
    assert args.min_composite == 0.6
    assert args.candidates is True
    args = _parser().parse_args(["pump", "open"])
    assert args.pump_action == "open"
    args = _parser().parse_args(["pump", "watch-mint", MINT_CAND])
    assert args.pump_action == "watch-mint"
    assert args.mint == MINT_CAND
    args = _parser().parse_args(["pump", "unwatch", MINT_CAND])
    assert args.pump_action == "unwatch"
    assert args.target == MINT_CAND

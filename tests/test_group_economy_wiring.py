"""P2 — the group economy country, wired live.

Pins: chatting verifies a member and pays the daily streak once;
in-group game coins land in the group purse (one purse); /work is
cooldown-bound and levels; /pay skims the tax to the treasury;
drops are treasury-escrowed, one claim per member, and run dry
honestly; a corrupt ledger FREEZES the economy instead of reading
as empty (the organ's silent-loss shape, fixed at the source).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.community.group_economy import (
    GroupEconomy,
    GroupEconomyError,
)
from nomorals.social.chat.group_surface import GroupStore, GroupSurface

KEY = "telegram-bot:-1001"
T0 = 1_800_000_000.0


def _surface(tmp_path: Path) -> GroupSurface:
    store = GroupStore(tmp_path / "groups.json").load()
    return GroupSurface(store)


def _fund(surface: GroupSurface, member: str, amount: int) -> None:
    econ = surface.economy
    econ.verify_member(KEY, member)
    econ.contribute_treasury  # noqa: B018 - presence check
    # seed via a treasury grant funded by a direct ledger award:
    econ.award(KEY, member, amount, "seed", now=T0)


def test_chatting_verifies_and_pays_streak_once(tmp_path):
    surface = _surface(tmp_path)
    surface.handle_message(KEY, "u1", text="hello", ts=T0)
    surface.handle_message(KEY, "u1", text="again", ts=T0 + 60)
    econ = surface.economy
    assert econ.is_verified(KEY, "u1")
    # streak_day pays 5 once per day; message activity pays 0
    assert econ.balance(KEY, "u1") == 5
    assert econ.activity_count(KEY, "u1", "message", "daily", T0) == 2


def test_game_coins_land_in_the_group_purse(tmp_path):
    surface = _surface(tmp_path)
    credited = surface.game_award(KEY, "u9", 40, "ttt", now=T0)
    assert credited == 40
    assert surface.economy.balance(KEY, "u9") == 40


def test_work_shift_pays_then_cools_down(tmp_path):
    surface = _surface(tmp_path)
    first = surface.work_shift(KEY, "u1", now=T0)
    assert first["ok"] and first["pay"] >= 10
    second = surface.work_shift(KEY, "u1", now=T0 + 60)
    assert not second["ok"] and second["reason"] == "cooldown"
    third = surface.work_shift(KEY, "u1", now=T0 + 3700)
    assert third["ok"]


def test_pay_skims_tax_to_treasury(tmp_path):
    surface = _surface(tmp_path)
    _fund(surface, "u1", 150)
    result = surface.pay(KEY, "u1", "u2", 100, now=T0)
    assert result["ok"] and result["fee"] == 2
    assert result["net"] == 98
    assert surface.economy.balance(KEY, "u2") == 98
    assert surface.economy.treasury_balance(KEY) == 2


def test_drop_lifecycle_escrow_claims_and_dry_pool(tmp_path):
    surface = _surface(tmp_path)
    _fund(surface, "admin", 150)
    opened = surface.open_drop(KEY, "admin", 100, 50, now=T0)
    assert opened["ok"]
    # funded from the admin's balance into the treasury escrow
    assert surface.economy.balance(KEY, "admin") == 50
    assert surface.economy.treasury_balance(KEY) == 100
    claim = surface.claim_drop(KEY, "u1", now=T0 + 10)
    assert claim["ok"] and claim["amount"] == 50
    again = surface.claim_drop(KEY, "u1", now=T0 + 20)
    assert not again["ok"] and again["reason"] == "already"
    last = surface.claim_drop(KEY, "u2", now=T0 + 30)
    assert last["ok"] and last["remaining"] == 0
    dry = surface.claim_drop(KEY, "u3", now=T0 + 40)
    assert not dry["ok"] and dry["reason"] in ("empty", "no_drop")


def test_corrupt_ledger_freezes_instead_of_emptying(tmp_path):
    surface = _surface(tmp_path)
    _fund(surface, "u1", 50)
    ledger = (tmp_path / "group_economy" / "telegram-bot_-1001"
              / "ledger.json")
    assert ledger.exists()
    ledger.write_text("{not json", encoding="utf-8")
    with pytest.raises(GroupEconomyError):
        surface.economy.balance(KEY, "u1")
    fresh = _surface(tmp_path)
    assert fresh.economy_frozen(KEY)
    assert fresh.game_award(KEY, "u2", 10, "ttt", now=T0) == 0
    assert fresh.work_shift(KEY, "u2", now=T0)["ok"] is False


def test_coin_sink_routes_player_keys_to_members(tmp_path):
    from nomorals.agents.partner.runtime_group import RuntimeGroupMixin

    surface = _surface(tmp_path)
    mixin = RuntimeGroupMixin()
    mixin._group_surface = lambda: surface  # type: ignore[method-assign]
    player = SimpleNamespace(key="telegram:4242")
    mixin._group_coin_sink(KEY, player, 25, "connect4")
    assert surface.economy.balance(KEY, "4242") == 25

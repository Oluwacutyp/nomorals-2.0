"""The economy as a SYSTEM (owner, 2026-11-03: "the group
economy is incomplete and broken" — /deposit all answered "not
sure what that does").  Pins the full Septorch-class loop on the
group ledger: bank vault (deposit/withdraw), daily + streaks,
rob → jail → bail, casino sinks feeding treasury + jackpot,
treasury-funded drops with reservation accounting, governance
(second-admin approvals above a threshold), group quests paying
the treasury, moderation fines, season rollover with champions,
monetary-policy throttling, and the context routing that keeps
/daily, /slots and /approve meaning their old things outside
groups.  Money assertions read the ledger — the only truth.

Rigs follow the house shapes: handlers through a
RuntimeGroupMixin stub (as the presentation/moderation tests),
dispatch through a real PartnerRuntime (as the chat-modes tests).
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

from nomorals.social.chat.group_surface import GroupStore, GroupSurface

KEY = "telegram-bot:-1001234567"


class _Adapter:
    def admin_member(self, chat, user_id):
        return {"status": "administrator"}


def _rig(tmp_path: Path, *, operator_id: str | None = None):
    from nomorals.agents.partner.runtime_group import RuntimeGroupMixin

    class _RT(RuntimeGroupMixin):
        def _group_adapter(self, platform):  # type: ignore[no-untyped-def]
            return _Adapter()

        def _is_operator(self, message):  # type: ignore[no-untyped-def]
            return operator_id is not None \
                and str(message.sender_id) == operator_id

    rt = _RT()
    surface = GroupSurface(GroupStore(tmp_path / "g.json").load())
    rt._group_surface_obj = surface
    return rt, surface


def _msg(*, sender_id="55", sender="Bola", username="bola",
         text="", meta=None):
    return SimpleNamespace(
        chat=SimpleNamespace(key=KEY, platform="telegram-bot",
                             kind="group", title="Builders"),
        sender_id=sender_id, sender=sender,
        sender_username=username, text=text, meta=meta or {},
        message_id="1")


def _seed(surface, amount=300, *, user_id="55", name="Bola",
          username="bola") -> int:
    """Presence (verifies the member) + game awards: real sparks
    through the real faucet.  The organ's earn caps are real, so
    seeding spreads chunks across past weeks — the way a member
    would actually have earned them.  Returns the wallet total
    (seed + today's streak spark)."""
    surface.handle_message(KEY, user_id, username=username,
                           name=name, text="hello group")
    base = time.time()
    remaining, step = amount, 0
    while remaining > 0:
        chunk = min(190, remaining)
        surface.game_award(KEY, user_id, chunk, "ttt",
                           now=base - step * 8 * 86400)
        remaining -= chunk
        step += 1
    return surface.economy.balance(KEY, user_id)


# ── bank: the command the owner hit ─────────────────────────────


def test_deposit_all_works_and_balance_shows_the_vault(tmp_path):
    rt, surface = _rig(tmp_path)
    w0 = _seed(surface, 300)
    out = rt._control_deposit("all", _msg())
    assert "deposited" in out.text and str(w0) in out.text, out
    econ = surface.economy
    assert econ.balance(KEY, "55") == 0
    assert econ.bank_balance(KEY, "55") == w0
    assert econ.total_balance(KEY, "55") == w0
    bal = rt._control_balance(_msg())
    assert "banked" in bal.text and str(w0) in bal.text, bal
    out = rt._control_withdraw("120", _msg())
    assert "withdrew" in out.text, out
    assert econ.balance(KEY, "55") == 120
    assert econ.bank_balance(KEY, "55") == w0 - 120
    out = rt._control_deposit("9999", _msg())
    assert "doesn't work" in str(out)  # honest refusal, no fake move
    assert econ.balance(KEY, "55") == 120


def test_daily_once_a_day_and_streak_grows(tmp_path):
    rt, surface = _rig(tmp_path)
    surface.handle_message(KEY, "55", username="bola", name="Bola",
                           text="hi")
    out = rt._control_group_daily(_msg())
    assert "daily claimed" in out.text and "+25" in out.text, out
    out = rt._control_group_daily(_msg())
    assert "already claimed" in str(out), out
    now = time.time()
    nxt = surface.daily_claim(KEY, "55", now=now + 86400 + 60)
    assert nxt["ok"] and nxt["streak"] == 2
    assert nxt["amount"] == 27  # base + streak step


# ── crime: rob → jail → bail ────────────────────────────────────


class _Rigged:
    """Deterministic rng for the organ's gambling/rob draws."""

    def __init__(self, *, draw: float, symbol: str = "🍒"):
        self._draw = draw
        self._symbol = symbol

    def random(self) -> float:
        return self._draw

    def uniform(self, a: float, b: float) -> float:
        return (a + b) / 2

    def choices(self, population, weights=None):  # noqa: ANN001
        return [self._symbol]


class _MixedReels:
    def __init__(self):
        self._symbols = iter(["🍒", "🍋", "🔔"])

    def choices(self, population, weights=None):  # noqa: ANN001
        return [next(self._symbols)]

    def random(self) -> float:
        return 0.9


def test_rob_success_moves_wallet_only_failure_jails(tmp_path):
    rt, surface = _rig(tmp_path)
    _seed(surface, 200)                       # Bola holds 200
    surface.deposit(KEY, "55", 150)           # 150 banked, 50 wallet
    econ = surface.economy
    surface.handle_message(KEY, "77", username="crim", name="Crim",
                           text="yo")
    won = surface.rob(KEY, "77", "55", rng=_Rigged(draw=0.1))
    assert won["ok"] and won["success"]
    assert won["amount"] == 9                 # 17.5% of the 55 wallet
    assert econ.balance(KEY, "55") == 46
    assert econ.bank_balance(KEY, "55") == 150    # vault untouched
    assert econ.balance(KEY, "77") == 14      # his streak 5 + the 9
    caught = surface.rob(KEY, "77", "55", rng=_Rigged(draw=0.9),
                         now=time.time() + 700)
    assert caught["ok"] and caught["jailed"]
    jail = surface.jail_state(KEY, "77")
    assert jail and jail["bail"] >= 10
    out = rt._control_work(_msg(sender_id="77", sender="Crim",
                                username="crim"))
    assert "jail" in str(out), out
    bail = surface.bail(KEY, "77")
    assert bail["ok"] and bail["cost"] == jail["bail"]
    assert surface.jail_state(KEY, "77") is None
    assert econ.treasury_balance(KEY) >= jail["bail"]


def test_slots_loss_feeds_treasury_and_jackpot_triple7_takes_pot(
        tmp_path):
    rt, surface = _rig(tmp_path)
    _seed(surface, 200)
    econ = surface.economy
    triple = econ.gamble_slots(KEY, "55", 10,
                               _Rigged(draw=0.5, symbol="🍋"))
    assert triple["ok"] and triple["payout"] == 40  # 🍋🍋🍋 ×4
    loss = econ.gamble_slots(KEY, "55", 10, _MixedReels())
    assert loss["ok"] and loss["payout"] == 0
    assert econ.treasury_balance(KEY) >= 5
    assert econ.jackpot_balance(KEY) >= 5
    pot_before = econ.jackpot_balance(KEY)
    win = econ.gamble_slots(KEY, "55", 10,
                            _Rigged(draw=0.5, symbol="7️⃣"))
    assert win["multiplier"] == 77
    assert win["jackpot"] == pot_before  # the whole pot, paid out
    assert econ.jackpot_balance(KEY) == 0


def test_coinflip_win_pays_house_edge_rate(tmp_path):
    rt, surface = _rig(tmp_path)
    _seed(surface, 100)
    econ = surface.economy
    out = econ.gamble_coinflip(KEY, "55", 10, "heads",
                               _Rigged(draw=0.1))   # flip = heads
    assert out["won"] and out["payout"] == 19
    out = econ.gamble_coinflip(KEY, "55", 10, "heads",
                               _Rigged(draw=0.9))   # flip = tails
    assert not out["won"] and out["payout"] == 0


# ── treasury drops, reservations, governance ────────────────────


def _fund_treasury(surface, amount=400) -> None:
    _seed(surface, amount)
    assert surface.economy.contribute_treasury(KEY, "55", amount)


def test_treasury_drop_reserves_against_grants(tmp_path):
    rt, surface = _rig(tmp_path)
    _fund_treasury(surface, 400)
    drop = surface.open_drop_from_treasury(KEY, 300, 100, by="7")
    assert drop["ok"], drop
    assert surface.reserved_treasury(KEY) == 300
    grant = surface.grant(KEY, "55", 150, by="7")
    assert not grant["ok"] and grant["available"] == 100
    grant = surface.grant(KEY, "55", 100, by="7")
    assert grant["ok"]
    claim = surface.claim_drop(KEY, "55")
    assert claim["ok"] and claim["amount"] == 100
    assert surface.reserved_treasury(KEY) == 200


def test_governance_second_admin_approves(tmp_path):
    rt, surface = _rig(tmp_path)  # adapter: everyone platform-admin
    _seed(surface, 900)
    assert surface.economy.contribute_treasury(KEY, "55", 900)
    out = rt._control_grant("@bola 600", _msg(sender_id="admin1",
                                              sender="Admin",
                                              username="boss"))
    text = out.text if hasattr(out, "text") else str(out)
    assert "proposal" in text and "SECOND" in text, out
    pid = next(iter(surface.open_proposals(KEY)))
    same = rt._control_approve(pid, _msg(sender_id="admin1",
                                         sender="Admin",
                                         username="boss"))
    assert "second" in same or "can't approve" in same, same
    done = rt._control_approve(pid, _msg(sender_id="admin2",
                                         sender="Admin",
                                         username="boss"))
    assert "approved and executed" in done, done
    econ = surface.economy
    assert econ.balance(KEY, "55") == 605  # his streak 5 + the 600
    assert econ.treasury_balance(KEY) == 300


def test_operator_grant_skips_governance(tmp_path):
    rt, surface = _rig(tmp_path, operator_id="7")
    _fund_treasury(surface, 900)
    out = rt._control_grant("@bola 600", _msg(sender_id="7",
                                              sender="Owner",
                                              username="boss"))
    assert "granted" in str(out), out
    assert surface.economy.balance(KEY, "55") == 605


# ── boards, policy, group quests, seasons, fines ────────────────


def test_top_board_counts_bank_and_skips_pseudo(tmp_path):
    rt, surface = _rig(tmp_path)
    _seed(surface, 100)
    _seed(surface, 50, user_id="77", name="Crim", username="crim")
    surface.deposit(KEY, "77", 50)
    rows = surface.top_holders(KEY)
    assert [r["member"] for r in rows] == ["55", "77"]
    assert rows[1]["total"] == 55 and rows[1]["bank"] == 50
    assert all(not r["member"].startswith("bank:") for r in rows)
    out = rt._control_top(_msg())
    assert "richest" in out and "Bola" in out, out


def test_group_quest_pays_treasury_once(tmp_path):
    rt, surface = _rig(tmp_path)
    surface.handle_message(KEY, "55", username="bola", name="Bola",
                           text="hi")
    econ = surface.economy
    now = time.time()
    for _i in range(500):
        econ.record_activity(KEY, "55", "message", now=now)
    extras = surface._extras(KEY)
    extras["quest_check_at"] = 0
    paid = surface.group_quest_cycle(KEY, now=now + 10)
    assert "Loud Week" in {p["name"] for p in paid}, paid
    treasury = econ.treasury_balance(KEY)
    assert treasury >= 200
    extras["quest_check_at"] = 0
    assert surface.group_quest_cycle(KEY, now=now + 20) == []
    assert econ.treasury_balance(KEY) == treasury


def test_season_rollover_crowns_and_archives(tmp_path):
    rt, surface = _rig(tmp_path)
    surface.handle_message(KEY, "55", username="bola", name="Bola",
                           text="hi")
    econ = surface.economy
    from nomorals.community.group_economy import season_key
    old = time.time() - 40 * 7 * 86400  # a long-past season
    econ.award(KEY, "55", 50, "old season work", now=old)
    extras = surface._extras(KEY)
    extras["season_seen"] = season_key(old)
    note = surface.season_cycle(KEY, now=time.time())
    assert note and "champions" in note and "Bola" in note, note
    assert extras["season_seen"] == season_key(time.time())
    archived = econ._load_json(KEY, f"archive-{season_key(old)}", [])
    assert len(archived) >= 1


def test_monetary_policy_throttles_the_faucet(tmp_path):
    rt, surface = _rig(tmp_path)
    surface.handle_message(KEY, "55", username="bola", name="Bola",
                           text="hi")
    surface.handle_message(KEY, "77", username="crim", name="Crim",
                           text="hi")
    econ = surface.economy
    now = time.time()
    for member in ("55", "77"):
        for day in range(4):
            econ.award(KEY, member, 200, "grind",
                       now=now - day * 86400)
    state = econ.issuance_state(KEY, now=now)
    assert state["earned"] >= 1000 and state["throttled"], state
    entry = econ.record_activity(KEY, "55", "streak_day", now=now)
    assert entry is not None and entry.amount == 2  # 5 halved
    out = rt._control_tax(_msg())
    assert "throttled" in out, out


def test_auto_mute_fine_moves_to_treasury(tmp_path):
    rt, surface = _rig(tmp_path)
    before = _seed(surface, 100, user_id="66", name="Spam",
                   username="spam")
    surface.update_settings(KEY, fine_amount=30)
    econ = surface.economy
    result = {}
    for i in range(8):  # flood: past the limit → a mute action
        result = surface.handle_message(
            KEY, "66", username="spam", name="Spam",
            text=f"spam {i} http://spammy.example/x",
            message_id=str(i))
    assert any(a.get("kind") == "mute"
               for a in result.get("actions") or []), result
    # exactly one fine for the episode, however many mutes fired
    assert econ.balance(KEY, "66") == before - 30, \
        econ.balance(KEY, "66")
    assert econ.treasury_balance(KEY) >= 30


def test_health_excludes_pseudo_members(tmp_path):
    rt, surface = _rig(tmp_path)
    _seed(surface, 100)
    surface.deposit(KEY, "55", 40)
    econ = surface.economy
    econ.gamble_slots(KEY, "55", 10, _MixedReels())
    health = surface.economy_health(KEY)
    assert health["holders"] == 1, health
    assert health["jackpot"] >= 5


# ── context routing through the real dispatch ───────────────────


def _full_runtime(tmp_path: Path):
    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import load_settings

    settings = load_settings(overrides={
        "home": str(tmp_path),
        "partner.platforms": "local",
        "chat.local_enabled": "true",
    })
    context = build_context(settings, with_executor=False,
                            with_tools=False)

    class _Router:
        def chat(self, *a, **k):  # noqa: ANN001, ANN202
            raise AssertionError("test must not call the model")

    context.router = _Router()
    return PartnerRuntime(context), context


def test_daily_and_approve_keep_dm_meanings(tmp_path):
    from nomorals.social.chat.base import (
        ChatKind, ChatMessage, ChatRef)
    from nomorals.social.chat.control import parse_control

    rt, context = _full_runtime(tmp_path)
    try:
        dm = ChatRef(platform="local", chat_id="owner",
                     kind=ChatKind.DM, title="")

        def run(text):
            assert parse_control(text) is not None, text
            msg = ChatMessage(chat=dm, sender="Owner",
                              sender_id="owner", text=text,
                              incoming=True)
            out = rt.handle_control(text, "local:owner",
                                    message=msg)
            return out.text if hasattr(out, "text") else str(out)

        daily = run("/daily")
        assert "group command" not in daily, daily
        approve = run("/approve nope-123")
        assert "group command" not in approve, approve
    finally:
        context.close()

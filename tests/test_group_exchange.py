"""The §25 conversion system: games coins ⇄ group sparks.

Pins the exchange as a living system, not a window: the rate
floats with the weekly pool inside its band and the pool
refills with the ISO week; the weekly allowance is progression
(level × mastery × gang rank); the reverse door is priced off
the same rate minus the spread so a round trip always loses;
every conversion journals on BOTH sides (estate kind "convert"
under one shared ref, games ledger source/sink "convert");
conversion-in counts as issuance for the §17 policy; refusals
are whole (nothing moves) and specific; §33 keeps the exchange
out of foreign rooms; and while a gang holds power, the
territory skim routes out of the converted amount into that
gang's treasury — no skim, and no gang credit, without one.

The wager purse binding is not touched by any of this (its own
suite pins it).
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

from nomorals.community.group_exchange import (
    EXCHANGE_DEFAULTS,
    allowance_for,
    forward_sparks,
    rate_for,
    reverse_coins,
)
from nomorals.games.engine import GameEngine
from nomorals.games.players import Player
from nomorals.social.chat.group_surface import GroupStore, GroupSurface
from nomorals.storage.db import Database

KEY = "telegram-bot:-1001234567"
FOREIGN = "telegram:-100999"
NOW = time.time()


# ── rigs ─────────────────────────────────────────────────────

def _surface(tmp_path: Path) -> GroupSurface:
    return GroupSurface(GroupStore(tmp_path / "g.json").load())


def _engine():
    db = Database(":memory:")
    db.migrate()

    class Ctx:
        pass

    ctx = Ctx()
    ctx.db = db
    engine = GameEngine(ctx, send=lambda chat, text: None)
    return engine, db


def _wallet(engine, player):
    from nomorals.agents.partner.runtime_group import _GamesCoinWallet

    return _GamesCoinWallet(engine, player)


def _rich_surface(tmp_path: Path, **overrides) -> GroupSurface:
    """A surface whose exchange allows big conversions (pool +
    allowance lifted) so pool mechanics can be exercised."""
    surface = _surface(tmp_path)
    cfg = dict(EXCHANGE_DEFAULTS)
    cfg.update({"pool_sparks": 20_000, "allowance_base": 100_000})
    cfg.update(overrides)
    surface.update_settings(KEY, exchange=cfg)
    return surface


def _convert_entries(surface, member=None):
    out = [e for e in surface.economy._ledger(KEY)
           if e.kind == "convert"]
    if member is not None:
        out = [e for e in out if e.member == member]
    return out


def _game_ledger(db, key):
    rows = db.execute(
        "SELECT kind, category, amount FROM game_ledger "
        "WHERE player_key = ?", (key,)).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


# ── the math: rate band, allowance ladder, the losing loop ───

def test_rate_band_moves_with_the_pool():
    cfg = dict(EXCHANGE_DEFAULTS)
    assert rate_for(cfg, 20_000, 20_000) == 14   # full pool: top
    assert rate_for(cfg, 10_000, 20_000) == 10   # half: the base
    assert rate_for(cfg, 0, 20_000) == 6         # drained: floor
    assert rate_for(cfg, 0, 0) == 10             # closed: base


def test_allowance_ladder_grows_and_caps():
    cfg = dict(EXCHANGE_DEFAULTS)
    base = allowance_for(cfg)
    assert base == 400
    assert allowance_for(cfg, level=21) == 800
    assert allowance_for(cfg, mastery_steps=5) == 600
    assert allowance_for(cfg, gang_rank=1) == 500
    assert allowance_for(cfg, gang_rank=3) == 800
    assert allowance_for(cfg, level=500, mastery_steps=500,
                         gang_rank=3) == 400 * 400 // 100
    # every rung strictly grows the allowance
    ladder = [allowance_for(cfg, level=lv) for lv in (1, 5, 11, 21)]
    assert ladder == sorted(ladder) and len(set(ladder)) == 4


def test_round_trip_always_loses_by_the_spread():
    cfg = dict(EXCHANGE_DEFAULTS)
    for rate in (6, 10, 14):
        sparks = forward_sparks(10_000, rate)
        back = reverse_coins(sparks, rate, cfg)
        assert back == 8_500  # exactly the 15% spread, any rate
        assert back < 10_000


# ── pool: drains, moves the rate, refills weekly ─────────────

def test_pool_drains_rate_worsens_and_week_refills(tmp_path):
    surface = _rich_surface(tmp_path)
    engine, _db = _engine()
    ada = Player.from_sender("telegram", "111", "Ada")
    engine.store.add_coins(ada, 200_000, reason="seed")
    q0 = surface.exchange_quote(KEY, "111", now=NOW)
    assert q0["rate"] == 14 and q0["pool"]["remaining"] == 20_000
    out = surface.exchange_convert(
        KEY, "111", "in", 50_000, wallet=_wallet(engine, ada),
        now=NOW)
    assert out["ok"] and out["gross"] == 7_000, out
    q1 = surface.exchange_quote(KEY, "111", now=NOW)
    assert q1["pool"]["remaining"] == 13_000
    assert q1["rate"] == 11  # the pool answered back
    # last week's conversion does not draw this week's pool
    bob = Player.from_sender("telegram", "222", "Bob")
    engine.store.add_coins(bob, 200_000, reason="seed")
    out = surface.exchange_convert(
        KEY, "222", "in", 10_000, wallet=_wallet(engine, bob),
        now=NOW - 8 * 86400)
    assert out["ok"], out
    q2 = surface.exchange_quote(KEY, "111", now=NOW)
    assert q2["pool"]["remaining"] == 13_000


def test_empty_pool_refuses_and_moves_nothing(tmp_path):
    surface = _rich_surface(tmp_path, pool_sparks=100)
    engine, _db = _engine()
    ada = Player.from_sender("telegram", "111", "Ada")
    engine.store.add_coins(ada, 100_000, reason="seed")
    out = surface.exchange_convert(
        KEY, "111", "in", 10_000, wallet=_wallet(engine, ada),
        now=NOW)
    assert not out["ok"] and out["reason"] == "pool", out
    assert engine.store.get_for(ada).coins == 100_000
    assert surface.economy.balance(KEY, "111") == 0
    assert _convert_entries(surface) == []


def test_closed_pool_is_stated_never_silent(tmp_path):
    surface = _rich_surface(tmp_path, pool_sparks=0)
    engine, _db = _engine()
    ada = Player.from_sender("telegram", "111", "Ada")
    engine.store.add_coins(ada, 10_000, reason="seed")
    quote = surface.exchange_quote(KEY, "111", now=NOW)
    assert quote["closed"] is True
    out = surface.exchange_convert(
        KEY, "111", "in", 1_000, wallet=_wallet(engine, ada),
        now=NOW)
    assert not out["ok"] and out["reason"] == "closed", out
    assert engine.store.get_for(ada).coins == 10_000


# ── allowances: per-member, standing-priced, whole refusals ──

def test_allowance_binds_and_refusal_moves_nothing(tmp_path):
    surface = _surface(tmp_path)  # default allowance base 400
    engine, _db = _engine()
    ada = Player.from_sender("telegram", "111", "Ada")
    engine.store.add_coins(ada, 100_000, reason="seed")
    # level 1: 400 sparks of allowance. 3000 coins at rate 14
    # gross 420 > 400 → refused whole, nothing moves.
    out = surface.exchange_convert(
        KEY, "111", "in", 3_000, wallet=_wallet(engine, ada),
        now=NOW)
    assert not out["ok"] and out["reason"] == "allowance", out
    assert out["allowance_left"] == 400
    assert engine.store.get_for(ada).coins == 100_000
    assert surface.economy.balance(KEY, "111") == 0
    # 2800 coins → 392 sparks fits; the next 100 coins do not.
    out = surface.exchange_convert(
        KEY, "111", "in", 2_800, wallet=_wallet(engine, ada),
        now=NOW)
    assert out["ok"] and out["sparks"] == 392, out
    out = surface.exchange_convert(
        KEY, "111", "in", 100, wallet=_wallet(engine, ada),
        now=NOW)
    assert not out["ok"] and out["reason"] == "allowance", out
    assert out["allowance_left"] == 8
    assert surface.economy.balance(KEY, "111") == 392


def test_higher_standing_converts_more(tmp_path):
    surface = _surface(tmp_path)
    engine, _db = _engine()
    ada = Player.from_sender("telegram", "111", "Ada")
    engine.store.add_coins(ada, 100_000, reason="seed")
    out = surface.exchange_convert(
        KEY, "111", "in", 3_000, wallet=_wallet(engine, ada),
        level=21, mastery_steps=10, now=NOW)
    # allowance 400 × (1 + 1.0 level + 1.0 mastery) = 1200 ≥ 420
    assert out["ok"] and out["sparks"] == 420, out


def test_max_takes_exactly_what_the_allowance_allows(tmp_path):
    surface = _surface(tmp_path)
    engine, _db = _engine()
    ada = Player.from_sender("telegram", "111", "Ada")
    engine.store.add_coins(ada, 5_000, reason="seed")
    out = surface.exchange_convert(
        KEY, "111", "in", None, wallet=_wallet(engine, ada),
        now=NOW)
    assert out["ok"], out
    assert out["sparks"] == 399  # 2857 coins × 14 // 100
    assert out["coins"] == 2_857
    assert out["quote"]["allowance_left"] == 1
    assert engine.store.get_for(ada).coins == 5_000 - 2_857


# ── journals: both sides, one ref; issuance sees it ──────────

def test_conversion_journals_on_both_sides(tmp_path):
    surface = _rich_surface(tmp_path)
    engine, db = _engine()
    ada = Player.from_sender("telegram", "111", "Ada")
    engine.store.add_coins(ada, 50_000, reason="seed")
    out = surface.exchange_convert(
        KEY, "111", "in", 10_000, wallet=_wallet(engine, ada),
        now=NOW)
    assert out["ok"], out
    entries = _convert_entries(surface, member="111")
    assert len(entries) == 1 and entries[0].amount == 1_400
    assert entries[0].ref.startswith("convert:") \
        and entries[0].ref.endswith(":in")
    assert _game_ledger(db, ada.key) == [("sink", "convert", 10_000)]
    # and back out: the reverse legs, again both sides.  The
    # pool draw moved the rate (14 → 13), so the reverse prices
    # off the CURRENT quote — still strictly losing.
    rate_now = surface.exchange_quote(KEY, "111", now=NOW)["rate"]
    assert rate_now == 13
    expected = reverse_coins(1_400, rate_now,
                             dict(EXCHANGE_DEFAULTS))
    out = surface.exchange_convert(
        KEY, "111", "out", 1_400, wallet=_wallet(engine, ada),
        now=NOW)
    assert out["ok"] and out["coins"] == expected, out
    assert expected < 10_000
    entries = _convert_entries(surface, member="111")
    assert [e.amount for e in entries] == [1_400, -1_400]
    assert entries[1].ref.endswith(":out")
    assert _game_ledger(db, ada.key) == [
        ("sink", "convert", 10_000), ("source", "convert", expected)]
    assert engine.store.get_for(ada).coins \
        == 50_000 - 10_000 + expected


def test_conversion_in_counts_as_issuance(tmp_path):
    surface = _rich_surface(tmp_path)
    engine, _db = _engine()
    ada = Player.from_sender("telegram", "111", "Ada")
    engine.store.add_coins(ada, 50_000, reason="seed")
    econ = surface.economy
    before = econ.issuance_state(KEY, now=NOW)
    out = surface.exchange_convert(
        KEY, "111", "in", 10_000, wallet=_wallet(engine, ada),
        now=NOW)
    assert out["ok"], out
    after = econ.issuance_state(KEY, now=NOW)
    assert after["earned"] - before["earned"] == 1_400
    out = surface.exchange_convert(
        KEY, "111", "out", 400, wallet=_wallet(engine, ada),
        now=NOW)
    assert out["ok"], out
    final = econ.issuance_state(KEY, now=NOW)
    assert final["sunk"] - after["sunk"] == 400


# ── refusals: funds, size ────────────────────────────────────

def test_insufficient_funds_refuse_both_ways(tmp_path):
    surface = _rich_surface(tmp_path)
    engine, _db = _engine()
    ada = Player.from_sender("telegram", "111", "Ada")
    engine.store.add_coins(ada, 100, reason="seed")
    out = surface.exchange_convert(
        KEY, "111", "in", 1_000, wallet=_wallet(engine, ada),
        now=NOW)
    assert not out["ok"] and out["reason"] == "funds_coins", out
    assert engine.store.get_for(ada).coins == 100
    surface.game_wager_credit(KEY, "111", 100, "seed sparks")
    out = surface.exchange_convert(
        KEY, "111", "out", 500, wallet=_wallet(engine, ada),
        now=NOW)
    assert not out["ok"] and out["reason"] == "funds_sparks", out
    assert surface.economy.balance(KEY, "111") == 100
    assert engine.store.get_for(ada).coins == 100
    assert _convert_entries(surface) == []


def test_too_small_refuses_with_the_minimum(tmp_path):
    surface = _rich_surface(tmp_path)
    engine, _db = _engine()
    ada = Player.from_sender("telegram", "111", "Ada")
    engine.store.add_coins(ada, 1_000, reason="seed")
    out = surface.exchange_convert(
        KEY, "111", "in", 3, wallet=_wallet(engine, ada), now=NOW)
    assert not out["ok"] and out["reason"] == "too_small", out
    assert out["min_coins"] == 8  # ceil(100 / 14)
    assert engine.store.get_for(ada).coins == 1_000


def test_exchange_off_refuses(tmp_path):
    surface = _rich_surface(tmp_path, enabled=False)
    engine, _db = _engine()
    ada = Player.from_sender("telegram", "111", "Ada")
    engine.store.add_coins(ada, 1_000, reason="seed")
    out = surface.exchange_convert(
        KEY, "111", "in", 1_000, wallet=_wallet(engine, ada),
        now=NOW)
    assert not out["ok"] and out["reason"] == "off", out
    assert engine.store.get_for(ada).coins == 1_000


# ── the territory skim ───────────────────────────────────────

def _power_gang(surface, now):
    """A five-member syndicate holding two districts."""
    econ = surface.economy
    for uid in ("31", "32", "33", "34", "35"):
        surface.game_wager_credit(KEY, uid, 900, "seed")
    organ = surface.conflict
    gang = organ.create_gang(KEY, "31", "Owls", "syndicate",
                             now=now)
    assert gang["ok"], gang
    for uid in ("32", "33", "34", "35"):
        out = organ.join_gang(KEY, uid, "Owls", now=now)
        assert out["ok"], out
    assert organ.gang_deposit(KEY, "31", 300, now=now)["ok"]
    assert organ.claim_territory(KEY, "31", "old_town",
                                 now=now)["ok"]
    assert organ.claim_territory(KEY, "31", "market_mile",
                                 now=now)["ok"]
    power = organ.power_gang(KEY, now=now)
    assert power is not None and power["name"] == "Owls"
    return power


def test_no_power_gang_no_skim(tmp_path):
    surface = _rich_surface(tmp_path)
    engine, _db = _engine()
    ada = Player.from_sender("telegram", "111", "Ada")
    engine.store.add_coins(ada, 50_000, reason="seed")
    quote = surface.exchange_quote(KEY, "111", now=NOW)
    assert quote["skim_pct"] == 0 and quote["skim_gang"] == ""
    out = surface.exchange_convert(
        KEY, "111", "in", 10_000, wallet=_wallet(engine, ada),
        now=NOW)
    assert out["ok"] and out["skim"] == 0
    assert out["sparks"] == out["gross"] == 1_400
    treasury = [e for e in _convert_entries(surface)
                if e.member == ""]
    assert treasury == []


def test_power_gang_skim_routes_out_of_the_conversion(tmp_path):
    surface = _rich_surface(tmp_path)
    engine, _db = _engine()
    power = _power_gang(surface, NOW)
    organ = surface.conflict
    gang_before = organ.gang_of(KEY, "31")["treasury"]
    ada = Player.from_sender("telegram", "111", "Ada")
    engine.store.add_coins(ada, 50_000, reason="seed")
    quote = surface.exchange_quote(KEY, "111", now=NOW)
    assert quote["skim_pct"] == 2 and quote["skim_gang"] == "Owls"
    out = surface.exchange_convert(
        KEY, "111", "in", 10_000, wallet=_wallet(engine, ada),
        now=NOW)
    assert out["ok"], out
    assert out["gross"] == 1_400 and out["skim"] == 28
    assert out["sparks"] == 1_372
    # the skim's estate leg sits in the treasury under the SAME
    # conversion ref as the member's credit
    entries = _convert_entries(surface)
    member_leg = [e for e in entries if e.member == "111"][0]
    skim_leg = [e for e in entries if e.member == ""][0]
    assert skim_leg.amount == 28
    assert skim_leg.ref.rsplit(":", 1)[0] \
        == member_leg.ref.rsplit(":", 1)[0]
    assert skim_leg.ref.endswith(":skim")
    # and the gang's own treasury number rose by the skim
    assert organ.gang_of(KEY, "31")["treasury"] \
        == gang_before + 28
    # the reverse door skims too: paid out of the sparks in
    out = surface.exchange_convert(
        KEY, "111", "out", 1_000, wallet=_wallet(engine, ada),
        now=NOW)
    assert out["ok"] and out["skim"] == 20, out
    assert organ.gang_of(KEY, "31")["treasury"] \
        == gang_before + 28 + 20
    skimout = [e for e in _convert_entries(surface)
               if e.ref.endswith(":skimout")]
    assert len(skimout) == 1 and skimout[0].amount == 20
    assert power["id"]


# ── the runtime surface: gate, quote, convert, settings ─────

class _Adapter:
    def __init__(self, admins=()):
        self.admins = set(admins)

    def admin_member(self, chat, user_id):
        return {"status": "administrator"
                if str(user_id) in self.admins else "member"}


def _rt(tmp_path: Path, engine, *, admins=("99",)):
    from nomorals.agents.partner.runtime_group import \
        RuntimeGroupMixin

    class _RT(RuntimeGroupMixin):
        def _group_adapter(self, platform):
            return _Adapter(admins)

        def _is_operator(self, message):
            return False

        def _game_engine(self):
            return engine

    rt = _RT()
    surface = GroupSurface(GroupStore(tmp_path / "g.json").load())
    rt._group_surface_obj = surface
    return rt, surface


def _msg(*, key=KEY, sender_id="55", sender="Bola",
         username="bola", platform="telegram-bot"):
    return SimpleNamespace(
        chat=SimpleNamespace(key=key, platform=platform,
                             kind="group", title="Builders"),
        sender_id=sender_id, sender=sender,
        sender_username=username, text="", meta={},
        message_id="1")


def test_runtime_gate_refuses_foreign_rooms(tmp_path):
    # the owner's own bug shape (§33): his userbot sits in a
    # foreign room his bot doesn't operate in — the exchange
    # must not run, or accrue anything, there.
    engine, _db = _engine()
    rt, surface = _rt(tmp_path, engine)
    out = rt._control_exchange_convert(
        "100", _msg(key=FOREIGN, sender_id="111",
                    platform="telegram"))
    assert "aren't running in this group" in str(out), out
    assert _convert_entries(surface) == []


def test_runtime_quote_convert_and_balance_line(tmp_path):
    engine, _db = _engine()
    rt, surface = _rt(tmp_path, engine)
    ada = Player.from_sender("telegram", "55", "Bola")
    engine.store.add_coins(ada, 50_000, reason="seed")
    quote = rt._control_exchange_convert("", _msg())
    text = getattr(quote, "text", str(quote))
    assert "100 coins → 14 sparks" in text, text
    assert "20,000 of 20,000 sparks left" in text, text
    done = rt._control_exchange_convert("1000", _msg())
    text = getattr(done, "text", str(done))
    assert "1,000 coins →" in text and "140" in text, text
    assert engine.store.get_for(ada).coins == 49_000
    assert surface.economy.balance(KEY, "55") == 140
    back = rt._control_exchange_convert("sparks 140", _msg())
    text = getattr(back, "text", str(back))
    # the forward leg drew the pool (rate 14 → 13), so the 140
    # sparks buy back at the moved rate, minus the spread
    expected = reverse_coins(140, 13, dict(EXCHANGE_DEFAULTS))
    assert f"{expected} coins" in text, text
    assert engine.store.get_for(ada).coins == 49_000 + expected
    bal = rt._control_balance(_msg())
    text = getattr(bal, "text", str(bal))
    assert "Exchange" in text and "pool" in text, text


def test_runtime_refusals_are_plain_and_specific(tmp_path):
    engine, _db = _engine()
    rt, surface = _rt(tmp_path, engine)
    out = rt._control_exchange_convert("1000", _msg())
    assert "games coins" in str(out), out  # no coins at all
    surface.update_settings(
        KEY, exchange={**EXCHANGE_DEFAULTS, "pool_sparks": 0})
    out = rt._control_exchange_convert("1000", _msg())
    assert "closed this week" in str(out), out


def test_gsettings_exchange_grammar(tmp_path):
    engine, _db = _engine()
    rt, surface = _rt(tmp_path, engine)
    admin = _msg(sender_id="99")
    out = rt._control_gsettings("exchange pool 5000", admin)
    assert "5,000" in out or "5000" in out, out
    cfg = surface.exchange_cfg(KEY)
    assert cfg["pool_sparks"] == 5000
    out = rt._control_gsettings("exchange skim 5", admin)
    assert surface.exchange_cfg(KEY)["skim_pct"] == 5
    out = rt._control_gsettings("exchange skim 90", admin)
    # the skim is hard-bounded to the §23 capture range
    assert surface.exchange_cfg(KEY)["skim_pct"] == 10
    out = rt._control_gsettings("exchange off", admin)
    assert surface.exchange_cfg(KEY)["enabled"] is False
    out = rt._control_gsettings("exchange bogus", admin)
    assert "usage:" in out, out


def test_currency_tail_guard_keeps_media_meaning():
    from nomorals.agents.partner.runtime_group import \
        RuntimeGroupMixin

    guard = RuntimeGroupMixin._convert_tail_is_currency
    assert guard("") and guard("1000") and guard("sparks 50")
    assert guard("max") and guard("coins all")
    assert not guard("sticker-to-mp4")
    assert not guard("mp4-to-mp3")
    assert not guard("gif-to-mp4 extra")

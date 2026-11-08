"""Failed-bank-heist consequence bundle (doc 4, owner-adopted
2026-11-08; rules settled by the owner, not re-derived here):

- A conviction requires a RESOLVED BUST on a bank/treasury
  target. Partial, decoy, empty, member-vault busts, malformed
  commands and failures before resolution never convict.
- One durable CASE per bust (group, rules version, heist
  identity, offenders + roles, evidence, settlement state),
  settled atomically over Batch B's substrate; a retried
  settlement never double-confiscates, never re-jails, never
  extends a blacklist.
- The whole crew is liable; settlement is per offender.
- Bank-deposit BLACKLIST: 48h from the bust, 96h on a repeat
  bust, auto-lift, enforced at the bank OPERATION itself —
  every credit whose destination is the offender's bank vault
  is refused, whoever sends it. Wallet receipts are not
  deposits and still land; withdrawals and bank debits are
  not deposits and still run.
- CONFISCATION = real property only (the offender's estate
  properties, with their attached state), into the BANK
  RECOVERY ESTATE under the case — never wallet, bank balance,
  gems or equipment; never an uninvolved member's property.
- Bail changes custody ONLY: blacklist and seizure stand.

Rigs follow the house shapes: GroupSurface over a tmp_path
GroupStore; a stub rng whose rolls the test chooses.
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.community.group_economy import (
    BankBlacklistedError,
)
from nomorals.social.chat.group_surface import (
    GroupStore,
    GroupSurface,
)

KEY = "telegram-bot:-1001234567"
HOUR = 3600.0


class _Rng:
    def __init__(self, roll: float = 0.99) -> None:
        self.roll = roll

    def random(self) -> float:
        return self.roll

    def uniform(self, a: float, b: float) -> float:
        return 1.0

    def randint(self, a: int, b: int) -> int:
        return a

    def choice(self, seq):
        return list(seq)[0]


def _surface(tmp_path: Path, name: str = "g.json") -> GroupSurface:
    surface = GroupSurface(GroupStore(tmp_path / name).load())
    cfg = dict(surface.store.settings(KEY)["conflict"])
    cfg["mode"] = "crime"
    surface.update_settings(KEY, conflict=cfg)
    return surface


def _seed(surface: GroupSurface, uid: str, amount: int,
          name: str | None = None) -> int:
    surface.handle_message(KEY, uid, username=f"u{uid}",
                           name=name or f"User {uid}",
                           text="hello group")
    base = time.time()
    remaining, step = amount, 0
    while remaining > 0:
        chunk = min(190, remaining)
        surface.game_award(KEY, uid, chunk, "ttt",
                           now=base - step * 8 * 86400)
        remaining -= chunk
        step += 1
    return surface.economy.balance(KEY, uid)


def _gang5(surface: GroupSurface, now: float) -> None:
    """Leader 11 + members 12..15 — the tier a bank score
    demands. 13 is in the gang but never rides the crew."""
    out = surface.gang_create(KEY, "11", "Night Owls", "street",
                              now=now)
    assert out["ok"], out
    for member in ("12", "13", "14", "15"):
        joined = surface.gang_join(KEY, member, "Night Owls",
                                   now=now)
        assert joined["ok"], joined


def _buy(surface: GroupSurface, uid: str, district: str,
         tier: str, now: float) -> str:
    out = surface.estate_buy(KEY, uid, district, tier, now=now)
    assert out["ok"], out
    return str(out["property"]["id"])


def _bank_bust(surface: GroupSurface, now: float, *,
               roll: float = 0.99) -> dict:
    start = surface.heist_start(KEY, "11", "bank", "harbour",
                                now=now)
    assert start["ok"], start
    assert surface.heist_join(KEY, "12", now=now)["ok"]
    out = surface.heist_go(KEY, "11", rng=_Rng(roll), now=now)
    assert out["ok"], out
    return out


def _rig(tmp_path: Path, now: float) -> GroupSurface:
    """Crew (11 leads, 12 rides), bystander 13 in the gang.
    11 owns two properties (one listed), 12 owns one,
    13 owns one. Everyone carries wallet + banked sparks."""
    surface = _surface(tmp_path)
    for uid, amount in (("11", 9000), ("12", 5000),
                        ("13", 5000), ("14", 800), ("15", 800)):
        _seed(surface, uid, amount)
    _gang5(surface, now)
    surface.deposit(KEY, "11", 1000, now=now)
    surface.deposit(KEY, "12", 700, now=now)
    p1 = _buy(surface, "11", "old_town", "room", now)
    p2 = _buy(surface, "11", "market_mile", "apartment", now)
    listed = surface.estate.list_property(KEY, "11", p2, 900,
                                          now=now)
    assert listed["ok"], listed
    p12 = _buy(surface, "12", "old_town", "room", now)
    p13 = _buy(surface, "13", "old_town", "room", now)
    props = {"11": {p1, p2}, "12": {p12}, "13": {p13}}
    return surface, props


def _owner_of(surface: GroupSurface, pid: str) -> str:
    state = surface.estate._load_state(KEY)
    return str((state.get("properties") or {})[pid]["owner"])


# ── the conviction itself ─────────────────────────────────────

def test_bank_bust_opens_one_case_and_settles_it(tmp_path):
    now = time.time()
    surface, _props = _rig(tmp_path, now)
    bank_before = surface.economy.bank_balance(KEY, "11")
    wallet_before = surface.economy.balance(KEY, "11")
    out = _bank_bust(surface, now)
    assert out["outcome"] == "bust"
    case = out.get("case")
    assert case, out
    assert case["state"] == "committed", case
    assert case["target"]["kind"] == "bank"
    assert case["rules_version"]
    assert case["heist_id"]
    offenders = case["offenders"]
    assert set(offenders) == {"11", "12"}
    assert offenders["11"]["role"] == "leader"
    assert offenders["12"]["role"] == "crew"
    # arrest is the existing jail, recorded on the case
    for member in ("11", "12"):
        jail = surface.jail_state(KEY, member, now=now)
        assert jail and jail["until"] > now + 40 * 60
        assert offenders[member]["jailed_until"] == jail["until"]
    # blacklist: 48h from the bust, per offender
    for member in ("11", "12"):
        assert offenders[member]["blacklist_hours"] == 48
        assert offenders[member]["blacklist_until"] \
            == pytest.approx(now + 48 * HOUR)
    # confiscation is property ONLY: money and gems untouched
    assert surface.economy.bank_balance(KEY, "11") == bank_before
    assert surface.economy.balance(KEY, "11") == wallet_before


def test_bust_seizes_crew_property_into_recovery_estate(tmp_path):
    now = time.time()
    surface, _props = _rig(tmp_path, now)
    mine_before = {p["id"] for p in
                   surface.estate.portfolio(KEY, "11")}
    crew_before = {p["id"] for p in
                   surface.estate.portfolio(KEY, "12")}
    bystander_before = {p["id"] for p in
                        surface.estate.portfolio(KEY, "13")}
    assert len(mine_before) == 2 and len(crew_before) == 1
    out = _bank_bust(surface, now)
    case = out["case"]
    custody = f"recovery:{case['id']}"
    # every crew property moved, under the case's custody
    for pid in mine_before | crew_before:
        assert _owner_of(surface, pid) == custody
    # the bystander's property never moved
    for pid in bystander_before:
        assert _owner_of(surface, pid) == "13"
    # portfolios tell the same story
    assert surface.estate.portfolio(KEY, "11") == []
    assert surface.estate.portfolio(KEY, "12") == []
    assert len(surface.estate.portfolio(KEY, "13")) == 1
    held = surface.estate.portfolio(KEY, custody)
    assert {p["id"] for p in held} == mine_before | crew_before
    # the manifest on the case is the snapshot of what was taken
    seized11 = case["offenders"]["11"]["seized"]
    assert {s["id"] for s in seized11} == mine_before
    assert {s["id"] for s in case["offenders"]["12"]["seized"]} \
        == crew_before
    # the listed property's listing died with the seizure —
    # seized property is not quietly still for sale
    market = surface.estate.market(KEY)
    assert all(row["property"] not in mine_before
               for row in market)
    # rent follows ownership: the offender collects nothing
    # from seized property anymore
    rent = surface.estate.collect_rent(KEY, "11",
                                       now=now + 3 * 86400)
    assert rent["gems"] == 0


def test_zero_property_offender_still_convicts(tmp_path):
    now = time.time()
    surface = _surface(tmp_path)
    for uid, amount in (("11", 3000), ("12", 900), ("13", 800),
                        ("14", 800), ("15", 800)):
        _seed(surface, uid, amount)
    _gang5(surface, now)
    out = _bank_bust(surface, now)
    case = out["case"]
    assert case["state"] == "committed"
    assert case["offenders"]["12"]["seized"] == []
    assert case["offenders"]["12"]["blacklist_hours"] == 48


def test_seizure_cancels_open_auction_and_refunds_the_bid(
        tmp_path):
    now = time.time()
    surface, props = _rig(tmp_path, now)
    pid = next(iter(props["12"]))
    auction = surface.estate.start_auction(KEY, "12", pid, 300,
                                           24, now=now)
    assert auction["ok"], auction
    bid = surface.estate.place_bid(KEY, "13", auction["auction"],
                                   500, now=now)
    assert bid["ok"], bid
    escrowed = surface.economy.bank_balance(KEY, "13")
    wallet13 = surface.economy.balance(KEY, "13")
    out = _bank_bust(surface, now)
    assert out["case"]["state"] == "committed"
    # the property sits in the recovery estate, the auction is
    # dead, and the standing bid came back out of escrow whole
    assert _owner_of(surface, pid) == \
        f"recovery:{out['case']['id']}"
    state = surface.estate._load_state(KEY)
    record = (state.get("auctions") or {})[auction["auction"]]
    assert record["open"] is False
    assert record.get("cancelled") == "seized"
    assert surface.economy.balance(KEY, "13") == wallet13 + 500
    assert escrowed == surface.economy.bank_balance(KEY, "13")


def test_seizure_carries_a_mortgage_as_encumbrance(tmp_path):
    now = time.time()
    surface, props = _rig(tmp_path, now)
    pid = next(p for p in props["11"]
               if not (surface.estate._load_state(KEY)
                       .get("properties") or {})[p]["listed"])
    loan = surface.estate.borrow(KEY, "11", pid, 50,
                                 cfg=surface.estate_cfg(KEY),
                                 now=now)
    assert loan["ok"], loan
    out = _bank_bust(surface, now)
    case = out["case"]
    seized = {s["id"]: s
              for s in case["offenders"]["11"]["seized"]}
    assert seized[pid]["mortgage"] == loan["mortgage"]["id"]
    state = surface.estate._load_state(KEY)
    prop = (state.get("properties") or {})[pid]
    assert prop["owner"] == f"recovery:{case['id']}"
    assert prop["mortgage"] == loan["mortgage"]["id"]
    # the debt itself is untouched: still the offender's, still
    # open, still scheduled against their bank vault
    debts = surface.estate.mortgages(KEY, "11")
    assert [m["id"] for m in debts] == [loan["mortgage"]["id"]]


def test_decoy_bank_result_never_convicts(tmp_path):
    now = time.time()
    surface, _props = _rig(tmp_path, now)
    # the bank's own decoy vault, installed at the bank scope,
    # plus a strong crew: the roll SUCCEEDS and still nobody
    # is convicted — a decoy is not a bust
    surface.gear_buy(KEY, "13", "decoy_vault", now=now)
    installed = surface.conflict.install_defense(
        KEY, "13", "decoy_vault", "bank:harbour", now=now)
    assert installed["ok"], installed
    surface.gear_buy(KEY, "11", "thermite_charge", now=now)
    surface.gear_buy(KEY, "11", "vault_drill", now=now)
    assert surface.economy.contribute_treasury(KEY, "13", 3000,
                                               now=now)
    start = surface.heist_start(KEY, "11", "bank", "harbour",
                                now=now)
    assert start["ok"], start
    assert surface.heist_join(KEY, "12", now=now)["ok"]
    out = surface.heist_go(KEY, "11", rng=_Rng(0.0), now=now)
    assert out["outcome"] == "decoy", out
    assert "case" not in out
    assert surface.jail_state(KEY, "11", now=now) is None
    assert surface.deposit(KEY, "11", 50, now=now)["ok"]


# ── the blacklist: enforced at the bank operation ─────────────

def test_blacklist_binds_every_vault_credit_route(tmp_path):
    now = time.time()
    surface, _props = _rig(tmp_path, now)
    _bank_bust(surface, now)
    econ = surface.economy
    # the typed-command route (surface deposit)
    out = surface.deposit(KEY, "11", 50, now=now)
    assert not out["ok"] and out["reason"] == "blacklisted", out
    assert out["case_id"]
    assert out["until"] == pytest.approx(now + 48 * HOUR)
    # the operation itself
    with pytest.raises(BankBlacklistedError):
        econ.bank_deposit(KEY, "11", 50, now=now)
    # a raw paired move into the vault — the disguised route
    with pytest.raises(BankBlacklistedError):
        econ._move(KEY, "13", econ.bank_member("11"), 50,
                   "transfer_in", "a favour, banked", now)
    # nothing moved anywhere
    assert econ.bank_balance(KEY, "11") == 1000
    # the bystander is not blacklisted: their deposit runs
    assert surface.deposit(KEY, "13", 50, now=now)["ok"]


def test_indirect_credits_are_judged_not_deposits(tmp_path):
    """The stated rule: the blacklist binds the VAULT, not the
    member's whole purse. A /pay lands in the wallet — not a
    deposit, it goes through; the offender banking it after is
    the deposit, and that is refused. Withdrawals and bank
    debits (mortgage-style) are not deposits either."""
    now = time.time()
    surface, _props = _rig(tmp_path, now)
    _bank_bust(surface, now)
    econ = surface.economy
    # a wallet transfer TO the offender still lands
    paid = surface.pay(KEY, "13", "11", 200, now=now)
    assert paid["ok"], paid
    # …but banking it is refused at the operation
    out = surface.deposit(KEY, "11", 200, now=now)
    assert not out["ok"] and out["reason"] == "blacklisted"
    # withdrawing what was already banked still works
    out = surface.withdraw(KEY, "11", 300, now=now)
    assert out["ok"], out
    assert econ.bank_balance(KEY, "11") == 700
    # a bank DEBIT (the mortgage/repair shape) still runs
    assert econ._move(KEY, econ.bank_member("11"), "", 100,
                      "fine", "a debit, not a deposit", now)
    assert econ.bank_balance(KEY, "11") == 600


def test_repeat_bust_blacklists_96_hours(tmp_path):
    now = time.time()
    surface, _props = _rig(tmp_path, now)
    first = _bank_bust(surface, now)
    assert first["case"]["offenders"]["11"]["blacklist_hours"] \
        == 48
    later = now + 49 * HOUR
    # the first blacklist has lifted; the crew runs it again
    second = _bank_bust(surface, later)
    case = second["case"]
    assert case["id"] != first["case"]["id"]
    slot = case["offenders"]["11"]
    assert slot["repeat"] is True
    assert slot["blacklist_hours"] == 96
    assert slot["blacklist_until"] == pytest.approx(
        later + 96 * HOUR)
    active = surface.economy.bank_blacklist(KEY, "11", now=later)
    assert active and active["case_id"] == case["id"]


def test_blacklist_lifts_automatically_and_status_says_so(
        tmp_path):
    now = time.time()
    surface, _props = _rig(tmp_path, now)
    out = _bank_bust(surface, now)
    case_id = out["case"]["id"]
    econ = surface.economy
    active = econ.bank_blacklist(KEY, "11", now=now + HOUR)
    assert active and active["case_id"] == case_id
    # one second past the window: lifted, and the book says so
    after = now + 48 * HOUR + 1
    assert econ.bank_blacklist(KEY, "11", now=after) is None
    lifted = econ.bank_blacklist_lifted(KEY, "11", now=after)
    assert lifted and lifted["case_id"] == case_id
    assert lifted["until"] == pytest.approx(now + 48 * HOUR)
    # deposits run again
    assert surface.deposit(KEY, "11", 25, now=after)["ok"]
    # the surface bank status carries both states
    view = surface.bank_status(KEY, "11", now=now + HOUR)
    assert view["blacklist"]["case_id"] == case_id
    view = surface.bank_status(KEY, "11", now=after)
    assert view["blacklist"] is None
    assert view["lifted"]["case_id"] == case_id


def test_bail_changes_custody_only(tmp_path):
    now = time.time()
    surface, _props = _rig(tmp_path, now)
    out = _bank_bust(surface, now)
    case = out["case"]
    bailed = surface.bail(KEY, "11", now=now + 60)
    assert bailed["ok"], bailed
    assert surface.jail_state(KEY, "11", now=now + 60) is None
    # blacklist stands
    dep = surface.deposit(KEY, "11", 50, now=now + 120)
    assert not dep["ok"] and dep["reason"] == "blacklisted"
    # seizure stands — the case's custody holds the whole
    # crew's seized property (11's two + 12's one)
    assert surface.estate.portfolio(KEY, "11") == []
    held = surface.estate.portfolio(
        KEY, f"recovery:{case['id']}")
    assert len(held) == 3


# ── settlement integrity ──────────────────────────────────────

def test_settlement_retry_never_double_settles(tmp_path):
    now = time.time()
    surface, _props = _rig(tmp_path, now)
    out = _bank_bust(surface, now)
    case = out["case"]
    again = surface.heist_case_settle(KEY, case["id"],
                                      now=now + 600)
    assert again["id"] == case["id"]
    assert again["state"] == "committed"
    for member in ("11", "12"):
        assert again["offenders"][member]["seized"] \
            == case["offenders"][member]["seized"]
        assert again["offenders"][member]["blacklist_until"] \
            == case["offenders"][member]["blacklist_until"]
        assert again["offenders"][member]["jailed_until"] \
            == case["offenders"][member]["jailed_until"]
    # exactly one case exists for the heist
    assert surface.economy.case_for_heist(
        KEY, case["heist_id"])["id"] == case["id"]
    # property acquired AFTER the snapshot is not seized,
    # and a retry does not reach for it either
    pid = _buy(surface, "12", "old_town", "room", now + 700)
    surface.heist_case_settle(KEY, case["id"], now=now + 800)
    assert _owner_of(surface, pid) == "12"


def test_settlement_failure_is_recovery_required_not_success(
        tmp_path, monkeypatch):
    now = time.time()
    surface, props = _rig(tmp_path, now)
    real_seize = surface.estate.seize_member_property

    def boom(*args, **kwargs):
        from nomorals.community.group_real_estate import GroupRealEstateError
        raise GroupRealEstateError("estate store on fire")

    monkeypatch.setattr(surface.estate, "seize_member_property",
                        boom)
    out = _bank_bust(surface, now)
    case = out["case"]
    # the bust is announced as a bust, the settlement is NOT
    # announced complete: jail + blacklist landed (they derive
    # from the case record), seizure is still owed
    assert case["state"] == "recovery_required", case
    assert surface.jail_state(KEY, "11", now=now)
    assert surface.economy.bank_blacklist(KEY, "11", now=now)
    assert len(surface.estate.portfolio(KEY, "11")) == 2
    # recovery: the same settle verb finishes the job, once
    monkeypatch.setattr(surface.estate, "seize_member_property",
                        real_seize)
    done = surface.heist_case_settle(KEY, case["id"],
                                     now=now + 300)
    assert done["state"] == "committed"
    assert surface.estate.portfolio(KEY, "11") == []
    assert {s["id"] for s in done["offenders"]["11"]["seized"]} \
        == props["11"]


def test_case_blacklist_and_jail_survive_restart(tmp_path):
    now = time.time()
    surface, _props = _rig(tmp_path, now)
    out = _bank_bust(surface, now)
    case_id = out["case"]["id"]
    # a fresh surface over the same files = a restarted process
    revived = _surface(tmp_path)
    case = revived.economy.heist_case(KEY, case_id)
    assert case and case["state"] == "committed"
    dep = revived.deposit(KEY, "11", 50, now=now + 60)
    assert not dep["ok"] and dep["reason"] == "blacklisted"
    assert revived.jail_state(KEY, "11", now=now + 60)
    assert revived.estate.portfolio(KEY, "11") == []


# ── what never convicts ───────────────────────────────────────

def test_partial_bank_result_never_convicts(tmp_path):
    now = time.time()
    surface, _props = _rig(tmp_path, now)
    # fund the treasury: a partial take must exist to land in
    # the partial band at all (an empty till reads "empty")
    assert surface.economy.contribute_treasury(KEY, "13", 3000,
                                               now=now)
    # a mid roll lands the partial band: nobody jailed, no case
    start = surface.heist_start(KEY, "11", "bank", "harbour",
                                now=now)
    assert start["ok"], start
    assert surface.heist_join(KEY, "12", now=now)["ok"]
    preview = surface.conflict.heist_preview(
        KEY, start["heist"]["id"], now=now)
    odds = float(preview["odds"])
    roll = odds + (1.0 - odds) * 0.10  # inside the partial band
    out = surface.heist_go(KEY, "11", rng=_Rng(roll), now=now)
    assert out["outcome"] == "partial", out
    assert "case" not in out
    assert surface.jail_state(KEY, "11", now=now) is None
    assert surface.deposit(KEY, "11", 50, now=now)["ok"]


def test_vault_bust_jails_but_never_convicts(tmp_path):
    """Robbing a MEMBER's vault is the street crime it always
    was: the bust jails the crew, but it is not a bank robbery
    — no case, no blacklist, no seizure."""
    now = time.time()
    surface, _props = _rig(tmp_path, now)
    start = surface.heist_start(KEY, "11", "vault", "13",
                                now=now)
    assert start["ok"], start
    assert surface.heist_join(KEY, "12", now=now)["ok"]
    out = surface.heist_go(KEY, "11", rng=_Rng(0.99), now=now)
    assert out["outcome"] == "bust", out
    assert "case" not in out
    assert surface.jail_state(KEY, "11", now=now)
    assert surface.deposit(KEY, "11", 50, now=now)["ok"]
    assert len(surface.estate.portfolio(KEY, "11")) == 2


def test_unresolved_and_malformed_never_convict(tmp_path):
    now = time.time()
    surface, _props = _rig(tmp_path, now)
    # no lobby at all
    out = surface.heist_go(KEY, "11", rng=_Rng(0.99), now=now)
    assert not out["ok"]
    # a crew too small to resolve
    start = surface.heist_start(KEY, "11", "bank", "harbour",
                                now=now)
    assert start["ok"], start
    out = surface.heist_go(KEY, "11", rng=_Rng(0.99), now=now)
    assert not out["ok"] and out["reason"] == "crew_small"
    assert surface.economy.case_for_heist(
        KEY, start["heist"]["id"]) is None
    assert surface.deposit(KEY, "11", 50, now=now)["ok"]


# ── chat layer: message-id op identity (Batch B remainder) ────

def _rt_rig(tmp_path: Path):
    from nomorals.agents.partner.runtime_group import RuntimeGroupMixin

    class _Adapter:
        def admin_member(self, chat, user_id):
            return {"status": "administrator"}

    class _RT(RuntimeGroupMixin):
        def _group_adapter(self, platform):  # type: ignore[no-untyped-def]
            return _Adapter()

        def _is_operator(self, message):  # type: ignore[no-untyped-def]
            return False

    rt = _RT()
    surface = _surface(tmp_path)
    rt._group_surface_obj = surface
    return rt, surface


def _msg(*, sender_id="55", sender="Bola", username="bola",
         message_id="1"):
    return SimpleNamespace(
        chat=SimpleNamespace(key=KEY, platform="telegram-bot",
                             kind="group", title="Builders"),
        sender_id=sender_id, sender=sender,
        sender_username=username, text="", meta={},
        message_id=message_id)


def _chat_seed(surface: GroupSurface, uid: str, amount: int,
               name: str) -> None:
    surface.handle_message(KEY, uid, username=f"u{uid}",
                           name=name, text="hello group")
    base = time.time()
    remaining, step = amount, 0
    while remaining > 0:
        chunk = min(190, remaining)
        surface.game_award(KEY, uid, chunk, "ttt",
                           now=base - step * 8 * 86400)
        remaining -= chunk
        step += 1


def test_chat_pay_retry_with_same_message_moves_money_once(
        tmp_path):
    rt, surface = _rt_rig(tmp_path)
    _chat_seed(surface, "55", 500, "Bola")
    _chat_seed(surface, "56", 100, "Ada")
    bal_before = surface.economy.balance(KEY, "56")
    msg = _msg(message_id="pay-msg-1")
    first = rt._control_pay("@u56 100", msg)
    assert "paid" in first.text, first
    bal_after_first = surface.economy.balance(KEY, "56")
    net = bal_after_first - bal_before  # gross minus the tax
    assert net > 0
    # the platform re-delivers the same message (a retry): the
    # member is NOT charged twice — the receipt answers
    second = rt._control_pay("@u56 100", msg)
    assert "paid" in second.text, second
    assert surface.economy.balance(KEY, "56") == bal_after_first
    # a genuinely new message is a new payment of the same net
    rt._control_pay("@u56 100", _msg(message_id="pay-msg-2"))
    assert surface.economy.balance(KEY, "56") \
        == bal_after_first + net


def test_chat_deposit_retry_with_same_message_moves_money_once(
        tmp_path):
    rt, surface = _rt_rig(tmp_path)
    _chat_seed(surface, "55", 500, "Bola")
    msg = _msg(message_id="dep-msg-1")
    rt._control_deposit("100", msg)
    assert surface.economy.bank_balance(KEY, "55") == 100
    rt._control_deposit("100", msg)
    assert surface.economy.bank_balance(KEY, "55") == 100
    rt._control_deposit("100", _msg(message_id="dep-msg-2"))
    assert surface.economy.bank_balance(KEY, "55") == 200


def test_chat_same_card_new_amount_is_a_new_operation(tmp_path):
    """Every tap on one card shares the card's message id. An
    exact retry (same amount) resolves to the receipt; a tap
    for a DIFFERENT amount is a new operation and executes —
    it never answers with the old receipt while claiming the
    new amount moved."""
    rt, surface = _rt_rig(tmp_path)
    _chat_seed(surface, "55", 500, "Bola")
    card_msg = _msg(message_id="card-7")
    rt._control_deposit("100", card_msg)
    assert surface.economy.bank_balance(KEY, "55") == 100
    rt._control_deposit("100", card_msg)  # exact retry: receipt
    assert surface.economy.bank_balance(KEY, "55") == 100
    rt._control_deposit("50", card_msg)   # same card, new op
    assert surface.economy.bank_balance(KEY, "55") == 150
    rt._control_deposit("50", card_msg)   # its own retry dedupes
    assert surface.economy.bank_balance(KEY, "55") == 150


def test_chat_blacklisted_deposit_names_the_case(tmp_path):
    rt, surface = _rt_rig(tmp_path)
    now = time.time()
    for uid, amount in (("55", 9000), ("56", 5000), ("57", 800),
                        ("58", 800), ("59", 800)):
        _chat_seed(surface, uid, amount, f"User {uid}")
    out = surface.gang_create(KEY, "55", "Night Owls", "street",
                              now=now)
    assert out["ok"], out
    for member in ("56", "57", "58", "59"):
        assert surface.gang_join(KEY, member, "Night Owls",
                                 now=now)["ok"]
    start = surface.heist_start(KEY, "55", "bank", "harbour",
                                now=now)
    assert start["ok"], start
    assert surface.heist_join(KEY, "56", now=now)["ok"]
    out = surface.heist_go(KEY, "55", rng=_Rng(0.99), now=now)
    assert out["outcome"] == "bust" and out["case"], out
    reply = rt._control_deposit(
        "50", _msg(sender_id="55", message_id="dep-after-bust"))
    text = getattr(reply, "text", str(reply))
    assert out["case"]["id"] in text, text
    assert "blacklist" in text.lower(), text

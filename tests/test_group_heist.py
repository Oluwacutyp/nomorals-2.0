"""Heists (§5b): bunkers, street scores, the solo bank robbery,
scouting and the targets board — over the shipped crew-heist
loop, never beside it.

Pins, in the brief's own order: a bunker is a real third pocket
(neither wallet nor bank — no police fine, no vault raid and no
wallet robbery can touch it), hide makes the member unreachable
by the sweep and by crime verbs and carries a cooldown, street
scores spawn inside a daily budget and expire (a convoy takes
treasury money in transit, a hoard pays as a system award and
never touches the treasury), the solo bank robbery needs no crew
and no gang and pays 8% capped at 400, and the scout math is the
same math ``resolve_heist`` runs.

Rigs follow the house shapes: GroupSurface over a tmp_path
GroupStore; a stub rng whose rolls the test chooses (the math is
the system's, the dice are the test's).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from nomorals.community.group_conflict import (
    BUNKER_TIERS,
    SCORE_KINDS,
    WANTED_RUNGS,
)
from nomorals.social.chat.group_surface import (
    GroupStore,
    GroupSurface,
)
from tests.frozen_clock import MIDDAY_TS

KEY = "telegram-bot:-1001234567"      # a claimed bot room


class _Rng:
    """The test's dice: random() returns the chosen roll;
    everything else takes the neutral middle."""

    def __init__(self, roll: float = 0.0) -> None:
        self.roll = roll

    def random(self) -> float:
        return self.roll

    def uniform(self, a: float, b: float) -> float:
        return 1.0

    def randint(self, a: int, b: int) -> int:
        return a

    def choice(self, seq):
        return list(seq)[0]


class _SeqRng(_Rng):
    """random() walks a list, then holds the last value."""

    def __init__(self, rolls: list[float]) -> None:
        super().__init__(rolls[0] if rolls else 0.0)
        self._rolls = list(rolls)
        self._i = 0

    def random(self) -> float:
        value = self._rolls[min(self._i, len(self._rolls) - 1)]
        self._i += 1
        return value


def _surface(tmp_path: Path, name: str = "g.json") -> GroupSurface:
    return GroupSurface(GroupStore(tmp_path / name).load())


def _mode(surface: GroupSurface, key: str, mode: str,
          **over) -> None:
    cfg = dict(surface.store.settings(key)["conflict"])
    cfg["mode"] = mode
    cfg.update(over)
    surface.update_settings(key, conflict=cfg)


def _seed(surface: GroupSurface, key: str, uid: str,
          amount: int, *, name: str | None = None) -> int:
    """Presence plus real sparks through the real faucet, spread
    across past weeks the way the economy rig seeds (earn caps
    are real)."""
    surface.handle_message(key, uid, username=f"u{uid}",
                           name=name or f"User {uid}",
                           text="hello group", ts=MIDDAY_TS)
    base = MIDDAY_TS
    remaining, step = amount, 0
    while remaining > 0:
        chunk = min(190, remaining)
        surface.game_award(key, uid, chunk, "ttt",
                           now=base - step * 8 * 86400)
        remaining -= chunk
        step += 1
    return surface.economy.balance(key, uid)


def _gems(surface: GroupSurface, key: str, uid: str,
          amount: int) -> None:
    """Fixture gems through a REAL faucet source (season)."""
    out = surface.estate.credit_gems(
        key, uid, amount, source="season", reason="test seed",
        now=MIDDAY_TS)
    assert out is not None


def _gang5(surface: GroupSurface, now: float) -> None:
    """Leader 11 + members 12..15 — the tier a bank score
    demands.  13 is in the gang but never rides the crew."""
    out = surface.gang_create(KEY, "11", "Night Owls", "street",
                              now=now)
    assert out["ok"], out
    for member in ("12", "13", "14", "15"):
        joined = surface.gang_join(KEY, member, "Night Owls",
                                   now=now)
        assert joined["ok"], joined


def _rig(tmp_path: Path, now: float) -> GroupSurface:
    """Crew (11 leads, 12 rides), bystander 13 in the gang,
    14/15 in the gang, 16 outside it.  Everyone carries wallet
    sparks; the treasury holds 3,000."""
    surface = _surface(tmp_path)
    _mode(surface, KEY, "crime")
    for uid, amount in (("11", 9000), ("12", 5000),
                        ("13", 5000), ("14", 800), ("15", 800),
                        ("16", 500)):
        _seed(surface, KEY, uid, amount)
    _gang5(surface, now)
    assert surface.economy.contribute_treasury(
        KEY, "13", 3000, now=now)
    # the rig's own messages run the conflict beat on the
    # global rng — it may spend the day's score budget and it
    # stamps conflict_cycle_at.  Reset both so every test
    # starts with a clean score day and a live hourly beat.
    state = surface.conflict._load_state(KEY)
    state["scores"] = {}
    state["score_day"] = ""
    state["score_count"] = 0
    surface.conflict._save_state(KEY, state)
    surface._extras(KEY)["conflict_cycle_at"] = 0
    surface.store.save()
    return surface


def _crime_index(surface: GroupSurface, actor: str, target: str,
                 events: int, *, now: float) -> None:
    """Journaled crime events until the bully index climbs —
    the honest way to grow wanted stars."""
    for _ in range(events):
        surface.conflict.note_crime(
            KEY, actor=actor, target=target, verb="rob",
            success=True, take=10, shielded=False, now=now)


# ── bunkers ────────────────────────────────────────────────────

def test_bunker_buy_tiers_and_upgrade_pays_difference(tmp_path):
    surface = _surface(tmp_path)
    _mode(surface, KEY, "crime")
    _seed(surface, KEY, "11", 3000)
    now = MIDDAY_TS
    econ = surface.economy
    before = econ.balance(KEY, "11")
    out = surface.bunker_buy(KEY, "11", 1, now=now)
    assert out["ok"], out
    assert out["bunker"]["tier"] == 1
    assert out["bunker"]["cap"] == BUNKER_TIERS[1]["cap"]
    assert econ.balance(KEY, "11") == before - 200
    # the price is a sink: it lands in the treasury
    assert econ.treasury_balance(KEY) == 200
    # upgrade to tier 3 pays only the difference (1300 sparks
    # + 2 gems), and the stash cap follows the tier
    _gems(surface, KEY, "11", 5)
    out = surface.bunker_buy(KEY, "11", 3, now=now)
    assert out["ok"], out
    assert out["sparks"] == 1300 and out["gems"] == 2
    assert out["bunker"]["cap"] == BUNKER_TIERS[3]["cap"]
    assert surface.estate.gems_balance(KEY, "11") == 3
    # unknown tier, and a member who cannot pay
    assert surface.bunker_buy(KEY, "11", 9, now=now)["reason"] \
        == "tier"
    assert surface.bunker_buy(KEY, "12", 3, now=now)["reason"] \
        == "sparks"


def test_bunker_stash_cap_and_take_roundtrip(tmp_path):
    surface = _surface(tmp_path)
    _mode(surface, KEY, "crime")
    _seed(surface, KEY, "11", 1000)
    now = MIDDAY_TS
    # the seed's "hello" message earns a few sparks of its
    # own — read the wallet, never assume the seed total
    before = surface.economy.balance(KEY, "11")
    assert surface.bunker_buy(KEY, "11", 1, now=now)["ok"]
    # the cap binds: 600 asked, 500 fits
    out = surface.bunker_stash(KEY, "11", 600, now=now)
    assert out["ok"] and out["amount"] == 500, out
    assert out["stored"] == 500
    assert surface.economy.balance(KEY, "11") == before - 200 - 500
    assert surface.bunker_stash(KEY, "11", 10, now=now)["reason"] \
        == "full"
    # take returns to the wallet; taking more than stored pays
    # only what is stored
    out = surface.bunker_take(KEY, "11", 700, now=now)
    assert out["ok"] and out["amount"] == 500, out
    assert surface.economy.balance(KEY, "11") == before - 200
    assert surface.bunker_take(KEY, "11", 10, now=now)["reason"] \
        == "empty"
    # no bunker, no stash
    assert surface.bunker_stash(KEY, "12", 10, now=now)["reason"] \
        == "none"


def test_bunker_stash_is_raid_proof_and_rob_proof(tmp_path):
    surface = _rig(tmp_path, MIDDAY_TS)
    now = MIDDAY_TS
    econ = surface.economy
    # 11 banks 1,000 and stashes 400 in a bolt-hole
    surface.deposit(KEY, "11", 1000, now=now)
    assert surface.bunker_buy(KEY, "11", 1, now=now)["ok"]
    assert surface.bunker_stash(KEY, "11", 400, now=now)["ok"]
    bank_before = econ.bank_balance(KEY, "11")
    # grow 11's wanted index to a raid rung (20 events: base
    # 20 + repeat points from the 3rd hit on — index 38, 4★)
    _crime_index(surface, "11", "13", 20, now=now)
    stars, index, _ = surface.conflict.wanted_stars(
        KEY, "11", now=now)
    assert stars >= 3 and index >= 24, (stars, index)
    rung = WANTED_RUNGS[stars]
    actions = surface.conflict.police_sweep(
        KEY, now=now, free_treasury=1000)
    assert actions and actions[0].get("raid"), actions
    assert econ.bank_balance(KEY, "11") == bank_before - int(
        bank_before * int(rung["bank_pct"]) / 100)
    # the raid took the bank; the stash never moved
    assert surface.conflict.bunker_of(KEY, "11")["sparks"] == 400
    # and a successful wallet robbery cannot take it either:
    # 12 robs 11's wallet (roll 0.0 always succeeds here) and
    # the take is a cut of the wallet alone
    wallet_before = econ.balance(KEY, "11")
    out = surface.rob(KEY, "12", "11", rng=_Rng(0.0), now=now)
    assert out["ok"] and out["success"], out
    assert econ.balance(KEY, "11") == wallet_before - int(
        out["amount"])
    assert surface.conflict.bunker_of(KEY, "11")["sparks"] == 400
    # the stash is promised treasury: the reservation counts it
    reserved = surface.conflict.reservations(KEY)
    assert reserved >= 400


def test_bunker_hide_blocks_police_and_crime(tmp_path):
    surface = _rig(tmp_path, MIDDAY_TS)
    now = MIDDAY_TS
    assert surface.bunker_buy(KEY, "11", 1, now=now)["ok"]
    _crime_index(surface, "11", "13", 20, now=now)
    out = surface.bunker_hide(KEY, "11", now=now)
    assert out["ok"], out
    assert out["until"] == now + 1800
    assert surface.conflict.bunker_hidden(KEY, "11", now=now)
    # the sweep cannot find them: no action, no rung advanced
    actions = surface.conflict.police_sweep(
        KEY, now=now, free_treasury=1000)
    assert actions == []
    # no crime verb can reach them either
    out = surface.crime_verb(KEY, "12", "11", "shoot",
                             rng=_Rng(0.0), now=now)
    assert out["reason"] == "target_hidden", out
    out = surface.rob(KEY, "12", "11", rng=_Rng(0.0), now=now)
    assert out["reason"] == "target_hidden", out
    # once the window ends the street finds them again
    later = now + 1801
    assert not surface.conflict.bunker_hidden(KEY, "11", now=later)
    actions = surface.conflict.police_sweep(
        KEY, now=later, free_treasury=1000)
    assert actions and actions[0]["member"] == "11"


def test_bunker_hide_cooldown(tmp_path):
    surface = _surface(tmp_path)
    _mode(surface, KEY, "crime")
    _seed(surface, KEY, "11", 1000)
    now = MIDDAY_TS
    assert surface.bunker_buy(KEY, "11", 1, now=now)["ok"]
    assert surface.bunker_hide(KEY, "11", now=now)["ok"]
    # still hidden: a second hide is refused, not stacked
    out = surface.bunker_hide(KEY, "11", now=now + 10)
    assert out["reason"] == "hidden"
    # window over, cooldown running: refused with a wait
    out = surface.bunker_hide(KEY, "11", now=now + 1801)
    assert out["reason"] == "cooldown", out
    assert out["wait_s"] == 12 * 3600 - 1
    # cooldown over: hide works again
    out = surface.bunker_hide(KEY, "11", now=now + 1801 + 12 * 3600)
    assert out["ok"], out


def test_bunker_needs_crime_mode(tmp_path):
    surface = _surface(tmp_path)
    _seed(surface, KEY, "11", 1000)
    now = MIDDAY_TS
    assert surface.bunker_buy(KEY, "11", 1, now=now)["reason"] \
        == "mode"
    assert surface.bunker_status(KEY, "11", now=now)["reason"] \
        == "mode"


# ── street scores ──────────────────────────────────────────────

def test_score_spawn_budget_and_expiry(tmp_path):
    surface = _surface(tmp_path)
    _mode(surface, KEY, "crime")
    now = MIDDAY_TS
    cfg = surface.conflict_cfg(KEY)
    assert cfg["scores_per_day"] == 1
    score = surface.conflict.cycle_scores(
        KEY, rng=_Rng(0.0), now=now, cfg=cfg)
    assert score is not None, score
    assert score["kind"] == "convoy"  # roll 0.0 < 0.5
    assert score["prize"] == SCORE_KINDS["convoy"]["prize"][0]
    assert score["district"] == "harbour"  # first, sorted
    # one open at a time and one a day: no second spawn
    assert surface.conflict.cycle_scores(
        KEY, rng=_Rng(0.0), now=now + 60, cfg=cfg) is None
    # the window passes: expiry closes it, and the next day a
    # new score can open
    expired = surface.conflict.expire_scores(
        KEY, now=now + int(cfg["score_ttl_s"]) + 1)
    assert len(expired) == 1
    assert surface.conflict.open_scores(
        KEY, now=now + int(cfg["score_ttl_s"]) + 1) == []
    score2 = surface.conflict.cycle_scores(
        KEY, rng=_Rng(0.0), now=now + 86400, cfg=cfg)
    assert score2 is not None


def test_convoy_heist_takes_from_treasury(tmp_path):
    surface = _rig(tmp_path, MIDDAY_TS)
    now = MIDDAY_TS
    cfg = surface.conflict_cfg(KEY)
    score = surface.conflict.cycle_scores(
        KEY, rng=_Rng(0.0), now=now, cfg=cfg)
    assert score is not None and score["kind"] == "convoy"
    treasury_before = surface.economy.treasury_balance(KEY)
    assert treasury_before == 3000
    start = surface.heist_start(KEY, "11", "convoy",
                                score["id"], now=now)
    assert start["ok"], start
    assert surface.heist_join(KEY, "12", now=now)["ok"]
    preview = surface.conflict.heist_preview(
        KEY, start["heist"]["id"], now=now)
    # naked crew (offense 4) vs the van (14 + harbour guard 5)
    assert preview["defense"] == 19, preview
    assert preview["offense"] == 4
    # even split, the remainder to the leader (none here)
    wallet11 = surface.economy.balance(KEY, "11")
    wallet12 = surface.economy.balance(KEY, "12")
    out = surface.heist_go(KEY, "11", rng=_Rng(0.0), now=now)
    assert out["ok"] and out["outcome"] == "success", out
    assert out["take"] == 300  # the manifest, in full
    assert surface.economy.treasury_balance(KEY) \
        == treasury_before - 300
    assert surface.economy.balance(KEY, "11") == wallet11 + 150
    assert surface.economy.balance(KEY, "12") == wallet12 + 150
    # the score is consumed by the attempt
    assert surface.conflict.open_scores(KEY, now=now) == []


def test_hoard_heist_pays_as_award_and_leaves_treasury(tmp_path):
    surface = _rig(tmp_path, MIDDAY_TS)
    now = MIDDAY_TS
    cfg = surface.conflict_cfg(KEY)
    # the seed's first 190-chunk lands today and the daily earn
    # cap is 200 — the hoard runs tomorrow, when the cap is
    # fresh (the same discipline the drop tests follow)
    late = now + 86400
    # chance 0.0 passes, kind 0.9 picks the hoard
    score = surface.conflict.cycle_scores(
        KEY, rng=_SeqRng([0.0, 0.9]), now=late, cfg=cfg)
    assert score is not None and score["kind"] == "hoard", score
    assert score["prize"] == SCORE_KINDS["hoard"]["prize"][0]
    treasury_before = surface.economy.treasury_balance(KEY)
    start = surface.heist_start(KEY, "11", "hoard", score["id"],
                                now=late)
    assert start["ok"], start
    assert surface.heist_join(KEY, "12", now=late)["ok"]
    out = surface.heist_go(KEY, "11", rng=_Rng(0.0), now=late)
    assert out["ok"] and out["outcome"] == "success", out
    assert out["take"] == 150  # the hoard, in full
    # the prize is a system award: the treasury never moves
    assert surface.economy.treasury_balance(KEY) \
        == treasury_before
    # and the award rides the ledger as an earn, referenced to
    # the score — not as a treasury move
    entries = surface.economy._load_json(KEY, "ledger", [])
    assert any(e.get("ref") == f"score:{score['id']}"
               and e.get("kind") == "earn" for e in entries)


def test_expired_score_refuses_start_and_closes_lobby(tmp_path):
    surface = _rig(tmp_path, MIDDAY_TS)
    now = MIDDAY_TS
    cfg = surface.conflict_cfg(KEY)
    score = surface.conflict.cycle_scores(
        KEY, rng=_Rng(0.0), now=now, cfg=cfg)
    assert score is not None
    late = now + int(cfg["score_ttl_s"]) + 1
    # a lobby formed in time still dies when the window closes
    start = surface.heist_start(KEY, "11", "convoy", score["id"],
                                now=now)
    assert start["ok"], start
    assert surface.heist_join(KEY, "12", now=now)["ok"]
    out = surface.heist_go(KEY, "11", rng=_Rng(0.0), now=late)
    assert out["reason"] == "expired", out
    state = surface.conflict._load_state(KEY)
    assert state["heists"][start["heist"]["id"]]["status"] \
        == "expired"
    # and a fresh start after the window is refused the same way
    out = surface.heist_start(KEY, "11", "convoy", score["id"],
                              now=late)
    assert out["reason"] == "expired", out


def test_conflict_cycle_announces_a_score(tmp_path):
    surface = _rig(tmp_path, MIDDAY_TS)
    now = MIDDAY_TS
    notices = surface.conflict_cycle(KEY, rng=_Rng(0.0), now=now)
    assert any("score is open" in n for n in notices), notices
    assert surface.conflict.open_scores(KEY, now=now)


# ── solo robbery ───────────────────────────────────────────────

def test_rob_bank_solo_success_and_bust(tmp_path):
    surface = _rig(tmp_path, MIDDAY_TS)
    now = MIDDAY_TS
    econ = surface.economy
    treasury_before = econ.treasury_balance(KEY)
    # naked member 11 vs the harbour bank (defense 12): odds
    # 0.5 + (2 - 12) / 60 — roll 0.0 succeeds
    out = surface.rob_bank(KEY, "11", "harbour",
                           rng=_Rng(0.0), now=now)
    assert out["ok"] and out["outcome"] == "success", out
    assert out["take"] == min(int(3000 * 0.08), 400)
    assert econ.treasury_balance(KEY) == treasury_before - 240
    # the attempt journals as a crime event, verb robbery
    state = surface.conflict._load_state(KEY)
    verbs = [e.get("verb") for e in state["journal"]
             if e.get("kind") == "crime"]
    assert "robbery" in verbs
    # member 12, fresh cooldown, rolls the bust: 30 minutes
    out = surface.rob_bank(KEY, "12", "harbour",
                           rng=_Rng(0.99), now=now)
    assert out["ok"] and out["outcome"] == "bust", out
    assert out["take"] == 0
    jail = surface.jail_state(KEY, "12", now=now)
    assert jail and jail["until"] > now + 29 * 60


def test_rob_bank_needs_no_gang_and_no_crew(tmp_path):
    surface = _rig(tmp_path, MIDDAY_TS)
    now = MIDDAY_TS
    # 16 is outside the gang entirely — the lone wolf still
    # rides, and the treasury stays out of solo reach only in
    # the sense that the take is small and capped
    out = surface.rob_bank(KEY, "16", "old_town",
                           rng=_Rng(0.0), now=now)
    assert out["ok"] and out["outcome"] == "success", out
    assert out["take"] == 240
    # the treasury itself is crew-only: no solo verb names it
    out = surface.conflict.rob_institution(
        KEY, "16", "treasury", "", rng=_Rng(0.0), now=now)
    assert out["reason"] == "target"


# ── scouting and the targets board ─────────────────────────────

def test_heist_targets_board(tmp_path):
    surface = _rig(tmp_path, MIDDAY_TS)
    now = MIDDAY_TS
    # 13's wallet funded the treasury in the rig, so the vault
    # belongs to 12 (wallet 5,000+)
    assert surface.deposit(KEY, "12", 3000, now=now)["ok"]
    cfg = surface.conflict_cfg(KEY)
    score = surface.conflict.cycle_scores(
        KEY, rng=_Rng(0.0), now=now, cfg=cfg)
    out = surface.heist_targets(KEY, now=now)
    assert out["ok"], out
    assert out["treasury"]["available"] == 3000
    assert out["treasury"]["defense"] == 20
    assert len(out["banks"]) == 4
    assert {b["district"] for b in out["banks"]} == {
        "old_town", "market_mile", "harbour", "heights"}
    assert all(b["defense"] >= 12 for b in out["banks"])
    assert len(out["scores"]) == 1
    assert out["scores"][0]["id"] == score["id"]
    assert out["scores"][0]["defense"] == 19
    assert out["vaults"] == 1
    assert out["vault_total"] == 3000


def test_scout_math_is_the_heist_math(tmp_path):
    surface = _rig(tmp_path, MIDDAY_TS)
    now = MIDDAY_TS
    start = surface.heist_start(KEY, "11", "treasury", "",
                                now=now)
    assert start["ok"], start
    assert surface.heist_join(KEY, "12", now=now)["ok"]
    preview = surface.conflict.heist_preview(
        KEY, start["heist"]["id"], now=now)
    scout = surface.heist_scout(KEY, "11", "treasury", "",
                                now=now)
    assert scout["ok"], scout
    # one rider, not two: the offense is the leader's alone
    assert scout["member_offense"] == 2
    assert scout["defense"] == preview["defense"] == 20
    assert scout["odds"] < preview["odds"]
    assert scout["needed_offense"] == 20  # defense + 4 * 0 heat
    assert surface.heist_scout(KEY, "11", "vault", "11",
                               now=now)["reason"] == "self"
    assert surface.heist_scout(KEY, "11", "bank", "nowhere",
                               now=now)["reason"] == "district"


# ── the brain tool ─────────────────────────────────────────────

class _Settings:
    """The tool's settings surface: workspace + owner id."""

    def __init__(self, workspace: str, owner: str) -> None:
        self.workspace_dir = workspace
        self._owner = owner

    def get(self, dotted: str, default=None):
        if dotted == "partner.owner_sender_ids":
            return self._owner
        return default


class _Registry:
    """A registry stub: register() only, no builtin imports."""

    def __init__(self, context) -> None:
        self.context = context
        self.tools: dict = {}

    def register(self, name, description="", capability=None):
        def deco(fn):
            self.tools[name] = fn
            return fn
        return deco


def _tool(tmp_path: Path):
    from nomorals.tools import group_heist

    context = SimpleNamespace(
        settings=_Settings(str(tmp_path), "11"))
    registry = _Registry(context)
    group_heist.register(registry)
    return registry.tools["group_heist"]


def _tool_surface(tmp_path: Path) -> GroupSurface:
    """The fixture surface over the SAME store the tool reads
    (default_surface: <ws>/data/social/groups.json)."""
    return GroupSurface(GroupStore(
        tmp_path / "data" / "social" / "groups.json").load())


def _tool_rig(tmp_path: Path, now: float) -> GroupSurface:
    surface = _tool_surface(tmp_path)
    _mode(surface, KEY, "crime")
    for uid, amount in (("11", 9000), ("12", 5000),
                        ("13", 5000), ("14", 800), ("15", 800)):
        _seed(surface, KEY, uid, amount)
    _gang5(surface, now)
    assert surface.economy.contribute_treasury(
        KEY, "13", 3000, now=now)
    surface.store.save()
    return surface


def test_group_heist_tool_targets_scout_bunker(tmp_path):
    _tool_rig(tmp_path, MIDDAY_TS)
    tool = _tool(tmp_path)
    out = tool(action="targets", group="-1001234567")
    assert out["ok"], out
    assert "treasury" in out["text"]
    assert "3,000" in out["text"]
    out = tool(action="scout", group="-1001234567",
               target="treasury")
    assert out["ok"], out
    assert "odds 20%" in out["text"], out
    out = tool(action="bunker", group="-1001234567",
               bunker="buy 1")
    assert out["ok"], out
    assert "Bolt-hole" in out["text"]
    out = tool(action="bunker", group="-1001234567",
               bunker="stash 400")
    assert out["ok"], out
    assert "400" in out["text"]
    out = tool(action="bunker", group="-1001234567", bunker="")
    assert out["ok"], out
    assert "400" in out["text"]


def test_group_heist_tool_unknown_group_is_honest(tmp_path):
    _tool_rig(tmp_path, MIDDAY_TS)
    tool = _tool(tmp_path)
    out = tool(action="targets", group="nowhere")
    assert out["ok"]
    assert "no record of a group matching" in out["text"]
    out = tool(action="nope", group="-1001234567")
    assert not out["ok"]

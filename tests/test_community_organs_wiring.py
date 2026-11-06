"""Dormant community organs pass (AA-04/05/18/21/22/27/30): the wiring.

Every test drives the PRODUCTION surface — the community chat
controls, the partner command catalog, or the main tool registry —
never the organ in isolation. On the pristine tree these fail:
control_miniapp has no wave-2 families, /party and /ajo do not
exist as commands, the tool registry carries no community tools,
SessionRegistry cannot share sessions across instances, and ending
a recorded room archives nothing.
"""

from __future__ import annotations

import re
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from nomorals.community.ajo import AjoStore, control_ajo
from nomorals.community.audio import AudioStore
from nomorals.community.audio_rooms import RoomStore, control_room
from nomorals.community.group_economy import GroupEconomy
from nomorals.community.groups import GroupStore, control_cgroup
from nomorals.community.miniapps import control_miniapp
from nomorals.community.miniapps2 import MiniApp2Store
from nomorals.community.orchestration import (
    ORCHESTRATION_JOB,
    SessionRegistry,
    ensure_orchestration_job,
    run_orchestration_tick,
)
from nomorals.community.parties import TRIVIA_POINTS, control_party
from nomorals.community.parties_quick import QuickPartyStore
from nomorals.community.token_economy import persistent_token_economy


def _ctx(root: str):
    return SimpleNamespace(
        settings=SimpleNamespace(community_dir=root))


def _chat(chat_id: str = "42"):
    return SimpleNamespace(kind="group", platform="telegram",
                           chat_id=chat_id, thread_id="")


def _id_after(text: str, marker: str) -> str:
    match = re.search(re.escape(marker) + r"\s+(\S+)", text)
    assert match, f"no id after {marker!r} in: {text}"
    return match.group(1)


# ── AA-04: miniapps wave 2 through /miniapp ─────────────────────────────


def test_miniapp_wave2_carpool_persists():
    root = tempfile.mkdtemp()
    ctx, chat = _ctx(root), _chat()
    out = control_miniapp("carpool new Lekki -> Ikeja | Sat 8:00 | 2",
                          context=ctx, chat=chat, sender_id="u1",
                          sender="Ada")
    ride_id = _id_after(out, "join")
    joined = control_miniapp(f"carpool join {ride_id}", context=ctx,
                             chat=chat, sender_id="u2", sender="Bob")
    assert "Bob is riding" in joined
    # State really persisted: a fresh store reads the ride + rider.
    store = MiniApp2Store(Path(root) / "miniapps2")
    rides = store.list("telegram:42", "carpools")
    assert len(rides) == 1 and rides[0]["driver_name"] == "Ada"
    assert [p["display_name"] for p in rides[0]["riders"]] == ["Bob"]


def test_miniapp_wave2_rota_market_league_split():
    root = tempfile.mkdtemp()
    ctx, chat = _ctx(root), _chat()

    rota_id = _id_after(control_miniapp(
        "rota new Cleaning", context=ctx, chat=chat, sender_id="u1",
        sender="Ada"), "shift")
    control_miniapp(f"rota shift {rota_id} 2026-11-07 morning",
                    context=ctx, chat=chat, sender_id="u1", sender="Ada")
    assigned = control_miniapp(f"rota assign {rota_id} 2026-11-07 morning",
                               context=ctx, chat=chat, sender_id="u2",
                               sender="Bob")
    assert "to Bob" in assigned

    listing_id = _id_after(control_miniapp(
        "market new sell Rice cooker | 15000 | barely used", context=ctx,
        chat=chat, sender_id="u1", sender="Ada"), "market offer")
    control_miniapp(f"market offer {listing_id} 12000", context=ctx,
                    chat=chat, sender_id="u2", sender="Bob")
    shown = control_miniapp(f"market show {listing_id}", context=ctx,
                            chat=chat, sender_id="u1", sender="Ada")
    offer_id = re.search(r"`(offer\w+)`", shown).group(1)
    deal = control_miniapp(f"market accept {listing_id} {offer_id}",
                           context=ctx, chat=chat, sender_id="u1",
                           sender="Ada")
    assert "Deal!" in deal
    store = MiniApp2Store(Path(root) / "miniapps2")
    listing = store.get("telegram:42", "listings", listing_id)
    assert listing is not None and listing["status"] == "closed"

    league_id = _id_after(control_miniapp(
        "league new Weekend League", context=ctx, chat=chat,
        sender_id="u1", sender="Ada"), "team")
    control_miniapp(f"league team {league_id} Rockets", context=ctx,
                    chat=chat, sender_id="u1", sender="Ada")
    control_miniapp(f"league team {league_id} Comets", context=ctx,
                    chat=chat, sender_id="u1", sender="Ada")
    board = control_miniapp(f"league standings {league_id}", context=ctx,
                            chat=chat, sender_id="u1", sender="Ada")
    team_ids = re.findall(r"`(team\w+)`", board)
    result = control_miniapp(
        f"league result {league_id} {team_ids[0]} 2-1 {team_ids[1]}",
        context=ctx, chat=chat, sender_id="u1", sender="Ada")
    assert "Result booked" in result
    board = control_miniapp(f"league standings {league_id}", context=ctx,
                            chat=chat, sender_id="u1", sender="Ada")
    assert "3 pts" in board

    bill_id = _id_after(control_miniapp(
        "split new House bills", context=ctx, chat=chat, sender_id="u1",
        sender="Ada"), "add")
    control_miniapp(f"split add {bill_id} 3000 dinner for ada,bob",
                    context=ctx, chat=chat, sender_id="u1", sender="Ada")
    shown = control_miniapp(f"split show {bill_id}", context=ctx,
                            chat=chat, sender_id="u1", sender="Ada")
    # "ada" in the for-list IS the payer Ada: one person, a 2-way
    # split — never a phantom second Ada owing herself.
    assert "Ada: is owed 1,500" in shown
    assert "ada: owes" not in shown
    assert "bob pays Ada 1,500" in shown


# ── AA-05: parties through /party ───────────────────────────────────────


def _make_quiz(ctx, chat) -> None:
    out = control_miniapp('new quiz "Capital of France?" Paris London '
                          'Berlin correct:1', context=ctx, chat=chat,
                          sender_id="u1", sender="Ada")
    assert "Created" in out


def _run_party(ctx, chat, title: str, *, trivia_win_by: str = "") -> str:
    control_party(f"new {title}", context=ctx, chat=chat,
                  sender_id="u1", sender="Ada")
    control_party("start", context=ctx, chat=chat, sender_id="u1",
                  sender="Ada")
    if trivia_win_by:
        control_party("trivia", context=ctx, chat=chat, sender_id="u1",
                      sender="Ada")
        control_party("answer 1", context=ctx, chat=chat,
                      sender_id=trivia_win_by, sender="Winner")
    return control_party("end", context=ctx, chat=chat, sender_id="u1",
                         sender="Ada")


def test_party_lifecycle_recap_and_lastnight():
    root = tempfile.mkdtemp()
    ctx, chat = _ctx(root), _chat()
    created = control_party("new Friday Turnup | amapiano", context=ctx,
                            chat=chat, sender_id="u1", sender="Ada")
    assert "Friday Turnup" in created
    control_party("rsvp yes", context=ctx, chat=chat, sender_id="u2",
                  sender="Bob")
    started = control_party("start", context=ctx, chat=chat,
                            sender_id="u1", sender="Ada")
    assert "LIVE" in started
    queued = control_party("request Soso", context=ctx, chat=chat,
                           sender_id="u2", sender="Bob")
    track_id = _id_after(queued, "vote")
    control_party(f"vote {track_id}", context=ctx, chat=chat,
                  sender_id="u1", sender="Ada")
    now = control_party("next", context=ctx, chat=chat, sender_id="u1",
                        sender="Ada")
    assert "Soso" in now
    control_party("wall what a night", context=ctx, chat=chat,
                  sender_id="u2", sender="Bob")
    recap = control_party("end", context=ctx, chat=chat, sender_id="u1",
                          sender="Ada")
    assert "that's a wrap" in recap and "Soso" in recap
    card = control_party("lastnight", context=ctx, chat=chat,
                         sender_id="u1", sender="Ada")
    assert "Last night" in card


def test_party_trivia_credits_the_one_group_economy():
    root = tempfile.mkdtemp()
    ctx, chat = _ctx(root), _chat()
    _make_quiz(ctx, chat)
    economy_root = Path(root) / "group_economy"
    group_id = "telegram:42"
    # The first three parties are rehearsals: a trivia win is scored
    # but credits NOTHING (dry-run), by the manager's own law.
    for n in range(3):
        _run_party(ctx, chat, f"Rehearsal {n}", trivia_win_by="u2")
    economy = GroupEconomy(economy_root)
    assert economy.balance(group_id, "u2") == 0
    # Fourth party is real. The winner is a verified member of the
    # token economy, so the credit lands in the shared ledger.
    facade = persistent_token_economy(group_id, data_dir=economy_root)
    facade.verify("u2", "fingerprint-u2")
    _run_party(ctx, chat, "The Real One", trivia_win_by="u2")
    economy = GroupEconomy(economy_root)
    assert economy.balance(group_id, "u2") == TRIVIA_POINTS


def test_party_quick_lifecycle_and_unique_ids():
    root = tempfile.mkdtemp()
    ctx, chat = _ctx(root), _chat()
    control_party("quick new Fast One", context=ctx, chat=chat,
                  sender_id="u1", sender="Ada")
    started = control_party("quick start", context=ctx, chat=chat,
                            sender_id="u1", sender="Ada")
    assert "LIVE" in started
    control_party("quick request Essence", context=ctx, chat=chat,
                  sender_id="u2", sender="Bob")
    control_party("quick vote 1", context=ctx, chat=chat, sender_id="u1",
                  sender="Ada")
    now = control_party("quick next", context=ctx, chat=chat,
                        sender_id="u1", sender="Ada")
    assert "Essence" in now
    recap = control_party("quick end", context=ctx, chat=chat,
                          sender_id="u1", sender="Ada")
    assert "wrap" in recap
    control_party("quick new Second Spin", context=ctx, chat=chat,
                  sender_id="u1", sender="Ada")
    store = QuickPartyStore(Path(root) / "parties_quick")
    parties = store.load("telegram:42")
    assert len(parties) == 2
    assert parties[0].party_id != parties[1].party_id


# ── AA-21: the token economy IS the group economy ───────────────────────


def test_token_economy_shares_the_group_economy_ledger():
    root = Path(tempfile.mkdtemp())
    facade = persistent_token_economy("g1", data_dir=root)
    facade.verify("alice", "fp-alice")
    facade.verify("bob", "fp-bob")
    entry = facade.earn("alice", 120, "mining")
    assert entry is not None
    assert facade.transfer("alice", "bob", 50)
    facade.to_treasury("alice", 20)

    economy = GroupEconomy(root)
    assert economy.balance("g1", "alice") == facade.balance("alice") == 50
    assert economy.balance("g1", "bob") == facade.balance("bob") == 50
    assert economy.treasury_balance("g1") == 20
    # One ledger only — the old parallel token files never appear.
    assert (root / "g1" / "ledger.json").exists()
    assert not list(root.rglob("token_ledger.json"))
    assert not list(root.rglob("token_balances.json"))


# ── AA-22: ajo through /ajo ─────────────────────────────────────────────


def test_ajo_full_cycle_nets_zero_through_chat():
    root = tempfile.mkdtemp()
    ctx, chat = _ctx(root), _chat()
    out = control_ajo("new 5000 Rent circle", context=ctx, chat=chat,
                      sender_id="u1", sender="Ada")
    ajo_id = _id_after(out, "join")
    members = [("u1", "Ada"), ("u2", "Bob"), ("u3", "Cid")]
    for uid, name in members[1:]:
        control_ajo(f"join {ajo_id}", context=ctx, chat=chat,
                    sender_id=uid, sender=name)
    for _round in range(3):
        for uid, name in members:
            paid = control_ajo(f"pay {ajo_id}", context=ctx, chat=chat,
                               sender_id=uid, sender=name)
            assert "paid" in paid
        payout = control_ajo(f"payout {ajo_id}", context=ctx, chat=chat,
                             sender_id="u1", sender="Ada")
        assert "Pot paid" in payout
    # Full cycle: everyone contributed 3 pots and took exactly one.
    store = AjoStore(Path(root) / "ajo")
    group = store.get(ajo_id)
    assert group is not None
    assert all(v == 0 for v in group.balances().values())
    balances = control_ajo(f"balance {ajo_id}", context=ctx, chat=chat,
                           sender_id="u1", sender="Ada")
    assert "Ada: 0" in balances


# ── AA-27: community tools in the production registry ───────────────────


def test_community_tools_registered_in_main_tool_registry():
    from nomorals.community.registry import COMMUNITY_TOOL_NAMES
    from nomorals.tools.registry import ToolRegistry

    registry = ToolRegistry().register_builtins()
    missing = [n for n in COMMUNITY_TOOL_NAMES if n not in registry._tools]
    assert missing == []
    failed = [f for f in registry._failed_modules
              if f.get("module") == "community"]
    assert failed == []
    roll = registry._tools["roll"]
    assert roll.capability == "community.fun"
    outcome = roll.fn("2d6")
    assert isinstance(outcome, str) and outcome.strip()


# ── AA-18: orchestration fleet + tick + job ─────────────────────────────


def test_orchestration_fleet_registry_tick_and_job():
    root = tempfile.mkdtemp()
    sessions_path = str(Path(root) / "sessions.json")
    first = SessionRegistry(path=sessions_path)
    first.register("device-a", kind="phone")
    second = SessionRegistry(path=sessions_path)
    assert "device-a" in second.health()  # shared across instances

    ctx, chat = _ctx(root), _chat()
    control_cgroup("schedule 0 hello fleet", context=ctx, chat=chat,
                   sender="Ada", sender_id="u1")
    sent: list[tuple[str, str, str]] = []

    def sender(platform: str, target: str, text: str) -> bool:
        sent.append((platform, target, text))
        return True

    summary = run_orchestration_tick(sender, context=ctx)
    assert summary["dispatched"] and summary["sent"] == 1
    assert sent == [("telegram", "telegram:42", "hello fleet")]
    again = run_orchestration_tick(sender, context=ctx)
    assert again["sent"] == 0 and len(sent) == 1  # one-shot: never twice

    devices = control_cgroup("devices", context=ctx, chat=chat,
                             sender="Ada", sender_id="u1")
    assert "Devices" in devices

    class FakeScheduler:
        def __init__(self) -> None:
            self.jobs: list[dict] = []

        def list_jobs(self):
            return list(self.jobs)

        def add(self, name, spec, kind, payload, routine=False):
            self.jobs.append({"name": name, "routine": routine})
            return "job-1"

    scheduler = FakeScheduler()
    ensure_orchestration_job(scheduler)
    ensure_orchestration_job(scheduler)  # idempotent
    assert [j["name"] for j in scheduler.jobs] == [ORCHESTRATION_JOB]
    assert scheduler.jobs[0]["routine"] is True


# ── AA-30: room recordings land in the audio archive ────────────────────


def test_room_end_archives_recording_and_clips():
    root = tempfile.mkdtemp()
    ctx, chat = _ctx(root), _chat()
    groups = GroupStore(data_dir=Path(root) / "cgroups")
    group = groups.create_group("Devs", "coding", created_by="u1")
    rooms = RoomStore(data_dir=Path(root) / "rooms")
    kw = dict(context=ctx, chat=chat, sender="Ada", sender_id="u1",
              store=rooms, group_store=groups)
    control_room(f"create {group.id} | Archive Test", **kw)
    room = rooms.list(community_id=group.id)[0]
    room.created_at = time.time() - 3600  # an hour-long room
    rooms.put(room)
    control_room(f"record {room.room_id} on", **kw)
    ended = control_room(f"end {room.room_id}", **kw)
    assert "saved to the archive" in ended

    archive = AudioStore(data_dir=Path(root) / "audio")
    recordings = archive.list_recordings(group.id)
    assert len(recordings) == 1
    rec = recordings[0]
    assert rec.room_id == room.room_id and rec.duration_s >= 3500

    listing = control_room(f"recordings {group.id}", **kw)
    assert "Archive Test" in listing
    cut = control_room(f"clip {group.id} {rec.id} 10 70 intros", **kw)
    assert "clip saved" in cut
    control_room(f"attach {group.id} {rec.id} Ada | welcome everyone", **kw)
    stats = control_room(f"speakers {group.id} {rec.id}", **kw)
    assert "Ada" in stats and "2 words" in stats


# ── catalog: /party + /ajo are real commands ────────────────────────────


def test_party_and_ajo_are_catalogued_commands():
    from nomorals.social.chat.control import (
        COMMAND_DETAILS,
        LIST_GROUPS,
        parse_control,
    )

    for text, kind in (("/party new Gala", "party"), ("/ajo list", "ajo")):
        command = parse_control(text)
        assert command is not None and command.kind == kind
    listed = {name for _group, names in LIST_GROUPS for name in names}
    assert {"party", "ajo"} <= listed
    assert {"party", "ajo"} <= set(COMMAND_DETAILS)

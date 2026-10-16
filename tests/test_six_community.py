"""Tests for the second-wave community modules.

Covers: ajo, miniapps2, audio, bridges, group_memory, persona, insights.

1. ISOLATION (the contract): AST walk + import-line grep over the NEW
   community files — zero imports of owner-data modules (memory,
   accounts, connectors, anything with "vault" in it). The check is
   import-aware: docstrings may discuss memory/accounts, but no import
   statement may touch them. Group memory state must live under the
   community dir, never in owner storage.
2. Ajo ledger: contributions + payout conserve money (balances sum to
   zero after a full round), turn order rotates, incomplete rounds
   refuse payout.
3. Split: shares sum exactly to the total; settlements zero balances.
4. Bridges: routing goes to OTHER platforms only, never the origin;
   with no injected senders nothing is sent.
"""

import ast
import re
from pathlib import Path

import pytest

from nomorals.community import (
    ajo as ajo_mod,
)
from nomorals.community import (
    audio as audio_mod,
)
from nomorals.community import (
    bridges as bridges_mod,
)
from nomorals.community import (
    group_memory as group_memory_mod,
)
from nomorals.community import (
    insights as insights_mod,
)
from nomorals.community import (
    miniapps2 as miniapps2_mod,
)
from nomorals.community import (
    persona as persona_mod,
)

COMMUNITY_DIR = Path(__file__).resolve().parent.parent / "nomorals" / "community"
NEW_FILES = (
    "ajo.py",
    "miniapps2.py",
    "audio.py",
    "bridges.py",
    "group_memory.py",
    "persona.py",
    "insights.py",
)

FORBIDDEN_IMPORT_PREFIXES = (
    "nomorals.memory",
    "nomorals.accounts",
    "nomorals.connectors",
    "memory",
    "accounts",
    "connectors",
)
FORBIDDEN_IMPORT_SUBSTRINGS = ("vault",)
# Relative imports inside the package resolve to these safe modules only.
# ("group_memory" is fine: it is the group-scoped store, not owner memory.)
SAFE_RELATIVE = {"group_memory", "core", "tools"}


def _imported_modules(tree: ast.AST) -> list[str]:
    mods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mods.append(node.module)
    return mods


def test_new_files_have_zero_forbidden_imports_ast():
    violations: list[str] = []
    for name in NEW_FILES:
        tree = ast.parse((COMMUNITY_DIR / name).read_text(encoding="utf-8"))
        for mod in _imported_modules(tree):
            if mod.startswith(FORBIDDEN_IMPORT_PREFIXES):
                violations.append(f"{name}: imports {mod}")
            if any(s in mod for s in FORBIDDEN_IMPORT_SUBSTRINGS):
                violations.append(f"{name}: imports {mod} (vault?)")
            # Relative imports (level > 0) show up with their bare module
            # name; guard the dangerous bare names too.
            if mod in ("memory", "accounts", "connectors"):
                violations.append(f"{name}: bare import {mod}")
    assert not violations, "isolation breach:\n" + "\n".join(violations)


def test_new_files_have_zero_forbidden_imports_grep():
    """Line-level grep: only import/from lines are checked, so
    docstrings discussing memory/accounts do not false-positive."""
    import_re = re.compile(r"^\s*(import|from)\s+(\S+)")
    violations: list[str] = []
    for name in NEW_FILES:
        for lineno, line in enumerate(
            (COMMUNITY_DIR / name).read_text(encoding="utf-8").splitlines(), 1
        ):
            match = import_re.match(line)
            if not match:
                continue
            mod = match.group(2)
            if mod.startswith(FORBIDDEN_IMPORT_PREFIXES):
                violations.append(f"{name}:{lineno}: {line.strip()}")
            if any(s in mod for s in FORBIDDEN_IMPORT_SUBSTRINGS):
                violations.append(f"{name}:{lineno}: {line.strip()}")
    assert not violations, "isolation breach (grep):\n" + "\n".join(violations)


def test_group_memory_state_lives_under_community_dir():
    root = group_memory_mod.default_community_root()
    assert str(root).endswith(str(Path(".devon") / "community"))
    assert ".devon" in root.parts
    # A root inside ~/.devon but outside ~/.devon/community is refused.
    with pytest.raises(ValueError):
        group_memory_mod.GroupMemoryStore(Path.home() / ".devon" / "owner_db")


def test_group_memory_cross_group_isolation(tmp_path):
    store = group_memory_mod.GroupMemoryStore(tmp_path)
    store.remember("group-a", "rule", "no spam", author_name="Ada")
    store.remember("group-b", "rule", "memes only", author_name="Bola")
    assert store.recall("group-a", "rule").value == "no spam"
    assert store.recall("group-b", "rule").value == "memes only"
    # A key that exists only in another group is invisible here.
    store.remember("group-a", "secret-a", "aaa")
    assert store.recall("group-b", "secret-a") is None
    # Search never crosses groups.
    assert [e.value for e in store.search("group-b", "aaa")] == []
    assert [e.value for e in store.search("group-a", "aaa")] == ["aaa"]
    # Namespaced keys carry the group id.
    assert store.recall("group-a", "rule").namespaced_key == "group-a:rule"
    # Path traversal via group id is refused.
    with pytest.raises(ValueError):
        store.remember("../escape", "k", "v")
    # Forgetting in one group leaves the other untouched.
    assert store.forget("group-a", "rule") is True
    assert store.recall("group-b", "rule").value == "memes only"


def test_persona_grounded_only_in_group_history(tmp_path):
    store = group_memory_mod.GroupMemoryStore(tmp_path)
    store.remember("g1", "persona:name", "GistBot")
    store.remember("g1", "persona:traits", "playful; short replies")
    store.remember("g1", "meeting_day", "Fridays")
    store.remember("g2", "meeting_day", "Mondays")
    prompt = persona_mod.persona_prompt(store, "g1")
    assert "GistBot" in prompt
    assert "playful" in prompt
    assert "Fridays" in prompt
    assert "Mondays" not in prompt  # other group's fact never leaks in
    assert "owner" in prompt.lower()  # the no-owner-data rule is stated
    # Empty history: honest, no invented facts.
    empty = persona_mod.persona_prompt(store, "g-fresh")
    assert "no stored history" in empty
    assert "Fridays" not in empty


# ── ajo ─────────────────────────────────────────────────────────────────
def _three_member_circle():
    group = ajo_mod.create_ajo("g1", "Friday Circle", 1000)
    group.add_member("u1", "Ada")
    group.add_member("u2", "Bola")
    group.add_member("u3", "Chidi")
    return group


def test_ajo_full_cycle_conserves_money():
    group = _three_member_circle()
    # Round 1: everyone pays, first member takes the pot.
    for uid in ("u1", "u2", "u3"):
        group.record_contribution(uid, 1000)
    payout = group.next_payout()
    assert payout["recipient_id"] == "u1"
    assert payout["amount_cents"] == 3000
    assert payout["round_complete"] is True
    group.record_payout()
    assert group.next_payout()["recipient_id"] == "u2"
    assert group.next_payout()["round_no"] == 2  # one payout per round
    assert group.member_balance("u1") == 1000 - 3000  # ahead in rotation
    assert group.member_balance("u2") == 1000
    assert group.member_balance("u3") == 1000
    assert sum(group.balances().values()) == 0  # money conserved
    # Round 2 just started: everyone owes again.
    assert {r["user_id"] for r in group.due_reminders()} == {"u1", "u2", "u3"}
    group.record_contribution("u1", 1000)
    assert {r["user_id"] for r in group.due_reminders()} == {"u2", "u3"}
    # Finish the full cycle: rounds 2 and 3 fund payouts to u2, u3.
    for uid in ("u2", "u3"):
        group.record_contribution(uid, 1000)
    group.record_payout()  # u2 takes round 2 pot
    for uid in ("u1", "u2", "u3"):
        group.record_contribution(uid, 1000)
    group.record_payout()  # u3 takes round 3 pot — cycle complete
    assert group.next_payout()["recipient_id"] == "u1"  # turn wraps
    assert group.balances() == {"u1": 0, "u2": 0, "u3": 0}


def test_ajo_payout_refused_on_incomplete_round():
    group = _three_member_circle()
    group.record_contribution("u1", 1000)
    with pytest.raises(ValueError, match="round incomplete"):
        group.record_payout()
    with pytest.raises(ValueError):
        group.record_contribution("ghost", 1000)
    with pytest.raises(ValueError):
        group.record_contribution("u1", 0)


def test_ajo_persistence_round_trip(tmp_path):
    store = ajo_mod.AjoStore(tmp_path)
    group = _three_member_circle()
    group.record_contribution("u1", 1000)
    store.put(group)
    loaded = store.get(group.id)
    assert loaded is not None
    assert loaded.contributed_this_round("u1") == 1000
    assert loaded.next_payout()["recipient_id"] == "u1"
    assert [g.id for g in store.list("g1")] == [group.id]
    assert store.remove(group.id) is True
    assert store.get(group.id) is None


# ── miniapps2 ───────────────────────────────────────────────────────────
def test_carpool_seats():
    ride = miniapps2_mod.CarpoolRide(
        id="r1",
        group_key="g1",
        route_from="Ikeja",
        route_to="Lekki",
        depart_at="Sat 8:00",
        seats_total=2,
        driver_id="d1",
        driver_name="Dele",
    )
    assert ride.seats_left == 2
    ride.add_rider("u1", "Ada")
    ride.add_rider("u2", "Bola")
    assert ride.is_full
    with pytest.raises(ValueError, match="full"):
        ride.add_rider("u3", "Chidi")
    assert ride.remove_rider("u1") is True
    assert ride.seats_left == 1


def test_rota_swap_and_unassigned():
    rota = miniapps2_mod.Rota(id="rt1", group_key="g1", title="Cleaning")
    rota.add_shift("2026-10-19", "morning")
    rota.add_shift("2026-10-19", "evening")
    rota.assign("2026-10-19", "morning", "u1", "Ada")
    assert len(rota.unassigned()) == 1
    rota.swap(("2026-10-19", "morning"), ("2026-10-19", "evening"))
    assert [s.slot for s in rota.shifts_for("u1")] == ["evening"]
    assert rota._find("2026-10-19", "morning").assignee_id == ""
    assert len(rota.unassigned()) == 1
    with pytest.raises(ValueError):
        rota.add_shift("2026-10-19", "morning")  # duplicate


def test_marketplace_offer_flow():
    listing = miniapps2_mod.Listing(
        id="l1",
        group_key="g1",
        seller_id="s1",
        seller_name="Sade",
        title="Bike",
        kind="sell",
        price_cents=50000,
    )
    cash = listing.make_offer("b1", "Bola", amount_cents=45000)
    barter = listing.make_offer("b2", "Ada", barter_text="2 bags of rice")
    assert barter.amount_cents is None
    with pytest.raises(ValueError):
        listing.make_offer("s1", "Sade", amount_cents=1)  # own listing
    chosen = listing.accept_offer(cash.id)
    assert chosen.status == "accepted"
    assert listing.status == "closed"
    assert listing.offers[1].status == "declined"  # losers auto-decline
    with pytest.raises(ValueError):
        listing.make_offer("b3", "Chidi", amount_cents=1)  # closed


def test_league_standings_order():
    league = miniapps2_mod.League(id="lg1", group_key="g1", name="Street League")
    a = league.add_team("Alpha")
    b = league.add_team("Beta")
    c = league.add_team("Gamma")
    league.record_result(a.id, 3, b.id, 1)  # Alpha 3 pts
    league.record_result(b.id, 2, c.id, 2)  # Beta 1, Gamma 1
    league.record_result(a.id, 0, c.id, 0)  # Alpha 4, Gamma 2
    assert [t.name for t in league.standings()] == ["Alpha", "Gamma", "Beta"]
    assert league.standings()[0].points == 4
    with pytest.raises(ValueError):
        league.record_result(a.id, 1, a.id, 1)  # can't play self


def test_split_sums_exactly_and_settlements_zero_balances():
    # 100 cents across 3 equal shares: 34 + 33 + 33 = 100 exactly.
    split = miniapps2_mod.split_by_shares(100, {"a": 1, "b": 1, "c": 1})
    assert sum(split.values()) == 100
    assert sorted(split.values()) == [33, 33, 34]
    # Weighted split also exact.
    split2 = miniapps2_mod.split_by_shares(100, {"a": 2, "b": 1})
    assert sum(split2.values()) == 100
    assert split2 == {"a": 67, "b": 33}
    with pytest.raises(ValueError):
        miniapps2_mod.split_by_shares(100, {"a": 0})
    # Settlements: A paid 900 for 3 → B and C each owe A 300.
    bill = miniapps2_mod.SplitBill(id="sb1", group_key="g1", title="Dinner")
    bill.add_receipt("a", "Ada", 900, {"a": 1, "b": 1, "c": 1})
    balances = bill.balances()
    assert balances == {"a": 600, "b": -300, "c": -300}
    assert sum(balances.values()) == 0
    transfers = bill.who_owes_whom()
    assert sum(t["amount_cents"] for t in transfers) == 600
    # Applying the transfers zeroes everyone out.
    applied = dict(balances)
    for t in transfers:
        applied[t["from_id"]] += t["amount_cents"]
        applied[t["to_id"]] -= t["amount_cents"]
    assert all(v == 0 for v in applied.values())


# ── audio ───────────────────────────────────────────────────────────────
def test_audio_transcript_and_speaker_stats(tmp_path):
    store = audio_mod.AudioStore(tmp_path)
    rec = audio_mod.Recording(
        id="rec1",
        group_key="g1",
        room_id="room1",
        title="Friday chat",
        started_ts=1000.0,
    )
    rec.finish(1600.0)
    assert rec.duration_s == 600.0
    store.put_recording(rec)
    clip = store.add_clip(rec, "highlight", 10.0, 70.0, "u1", "Ada")
    assert clip.duration_s == 60.0
    with pytest.raises(ValueError):
        store.add_clip(rec, "too long", 0.0, 9999.0)  # past recording end
    # No transcript yet → no stats, never invented.
    assert store.speaker_stats_for("g1", "rec1") == {}
    assert store.transcript_text("g1", "rec1") == ""
    # Transcript is ACCEPTED text, never generated.
    store.attach_transcript(
        "g1",
        "rec1",
        [
            audio_mod.TranscriptSegment("u1", "Ada", 0.0, 60.0, "hello all"),
            audio_mod.TranscriptSegment("u2", "Bola", 60.0, 90.0, "hi"),
            audio_mod.TranscriptSegment("u1", "Ada", 90.0, 150.0, "let us start"),
        ],
        source="manual",
    )
    stats = store.speaker_stats_for("g1", "rec1")
    assert stats["u1"]["talk_seconds"] == 120.0
    assert stats["u2"]["talk_seconds"] == 30.0
    assert stats["u1"]["segments"] == 2
    assert stats["u1"]["words"] == 5
    assert stats["u1"]["share"] == pytest.approx(0.8)
    assert "Ada: hello all" in store.transcript_text("g1", "rec1")


# ── bridges ─────────────────────────────────────────────────────────────
def _linked_roster():
    roster = bridges_mod.UnifiedRoster()
    ada = roster.add_member("Ada")
    roster.link(ada.id, "telegram", "tg-ada")
    roster.link(ada.id, "whatsapp", "wa-ada")
    roster.link(ada.id, "discord", "dc-ada")
    return roster, ada


def test_bridges_route_to_other_platforms_only():
    roster, ada = _linked_roster()
    targets = {
        "telegram": "tg-group",
        "whatsapp": "wa-group",
        "discord": "dc-channel",
    }
    deliveries = bridges_mod.plan_bridge(
        roster, "telegram", "tg-ada", "hello all", targets
    )
    assert {d.to_platform for d in deliveries} == {"whatsapp", "discord"}
    assert all(d.to_target for d in deliveries)
    assert all("Ada @ telegram" in d.text for d in deliveries)
    assert all(d.sent is False for d in deliveries)  # plan never sends
    # No targets → no deliveries, nothing guessed.
    assert bridges_mod.plan_bridge(roster, "telegram", "tg-ada", "hi", {}) == []
    # Unknown origin user still bridges, attributed by platform id.
    stranger = bridges_mod.plan_bridge(
        roster, "discord", "dc-stranger", "hey", targets
    )
    assert {d.to_platform for d in stranger} == {"telegram", "whatsapp"}
    assert all("dc-stranger @ discord" in d.text for d in stranger)


def test_bridges_never_send_without_injected_sender():
    roster, _ = _linked_roster()
    targets = {"whatsapp": "wa-group", "discord": "dc-channel"}
    sent: list[tuple[str, str]] = []
    # No senders at all → nothing sent.
    deliveries = bridges_mod.bridge_message(
        roster, "telegram", "tg-ada", "hello", targets
    )
    assert all(d.sent is False for d in deliveries)
    assert sent == []
    # Only WhatsApp sender injected → only WhatsApp is sent.
    deliveries = bridges_mod.bridge_message(
        roster,
        "telegram",
        "tg-ada",
        "hello",
        targets,
        senders={"whatsapp": lambda target, text: sent.append((target, text))},
    )
    by_platform = {d.to_platform: d for d in deliveries}
    assert by_platform["whatsapp"].sent is True
    assert by_platform["discord"].sent is False  # no sender → not sent
    assert sent == [("wa-group", "[Ada @ telegram] hello")]
    # Sender failure is captured per-delivery, not fatal.
    def _boom(target: str, text: str) -> None:
        raise RuntimeError("down")

    deliveries = bridges_mod.bridge_message(
        roster, "telegram", "tg-ada", "hi", targets, senders={"whatsapp": _boom}
    )
    assert by_platform["whatsapp"].sent is True  # previous call unaffected
    failed = [d for d in deliveries if d.to_platform == "whatsapp"][0]
    assert failed.sent is False and failed.error == "down"


def test_bridges_roster_linking_and_persistence(tmp_path):
    roster = bridges_mod.UnifiedRoster()
    ada = roster.add_member("Ada")
    bola = roster.add_member("Bola")
    roster.link(ada.id, "telegram", "tg-ada")
    roster.link(bola.id, "telegram", "tg-bola")
    # Relinking the same platform id moves it, never duplicates.
    roster.link(bola.id, "telegram", "tg-ada")
    assert roster.resolve("telegram", "tg-ada").id == bola.id
    assert roster.member_platforms(ada.id) == {}
    with pytest.raises(ValueError):
        roster.link(ada.id, "signal", "sig-ada")  # unknown platform
    store = bridges_mod.BridgeStore(tmp_path)
    store.save("g1", roster)
    assert store.load("g1").resolve("telegram", "tg-ada").id == bola.id


# ── insights ────────────────────────────────────────────────────────────
def test_insights_engagement_quiet_turnout():
    now = 1_000_000.0
    members = [
        {"member_id": "u1", "display_name": "Ada"},
        {"member_id": "u2", "display_name": "Bola"},
        {"member_id": "u3", "display_name": "Chidi"},
    ]
    messages = [
        {"member_id": "u1", "ts": now - 100},
        {"member_id": "u1", "ts": now - 200},
        {"member_id": "u2", "ts": now - 30 * 86400},  # 30 days ago
    ]
    eng = insights_mod.engagement(messages)
    assert eng["total"] == 3
    assert eng["active_members"] == 2
    assert eng["per_member"][0]["member_id"] == "u1"
    assert eng["per_member"][0]["share"] == pytest.approx(2 / 3)
    quiet = insights_mod.quiet_members(members, messages, days=14, now=now)
    quiet_ids = [q["member_id"] for q in quiet]
    assert "u2" in quiet_ids  # last message 30 days ago
    assert "u3" in quiet_ids  # never posted
    assert "u1" not in quiet_ids
    never = [q for q in quiet if q["member_id"] == "u3"][0]
    assert never["last_message_ts"] is None and never["days_since_last"] is None
    events = [
        {
            "event_id": "e1",
            "title": "Meetup",
            "rsvp": ["u1", "u2", "u3"],
            "attended": ["u1", "u4"],  # u4 walked in
        },
        {"event_id": "e2", "title": "No RSVP", "rsvp": [], "attended": ["u1"]},
    ]
    result = insights_mod.turnout(events)
    first = result["events"][0]
    assert first["rsvp"] == 3 and first["attended"] == 2
    assert first["no_shows"] == ["u2", "u3"]
    assert first["walk_ins"] == ["u4"]
    assert first["rate"] == pytest.approx(2 / 3)
    assert result["events"][1]["rate"] is None  # no RSVPs → no rate, not 0
    assert result["overall_rate"] == pytest.approx(3 / 3)
    snap = insights_mod.summary(members, messages, events, now=now)
    assert snap["engagement"]["total"] == 3
    assert len(snap["quiet_members"]) == 2
    assert snap["turnout"]["total_rsvp"] == 3

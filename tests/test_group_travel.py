"""Offline tests for the group-travel stack (#74).

Covers: polls + voting + deadlines, expense splitting, minimized
settlement, per-person budgets, per-traveler legs + merged view,
propose-vs-agree duality, group isolation, and the /gtrip chat surface.
"""

import ast
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.travel.groups import (
    GroupTripStore,
    control_gtrip,
    merged_itinerary,
    parse_naira_kobo,
    settle_balances,
)
from nomorals.travel.itinerary import Flight


@pytest.fixture()
def store(tmp_path):
    return GroupTripStore("test:group1", db_path=str(tmp_path / "g.db"))


@pytest.fixture()
def store2(tmp_path):
    return GroupTripStore("test:group2", db_path=str(tmp_path / "g2.db"))


@pytest.fixture()
def trip(store):
    return store.create_trip("Lagos Trip", ["Ada", "Mama", "Emeka"],
                             created_by="Ada")


# ── trips ──────────────────────────────────────────────────────────────────


def test_create_trip(store):
    t = store.create_trip("Beach", ["Ada", "Ada", "Mama"], created_by="Ada")
    assert t.id.startswith("gtrip_")
    assert t.members == ["Ada", "Mama"]  # deduped, creator included
    assert store.get_trip(t.id).name == "Beach"


def test_add_member(store, trip):
    store.add_member(trip.id, "Zainab")
    assert "Zainab" in store.get_trip(trip.id).members
    store.add_member(trip.id, "Zainab")  # idempotent
    assert store.get_trip(trip.id).members.count("Zainab") == 1


def test_unknown_trip(store):
    assert store.get_trip("nope") is None
    assert store.create_poll("nope", "q", ["a", "b"]) is None


# ── polls ──────────────────────────────────────────────────────────────────


def test_poll_vote_result(store, trip):
    p = store.create_poll(trip.id, "Where to eat?", ["suya", "pizza"],
                          created_by="Ada")
    assert p is not None
    assert store.vote(p.id, "Ada", "suya")
    assert store.vote(p.id, "Mama", 1)          # by index
    assert store.vote(p.id, "Emeka", "2")       # "2" = second option
    res = store.poll_result(p.id)
    assert res["counts"] == [1, 2]
    assert res["total"] == 3
    assert res["winner"] == 1
    assert not res["closed"]


def test_poll_one_vote_per_member(store, trip):
    p = store.create_poll(trip.id, "Q?", ["a", "b"])
    assert store.vote(p.id, "Ada", "a")
    assert store.vote(p.id, "Ada", "b")  # changed vote replaces
    res = store.poll_result(p.id)
    assert res["counts"] == [0, 1]


def test_poll_deadline(store, trip):
    import time
    p = store.create_poll(trip.id, "Q?", ["a", "b"], deadline=time.time() - 1)
    assert p.is_closed()
    assert not store.vote(p.id, "Ada", "a")  # closed → no vote
    p2 = store.create_poll(trip.id, "Q2?", ["a", "b"],
                           deadline=time.time() + 3600)
    assert not p2.is_closed()
    assert store.vote(p2.id, "Ada", "a")


def test_poll_needs_two_options(store, trip):
    assert store.create_poll(trip.id, "Q?", ["only"]) is None
    assert store.vote("nope", "Ada", "a") is False


# ── expenses + settlement ──────────────────────────────────────────────────


def test_expense_split_and_settlement(store, trip):
    # Ada pays 9000 for all three → each owes 3000
    store.add_expense(trip.id, "Ada", 900_000, None, "dinner")
    bal = store.balances(trip.id)
    assert bal["Ada"] == 600_000      # paid 9000, owes 3000
    assert bal["Mama"] == -300_000
    assert bal["Emeka"] == -300_000
    transfers = store.settlement(trip.id)
    assert len(transfers) == 2  # minimized: 2 debtors → 1 creditor
    total_moved = sum(a for _, _, a in transfers)
    assert total_moved == 600_000
    # every transfer goes to Ada
    assert all(c == "Ada" for _, c, _ in transfers)


def test_settlement_minimized_pure():
    # A owes 100, B owes 100, C is owed 200 → 2 transfers, not 3+
    t = settle_balances({"a": -100, "b": -100, "c": 200})
    assert len(t) == 2
    assert sum(a for _, _, a in t) == 200


def test_settlement_chain_minimized():
    # a owes 50, b is owed 30, c is owed 20 → 2 transfers
    t = settle_balances({"a": -50, "b": 30, "c": 20})
    assert len(t) == 2


def test_settlement_nothing_owed():
    assert settle_balances({}) == []
    assert settle_balances({"a": 0, "b": 0}) == []


def test_expense_subset(store, trip):
    # Mama pays 6000 for Ada + Mama only
    store.add_expense(trip.id, "Mama", 600_000, ["Ada", "Mama"], "taxi")
    bal = store.balances(trip.id)
    assert bal["Mama"] == 300_000
    assert bal["Ada"] == -300_000
    assert bal.get("Emeka", 0) == 0


def test_expense_validation(store, trip):
    assert store.add_expense("nope", "Ada", 100, None, "x") is None
    assert store.add_expense(trip.id, "Ada", 0, None, "x") is None
    assert store.add_expense(trip.id, "Ada", -5, None, "x") is None


# ── budgets ────────────────────────────────────────────────────────────────


def test_budget_and_spending(store, trip):
    assert store.set_budget(trip.id, "Ada", 1_000_000)
    store.add_expense(trip.id, "Mama", 900_000, None, "dinner")
    assert store.spending(trip.id, "Ada") == 300_000
    assert store.get_budget(trip.id, "Ada") == 1_000_000
    assert store.get_budget(trip.id, "Nobody") == 0
    assert not store.set_budget("nope", "Ada", 100)


# ── legs + merged itinerary ────────────────────────────────────────────────


def test_legs_and_merged_view(store, trip):
    f1 = Flight(flight_number="BA075", origin="LOS", destination="LHR",
                departs="2026-12-01T22:45", pnr="ABC123")
    f2 = Flight(flight_number="BA076", origin="LHR", destination="LOS",
                departs="2026-12-05T10:00")
    store.add_leg(trip.id, "Ada", f1)
    store.add_leg(trip.id, "Mama", f2, note="window seat")
    legs = store.list_legs(trip.id)
    assert len(legs) == 2
    view = store.merged_itinerary(trip.id)
    assert "Ada" in view and "BA075" in view
    assert "Mama" in view and "BA076" in view
    assert "ABC123" in view


def test_merged_itinerary_empty(store, trip):
    view = store.merged_itinerary(trip.id)
    assert "nothing planned yet" in view
    assert store.merged_itinerary("nope") == "no such group trip."


def test_merged_itinerary_pure():
    from nomorals.travel.groups import GroupTrip
    t = GroupTrip(id="x", name="Test")
    view = merged_itinerary(t, [], [])
    assert "Test" in view


# ── propose vs agree ───────────────────────────────────────────────────────


def test_propose_agree_duality(store, trip):
    p = store.propose(trip.id, "Ada", "Visit Lekki market")
    assert p.status == "proposed"
    # proposals and agreed are different objects
    assert store.list_proposals(trip.id, "proposed") != []
    assert store.list_proposals(trip.id, "agreed") == []
    agreed = store.agree(trip.id, p.id)
    assert agreed.status == "agreed"
    assert store.list_proposals(trip.id, "proposed") == []
    assert len(store.list_proposals(trip.id, "agreed")) == 1
    # agreed items show in the itinerary
    assert "Lekki market" in store.merged_itinerary(trip.id)


def test_propose_validation(store, trip):
    assert store.propose("nope", "Ada", "x") is None
    assert store.propose(trip.id, "Ada", "   ") is None
    assert store.agree(trip.id, "nope") is None


# ── parsing ────────────────────────────────────────────────────────────────


def test_parse_naira_kobo():
    assert parse_naira_kobo("5000") == 500_000
    assert parse_naira_kobo("5k") == 500_000
    assert parse_naira_kobo("₦5,000") == 500_000
    assert parse_naira_kobo("2.5k") == 250_000
    assert parse_naira_kobo("garbage") is None
    assert parse_naira_kobo("") is None


# ── isolation ──────────────────────────────────────────────────────────────


def test_group_isolation(store, store2):
    t1 = store.create_trip("Trip A", ["Ada"])
    t2 = store2.create_trip("Trip B", ["Mama"])
    assert [t.name for t in store.list_trips()] == ["Trip A"]
    assert [t.name for t in store2.list_trips()] == ["Trip B"]
    assert store.get_trip(t2.id) is None  # not visible across groups


def test_no_owner_scoped_imports():
    """groups.py must not import memory/accounts/vaults/connectors."""
    path = Path(__file__).resolve().parent.parent / "nomorals" / "travel" / "groups.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.append(node.module)
    for mod in mods:
        assert not mod.startswith(
            ("nomorals.memory", "nomorals.accounts", "nomorals.connectors")), mod
        assert "vault" not in mod, mod


# ── chat surface ───────────────────────────────────────────────────────────


def _group_chat():
    return SimpleNamespace(kind="group", platform="test", id="g1")


def _dm_chat():
    return SimpleNamespace(kind="dm", platform="test", id="d1")


def test_chat_refuses_dm():
    out = control_gtrip("new Trip", chat=_dm_chat(), sender="Ada")
    assert "group chats" in out


def test_chat_full_flow(tmp_path):
    import nomorals.travel.groups as g
    g._DEFAULT_DIR = tmp_path  # noqa: SLF001 — isolate test state
    chat = _group_chat()
    out = control_gtrip("new Lagos Trip", chat=chat, sender="Ada")
    assert "Group trip created" in out
    out = control_gtrip("poll where to eat? | suya | pizza | deadline 2h",
                        chat=chat, sender="Ada")
    assert "Vote:" in out
    poll_id = out.split("/gtrip vote ")[1].split(" ")[0]
    out = control_gtrip(f"vote {poll_id} suya", chat=chat, sender="Mama")
    assert "counted" in out
    out = control_gtrip(f"result {poll_id}", chat=chat, sender="Ada")
    assert "suya" in out
    out = control_gtrip("expense 5k dinner", chat=chat, sender="Ada")
    assert "₦5,000" in out
    out = control_gtrip("settle", chat=chat, sender="Ada")
    assert "Settle up" in out or "settled" in out
    out = control_gtrip("propose visit the beach", chat=chat, sender="Mama")
    assert "Proposed" in out
    out = control_gtrip("itinerary", chat=chat, sender="Ada")
    assert "Lagos Trip" in out


def test_chat_never_raises():
    # garbage input on every subcommand → error strings, never exceptions
    chat = _group_chat()
    for tail in ["", "poll", "vote", "expense xyz", "settle", "budget",
                 "spending", "leg", "propose", "agree nope", "itinerary",
                 "frobnicate"]:
        out = control_gtrip(tail, chat=chat, sender="Ada")
        assert isinstance(out, str) and out


def test_chat_usage():
    out = control_gtrip("help", chat=_group_chat(), sender="Ada")
    assert "/gtrip new" in out and "/gtrip settle" in out

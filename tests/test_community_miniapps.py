"""Tests for group mini-apps: poll / expenses / rsvp logic, store, chat wiring."""

import json
from pathlib import Path

import pytest

from nomorals.community import miniapps
from nomorals.community.miniapps import (
    MiniAppStore,
    apply_action,
    create_miniapp,
    render_miniapp,
    simplify_debts,
    web_url,
)

GROUP = "telegram:123"


@pytest.fixture
def store(tmp_path: Path) -> MiniAppStore:
    return MiniAppStore(data_dir=tmp_path / "miniapps")


# ── creation ───────────────────────────────────────────────────────────────


def test_create_poll_needs_two_options():
    with pytest.raises(ValueError):
        create_miniapp("poll", GROUP, "Q?", options=["only"])
    app = create_miniapp("poll", GROUP, "Q?", options=["a", "b"])
    assert app.kind == "poll" and app.id.startswith("poll_")


def test_create_unknown_kind():
    with pytest.raises(ValueError):
        create_miniapp("leaderboard", GROUP, "Nope")


def test_web_url_is_honestly_none():
    app = create_miniapp("poll", GROUP, "Q?", options=["a", "b"])
    assert web_url(app) is None  # chat-native is the surface; no fake hosting


# ── poll ───────────────────────────────────────────────────────────────────


def test_poll_vote_and_change():
    app = create_miniapp("poll", GROUP, "Best?", options=["x", "y"])
    app, r1 = apply_action(app, "u1", "Ada", "vote", ["1"])
    assert "voted" in r1
    app, r2 = apply_action(app, "u1", "Ada", "vote", ["2"])
    assert "changed" in r2
    assert app.state["votes"] == {"u1": 1}  # one vote per user


def test_poll_results_math():
    app = create_miniapp("poll", GROUP, "Best?", options=["x", "y"])
    for uid, name, opt in [("u1", "A", "1"), ("u2", "B", "2"), ("u3", "C", "2")]:
        app, _ = apply_action(app, uid, name, "vote", [opt])
    text = render_miniapp(app)
    assert "1 vote(s) (33%)" in text
    assert "2 vote(s) (67%)" in text


def test_poll_close_blocks_votes():
    app = create_miniapp("poll", GROUP, "Best?", options=["x", "y"])
    app, _ = apply_action(app, "u1", "Ada", "close", [])
    assert app.state["closed"] is True
    app, reply = apply_action(app, "u2", "Bola", "vote", ["1"])
    assert "closed" in reply
    assert "u2" not in app.state["votes"]


def test_poll_bad_option_never_raises():
    app = create_miniapp("poll", GROUP, "Best?", options=["x", "y"])
    for bad in [[], ["0"], ["3"], ["abc"]]:
        app, reply = apply_action(app, "u1", "Ada", "vote", bad)
        assert "u1" not in app.state["votes"]
        assert isinstance(reply, str) and reply


# ── expenses ───────────────────────────────────────────────────────────────


def test_expenses_balances_and_settlement():
    app = create_miniapp("expenses", GROUP, "Trip")
    app, _ = apply_action(app, "a", "Ada", "expense", ["90", "dinner", "for Ada,Bola,Chidi"])
    bal = miniapps._balances(app.state)
    # Ada paid 9000c, split 3 ways → Ada +6000, others -3000
    assert bal == {"a": 6000, "name:bola": -3000, "name:chidi": -3000}
    transfers = simplify_debts(bal)
    assert sorted(transfers) == sorted([("name:bola", "a", 3000), ("name:chidi", "a", 3000)])
    text = render_miniapp(app)
    assert "90.00" in text and "Bola → Ada: 30.00" in text


def test_expenses_settle_clears_debt():
    app = create_miniapp("expenses", GROUP, "Trip")
    app, _ = apply_action(app, "a", "Ada", "expense", ["60", "taxi", "for Ada,Bola"])
    app, _ = apply_action(app, "x", "Sys", "settle", ["name:bola", "a", "30"])
    assert miniapps._balances(app.state) == {}


def test_expenses_bad_amount_never_raises():
    app = create_miniapp("expenses", GROUP, "Trip")
    app, reply = apply_action(app, "a", "Ada", "expense", ["abc", "dinner"])
    assert not app.state["expenses"]
    assert "Couldn't parse" in reply


def test_settlement_minimal_transfers():
    # A is owed 50, B owes 30, C owes 20 → two transfers, not three
    transfers = simplify_debts({"a": 5000, "b": -3000, "c": -2000})
    assert len(transfers) == 2
    assert sum(t[2] for t in transfers) == 5000


# ── rsvp ───────────────────────────────────────────────────────────────────


def test_rsvp_counts_and_nudge():
    app = create_miniapp("rsvp", GROUP, "Game night", date="2026-10-20")
    app, _ = apply_action(app, "u1", "Ada", "yes", [])
    app, _ = apply_action(app, "u2", "Bola", "maybe", [])
    app, _ = apply_action(app, "u3", "Chidi", "nudge", [])  # seen but silent
    app, reply = apply_action(app, "u1", "Ada", "nudge", [])
    assert "Chidi" in reply and "Ada" not in reply and "Bola" not in reply
    text = render_miniapp(app)
    assert "Yes (1)" in text and "Maybe (1)" in text


def test_rsvp_change_answer():
    app = create_miniapp("rsvp", GROUP, "E")
    app, _ = apply_action(app, "u1", "Ada", "yes", [])
    app, _ = apply_action(app, "u1", "Ada", "no", [])
    assert app.state["responses"] == {"u1": "no"}


# ── store ──────────────────────────────────────────────────────────────────


def test_store_round_trip(store: MiniAppStore):
    app = create_miniapp("poll", GROUP, "Q?", options=["a", "b"])
    store.put(app)
    loaded = store.load(GROUP)
    assert len(loaded) == 1 and loaded[0].id == app.id
    assert loaded[0].state["options"] == ["a", "b"]


def test_store_group_isolation(store: MiniAppStore):
    store.put(create_miniapp("poll", GROUP, "Q?", options=["a", "b"]))
    store.put(create_miniapp("rsvp", "discord:999", "E"))
    assert len(store.load(GROUP)) == 1
    assert len(store.load("discord:999")) == 1


def test_store_malformed_file_returns_empty(tmp_path: Path):
    d = tmp_path / "miniapps"
    d.mkdir()
    (d / "telegram_123.json").write_text("{not json", encoding="utf-8")
    assert MiniAppStore(data_dir=d).load(GROUP) == []


def test_store_prefix_id_lookup(store: MiniAppStore):
    app = create_miniapp("poll", GROUP, "Q?", options=["a", "b"])
    store.put(app)
    assert store.get(GROUP, app.id[:6]).id == app.id


# ── chat wiring ────────────────────────────────────────────────────────────


class _Chat:
    def __init__(self, kind="group", platform="telegram", chat_id="123", thread_id=""):
        self.kind = kind
        self.platform = platform
        self.chat_id = chat_id
        self.thread_id = thread_id


class _Settings:
    def __init__(self, community_dir):
        self.community_dir = community_dir


class _Ctx:
    def __init__(self, community_dir):
        self.settings = _Settings(community_dir)


def test_control_dm_refused(tmp_path: Path):
    ctx = _Ctx(str(tmp_path))
    reply = miniapps.control_miniapp("list", context=ctx,
                                     chat=_Chat(kind="dm"),
                                     sender_id="u1", sender="Ada")
    assert "group chats" in reply


def test_control_new_list_show(tmp_path: Path):
    ctx = _Ctx(str(tmp_path))
    chat = _Chat()
    r = miniapps.control_miniapp('new poll "Best?" x y', context=ctx, chat=chat,
                                 sender_id="u1", sender="Ada")
    assert "Created" in r
    r = miniapps.control_miniapp("list", context=ctx, chat=chat,
                                 sender_id="u1", sender="Ada")
    assert "Best?" in r
    app_id = miniapps.MiniAppStore(data_dir=tmp_path).load("telegram:123")[0].id
    r = miniapps.control_miniapp(f"vote {app_id} 2", context=ctx, chat=chat,
                                 sender_id="u1", sender="Ada")
    assert "voted" in r


def test_control_thread_isolation(tmp_path: Path):
    ctx = _Ctx(str(tmp_path))
    miniapps.control_miniapp('new poll "Q?" a b', context=ctx, chat=_Chat(thread_id="t1"),
                             sender_id="u1", sender="Ada")
    r = miniapps.control_miniapp("list", context=ctx, chat=_Chat(thread_id="t2"),
                                 sender_id="u1", sender="Ada")
    assert "No mini-apps" in r


def test_control_never_raises_on_garbage(tmp_path: Path):
    ctx = _Ctx(str(tmp_path))
    chat = _Chat()
    for tail in ["", "bogus", "vote", "vote nope 1", "new bogus x"]:
        assert isinstance(miniapps.control_miniapp(tail, context=ctx, chat=chat,
                                                   sender_id="u1", sender="Ada"), str)

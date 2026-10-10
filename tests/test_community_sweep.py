"""Sweep tests for the community module upgrade (2026-10-10).

Covers everything added in the mined-then-built upgrade: roles,
announcements, recurrence engine, capacity/waitlists, approval voting,
quiz mini-apps, Splitwise-grade expenses, audio-room co-hosts/scheduling,
and the extended fun registry. All stores are tmp-dir scoped.
"""

import time
from pathlib import Path

import pytest

from nomorals.community import audio_rooms as ar
from nomorals.community import events as evmod
from nomorals.community import groups as gmod
from nomorals.community import meetups as mmod
from nomorals.community import miniapps
from nomorals.community.miniapps import (
    MiniAppStore, apply_action, create_miniapp, render_miniapp,
)
from nomorals.community.registry import (
    COMMUNITY_TOOL_NAMES, coin, community_registry, pick, roll, shuffle,
)


def _future(hours=48):
    return time.time() + hours * 3600.0


# ── groups: roles, announcements, pins, search, gates ─────────────────────


def test_group_roles_and_gates(tmp_path: Path):
    s = gmod.GroupStore(data_dir=tmp_path / "g")
    g = s.create_group("Devs", "code", created_by="Ada")
    assert g.role_of("Ada") == "owner"
    s.join_group(g.id, "Bob")
    # member can't announce
    assert s.announce(g.id, "Bob", "hi") is None
    # owner can
    assert s.announce(g.id, "Ada", "rules!") is not None
    # promote
    ok, msg = s.set_role(g.id, "Ada", "Bob", "admin")
    assert ok and "admin" in msg
    # admin can now announce, non-owner can't change roles
    assert s.announce(g.id, "Bob", "update") is not None
    ok2, _ = s.set_role(g.id, "Bob", "Ada", "member")
    assert not ok2  # only owner
    # founder keeps owner
    ok3, _ = s.set_role(g.id, "Ada", "Ada", "member")
    assert not ok3


def test_group_pins_search_welcome_gate(tmp_path: Path):
    s = gmod.GroupStore(data_dir=tmp_path / "g")
    g = s.create_group("Hikers", "trails and tea", created_by="Ada")
    assert s.set_welcome(g.id, "Ada", "Say hi on arrival!")
    assert s.set_join_question(g.id, "Ada", "What's your pace?")
    msg = s.join_group(g.id, "Cara")
    assert "pace" in msg  # gate question surfaced
    assert s.answer_join_question(g.id, "Cara", "slow and steady")
    assert s.get(g.id).join_answers["Cara"] == "slow and steady"
    # non-owner can't set welcome
    assert not s.set_welcome(g.id, "Cara", "x")
    p = s.post(g.id, "Ada", "trail saturday")
    assert p is not None
    assert s.pin(g.id, "Ada", p.id)
    assert [x.id for x in s.pinned_posts(g.id)] == [p.id]
    assert not s.pin(g.id, "Cara", p.id)  # member can't pin
    assert s.unpin(g.id, "Ada", p.id)
    assert s.pinned_posts(g.id) == []
    # search
    assert s.search("hiker")[0].id == g.id
    assert s.search("TRAILS")[0].id == g.id
    assert s.search("zzz") == []


def test_group_reputation_and_card(tmp_path: Path):
    gs = gmod.GroupStore(data_dir=tmp_path / "g")
    es = evmod.EventStore(data_dir=tmp_path / "e")
    ms = mmod.MeetupStore(data_dir=tmp_path / "m", events=es)
    g = gs.create_group("Runners", "run", created_by="Ada")
    past = es.create_event(g.id, "Old run", time.time() - 7200.0)
    es.rsvp(past.id, "Ada", "yes")
    es.rsvp(past.id, "Bob", "yes")
    ms.check_in(past.id, "Ada")  # Ada showed, Bob no-showed
    rep = gs.member_reputation("Ada", events_store=es, meetups_store=ms)
    assert rep["score"] == 1 and rep["shows"] == 1
    rep_b = gs.member_reputation("Bob", events_store=es, meetups_store=ms)
    assert rep_b["score"] == -1 and rep_b["no_shows"] == 1
    # group card renders events + header
    fut = es.create_event(g.id, "Next run", _future(), where="Park")
    card = gmod.format_group_card(gs, gs.get(g.id), events_store=es)
    assert "Runners" in card and "Next run" in card and "👑" in card
    assert isinstance(gmod.icebreaker(seed=1), str)
    # member activity
    gs.post(g.id, "Ada", "hello")
    assert gs.member_activity(g.id)[0] == ("Ada", 1)


# ── events: recurrence engine ──────────────────────────────────────────────


def test_recurrence_parse_and_describe():
    assert evmod.parse_recurrence("daily")["freq"] == "daily"
    assert evmod.parse_recurrence("daily:3")["interval"] == 3
    w = evmod.parse_recurrence("weekly:MO,WE,FR")
    assert w["byweekday"] == [0, 2, 4]
    m = evmod.parse_recurrence("monthly:15")
    assert m["bymonthday"] == 15
    assert evmod.parse_recurrence("monthly:-1")["bymonthday"] == -1
    assert evmod.parse_recurrence("daily:2;count=5")["count"] == 5
    u = evmod.parse_recurrence("weekly;until=2026-12-31 23:59")
    assert u["until"] > 0
    assert evmod.parse_recurrence("yearly") is None
    assert evmod.parse_recurrence("") is None
    assert "Mon" in evmod.describe_recurrence("weekly:MO,WE")
    assert "day" in evmod.describe_recurrence("daily:2")


def test_daily_and_monthly_occurrences():
    now = time.time()
    spec = evmod.parse_recurrence("daily:2")
    occs = evmod.occurrences_after(now, spec, after=now, n=3)
    assert len(occs) == 3
    assert all(occs[i + 1] - occs[i] == pytest.approx(2 * 86400.0, abs=2)
               for i in range(2))
    spec_m = evmod.parse_recurrence("monthly")
    occs_m = evmod.occurrences_after(now, spec_m, after=now, n=2)
    assert len(occs_m) == 2
    # exdates skip
    ex = [occs[0]]
    occs2 = evmod.occurrences_after(now, spec, exdates=ex, after=now, n=3)
    assert all(abs(o - occs[0]) > 3600 for o in occs2)
    # terminators bound the series from its origin
    spec_c = evmod.parse_recurrence("daily:1;count=3")
    occs_c = evmod.occurrences_after(now + 86400, spec_c, after=now, n=10)
    assert len(occs_c) == 2  # DTSTART counts as occurrence #1
    spec_c1 = evmod.parse_recurrence("daily:1;count=1")
    assert evmod.occurrences_after(now + 86400, spec_c1, after=now, n=10) == []
    spec_u = evmod.parse_recurrence("daily;until=2020-01-01 00:00")
    assert evmod.occurrences_after(now, spec_u, after=now, n=3) == []


def test_event_store_recurring_engine(tmp_path: Path):
    s = evmod.EventStore(data_dir=tmp_path / "e")
    ev = s.create_event("g1", "Daily sync", _future(24), recurrence="daily:2",
                        description="standup", link="https://x", capacity=10)
    assert ev is not None and ev.description == "standup"
    occs = s.next_occurrences(ev.id, 3)
    assert len(occs) == 3
    assert all(o > ev.starts_at for o in occs)
    assert ev.is_recurring()
    # bad recurrence rejected loudly
    assert s.create_event("g1", "Bad", _future(), recurrence="yearly") is None
    # weekly multi-day
    ev2 = s.create_event("g1", "Gym", _future(24), recurrence="weekly:MO,WE")
    occs2 = s.next_occurrences(ev2.id, 4)
    assert len(occs2) == 4
    assert {time.localtime(o).tm_wday for o in occs2} <= {0, 2}
    # skip a date
    assert s.skip_occurrence(ev2.id, occs2[0])
    occs3 = s.next_occurrences(ev2.id, 4)
    assert all(abs(o - occs2[0]) > 3600 for o in occs3)


def test_capacity_waitlist_guests(tmp_path: Path):
    s = evmod.EventStore(data_dir=tmp_path / "e")
    ev = s.create_event("g1", "Dinner", _future(), capacity=3)
    assert s.rsvp(ev.id, "Ada", "yes") == "yes"
    # +1 guest uses seats too
    assert s.rsvp(ev.id, "Bob", "yes", guests=1) == "yes"
    assert s.get(ev.id).seats_used() == 3  # Ada + Bob + Bob's guest
    # full → waitlist
    assert s.rsvp(ev.id, "Cara", "yes") == "waitlist"
    assert s.waitlist(ev.id) == ["Cara"]
    counts = s.rsvp_counts(ev.id)
    assert counts["waitlist"] == 1 and counts["yes"] == 2
    # cancellation promotes the head
    promoted = s.cancel_rsvp(ev.id, "Ada")
    assert promoted == "Cara"
    assert s.get(ev.id).rsvps["Cara"] == "yes"
    assert s.waitlist(ev.id) == []


def test_rsvp_window_and_proxy(tmp_path: Path):
    s = evmod.EventStore(data_dir=tmp_path / "e")
    now = time.time()
    ev = s.create_event("g1", "Gala", _future(72),
                        rsvp_opens_at=now + 3600.0)
    assert s.rsvp(ev.id, "Ada", "yes", now=now) == ""  # window not open
    ev2 = s.create_event("g1", "Gala2", _future(72),
                         rsvp_closes_at=now - 10.0)
    assert s.rsvp(ev2.id, "Ada", "yes", now=now) == ""  # window closed
    # proxy RSVP attaches on self-RSVP
    ev3 = s.create_event("g1", "Gala3", _future(72))
    assert s.rsvp_for(ev3.id, "Zed", "yes", by="Ada")
    assert s.get(ev3.id).proxy["Zed"] == "Ada"
    assert s.rsvp(ev3.id, "Zed", "maybe") == "maybe"
    assert "Zed" not in s.get(ev3.id).proxy  # attached, not duplicated


def test_event_edit_my_events_reminders(tmp_path: Path):
    s = evmod.EventStore(data_dir=tmp_path / "e")
    now = time.time()
    ev = s.create_event("g1", "Big day", now + 3 * 86400.0, where="Hall",
                        description="d" * 10)
    s.edit_event(ev.id, title="Bigger day", capacity=50)
    got = s.get(ev.id)
    assert got.title == "Bigger day" and got.capacity == 50
    s.rsvp(ev.id, "Ada", "yes")
    mine = s.my_events("Ada")
    assert [e.id for e in mine] == [ev.id]
    # 3 days out → only the 7-day window fires
    due = s.check_reminders(now=now)
    assert len(due) == 1 and "in a week" in due[0]["text"]
    assert s.check_reminders(now=now) == []
    # card renders the new fields
    text = evmod.EventStore.format_event(s.get(ev.id))
    assert "Bigger day" in text and "50 seats" in text


# ── meetups: approval voting, blind polls, adoption, reliability ────────────


def _meetup_stores(tmp_path: Path):
    es = evmod.EventStore(data_dir=tmp_path / "e")
    ms = mmod.MeetupStore(data_dir=tmp_path / "m", events=es)
    return es, ms


def test_approval_voting_and_blind(tmp_path: Path):
    es, ms = _meetup_stores(tmp_path)
    ev = es.create_event("g1", "Dinner", _future())
    poll = ms.venue_poll(ev.id, ["A", "B", "C"], multi=True, blind=True,
                         closes_in_hours=2)
    assert poll.multi and poll.blind and poll.closes_at > 0
    assert ms.vote_approval(poll.id, "Ada", ["0", "1"])  # A and B
    assert ms.vote_approval(poll.id, "Bob", ["B"])       # just B
    assert not ms.vote_approval(poll.id, "Zed", ["9"])   # bad option
    assert ms.poll_results(poll.id) == {"A": 1, "B": 2, "C": 0}
    assert ms.get_poll(poll.id).winner() == "B"
    # blind: render hides counts until closed
    text = mmod.render_poll(ms.get_poll(poll.id))
    assert "🙈 blind" in text and "1 vote" not in text
    ms.close_poll(poll.id)
    text2 = mmod.render_poll(ms.get_poll(poll.id))
    assert "2 votes" in text2
    # rich venue options
    poll2 = ms.venue_poll(ev.id, [{"name": "X", "address": "12 Rd",
                                   "link": "https://x"} , "Y"])
    assert ms.vote_poll(poll2.id, "Ada", "0")
    assert "12 Rd" in mmod.render_poll(ms.get_poll(poll2.id))


def test_adopt_winner_and_due_polls(tmp_path: Path):
    es, ms = _meetup_stores(tmp_path)
    ev = es.create_event("g1", "Lunch", _future(), where="TBD")
    poll = ms.venue_poll(ev.id, ["Suya spot", "Mama put"], closes_in_hours=0.0001)
    ms.vote_poll(poll.id, "Ada", "0")
    ms.vote_poll(poll.id, "Bob", "0")
    import time as _t
    _t.sleep(0.5)
    assert [p.id for p in ms.due_polls()] == [poll.id]
    res = ms.adopt_winner(poll.id)
    assert res["ok"] and res["venue"] == "Suya spot"
    assert "Suya spot" in res["notify"]
    assert es.get(ev.id).where == "Suya spot"
    assert ms.get_poll(poll.id).closed
    assert ms.adopt_winner("nope")["ok"] is False


def test_reliability_and_quick_meetup(tmp_path: Path):
    es, ms = _meetup_stores(tmp_path)
    past = es.create_event("g1", "Old", time.time() - 7200.0)
    es.rsvp(past.id, "Ada", "yes")
    es.rsvp(past.id, "Bob", "yes")
    ms.check_in(past.id, "Ada")
    r = ms.reliability("Ada")
    assert r["score"] == 1
    r2 = ms.reliability("Bob")
    assert r2["score"] == -1 and "flaky" in r2["label"]
    t = ms.turnout(past.id)
    assert t["reliability"]["Bob"]["score"] == -1
    # host roll-call
    assert ms.host_check_in(past.id, ["Cara", "Dele"]) == 2
    # instant meetup
    res = ms.quick_meetup("g1", "Drinks", _future(5), ["Bar A", "Bar B"],
                          created_by="Ada")
    assert res["ok"]
    assert "Spontaneous meetup" in res["notify"]
    assert res["poll"] is not None


# ── miniapps: quiz, multi polls, split strategies, capacity ─────────────────


def test_quiz_lifecycle():
    app = create_miniapp("quiz", "g", "Capital?",
                         options=["Lagos", "Abuja", "Kano"], correct=1)
    assert app.state["correct"] == 1
    app, reply = apply_action(app, "u1", "Ada", "answer", ["2"])
    assert "Correct" in reply and "+2" in reply  # first-try bonus
    assert app.state["scores"]["u1"] == 2
    app, reply = apply_action(app, "u2", "Bola", "answer", ["1"])
    assert "Not quite" in reply
    assert app.state["streaks"]["u2"] == 0
    app, reply = apply_action(app, "u2", "Bola", "answer", ["2"])
    assert "Correct" in reply and "+1 pt" in reply  # later try = 1 pt
    assert "🔥" not in reply  # streak is only 1
    app, reply = apply_action(app, "u1", "Ada", "scoreboard", [])
    assert "🥇" in reply and "Ada" in reply
    app, reply = apply_action(app, "u1", "Ada", "close", [])
    assert "🔒 Quiz closed" in reply and "✅" in reply  # answer revealed
    app, reply = apply_action(app, "u2", "Bola", "answer", ["2"])
    assert "closed" in reply.lower()
    # quiz renders
    assert "🧠" in render_miniapp(app)


def test_multi_poll_ballots_and_comments():
    app = create_miniapp("poll", "g", "Lunch?", options=["A", "B", "C"],
                         multi=True)
    app, reply = apply_action(app, "u1", "Ada", "vote", ["1", "2"])
    assert "approves" in reply
    app, reply = apply_action(app, "u2", "Bo", "vote", ["2"])
    text = render_miniapp(app)
    assert "multi-answer" in text
    app, reply = apply_action(app, "u1", "Ada", "comment", ["tacos", "forever"])
    assert "Comment added" in reply
    assert "tacos forever" in render_miniapp(app)
    # deadline
    app, reply = apply_action(app, "u1", "Ada", "deadline", ["2"])
    assert "closes in 2h" in reply
    assert app.state["closes_at"] > time.time()
    # due_polls surfaces it once expired
    st = MiniAppStore(data_dir=__import__("tempfile").mkdtemp())
    app.state["closes_at"] = time.time() - 1
    st.put(app)
    assert [a.id for a in miniapps.due_polls(st, "g")] == [app.id]


def test_expense_split_strategies():
    # exact split, sum-validated
    app = create_miniapp("expenses", "g", "Trip")
    app, reply = apply_action(app, "a", "Ada", "expense",
                              ["90", "dinner", "for Ada,Bola", "split:exact:60,30"])
    assert "Recorded" in reply
    bal = miniapps._balances(app.state)
    assert bal == {"a": 3000, "name:bola": -3000}  # Ada paid 90, owed 60 back
    # exact that doesn't add up → loud rejection (the classic bug)
    app, reply = apply_action(app, "a", "Ada", "expense",
                              ["90", "dinner", "for Ada,Bola", "split:exact:50,30"])
    assert "add up to" in reply and len(app.state["expenses"]) == 1
    # percent split, must sum to 100
    app, reply = apply_action(app, "a", "Ada", "expense",
                              ["100", "taxi", "for Ada,Bola", "split:percent:70,30"])
    assert "Recorded" in reply
    shares = app.state["expenses"][-1]["shares"]
    assert shares["a"] == 7000 and shares["name:bola"] == 3000
    app, reply = apply_action(app, "a", "Ada", "expense",
                              ["100", "x", "for Ada,Bola", "split:percent:70,20"])
    assert "100" in reply and len(app.state["expenses"]) == 2
    # rounding drift fixed deterministically
    app2 = create_miniapp("expenses", "g", "T2")
    app2, _ = apply_action(app2, "a", "Ada", "expense",
                           ["10", "snack", "for Ada,Bola,Chidi",
                            "split:percent:33.33,33.33,33.34"])
    shares2 = app2.state["expenses"][-1]["shares"]
    assert sum(shares2.values()) == 1000
    assert "(percent)" in render_miniapp(app2)


def test_expense_edit_del_summary_currency():
    app = create_miniapp("expenses", "g", "Trip")
    app, _ = apply_action(app, "a", "Ada", "expense",
                          ["90", "dinner", "for Ada,Bola", "cat:food"])
    exp_id = app.state["expenses"][0]["id"]
    app, reply = apply_action(app, "a", "Ada", "edit", [exp_id, "120", "dinner party"])
    assert "120.00" in reply
    assert miniapps._balances(app.state)["a"] == 6000  # paid 120, split 2 ways
    app, reply = apply_action(app, "a", "Ada", "currency", ["₦"])
    assert app.state["currency"] == "₦"
    assert "₦120.00" in render_miniapp(app)
    app, reply = apply_action(app, "a", "Ada", "summary", [])
    assert "food" in reply and "By month" in reply and "Paid by" in reply
    app, reply = apply_action(app, "a", "Ada", "del", [exp_id])
    assert "deleted" in reply and not app.state["expenses"]
    assert miniapps._balances(app.state) == {}
    app, reply = apply_action(app, "a", "Ada", "del", ["nope"])
    assert "No expense" in reply


def test_rsvp_miniapp_capacity_waitlist():
    app = create_miniapp("rsvp", "g", "Game night", date="2026-10-20",
                         capacity=2)
    app, r1 = apply_action(app, "u1", "Ada", "yes", [])
    app, r2 = apply_action(app, "u2", "Bola", "yes", ["+1"])
    assert "+1" in r2
    assert app.state["guests"]["u2"] == 1
    app, r3 = apply_action(app, "u3", "Chidi", "yes", [])
    assert "waitlist" in r3 and app.state["waitlist"] == ["u3"]
    text = render_miniapp(app)
    assert "2/2 seats" in text and "Waitlist (1)" in text
    # someone leaves → waitlist head promoted
    app, _ = apply_action(app, "u1", "Ada", "no", [])
    assert app.state["responses"]["u3"] == "yes" and app.state["waitlist"] == []
    # capacity action
    app, r = apply_action(app, "u1", "Ada", "capacity", ["0"])
    assert "unlimited" in r


def test_panels_due_and_digest(tmp_path: Path):
    st = miniapps.PanelStore(data_dir=tmp_path / "p")
    p = miniapps.create_panel("Runs", "owner", "strava", store=st,
                              refresh_cadence="hourly")
    p.last_refresh = time.time() - 7200.0
    st.put(p)
    assert [x.id for x in miniapps.due_refreshes("owner", store=st)] == [p.id]
    miniapps.refresh_panel(p.id, "owner", store=st,
                           fetcher=lambda c: {"km": 42, "runs": 5})
    d = miniapps.digest("owner", store=st)
    assert "Runs" in d and "km" in d
    # honest when unwired
    p2 = miniapps.create_panel("Cash", "owner", "mono", store=st)
    d2 = miniapps.digest("owner", store=st,
                         fetcher=lambda c: {"_error": "nope"})
    assert "nope" in d2


# ── audio rooms: cohosts, lock, invites, schedule, stats ────────────────────


def _room_store(tmp_path: Path):
    gs = gmod.GroupStore(data_dir=tmp_path / "g")
    g = gs.create_group("Club", "talk", created_by="Ada")
    return ar.RoomStore(data_dir=tmp_path / "r"), gs, g


def test_room_cohosts_lock_muteall(tmp_path: Path):
    st, gs, g = _room_store(tmp_path)
    res = ar.create_room(g.id, "AMA", "host1", "Host", store=st, group_store=gs)
    assert res["ok"]
    rid = res["room"].room_id
    # stranger can't co-host themselves
    assert not ar.add_cohost(rid, "x", "X", "stranger", store=st)["ok"]
    # host names a co-host
    assert ar.add_cohost(rid, "c1", "Cara", "host1", store=st)["ok"]
    # co-host can mute (host powers)
    ar.join_room(rid, "s1", "Sam", store=st)
    ar.promote_speaker(rid, "s1", "host1", store=st)
    m = ar.mute_user(rid, "s1", "c1", True, store=st)
    assert m["ok"] and m["action"] == "muted"
    # co-host can mute-all
    ar.join_room(rid, "s2", "Sol", store=st)
    ar.promote_speaker(rid, "s2", "c1", store=st)
    ma = ar.mute_all(rid, "c1", True, store=st)
    assert ma["ok"] and ma["count"] == 1  # s1 was already muted
    # lock rejects joins
    assert ar.lock_room(rid, "c1", True, store=st)["locked"]
    assert not ar.join_room(rid, "late", "Late", store=st)["ok"]
    assert ar.lock_room(rid, "host1", False, store=st)["locked"] is False
    assert ar.join_room(rid, "late", "Late", store=st)["ok"]
    # remove co-host
    assert ar.remove_cohost(rid, "c1", "host1", store=st)["removed"]
    assert not ar.mute_all(rid, "c1", True, store=st)["ok"]


def test_room_invites_reactions_agenda(tmp_path: Path):
    st, gs, g = _room_store(tmp_path)
    res = ar.create_room(g.id, "Jam", "host1", "Host", store=st, group_store=gs)
    rid = res["room"].room_id
    inv = ar.mint_invite(rid, "host1", "speaker", store=st)
    assert inv["ok"] and inv["role"] == "speaker"
    j = ar.join_with_code(inv["code"], "vip", "Vera", store=st)
    assert j["ok"] and j["role"] == "speaker"
    assert not ar.join_with_code("deadbeef", "x", "X", store=st)["ok"]
    # reactions
    assert ar.react(rid, "vip", "🔥", store=st)["ok"]
    assert ar.react(rid, "vip", "🔥", store=st)["reactions"]["🔥"] == 2
    # agenda + hand-raise note
    assert ar.set_agenda(rid, "host1", "1. intros 2. Q&A", store=st)["ok"]
    ar.raise_hand(rid, "late", "Late", store=st, note="Q about the venue")
    room = st.get(rid)
    assert room.agenda.startswith("1. intros")
    assert room.hand_raise_queue[-1].note == "Q about the venue"
    card = ar.render_room(room)
    assert "Q about the venue" in card and "1. intros" in card and "🔥×2" in card


def test_room_scheduling_and_stats(tmp_path: Path):
    st, gs, g = _room_store(tmp_path)
    res = ar.create_room(g.id, "Future talk", "host1", "Host", store=st,
                         group_store=gs, starts_at=time.time() + 5,
                         agenda="deep dive")
    rid = res["room"].room_id
    assert res["room"].state == "scheduled"
    assert ar.due_starts(now=time.time(), store=st) == []
    due = ar.due_starts(now=time.time() + 10, store=st)
    assert [r.room_id for r in due] == [rid]
    # second call: announced once
    assert ar.due_starts(now=time.time() + 10, store=st) == []
    # lobby check-in before go-live
    assert ar.lobby_checkin(rid, "eager", "Eager", store=st)["lobby_count"] == 1
    assert ar.go_live(rid, "host1", store=st)["ok"]
    assert st.get(rid).state == "live"
    # captions → WebVTT
    ar.add_caption(rid, "welcome everyone", store=st)
    vtt = ar.export_captions_vtt(rid, store=st)
    assert vtt["ok"] and vtt["vtt"].startswith("WEBVTT")
    assert "welcome everyone" in vtt["vtt"]
    # stats
    ar.join_room(rid, "a", "A", store=st)
    ar.join_room(rid, "b", "B", store=st)
    stats = ar.room_stats(rid, store=st)
    assert stats["ok"] and stats["peak_headcount"] >= 3
    # hand-raise expiry
    room = st.get(rid)
    room.hand_raise_expiry = 0.01
    st.put(room)
    ar.raise_hand(rid, "old", "Old", store=st)
    room = st.get(rid)
    room.hand_raise_queue[0].hand_raised_at = time.time() - 100
    st.put(room)
    ar.raise_hand(rid, "new", "New", store=st)
    room = st.get(rid)
    assert [p.user_id for p in room.hand_raise_queue] == ["new"]


def test_room_chat_commands(tmp_path: Path, monkeypatch):
    import os
    monkeypatch.setenv("HOME", str(tmp_path))
    gs = gmod.GroupStore(data_dir=tmp_path / "g")
    g = gs.create_group("Club", "talk", created_by="Ada")
    out = ar.control_room(f"create {g.id} | Test room | agenda:hello",
                          sender="Host", sender_id="host1",
                          group_store=gs)
    assert "room is live" in out
    rid = out.split("`")[1]
    out2 = ar.control_room(f"raise {rid} question about X",
                           sender="Sam", sender_id="s1")
    assert "#1 in the queue" in out2
    out3 = ar.control_room(f"react {rid} 🔥", sender="Sam", sender_id="s1")
    assert "🔥" in out3
    out4 = ar.control_room(f"cohost {rid} s1", sender="Host", sender_id="host1")
    assert "co-host" in out4
    out5 = ar.control_room(f"stats {rid}", sender="Host", sender_id="host1")
    assert "peak" in out5
    assert "room — live audio rooms" in ar.control_room("bogus")


# ── registry: dice grammar + fun tools ──────────────────────────────────────


class SeededRng:
    def __init__(self, seed=7):
        self._r = __import__("random").Random(seed)

    def randint(self, a, b):
        return self._r.randint(a, b)

    def choice(self, seq):
        return self._r.choice(seq)

    def shuffle(self, x):
        return self._r.shuffle(x)


def test_roll_grammar():
    rng = SeededRng(7)
    out = roll("2d6", _rng=rng)
    assert out.startswith("🎲 2d6 →") and "=" in out
    # keep highest
    out = roll("4d6k3", _rng=SeededRng(7))
    assert "keep" in out
    # exploding
    out = roll("4d6!", _rng=SeededRng(7))
    assert "🎲" in out
    # composite + description
    out = roll("2d8+1d6 # damage", _rng=SeededRng(7))
    assert "(damage)" in out
    # advantage
    out = roll("d20 adv", _rng=SeededRng(7))
    assert "advantage" in out and "→" in out
    out = roll("d20 dis", _rng=SeededRng(7))
    assert "disadvantage" in out
    # Fate dice
    out = roll("4dF", _rng=SeededRng(7))
    assert "🎲 4dF" in out
    # crit callout eventually happens with a fixed seed sweep
    seen = {roll("d20", _rng=SeededRng(s)) for s in range(60)}
    assert any("CRIT" in s or "fumble" in s for s in seen)
    # garbage never raises
    assert "can't parse" in roll("banana")
    assert isinstance(roll(""), str)


def test_coin_pick_shuffle():
    out = coin(5, _rng=SeededRng(3))
    assert "5 flips" in out and "Heads" in out
    assert coin().startswith("🪙")
    assert pick("pizza, sushi, tacos", _rng=SeededRng(1)) in {
        "🎯 pizza", "🎯 sushi", "🎯 tacos"}
    assert 'Give me options' in pick("")
    out = shuffle("a, b, c", _rng=SeededRng(1))
    assert out.startswith("🔀") and "a" in out and "b" in out
    assert "at least 2" in shuffle("solo")


def test_registry_allowlist_extended():
    reg = community_registry()
    assert set(reg._tools) == set(COMMUNITY_TOOL_NAMES)
    assert "pick" in COMMUNITY_TOOL_NAMES and "shuffle" in COMMUNITY_TOOL_NAMES
    from nomorals.community.policy import COMMUNITY_CAPABILITIES, is_community_capability
    for name, spec in reg._tools.items():
        assert is_community_capability(spec.capability)
        assert spec.capability in COMMUNITY_CAPABILITIES

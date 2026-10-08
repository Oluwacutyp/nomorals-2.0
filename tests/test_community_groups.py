"""Build-map #84: community groups + events + meetups. All offline."""

import ast
import time
from pathlib import Path

import pytest

from nomorals.community import events as evmod
from nomorals.community import groups as gmod
from nomorals.community import meetups as mmod
from nomorals.community.events import CommunityEvent, EventStore
from nomorals.community.groups import GroupStore
from nomorals.community.meetups import MeetupStore

COMMUNITY_DIR = Path(__file__).resolve().parent.parent / "nomorals" / "community"
FORBIDDEN = ("nomorals.memory", "nomorals.accounts", "nomorals.connectors", "vault")


@pytest.fixture()
def tmp(tmp_path):
    return tmp_path


# ── isolation gate ───────────────────────────────────────────────────────


def test_no_forbidden_imports_in_new_modules():
    for name in ("groups.py", "events.py", "meetups.py"):
        tree = ast.parse((COMMUNITY_DIR / name).read_text(encoding="utf-8"))
        mods: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.append(node.module)
        for mod in mods:
            assert not any(mod.startswith(f) for f in FORBIDDEN[:3]), \
                f"{name}: imports forbidden {mod}"
            assert "vault" not in mod, f"{name}: vault-ish import {mod}"


# ── groups ───────────────────────────────────────────────────────────────


def test_create_join_list(tmp):
    s = GroupStore(data_dir=tmp / "g")
    g = s.create_group("Lagos Devs", "build things together", created_by="Ada")
    assert g is not None and g.id.startswith("grp_")
    assert "Ada" in g.members
    assert s.join_group(g.id, "Bob")
    assert "Bob" in s.get(g.id).members
    # idempotent join
    assert s.join_group(g.id, "Bob")
    assert s.get(g.id).members.count("Bob") == 1
    assert len(s.list_groups()) == 1
    assert not s.join_group("nope", "Ada")
    assert s.create_group("", "topic") is None


def test_post_feed_requires_membership(tmp):
    s = GroupStore(data_dir=tmp / "g")
    g = s.create_group("Hikers", "weekend trails", created_by="Ada")
    # non-member can't post
    assert s.post(g.id, "Zed", "hello") is None
    p = s.post(g.id, "Ada", "Trail on Saturday?")
    assert p is not None
    assert s.feed(g.id)[0].text == "Trail on Saturday?"
    # empty text / author rejected
    assert s.post(g.id, "Ada", "  ") is None
    assert s.post(g.id, "", "hi") is None


def test_quiet_groups_and_nudge(tmp):
    s = GroupStore(data_dir=tmp / "g")
    g = s.create_group("Quiet", "zzz", created_by="Ada")
    # created long ago, never posted
    assert s.quiet_groups(days=7, now=time.time()) and \
        s.quiet_groups(days=7, now=time.time())[0].id == g.id
    s.post(g.id, "Ada", "hello")
    assert s.quiet_groups(days=7, now=time.time() + 60) == []


def test_discovery_warm_matches(tmp):
    s = GroupStore(data_dir=tmp / "g")
    a = s.create_group("A", "a", created_by="Ada")
    b = s.create_group("B", "b", created_by="Ada")
    s.join_group(a.id, "Bob")
    s.join_group(a.id, "Cara")
    s.join_group(b.id, "Bob")
    top = gmod.warm_matches(s, "Ada")
    assert ("Bob", 2) in top
    assert ("Cara", 1) in top
    # Bob before Cara (more shared groups)
    assert top[0][0] == "Bob"
    # nobody in groups → no matches
    assert gmod.warm_matches(s, "Zed") == []
    # chat path never raises
    assert isinstance(gmod.control_cgroup("discover", sender="Ada"), str)


def test_leave_group(tmp):
    s = GroupStore(data_dir=tmp / "g")
    g = s.create_group("X", "x", created_by="Ada")
    assert s.leave_group(g.id, "Ada")
    assert "Ada" not in s.get(g.id).members
    assert not s.leave_group(g.id, "Ada")


# ── events ───────────────────────────────────────────────────────────────


def _future(hours=48):
    return time.time() + hours * 3600.0


def test_event_lifecycle(tmp):
    s = EventStore(data_dir=tmp / "e")
    ev = s.create_event("grp_1", "Hackathon", _future(), where="Lagos",
                        created_by="Ada")
    assert ev is not None and ev.id.startswith("evt_")
    # no calendar provider → honest empty ref
    assert ev.calendar_ref == ""
    assert s.rsvp(ev.id, "Ada", "yes")
    assert s.rsvp(ev.id, "Bob", "maybe")
    assert not s.rsvp(ev.id, "Cara", "definitely")
    counts = s.rsvp_counts(ev.id)
    assert counts == {"yes": 1, "no": 0, "maybe": 1}
    assert s.missing_rsvps(ev.id, ["Ada", "Bob", "Cara"]) == ["Cara"]
    assert s.list_events("grp_1")[0].id == ev.id
    # bad input
    assert s.create_event("grp_1", "", _future()) is None
    assert s.create_event("grp_1", "X", 0) is None


def test_calendar_sync_provider(tmp):
    calls = []
    s = EventStore(data_dir=tmp / "e",
                   calendar_provider=lambda d: calls.append(d) or "cal_123")
    ev = s.create_event("grp_1", "Meetup", _future(), where="Ikeja")
    assert ev.calendar_ref == "cal_123"
    assert calls and calls[0]["title"] == "Meetup"
    # broken provider never breaks events
    s2 = EventStore(data_dir=tmp / "e2",
                    calendar_provider=lambda d: 1 / 0)
    ev2 = s2.create_event("grp_1", "M2", _future())
    assert ev2.calendar_ref == ""


def test_reminders_fire_once(tmp):
    s = EventStore(data_dir=tmp / "e")
    now = time.time()
    ev = s.create_event("grp_1", "Soon", now + 1800.0, created_by="Ada")
    s.rsvp(ev.id, "Ada", "yes")
    due = s.check_reminders(now=now)
    # 30 min out → both the 24h and 1h windows fire
    assert len(due) == 2
    assert all("Soon" in d["text"] for d in due)
    assert due[0]["members"] == ["Ada"]
    # second call: both windows already sent → nothing new
    assert s.check_reminders(now=now) == []
    # past events never remind
    s.create_event("grp_1", "Old", now - 100.0)
    assert all("Old" not in d["text"] for d in s.check_reminders(now=now))


def test_post_event_summary(tmp):
    s = EventStore(data_dir=tmp / "e")
    now = time.time()
    ev = s.create_event("grp_1", "Past", now - 7200.0)
    got = s.post_event_summary(ev.id, "Great turnout!", now=now)
    assert got is not None and got.summary == "Great turnout!"
    # future event → no summary
    ev2 = s.create_event("grp_1", "Future", now + 7200.0)
    assert s.post_event_summary(ev2.id, "x", now=now) is None


def test_weekly_ritual_expansion(tmp):
    s = EventStore(data_dir=tmp / "e")
    # next Wednesday-ish: pick a future weekday explicitly
    now = time.time()
    target_wd = (time.localtime(now).tm_wday + 2) % 7
    day = time.localtime(now + 86400.0)
    starts = time.mktime((day.tm_year, day.tm_mon, day.tm_mday,
                          18, 0, 0, 0, 0, -1))
    ev = s.create_event("grp_1", "Weekly standup", starts,
                        recurrence="weekly", weekday=target_wd)
    assert ev.recurrence == "weekly"
    occs = s.next_occurrences(ev.id, 3, now=now)
    assert len(occs) == 3
    assert all(time.localtime(o).tm_wday == target_wd for o in occs)
    insts = s.expand_recurrence(ev.id, 2, now=now)
    assert len(insts) == 2
    assert len(s.list_events("grp_1")) >= 3
    # non-ritual → nothing
    ev2 = s.create_event("grp_1", "Once", starts)
    assert s.next_occurrences(ev2.id) == []


def test_cancel_event(tmp):
    s = EventStore(data_dir=tmp / "e")
    ev = s.create_event("grp_1", "Gone", _future())
    assert s.cancel_event(ev.id)
    assert s.get(ev.id) is None
    assert not s.cancel_event("nope")


def test_format_event():
    ev = CommunityEvent(id="evt_1", group_id="g", title="T",
                        starts_at=time.time() + 3600.0, where="Lekki")
    ev.rsvps = {"a": "yes"}
    text = EventStore.format_event(ev)
    assert "T" in text and "Lekki" in text and "1 going" in text


# ── meetups ──────────────────────────────────────────────────────────────


def test_venue_poll_lifecycle(tmp):
    es = EventStore(data_dir=tmp / "e")
    ms = MeetupStore(data_dir=tmp / "m", events=es)
    ev = es.create_event("grp_1", "Dinner", _future())
    assert ms.venue_poll("nope", ["A", "B"]) is None
    assert ms.venue_poll(ev.id, ["Only"]) is None  # needs ≥2 options
    poll = ms.venue_poll(ev.id, ["Suya spot", "Mama put"])
    assert poll is not None
    assert ms.vote_poll(poll.id, "Ada", "0")
    assert ms.vote_poll(poll.id, "Bob", "Mama put")
    assert not ms.vote_poll(poll.id, "Cara", "9")  # bad index
    results = ms.poll_results(poll.id)
    assert results == {"Suya spot": 1, "Mama put": 1}
    fresh = ms.get_poll(poll.id)
    assert fresh is not None and fresh.winner() in ("Suya spot", "Mama put")
    assert len(ms.polls_for(ev.id)) == 1


def test_attendees_checkin_turnout(tmp):
    es = EventStore(data_dir=tmp / "e")
    ms = MeetupStore(data_dir=tmp / "m", events=es)
    ev = es.create_event("grp_1", "Dinner", _future())
    es.rsvp(ev.id, "Ada", "yes")
    es.rsvp(ev.id, "Bob", "yes")
    es.rsvp(ev.id, "Cara", "no")
    assert ms.attendees(ev.id) == ["Ada", "Bob"]
    assert ms.check_in(ev.id, "Ada")
    assert ms.check_in(ev.id, "Ada")  # idempotent
    assert ms.checked_in(ev.id) == ["Ada"]
    t = ms.turnout(ev.id)
    assert t["rsvp_yes"] == 2 and t["checked_in"] == 1
    assert t["no_shows"] == ["Bob"]
    assert not ms.check_in("nope", "Ada")


def test_day_of_reminders(tmp):
    es = EventStore(data_dir=tmp / "e")
    ms = MeetupStore(data_dir=tmp / "m", events=es)
    now = time.time()
    ev = es.create_event("grp_1", "Tonight", now + 3 * 3600.0, where="VI")
    es.rsvp(ev.id, "Ada", "yes")
    due = ms.day_of_reminders(now=now)
    assert len(due) == 1 and "Tonight" in due[0]["text"]
    assert due[0]["members"] == ["Ada"]
    # far-future event → no day-of reminder
    es.create_event("grp_1", "Later", now + 30 * 3600.0)
    assert len(ms.day_of_reminders(now=now)) == 1


# ── chat controls ────────────────────────────────────────────────────────


def test_cgroup_chat_full_flow():
    gmod.GroupStore().data_dir  # default store — chat uses real HOME; use control lightly
    # usage / unknown
    assert "cgroup new" in gmod.control_cgroup("")
    assert "cgroup new" in gmod.control_cgroup("bogus")
    # never raises on garbage
    assert isinstance(gmod.control_cgroup("post", sender=""), str)


def test_event_chat_parse():
    out = evmod.control("", sender="Ada")
    assert "event new" in out
    assert "Date format" in evmod._control(
        "event new g1 | T | not-a-date", None, None, "Ada", "")
    assert isinstance(evmod.control("rsvp", sender=""), str)


def test_meetup_chat_parse():
    out = mmod.control("", sender="Ada")
    assert "meetup poll" in out
    assert "Usage" in mmod._control("meetup poll g1", None, None, "Ada", "")
    assert isinstance(mmod.control(None, sender=""), str)


def test_parse_when():
    assert evmod._parse_when("2026-10-10 18:00") > 0
    assert evmod._parse_when("garbage") == 0.0
    assert evmod._parse_when("") == 0.0


def test_ensure_schedule_never_raises():
    assert evmod.ensure_schedule(object()) in (True, False)
    assert evmod.ensure_schedule(None) in (True, False)

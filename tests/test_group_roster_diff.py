"""Scheduled roster diffing (§9.1 / §3): the live member list,
polled on a schedule per group, diffed against the last snapshot.

Two halves: the organ (nomorals/social/chat/group_roster_diff.py —
settings, due, fetch, diff, tick, job) on a bare GroupSurface, and
the chat surface (/gsettings roster grammar, the console roster
section, the /groups dashboard line) through the RuntimeGroupMixin
rig from test_group_cleanup.py.

Design laws under test: the diff posts nothing, ever — every change
lands in the held audit log; a first run seeds the snapshot and
reports no joins; leaves are never asserted from a truncated list;
a group with no member-list route is skipped with the reason
recorded, never faked; one bad group never kills the tick.
"""

from __future__ import annotations

import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.social.chat import group_roster_diff as rd
from nomorals.social.chat.group_surface import (
    GroupStore,
    GroupSurface,
    _default_settings,
)

KEY = "telegram:-1001"
BOTKEY = "telegram-bot:-1002"
CHANNEL = "telegram-bot:-100333"
NOW = 1780000000.0  # a fixed Monday 03:00-ish anchor is not
# assumed; the tests pass explicit `now` values instead.
DAY = 86400.0


def _row(uid, name, *, username="", bot=False, deleted=False):
    return {"user_id": uid, "name": name, "username": username,
            "bot": bot, "deleted": deleted,
            "status_kind": "", "was_online": None,
            "last_name": ""}


def _live(uid, name, *, username="", bot=False, deleted=False):
    """A live-list row in the adapter's shape (id, not user_id)."""
    return {"id": uid, "name": name, "username": username,
            "bot": bot, "deleted": deleted,
            "status_kind": "", "was_online": None,
            "last_name": ""}


class _UserbotAdapter:
    """MTProto-shaped fake: full member list + admin list."""

    def __init__(self, members=(), *, admins=("admin1",),
                 truncated=False):
        self._members = list(members)
        self._admins = list(admins)
        self._truncated = truncated
        self.list_calls = 0

    def admin_members_full(self, chat):
        self.list_calls += 1
        return {"ok": True, "members": list(self._members),
                "count": len(self._members),
                "truncated": self._truncated}

    def admin_administrators(self, chat):
        return {"ok": True, "administrators": [
            {"id": uid, "status": "administrator"}
            for uid in self._admins]}


class _CappedAdapter:
    """MTProto-shaped fake with only the capped list."""

    def __init__(self, members=()):
        self._members = list(members)

    def admin_members(self, chat, limit=100):
        return {"ok": True,
                "members": list(self._members)[:limit],
                "count": len(self._members)}


class _BotAdapter:
    """Bot API-shaped fake: NO member list — per-member probes."""

    def __init__(self, profiles=None):
        self._profiles = profiles or {}

    def admin_member_profile(self, chat, user_id):
        prof = self._profiles.get(str(user_id))
        if prof is None:
            return {"ok": False, "error": "user not found"}
        return {"ok": True, "status": prof["status"],
                "user": prof["user"]}


def _surface(tmp_path: Path) -> GroupSurface:
    return GroupSurface(GroupStore(tmp_path / "g.json").load())


def _seed_ledger(surface, key=KEY, *, members=("u1", "u2")):
    for uid in members:
        surface.note_join(key, uid, username=f"user{uid[1:]}",
                          name=f"User {uid[1:]}", ts=NOW)
    return surface


def _msg(key=KEY, platform="telegram", *, sender_id="admin1"):
    return SimpleNamespace(
        chat=SimpleNamespace(key=key, platform=platform,
                             kind="group", title="Builders",
                             chat_id=key.split(":", 1)[1]),
        sender_id=sender_id, sender="Admin",
        sender_username="boss", text="", meta={}, reply_to="",
        message_id="900")


def _rig(tmp_path: Path, adapter, *, key=KEY, designate=True):
    from nomorals.agents.partner.runtime_group import RuntimeGroupMixin

    surface = _surface(tmp_path)
    if designate:
        surface.set_designated(key, True, by="owner")
    mixin = RuntimeGroupMixin()
    mixin._group_surface = lambda: surface  # type: ignore[method-assign]
    mixin._group_adapter = lambda platform: adapter  # type: ignore[method-assign]
    mixin._operator_or_false = lambda message: False  # type: ignore[method-assign]
    mixin._group_admin_ok = lambda message: True  # type: ignore[method-assign]
    mixin._console_is_admin = lambda k, sender: True  # type: ignore[method-assign]
    return mixin, surface


def _text(out) -> str:
    return getattr(out, "text", out)


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(__import__("tempfile").mkdtemp())
        self.surface = _surface(self._tmp)

    def test_defaults_are_daily_at_3(self):
        self.assertIn("roster_diff", _default_settings())
        cfg = rd.roster_diff_settings(self.surface, KEY)
        self.assertEqual(cfg["cadence"], "daily")
        self.assertEqual(cfg["hour"], 3)

    def test_merge_drops_unknown_keys_and_clamps(self):
        entry = self.surface.store.group(KEY)
        entry.setdefault("settings", {})["roster_diff"] = {
            "cadence": "sometimes", "hour": 99, "bogus": 1}
        cfg = rd.roster_diff_settings(self.surface, KEY)
        self.assertEqual(cfg["cadence"], "off")
        self.assertEqual(cfg["hour"], 23)
        self.assertNotIn("bogus", cfg)

    def test_due_off_never(self):
        self.assertFalse(rd.roster_diff_due(
            {"cadence": "off", "hour": 3, "last_run_ts": 0.0},
            NOW))

    def test_due_daily_after_interval_and_hour(self):
        # 02:00 local; last run a full day ago; hour 2: due.
        base = time.mktime((2026, 11, 15, 2, 0, 0, 0, 0, -1))
        cfg = {"cadence": "daily", "hour": 2,
               "last_run_ts": base - DAY}
        self.assertTrue(rd.roster_diff_due(cfg, base))

    def test_due_daily_before_hour(self):
        # 02:00 local; last run a full day ago; hour 3: not yet.
        base = time.mktime((2026, 11, 15, 2, 0, 0, 0, 0, -1))
        cfg = {"cadence": "daily", "hour": 3,
               "last_run_ts": base - DAY}
        self.assertFalse(rd.roster_diff_due(cfg, base))

    def test_due_weekly(self):
        base = time.mktime((2026, 11, 15, 2, 0, 0, 0, 0, -1))
        cfg = {"cadence": "weekly", "hour": 2,
               "last_run_ts": base - 3 * DAY}
        self.assertFalse(rd.roster_diff_due(cfg, base))
        cfg["last_run_ts"] = base - 8 * DAY
        self.assertTrue(rd.roster_diff_due(cfg, base))

    def test_due_daily_within_interval(self):
        cfg = {"cadence": "daily", "hour": 0,
               "last_run_ts": NOW}
        self.assertFalse(rd.roster_diff_due(cfg, NOW + 3600))

    def test_set_schedule_ok_and_audited(self):
        out = rd.set_roster_diff_schedule(
            self.surface, KEY, cadence="weekly", hour=5, by="a1")
        self.assertEqual(out["settings"]["cadence"], "weekly")
        self.assertEqual(out["settings"]["hour"], 5)
        self.assertIn("next diff", out["next"])
        kinds = [row.get("kind")
                 for row in self.surface.audit(KEY)]
        self.assertIn("roster_diff_schedule", kinds)

    def test_set_schedule_refuses_bad_cadence_and_hour(self):
        with self.assertRaises(rd.GroupRosterDiffRefused):
            rd.set_roster_diff_schedule(
                self.surface, KEY, cadence="sometimes")
        with self.assertRaises(rd.GroupRosterDiffRefused):
            rd.set_roster_diff_schedule(
                self.surface, KEY, cadence="daily", hour=24)

    def test_describe(self):
        self.assertEqual(
            rd.describe_roster_diff({"cadence": "off"}), "off")
        line = rd.describe_roster_diff({
            "cadence": "daily", "hour": 3,
            "last_run_ts": NOW,
            "last_diff": {"joined": 2, "left": 1, "changed": 3,
                          "deleted": 0, "roles": 0,
                          "seeded": False, "truncated": False}})
        self.assertIn("daily at 03:00", line)
        self.assertIn("+2 joined", line)
        self.assertIn("−1 left", line)
        self.assertIn("~3 changed", line)


class DiffTests(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(__import__("tempfile").mkdtemp())
        self.surface = _surface(self._tmp)

    def _rows(self, *specs):
        return [_row(*s[0:2], **(s[2] if len(s) > 2 else {}))
                for s in specs]

    def test_first_run_seeds_no_joins(self):
        _seed_ledger(self.surface, members=("u1", "u2"))
        joins_before = sum(
            1 for row in self.surface.audit(KEY)
            if row.get("kind") == "join")
        rows = self._rows(("u1", "User 1"), ("u2", "User 2"),
                          ("u3", "User 3"))
        out = rd.diff_group_roster(self.surface, KEY, rows,
                                   now=NOW + DAY)
        self.assertTrue(out["seeded"])
        self.assertEqual(out["joined"], [])
        self.assertEqual(out["left"], [])
        # u3 is on the ledger now, silently: no join stamp, and
        # no NEW join audit row for the seeding.
        member = self.surface.members(KEY)["u3"]
        self.assertIsNone(member.get("joined_ts"))
        kinds = [row.get("kind")
                 for row in self.surface.audit(KEY)]
        self.assertIn("roster_diff", kinds)
        self.assertEqual(
            sum(1 for kind in kinds if kind == "join"),
            joins_before)
        state = self.surface._extras(KEY)["roster_diff"]
        self.assertEqual(
            set(state["snapshot"]), {"u1", "u2", "u3"})

    def test_first_run_asserts_leaves(self):
        _seed_ledger(self.surface, members=("u1", "u2", "u3"))
        rows = self._rows(("u1", "User 1"), ("u2", "User 2"))
        out = rd.diff_group_roster(self.surface, KEY, rows,
                                   now=NOW + DAY)
        self.assertTrue(out["seeded"])
        self.assertEqual(out["left"], ["u3"])
        member = self.surface.members(KEY)["u3"]
        self.assertIsNotNone(member.get("left_ts"))

    def test_second_run_join_change_leave(self):
        _seed_ledger(self.surface, members=("u1", "u2", "u3"))
        seed_rows = self._rows(
            ("u1", "User 1", {"username": "user1"}),
            ("u2", "User 2", {"username": "user2"}),
            ("u3", "User 3", {"username": "user3"}))
        rd.diff_group_roster(self.surface, KEY, seed_rows,
                             now=NOW + DAY)
        rows = self._rows(
            ("u1", "User 1", {"username": "user1"}),
            ("u2", "User Two", {"username": "user2new"}),
            ("u4", "User 4", {"username": "user4"}))
        out = rd.diff_group_roster(self.surface, KEY, rows,
                                   now=NOW + 2 * DAY)
        self.assertFalse(out["seeded"])
        self.assertEqual(out["joined"], ["u4"])
        self.assertEqual(out["left"], ["u3"])
        self.assertEqual(
            [(c["user_id"], c["kind"]) for c in out["changed"]],
            [("u2", "username_changed"),
             ("u2", "name_changed")])
        history = self.surface.members(KEY)["u2"].get(
            "identity_history") or []
        self.assertTrue(any(
            h.get("kind") == "username_changed" and
            h.get("new") == "user2new" for h in history))
        kinds = [row.get("kind")
                 for row in self.surface.audit(KEY)]
        self.assertIn("identity_change", kinds)
        self.assertIn("join", kinds)
        self.assertIn("leave", kinds)

    def test_truncated_never_asserts_leaves(self):
        _seed_ledger(self.surface, members=("u1", "u2"))
        rd.diff_group_roster(
            self.surface, KEY,
            self._rows(("u1", "User 1"), ("u2", "User 2")),
            now=NOW + DAY)
        out = rd.diff_group_roster(
            self.surface, KEY, self._rows(("u1", "User 1")),
            now=NOW + 2 * DAY, truncated=True)
        self.assertEqual(out["left"], [])
        self.assertTrue(out["truncated"])
        self.assertIsNone(
            self.surface.members(KEY)["u2"].get("left_ts"))

    def test_role_changes_not_asserted_on_seed(self):
        _seed_ledger(self.surface, members=("u1", "u2"))
        out = rd.diff_group_roster(
            self.surface, KEY,
            self._rows(("u1", "User 1"), ("u2", "User 2")),
            now=NOW + DAY, admin_ids={"admin1"})
        self.assertEqual(out["roles"], [])
        out = rd.diff_group_roster(
            self.surface, KEY,
            self._rows(("u1", "User 1"), ("u2", "User 2")),
            now=NOW + 2 * DAY, admin_ids={"admin1", "u2"})
        self.assertEqual(out["roles"],
                         [{"user_id": "u2", "admin": True}])
        out = rd.diff_group_roster(
            self.surface, KEY,
            self._rows(("u1", "User 1"), ("u2", "User 2")),
            now=NOW + 3 * DAY, admin_ids={"u2"})
        self.assertEqual(out["roles"],
                         [{"user_id": "admin1",
                           "admin": False}])

    def test_rejoin_clears_left(self):
        _seed_ledger(self.surface, members=("u1", "u2"))
        rd.diff_group_roster(
            self.surface, KEY,
            self._rows(("u1", "User 1"), ("u2", "User 2")),
            now=NOW + DAY)
        rd.diff_group_roster(
            self.surface, KEY, self._rows(("u2", "User 2")),
            now=NOW + 2 * DAY)
        self.assertIsNotNone(
            self.surface.members(KEY)["u1"].get("left_ts"))
        out = rd.diff_group_roster(
            self.surface, KEY,
            self._rows(("u1", "User 1"), ("u2", "User 2")),
            now=NOW + 3 * DAY)
        self.assertIn("u1", out["joined"])
        self.assertIsNone(
            self.surface.members(KEY)["u1"].get("left_ts"))

    def test_deleted_flag_roundtrip(self):
        _seed_ledger(self.surface, members=("u1", "u2"))
        rd.diff_group_roster(
            self.surface, KEY,
            self._rows(("u1", "User 1"), ("u2", "User 2")),
            now=NOW + DAY)
        out = rd.diff_group_roster(
            self.surface, KEY,
            self._rows(("u1", "User 1", {"deleted": True}),
                      ("u2", "User 2")),
            now=NOW + 2 * DAY)
        self.assertEqual(out["deleted"], ["u1"])
        self.assertTrue(
            self.surface.members(KEY)["u1"].get("deleted"))
        out = rd.diff_group_roster(
            self.surface, KEY,
            self._rows(("u1", "User 1"), ("u2", "User 2")),
            now=NOW + 3 * DAY)
        self.assertEqual(out["deleted"], [])
        self.assertFalse(
            self.surface.members(KEY)["u1"].get("deleted"))

    def test_last_diff_counts_land_in_settings(self):
        _seed_ledger(self.surface, members=("u1",))
        rd.diff_group_roster(
            self.surface, KEY, self._rows(("u1", "User 1")),
            now=NOW + DAY)
        rd.diff_group_roster(
            self.surface, KEY,
            self._rows(("u1", "User 1"), ("u2", "User 2")),
            now=NOW + 2 * DAY)
        cfg = rd.roster_diff_settings(self.surface, KEY)
        self.assertEqual(cfg["last_diff"]["joined"], 1)
        self.assertEqual(cfg["last_diff"]["total_live"], 2)


class FetchTests(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(__import__("tempfile").mkdtemp())
        self.surface = _surface(self._tmp)

    def test_full_list_wins_and_carries_admins(self):
        adapter = _UserbotAdapter(
            [_live("u1", "User 1", username="user1"),
             _live("u2", "User 2")], admins=("admin1", "u2"))
        out = rd.fetch_live_roster([("telegram", adapter)], KEY)
        self.assertTrue(out["ok"])
        self.assertEqual(out["source"],
                         "the userbot's full member list")
        self.assertEqual(
            {r["user_id"] for r in out["rows"]}, {"u1", "u2"})
        self.assertEqual(out["admin_ids"], {"admin1", "u2"})
        self.assertFalse(out["truncated"])

    def test_capped_fallback(self):
        adapter = _CappedAdapter(
            [_live(f"u{i}", f"User {i}") for i in range(3)])
        out = rd.fetch_live_roster([("telegram", adapter)], KEY)
        self.assertTrue(out["ok"])
        self.assertIn("first 500", out["source"])
        self.assertEqual(len(out["rows"]), 3)

    def test_probe_path_small_ledger(self):
        _seed_ledger(self.surface, members=("u1", "u2"))
        adapter = _BotAdapter({
            "u1": {"status": "member",
                   "user": {"id": "u1", "first_name": "User",
                            "last_name": "One",
                            "username": "user1",
                            "is_bot": False}},
            "u2": {"status": "left",
                   "user": {"id": "u2", "first_name": "User",
                            "last_name": "Two",
                            "username": "user2",
                            "is_bot": False}},
        })
        out = rd.fetch_live_roster(
            [("telegram-bot", adapter)], KEY, surface=self.surface)
        self.assertTrue(out["ok"])
        self.assertEqual(
            {r["user_id"] for r in out["rows"]}, {"u1"})
        self.assertEqual(out["rows"][0]["username"], "user1")

    def test_probe_path_refuses_large_ledger(self):
        _seed_ledger(self.surface,
                     members=tuple(f"u{i}" for i in range(201)))
        adapter = _BotAdapter({})
        out = rd.fetch_live_roster(
            [("telegram-bot", adapter)], KEY, surface=self.surface)
        self.assertFalse(out["ok"])
        self.assertIn("too large", out["error"])

    def test_no_route_is_an_honest_failure(self):
        out = rd.fetch_live_roster([], KEY, surface=self.surface)
        self.assertFalse(out["ok"])
        self.assertIn("no member-list route", out["error"])


class TickTests(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(__import__("tempfile").mkdtemp())
        self.surface = _surface(self._tmp)

    def _fetch(self, rows, *, admins=None):
        def _call(adapters, key, surface):
            return {"ok": True, "rows": list(rows),
                    "source": "stub", "truncated": False,
                    "admin_ids": admins}
        return _call

    def test_tick_diffs_due_claimed_group(self):
        _seed_ledger(self.surface, members=("u1", "u2"))
        self.surface.set_designated(KEY, True, by="owner")
        rows = [_row("u1", "User 1"), _row("u2", "User 2"),
                _row("u3", "User 3")]
        out = rd.run_roster_diff_tick_once(
            surface=self.surface, now=NOW + DAY,
            fetch=self._fetch(rows))
        self.assertEqual(len(out["diffed"]), 1)
        self.assertTrue(out["diffed"][0]["seeded"])
        self.assertEqual(out["failed"], [])
        kinds = [row.get("kind")
                 for row in self.surface.audit(KEY)]
        self.assertIn("roster_diff", kinds)

    def test_tick_skips_not_due(self):
        _seed_ledger(self.surface, members=("u1",))
        self.surface.set_designated(KEY, True, by="owner")
        rd.set_roster_diff_schedule(
            self.surface, KEY, cadence="daily", hour=0)
        # A fresh schedule has last_run 0, so the first tick is
        # due; the second, an hour later, is inside the daily
        # interval and skips.
        out = rd.run_roster_diff_tick_once(
            surface=self.surface, now=NOW,
            fetch=self._fetch([_row("u1", "User 1")]))
        self.assertEqual(len(out["diffed"]), 1)
        out = rd.run_roster_diff_tick_once(
            surface=self.surface, now=NOW + 3600,
            fetch=self._fetch([_row("u1", "User 1")]))
        self.assertEqual(out["diffed"], [])
        self.assertEqual(out["skipped"], 1)

    def test_tick_skips_unclaimed_and_channels(self):
        _seed_ledger(self.surface, members=("u1",))
        # KEY is NOT designated here; CHANNEL carries the channel
        # marker, so the tick skips it as a channel, not a group.
        centry = self.surface.store.group(CHANNEL)
        centry.setdefault("extras", {})["channel"] = {
            "title": "Signals"}
        out = rd.run_roster_diff_tick_once(
            surface=self.surface, now=NOW + DAY,
            fetch=self._fetch([_row("u1", "User 1")]))
        self.assertEqual(out["diffed"], [])
        self.assertEqual(out["failed"], [])
        self.assertEqual(out["skipped"], 1)

    def test_tick_fetch_failure_does_not_kill_pass(self):
        for key in (KEY, BOTKEY):
            _seed_ledger(self.surface, key=key, members=("u1",))
            self.surface.set_designated(key, True, by="owner")

        def _call(adapters, key, surface):
            if key == KEY:
                return {"ok": False, "error": "boom"}
            return {"ok": True,
                    "rows": [_row("u1", "User 1")],
                    "source": "stub", "truncated": False,
                    "admin_ids": None}

        out = rd.run_roster_diff_tick_once(
            surface=self.surface, now=NOW + DAY, fetch=_call)
        self.assertEqual(len(out["diffed"]), 1)
        self.assertEqual(len(out["failed"]), 1)
        self.assertEqual(out["failed"][0]["group"], KEY)

    def test_tick_without_adapters_reports_no_route(self):
        _seed_ledger(self.surface, members=("u1",))
        self.surface.set_designated(KEY, True, by="owner")
        out = rd.run_roster_diff_tick_once(
            surface=self.surface, now=NOW + DAY)
        self.assertEqual(out["diffed"], [])
        self.assertEqual(len(out["failed"]), 1)
        self.assertIn("no member-list route",
                      out["failed"][0]["error"])

    def test_ensure_job_idempotent(self):
        class _Sched:
            def __init__(self):
                self.jobs = []

            def list_jobs(self):
                return list(self.jobs)

            def add(self, name, spec, kind, payload, routine):
                job = {"name": name, "spec": spec,
                       "id": f"j{len(self.jobs)}"}
                self.jobs.append(job)
                return job

            def remove(self, job_id):
                self.jobs = [j for j in self.jobs
                             if j["id"] != job_id]

        sched = _Sched()
        first = rd.ensure_group_roster_diff_job(sched)
        second = rd.ensure_group_roster_diff_job(sched)
        self.assertIs(first, second)
        self.assertEqual(len(sched.jobs), 1)
        # A stale spec is replaced, not duplicated.
        sched.jobs[0]["spec"] = "every 2h"
        third = rd.ensure_group_roster_diff_job(sched)
        self.assertEqual(third["spec"], rd.ROSTER_DIFF_SPEC)
        self.assertEqual(len(sched.jobs), 1)


class GrammarTests(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(__import__("tempfile").mkdtemp())
        self.adapter = _UserbotAdapter(
            [_live("u1", "User 1", username="user1"),
             _live("u2", "User 2", username="user2")])
        self.mixin, self.surface = _rig(
            self._tmp, self.adapter)

    def test_panel_shows_roster_block(self):
        out = _text(self.mixin._control_gsettings(
            "", _msg()))
        self.assertIn("Roster diff", out)
        self.assertIn("daily at 03:00", out)
        self.assertIn("/gsettings roster off|daily|weekly",
                      out)

    def test_grammar_sets_cadence_and_hour(self):
        out = self.mixin._control_gsettings(
            "roster weekly hour 5", _msg())
        self.assertIn("weekly at 05:00", out)
        cfg = rd.roster_diff_settings(self.surface, KEY)
        self.assertEqual(cfg["cadence"], "weekly")
        self.assertEqual(cfg["hour"], 5)

    def test_grammar_off(self):
        out = self.mixin._control_gsettings("roster off", _msg())
        self.assertIn("roster diff for", out)
        self.assertIn("off", out)
        cfg = rd.roster_diff_settings(self.surface, KEY)
        self.assertEqual(cfg["cadence"], "off")

    def test_grammar_bad_cadence_and_hour(self):
        out = self.mixin._control_gsettings(
            "roster sometimes", _msg())
        self.assertIn("usage", out)
        out = self.mixin._control_gsettings(
            "roster daily hour 25", _msg())
        self.assertIn("hour is 0–23", out)
        out = self.mixin._control_gsettings(
            "roster daily hour x", _msg())
        self.assertIn("hour is a number", out)

    def test_grammar_status(self):
        out = self.mixin._control_gsettings("roster status",
                                            _msg())
        self.assertIn("roster diff for", out)
        self.assertIn("daily at 03:00", out)

    def test_grammar_now_seeds_then_reports(self):
        _seed_ledger(self.surface, members=("u1", "u2"))
        out = self.mixin._control_gsettings("roster now", _msg())
        self.assertIn("first run", out)
        self.assertIn("2 members live", out)
        # Second run: a new member joins, one renames.
        self.adapter._members = [
            _live("u1", "User 1", username="user1"),
            _live("u2", "User Two", username="user2"),
            _live("u3", "User 3", username="user3")]
        out = self.mixin._control_gsettings("roster now", _msg())
        self.assertIn("+1 joined", out)
        self.assertIn("~1 changed", out)

    def test_grammar_now_unclaimed(self):
        # A fresh store: the setUp rig designated KEY in its own
        # store, so an unclaimed check needs a clean one.
        import tempfile

        fresh = Path(tempfile.mkdtemp())
        mixin, _surface = _rig(fresh, self.adapter,
                               designate=False)
        out = mixin._control_gsettings("roster now", _msg())
        self.assertIn("isn't switched on", out)

    def test_grammar_now_no_route(self):
        mixin, surface = _rig(self._tmp, _BotAdapter({}),
                              designate=True)
        _seed_ledger(surface, members=tuple(
            f"u{i}" for i in range(201)))
        out = mixin._control_gsettings("roster now", _msg())
        self.assertIn("couldn't run", out)
        self.assertIn("too large", out)

    def test_console_menu_and_section(self):
        out = _text(self.mixin._console_group_menu(
            self.surface, KEY, _msg(), "g1"))
        self.assertIn("roster diff:", out)
        out = _text(self.mixin._console_section(
            self.surface, KEY, _msg(), "g1", "roster"))
        self.assertIn("daily at 03:00", out)
        self.assertIn("<code>now</code>", out)

    def test_console_roster_words(self):
        out = self.mixin._console_apply(
            self.surface, KEY, _msg(), "admin1", "roster",
            ["weekly", "hour", "6"])
        self.assertIn("weekly at 06:00", out)
        cfg = rd.roster_diff_settings(self.surface, KEY)
        self.assertEqual(cfg["cadence"], "weekly")

    def test_dashboard_line(self):
        lines = self.mixin._groups_group_lines(
            self.surface, KEY, "g1")
        self.assertTrue(any(
            line.startswith("roster diff ") for line in lines))
        self.assertTrue(any(
            "daily at 03:00" in line for line in lines))

    def test_note_leave_is_silent(self):
        _seed_ledger(self.surface, members=("u1",))
        self.surface.note_leave(KEY, "u1", ts=NOW)
        member = self.surface.members(KEY)["u1"]
        self.assertEqual(member.get("left_ts"), NOW)
        kinds = [row.get("kind")
                 for row in self.surface.audit(KEY)]
        self.assertIn("leave", kinds)

    def test_record_leave_still_renders_goodbye(self):
        _seed_ledger(self.surface, members=("u1",))
        self.surface.update_settings(KEY, goodbye_enabled=True)
        out = self.surface.record_leave(KEY, "u1", ts=NOW)
        self.assertIn("User 1", out)
        self.assertIsNotNone(
            self.surface.members(KEY)["u1"].get("left_ts"))


if __name__ == "__main__":
    unittest.main()

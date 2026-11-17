"""WhatsApp group health analytics (census §6 #6).

The vault already holds everything a health report needs — the
member ledger (message_count / last_seen / joined_ts / left_ts),
the 500-row audit ring (join/leave rows with timestamps), the
10-day rhythm counters (messages/joins/leaves per local day) and
the roster-diff state — but nothing turns it into health: no
churn number, no rhythm shape, no silent-member list, no
cross-group overview, and the scheduled roster diff cannot even
fetch a live WhatsApp roster (fetch_live_roster only knows the
Telegram-shaped admin_members verbs; the bridge's
group_participants verb is never tried).

These tests drive the new organ (nomorals/social/chat/group_health.py)
on a bare GroupSurface, the WhatsApp branch of fetch_live_roster,
and the /ghealth chat surface through the RuntimeGroupMixin rig.
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.social.chat import group_health as gh
from nomorals.social.chat import group_roster_diff as rd
from nomorals.social.chat.base import ChatKind
from nomorals.social.chat.control import (
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    LIST_GROUPS,
    LIST_ONELINERS,
    PUBLIC_COMMANDS,
    help_text,
    list_catalog,
)
from nomorals.social.chat.group_surface import (
    GroupStore,
    GroupSurface,
)

WA = "whatsapp:120363025000000@g.us"
WA2 = "whatsapp:120363025000001@g.us"
TG = "telegram:-100111"
NOW = 1780000000.0  # fixed anchor; tests pass explicit `now`
DAY = 86400.0


def _surface(tmp_path: Path) -> GroupSurface:
    return GroupSurface(GroupStore(tmp_path / "g.json").load())


def _seed_group(surface, key=WA, *, members=(), now=NOW):
    """Seed a ledger via note_join, then let the caller mutate
    last_seen / message_count on the plain ledger dicts."""
    for uid, name in members:
        surface.note_join(key, uid, username=name.lower(),
                          name=name, ts=now)
    return surface


def _member(surface, key, uid):
    return surface.store.group(key)["members"][uid]


def _set_rhythm(surface, key, days):
    surface._extras(key)["rhythm_days"] = dict(days)


def _audit_row(surface, key, kind, ts, **details):
    entry = surface.store.group(key)
    entry["audit"].append({"ts": ts, "kind": kind, **details})


def _group_msg(key=WA, *, sender_id="admin1", platform="whatsapp"):
    return SimpleNamespace(
        chat=SimpleNamespace(key=key, platform=platform,
                             kind=ChatKind.GROUP,
                             chat_id=key.split(":", 1)[1]),
        sender_id=sender_id, sender="Admin",
        sender_username="boss", text="", meta={}, reply_to="",
        message_id="900")


def _dm_msg(sender_id="owner1", *, platform="whatsapp"):
    chat = SimpleNamespace(key=f"{platform}:{sender_id}",
                           platform=platform, kind=ChatKind.DM,
                           chat_id=sender_id)
    return SimpleNamespace(chat=chat, text="", meta={},
                           sender="someone", sender_id=sender_id)


def _rig(tmp_path: Path, adapter, *, key=WA, designate=True,
         operator=False, admin_ok=True, console_admin=True):
    from nomorals.agents.partner.runtime_group import RuntimeGroupMixin

    surface = _surface(tmp_path)
    if designate:
        surface.set_designated(key, True, by="owner")
    mixin = RuntimeGroupMixin()
    mixin._group_surface = lambda: surface  # type: ignore[method-assign]
    mixin._group_adapter = lambda platform: adapter  # type: ignore[method-assign]
    mixin._operator_or_false = lambda message: operator  # type: ignore[method-assign]
    mixin._group_admin_ok = lambda message: admin_ok  # type: ignore[method-assign]
    mixin._console_is_admin = lambda k, sender: console_admin  # type: ignore[method-assign]
    return mixin, surface


def _text(out) -> str:
    return getattr(out, "text", out)


class _WAAdapter:
    """WhatsApp-shaped fake: the bridge's group roster only."""

    def __init__(self, participants=(), *, connected=True):
        self._parts = list(participants)
        self._connected = connected

    def group_participants(self, chat):
        return [{"jid": jid, "name": name, "role": role}
                for jid, name, role in self._parts]


class _NoRouteAdapter:
    """No member-list verb at all."""


class OrganTests(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self.surface = _surface(self._tmp)

    def test_unknown_group_returns_none(self):
        self.assertIsNone(gh.build_group_health(
            self.surface, WA, now=NOW))

    def test_silent_members_by_last_seen(self):
        _seed_group(self.surface, members=(
            ("u1", "Ada"), ("u2", "Bo"), ("u3", "Cy")))
        # Ada spoke yesterday; Bo last spoke 20 days ago;
        # Cy never spoke (joined 30 days ago).
        _member(self.surface, WA, "u1")["last_seen"] = NOW - DAY
        _member(self.surface, WA, "u1")["message_count"] = 40
        _member(self.surface, WA, "u2")["first_seen"] = NOW - 30 * DAY
        _member(self.surface, WA, "u2")["last_seen"] = NOW - 20 * DAY
        _member(self.surface, WA, "u2")["message_count"] = 12
        _member(self.surface, WA, "u3")["first_seen"] = NOW - 30 * DAY
        _member(self.surface, WA, "u3")["last_seen"] = NOW - 30 * DAY
        _member(self.surface, WA, "u3")["message_count"] = 0
        health = gh.build_group_health(self.surface, WA, now=NOW)
        self.assertEqual(health["members_total"], 3)
        self.assertEqual(health["active_7d"], 1)
        self.assertEqual(health["silent_count"], 2)
        silent_names = [m["name"] for m in health["silent"]]
        self.assertEqual(silent_names, ["Cy", "Bo"])
        self.assertEqual(health["never_spoke"], 1)
        self.assertEqual(
            [m["days_quiet"] for m in health["silent"]], [30, 20])

    def test_new_member_is_not_silent(self):
        _seed_group(self.surface, members=(("u1", "Ada"),))
        # Joined yesterday, never spoke — new, not silent.
        _member(self.surface, WA, "u1")["first_seen"] = NOW - DAY
        _member(self.surface, WA, "u1")["last_seen"] = NOW - DAY
        health = gh.build_group_health(self.surface, WA, now=NOW)
        self.assertEqual(health["silent_count"], 0)

    def test_left_bot_and_deleted_are_not_silent(self):
        _seed_group(self.surface, members=(
            ("u1", "Ada"), ("u2", "Bo"), ("u3", "Cy"),
            ("u4", "Dee")))
        for uid in ("u1", "u2", "u3", "u4"):
            _member(self.surface, WA, uid)["first_seen"] = NOW - 30 * DAY
            _member(self.surface, WA, uid)["last_seen"] = NOW - 30 * DAY
        _member(self.surface, WA, "u1")["left_ts"] = NOW - 5 * DAY
        _member(self.surface, WA, "u2")["is_bot"] = True
        _member(self.surface, WA, "u3")["deleted"] = True
        health = gh.build_group_health(self.surface, WA, now=NOW)
        self.assertEqual(health["members_total"], 1)
        self.assertEqual(health["members_left"], 1)
        self.assertEqual(health["bots"], 1)
        self.assertEqual(health["deleted"], 1)
        self.assertEqual(
            [m["name"] for m in health["silent"]], ["Dee"])

    def test_churn_from_rhythm_days(self):
        _seed_group(self.surface, members=(("u1", "Ada"),))
        # NOW is 2026-05-28, so the 7-day window is May 22–28.
        _set_rhythm(self.surface, WA, {
            "2026-05-24": {"messages": 12, "joins": 2, "leaves": 1},
            "2026-05-25": {"messages": 0, "joins": 0, "leaves": 0},
            "2026-05-26": {"messages": 30, "joins": 0, "leaves": 3},
            "2026-05-27": {"messages": 8, "joins": 1, "leaves": 0},
        })
        health = gh.build_group_health(self.surface, WA, now=NOW,
                                       window_days=7)
        self.assertEqual(health["msgs_7d"], 50)
        self.assertEqual(health["joins_7d"], 3)
        self.assertEqual(health["leaves_7d"], 4)
        self.assertEqual(health["busiest_day"], ("2026-05-26", 30))
        self.assertEqual(health["quiet_days"], ["2026-05-25"])

    def test_newcomers_and_leavers_from_audit(self):
        _seed_group(self.surface, members=(("u1", "Ada"),))
        _audit_row(self.surface, WA, "join", NOW - 2 * DAY,
                   user_id="u9", name="Nia")
        _audit_row(self.surface, WA, "join", NOW - 40 * DAY,
                   user_id="u8", name="Old")
        _audit_row(self.surface, WA, "leave", NOW - DAY,
                   user_id="u7", name="Mia")
        health = gh.build_group_health(self.surface, WA, now=NOW)
        self.assertEqual(
            [m["name"] for m in health["newcomers_7d"]], ["Nia"])
        self.assertEqual(
            [m["name"] for m in health["leavers_7d"]], ["Mia"])

    def test_audit_window_completeness_is_honest(self):
        _seed_group(self.surface, members=(("u1", "Ada"),))
        # Audit only reaches back 2 days — the 7-day window is
        # NOT fully covered, so the report must say so.
        _audit_row(self.surface, WA, "join", NOW - 2 * DAY,
                   user_id="u9", name="Nia")
        health = gh.build_group_health(self.surface, WA, now=NOW)
        self.assertFalse(health["audit_window_complete"])
        self.assertAlmostEqual(
            health["audit_covers_since"], NOW - 2 * DAY)
        # An old row covers the window.
        _audit_row(self.surface, WA, "join", NOW - 10 * DAY,
                   user_id="u8", name="Old")
        health = gh.build_group_health(self.surface, WA, now=NOW)
        self.assertTrue(health["audit_window_complete"])

    def test_verdict_quiet(self):
        _seed_group(self.surface, members=(("u1", "Ada"),))
        _set_rhythm(self.surface, WA, {
            "2026-05-27": {"messages": 0, "joins": 0, "leaves": 0},
        })
        health = gh.build_group_health(self.surface, WA, now=NOW)
        self.assertIn("quiet", health["verdict"])

    def test_verdict_churning(self):
        _seed_group(self.surface, members=(("u1", "Ada"),))
        _set_rhythm(self.surface, WA, {
            "2026-05-26": {"messages": 5, "joins": 0, "leaves": 3},
            "2026-05-27": {"messages": 5, "joins": 1, "leaves": 2},
        })
        health = gh.build_group_health(self.surface, WA, now=NOW)
        self.assertIn("churning", health["verdict"])

    def test_verdict_mostly_silent(self):
        _seed_group(self.surface, members=(
            ("u1", "Ada"), ("u2", "Bo"), ("u3", "Cy"),
            ("u4", "Dee"), ("u5", "Eli")))
        for uid in ("u1", "u2", "u3", "u4", "u5"):
            _member(self.surface, WA, uid)["first_seen"] = NOW - 30 * DAY
            _member(self.surface, WA, uid)["last_seen"] = NOW - 30 * DAY
        _member(self.surface, WA, "u1")["last_seen"] = NOW - DAY
        _set_rhythm(self.surface, WA, {
            "2026-05-27": {"messages": 4, "joins": 0, "leaves": 0},
        })
        health = gh.build_group_health(self.surface, WA, now=NOW)
        self.assertIn("gone quiet", health["verdict"])

    def test_verdict_healthy(self):
        _seed_group(self.surface, members=(
            ("u1", "Ada"), ("u2", "Bo")))
        for uid in ("u1", "u2"):
            _member(self.surface, WA, uid)["last_seen"] = NOW - DAY
            _member(self.surface, WA, uid)["message_count"] = 5
        _set_rhythm(self.surface, WA, {
            "2026-05-27": {"messages": 20, "joins": 0, "leaves": 0},
        })
        health = gh.build_group_health(self.surface, WA, now=NOW)
        self.assertIn("healthy", health["verdict"])

    def test_render_names_no_raw_ids(self):
        _seed_group(self.surface, members=(("u1", "Ada"),))
        _member(self.surface, WA, "u1")["first_seen"] = NOW - 30 * DAY
        _member(self.surface, WA, "u1")["last_seen"] = NOW - 30 * DAY
        health = gh.build_group_health(self.surface, WA, now=NOW)
        text = gh.render_group_health(health)
        self.assertIn("Ada", text)
        self.assertNotIn("u1", text)
        self.assertNotIn(WA, text)
        self.assertIn("silent", text.lower())

    def test_render_caps_silent_list(self):
        members = tuple((f"u{i}", f"User{i}") for i in range(15))
        _seed_group(self.surface, members=members)
        for uid, _ in members:
            _member(self.surface, WA, uid)["first_seen"] = NOW - 30 * DAY
            _member(self.surface, WA, uid)["last_seen"] = NOW - 30 * DAY
        health = gh.build_group_health(self.surface, WA, now=NOW)
        text = gh.render_group_health(health)
        self.assertIn("and 5 more", text)

    def test_overview_is_whatsapp_only(self):
        _seed_group(self.surface, WA, members=(("u1", "Ada"),))
        _seed_group(self.surface, WA2, members=(("u2", "Bo"),))
        _seed_group(self.surface, TG, members=(("u3", "Cy"),))
        # A WhatsApp channel is not a group.
        chan = "whatsapp:120363025000009@newsletter"
        _seed_group(self.surface, chan, members=(("u4", "Dee"),))
        self.surface.store.group(chan).setdefault("extras", {})[
            "channel"] = {"title": "Announcements"}
        overview = gh.build_overview(self.surface, now=NOW)
        self.assertEqual(
            [g["group_key"] for g in overview["groups"]],
            [WA, WA2])
        self.assertEqual(overview["totals"]["groups"], 2)
        self.assertEqual(overview["totals"]["members"], 2)

    def test_overview_render_one_line_per_group(self):
        _seed_group(self.surface, WA, members=(("u1", "Ada"),))
        _member(self.surface, WA, "u1")["last_seen"] = NOW - DAY
        overview = gh.build_overview(self.surface, now=NOW)
        text = gh.render_overview(overview)
        self.assertIn("1 WhatsApp group", text)
        self.assertIn("1 member", text)
        self.assertIn(WA, text)


class FetchBranchTests(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self.surface = _surface(self._tmp)

    def test_group_participants_branch(self):
        adapter = _WAAdapter([
            ("120363011111111@lid", "Ada", "admin"),
            ("120363022222222@lid", "Bo", ""),
        ])
        fetched = rd.fetch_live_roster(
            [("whatsapp", adapter)], WA, surface=self.surface)
        self.assertTrue(fetched["ok"])
        self.assertEqual(len(fetched["rows"]), 2)
        self.assertEqual(fetched["source"], "the bridge's group roster")
        self.assertEqual(
            fetched["admin_ids"], {"120363011111111@lid"})

    def test_group_participants_empty_is_not_a_roster(self):
        fetched = rd.fetch_live_roster(
            [("whatsapp", _WAAdapter([]))], WA,
            surface=self.surface)
        self.assertFalse(fetched["ok"])

    def test_no_route_still_honest(self):
        fetched = rd.fetch_live_roster(
            [("whatsapp", _NoRouteAdapter())], WA,
            surface=self.surface)
        self.assertFalse(fetched["ok"])
        self.assertIn("no member-list route", fetched["error"])

    def test_refresh_diffs_whatsapp_roster(self):
        _seed_group(self.surface, members=(("u1", "Ada"),))
        adapter = _WAAdapter([
            ("u1", "Ada", "admin"),
            ("u2", "Bo", ""),
        ])
        # First diff seeds the snapshot silently; the refresh
        # after it is the run that reports the join.
        rd.diff_group_roster(
            self.surface, WA,
            [{"user_id": "u1", "name": "Ada", "username": "ada",
              "bot": False, "deleted": False, "status_kind": "",
              "was_online": None, "last_name": ""}],
            now=NOW - DAY)
        result = gh.refresh_live_roster(
            self.surface, WA, [("whatsapp", adapter)], now=NOW)
        self.assertTrue(result["ok"])
        self.assertEqual(result["joined"], ["u2"])
        self.assertFalse(result.get("seeded"))


class HandlerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())

    def _seed_wa(self, surface, *, members=(("u1", "Ada"),)):
        # The handler reads the REAL clock (build_group_health
        # with now=None), so handler seeds anchor on real now —
        # the fixed NOW anchor belongs to the organ tests only.
        now = time.time()
        _seed_group(surface, members=members, now=now)
        _member(surface, WA, "u1")["last_seen"] = now - DAY
        _member(surface, WA, "u1")["message_count"] = 9
        _set_rhythm(surface, WA, {
            time.strftime("%Y-%m-%d", time.gmtime(now)): {
                "messages": 9, "joins": 0, "leaves": 0},
        })
        return now

    def test_in_group_report(self):
        mixin, surface = _rig(self._tmp, _WAAdapter())
        self._seed_wa(surface)
        out = _text(mixin._control_ghealth("", _group_msg()))
        self.assertIn("Ada", out)
        self.assertIn("healthy", out)

    def test_in_group_non_admin_refused(self):
        mixin, surface = _rig(self._tmp, _WAAdapter(), admin_ok=False)
        self._seed_wa(surface)
        out = _text(mixin._control_ghealth("", _group_msg()))
        self.assertIn("admin", out.lower())
        self.assertNotIn("healthy", out)

    def test_telegram_group_refused_honestly(self):
        mixin, surface = _rig(self._tmp, _WAAdapter())
        self._seed_wa(surface)
        out = _text(mixin._control_ghealth(
            "", _group_msg(key=TG, platform="telegram")))
        self.assertIn("WhatsApp", out)

    def test_dm_overview_operator(self):
        mixin, surface = _rig(self._tmp, _WAAdapter(), operator=True)
        self._seed_wa(surface)
        _seed_group(surface, WA2, members=(("u2", "Bo"),))
        out = _text(mixin._control_ghealth("", _dm_msg()))
        self.assertIn("2 WhatsApp groups", out)

    def test_dm_overview_non_operator_sees_own_only(self):
        mixin, surface = _rig(self._tmp, _WAAdapter(),
                              operator=False, console_admin=False)
        self._seed_wa(surface)
        out = _text(mixin._control_ghealth("", _dm_msg()))
        self.assertIn("don't admin any", out)

    def test_dm_selector_single_match(self):
        mixin, surface = _rig(self._tmp, _WAAdapter(), operator=True)
        self._seed_wa(surface)
        surface.store.group(WA)["title"] = "Devon Builders"
        out = _text(mixin._control_ghealth(
            "builders", _dm_msg()))
        self.assertIn("healthy", out)

    def test_dm_selector_no_match(self):
        mixin, surface = _rig(self._tmp, _WAAdapter(), operator=True)
        self._seed_wa(surface)
        out = _text(mixin._control_ghealth("zzz", _dm_msg()))
        self.assertIn("no WhatsApp group", out)

    def test_dm_selector_ambiguous(self):
        mixin, surface = _rig(self._tmp, _WAAdapter(), operator=True)
        self._seed_wa(surface)
        _seed_group(surface, WA2, members=(("u2", "Bo"),))
        surface.store.group(WA)["title"] = "Devon Builders"
        surface.store.group(WA2)["title"] = "Devon Traders"
        out = _text(mixin._control_ghealth("devon", _dm_msg()))
        self.assertIn("more than one", out)

    def test_refresh_in_group(self):
        adapter = _WAAdapter([
            ("u1", "Ada", "admin"),
            ("u2", "Bo", ""),
        ])
        mixin, surface = _rig(self._tmp, adapter)
        seeded_now = self._seed_wa(surface)
        # First diff seeds the snapshot silently; the refresh
        # after it is the run that reports the join.
        rd.diff_group_roster(
            surface, WA,
            [{"user_id": "u1", "name": "Ada", "username": "ada",
              "bot": False, "deleted": False, "status_kind": "",
              "was_online": None, "last_name": ""}],
            now=seeded_now - DAY)
        out = _text(mixin._control_ghealth(
            "refresh", _group_msg()))
        self.assertIn("+1 joined", out)
        self.assertIn("healthy", out)

    def test_refresh_needs_claim(self):
        mixin, surface = _rig(self._tmp, _WAAdapter(),
                              designate=False)
        self._seed_wa(surface)
        out = _text(mixin._control_ghealth(
            "refresh", _group_msg()))
        self.assertIn("isn't switched on", out)

    def test_board_line_for_whatsapp_group(self):
        mixin, surface = _rig(self._tmp, _WAAdapter())
        self._seed_wa(surface)
        lines = mixin._groups_group_lines(surface, WA, "g1")
        joined = "\n".join(lines)
        self.assertIn("health", joined)
        self.assertIn("1 spoke", joined)


class RegistryTests(unittest.TestCase):
    def test_ghealth_registered_like_groups(self):
        self.assertIn("ghealth", CONTROL_COMMANDS)
        self.assertIn("ghealth", PUBLIC_COMMANDS)
        page = COMMAND_DETAILS["ghealth"]
        self.assertTrue(page["what"] and page["usage"]
                        and page["example"])
        self.assertIn("ghealth", LIST_ONELINERS)
        slots = [group for group, kinds in LIST_GROUPS
                 if "ghealth" in kinds]
        self.assertEqual(len(slots), 1)
        self.assertIn("/ghealth", help_text())
        self.assertIn("/ghealth", list_catalog())

    def test_public_count_pin(self):
        # 262 pre-digest; /digest adds one public command.
        self.assertEqual(len(PUBLIC_COMMANDS), 274)


if __name__ == "__main__":
    unittest.main()

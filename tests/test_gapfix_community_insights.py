"""Gapfix R2-27: /ghealth routes through community/insights.

Failing-first: the organ calls below do not exist in
group_health.py yet, so tests 1-6 fail before the wiring and
pass after it. Test 7 is a guard pin (passes before and after):
no turnout data is ever invented.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nomorals.community import insights as insights_mod
from nomorals.social.chat import group_health as gh
from nomorals.social.chat.group_surface import GroupStore, GroupSurface

WA = "whatsapp:120363025000000@g.us"
NOW = 1780000000.0  # fixed anchor; tests pass explicit `now`
DAY = 86400.0


def _surface(tmp_path: Path) -> GroupSurface:
    return GroupSurface(GroupStore(tmp_path / "g.json").load())


def _seed_group(surface, key=WA, *, members=(), now=NOW):
    for uid, name in members:
        surface.note_join(key, uid, username=name.lower(),
                          name=name, ts=now)
    return surface


def _member(surface, key, uid):
    return surface.store.group(key)["members"][uid]


def _seed_trio(surface):
    """Ada spoke yesterday (40 msgs); Bo last spoke 20d ago (12);
    Cy never spoke (joined 30d ago)."""
    _seed_group(surface, members=(
        ("u1", "Ada"), ("u2", "Bo"), ("u3", "Cy")))
    _member(surface, WA, "u1")["last_seen"] = NOW - DAY
    _member(surface, WA, "u1")["message_count"] = 40
    _member(surface, WA, "u2")["first_seen"] = NOW - 30 * DAY
    _member(surface, WA, "u2")["last_seen"] = NOW - 20 * DAY
    _member(surface, WA, "u2")["message_count"] = 12
    _member(surface, WA, "u3")["first_seen"] = NOW - 30 * DAY
    _member(surface, WA, "u3")["last_seen"] = NOW - 30 * DAY
    _member(surface, WA, "u3")["message_count"] = 0


class OrganDelegationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self.surface = _surface(self._tmp)

    def test_quiet_comes_from_organ(self):
        _seed_trio(self.surface)
        real = insights_mod.quiet_members
        calls: list = []

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return real(*args, **kwargs)

        with mock.patch.object(insights_mod, "quiet_members", spy):
            health = gh.build_group_health(self.surface, WA, now=NOW)
        self.assertEqual(len(calls), 1)
        args, kwargs = calls[0]
        days = kwargs.get("days", args[2] if len(args) > 2 else None)
        now = kwargs.get("now", args[3] if len(args) > 3 else None)
        min_age = kwargs.get("min_age_days")
        self.assertEqual(days, gh.SILENT_DAYS)
        self.assertEqual(now, NOW)
        self.assertEqual(min_age, gh.SILENT_DAYS)
        # Behaviour preserved: the same silent list as before.
        self.assertEqual(
            [m["name"] for m in health["silent"]], ["Cy", "Bo"])
        self.assertEqual(
            [m["days_quiet"] for m in health["silent"]], [30, 20])

    def test_engagement_agrees_with_organ(self):
        _seed_trio(self.surface)
        health = gh.build_group_health(self.surface, WA, now=NOW)
        expected = insights_mod.engagement_from_counts(
            {"u1": 40, "u2": 12, "u3": 0})
        self.assertEqual(health["engagement"], expected)
        self.assertEqual(health["engagement"]["total"], 52)
        self.assertEqual(health["engagement"]["active_members"], 2)
        self.assertEqual(
            health["engagement"]["per_member"][0]["member_id"], "u1")

    def test_top_talkers_rendered(self):
        _seed_trio(self.surface)
        health = gh.build_group_health(self.surface, WA, now=NOW)
        text = gh.render_group_health(health)
        self.assertIn("Top talkers (all time): Ada (40) · Bo (12)",
                      text)

    def test_silent_never_spoke_renders_never(self):
        _seed_group(self.surface, members=(("u4", "Dee"),))
        _member(self.surface, WA, "u4")["first_seen"] = NOW - 30 * DAY
        _member(self.surface, WA, "u4")["last_seen"] = 0.0
        _member(self.surface, WA, "u4")["message_count"] = 0
        health = gh.build_group_health(self.surface, WA, now=NOW)
        self.assertEqual(health["silent_count"], 1)
        entry = health["silent"][0]
        self.assertEqual(entry["name"], "Dee")
        # Never spoke is reported, never faked as a giant day count.
        self.assertIsNone(entry["days_quiet"])
        text = gh.render_group_health(health)
        self.assertIn("Dee (never spoke)", text)

    def test_turnout_not_invented(self):
        _seed_trio(self.surface)
        health = gh.build_group_health(self.surface, WA, now=NOW)
        self.assertNotIn("turnout", health)
        text = gh.render_group_health(health)
        self.assertNotIn("RSVP", text)


class OrganExtensionTests(unittest.TestCase):
    def test_newcomer_guard_lives_in_organ(self):
        now = 1_000_000.0
        members = [{"member_id": "u1", "display_name": "New",
                    "first_seen": now - DAY}]
        # Never posted, joined yesterday: quiet without the guard,
        # not quiet with it.
        self.assertEqual(
            [q["member_id"] for q in insights_mod.quiet_members(
                members, [], days=14, now=now)],
            ["u1"])
        self.assertEqual(
            insights_mod.quiet_members(
                members, [], days=14, now=now, min_age_days=14),
            [])
        # Joined 30 days ago: quiet even with the guard.
        members[0]["first_seen"] = now - 30 * DAY
        self.assertEqual(
            [q["member_id"] for q in insights_mod.quiet_members(
                members, [], days=14, now=now, min_age_days=14)],
            ["u1"])

    def test_engagement_from_counts(self):
        result = insights_mod.engagement_from_counts(
            {"u1": 40, "u2": 12, "u3": 0})
        self.assertEqual(result["total"], 52)
        self.assertEqual(result["active_members"], 2)
        self.assertEqual(result["per_member"][0]["member_id"], "u1")
        self.assertAlmostEqual(
            result["per_member"][0]["share"], 40 / 52)
        self.assertEqual(
            insights_mod.engagement_from_counts({}),
            {"total": 0, "active_members": 0, "per_member": []})
        with self.assertRaises(ValueError):
            insights_mod.engagement_from_counts({"u1": -1})

    def test_engagement_and_from_counts_agree(self):
        messages = [{"member_id": "u1", "ts": 100},
                    {"member_id": "u1", "ts": 200},
                    {"member_id": "u2", "ts": 300}]
        self.assertEqual(
            insights_mod.engagement(messages),
            insights_mod.engagement_from_counts({"u1": 2, "u2": 1}))


if __name__ == "__main__":
    unittest.main()

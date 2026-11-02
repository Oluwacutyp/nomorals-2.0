"""Pins for nomorals/games/titles.py (tail-116): behavior + NOT-recording.

Sites pinned (all exclusions — fail-closed withholdings on a
best-effort enrichment read, plus a never-raise display shell):

* 199 — ``check_unlocks``: the achievements read (deferred import +
  call) raising degrades to an empty achievement set. The failure
  withholds achievement-gated unlocks for THIS call only: grants are
  additive and recomputed every call, so the next healthy call grants
  the rest; the default title is granted independently and persists.
  Nothing is unlocked that was not earned.
* 206 — ``check_unlocks``: the skills read raising degrades the same
  way (empty learned set; skill-gated titles withheld this call).
* 272 — module-level ``active_title``: the never-raise convenience
  shell around ``TitleStore.active`` (whose own guard at the read
  site already degrades to "" with a debug log). A store that raises
  outside that inner guard — e.g. at construction — still yields ""
  rather than breaking a display surface.

NOT-recording: the 199/206 guards bind nothing and emit nothing;
``titles._log`` is patched with a MagicMock around the crash calls
and ``method_calls`` must stay empty. The 272 wrapper likewise adds
no recording of its own. This file contains NO except handlers:
triggers are mock side_effects raised inside the target's own
handlers.
"""
import unittest
from unittest import mock

from nomorals.games import achievements as ach_mod
from nomorals.games import skills as skills_mod
from nomorals.games import titles as titles_mod
from nomorals.games.titles import TitleStore, active_title
from nomorals.storage.db import Database


def _store():
    return TitleStore(Database(":memory:"))


def _boom(exc):
    def _raise(*a, **k):
        raise exc
    return _raise


_SKILL_TITLE = next(t for t in titles_mod.TITLE_CATALOG
                    if t.unlock.startswith("skill:"))
_SKILL_SLUG = _SKILL_TITLE.unlock.split(":", 1)[1]


class TestAchievementsGuard199(unittest.TestCase):
    def test_crash_withholds_achievement_titles_for_this_call(self):
        store = _store()
        with mock.patch.object(ach_mod, "get_achievements",
                               _boom(RuntimeError("ach dead"))), \
                mock.patch.object(titles_mod, "_log") as log:
            got = store.check_unlocks("p1", learned_skills=set())
        self.assertEqual(got, [])
        self.assertIn("novice", store.unlocked("p1"))
        self.assertNotIn("gladiator", store.unlocked("p1"))
        self.assertEqual(log.method_calls, [])

    def test_healthy_control_grants_achievement_title(self):
        store = _store()
        with mock.patch.object(ach_mod, "get_achievements",
                               lambda db, pk: [{"id": "arena_win"}]):
            got = store.check_unlocks("p1", learned_skills=set())
        self.assertIn("Gladiator", got)


class TestSkillsGuard206(unittest.TestCase):
    def test_crash_withholds_skill_titles_for_this_call(self):
        store = _store()
        with mock.patch.object(skills_mod.SkillStore, "learned",
                               _boom(RuntimeError("skills dead"))), \
                mock.patch.object(titles_mod, "_log") as log:
            got = store.check_unlocks("p2", unlocked_achievements=set())
        self.assertNotIn(_SKILL_TITLE.name, got)
        self.assertEqual(log.method_calls, [])

    def test_healthy_control_grants_skill_title(self):
        store = _store()
        with mock.patch.object(skills_mod.SkillStore, "learned",
                               lambda self, pk: {_SKILL_SLUG}):
            got = store.check_unlocks("p2", unlocked_achievements=set())
        self.assertIn(_SKILL_TITLE.name, got)


class _ExplodingDB:
    def query(self, *a, **k):
        raise RuntimeError("db dead")

    def execute(self, *a, **k):
        raise RuntimeError("db dead")


class _CtorBoomStore:
    def __init__(self, db):
        raise RuntimeError("ctor dead")


class TestActiveTitleWrapper272(unittest.TestCase):
    def test_exploding_db_yields_empty(self):
        self.assertEqual(active_title(_ExplodingDB(), "p1"), "")

    def test_store_construction_raise_yields_empty(self):
        with mock.patch.object(titles_mod, "TitleStore", _CtorBoomStore), \
                mock.patch.object(titles_mod, "_log") as log:
            self.assertEqual(active_title(object(), "p1"), "")
        self.assertEqual(log.method_calls, [])

    def test_healthy_control_returns_active_title_name(self):
        db = Database(":memory:")
        store = TitleStore(db)
        store.unlock("p3", "gladiator")
        self.assertTrue(store.set_active("p3", "gladiator"))
        self.assertEqual(active_title(db, "p3"), "Gladiator")


if __name__ == "__main__":
    unittest.main()

"""Player display-name authority across aliased Telegram endpoints.

Covers:
- The userbot (``telegram``) is the name authority: its sightings always
  refresh the stored display name, so a rename takes effect.
- A ``telegram-bot`` sighting never overwrites a userbot-set name, but
  may fill in a name when none exists yet (bot-only users can rename).
- Sightings from unrelated platforms always refresh.
- ``set_display_name`` is an explicit, always-applied rename.
"""
from __future__ import annotations

import unittest

from nomorals.games.players import PlayerStore
from nomorals.storage.db import Database


def make_store() -> PlayerStore:
    db = Database(":memory:")
    db.migrate()
    return PlayerStore(db)


class UserbotAuthorityTests(unittest.TestCase):
    def test_userbot_name_wins_over_bot_name(self):
        store = make_store()
        # First sighting via the BotFather bot: username-style name.
        prof = store.get("telegram:999", name="chfjdhx",
                         platform="telegram-bot")
        self.assertEqual(prof.name, "chfjdhx")
        # Same human via the userbot: display name takes over.
        prof = store.get("telegram:999", name="Mary", platform="telegram")
        self.assertEqual(prof.name, "Mary")
        self.assertEqual(prof.platform, "telegram")

    def test_bot_sighting_does_not_clobber_userbot_name(self):
        store = make_store()
        store.get("telegram:999", name="Mary", platform="telegram")
        prof = store.get("telegram:999", name="chfjdhx",
                         platform="telegram-bot")
        self.assertEqual(prof.name, "Mary")
        self.assertEqual(prof.platform, "telegram")

    def test_userbot_rename_propagates(self):
        store = make_store()
        store.get("telegram:999", name="Mary", platform="telegram")
        prof = store.get("telegram:999", name="Maria", platform="telegram")
        self.assertEqual(prof.name, "Maria")

    def test_bot_only_user_can_rename(self):
        store = make_store()
        store.get("telegram:999", name="chfjdhx", platform="telegram-bot")
        prof = store.get("telegram:999", name="chfjdhx2",
                         platform="telegram-bot")
        self.assertEqual(prof.name, "chfjdhx2")

    def test_unrelated_platform_always_refreshes(self):
        store = make_store()
        store.get("discord:999", name="Old", platform="discord")
        prof = store.get("discord:999", name="New", platform="discord")
        self.assertEqual(prof.name, "New")

    def test_no_platform_no_touch(self):
        store = make_store()
        store.get("telegram:999", name="Mary", platform="telegram")
        # Bare reads must not wipe or change the name.
        prof = store.get("telegram:999")
        self.assertEqual(prof.name, "Mary")
        prof = store.get("telegram:999", name="chfjdhx")
        self.assertEqual(prof.name, "Mary")

    def test_name_persists_across_reads(self):
        store = make_store()
        store.get("telegram:999", name="Mary", platform="telegram")
        store.get("telegram:999", name="Maria", platform="telegram")
        prof = store.get("telegram:999")
        self.assertEqual(prof.name, "Maria")


class SetDisplayNameTests(unittest.TestCase):
    def test_explicit_rename_applies(self):
        store = make_store()
        store.get("telegram:999", name="Mary", platform="telegram")
        prof = store.set_display_name("telegram:999", "Maria",
                                      platform="telegram")
        self.assertEqual(prof.name, "Maria")
        self.assertEqual(store.get("telegram:999").name, "Maria")

    def test_explicit_rename_creates_row(self):
        store = make_store()
        prof = store.set_display_name("telegram:999", "Mary",
                                      platform="telegram")
        self.assertEqual(prof.name, "Mary")

    def test_explicit_rename_noop_when_same(self):
        store = make_store()
        store.get("telegram:999", name="Mary", platform="telegram")
        prof = store.set_display_name("telegram:999", "Mary")
        self.assertEqual(prof.name, "Mary")

    def test_explicit_rename_empty_ignored(self):
        store = make_store()
        store.get("telegram:999", name="Mary", platform="telegram")
        prof = store.set_display_name("telegram:999", "")
        self.assertEqual(prof.name, "Mary")


if __name__ == "__main__":
    unittest.main()

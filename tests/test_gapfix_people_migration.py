"""Gapfix R3-56: the people migration actually runs.

Failing-first: on the unmodified tree there is no production caller
of ``migrate_from_relationships`` at all — no ``/people migrate``
verb and no boot sync — so every test here fails before the fix:
the verb falls through to the usage line, the helper does not exist,
and ``start()`` never mentions it.

The tests drive the real ``/people`` handler against a real
PartnerRuntime on a throwaway home (the test_reachability_ws2
pattern), with the legacy tracker seeded through the real
``RelationshipTracker`` (which persists its JSON on every note).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef
from nomorals.social.relationships import RelationshipTracker

OWNER_ID = "5478650254"


def _make_runtime():
    tmp = tempfile.TemporaryDirectory(prefix="nm-test-gapfix56-")
    settings = load_settings(overrides={
        "home": tmp.name,
        "partner.platforms": "telegram",
        "partner.owner_chats": OWNER_ID,
        "partner.owner_sender_ids": OWNER_ID,
    })
    context = build_context(settings, with_executor=False, with_tools=False)
    runtime = PartnerRuntime(context, gateway=None)
    return runtime, tmp


def _owner_msg(chat_key: str) -> ChatMessage:
    platform, _, chat_id = chat_key.partition(":")
    return ChatMessage(
        chat=ChatRef(platform=platform or "telegram",
                     chat_id=chat_id or "1", kind=ChatKind.DM),
        incoming=True, text="", sender="owner", sender_id=OWNER_ID,
        meta={"is_owner": True})


class PeopleMigrationTests(unittest.TestCase):
    def setUp(self):
        self.runtime, self.tmp = _make_runtime()
        self.home = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, text: str) -> str:
        out = self.runtime.handle_control(
            text, "telegram:1", message=_owner_msg("telegram:1"))
        return out.text if hasattr(out, "text") else out

    def _seed_tracker(self) -> RelationshipTracker:
        tracker = RelationshipTracker(
            path=str(self.home / "data" / "social_relationships.json"))
        tracker.note_inbound("ada", display_name="Ada", topic="interview")
        tracker.note_inbound("ada", display_name="Ada")
        tracker.note_outbound("ada", owner_initiated=True)
        tracker.add_note("ada", "Birthday in June")
        tracker.note_inbound("grace", display_name="Grace")
        tracker.add_note("grace", "Loves chess")
        return tracker

    def test_migrate_verb_moves_tracker_people_into_spine(self):
        self._seed_tracker()
        self.assertIn("no people recorded", self._run("/people list"))
        reply = self._run("/people migrate")
        self.assertIn("migrated 2 people", reply)
        listed = self._run("/people list")
        self.assertIn("Ada", listed)
        self.assertIn("Grace", listed)
        shown = self._run("/people show Ada")
        self.assertIn("Birthday in June", shown)
        self.assertIn("inbound 2", shown)
        self.assertIn("outbound 1", shown)

    def test_migrate_is_idempotent_via_verb(self):
        self._seed_tracker()
        self.assertIn("migrated 2 people", self._run("/people migrate"))
        second = self._run("/people migrate")
        self.assertIn("nothing new to migrate", second)
        shown = self._run("/people show Ada")
        # The note appears exactly once — re-migration must not dupe it.
        self.assertEqual(shown.count("Birthday in June"), 1)
        self.assertIn("inbound 2", shown)  # counters not doubled

    def test_migrate_leaves_spine_added_people_untouched(self):
        self._seed_tracker()
        self.assertIn("Alan", self._run("/people add Alan Turing"))
        self._run("/people note Alan Turing Store-side note")
        self.assertIn("migrated 2 people", self._run("/people migrate"))
        shown = self._run("/people show Alan")
        self.assertIn("Store-side note", shown)
        # A store-side note on a migrated person survives re-migration.
        self._run("/people note Ada Spine note")
        self._run("/people migrate")
        ada = self._run("/people show Ada")
        self.assertIn("Birthday in June", ada)
        self.assertIn("Spine note", ada)

    def test_migrate_with_no_tracker_file(self):
        # Fresh home: no social_relationships.json at all. Must not
        # crash — an empty tracker migrates nothing.
        reply = self._run("/people migrate")
        self.assertIn("nothing new to migrate", reply)
        self.assertIn("no people recorded", self._run("/people list"))

    def test_boot_sync_helper_runs_migration(self):
        self._seed_tracker()
        inserted = self.runtime._sync_people_from_tracker()  # noqa: SLF001
        self.assertEqual(inserted, 2)
        self.assertEqual(
            self.runtime._sync_people_from_tracker(), 0)  # noqa: SLF001
        self.assertIn("Ada", self._run("/people list"))

    def test_start_calls_people_sync(self):
        # The boot call site: start() must invoke the sync helper so
        # the history moves without the owner remembering a verb.
        # (The helper itself is unit-tested above; this pins the
        # wiring, not the behavior.)
        path = Path("nomorals/agents/partner/runtime.py")
        text = path.read_text(encoding="utf-8")
        start = text.index("    def start(self)")
        body = text[start:text.index("\n    def ", start + 10)]
        self.assertIn("_sync_people_from_tracker", body)


if __name__ == "__main__":
    unittest.main()

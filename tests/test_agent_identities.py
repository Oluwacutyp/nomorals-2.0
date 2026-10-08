"""Agent identities — Devon as a somebody for delegated tasks.

Covers IdentityStore CRUD and the ProactiveEngine task-identity wiring:
set/clear/acting_as, suggestion tagging, identity signatures.
Offline by design.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import unittest

from nomorals.agents.identities import AgentIdentity, IdentityStore
from nomorals.agents.proactive import ProactiveEngine, Suggestion


def temp_db() -> str:
    return tempfile.mktemp(prefix="nm-ident-", suffix=".db")


class IdentityStoreTests(unittest.TestCase):
    def setUp(self):
        self.db = temp_db()

    def test_create_and_get(self):
        store = IdentityStore(db_path=self.db)
        ident = store.create("Devon Research", email="devon.research@example.com",
                             handle="@devon_research", purpose="research tasks")
        self.assertIsNotNone(ident)
        self.assertTrue(ident.id.startswith("ident_"))
        fetched = store.get(ident.id)
        self.assertEqual(fetched.name, "Devon Research")
        self.assertEqual(fetched.email, "devon.research@example.com")
        self.assertEqual(fetched.handle, "@devon_research")

    def test_create_requires_name(self):
        store = IdentityStore(db_path=self.db)
        self.assertIsNone(store.create(""))
        self.assertIsNone(store.create("   "))

    def test_list_and_remove(self):
        store = IdentityStore(db_path=self.db)
        a = store.create("Devon Research")
        b = store.create("Devon Scheduler")
        names = [i.name for i in store.list()]
        self.assertIn("Devon Research", names)
        self.assertIn("Devon Scheduler", names)
        self.assertTrue(store.remove(a.id))
        self.assertIsNone(store.get(a.id))
        self.assertFalse(store.remove("nope"))

    def test_signature(self):
        ident = AgentIdentity(name="Devon Research", handle="@devon_research")
        self.assertEqual(ident.signature(), "Devon Research @devon_research")
        self.assertEqual(ident.label(), "Devon Research")

    def test_never_raises_on_garbage(self):
        store = IdentityStore(db_path=self.db)
        self.assertIsNone(store.get(None))
        self.assertIsNone(store.get(""))
        self.assertFalse(store.remove(None))
        self.assertFalse(store.remove(""))


class ProactiveIdentityTests(unittest.TestCase):
    def setUp(self):
        self.engine = ProactiveEngine()

    def test_default_is_devon_herself(self):
        self.assertIsNone(self.engine.acting_as)
        self.assertEqual(self.engine.identity_signature(), "Devon")

    def test_set_and_clear_identity(self):
        ident = AgentIdentity(id="ident_1", name="Devon Research",
                              handle="@devon_research")
        self.assertTrue(self.engine.set_identity(ident))
        self.assertIs(self.engine.acting_as, ident)
        self.assertEqual(self.engine.identity_signature(),
                         "Devon Research @devon_research")
        self.engine.clear_identity()
        self.assertIsNone(self.engine.acting_as)
        self.assertEqual(self.engine.identity_signature(), "Devon")

    def test_suggestions_tagged_with_identity(self):
        ident = AgentIdentity(id="ident_9", name="Devon Scheduler")
        self.engine.set_identity(ident)

        async def fake_gen(user_id, context):
            return [Suggestion(suggestion_id="s1", text="do the thing",
                               suggestion_type="pattern", action="x",
                               confidence=0.9, priority=1)]

        self.engine._suggestion_generators = [fake_gen]
        out = asyncio.run(self.engine.get_suggestions("u1"))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].metadata.get("identity_id"), "ident_9")
        self.assertEqual(out[0].metadata.get("identity_label"), "Devon Scheduler")

    def test_suggestions_untagged_without_identity(self):
        async def fake_gen(user_id, context):
            return [Suggestion(suggestion_id="s1", text="do the thing",
                               suggestion_type="pattern", action="x",
                               confidence=0.9, priority=1)]

        self.engine._suggestion_generators = [fake_gen]
        out = asyncio.run(self.engine.get_suggestions("u1"))
        self.assertNotIn("identity_id", out[0].metadata)

    def test_identity_store_roundtrip_with_engine(self):
        store = IdentityStore(db_path=temp_db())
        ident = store.create("Devon Research", handle="@devon_research")
        self.engine.set_identity(store.get(ident.id))
        self.assertEqual(self.engine.acting_as.name, "Devon Research")


if __name__ == "__main__":
    unittest.main()

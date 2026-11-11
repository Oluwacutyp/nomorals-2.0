"""/glossary — the docs 19+21 T2 glossary management surface, pinned.

The T2 glossary store shipped with no surface: entries could be
enforced by the translator but never seen or edited from chat.
This suite pins the owner surface that closes that remainder:

* the board renders every glossary in the store with per-scope
  counts (owner / domain / global — the store's own SCOPES), an
  empty scope named empty, and where the entries live;
* set writes THROUGH the shipped store (its own put_term), and a
  fresh store over the same backend sees the entry through the
  translator's own resolution read (terms_for) — with the store's
  precedence intact: an owner-scope row beats domain beats global;
* set-replace names old → new; del removes through remove_term,
  and an absent term (or a term at another scope) answers in
  words — never a fake success;
* the fingerprint consequence is real at the shipped seam: an
  edit changes GlossaryStore.fingerprint, and the reply says
  cached translations under the old fingerprint stop matching;
* /glossary is owner-only at the gating layer (CONTROL_COMMANDS,
  never PUBLIC_COMMANDS — the public count does not move) and at
  the full-runtime choke point;
* no legitimacy language anywhere in the surface copy.

Every store path in this suite is a per-test tmp path — never a
fixed shared root (the /craft pass's ship-blocking lesson).
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.partner.runtime_system import RuntimeSystemMixin
from nomorals.llm.glossary import GlossaryStore
from nomorals.social.chat.control import (
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    PUBLIC_COMMANDS,
    caller_may_run,
    detailed_help,
    parse_control,
)
from nomorals.storage.db import Database


def _tmp(testcase: unittest.TestCase) -> Path:
    tmp = Path(tempfile.mkdtemp(dir=os.environ.get("NM_TEST_TMPDIR")))
    testcase.addCleanup(
        lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
    return tmp


def _file_store(testcase: unittest.TestCase) -> GlossaryStore:
    """A store over a per-test home (the JSON fallback backend)."""
    return GlossaryStore(None, str(_tmp(testcase)))


def _db_store(testcase: unittest.TestCase) -> GlossaryStore:
    """A store over a per-test migrated database backend."""
    db = Database(str(_tmp(testcase) / "test.db"))
    db.migrate()
    return GlossaryStore(db, "")


class GlossaryRuntime(RuntimeSystemMixin):
    """The system mixin alone, with a glossary store injected the
    way the handler reads it."""

    def __init__(self, store):
        self.context = SimpleNamespace(db=None, settings=None)
        self._glossary_store_instance = store


def _seeded(testcase: unittest.TestCase, store: GlossaryStore,
            ) -> GlossaryStore:
    store.put_term("legal", "bank", "Banco Central", scope="owner")
    store.put_term("legal", "court", "tribunal", scope="domain")
    store.put_term("legal", "filing", "expediente", scope="global")
    store.put_term("legal", "affidavit", "declaración jurada",
                   source_lang="en", target_lang="es", scope="owner")
    store.put_term("pets", "treat", "premio", scope="global")
    return store


# ── the board ────────────────────────────────────────────────────


class BoardTests(unittest.TestCase):

    def test_board_renders_scopes_counts_and_location(self):
        store = _seeded(self, _file_store(self))
        out = GlossaryRuntime(store)._control_glossary("")
        self.assertIn("Translation glossaries", out)
        self.assertIn("legal — 4 entries", out)
        self.assertIn("owner: 2", out)
        self.assertIn("domain: 1", out)
        self.assertIn("global: 1", out)
        self.assertIn("pets — 1 entry", out)
        # empty scopes named as empty, never a quiet zero
        self.assertIn("owner: 0 (empty)", out)
        self.assertIn("domain: 0 (empty)", out)
        # where the entries live — the file backend's own words
        self.assertIn("glossary.json", out)
        self.assertIn("2 glossaries · 5 entries in all", out)

    def test_board_over_the_database_backend(self):
        store = _seeded(self, _db_store(self))
        out = GlossaryRuntime(store)._control_glossary("board")
        self.assertIn("translate_glossary table", out)
        self.assertIn("legal — 4 entries", out)

    def test_board_on_an_empty_store_says_so(self):
        out = GlossaryRuntime(_file_store(self))._control_glossary("")
        self.assertIn("no glossaries yet", out)
        self.assertIn("the store is empty", out)


# ── list ─────────────────────────────────────────────────────────


class ListTests(unittest.TestCase):

    def test_list_groups_by_glossary_and_scope(self):
        store = _seeded(self, _file_store(self))
        out = GlossaryRuntime(store)._control_glossary("list")
        self.assertIn("legal · owner scope", out)
        self.assertIn("bank → Banco Central", out)
        self.assertIn("court → tribunal", out)
        self.assertIn("filing → expediente", out)
        self.assertIn("pets · global scope", out)
        self.assertIn("treat → premio", out)
        # pair-specific rows carry their pair
        self.assertIn("affidavit → declaración jurada  (en→es)", out)

    def test_list_one_scope_across_glossaries(self):
        store = _seeded(self, _file_store(self))
        out = GlossaryRuntime(store)._control_glossary("list global")
        self.assertIn("filing → expediente", out)
        self.assertIn("treat → premio", out)
        self.assertNotIn("bank → Banco Central", out)

    def test_list_one_glossary(self):
        store = _seeded(self, _file_store(self))
        out = GlossaryRuntime(store)._control_glossary("list pets")
        self.assertIn("treat → premio", out)
        self.assertNotIn("bank", out)

    def test_list_unknown_glossary_answers_in_words(self):
        store = _seeded(self, _file_store(self))
        out = GlossaryRuntime(store)._control_glossary("list ghost")
        self.assertIn("no glossary named 'ghost'", out)
        self.assertIn("legal", out)

    def test_list_empty_scope_answers_in_words(self):
        store = _seeded(self, _file_store(self))
        out = GlossaryRuntime(
            store)._control_glossary("list pets owner")
        self.assertIn("no entries at owner scope", out)
        self.assertIn("empty", out)


# ── set: through the store, visible to the translator's read ─────


class SetTests(unittest.TestCase):

    def test_set_lands_and_terms_for_sees_it(self):
        home = str(_tmp(self))
        rt = GlossaryRuntime(GlossaryStore(None, home))
        out = rt._control_glossary(
            "set legal owner bank = Banco Central")
        self.assertIn("set in the legal glossary (owner scope)", out)
        self.assertIn("bank → Banco Central", out)
        # a FRESH store over the same backend — the translator's
        # own resolution read — returns the entry
        fresh = GlossaryStore(None, home)
        terms = fresh.terms_for("legal", "en", "es")
        self.assertEqual(
            [(t.source_term, t.target_term, t.scope) for t in terms],
            [("bank", "Banco Central", "owner")])

    def test_set_owner_scope_beats_global_in_resolution(self):
        store = _file_store(self)
        store.put_term("legal", "bank", "the global reading",
                       scope="global")
        rt = GlossaryRuntime(store)
        rt._control_glossary("set legal owner bank = Banco Central")
        resolved = store.terms_for("legal", "en", "es")
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].target_term, "Banco Central")
        self.assertEqual(resolved[0].scope, "owner")

    def test_set_replace_reports_old_to_new(self):
        store = _file_store(self)
        store.put_term("legal", "bank", "Old Bank", scope="global")
        out = GlossaryRuntime(store)._control_glossary(
            "set legal owner bank = New Bank")
        self.assertIn("replaced in the legal glossary", out)
        self.assertIn("bank → New Bank", out)
        self.assertIn("was Old Bank, at global scope", out)

    def test_set_changes_the_fingerprint(self):
        store = _file_store(self)
        before = store.fingerprint("legal", "", "")
        out = GlossaryRuntime(store)._control_glossary(
            "set legal global bank = Banco")
        self.assertNotEqual(store.fingerprint("legal", "", ""),
                            before)
        # and the reply states the cache consequence in words
        self.assertIn("fingerprint changed", out)
        self.assertIn("stop matching", out)

    def test_set_with_a_bad_scope_writes_nothing(self):
        store = _file_store(self)
        out = GlossaryRuntime(store)._control_glossary(
            "set legal personal bank = Banco")
        self.assertIn("is not a glossary scope", out)
        self.assertIn("Nothing was written", out)
        self.assertEqual(store.entries("legal"), [])

    def test_set_without_an_equals_is_usage(self):
        out = GlossaryRuntime(_file_store(self))._control_glossary(
            "set legal owner bank")
        self.assertIn("usage: /glossary set", out)

    def test_set_multiword_term_and_translation(self):
        store = _file_store(self)
        rt = GlossaryRuntime(store)
        rt._control_glossary(
            "set legal domain power of attorney = poder notarial")
        terms = store.terms_for("legal", "en", "es")
        self.assertEqual(terms[0].source_term, "power of attorney")
        self.assertEqual(terms[0].target_term, "poder notarial")


# ── del ──────────────────────────────────────────────────────────


class DelTests(unittest.TestCase):

    def test_del_removes_and_the_lookup_then_misses(self):
        store = _file_store(self)
        store.put_term("legal", "bank", "Banco Central",
                       scope="owner")
        out = GlossaryRuntime(store)._control_glossary(
            "del legal owner bank")
        self.assertIn("removed from the legal glossary", out)
        self.assertIn("bank → Banco Central is gone", out)
        self.assertEqual(store.terms_for("legal", "en", "es"), [])
        self.assertIn("fingerprint changed", out)

    def test_del_absent_term_answers_in_words(self):
        store = _seeded(self, _file_store(self))
        out = GlossaryRuntime(store)._control_glossary(
            "del legal owner ghost")
        self.assertIn("there is no 'ghost' in the legal glossary",
                      out)
        self.assertIn("nothing was removed", out)

    def test_del_at_the_wrong_scope_removes_nothing(self):
        store = _file_store(self)
        store.put_term("legal", "bank", "Banco Central",
                       scope="owner")
        out = GlossaryRuntime(store)._control_glossary(
            "del legal global bank")
        self.assertIn("at owner scope, not global", out)
        self.assertIn("nothing was removed", out)
        self.assertEqual(len(store.entries("legal")), 1)

    def test_del_without_a_scope(self):
        store = _file_store(self)
        store.put_term("pets", "treat", "premio", scope="global")
        out = GlossaryRuntime(store)._control_glossary(
            "del pets treat")
        self.assertIn("removed from the pets glossary", out)
        self.assertEqual(store.entries("pets"), [])


# ── copy discipline ──────────────────────────────────────────────


class CopyTests(unittest.TestCase):

    def test_no_legitimacy_language_anywhere(self):
        store = _seeded(self, _file_store(self))
        rt = GlossaryRuntime(store)
        outputs = [
            rt._control_glossary(""),
            rt._control_glossary("list"),
            rt._control_glossary("set legal owner fine = multa"),
            rt._control_glossary("del legal owner fine"),
            rt._control_glossary("del legal owner ghost"),
            COMMAND_DETAILS["glossary"]["what"],
            COMMAND_DETAILS["glossary"]["usage"],
        ]
        for text in outputs:
            self.assertNotIn("legitim", text.lower(), text)
            self.assertNotIn("not allowed", text.lower(), text)
            self.assertNotIn("policy", text.lower(), text)

    def test_unknown_verb_is_usage(self):
        out = GlossaryRuntime(_file_store(self))._control_glossary(
            "frobnicate")
        self.assertIn("unknown /glossary verb 'frobnicate'", out)
        self.assertIn("usage: /glossary", out)


# ── gating: owner-only, end to end ───────────────────────────────


class GatingTests(unittest.TestCase):

    def test_gating_layer(self):
        self.assertIn("glossary", CONTROL_COMMANDS)
        # owner-only: never on the public list. (The public catalog's
        # exact size is pinned once, in test_craft_surface — this
        # suite asserts only the property it owns.)
        self.assertNotIn("glossary", PUBLIC_COMMANDS)
        self.assertFalse(caller_may_run("glossary", False))
        self.assertTrue(caller_may_run("glossary", True))
        cmd = parse_control("/glossary list")
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.kind, "glossary")

    def test_public_help_serves_no_glossary_page(self):
        page = detailed_help("glossary", public_only=True)
        self.assertNotIn("the translation glossary board", page)
        owner_page = detailed_help("glossary")
        self.assertIn("the translation glossary board", owner_page)


class RuntimeDispatchTests(unittest.TestCase):
    """The full PartnerRuntime path: /glossary dispatches for the
    owner and is refused for anyone else at the grant choke
    point."""

    @classmethod
    def setUpClass(cls):
        from nomorals.agents.context import build_context
        from nomorals.agents.partner_runtime import PartnerRuntime
        from nomorals.core.config import load_settings
        from nomorals.social.chat.base import ChatAdapter, SendResult
        from nomorals.social.chat.gateway import ChatGateway

        class FakeAdapter(ChatAdapter):
            def __init__(self, name):
                super().__init__(media_dir="/tmp/nm-test-media")
                self.name = name

            def run(self, handler):
                pass

            def send(self, chat, text, *, reply_to="", buttons=None,
                     **kwargs):
                return SendResult(ok=True, platform=self.name,
                                  message_id="m1")

        tmp = tempfile.TemporaryDirectory(
            dir=os.environ.get("NM_TEST_TMPDIR"),
            prefix="nm-test-glossary-surface-")
        cls._tmp = tmp
        settings = load_settings(overrides={
            "home": tmp.name,
            "partner.platforms": "telegram",
            "partner.owner_chats": "5478650254",
            "partner.owner_sender_ids": "5478650254",
        })
        context = build_context(settings, with_executor=False,
                                with_tools=False)
        gateway = ChatGateway({"telegram": FakeAdapter("telegram")},
                              db=context.db)
        cls.runtime = PartnerRuntime(context, gateway=gateway)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_owner_dispatch_renders_the_board(self):
        reply = self.runtime.handle_control(
            "/glossary", "telegram:1", message=None)
        self.assertIn("Translation glossaries", reply)
        self.assertNotIn("unknown command", reply)

    def test_non_owner_is_refused_at_the_grant(self):
        with mock.patch.object(self.runtime, "_is_operator",
                               return_value=False):
            reply = self.runtime.handle_control(
                "/glossary", "telegram:999",
                message=SimpleNamespace())
        self.assertNotIn("Translation glossaries", reply)
        self.assertIn("owner", reply)


if __name__ == "__main__":
    unittest.main()

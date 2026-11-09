"""Cockpit: /skills + /delegate catalog registration and handlers,
riding the Living Chat surfaces (no second control surface)."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.social.chat import control


class CatalogTests(unittest.TestCase):
    def test_commands_registered_everywhere_the_catalog_requires(self):
        for kind in ("skills", "delegate"):
            self.assertIn(kind, control.CONTROL_COMMANDS)
            self.assertIn(kind, control.COMMAND_DETAILS)
            self.assertIn(kind, control.LIST_ONELINERS)
            grouped = {name for _title, names in control.LIST_GROUPS
                       for name in names}
            self.assertIn(kind, grouped)

    def test_commands_are_owner_only(self):
        for kind in ("skills", "delegate"):
            self.assertNotIn(kind, control.PUBLIC_COMMANDS)
            self.assertFalse(control.caller_may_run(kind, False))
            self.assertTrue(control.caller_may_run(kind, True))


class _FakeHost:
    def __init__(self):
        self.store = SimpleNamespace(get=lambda card_id: None)
        self.published = []
        self.revised = []

    def publish(self, card, chat):
        self.published.append(card)
        return card

    def revise(self, card, chat, **changes):
        self.revised.append((card, changes))
        return card


def _handler(test: unittest.TestCase, db=None):
    from nomorals.agents.partner.runtime_expression import (
        RuntimeExpressionMixin,
    )

    handler = RuntimeExpressionMixin()
    handler.context = SimpleNamespace(db=db)
    tmp = tempfile.mkdtemp(dir=os.environ.get("NM_TEST_TMPDIR") or None)
    test.addCleanup(lambda: shutil.rmtree(tmp, True))
    env = mock.patch.dict(os.environ, {"NM_HOME": tmp})
    env.start()
    test.addCleanup(env.stop)
    return handler


class SkillsHandlerTests(unittest.TestCase):
    def test_list_show_search(self):
        handler = _handler(self)
        listing = handler._control_skillbook("", None)
        self.assertIn("mission-execution", listing)
        self.assertIn("accountable-delegation", listing)
        shown = handler._control_skillbook("show mission-execution", None)
        self.assertIn("Mission execution", shown)
        routed = handler._control_skillbook(
            "search delegate work and verify claims", None)
        self.assertIn("accountable-delegation", routed)
        missing = handler._control_skillbook("show nope-nothing", None)
        self.assertIn("no skill called", missing)

    def test_list_publishes_a_living_card(self):
        handler = _handler(self)
        host = _FakeHost()
        handler._living_card_host = lambda: host
        chat = SimpleNamespace(key="telegram:1", platform="telegram",
                               chat_id="1", kind=None)
        message = SimpleNamespace(chat=chat)
        out = handler._control_skillbook("", message)
        self.assertIn("card is live", out)
        self.assertEqual(len(host.published), 1)
        card = host.published[0]
        self.assertEqual(card.family, "skills")
        self.assertTrue(card.actions)
        self.assertTrue(all(a.data.startswith("skills show ")
                            for a in card.actions))


class DelegateHandlerTests(unittest.TestCase):
    def test_empty_and_populated_states(self):
        from nomorals.agents.delegation import DelegationStore
        from nomorals.storage.db import Database

        db = Database(":memory:")
        db.migrate()
        handler = _handler(self, db=db)
        empty = handler._control_delegate("", None)
        self.assertIn("no work contracts yet", empty)
        store = DelegationStore(db)
        contract = store.create_contract(
            goal="audit the auth layer", worker="security:critic",
            verifier="verifier-amy")
        claim = store.submit_evidence(
            contract.id, kind="audit result", claim="no criticals",
            producer="security:critic")
        listing = handler._control_delegate("", None)
        self.assertIn(contract.id, listing)
        shown = handler._control_delegate(f"show {contract.id}", None)
        self.assertIn("claim ledger:", shown)
        self.assertIn("no criticals", shown)
        ledger = handler._control_delegate("claims", None)
        self.assertIn("no criticals", ledger)
        self.assertEqual(claim.contract_id, contract.id)

    def test_delegate_without_db_answers_honestly(self):
        handler = _handler(self, db=None)
        out = handler._control_delegate("", None)
        self.assertIn("unavailable", out)


if __name__ == "__main__":
    unittest.main()

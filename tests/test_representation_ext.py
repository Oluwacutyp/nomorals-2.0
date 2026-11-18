"""Representation-ledger extension (lane 63): authority, audit log,
/representations verb, representation_lookup tool, group-announcement
hook.  Failing-first: every test here fails on the f073d80c base.
"""

from __future__ import annotations

import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.cognition.representation import (
    RepresentationLedger,
    _ledger_for,
)

OLD_DDL = """
CREATE TABLE cog_representation (
    id TEXT PRIMARY KEY,
    action_type TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    outcome TEXT NOT NULL DEFAULT 'unknown',
    owner_feedback INTEGER,
    feedback_text TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""


def _seed(ledger, **kw):
    return ledger.log_action(
        kw.pop("action_type", "message"),
        kw.pop("description", "test action"),
        outcome=kw.pop("outcome", "success"),
        metadata=kw.pop("metadata", None),
        authority=kw.pop("authority", None),
    )


class AuthorityTests(unittest.TestCase):
    def test_authority_round_trip(self):
        ledger = RepresentationLedger(":memory:")
        rec = _seed(ledger, authority="owner-confirmed",
                    metadata={"counterparty": "Chidi"})
        self.assertEqual(rec.authority, "owner-confirmed")
        again = ledger.get(rec.id)
        assert again is not None
        self.assertEqual(again.authority, "owner-confirmed")

    def test_authority_defaults_null(self):
        ledger = RepresentationLedger(":memory:")
        rec = _seed(ledger)
        self.assertIsNone(rec.authority)
        again = ledger.get(rec.id)
        assert again is not None
        self.assertIsNone(again.authority)

    def test_authority_column_added_to_old_db(self):
        # A representation.db written before this change has no
        # authority column: opening it must add the column, keep the
        # old rows readable (authority None), and accept new rows.
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "representation.db")
            conn = sqlite3.connect(path)
            conn.executescript(OLD_DDL)
            conn.execute(
                "INSERT INTO cog_representation (id, action_type,"
                " description, outcome, created_at, updated_at)"
                " VALUES ('rep_old', 'post', 'old row', 'success', ?, ?)",
                (time.time(), time.time()))
            conn.commit()
            conn.close()
            ledger = RepresentationLedger(path)
            old = ledger.get("rep_old")
            assert old is not None
            self.assertIsNone(old.authority)
            new = _seed(ledger, authority="owner-direct")
            self.assertEqual(ledger.get(new.id).authority, "owner-direct")

    def test_extended_ledger_passes_authority(self):
        from nomorals.cognition.ledger_ext import (
            ExtendedRepresentationLedger,
        )

        ext = ExtendedRepresentationLedger(RepresentationLedger(":memory:"))
        rec = ext.record_autonomous_action(
            "send", counterparty="Chidi", outcome="success",
            description="mailed Chidi", authority="owner-confirmed")
        self.assertEqual(rec.authority, "owner-confirmed")
        self.assertEqual(ext.ledger.get(rec.id).authority,
                         "owner-confirmed")


class AuditTests(unittest.TestCase):
    def test_create_is_audited(self):
        ledger = RepresentationLedger(":memory:")
        rec = _seed(ledger, authority="owner-confirmed")
        trail = ledger.audit_trail(action_id=rec.id)
        self.assertEqual(len(trail), 1)
        self.assertEqual(trail[0]["event"], "create")
        self.assertEqual(trail[0]["action_id"], rec.id)

    def test_outcome_and_counterfactual_are_audited(self):
        ledger = RepresentationLedger(":memory:")
        rec = _seed(ledger)
        ledger.record_outcome(rec.id, "success", owner_feedback="up")
        ledger.record_counterfactual(rec.id, "ask first", "owner prefers it")
        events = [row["event"]
                  for row in ledger.audit_trail(action_id=rec.id)]
        self.assertIn("outcome", events)
        self.assertIn("counterfactual", events)

    def test_audit_trail_newest_first_and_filtered(self):
        ledger = RepresentationLedger(":memory:")
        first = _seed(ledger, description="first")
        second = _seed(ledger, description="second")
        trail = ledger.audit_trail()
        self.assertEqual(trail[0]["action_id"], second.id)
        self.assertEqual(trail[1]["action_id"], first.id)
        self.assertEqual(
            [row["action_id"] for row in ledger.audit_trail(
                action_id=first.id)],
            [first.id])

    def test_audit_never_raises_and_empty_trail(self):
        ledger = RepresentationLedger(":memory:")
        self.assertEqual(ledger.audit_trail(), [])
        ledger.audit("rep_nope", "owner", "read", "nothing there")
        self.assertEqual(len(ledger.audit_trail(action_id="rep_nope")), 1)

    def test_export_import_carry_authority_and_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = RepresentationLedger(":memory:")
            rec = _seed(src, authority="owner-confirmed")
            src.record_outcome(rec.id, "success", owner_feedback="up")
            out = Path(tmp) / "export.json"
            src.export_json(out)
            dst = RepresentationLedger(":memory:")
            counts = dst.import_json(out)
            self.assertEqual(counts["actions"], 1)
            self.assertEqual(dst.get(rec.id).authority, "owner-confirmed")
            events = [row["event"] for row in dst.audit_trail(
                action_id=rec.id)]
            self.assertIn("create", events)
            self.assertIn("outcome", events)


class LookupTests(unittest.TestCase):
    def _seeded(self):
        ledger = RepresentationLedger(":memory:")
        _seed(ledger, description="emailed Chidi the deck",
              metadata={"counterparty": "Chidi"}, authority="owner-confirmed")
        _seed(ledger, description="texted Ada",
              metadata={"counterparty": "Ada"}, authority="owner-direct")
        _seed(ledger, action_type="purchase", description="bought the book",
              metadata={"counterparty": "Chidi"})
        return ledger

    def test_lookup_by_counterparty_substring_case_insensitive(self):
        ledger = self._seeded()
        found = ledger.lookup(counterparty="chidi")
        self.assertEqual(len(found), 2)
        self.assertTrue(all(
            (r.metadata or {}).get("counterparty") == "Chidi"
            for r in found))

    def test_lookup_by_action_type(self):
        ledger = self._seeded()
        found = ledger.lookup(counterparty="Chidi", action_type="purchase")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].action_type, "purchase")

    def test_lookup_since_days(self):
        ledger = RepresentationLedger(":memory:")
        ledger.log_action(
            "message", "old one", outcome="success",
            metadata={"counterparty": "Chidi"},
            created_at=time.time() - 40 * 86400.0)
        _seed(ledger, description="new one",
              metadata={"counterparty": "Chidi"})
        found = ledger.lookup(counterparty="Chidi", since_days=30.0)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].description, "new one")

    def test_lookup_no_match_is_empty(self):
        ledger = self._seeded()
        self.assertEqual(ledger.lookup(counterparty="Nobody"), [])


class ToolTests(unittest.TestCase):
    def _registry(self, tmp):
        from nomorals.tools.registry import ToolRegistry

        settings = SimpleNamespace(home=tmp, home_path=tmp)
        ctx = SimpleNamespace(settings=settings)
        reg = ToolRegistry(ctx)
        reg.register_builtins()
        return reg

    def test_tool_registered(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._registry(tmp)
            self.assertIn("representation_lookup", reg._tools)

    def test_tool_lookup_answers_what_did_you_tell_x(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = _ledger_for(SimpleNamespace(home_path=tmp))
            _seed(ledger, description="emailed Chidi the deck",
                  metadata={"counterparty": "Chidi"},
                  authority="owner-confirmed")
            reg = self._registry(tmp)
            fn = reg._tools["representation_lookup"].fn
            out = fn(action="lookup", counterparty="Chidi")
            self.assertTrue(out.get("ok"))
            self.assertEqual(len(out.get("records", [])), 1)
            self.assertIn("Chidi", out.get("text", ""))
            # The tool read audits itself as an owner read.
            trail = ledger.audit_trail()
            self.assertTrue(any(
                row["event"] == "read" and row["actor"] == "owner"
                for row in trail))

    def test_tool_unknown_counterparty_is_honest(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._registry(tmp)
            fn = reg._tools["representation_lookup"].fn
            out = fn(action="lookup", counterparty="Nobody")
            self.assertTrue(out.get("ok"))
            self.assertEqual(out.get("records", []), [])
            self.assertIn("nothing on record", out.get("text", "").lower())

    def test_tool_show_and_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = _ledger_for(SimpleNamespace(home_path=tmp))
            rec = _seed(ledger, description="posted the update",
                        authority="owner-designated")
            reg = self._registry(tmp)
            fn = reg._tools["representation_lookup"].fn
            shown = fn(action="show", record_id=rec.id)
            self.assertTrue(shown.get("ok"))
            self.assertIn("owner-designated", shown.get("text", ""))
            audited = fn(action="audit", record_id=rec.id)
            self.assertTrue(audited.get("ok"))
            self.assertTrue(any(
                row["event"] == "create"
                for row in audited.get("trail", [])))


class ParserTests(unittest.TestCase):
    def test_what_did_you_tell_x_phrasings(self):
        from nomorals.tools.representation import (
            parse_representation_lookup,
        )

        for text in (
            "what did you tell Chidi for me?",
            "what did you tell Chidi for me",
            "what did you say to Ada?",
            "what have you told Chidi",
            "what did you tell the group for me?",
        ):
            with self.subTest(text=text):
                name = parse_representation_lookup(text)
                self.assertIsNotNone(name, text)
        self.assertEqual(
            parse_representation_lookup("what did you tell Chidi for me?"),
            "Chidi")
        self.assertEqual(
            parse_representation_lookup("what did you say to Ada?"),
            "Ada")

    def test_parser_ignores_other_questions(self):
        from nomorals.tools.representation import (
            parse_representation_lookup,
        )

        for text in (
            "tell Chidi I said hi",
            "what did you do today?",
            "how did i do",
            "what is the weather?",
        ):
            self.assertIsNone(parse_representation_lookup(text), text)


def _make_runtime(tmp):
    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import load_settings
    from nomorals.social.chat.base import ChatAdapter, SendResult
    from nomorals.social.chat.gateway import ChatGateway

    class FakeAdapter(ChatAdapter):
        def __init__(self, name):
            super().__init__(media_dir="/tmp/nm-test-media")
            self.name = name
            self.sent = []

        def run(self, handler):
            pass

        def send(self, chat, text, *, reply_to="", buttons=None, **kwargs):
            self.sent.append(text)
            return SendResult(ok=True, platform=self.name,
                              message_id=f"m{len(self.sent)}")

    settings = load_settings(overrides={
        "home": tmp,
        "partner.platforms": "telegram",
        "partner.owner_chats": "5478650254",
        "partner.owner_sender_ids": "5478650254",
    })
    context = build_context(settings, with_executor=False, with_tools=False)
    adapter = FakeAdapter("telegram")
    gateway = ChatGateway({"telegram": adapter}, db=context.db)
    runtime = PartnerRuntime(context, gateway=gateway)
    return runtime, adapter, settings


class VerbTests(unittest.TestCase):
    def test_verb_parses(self):
        from nomorals.social.chat.control import parse_control

        cmd = parse_control("/representations")
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.kind, "representations")

    def test_verb_list_shows_records_with_authority(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime, _adapter, settings = _make_runtime(tmp)
            ledger = _ledger_for(settings)
            _seed(ledger, description="emailed Chidi the deck",
                  metadata={"counterparty": "Chidi"},
                  authority="owner-confirmed")
            out = runtime._control_representations("", chat_key="t")
            self.assertIn("Chidi", out)
            self.assertIn("owner-confirmed", out)

    def test_verb_show_renders_full_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime, _adapter, settings = _make_runtime(tmp)
            ledger = _ledger_for(settings)
            rec = _seed(ledger, description="posted the update",
                        authority="owner-designated")
            out = runtime._control_representations(f"show {rec.id}",
                                                   chat_key="t")
            self.assertIn(rec.id, out)
            self.assertIn("owner-designated", out)
            self.assertIn("audit", out.lower())

    def test_verb_audit_lists_trail(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime, _adapter, settings = _make_runtime(tmp)
            ledger = _ledger_for(settings)
            rec = _seed(ledger, description="posted the update")
            out = runtime._control_representations("audit", chat_key="t")
            self.assertIn("create", out)
            self.assertIn(rec.id, out)

    def test_verb_show_unknown_id_is_honest(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime, _adapter, settings = _make_runtime(tmp)
            out = runtime._control_representations("show rep_nope",
                                                   chat_key="t")
            self.assertIn("no record", out.lower())

    def test_verb_owner_only(self):
        from nomorals.social.chat.base import ChatKind, ChatRef

        with tempfile.TemporaryDirectory() as tmp:
            runtime, _adapter, settings = _make_runtime(tmp)
            stranger = SimpleNamespace(
                sender_id="999",
                chat=ChatRef(platform="telegram", chat_id="999",
                             kind=ChatKind.DM),
                meta={})
            out = runtime.handle_control("/representations", "t",
                                         message=stranger)
            self.assertEqual(out, "that one's just for the owner.")

    def test_verb_list_read_is_audited(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime, _adapter, settings = _make_runtime(tmp)
            ledger = _ledger_for(settings)
            _seed(ledger, description="posted the update")
            runtime._control_representations("", chat_key="t")
            trail = ledger.audit_trail()
            self.assertTrue(any(
                row["event"] == "read" and row["actor"] == "owner"
                for row in trail))


class GroupHookTests(unittest.TestCase):
    def test_group_send_logs_post_with_designation(self):
        from nomorals.social.chat.base import ChatKind, ChatRef

        with tempfile.TemporaryDirectory() as tmp:
            runtime, adapter, settings = _make_runtime(tmp)
            chat = ChatRef(platform="telegram", chat_id="test-group-1",
                           kind=ChatKind.GROUP)
            runtime._send(chat, "hello room, weekly update")
            self.assertEqual(adapter.sent, ["hello room, weekly update"])
            ledger = _ledger_for(settings)
            recs = ledger.recent(action_type="post")
            self.assertEqual(len(recs), 1)
            self.assertIn("weekly update", recs[0].description)
            self.assertEqual(recs[0].authority, "owner-designated")
            self.assertEqual(
                (recs[0].metadata or {}).get("group_key"),
                "telegram:test-group-1")


if __name__ == "__main__":
    unittest.main()

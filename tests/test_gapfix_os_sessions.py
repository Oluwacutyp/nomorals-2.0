"""Gapfix R2-37: OS session adapters wired into the SessionBridge.

Failing-first: the bridge exposes no adapter enumeration yet, so every
test that asks for an adapter-projected row must fail before the wiring
lands. Contract under test (from the survey):

* ``list_unified()`` returns the bridge's own active rows PLUS the
  adapter-projected views (account / voice / generic), each carrying
  its adapter kind in ``state["kind"]`` — never silently dropped and
  never re-labelled as a bridge session.
* With no sources registered, ``list_unified()`` is exactly
  ``list_active()`` (no regression on the bridge's own rows).
* A failing adapter/source degrades honestly: the broken concept is
  named as unavailable (``degraded: True`` + the error), enumeration
  continues, and nothing fake is shown.
* One row per real session: two sources claiming the same concept
  yield one row (dedup by conversation id).
* Account projection is read-only: enumeration calls ``peek_session``
  (no touch), never ``get_session``.
* Projected ids are deterministic per (kind, conversation id), so a
  row shown by ``list`` stays addressable by ``show``.
"""

from __future__ import annotations

import argparse
import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from nomorals.os.session_bridge import SessionBridge

# ── fakes ────────────────────────────────────────────────────────────────

class _FakeAccount:
    def __init__(self, service="github", username="u", *, valid=True):
        self.service = service
        self.username = username
        self.scope = "s"
        self.oauth_token = None
        self.metadata = {"a": 1}
        self.created_at = "c"
        self.last_used = "l"
        self._valid = valid

    def is_valid(self):
        return self._valid


class _FakeManager:
    """Duck-typed accounts SessionManager with call recording."""

    def __init__(self, accounts, *, fail_peek_for=()):
        self._accounts = dict(accounts)  # (service, username) -> account
        self._fail_peek_for = set(fail_peek_for)
        self.peek_calls: list[tuple[str, str]] = []
        self.get_calls: list[tuple[str, str]] = []

    def list_pairs(self):
        return sorted(self._accounts)

    def peek_session(self, service, username):
        self.peek_calls.append((service, username))
        if (service, username) in self._fail_peek_for:
            raise RuntimeError("vault read boom")
        return self._accounts.get((service, username))

    def get_session(self, service, username):
        self.get_calls.append((service, username))
        acct = self._accounts.get((service, username))
        if acct is None:
            acct = _FakeAccount(service, username)
            self._accounts[(service, username)] = acct
        return acct


class _FakeVoice:
    def __init__(self, session_id="voice-1", state="listening"):
        self.session_id = session_id
        self.state = state
        self.device_id = "mic-1"
        self.profile = "default"
        self.consent = object()
        self.stats = {"turns": 3, "barge_ins": 0,
                      "end_reason": None, "error": None}


def _bridge_with_chat():
    bridge = SessionBridge(db=":memory:")
    bridge.session_for_cli()
    return bridge


class UnifiedEnumerationTests(unittest.TestCase):
    def test_unified_includes_account_and_voice(self):
        manager = _FakeManager({("github", "u"): _FakeAccount()})
        bridge = _bridge_with_chat()
        bridge.register_account_manager(manager)
        bridge.register_adapter_source("voice", lambda: [_FakeVoice()])
        rows = bridge.list_unified()
        by_kind = {}
        for row in rows:
            by_kind.setdefault(row.state.get("kind", "<bridge>"), []).append(row)
        # the bridge's own CLI row is still there, unlabelled by any adapter
        self.assertEqual(len(by_kind["<bridge>"]), 1)
        self.assertEqual(by_kind["<bridge>"][0].frontend, "cli")
        # the account row came through the adapter, kind intact
        (acct,) = by_kind["account"]
        self.assertEqual(acct.frontend, "api")
        self.assertEqual(acct.state["service"], "github")
        self.assertEqual(acct.state["username"], "u")
        self.assertTrue(acct.state["valid"])
        self.assertEqual(acct.conversation_id, "github:u")
        # the voice row came through the adapter, kind intact
        (voice,) = by_kind["voice"]
        self.assertEqual(voice.frontend, "voice")
        self.assertEqual(voice.state["voice_state"], "listening")
        self.assertEqual(voice.conversation_id, "voice-1")

    def test_no_sources_unified_equals_active(self):
        bridge = _bridge_with_chat()
        self.assertEqual([s.id for s in bridge.list_unified()],
                         [s.id for s in bridge.store.list_active()])

    def test_adapter_failure_degrades_honestly(self):
        manager = _FakeManager(
            {("github", "u"): _FakeAccount(),
             ("gitlab", "v"): _FakeAccount("gitlab", "v")},
            fail_peek_for={("gitlab", "v")})
        bridge = _bridge_with_chat()
        bridge.register_account_manager(manager)
        rows = bridge.list_unified()
        kinds = [(r.state.get("kind"), r.state.get("degraded", False))
                 for r in rows]
        # the good account still projects; the broken one is NAMED, not faked
        self.assertIn(("account", False), kinds)
        degraded = [r for r in rows if r.state.get("degraded")]
        self.assertEqual(len(degraded), 1)
        self.assertEqual(degraded[0].state["service"], "gitlab")
        self.assertEqual(degraded[0].state["username"], "v")
        self.assertIn("vault read boom", degraded[0].state["error"])
        # and the bridge's own row survived the failure
        self.assertTrue(any(r.frontend == "cli" for r in rows))

    def test_failing_source_does_not_kill_enumeration(self):
        bridge = _bridge_with_chat()

        def _boom():
            raise RuntimeError("source exploded")

        bridge.register_adapter_source("voice", _boom)
        rows = bridge.list_unified()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].frontend, "cli")

    def test_no_duplicates_when_two_sources_claim_one_concept(self):
        voice = _FakeVoice()
        bridge = _bridge_with_chat()
        bridge.register_adapter_source("voice", lambda: [voice])
        bridge.register_adapter_source("voice", lambda: [voice])
        rows = bridge.list_unified()
        self.assertEqual(
            [r.conversation_id for r in rows].count("voice-1"), 1)

    def test_account_projection_prefers_peek_never_get(self):
        manager = _FakeManager({("github", "u"): _FakeAccount()})
        bridge = _bridge_with_chat()
        bridge.register_account_manager(manager)
        bridge.list_unified()
        self.assertEqual(manager.peek_calls, [("github", "u")])
        self.assertEqual(manager.get_calls, [])

    def test_projected_ids_stable_across_calls(self):
        manager = _FakeManager({("github", "u"): _FakeAccount()})
        bridge = _bridge_with_chat()
        bridge.register_account_manager(manager)
        bridge.register_adapter_source("voice", lambda: [_FakeVoice()])
        first = {r.conversation_id: r.id for r in bridge.list_unified()}
        second = {r.conversation_id: r.id for r in bridge.list_unified()}
        self.assertEqual(first["github:u"], second["github:u"])
        self.assertEqual(first["voice-1"], second["voice-1"])

    def test_unknown_kind_rejected_at_registration(self):
        bridge = _bridge_with_chat()
        with self.assertRaises(ValueError):
            bridge.register_adapter_source("nope", lambda: [])

    def test_generic_payload_projects(self):
        bridge = _bridge_with_chat()
        bridge.register_adapter_source(
            "generic", lambda: [{"session_id": "g1", "status": "open"}])
        rows = bridge.list_unified()
        generic = [r for r in rows if r.state.get("kind") == "generic"]
        self.assertEqual(len(generic), 1)
        self.assertEqual(generic[0].conversation_id, "g1")
        self.assertEqual(generic[0].state["status"], "open")


class AccountSourceHelperTests(unittest.TestCase):
    def test_try_register_returns_false_when_vault_locked(self):
        from nomorals.os.session_bridge import try_register_account_source

        bridge = _bridge_with_chat()
        with patch("nomorals.accounts.vault.open_vault",
                   side_effect=RuntimeError("locked")):
            self.assertFalse(
                try_register_account_source(bridge, ":memory:"))
        # bridge still enumerates its own rows afterwards
        self.assertEqual(len(bridge.list_unified()), 1)


class SessionCommandTests(unittest.TestCase):
    def _run(self, task, *, context=None):
        from nomorals.cmdline.commands.session import _cmd_session

        args = argparse.Namespace(task=task.split(), json=False)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_session(args, context or SimpleNamespace(db=":memory:"))
        return rc, buf.getvalue()

    def test_list_shows_adapter_rows(self):
        manager = _FakeManager({("github", "u"): _FakeAccount()})

        def _fake_try_register(bridge, db, *, context=None):
            bridge.register_account_manager(manager)
            return True

        with patch("nomorals.os.session_bridge.try_register_account_source",
                   side_effect=_fake_try_register):
            rc, out = self._run("list")
        self.assertEqual(rc, 0)
        self.assertIn("github:u", out)
        self.assertIn("kind=account", out)

    def test_show_falls_back_to_unified_view(self):
        manager = _FakeManager({("github", "u"): _FakeAccount()})

        def _fake_try_register(bridge, db, *, context=None):
            bridge.register_account_manager(manager)
            return True

        with patch("nomorals.os.session_bridge.try_register_account_source",
                   side_effect=_fake_try_register):
            _, out = self._run("list")
            adapter_id = None
            for line in out.splitlines():
                if "github:u" in line:
                    adapter_id = line.split()[0]
            self.assertIsNotNone(adapter_id)
            rc, shown = self._run(f"show {adapter_id}")
        self.assertEqual(rc, 0)
        self.assertIn("github:u", shown)

    def test_end_on_adapter_view_refuses_honestly(self):
        manager = _FakeManager({("github", "u"): _FakeAccount()})

        def _fake_try_register(bridge, db, *, context=None):
            bridge.register_account_manager(manager)
            return True

        with patch("nomorals.os.session_bridge.try_register_account_source",
                   side_effect=_fake_try_register):
            _, out = self._run("list")
            adapter_id = None
            for line in out.splitlines():
                if "github:u" in line:
                    adapter_id = line.split()[0]
            self.assertIsNotNone(adapter_id)
            rc, msg = self._run(f"end {adapter_id}")
        self.assertEqual(rc, 1)
        self.assertIn("adapter view", msg)
        self.assertIn(adapter_id, msg)


if __name__ == "__main__":
    unittest.main()

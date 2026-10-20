"""/artifact chat command: registry, owner gate, resolve, send.

The three pieces existed but were never wired together: /db could show
artifact *rows* as text, ArtifactStore could resolve/read/export, and
file_send could deliver a file — but no chat command connected them.
These tests pin the wiring: /artifact list|info|send resolves through
ArtifactStore (id, artifact:// URI, artm:// URI, or alias) and sends
the real bytes to the current chat through the file_send tool.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef
from nomorals.social.chat.control import (
    COMMAND_DETAILS, CONTROL_COMMANDS, LIST_GROUPS, LIST_ONELINERS,
    PUBLIC_COMMANDS, caller_may_run, parse_control)
from nomorals.storage.artifacts import ArtifactStore
from nomorals.storage.blob import BlobStore


def _msg(text: str, *, chat_key: str = "telegram:999",
         owner: bool = True) -> ChatMessage:
    platform, _, chat_id = chat_key.partition(":")
    return ChatMessage(
        chat=ChatRef(platform=platform or "telegram",
                     chat_id=chat_id or "999", kind=ChatKind.DM),
        incoming=True, text=text,
        sender="owner" if owner else "stranger",
        sender_id="5478650254" if owner else "999",
        meta={"is_owner": True} if owner else {},
    )


class _RecordingGateway:
    adapters: dict = {}

    def __init__(self) -> None:
        self.sent: list = []

    def status(self) -> dict:
        return {}

    def set_rate_limit(self, *a: Any, **k: Any) -> None:
        pass

    def typing(self, *a: Any, **k: Any) -> None:
        pass

    def send(self, platform: str, chat: Any, text: str,
             **kwargs: Any) -> SimpleNamespace:
        self.sent.append((platform, text))
        return SimpleNamespace(ok=True, error="")


class _FakeTools:
    """The file_send seam: records calls, returns a canned outcome."""

    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.calls: list[tuple[str, dict]] = []

    def call(self, name: str, **kwargs: Any) -> SimpleNamespace:
        self.calls.append((name, kwargs))
        if name != "file_send":
            return SimpleNamespace(
                ok=False, value=None,
                error=SimpleNamespace(message=f"unexpected tool {name}"))
        if not self.ok:
            return SimpleNamespace(
                ok=False, value=None,
                error=SimpleNamespace(message="gateway exploded"))
        path = kwargs["path"]
        return SimpleNamespace(
            ok=True,
            value={"path": path, "bytes": os.path.getsize(path)},
            error=None)


class ArtifactRegistrationTest(unittest.TestCase):
    def test_registered_everywhere(self) -> None:
        self.assertIn("artifact", CONTROL_COMMANDS)
        self.assertIn("artifact", COMMAND_DETAILS)
        self.assertIn("artifact", LIST_ONELINERS)
        grouped = [c for _, cs in LIST_GROUPS for c in cs]
        self.assertEqual(grouped.count("artifact"), 1)
        self.assertIn("/artifact", COMMAND_DETAILS["artifact"]["usage"])

    def test_owner_only(self) -> None:
        self.assertNotIn("artifact", PUBLIC_COMMANDS)
        self.assertFalse(caller_may_run("artifact", False))
        self.assertTrue(caller_may_run("artifact", True))

    def test_parse_control(self) -> None:
        cmd = parse_control("/artifact send abc123")
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertEqual(cmd.kind, "artifact")
        self.assertEqual(cmd.arg, "send")
        self.assertEqual(cmd.tail, "send abc123")
        self.assertIsNotNone(parse_control("/artifact"))
        self.assertIsNotNone(parse_control("/artifact list"))


class ArtifactCommandTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-artifact-cmd-")
        settings = load_settings(overrides={
            "home": self.tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
            "partner.owner_chats": "5478650254",
            "partner.owner_sender_ids": "5478650254",
        })
        self.context = build_context(settings, with_executor=False,
                                     with_tools=False)

        class _Router:
            def chat(self, *a: Any, **k: Any) -> Any:
                raise AssertionError("test must not call the model")

        self.context.router = _Router()
        self.tools = _FakeTools()
        self.context.tools = self.tools
        self.runtime = PartnerRuntime(self.context,
                                      gateway=_RecordingGateway())
        # Seed through the same store construction the handler uses:
        # context db + blobs beside the db file.
        db = self.context.db
        blob_dir = Path(db.path).parent / "blobs"
        self.store = ArtifactStore(db, BlobStore(db, blob_dir))
        self.report = self.store.put_text(
            "# Q3 report\n\nrevenue grew.", type="report",
            creator="analyst", metadata={"title": "Q3 report"})
        self.data = self.store.put_json({"rows": [1, 2, 3]}, type="json",
                                        creator="datasci")
        self.store.set_alias("latest-report", self.report.id)

    def tearDown(self) -> None:
        try:
            self.runtime.stop()
        except Exception:
            pass
        self.context.close()
        self.tmp.cleanup()

    def _run(self, text: str, *, owner: bool = True,
             chat_key: str = "telegram:999") -> str:
        return self.runtime.handle_control(
            text, chat_key, message=_msg(text, chat_key=chat_key,
                                         owner=owner))

    # ── list / info ─────────────────────────────────────────────────────
    def test_list_shows_recent_artifacts(self) -> None:
        out = self._run("/artifact list")
        self.assertIn(self.report.id, out)
        self.assertIn(self.data.id, out)
        self.assertIn("report", out)
        self.assertIn("latest-report", out)  # alias is listed too

    def test_bare_artifact_lists(self) -> None:
        out = self._run("/artifact")
        self.assertIn(self.report.id, out)

    def test_info_by_id(self) -> None:
        out = self._run(f"/artifact info {self.report.id}")
        self.assertIn(self.report.id, out)
        self.assertIn("report", out)
        self.assertIn(str(self.report.size), out)

    def test_info_by_alias(self) -> None:
        out = self._run("/artifact info latest-report")
        self.assertIn(self.report.id, out)

    def test_info_by_uri(self) -> None:
        out = self._run(f"/artifact info artifact://{self.data.id}")
        self.assertIn(self.data.id, out)

    # ── send ────────────────────────────────────────────────────────────
    def test_send_by_id_delivers_real_bytes(self) -> None:
        out = self._run(f"/artifact send {self.report.id}")
        self.assertIn("sent", out)
        self.assertEqual(len(self.tools.calls), 1)
        name, kwargs = self.tools.calls[0]
        self.assertEqual(name, "file_send")
        self.assertEqual(kwargs["platform"], "telegram")
        self.assertEqual(kwargs["chat_id"], "999")
        sent_bytes = Path(kwargs["path"]).read_bytes()
        self.assertEqual(sent_bytes, self.store.read(self.report.id))

    def test_send_by_alias(self) -> None:
        out = self._run("/artifact send latest-report")
        self.assertIn("sent", out)
        self.assertEqual(len(self.tools.calls), 1)

    def test_send_by_artm_uri(self) -> None:
        out = self._run(f"/artifact send artm://{self.data.id}")
        self.assertIn("sent", out)
        self.assertEqual(len(self.tools.calls), 1)
        _, kwargs = self.tools.calls[0]
        self.assertEqual(Path(kwargs["path"]).read_bytes(),
                         self.store.read(self.data.id))

    def test_send_failure_is_honest(self) -> None:
        self.tools.ok = False
        out = self._run(f"/artifact send {self.report.id}")
        self.assertIn("failed", out)
        self.assertIn("gateway exploded", out)
        self.assertNotIn("📎 sent", out)

    def test_send_missing_blob_is_honest(self) -> None:
        # Destroy the blob bytes behind the artifact's back: the read
        # must surface as an honest error, never a crash or a fake send.
        blob_root = Path(self.context.db.path).parent / "blobs"
        for path in blob_root.rglob("*"):
            if path.is_file():
                path.unlink()
        out = self._run(f"/artifact send {self.report.id}")
        self.assertIn("couldn't", out)
        self.assertEqual(self.tools.calls, [])

    # ── honesty + gate ──────────────────────────────────────────────────
    def test_unknown_ref_lists_what_exists(self) -> None:
        out = self._run("/artifact info no-such-thing")
        self.assertIn("unknown artifact", out)
        self.assertIn(self.report.id, out)  # shows what DOES exist
        out = self._run("/artifact send no-such-thing")
        self.assertIn("unknown artifact", out)
        self.assertEqual(self.tools.calls, [])

    def test_non_owner_is_refused(self) -> None:
        out = self._run("/artifact list", owner=False)
        self.assertEqual(out, "that one's just for the owner.")
        out = self._run(f"/artifact send {self.report.id}", owner=False)
        self.assertEqual(out, "that one's just for the owner.")
        self.assertEqual(self.tools.calls, [])

    def test_usage_for_missing_ref(self) -> None:
        out = self._run("/artifact send")
        self.assertIn("usage", out)
        out = self._run("/artifact info")
        self.assertIn("usage", out)
        out = self._run("/artifact frobnicate")
        self.assertIn("usage", out)


if __name__ == "__main__":
    unittest.main()

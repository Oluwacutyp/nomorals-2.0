"""Tests for the chat /trendops formats list with a never-expires format.

Proves: a format whose ``expires_at`` is 0 (never expires — reachable
via ``FormatLibrary.new_format(expires_days=0)`` or any spec added
without an expiry) lists through the chat path without crashing, and
shows the honest "never expires" wording the CLI already uses. Before
the fix the list died with ``trendops failed: unsupported format
string passed to NoneType.__format__`` — the ``f"{days_left:.0f}"``
format blew up on the ``None`` that ``TrendFormat.days_left()``
returns for a never-expires format — so the owner saw a failure line
instead of the format list. An expiring format in the same list still
shows its days left, and the empty list still answers honestly.
"""

from __future__ import annotations

import contextlib
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.media.trendops import TrendOps

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_artifact_command import _FakeTools, _RecordingGateway  # noqa: E402
from test_trendops import _test_format  # noqa: E402

_OWNER_ID = "5478650254"


class TestChatFormatsNeverExpires(unittest.TestCase):
    """Chat /trendops formats over a tmp home, DEFAULT_ROOT doubled."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-trendops-listfix-")
        self.trend_root = Path(self.tmp.name) / "trendops"
        settings = load_settings(overrides={
            "home": self.tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
            "partner.owner_chats": _OWNER_ID,
            "partner.owner_sender_ids": _OWNER_ID,
        })
        self.context = build_context(settings, with_executor=False,
                                     with_tools=False)

        class _Router:
            def chat(self, *a: Any, **k: Any) -> Any:
                raise AssertionError("test must not call the model")

        self.context.router = _Router()
        self.context.tools = _FakeTools()
        self.runtime = PartnerRuntime(self.context,
                                      gateway=_RecordingGateway())
        self._patcher = mock.patch(
            "nomorals.media.trendops.editor.DEFAULT_ROOT",
            self.trend_root)
        self._patcher.start()
        self.ops = TrendOps(context=self.context)

    def tearDown(self) -> None:
        self._patcher.stop()
        with contextlib.suppress(Exception):
            self.runtime.stop()
        self.context.close()
        self.tmp.cleanup()

    def _formats(self) -> str:
        reply = self.runtime._control_trendops("formats")
        assert isinstance(reply, str), reply
        return reply

    def test_never_expires_format_lists_with_no_expiry_wording(self):
        # A never-expires format, built through the real from_dict
        # path (expires_at 0.0 — the same shape new_format(expires_days=0)
        # and any expiry-less spec produce), beside an expiring one.
        self.ops.add_format(_test_format(slug="forever", name="Forever",
                                         expires_at=0.0))
        self.ops.add_format(_test_format(slug="testcut", name="Test cut"))
        reply = self._formats()
        assert not reply.startswith("trendops failed:"), reply
        lines = reply.splitlines()
        forever = [ln for ln in lines if ln.startswith("• forever ")]
        assert len(forever) == 1, reply
        assert "never expires" in forever[0], reply
        cut = [ln for ln in lines if ln.startswith("• testcut ")]
        assert len(cut) == 1, reply
        assert "d left" in cut[0], reply

    def test_empty_library_still_answers_honestly(self):
        # A fresh library auto-seeds roadster on every fresh instance,
        # so the empty branch is only reachable when formats exist but
        # none are live: remove the seed, leave one expired format
        # behind (files present, so no re-seed), and the list is empty.
        import time

        self.ops.remove_format("roadster")
        self.ops.add_format(_test_format(
            slug="old", name="Old", expires_at=time.time() - 3600.0))
        reply = self._formats()
        assert reply.startswith("no live trend formats"), reply

    def test_propose_with_never_expires_format_stays_safe(self):
        # The sibling surface: propose's "why" already guards the None
        # (editor.py), so a never-expires format proposes without a
        # failure line — pinned here so a future edit cannot break it
        # the way the list broke.
        self.ops.add_format(_test_format(slug="forever", name="Forever",
                                         expires_at=0.0))
        reply = self.runtime._control_trendops("propose")
        assert isinstance(reply, str), reply
        assert not reply.startswith("trendops failed:"), reply
        assert "Forever is live" in reply, reply
        assert "/trendops apply forever <footage>" in reply, reply

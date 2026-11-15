"""Tests for TrendOps → agent-drafts wiring.

Proves: a finished TrendOps render becomes a post draft (MP4 path +
format name + ready re-cut command in metadata, media entry carrying
the MP4) and the owner is asked "post directly or send for review?"
through the drafts organ's own propose_post — and the draft resolves
(post / discard) through the existing callback route. The render
itself is unchanged: same MP4, same report, only the delivery after
it changes.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.media.trendops import TrendOps
from nomorals.media.trendops.editor import draft_for_render
from nomorals.social import drafts as drafts_mod
from nomorals.social.drafts import (
    DISCARDED,
    PENDING_REVIEW,
    POSTED,
    DraftQueue,
)

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_artifact_command import _FakeTools, _RecordingGateway  # noqa: E402
from test_trendops import _mp4, _test_format  # noqa: E402

NEEDS_FFMPEG = pytest.mark.skipif(
    os.system("ffmpeg -hide_banner -version > /dev/null 2>&1") != 0,
    reason="ffmpeg not installed",
)

_REAL_QUEUE = DraftQueue
_REAL_TRENDOPS = TrendOps
_OWNER_ID = "5478650254"


# ---------------------------------------------------------------------------
# draft_for_render — pure data shaping over a real render report
# ---------------------------------------------------------------------------

@pytest.fixture()
def root(tmp_path):
    return tmp_path / "trendops"


@pytest.fixture()
def ops(root):
    return TrendOps(root=root)


@pytest.fixture()
def workdir(tmp_path):
    d = tmp_path / "footage"
    d.mkdir()
    return d


@NEEDS_FFMPEG
class TestDraftForRender:
    def test_render_report_becomes_draft(self, ops, workdir):
        clip = _mp4(workdir)
        ops.add_format(_test_format())
        report = ops.apply_format(str(clip), "testcut",
                                  beats=[1.0, 2.0, 3.0])
        assert report["ok"] is True
        content, metadata = draft_for_render(
            report, footage=[str(clip)])
        assert "Test cut" in content
        assert metadata["origin"] == "trendops"
        assert metadata["format"] == "testcut"
        assert metadata["format_name"] == "Test cut"
        assert metadata["output"] == report["output"]
        assert Path(metadata["output"]).is_file()
        assert metadata["media"] == [{
            "path": report["output"],
            "name": Path(report["output"]).name,
            "mime": "video/mp4"}]
        assert metadata["apply"].startswith("/trendops apply testcut")
        assert str(clip) in metadata["apply"]
        assert metadata["duration"] == report["duration"]
        assert metadata["segments"] == report["segments"]
        assert metadata["review"] == report["review"]

    def test_caption_becomes_post_text(self, ops, workdir):
        clip = _mp4(workdir)
        ops.add_format(_test_format())
        report = ops.apply_format(
            str(clip), "testcut", caption="Me and Bro in 2030",
            beats=[1.0, 2.0, 3.0])
        content, metadata = draft_for_render(
            report, footage=[str(clip)], caption="Me and Bro in 2030")
        assert content == "Me and Bro in 2030"
        assert "Me and Bro in 2030" in metadata["apply"]


# ---------------------------------------------------------------------------
# end to end: /trendops apply → draft proposed → post / discard resolves
# ---------------------------------------------------------------------------

class _FakeSocialManager:
    """The publisher seam: records publishes, posts to one platform.

    Accepts the media_paths / reply_to extras execute_post threads
    through for a media-carrying draft."""

    instances: list[_FakeSocialManager] = []

    def __init__(self, context: Any) -> None:
        self.context = context
        self.published: list[tuple[str, Any, dict]] = []

    def register_builtins(self) -> _FakeSocialManager:
        _FakeSocialManager.instances.append(self)
        return self

    def publish(self, content: str, platforms: Any = None,
                actor: str = "user", confirmation: str = "",
                **extra: Any) -> Any:
        self.published.append((content, platforms, extra))
        return SimpleNamespace(
            posted=["x"], failed=[], error="",
            results=[SimpleNamespace(
                platform="x", ok=True, external_id="ext1", url="",
                metrics={"media_offered": 1, "media_carried": 1})])


class _RuntimeRig(unittest.TestCase):
    """PartnerRuntime over a tmp home (the button-fix harness shape),
    with the DraftQueue factory, the SocialManager, and the TrendOps
    root doubled — the dispatch, the render, the draft lifecycle
    and the callback router stay real."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-trendops-drafts-")
        self.db_path = str(Path(self.tmp.name) / "drafts.db")
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
        _FakeSocialManager.instances = []
        self._patchers = [
            mock.patch.object(
                drafts_mod, "DraftQueue",
                lambda *a, **k: _REAL_QUEUE(db_path=self.db_path)),
            mock.patch("nomorals.social.SocialManager",
                       _FakeSocialManager),
            mock.patch(
                "nomorals.media.trendops.editor.DEFAULT_ROOT",
                self.trend_root),
        ]
        for patcher in self._patchers:
            patcher.start()
        # Seed the test format on the tmp library the control path reads.
        seed_ops = _REAL_TRENDOPS(context=self.context)
        seed_ops.add_format(_test_format())

    def tearDown(self) -> None:
        for patcher in self._patchers:
            patcher.stop()
        with contextlib.suppress(Exception):
            self.runtime.stop()
        self.context.close()
        self.tmp.cleanup()

    def _draft(self, draft_id: str) -> Any:
        queue = _REAL_QUEUE(db_path=self.db_path)
        try:
            return queue.get(draft_id)
        finally:
            queue.close()

    def _status(self, draft_id: str) -> str:
        draft = self._draft(draft_id)
        assert draft is not None
        return draft.status

    def _apply(self, clip: str) -> Any:
        return self.runtime._control_trendops(
            f"apply testcut {clip} --beats 1.0,2.0,3.0")

    @staticmethod
    def _buttons(reply: Any) -> list[tuple[str, str]]:
        keyboard = getattr(reply, "buttons", None) or []
        out: list[tuple[str, str]] = []
        for row in keyboard:
            for pair in row:
                out.append((str(pair[0]), str(pair[1])))
        return out

    @NEEDS_FFMPEG
    def test_apply_creates_draft_and_asks(self):
        workdir = Path(self.tmp.name) / "footage"
        workdir.mkdir()
        clip = _mp4(workdir)
        reply = self._apply(str(clip))
        text = str(getattr(reply, "text", "") or "")
        assert "cut to Test cut" in text
        assert "Post directly or send for review?" in text
        buttons = self._buttons(reply)
        post = [cb for _label, cb in buttons
                if cb.startswith("draft:post:")]
        assert len(post) == 1
        draft_id = post[0].split(":")[2]
        draft = self._draft(draft_id)
        assert draft is not None
        assert draft.status == PENDING_REVIEW
        assert draft.metadata["origin"] == "trendops"
        assert draft.metadata["format"] == "testcut"
        assert draft.metadata["format_name"] == "Test cut"
        assert draft.metadata["output"].endswith(".mp4")
        assert Path(draft.metadata["output"]).is_file()
        assert draft.metadata["media"][0]["path"] == \
            draft.metadata["output"]
        assert draft.metadata["media"][0]["mime"] == "video/mp4"
        assert draft.metadata["apply"].startswith(
            "/trendops apply testcut")

    @NEEDS_FFMPEG
    def test_draft_post_resolves(self):
        workdir = Path(self.tmp.name) / "footage"
        workdir.mkdir()
        clip = _mp4(workdir)
        reply = self._apply(str(clip))
        post = [cb for _label, cb in self._buttons(reply)
                if cb.startswith("draft:post:")]
        draft_id = post[0].split(":")[2]
        out = self.runtime._draft_callback(
            f"draft:post:{draft_id}", chat_key="telegram:999")
        assert "posted to" in out
        assert self._status(draft_id) == POSTED
        assert len(_FakeSocialManager.instances) == 1
        published = _FakeSocialManager.instances[0].published
        assert len(published) == 1
        content, _platforms, extra = published[0]
        assert "Test cut" in content
        media_paths = extra.get("media_paths") or []
        assert len(media_paths) == 1
        assert Path(media_paths[0]).suffix == ".mp4"

    @NEEDS_FFMPEG
    def test_draft_discard_resolves(self):
        workdir = Path(self.tmp.name) / "footage"
        workdir.mkdir()
        clip = _mp4(workdir)
        reply = self._apply(str(clip))
        post = [cb for _label, cb in self._buttons(reply)
                if cb.startswith("draft:post:")]
        draft_id = post[0].split(":")[2]
        out = self.runtime._draft_callback(
            f"draft:discard:{draft_id}", chat_key="telegram:999")
        assert "discarded" in out
        assert self._status(draft_id) == DISCARDED

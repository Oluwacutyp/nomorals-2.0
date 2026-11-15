"""Tests for TrendOps → Museum artifacts.

Proves: a finished render writes a JSON manifest next to the MP4
(format, duration, canvas, beats, segments, the WatchVideo verdict,
a draft-id slot); the Museum's trendops exhibitor reads the renders
dir at census time and each cut lands as a kept artifact — findable
by id and by unique prefix, listed on the trendops-renders shelf,
retrievable with its MP4 path, format, QA verdict and draft id; a
render with no manifest still exhibits; a corrupt manifest never
kills the shelf; a deleted MP4 leaves no artifact (the file IS the
artifact); and the chat path stamps the draft id onto the manifest, so the card
names the draft that carries the cut. The render itself is
unchanged: same MP4, same report, only the record after it is new.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.media.trendops import TrendOps
from nomorals.media.trendops.editor import (
    stamp_render_draft,
    write_render_manifest,
)
from nomorals.museum import Museum, trendops_exhibitor
from nomorals.social import drafts as drafts_mod
from nomorals.social.drafts import DraftQueue

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_artifact_command import _FakeTools, _RecordingGateway  # noqa: E402
from test_trendops import _mp4, _test_format  # noqa: E402
from test_trendops_watchvideo import _testsrc  # noqa: E402

NEEDS_FFMPEG = pytest.mark.skipif(
    os.system("ffmpeg -hide_banner -version > /dev/null 2>&1") != 0,
    reason="ffmpeg not installed",
)

_REAL_QUEUE = DraftQueue
_REAL_TRENDOPS = TrendOps
_OWNER_ID = "5478650254"


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


def _render(ops, workdir, *, clip_fn=_mp4, **kwargs):
    clip = clip_fn(workdir)
    ops.add_format(_test_format())
    return ops.apply_format(str(clip), "testcut",
                            beats=[1.0, 2.0, 3.0], **kwargs)


# ---------------------------------------------------------------------------
# manifest — written by the render, read by the exhibitor
# ---------------------------------------------------------------------------

@NEEDS_FFMPEG
class TestRenderManifest:
    def test_manifest_written_next_to_mp4(self, ops, workdir):
        report = _render(ops, workdir, clip_fn=_testsrc)
        output = Path(report["output"])
        assert output.is_file()
        manifest_path = output.with_suffix(".json")
        assert manifest_path.is_file()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["format"] == "testcut"
        assert manifest["format_name"] == "Test cut"
        assert manifest["output"] == str(output)
        assert manifest["duration"] == report["duration"]
        assert manifest["beats_used"] == 3
        assert manifest["segments"] == report["segments"]
        assert manifest["review"] == report["review"]
        assert manifest["draft_id"] is None
        assert manifest["created_at"] > 0

    def test_manifest_carries_real_qa_verdict(self, ops, workdir):
        report = _render(ops, workdir, clip_fn=_testsrc)
        manifest = json.loads(
            Path(report["output"]).with_suffix(".json").read_text(
                encoding="utf-8"))
        qa = manifest["review"]["qa"]
        assert qa["status"] == "pass"
        assert len(qa["checks"]["cuts"]["found"]) == 3

    def test_solid_clip_manifest_says_unavailable(self, ops, workdir):
        report = _render(ops, workdir)
        manifest = json.loads(
            Path(report["output"]).with_suffix(".json").read_text(
                encoding="utf-8"))
        assert manifest["review"]["qa"]["status"] == "qa_unavailable"

    def test_write_manifest_never_raises(self):
        assert write_render_manifest({}) is None
        assert write_render_manifest({"output": "/no/such/dir/out.mp4"}) \
            is None

    def test_stamp_sets_draft_id(self, ops, workdir):
        report = _render(ops, workdir, clip_fn=_testsrc)
        assert stamp_render_draft(report["output"], "draft-123") is True
        manifest = json.loads(
            Path(report["output"]).with_suffix(".json").read_text(
                encoding="utf-8"))
        assert manifest["draft_id"] == "draft-123"
        # the rest of the record survives the stamp
        assert manifest["format"] == "testcut"
        assert manifest["review"]["qa"]["status"] == "pass"

    def test_stamp_missing_manifest_is_quiet_false(self, tmp_path):
        assert stamp_render_draft(str(tmp_path / "ghost.mp4"),
                                  "draft-1") is False


# ---------------------------------------------------------------------------
# exhibitor — the renders dir as a Museum shelf
# ---------------------------------------------------------------------------

@NEEDS_FFMPEG
class TestTrendopsExhibitor:
    def test_render_becomes_artifact(self, ops, workdir):
        report = _render(ops, workdir, clip_fn=_testsrc)
        cards = trendops_exhibitor(ops.root)
        assert len(cards) == 1
        card = cards[0]
        stem = Path(report["output"]).stem
        assert card["id"] == f"trendops-render:{stem}"
        assert card["kind"] == "trendops-render"
        assert card["shelf"] == "trendops-renders"
        assert card["path"] == report["output"]
        assert card["size"] > 0
        assert card["hash"]
        assert "Test cut" in card["title"]
        assert card["provenance"]["tool"] == "trendops"
        assert card["provenance"]["ref"] == "testcut"
        assert card["provenance"]["qa"] == "pass"
        assert card["provenance"]["draft_id"] == ""
        assert "format:testcut" in card["tags"]
        assert "qa:pass" in card["tags"]
        assert "trendops" in card["tags"]

    def test_stamped_draft_on_card(self, ops, workdir):
        report = _render(ops, workdir, clip_fn=_testsrc)
        stamp_render_draft(report["output"], "draft-123")
        card = trendops_exhibitor(ops.root)[0]
        assert card["provenance"]["draft_id"] == "draft-123"
        assert "draft:draft-123" in card["tags"]

    def test_museum_census_find_render(self, ops, workdir):
        report = _render(ops, workdir, clip_fn=_testsrc)
        museum = Museum(trendops_root=ops.root)
        res = museum.census()
        assert res["by_shelf"].get("trendops-renders") == 1
        assert res["errors"] == {}
        stem = Path(report["output"]).stem
        found = museum.find(f"trendops-render:{stem}")
        assert found is not None
        assert found["path"] == report["output"]
        assert found["sendable"] is True
        # find() re-runs the census, so the prefix hit is an equal
        # card, not the identical dict.
        assert museum.find("trendops-render:") == found
        assert museum.find("trendops-render:nope") is None
        text = museum.render(res)
        assert "Test cut" in text
        assert "trendops-renders" in text

    def test_no_renders_is_empty_not_error(self, tmp_path):
        assert trendops_exhibitor(tmp_path / "trendops") == []
        museum = Museum(trendops_root=tmp_path / "trendops")
        res = museum.census()
        assert res["count"] == 0
        assert res["errors"] == {}

    def test_render_without_manifest_still_exhibits(self, ops, workdir):
        report = _render(ops, workdir, clip_fn=_testsrc)
        Path(report["output"]).with_suffix(".json").unlink()
        cards = trendops_exhibitor(ops.root)
        assert len(cards) == 1
        assert cards[0]["kind"] == "trendops-render"
        assert cards[0]["path"] == report["output"]

    def test_corrupt_manifest_still_exhibits(self, ops, workdir):
        report = _render(ops, workdir, clip_fn=_testsrc)
        Path(report["output"]).with_suffix(".json").write_text(
            "{not json", encoding="utf-8")
        museum = Museum(trendops_root=ops.root)
        res = museum.census()
        assert res["by_shelf"].get("trendops-renders") == 1
        # handled inside the exhibitor (recorded in the failure
        # ledger), not an exhibitor-level error — the shelf stands.
        assert res["errors"] == {}

    def test_deleted_mp4_leaves_no_artifact(self, ops, workdir):
        report = _render(ops, workdir, clip_fn=_testsrc)
        stem = Path(report["output"]).stem
        Path(report["output"]).unlink()
        museum = Museum(trendops_root=ops.root)
        res = museum.census()
        assert res["by_shelf"].get("trendops-renders", 0) == 0
        # the file IS the artifact — no file, no card, nothing to send.
        assert museum.find(f"trendops-render:{stem}") is None


# ---------------------------------------------------------------------------
# end to end: /trendops apply → draft → manifest stamped → card names it
# ---------------------------------------------------------------------------

class _RuntimeRig(unittest.TestCase):
    """PartnerRuntime over a tmp home (the button-fix harness shape),
    with the DraftQueue factory and the TrendOps root doubled — the
    dispatch, the render, the draft lifecycle and the manifest stamp
    stay real."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-trendops-museum-")
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
        self._patchers = [
            mock.patch.object(
                drafts_mod, "DraftQueue",
                lambda *a, **k: _REAL_QUEUE(db_path=self.db_path)),
            mock.patch(
                "nomorals.media.trendops.editor.DEFAULT_ROOT",
                self.trend_root),
        ]
        for patcher in self._patchers:
            patcher.start()
        seed_ops = _REAL_TRENDOPS(context=self.context)
        seed_ops.add_format(_test_format())

    def tearDown(self) -> None:
        for patcher in self._patchers:
            patcher.stop()
        with contextlib.suppress(Exception):
            self.runtime.stop()
        self.context.close()
        self.tmp.cleanup()

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
    def test_apply_stamps_draft_onto_museum_card(self):
        workdir = Path(self.tmp.name) / "footage"
        workdir.mkdir()
        clip = _testsrc(workdir)
        reply = self._apply(str(clip))
        text = str(getattr(reply, "text", "") or "")
        assert "cut to Test cut" in text
        assert "Post directly or send for review?" in text
        post = [cb for _label, cb in self._buttons(reply)
                if cb.startswith("draft:post:")]
        assert len(post) == 1
        draft_id = post[0].split(":")[2]
        renders = self.trend_root / "renders"
        mp4s = sorted(renders.glob("*.mp4"))
        assert len(mp4s) == 1
        manifest = json.loads(
            mp4s[0].with_suffix(".json").read_text(encoding="utf-8"))
        assert manifest["draft_id"] == draft_id
        card = trendops_exhibitor(self.trend_root)[0]
        assert card["provenance"]["draft_id"] == draft_id
        assert f"draft:{draft_id}" in card["tags"]
        assert card["provenance"]["qa"] == "pass"
        museum = Museum(trendops_root=self.trend_root)
        found = museum.find(f"trendops-render:{mp4s[0].stem}")
        assert found is not None
        assert found["sendable"] is True

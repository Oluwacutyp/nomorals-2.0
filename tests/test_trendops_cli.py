"""Tests for ``nm trendops`` — CLI parity for TrendOps.

Proves: the command is registered (parser + dispatch table + alias),
every action dispatches through the same TrendOps methods chat uses
(list/get/add/remove/propose/capture/apply), the output carries the
same facts as the chat reply, ``--json`` output works, and failures
are clean (no traceback, non-zero exit). The CLI path creates no
draft — the render manifest it writes keeps ``draft_id`` None.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import pytest

from nomorals.cmdline import _COMMANDS, _parser
from nomorals.cmdline.commands.media import _cmd_trendops
from nomorals.media.trendops import TrendFormat, TrendOps

NEEDS_FFMPEG = pytest.mark.skipif(
    os.system("ffmpeg -hide_banner -version > /dev/null 2>&1") != 0,
    reason="ffmpeg not installed",
)


@pytest.fixture()
def root(tmp_path):
    return tmp_path / "trendops"


@pytest.fixture()
def workdir(tmp_path):
    d = tmp_path / "footage"
    d.mkdir()
    return d


def _mp4(workdir, name="clip.mp4", color="red", seconds=4.0, w=320, h=568):
    from nomorals.media_edit.videos import run_ffmpeg

    p = workdir / name
    run_ffmpeg(["-f", "lavfi", "-i",
                f"color=c={color}:s={w}x{h}:d={seconds}:r=30",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)],
               duration=seconds)
    return p


def _test_format(**overrides):
    data = {
        "slug": "testcut",
        "name": "Test cut",
        "ramp_curve": [[0.0, 0.5, 0.5], [0.5, 1.0, 1.0]],
        "cut_rule": "beats",
        "canvas": {"width": 320, "height": 568, "fps": 15},
        "expires_at": time.time() + 86400.0,
    }
    data.update(overrides)
    return TrendFormat.from_dict(data)


def _run(argv, context=None):
    """Parse argv through the real parser and run the handler.

    Returns (rc, stdout, stderr). Context defaults to None: the CLI
    actions under test need none — TrendOps takes the root from
    --root. Capture needs a (dummy) context for the download chain,
    so its tests pass one explicitly.
    """
    args = _parser().parse_args(argv)
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = _cmd_trendops(args, context)
    return rc, out.getvalue(), err.getvalue()


class _FakeReport:
    def __init__(self, ok, path):
        self.ok = ok
        self.path = path

    def summary(self):
        return "fake failure"


def _fake_downloader_cls(clip):
    class _FakeDownloader:
        def __init__(self, context):
            self.clip = clip

        def download(self, url, dest_dir=None, **_ignored):
            dest = Path(dest_dir) / "captured.mp4"
            shutil.copyfile(self.clip, dest)
            return _FakeReport(True, str(dest))

    return _FakeDownloader


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_parser_has_trendops(self, root):
        args = _parser().parse_args(
            ["trendops", "list", "--root", str(root)])
        assert args.trendops_action == "list"

    def test_alias_trend_parses(self, root):
        args = _parser().parse_args(["trend", "list", "--root", str(root)])
        assert args.trendops_action == "list"

    def test_dispatch_table_has_trendops(self):
        assert "trendops" in _COMMANDS
        assert callable(_COMMANDS["trendops"].handler)


# ---------------------------------------------------------------------------
# list / get / add / remove
# ---------------------------------------------------------------------------

class TestListGetAddRemove:
    def test_list_shows_seeded_roadster(self, root):
        rc, out, _err = _run(["trendops", "list", "--root", str(root)])
        assert rc == 0
        assert "roadster" in out
        assert "gta" in out
        assert "30d left" in out

    def test_list_json(self, root):
        rc, out, _err = _run(
            ["trendops", "list", "--root", str(root), "--json"])
        assert rc == 0
        data = json.loads(out)
        slugs = [f["slug"] for f in data["formats"]]
        assert "roadster" in slugs

    def test_get_shows_format(self, root):
        TrendOps(root=root).add_format(_test_format())
        rc, out, _err = _run(["trendops", "get", "testcut",
                              "--root", str(root)])
        assert rc == 0
        assert "testcut" in out
        assert "Test cut" in out

    def test_get_unknown_fails_clean(self, root):
        rc, out, err = _run(["trendops", "get", "nope",
                             "--root", str(root)])
        assert rc == 1
        assert "unknown trend format" in err
        assert "Traceback" not in err

    def test_add_from_spec_file(self, root, tmp_path):
        spec = tmp_path / "spec.json"
        spec.write_text(json.dumps(_test_format(
            slug="cliadded", name="CLI added").to_dict()))
        rc, out, _err = _run(["trendops", "add", "--spec-file", str(spec),
                              "--root", str(root)])
        assert rc == 0
        assert "cliadded" in out
        assert "cliadded" in TrendOps(root=root).library.list()

    def test_add_from_spec_string(self, root):
        spec = json.dumps(_test_format(
            slug="clistr", name="CLI string").to_dict())
        rc, out, _err = _run(["trendops", "add", "--spec", spec,
                              "--root", str(root)])
        assert rc == 0
        assert "clistr" in out

    def test_add_needs_a_spec(self, root):
        rc, _out, err = _run(["trendops", "add", "--root", str(root)])
        assert rc == 2
        assert "--spec" in err

    def test_remove(self, root):
        ops = TrendOps(root=root)
        ops.add_format(_test_format())
        rc, out, _err = _run(["trendops", "remove", "testcut",
                              "--root", str(root)])
        assert rc == 0
        assert "removed" in out
        assert "testcut" not in ops.library.list()

    def test_remove_missing_names_it(self, root):
        rc, out, _err = _run(["trendops", "remove", "nope",
                              "--root", str(root)])
        assert rc == 0
        assert "no trend format" in out


# ---------------------------------------------------------------------------
# propose
# ---------------------------------------------------------------------------

class TestPropose:
    def test_propose_ranks_by_expiry(self, root):
        now = time.time()
        ops = TrendOps(root=root)
        ops.add_format(_test_format(slug="soon",
                                    expires_at=now + 3600.0))
        ops.add_format(_test_format(slug="later",
                                    expires_at=now + 30 * 86400.0))
        rc, out, _err = _run(["trendops", "propose",
                              "--root", str(root)])
        assert rc == 0
        assert out.index("soon") < out.index("later")
        assert "/trendops apply soon" in out

    def test_propose_empty(self, root):
        # Expire the seeded roadster instead of removing it: an empty
        # formats dir re-seeds roadster on the next library open, so
        # only an expired record leaves propose with nothing live.
        ops = TrendOps(root=root)
        fmt = ops.get_format("roadster")
        fmt.expires_at = time.time() - 10.0
        ops.library.save(fmt)
        rc, out, _err = _run(["trendops", "propose",
                              "--root", str(root)])
        assert rc == 0
        assert "no live trend formats" in out


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------

class TestCapture:
    def test_capture_with_fake_downloader(self, root, workdir):
        clip = _mp4(workdir)
        with mock.patch(
                "nomorals.media.downloader.MediaDownloader",
                _fake_downloader_cls(clip)):
            rc, out, _err = _run([
                "trendops", "capture", "https://tiktok.example/@x/video/1",
                "--name", "caught", "--niche", "gta",
                "--root", str(root)], context=object())
        assert rc == 0
        assert "caught" in out
        assert "cut it: nm trendops apply caught" in out
        assert "caught" in TrendOps(root=root).library.list()

    def test_capture_download_failure_fails_closed(self, root):
        class _Fail:
            def __init__(self, context):
                pass

            def download(self, url, dest_dir=None, **_ignored):
                return _FakeReport(False, "")

        with mock.patch("nomorals.media.downloader.MediaDownloader",
                        _Fail):
            rc, _out, err = _run([
                "trendops", "capture", "https://tiktok.example/@x/video/1",
                "--name", "nope", "--root", str(root)], context=object())
        assert rc == 1
        assert "could not catch" in err
        assert "Traceback" not in err


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

@NEEDS_FFMPEG
class TestApply:
    def test_apply_renders_real_clip(self, root, workdir):
        clip = _mp4(workdir)
        TrendOps(root=root).add_format(_test_format())
        rc, out, _err = _run([
            "trendops", "apply", "testcut", str(clip),
            "--beats", "1.0,2.0,3.0", "--root", str(root)])
        assert rc == 0
        assert "cut to Test cut" in out
        assert "4 segments" in out
        assert "3 beats" in out
        out_path = Path(out.split("→", 1)[1].split("\n", 1)[0].strip())
        assert out_path.exists() and out_path.stat().st_size > 0

    def test_apply_creates_no_draft(self, root, workdir):
        clip = _mp4(workdir)
        TrendOps(root=root).add_format(_test_format())
        rc, out, _err = _run([
            "trendops", "apply", "testcut", str(clip),
            "--beats", "1.0,2.0", "--root", str(root)])
        assert rc == 0
        out_path = Path(out.split("→", 1)[1].split("\n", 1)[0].strip())
        manifest = out_path.with_suffix(".json")
        assert manifest.exists()
        assert json.loads(manifest.read_text())["draft_id"] is None

    def test_apply_missing_footage_fails_clean(self, root):
        TrendOps(root=root).add_format(_test_format())
        rc, _out, err = _run([
            "trendops", "apply", "testcut", "/nope/missing.mp4",
            "--beats", "1.0", "--root", str(root)])
        assert rc == 1
        assert "no such footage" in err
        assert "Traceback" not in err

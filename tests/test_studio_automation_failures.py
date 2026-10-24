"""Pins for Pass #115: studio automation failure ledger.

Every silent handler in nomorals/media/studio_automation.py records into
nomorals/media/studio_automation_failures.py; return values, control flow,
and fallback defaults are unchanged.
"""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from nomorals.media import studio_automation as sa
from nomorals.media.studio_automation_failures import (
    Failure,
    clear_failures,
    failure_count,
    format_failures,
    recent_failures,
    record_failure,
)


@pytest.fixture(autouse=True)
def _clean():
    clear_failures()


def _sites():
    return [f.site for f in recent_failures(100)]


# --- ledger basics -------------------------------------------------------

def test_ledger_record_and_recent():
    record_failure("pin.site", ValueError("boom"))
    assert failure_count() == 1
    (entry,) = recent_failures()
    assert isinstance(entry, Failure)
    assert entry.site == "pin.site"
    assert entry.exc_type == "ValueError"
    assert entry.message == "boom"


def test_ledger_bound():
    for _ in range(250):
        record_failure("pin.site", RuntimeError("x"))
    assert failure_count() == 250
    assert len(recent_failures(1000)) == 200


def test_ledger_unprintable_exc():
    class BadStr(Exception):
        def __str__(self):
            raise RuntimeError("nope")

    record_failure("pin.site", BadStr())
    (entry,) = recent_failures()
    assert "unprintable" in entry.message


def test_format_render_and_empty():
    assert format_failures() == "no studio automation failures recorded."
    record_failure("pin.site", KeyError("k"))
    rendered = format_failures()
    assert "pin.site" in rendered and "KeyError" in rendered


def test_direct_import_surface():
    import nomorals.media.studio_automation_failures as led

    assert led.__all__ == ["Failure", "record_failure", "recent_failures",
                           "failure_count", "clear_failures", "format_failures"]


# --- real converted sites ------------------------------------------------

def test_detect_silences_silence_end_failure(monkeypatch):
    monkeypatch.setattr(sa, "_ffmpeg", lambda: "ffmpeg")
    stderr = (
        "[ffmpeg] silence_start: 1.000\n"
        "[ffmpeg] silence_end: garbage\n"
        "[ffmpeg] silence_start: 5.000\n"
        "[ffmpeg] silence_end: 6.000 | silence_duration: 1.000\n"
    )
    monkeypatch.setattr(sa, "_run", lambda cmd, **kw: SimpleNamespace(stderr=stderr))
    out = sa.detect_silences("in.mp4")
    assert len(out) == 1  # the malformed segment dropped, the good pair kept
    assert out[0].start == 5.0 and out[0].end == 6.0
    assert _sites() == ["detect_silences.silence_end"]


def test_detect_silences_healthy_records_nothing(monkeypatch):
    monkeypatch.setattr(sa, "_ffmpeg", lambda: "ffmpeg")
    stderr = (
        "[ffmpeg] silence_start: 1.000\n"
        "[ffmpeg] silence_end: 2.500 | silence_duration: 1.500\n"
    )
    monkeypatch.setattr(sa, "_run", lambda cmd, **kw: SimpleNamespace(stderr=stderr))
    out = sa.detect_silences("in.mp4")
    assert len(out) == 1 and out[0].duration == 1.5
    assert _sites() == []


def test_detect_scenes_showinfo_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(sa, "_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(sa, "_ffprobe", lambda: None)  # skip duration probe
    stderr = (
        "frame:0 showinfo pts_time:2.500\n"
        "showinfo pts_time: garbage\n"
        "frame:1 showinfo pts_time:5.000\n"
    )
    monkeypatch.setattr(sa, "_run", lambda cmd, **kw: SimpleNamespace(stderr=stderr))
    scenes = sa.detect_scenes("in.mp4", workdir=str(tmp_path))
    assert len(scenes) == 3  # malformed cut dropped, good cuts kept
    assert _sites() == ["detect_scenes.showinfo"]


def test_detect_scenes_duration_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(sa, "_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(sa, "_ffprobe", lambda: "ffprobe")

    def fake_run(cmd, **kw):
        if any("select=" in str(a) for a in cmd):
            return SimpleNamespace(stderr="frame:0 showinfo pts_time:2.500\n",
                                   stdout="")
        return SimpleNamespace(stderr="", stdout="not-a-number")

    monkeypatch.setattr(sa, "_run", fake_run)
    scenes = sa.detect_scenes("in.mp4", workdir=str(tmp_path))
    assert len(scenes) == 2  # duration fell back to last cut + 1.0
    assert scenes[-1].end == 3.5
    assert _sites() == ["detect_scenes.duration"]


def test_auto_reframe_probe_failure(monkeypatch):
    monkeypatch.setattr(sa, "_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(sa, "_ffprobe", lambda: "ffprobe")
    monkeypatch.setattr(
        sa, "_run",
        lambda cmd, **kw: SimpleNamespace(stdout="this is not json",
                                           stderr="", returncode=0))
    result = sa.auto_reframe("in.mp4")
    assert result == {"ok": False, "reason": "could not probe video dimensions"}
    assert _sites() == ["auto_reframe.probe"]


def test_auto_reframe_healthy_probe_records_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(sa, "_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(sa, "_ffprobe", lambda: "ffprobe")
    probe = json.dumps({"streams": [{"width": 1920, "height": 1080}],
                        "format": {"duration": "10.0"}})

    def fake_run(cmd, **kw):
        if any("-select_streams" in str(a) for a in cmd):
            return SimpleNamespace(stdout=probe, stderr="", returncode=0)
        if any("-filter_complex" in str(a) for a in cmd):
            Path(cmd[-1]).touch()  # the render "succeeds"
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(sa, "_run", fake_run)
    result = sa.auto_reframe("in.mp4", out=str(tmp_path / "reframed.mp4"))
    assert result["ok"] is True
    assert "auto_reframe.probe" not in _sites()
    assert _sites() == []


# --- subsystem remainder ---------------------------------------------------

def _silent_handlers_without_record():
    """All-pass except handlers in the module that never record.

    Keyed by function name, never by line number.
    """
    path = Path(sa.__file__)
    tree = ast.parse(path.read_text())
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.ExceptHandler):
                continue
            body = [s for s in sub.body
                    if not (isinstance(s, ast.Expr)
                            and isinstance(s.value, ast.Constant)
                            and isinstance(s.value.value, str))]
            if body and all(isinstance(s, ast.Pass) for s in body):
                has_record = any(
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name)
                    and n.func.id == "record_failure"
                    for n in ast.walk(sub))
                if not has_record:
                    found.add(node.name)
    return found


def test_subsystem_silent_handlers_all_recorded():
    # No deliberate exclusions in this cluster: every silent handler records.
    assert _silent_handlers_without_record() == set()

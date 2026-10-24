"""Pins for Pass #120: render failures ledger (fidelity + genedit).

``fidelity`` and ``genedit`` degrade honestly in return-value shape
but silently about the underlying error: a broker whose ``select``
raises looks like a registry with no ``select``; a registry whose
``cards`` raises looks empty; a card with a malformed quality looks
unrated; a broken prompt engine looks like plain prompts; a broken
imggen import looks like passthrough-by-design; a failed keyframe
restyle looks like a full img2img shot. Each now leaves a trace in
``nomorals.media.render_failures`` while returning exactly what it
returned before.
"""

import shutil
import subprocess
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

from nomorals.media import fidelity, genedit
from nomorals.media.render_failures import (
    clear_failures,
    failure_count,
    format_failures,
    recent_failures,
    record_failure,
)


@pytest.fixture(autouse=True)
def _clean_ledger():
    clear_failures()
    yield
    clear_failures()


def _sites():
    return [f.site for f in recent_failures(100)]


# ── ledger basics ──────────────────────────────────────────────────────────

def test_record_recent_format_and_clear():
    record_failure("test.site", ValueError("boom"))
    entries = recent_failures()
    assert len(entries) == 1
    assert entries[0].site == "test.site"
    assert entries[0].exc_type == "ValueError"
    assert entries[0].message == "boom"
    assert "test.site: ValueError: boom" in format_failures()
    assert failure_count() == 1
    clear_failures()
    assert recent_failures() == []
    assert failure_count() == 0


def test_ring_bound_keeps_200_and_count_keeps_total():
    for i in range(250):
        record_failure(f"test.site{i % 3}", RuntimeError(f"n{i}"))
    assert len(recent_failures(1000)) == 200
    assert failure_count() == 250


def test_unprintable_exception_is_safe():
    class BadStr(Exception):
        def __str__(self):
            raise RuntimeError("no str for you")

    record_failure("test.bad", BadStr())
    entry = recent_failures()[0]
    assert entry.exc_type == "BadStr"
    assert "unprintable" in entry.message


def test_format_empty_state():
    assert format_failures() == "no recorded render failures"


def test_direct_import_surface():
    import nomorals.media.render_failures as rf

    assert callable(rf.record_failure)
    assert callable(rf.recent_failures)
    assert callable(rf.failure_count)
    assert callable(rf.clear_failures)
    assert callable(rf.format_failures)


def test_fidelity_all_unchanged():
    assert fidelity.__all__ == [
        "QualityLadder", "QUALITY_LADDER", "PostStep", "FidelityPipeline",
        "pick_best_model", "route", "POST_STEPS",
    ]


# ── fidelity sites ─────────────────────────────────────────────────────────

def test_card_quality_malformed_records_and_demotes():
    # A present-but-unparseable quality is a data failure, not the
    # designed "no quality attr" path (that is the hasattr check).
    # NOTE: the attr loop serves object cards only — dict cards read
    # "quality" directly with no handler, so the pins use objects.
    from types import SimpleNamespace

    assert fidelity._card_quality(SimpleNamespace(quality="high")) == 0.0
    assert _sites() == ["fidelity._card_quality"]


def test_card_quality_malformed_falls_to_score_but_still_records():
    from types import SimpleNamespace

    assert fidelity._card_quality(SimpleNamespace(quality="high", score=0.9)) == 0.9
    assert _sites() == ["fidelity._card_quality"]


def test_card_quality_clean_records_nothing():
    from types import SimpleNamespace

    assert fidelity._card_quality(SimpleNamespace(quality=0.7)) == 0.7
    assert fidelity._card_quality(SimpleNamespace(score=0.4)) == 0.4
    assert fidelity._card_quality(SimpleNamespace()) == 0.0
    assert _sites() == []


class _BrokerSelectFails:
    def select(self, task):
        raise RuntimeError("broker down")

    def cards(self):
        return [{"id": "m1", "capabilities": ["t"], "quality": 0.5}]


def test_route_broker_select_failure_records_and_falls_to_cards():
    res = fidelity.route("t", _BrokerSelectFails())
    assert res["model_id"] == "m1"
    assert res["source"] == "registry"
    assert _sites() == ["fidelity.route:broker_select"]


class _BrokerSelectWorks:
    def select(self, task):
        return {"id": "m0", "capabilities": ["t"], "quality": 0.8}

    def cards(self):
        return [{"id": "m1", "capabilities": ["t"], "quality": 0.5}]


def test_route_broker_select_ok_records_nothing():
    res = fidelity.route("t", _BrokerSelectWorks())
    assert res["model_id"] == "m0"
    assert res["source"] == "broker"
    assert _sites() == []


class _CardsFails:
    def cards(self):
        raise RuntimeError("registry down")


def test_route_registry_cards_failure_records_and_reports_no_candidate():
    res = fidelity.route("t", _CardsFails())
    assert res["model_id"] is None
    assert res["reason"] == "no candidate serves this task at the quality bar"
    assert _sites() == ["fidelity.route:registry_cards"]


# ── genedit sites ──────────────────────────────────────────────────────────

def _install_stub(name, module):
    saved = sys.modules.get(name)
    sys.modules[name] = module
    return saved


def _restore_stub(name, saved):
    if saved is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = saved


def test_structure_prompt_failure_records_and_returns_plain():
    stub = types.ModuleType("nomorals.media.directed.prompt_engine")

    def _boom(*args, **kwargs):
        raise RuntimeError("prompt engine down")

    stub.structure = _boom
    saved = _install_stub("nomorals.media.directed.prompt_engine", stub)
    try:
        assert genedit._structure_prompt("hello", "ltx", 5.0) == ("hello", "")
    finally:
        _restore_stub("nomorals.media.directed.prompt_engine", saved)
    assert _sites() == ["genedit._structure_prompt"]


def _fake_subprocess(monkeypatch, tmp_path):
    """Fake ffmpeg: extraction creates 8 frames, reassembly is a no-op."""
    calls = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if "-vf" in args:
            out_pat = [a for a in args if a.endswith("f_%04d.png")][0]
            frames_dir = Path(out_pat).parent
            frames_dir.mkdir(parents=True, exist_ok=True)
            for i in range(8):
                (frames_dir / f"f_{i:04d}.png").write_bytes(b"fake-frame")
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_video_to_shot_imgpipe_import_failure_records_and_passthrough(tmp_path, monkeypatch):
    # A stub parent makes `from .imggen import pipeline` raise on import
    # — the import failure, not the designed "no img2img attr" path
    # (that is the hasattr check).
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    monkeypatch.setattr(genedit, "_ffmpeg", lambda: "/usr/bin/ffmpeg")
    _fake_subprocess(monkeypatch, tmp_path)
    saved_parent = sys.modules.get("nomorals.media.imggen")
    saved_pipe = sys.modules.get("nomorals.media.imggen.pipeline")
    sys.modules["nomorals.media.imggen"] = types.ModuleType("nomorals.media.imggen")
    sys.modules.pop("nomorals.media.imggen.pipeline", None)
    try:
        shot = genedit.video_to_shot(str(video), "a cat", workdir=str(tmp_path / "wd"))
    finally:
        _restore_stub("nomorals.media.imggen", saved_parent)
        _restore_stub("nomorals.media.imggen.pipeline", saved_pipe)
    assert shot.backend == "passthrough"
    assert _sites() == ["genedit.video_to_shot:imgpipe_import"]


def test_video_to_shot_keyframe_failure_records_but_still_reports_img2img(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    monkeypatch.setattr(genedit, "_ffmpeg", lambda: "/usr/bin/ffmpeg")
    _fake_subprocess(monkeypatch, tmp_path)
    import nomorals.media.imggen  # noqa: F401 — real parent, stub pipeline below

    calls = {"n": 0}
    stub = types.ModuleType("nomorals.media.imggen.pipeline")

    def _img2img(src, prompt, strength=None, out_path=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("img2img down")
        shutil.copy(src, out_path)

    stub.img2img = _img2img
    saved_pipe = _install_stub("nomorals.media.imggen.pipeline", stub)
    try:
        shot = genedit.video_to_shot(str(video), "a cat", workdir=str(tmp_path / "wd"))
    finally:
        _restore_stub("nomorals.media.imggen.pipeline", saved_pipe)
    # The caller still sees a full img2img shot — that invisibility is
    # exactly why the per-frame failure is recorded.
    assert shot.backend == "img2img"
    assert _sites() == ["genedit.video_to_shot:keyframe_img2img"]
    styled = tmp_path / "wd" / "styled"
    assert (styled / "s_0000.png").exists()


# ── scan pin ───────────────────────────────────────────────────────────────

def test_scan_remainder_empty_suppression_off():
    from nomorals.tools import error_scan as es

    orig = es._noqa_suppresses
    es._noqa_suppresses = lambda line, rule: False
    root = Path(__file__).resolve().parents[1]
    try:
        report = es.scan([
            str(root / "nomorals/media/fidelity.py"),
            str(root / "nomorals/media/genedit.py"),
            str(root / "nomorals/media/render_failures.py"),
        ])
    finally:
        es._noqa_suppresses = orig
    assert report.findings == []

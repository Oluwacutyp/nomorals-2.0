"""T1.6: the GoogleFlow video route takes its connector factory injected."""

from __future__ import annotations

import pytest

import nomorals.media_edit.video_models as vm
from nomorals.media_edit.video_models import VideoModelError, generate_video


class _FakeFlowConn:
    def generate_video(self, prompt, **kw):
        return {"video_bytes": b"FAKEMP4"}


def test_googleflow_route_uses_injected_factory(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    out = generate_video("a cat", budget="hero", profile="termux",
                         out_dir=str(tmp_path), confirmed=True,
                         connector_factory=lambda: _FakeFlowConn())
    assert out.exists()
    assert out.read_bytes() == b"FAKEMP4"


def test_googleflow_route_without_factory_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    old = vm._CONNECTOR_FACTORY
    vm._CONNECTOR_FACTORY = None
    try:
        with pytest.raises(VideoModelError, match="no connector factory"):
            generate_video("a cat", budget="hero", profile="termux",
                           out_dir=str(tmp_path), confirmed=True)
    finally:
        vm._CONNECTOR_FACTORY = old

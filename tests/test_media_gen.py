"""Tests for the media generation spine tools (nomorals/tools/media_gen)."""

import pytest

from nomorals.tools.registry import ToolRegistry


@pytest.fixture(scope="module")
def registry():
    r = ToolRegistry()
    r.register_builtins()
    return r


def test_all_six_registered(registry):
    for name in ("image_generate", "image_variation", "image_inpaint",
                 "image_outpaint", "image_batch", "video_generate",
                 "video_animate"):
        assert name in registry._tools, f"{name} not registered"


def test_style_preset_applied():
    from nomorals.tools.media_gen import _apply_style_preset
    out = _apply_style_preset("a cat", "cinematic")
    assert out.startswith("a cat")
    assert "cinematic" in out


def test_unknown_style_falls_back_to_raw():
    from nomorals.tools.media_gen import _apply_style_preset
    assert _apply_style_preset("a cat", "nonexistent") == "a cat"


def test_image_generate_rejects_empty_prompt(registry):
    tool = registry._tools["image_generate"]
    with pytest.raises(Exception):
        tool.func("", save_to="/tmp")


def test_video_generate_rejects_empty_prompt(registry):
    tool = registry._tools["video_generate"]
    with pytest.raises(Exception):
        tool.func("")


def test_image_batch_rejects_empty_list(registry):
    tool = registry._tools["image_batch"]
    with pytest.raises(Exception):
        tool.func([])

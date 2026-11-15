"""Tests for the one-call poster pipeline (nomorals/media/poster.py).

The question (queue E-4): Devon has generated art (imggen Studio) and
real PIL typography (upgrade.typography.composite_text), but no single
call ever wired them into one finished poster — generated art as the
background with exact typography composited on top. These tests pin
that pipeline: one call, prompt + text spec in, a real poster PNG out.

The generation half is injected in tests (a deterministic PIL gradient
is real image data, not a stub of the pipeline — the generate →
composite → save path under test is the production path). The default
generate is the real Studio; with no torch/checkpoint it fails
honestly instead of faking art.
"""

from __future__ import annotations

import pytest
from PIL import Image


def _gradient_art(path, size=(360, 640)):
    """Deterministic stand-in art: a real vertical-gradient PNG.

    The pipeline under test is generate → composite → save; only the
    art source is swapped, exactly like upgrade.run_pipeline's
    caller-supplied ``generate`` hook.
    """
    img = Image.new("RGB", size)
    px = img.load()
    w, h = size
    for y in range(h):
        r = 20 + (y * 40 // h)
        g = 10 + (y * 30 // h)
        b = 80 + (y * 120 // h)
        for x in range(w):
            px[x, y] = (r, g, b)
    img.save(path)
    return str(path)


def _fake_generate_factory(tmp_path, seen=None):
    """A generate callable with Studio.generate's shape: prompt +
    kwargs in, [saved art path] out."""
    def fake_generate(prompt, **kwargs):
        if seen is not None:
            seen["prompt"] = prompt
            seen["kwargs"] = dict(kwargs)
        art = _gradient_art(tmp_path / "art.png")
        return [art]
    return fake_generate


def _differing_pixels(a_path, b_path):
    """Count of pixels that differ between two same-size images."""
    with Image.open(a_path) as a, Image.open(b_path) as b:
        assert a.size == b.size
        a_rgb = a.convert("RGB")
        b_rgb = b.convert("RGB")
        return sum(1 for x, y in zip(a_rgb.getdata(), b_rgb.getdata(),
                                     strict=True)
                   if x != y)


# ---------------------------------------------------------------------------
# Core pipeline: make_poster
# ---------------------------------------------------------------------------

def test_make_poster_end_to_end(tmp_path):
    from nomorals.media.poster import make_poster

    seen: dict = {}
    out = make_poster(
        "a Lagos owambe at night, string lights",
        [{"text": "OWAMBE", "position": "center", "size": 96},
         {"text": "Saturday 8 PM", "position": "bottom", "size": 48}],
        generate=_fake_generate_factory(tmp_path, seen),
        save_to=str(tmp_path))
    assert seen["prompt"] == "a Lagos owambe at night, string lights"
    assert seen["kwargs"].get("aspect_ratio") == "9:16"
    poster_path = out["path"]
    assert poster_path.endswith(".png")
    with Image.open(poster_path) as img:
        assert img.size == (360, 640)
        assert img.format == "PNG"
    # The poster is the art WITH text: pixels differ from the bare art,
    # and the art file itself is never mutated by the composite.
    assert _differing_pixels(out["art_path"], poster_path) > 100
    assert out["prompt"] == "a Lagos owambe at night, string lights"
    assert [t["text"] for t in out["texts"]] == ["OWAMBE", "Saturday 8 PM"]
    # Sidecar prompt for reproducibility, mirroring Studio._finalize.
    sidecar = poster_path + ".txt"
    with open(sidecar) as f:
        body = f.read()
    assert "a Lagos owambe at night" in body
    assert "OWAMBE" in body


def test_make_poster_string_layers_get_defaults(tmp_path):
    from nomorals.media.poster import make_poster

    out = make_poster(
        "a market at dawn",
        ["MARKET DAY"],
        generate=_fake_generate_factory(tmp_path),
        save_to=str(tmp_path))
    assert len(out["texts"]) == 1
    assert out["texts"][0]["text"] == "MARKET DAY"
    assert out["texts"][0]["position"] == "bottom"
    assert out["texts"][0]["size"] == 48
    assert _differing_pixels(out["art_path"], out["path"]) > 100


def test_make_poster_single_textcomposite_and_single_string(tmp_path):
    from nomorals.media.poster import make_poster
    from nomorals.media.upgrade.typography import TextComposite

    out = make_poster(
        "a beach at noon",
        TextComposite(text="BEACH", position="top", size=72),
        generate=_fake_generate_factory(tmp_path),
        save_to=str(tmp_path))
    assert out["texts"][0]["position"] == "top"
    out2 = make_poster(
        "a beach at noon",
        "BEACH DAY",
        generate=_fake_generate_factory(tmp_path),
        save_to=str(tmp_path))
    assert out2["texts"][0]["text"] == "BEACH DAY"


def test_make_poster_save_to_file_path(tmp_path):
    from nomorals.media.poster import make_poster

    target = tmp_path / "my-poster.png"
    out = make_poster(
        "a mountain",
        ["SUMMIT"],
        generate=_fake_generate_factory(tmp_path),
        save_to=str(target))
    assert out["path"] == str(target)
    assert target.is_file()


def test_make_poster_rejects_empty_prompt(tmp_path):
    from nomorals.media.poster import make_poster

    with pytest.raises(ValueError, match="prompt"):
        make_poster("", ["TITLE"],
                    generate=_fake_generate_factory(tmp_path),
                    save_to=str(tmp_path))


def test_make_poster_rejects_no_text(tmp_path):
    from nomorals.media.poster import make_poster

    with pytest.raises(ValueError, match="text"):
        make_poster("a mountain", [],
                    generate=_fake_generate_factory(tmp_path),
                    save_to=str(tmp_path))


def test_make_poster_seed_and_steps_reach_generate(tmp_path):
    from nomorals.media.poster import make_poster

    seen: dict = {}
    make_poster(
        "a mountain", ["SUMMIT"], seed=7, steps=25,
        generate=_fake_generate_factory(tmp_path, seen),
        save_to=str(tmp_path))
    assert seen["kwargs"].get("seed") == 7
    assert seen["kwargs"].get("steps") == 25


def test_make_poster_generation_failure_is_honest(tmp_path):
    """A failing generate is never papered over with fake art."""
    from nomorals.media.imggen import ImgGenError
    from nomorals.media.poster import make_poster

    def dying_generate(prompt, **kwargs):
        raise ImgGenError("native image generation needs torch")

    with pytest.raises(ImgGenError, match="torch"):
        make_poster("a mountain", ["SUMMIT"], generate=dying_generate,
                    save_to=str(tmp_path))


def test_make_poster_default_generate_fails_honestly_without_torch(
        tmp_path, monkeypatch):
    """The real path (no injected generate) fails honestly on a box
    with no torch — never a fake poster."""
    from nomorals.media.imggen import ImgGenError
    from nomorals.media.poster import make_poster

    monkeypatch.setattr("nomorals.media.imggen.studio.TORCH_AVAILABLE",
                        False)
    with pytest.raises(ImgGenError):
        make_poster("a mountain", ["SUMMIT"], save_to=str(tmp_path))


# ---------------------------------------------------------------------------
# Tool action: poster_generate
# ---------------------------------------------------------------------------

def test_poster_tool_registered():
    from nomorals.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register_builtins()
    assert "poster_generate" in registry._tools


def test_poster_tool_wires_prompt_style_and_out_dir(
        tmp_path, monkeypatch):
    """The tool applies the style preset and hands the pipeline its
    out dir — the composition itself is proven by the core tests."""
    from types import SimpleNamespace

    from nomorals.tools.registry import ToolRegistry

    seen: dict = {}

    def fake_make_poster(prompt, texts, **kwargs):
        seen.update(prompt=prompt, texts=texts, kwargs=kwargs)
        return {"path": str(tmp_path / "poster.png"), "art_path": "",
                "prompt": prompt, "texts": [], "aspect": "9:16",
                "size": (360, 640)}

    monkeypatch.setattr("nomorals.media.poster.make_poster",
                        fake_make_poster)
    registry = ToolRegistry(
        context=SimpleNamespace(workspace=str(tmp_path)))
    registry.register_builtins()
    tool = registry._tools["poster_generate"]
    out = tool.fn(
        "a cat", [{"text": "CAT", "position": "center", "size": 96}],
        style="cinematic")
    assert "cinematic" in seen["prompt"]
    assert seen["prompt"].startswith("a cat")
    assert seen["texts"][0]["text"] == "CAT"
    assert seen["kwargs"]["save_to"].endswith("generated/images")
    assert out["path"] == str(tmp_path / "poster.png")


def test_poster_tool_rejects_empty_prompt(tmp_path):
    from types import SimpleNamespace

    from nomorals.tools.registry import ToolRegistry

    registry = ToolRegistry(
        context=SimpleNamespace(workspace=str(tmp_path)))
    registry.register_builtins()
    tool = registry._tools["poster_generate"]
    with pytest.raises(ValueError, match="prompt"):
        tool.fn("", ["TITLE"])


# ---------------------------------------------------------------------------
# Chat command: /poster
# ---------------------------------------------------------------------------

def _chat_mixin():
    from types import SimpleNamespace

    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    class M(RuntimeMediaMixin):
        def __init__(self):
            self.context = SimpleNamespace()

    return M()


def test_poster_chat_empty_usage():
    out = _chat_mixin()._control_poster("")
    assert "/poster" in out
    assert "::" in out


def test_poster_chat_parses_prompt_title_and_footer(
        tmp_path, monkeypatch):
    seen: dict = {}

    def fake_make_poster(prompt, texts, **kwargs):
        seen.update(prompt=prompt, texts=texts, kwargs=kwargs)
        return {"path": str(tmp_path / "poster.png"), "art_path": "",
                "prompt": prompt, "texts": texts, "aspect": "9:16",
                "size": (360, 640)}

    monkeypatch.setattr("nomorals.media.poster.make_poster",
                        fake_make_poster)
    out = _chat_mixin()._control_poster(
        "a Lagos owambe at night :: OWAMBE :: Saturday 8 PM --ar 9:16")
    assert seen["prompt"] == "a Lagos owambe at night"
    assert seen["texts"][0]["text"] == "OWAMBE"
    assert seen["texts"][0]["position"] == "center"
    assert seen["texts"][0]["size"] == 96
    assert seen["texts"][1]["text"] == "Saturday 8 PM"
    assert seen["texts"][1]["position"] == "bottom"
    assert seen["texts"][1]["size"] == 48
    assert seen["kwargs"].get("aspect") == "9:16"
    assert str(tmp_path / "poster.png") in out


def test_poster_chat_position_prefix(tmp_path, monkeypatch):
    seen: dict = {}

    def fake_make_poster(prompt, texts, **kwargs):
        seen.update(texts=texts)
        return {"path": str(tmp_path / "poster.png"), "art_path": "",
                "prompt": prompt, "texts": texts, "aspect": "9:16",
                "size": (360, 640)}

    monkeypatch.setattr("nomorals.media.poster.make_poster",
                        fake_make_poster)
    _chat_mixin()._control_poster("a mountain :: top:SUMMIT")
    assert seen["texts"][0]["position"] == "top"
    assert seen["texts"][0]["text"] == "SUMMIT"


def test_poster_chat_missing_text_returns_usage(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("make_poster must not run without text")

    monkeypatch.setattr("nomorals.media.poster.make_poster", boom)
    out = _chat_mixin()._control_poster("just an art prompt")
    assert "/poster" in out
    assert "usage" in out.lower() or "::" in out


def test_poster_chat_never_raises(tmp_path, monkeypatch):
    from nomorals.media.imggen import ImgGenError

    def dying_generate(self, prompt, **kwargs):
        raise ImgGenError("native image generation needs torch")

    monkeypatch.setattr(
        "nomorals.media.imggen.studio.Studio.generate", dying_generate)
    out = _chat_mixin()._control_poster("a mountain :: SUMMIT")
    assert "poster failed" in out.lower()

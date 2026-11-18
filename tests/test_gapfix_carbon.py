"""52-websites #19 (Carbon): code, photographed beautifully.

The organ renders code to a presentation image from Pygments'
own token stream — the highlighter is the authority, never a
hand-rolled approximation. These tests prove the contract:

- highlighted output for a fixture snippet matches the
  highlighter's tokenisation exactly (token type + text, from
  Pygments' own lexer, called independently here);
- golden renders per theme: deterministic (same card, same PNG
  bytes, same hash) and distinct per theme, with the style's
  own background colour at a known pixel;
- chrome (traffic-light dots, title bar) and the brand stamp
  are present pixels, not asserted labels;
- the card lands as a Studio image-document, and Studio work
  continues from it (a second op replays);
- the brain tool renders through the Studio, and a verified
  dev answer offers the card while an unverified one does not.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest
from PIL import Image
from pygments import lex as _pyg_lex
from pygments.lexers import get_lexer_by_name

from nomorals.media.imggen.codecard import (
    MAX_CODE_LINES,
    CodeCard,
    CodeCardError,
    list_themes,
    render_card,
    render_code_image,
    resolve_language,
    tokenize,
)
from nomorals.media.imggen.document import ImageDocument
from nomorals.media.imggen.studio import Studio
from nomorals.research.dev_answer import format_dev_answer_reply
from nomorals.tools import codecard as codecard_tool

PY_SNIPPET = (
    'def greet(name):\n'
    '    # say hello\n'
    '    return f"Hello, {name}!"\n'
)

JS_SNIPPET = (
    'const add = (a, b) => {\n'
    '  // sum two numbers\n'
    '  return a + b;\n'
    '};\n'
)


class _Registry:
    def __init__(self, context):
        self.context = context
        self.handlers = {}

    def register(self, name, **meta):
        def deco(fn):
            self.handlers[name] = fn
            return fn
        return deco


class _Ctx:
    def __init__(self, home, workspace):
        self.settings = _Settings(home)
        self.workspace = str(workspace)


class _Settings:
    def __init__(self, home):
        self.home = str(home)
        self.workspace_dir = str(home)


def _png_bytes(render) -> bytes:
    buf = io.BytesIO()
    render.image.save(buf, format="PNG")
    return buf.getvalue()


def _outer_pixel(render, x: int, y: int):
    return render.image.getpixel((x, y))


# ---------------------------------------------------------------------------
# Tokenisation: the highlighter is the authority
# ---------------------------------------------------------------------------

def test_python_tokens_match_pygments_own_lexer():
    """The organ's token stream IS Pygments' token stream."""
    mine = tokenize(PY_SNIPPET, "python")
    theirs = [(str(t), text)
              for t, text in _pyg_lex(PY_SNIPPET,
                                      get_lexer_by_name("python"))]
    assert mine == theirs
    assert ("Token.Keyword", "def") in mine
    assert ("Token.Comment.Single", "# say hello") in mine
    assert ("Token.Name.Function", "greet") in mine


def test_javascript_tokens_match_pygments_own_lexer():
    mine = tokenize(JS_SNIPPET, "javascript")
    theirs = [(str(t), text)
              for t, text in _pyg_lex(JS_SNIPPET,
                                      get_lexer_by_name("javascript"))]
    assert mine == theirs
    assert ("Token.Keyword.Declaration", "const") in mine


def test_language_alias_resolves_to_the_same_lexer():
    lexer, name, plain, _note = resolve_language("py")
    assert name == "Python"
    assert plain is False
    assert tokenize(PY_SNIPPET, "py") == tokenize(PY_SNIPPET, "python")


def test_unknown_language_is_plain_text_and_named():
    lexer, name, plain, note = resolve_language("xyzlang")
    assert plain is True
    assert name == "text"
    assert "xyzlang" in note
    render = render_code_image(PY_SNIPPET, language="xyzlang")
    assert any("xyzlang" in n for n in render.notes)
    assert render.language == "text"


def test_error_token_wears_its_style_background():
    """A style-set token background is painted, not dropped:
    monokai gives Error tokens the #1e0010 wash."""
    render = render_code_image("x = `\n", language="python",
                               theme="monokai")
    band = render.image.crop(
        (0, 28 + 40 + 16, render.width, 28 + 40 + 16 + 21))
    assert (0x1E, 0x00, 0x10) in set(band.getdata())


def test_unknown_theme_falls_back_to_dracula_and_says_so():
    render = render_code_image(PY_SNIPPET, theme="no-such-theme")
    assert render.theme == "dracula"
    assert render.style == "dracula"
    assert any("no-such-theme" in n for n in render.notes)


# ---------------------------------------------------------------------------
# Golden renders: deterministic per theme, distinct across themes
# ---------------------------------------------------------------------------

def test_same_card_renders_the_same_png_bytes():
    first = render_code_image(PY_SNIPPET, language="python",
                              theme="dracula", title="greet.py")
    second = render_code_image(PY_SNIPPET, language="python",
                               theme="dracula", title="greet.py")
    assert first.render_hash == second.render_hash
    assert _png_bytes(first) == _png_bytes(second)
    assert hashlib.sha256(_png_bytes(first)).hexdigest() \
        == first.render_hash


def test_every_theme_renders_and_differs():
    renders = {th["name"]: render_code_image(
        PY_SNIPPET, language="python", theme=th["name"])
        for th in list_themes()}
    assert len(list_themes()) == 8
    hashes = {r.render_hash for r in renders.values()}
    assert len(hashes) == 8, "two themes rendered identically"
    for name, render in renders.items():
        assert render.theme == name
        assert render.lines == 3
        assert render.tokens > 0


def test_code_pane_background_is_the_styles_own():
    """A known pixel inside the code pane is the style's bg."""
    dracula = render_code_image(PY_SNIPPET, language="python",
                                theme="dracula")
    light = render_code_image(PY_SNIPPET, language="python",
                              theme="solarized-light")
    # (40, 90) sits inside the code pane for this snippet.
    assert _outer_pixel(dracula, 40, 90) == (0x28, 0x2A, 0x36)
    assert _outer_pixel(light, 40, 90) == (0xFD, 0xF6, 0xE3)


def test_render_card_dataclass_renders():
    card = CodeCard(code=PY_SNIPPET, language="python",
                    theme="nord", title="greet.py")
    render = render_card(card)
    assert render.theme == "nord"
    assert render.font == "JetBrains Mono"
    assert render.lexer == "Python"


# ---------------------------------------------------------------------------
# Chrome, watermark, line numbers
# ---------------------------------------------------------------------------

def test_traffic_light_dots_are_present_pixels():
    render = render_code_image(PY_SNIPPET, language="python",
                               theme="dracula", title="greet.py")
    # Dot centres: outer_pad 28 + 22 + i*22, title bar centre y.
    assert _outer_pixel(render, 50, 48) == (0xFF, 0x5F, 0x57)
    assert _outer_pixel(render, 72, 48) == (0xFE, 0xBC, 0x2E)
    assert _outer_pixel(render, 94, 48) == (0x28, 0xC8, 0x40)


def test_chrome_off_removes_the_title_bar():
    with_chrome = render_code_image(PY_SNIPPET, language="python",
                                    chrome=True)
    without = render_code_image(PY_SNIPPET, language="python",
                                chrome=False)
    assert without.height == with_chrome.height - 40
    assert _outer_pixel(without, 50, 48) != (0xFF, 0x5F, 0x57)


def test_watermark_stamp_is_present_and_removable():
    stamped = render_code_image(PY_SNIPPET, language="python",
                                watermark=True)
    plain = render_code_image(PY_SNIPPET, language="python",
                              watermark=False)
    assert stamped.height == plain.height + 26
    # The footer band of the stamped card is not blank backdrop.
    footer = stamped.image.crop(
        (0, stamped.height - 26, stamped.width, stamped.height))
    pixels = set(footer.getdata())
    assert len(pixels) > 1, "watermark band is blank"


def test_line_numbers_widen_the_card():
    numbered = render_code_image(PY_SNIPPET, language="python",
                                 line_numbers=True)
    plain = render_code_image(PY_SNIPPET, language="python",
                              line_numbers=False)
    assert numbered.width > plain.width


def test_long_line_wraps_without_losing_chars():
    long_code = "x = '" + "a" * 200 + "'\n"
    render = render_code_image(long_code, language="python")
    # One 200-char line wraps to 2 visual lines at 120 columns.
    assert render.lines == 1
    assert render.height > render_code_image(
        "x = 'a'\n", language="python").height
    # The wrapped card is no wider than a 120-char card.
    cap = render_code_image("x = '" + "a" * 116 + "'\n",
                            language="python")
    assert render.width == cap.width


# ---------------------------------------------------------------------------
# Refusals name the limit
# ---------------------------------------------------------------------------

def test_empty_code_is_refused():
    with pytest.raises(CodeCardError, match="empty"):
        render_code_image("   \n  ")


def test_line_cap_is_refused_by_name():
    code = "\n".join(f"line {i}" for i in range(MAX_CODE_LINES + 1))
    with pytest.raises(CodeCardError, match=str(MAX_CODE_LINES)):
        render_code_image(code)


def test_font_size_bounds_are_refused():
    with pytest.raises(CodeCardError, match="font_size"):
        render_code_image(PY_SNIPPET, font_size=9)
    with pytest.raises(CodeCardError, match="font_size"):
        render_code_image(PY_SNIPPET, font_size=29)


# ---------------------------------------------------------------------------
# The Studio document flow
# ---------------------------------------------------------------------------

def test_code_card_document_is_a_real_document(tmp_path):
    studio = Studio()
    doc = studio.code_card_document(
        PY_SNIPPET, language="python", theme="dracula",
        title="greet.py", save_to=str(tmp_path))
    assert isinstance(doc, ImageDocument)
    assert doc.source_prompt == PY_SNIPPET
    assert "codecard" in doc.source_rung
    assert "dracula" in doc.source_rung
    assert any("render_hash" in n for n in doc.notes)
    assert Path(doc.source).exists()
    replayed = doc.render()
    assert replayed.image.size == Image.open(doc.source).size


def test_code_card_renders_and_saves_the_document(tmp_path):
    studio = Studio()
    result = studio.code_card(
        PY_SNIPPET, language="python", theme="monokai",
        title="greet.py", save_to=str(tmp_path))
    assert Path(result["paths"][0]).exists()
    assert Path(result["document"]).exists()
    assert result["theme"] == "monokai"
    assert result["language"] == "Python"
    assert result["font"] == "JetBrains Mono"
    assert len(result["render_hash"]) == 64
    loaded = ImageDocument.load(result["document"])
    assert loaded.source_prompt == PY_SNIPPET
    # The delivered PNG is the organ's pixels, replayed.
    organ = render_code_image(PY_SNIPPET, language="python",
                              theme="monokai", title="greet.py")
    delivered = Image.open(result["paths"][0]).convert("RGB")
    assert list(delivered.getdata()) == list(
        organ.image.convert("RGB").getdata())


def test_code_card_document_continues_with_a_second_op(tmp_path):
    """Studio work continues from the card: a second op replays."""
    studio = Studio()
    doc = studio.code_card_document(
        PY_SNIPPET, language="python", theme="dracula",
        save_to=str(tmp_path))
    doc.blur(radius=2.0)
    paths = studio.render_document(doc, save_to=str(tmp_path))
    assert Path(paths[0]).exists()
    assert Image.open(paths[0]).size == Image.open(doc.source).size


# ---------------------------------------------------------------------------
# The brain tool
# ---------------------------------------------------------------------------

def _tool(tmp_path):
    reg = _Registry(_Ctx(tmp_path, tmp_path))
    codecard_tool.register(reg)
    return reg.handlers["code_card"]


def test_tool_renders_through_the_studio(tmp_path):
    fn = _tool(tmp_path)
    result = fn(action="render", code=PY_SNIPPET, language="python",
                theme="dracula", title="greet.py")
    assert result["ok"] is True
    assert Path(result["paths"][0]).exists()
    assert Path(result["document"]).exists()
    assert result["render_hash"]


def test_tool_reads_code_from_a_file(tmp_path):
    src = tmp_path / "snippet.py"
    src.write_text(PY_SNIPPET, encoding="utf-8")
    fn = _tool(tmp_path)
    result = fn(action="render", file=str(src), language="python")
    assert result["ok"] is True
    assert result["language"] == "Python"


def test_tool_themes_and_status(tmp_path):
    fn = _tool(tmp_path)
    themes = fn(action="themes")
    assert themes["ok"] is True
    assert len(themes["themes"]) == 8
    assert themes["default"] == "dracula"
    status = fn(action="status")
    assert status["ok"] is True
    assert status["pygments"] is True
    assert status["font"] == "JetBrains Mono"


def test_tool_refuses_empty_code_and_unknown_action(tmp_path):
    fn = _tool(tmp_path)
    assert fn(action="render", code="  ")["ok"] is False
    assert fn(action="explode")["ok"] is False
    assert fn(action="render",
              file="/no/such/file.py")["ok"] is False


# ---------------------------------------------------------------------------
# The offer after a code fix lands
# ---------------------------------------------------------------------------

def _answer(verified: bool) -> dict:
    return {
        "ok": True,
        "answer": "Use a list comprehension.",
        "code_blocks": [{
            "code": "[x * 2 for x in xs]",
            "language": "python",
            "verified_by_execution": verified,
            "verification": {"seconds": 0.12} if verified
                            else {"reason": "no runner"},
        }],
        "citations": [],
    }


def test_verified_answer_offers_the_card():
    reply = format_dev_answer_reply(_answer(True))
    assert "✅ verified" in reply
    assert 'screenshot this code' in reply


def test_unverified_answer_offers_nothing():
    reply = format_dev_answer_reply(_answer(False))
    assert "⚠️ not verified" in reply
    assert "screenshot this code" not in reply


def test_answer_without_code_offers_nothing():
    reply = format_dev_answer_reply({
        "ok": True, "answer": "No code here.",
        "code_blocks": [], "citations": []})
    assert "screenshot this code" not in reply


# ---------------------------------------------------------------------------
# The Ray.so preset pack (52-websites #20, folded into §19)
# ---------------------------------------------------------------------------

def test_ray_preset_renders_deterministically():
    from nomorals.media.imggen.codecard import list_presets

    first = render_code_image(PY_SNIPPET, language="python",
                              preset="ray")
    second = render_code_image(PY_SNIPPET, language="python",
                               preset="ray")
    assert first.render_hash == second.render_hash
    assert first.preset == "ray"
    assert first.theme == "dracula"
    assert first.gradient == ("#2b2d5c", "#14152a")
    assert first.notes == []
    presets = {p["name"]: p for p in list_presets()}
    assert sorted(presets) == ["carbon", "ray", "ray-light",
                               "ray-midnight"]
    assert presets["ray"]["theme"] == "dracula"
    assert presets["carbon"]["theme"] is None


def test_ray_preset_fixes_its_knobs_and_names_overrides():
    render = render_code_image(
        PY_SNIPPET, language="python", preset="ray", theme="nord",
        chrome=False, font_size=20)
    assert render.theme == "dracula"
    assert render.preset == "ray"
    assert any("fixes theme to 'dracula'" in n
               for n in render.notes)
    assert any("fixes chrome to True" in n for n in render.notes)
    assert any("fixes font_size to 15" in n for n in render.notes)
    # Chrome was forced back on: the title bar is there.
    carbon_plain = render_code_image(
        PY_SNIPPET, language="python", preset="carbon",
        chrome=False)
    assert render.height == carbon_plain.height + 40


def test_ray_light_is_a_light_frame_with_gradient():
    render = render_code_image(PY_SNIPPET, language="python",
                               preset="ray-light")
    assert render.theme == "solarized-light"
    assert render.gradient == ("#fdf6e3", "#e8e0c8")
    top = _outer_pixel(render, 2, 2)
    bottom = _outer_pixel(render, 2, render.height - 2)
    assert top != bottom
    assert all(c > 200 for c in top), "ray-light top must be light"


def test_ray_gradient_present_carbon_solid():
    ray = render_code_image(PY_SNIPPET, language="python",
                            preset="ray")
    assert _outer_pixel(ray, 2, 2) != _outer_pixel(
        ray, 2, ray.height - 2)
    carbon = render_code_image(PY_SNIPPET, language="python",
                               preset="carbon")
    assert carbon.gradient is None
    assert _outer_pixel(carbon, 2, 2) == _outer_pixel(
        carbon, 2, carbon.height - 2)


def test_ray_midnight_is_a_dark_frame():
    render = render_code_image(PY_SNIPPET, language="python",
                               preset="ray-midnight")
    assert render.theme == "one-dark"
    top = _outer_pixel(render, 2, 2)
    assert all(c < 100 for c in top), "midnight top must be dark"


def test_unknown_preset_is_refused_by_name():
    with pytest.raises(CodeCardError, match="unknown code preset"):
        render_code_image(PY_SNIPPET, preset="no-such-preset")
    with pytest.raises(CodeCardError, match="ray"):
        render_code_image(PY_SNIPPET, preset="no-such-preset")


def test_carbon_preset_honours_knobs():
    render = render_code_image(PY_SNIPPET, language="python",
                               preset="carbon", theme="nord",
                               chrome=False, font_size=20)
    assert render.theme == "nord"
    assert render.notes == []
    assert render.height == render_code_image(
        PY_SNIPPET, language="python", preset="carbon",
        theme="nord", font_size=20).height - 40


def test_preset_through_studio_and_tool(tmp_path):
    studio = Studio()
    result = studio.code_card(
        PY_SNIPPET, language="python", preset="ray-light",
        save_to=str(tmp_path))
    assert result["preset"] == "ray-light"
    assert result["theme"] == "solarized-light"
    assert Path(result["paths"][0]).exists()
    fn = _tool(tmp_path)
    presets = fn(action="presets")
    assert presets["ok"] is True
    assert len(presets["presets"]) == 4
    assert presets["default"] == "carbon"
    rendered = fn(action="render", code=PY_SNIPPET,
                  language="python", preset="ray")
    assert rendered["ok"] is True
    assert rendered["preset"] == "ray"
    assert rendered["theme"] == "dracula"
    refused = fn(action="render", code=PY_SNIPPET, preset="nope")
    assert refused["ok"] is False
    assert "unknown code preset" in refused["error"]

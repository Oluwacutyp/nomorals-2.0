"""Tests for nomorals/media_edit/edittext.py — all offline, no tesseract needed."""

import pytest
from PIL import Image, ImageDraw

from nomorals.media_edit import edittext
from nomorals.media_edit.edittext import TextRegion, _best_match, _fit_font_size
from nomorals.media_edit.edittext import _sample_text_color
from nomorals.media_edit.images import MediaEditError


def _fake_data(words):
    """Build a fake pytesseract image_to_data DICT from word specs.

    words: list of (text, l, t, w, h, conf)
    """
    n = len(words)
    return {
        "text": [w[0] for w in words],
        "left": [w[1] for w in words],
        "top": [w[2] for w in words],
        "width": [w[3] for w in words],
        "height": [w[4] for w in words],
        "conf": [w[5] for w in words],
        "block_num": [1] * n,
        "par_num": [1] * n,
        "line_num": [1 if i < 2 else 2 for i in range(n)],
    }


class _FakeTesseract:
    class Output:
        DICT = "dict"

    def __init__(self, data):
        self._data = data

    def image_to_data(self, image, lang="eng", output_type=None):
        return self._data


def _img(w=400, h=200, color="white"):
    return Image.new("RGB", (w, h), color)


# ── detect_text ──────────────────────────────────────────────────────────

def test_detect_groups_words_into_lines(monkeypatch):
    data = _fake_data([
        ("SALE", 10, 20, 80, 30, 95.0),
        ("TODAY", 95, 20, 90, 30, 93.0),
        ("50%", 10, 70, 60, 30, 90.0),
    ])
    monkeypatch.setattr(edittext, "_require_tesseract",
                        lambda: _FakeTesseract(data))
    regions = edittext.detect_text(_img())
    assert len(regions) == 2
    assert regions[0].text == "SALE TODAY"
    assert regions[0].box == (10, 20, 185, 50)
    assert regions[1].text == "50%"
    assert regions[0].confidence == pytest.approx(94.0)


def test_detect_skips_empty_and_low_conf(monkeypatch):
    data = _fake_data([
        ("", 10, 20, 80, 30, 95.0),
        ("OK", 10, 20, 40, 30, -1.0),
    ])
    monkeypatch.setattr(edittext, "_require_tesseract",
                        lambda: _FakeTesseract(data))
    assert edittext.detect_text(_img()) == []


def test_detect_never_raises_on_weird_image(monkeypatch):
    class Boom:
        class Output:
            DICT = "dict"

        def image_to_data(self, *a, **k):
            raise RuntimeError("tesseract exploded")
    monkeypatch.setattr(edittext, "_require_tesseract", lambda: Boom())
    assert edittext.detect_text(_img()) == []


def test_detect_missing_ocr_raises_pip_hint(monkeypatch):
    from nomorals.documents.errors import DocumentError
    def boom():
        raise DocumentError("OCR needs the optional 'pytesseract' package: "
                            "pip install pytesseract pdf2image")
    monkeypatch.setattr("nomorals.documents.ocr._require_pytesseract", boom)
    # edittext imports _require_pytesseract lazily inside _require_tesseract
    with pytest.raises(DocumentError, match="pip install"):
        edittext.detect_text(_img())


# ── _best_match ──────────────────────────────────────────────────────────

def _regions():
    return [
        TextRegion(box=(0, 0, 100, 30), text="SALE TODAY", confidence=94.0),
        TextRegion(box=(0, 50, 80, 30), text="50% OFF", confidence=90.0),
    ]


def test_best_match_exact():
    assert _best_match(_regions(), "SALE TODAY").text == "SALE TODAY"


def test_best_match_fuzzy():
    # "SALE" alone still matches "SALE TODAY" above threshold
    assert _best_match(_regions(), "sale").text == "SALE TODAY"


def test_best_match_picks_highest():
    regions = [TextRegion(box=(0, 0, 10, 10), text="SALE", confidence=90.0),
               TextRegion(box=(0, 0, 10, 10), text="SALE TODAY ONLY",
                          confidence=90.0)]
    assert _best_match(regions, "sale today").text == "SALE TODAY ONLY"


def test_best_match_below_threshold_returns_none():
    assert _best_match(_regions(), "elephant") is None
    assert _best_match(_regions(), "") is None
    assert _best_match([], "sale") is None


# ── replace_text error paths ─────────────────────────────────────────────

def test_replace_no_text_detected(monkeypatch):
    monkeypatch.setattr(edittext, "detect_text", lambda img, lang="eng": [])
    with pytest.raises(MediaEditError, match="no text detected"):
        edittext.replace_text(_img(), "SALE", "SOLD")


def test_replace_no_match_lists_found(monkeypatch):
    monkeypatch.setattr(edittext, "detect_text",
                        lambda img, lang="eng": _regions())
    with pytest.raises(MediaEditError, match="couldn't find 'xyz'") as ei:
        edittext.replace_text(_img(), "xyz", "SOLD")
    assert "SALE TODAY" in str(ei.value)
    assert "50% OFF" in str(ei.value)


def test_replace_needs_old_and_new():
    with pytest.raises(MediaEditError, match="needs the text to find"):
        edittext.replace_text(_img(), "", "SOLD")
    with pytest.raises(MediaEditError, match="needs the replacement"):
        edittext.replace_text(_img(), "SALE", "  ")


# ── replace_text happy path (mocked OCR + inpaint) ───────────────────────

def _white_with_black_text():
    img = Image.new("RGB", (400, 120), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([50, 30, 200, 80], fill="black")
    return img


def test_replace_end_to_end_pil_backend(monkeypatch):
    regions = [TextRegion(box=(50, 30, 200, 80), text="SALE",
                          confidence=96.0)]
    monkeypatch.setattr(edittext, "detect_text",
                        lambda img, lang="eng": regions)
    out, meta = edittext.replace_text(_white_with_black_text(), "SALE",
                                      "SOLD", inpaint_backend="pil")
    assert out.size == (400, 120)
    assert meta["found"] == "SALE"
    assert meta["inpaint_backend"] == "pil"
    assert meta["box"] == (50, 30, 200, 80)
    assert meta["font_size"] >= 8
    # colour sampled from the black text block
    assert sum(meta["colour"]) < 150


def test_replace_uses_comfy_when_available(monkeypatch):
    regions = [TextRegion(box=(50, 30, 200, 80), text="SALE",
                          confidence=96.0)]
    monkeypatch.setattr(edittext, "detect_text",
                        lambda img, lang="eng": regions)
    monkeypatch.setattr("nomorals.media_edit.comfy.comfy_available",
                        lambda: (True, ""))
    called = {}

    class FakeBackend:
        def inpaint(self, image, mask, prompt):
            called["prompt"] = prompt
            return image
    monkeypatch.setattr("nomorals.media_edit.comfy.ComfyUIBackend",
                        lambda: FakeBackend())
    out, meta = edittext.replace_text(_white_with_black_text(), "SALE",
                                      "SOLD", inpaint_backend="auto")
    assert meta["inpaint_backend"] == "comfy"
    assert "no text" in called["prompt"]


def test_replace_cv2_when_comfy_down(monkeypatch):
    regions = [TextRegion(box=(50, 30, 200, 80), text="SALE",
                          confidence=96.0)]
    monkeypatch.setattr(edittext, "detect_text",
                        lambda img, lang="eng": regions)
    monkeypatch.setattr("nomorals.media_edit.comfy.comfy_available",
                        lambda: (False, "down"))
    out, meta = edittext.replace_text(_white_with_black_text(), "SALE",
                                      "SOLD", inpaint_backend="auto")
    assert meta["inpaint_backend"] == "cv2"


# ── helpers ──────────────────────────────────────────────────────────────

def test_sample_text_color_dark_on_light():
    # black text block on white: the sampled colour must be dark
    colour = _sample_text_color(_white_with_black_text(), (50, 30, 200, 80))
    assert sum(colour) < 120


def test_sample_text_color_light_on_dark():
    img = Image.new("RGB", (200, 100), "black")
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, 100, 60], fill="white")
    colour = _sample_text_color(img, (20, 20, 100, 60))
    assert sum(colour) > 400


def test_fit_font_size_shrinks_long_text():
    img = _img()
    d = ImageDraw.Draw(img)
    big = _fit_font_size(d, "HI", 300, 60)
    small = _fit_font_size(d, "THIS IS A VERY LONG REPLACEMENT STRING", 300, 60)
    assert small <= big
    assert big >= 8 and small >= 8


def test_op_registered():
    from nomorals.media_edit.images import _OP_FUNCS, OP_ALLOWLIST
    assert "edittext" in _OP_FUNCS and "edittext" in OP_ALLOWLIST


# ── NL intent patterns ───────────────────────────────────────────────────

def test_nl_patterns():
    from nomorals.agents.coremind import _image_intent
    good = [
        'replace "SALE" with "SOLD" in this image',
        "replace 'SALE' with 'SOLD'",
        'change the text "hello" to "goodbye"',
        'change text "a" to "b" in the photo',
        "replace SALE with SOLD in this image",
    ]
    for text in good:
        intent = _image_intent(text)
        assert intent is not None and intent.action == "edittext", text
        assert intent.meta["old_text"] and intent.meta["new_text"], text
    bad = [
        "replace the tyres with new ones",
        "can you replace my phone",
        "I want to change the world",
        "draw a conclusion",
        "remove the background",
    ]
    for text in bad:
        intent = _image_intent(text)
        assert intent is None or intent.action != "edittext", text

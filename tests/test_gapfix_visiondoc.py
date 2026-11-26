"""Picture-to-document lane (gapfix146).

Covers: the model correction pass and its guards, the observation-only
content signals, the live builder listing, all five builders, the
one-step picture_to_document flow (choice, explicit format, invented
numbers, empty extraction, missing rungs, multi-picture), the
vision_document tool (text + listing, no file), the no-mapping rule,
and the /vision handler branches (document + read-text delivery).
"""

import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from nomorals.agents.partner.runtime_media import RuntimeMediaMixin
from nomorals.core.errors import ToolError
from nomorals.llm.base import LLMResponse
from nomorals.tools import vision as vision_mod
from nomorals.tools import vision_document as vd
from nomorals.tools.registry import ToolRegistry

LETTER_RAW = (
    "Dear Sir,\n\nI am writting to complian about the invoce dated "
    "12 March 2026 for the amount of 15000 naira. The total is 15000."
    "\n\nYours faithfully,\nPeace"
)
LETTER_FIXED = (
    "Dear Sir,\n\nI am writing to complain about the invoice dated "
    "12 March 2026 for the amount of 15000 naira. The total is 15000."
    "\n\nYours faithfully,\nPeace"
)
CHOICE_WORD = {
    "tool": "document_word",
    "reason": "a letter reads best as a Word document",
    "title": "Complaint letter",
    "sections": [{"heading": "", "level": 1, "text": LETTER_FIXED,
                  "kind": "text"}],
    "tables": [],
}
SHAPE_HTML = {
    "title": "Complaint letter",
    "sections": [{"heading": "", "level": 1, "text": LETTER_FIXED,
                  "kind": "text"}],
    "tables": [],
}
TABLE_STRUCTURE = {
    "title": "Price list",
    "sections": [{"heading": "", "level": 1, "text": "Prices this week.",
                  "kind": "text"}],
    "tables": [{"name": "Prices", "headers": ["Item", "Price"],
                "rows": [["Rice", "15000"], ["Beans", "12000"]]}],
}


class FakeRouter:
    """Routes by prompt shape: correction / choice / shape / transcribe."""

    def __init__(self, *, correction=LETTER_FIXED, choice=None,
                 shape=None, transcribe=LETTER_RAW, provider="groq"):
        self.correction = correction
        self.choice = choice if choice is not None else CHOICE_WORD
        self.shape = shape if shape is not None else SHAPE_HTML
        self.transcribe = transcribe
        self.provider = provider
        self.prompts: list[str] = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        if "correcting text extracted by OCR" in prompt:
            # identity correction: return the OCR text as given
            text = (prompt.split("OCR text:\n", 1)[1]
                    if self.correction is None else self.correction)
            return LLMResponse(text=text, provider=self.provider,
                               model="test")
        if "choosing how to turn" in prompt:
            return LLMResponse(text=json.dumps(self.choice),
                               provider=self.provider, model="test")
        if "shaping corrected text" in prompt:
            return LLMResponse(text=json.dumps(self.shape),
                               provider=self.provider, model="test")
        return LLMResponse(text=self.transcribe, provider=self.provider,
                           model="test")

    def describe_image(self, data, instruction):
        return LLMResponse(text=self.transcribe, provider=self.provider,
                           model="test")


class RaisingRouter(FakeRouter):
    def complete(self, prompt):
        raise RuntimeError("provider down")


def _ctx(tmp_path, router=None):
    settings = SimpleNamespace(
        partner=SimpleNamespace(vision=SimpleNamespace(log_calls=False)),
        workspace_dir=str(tmp_path),
    )
    return SimpleNamespace(settings=settings,
                           router=router if router is not None else FakeRouter())


def _registry(ctx):
    reg = ToolRegistry(ctx)
    vision_mod.register(reg)
    return reg


def _fake_ocr(text, method="native-ocr"):
    return {"text": text, "method": method, "provider": "native",
            "model": "tesseract", "source": "test"}


def _png_bytes():
    img = Image.new("RGB", (40, 20), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── correction pass ──────────────────────────────────────────────────────────


def test_correct_text_applies_confident_corrections(tmp_path):
    ctx = _ctx(tmp_path)
    out = vd.correct_text(ctx, LETTER_RAW, source="test")
    assert out["corrected"] is True
    assert out["text"] == LETTER_FIXED
    assert "writing" in out["text"] and "writting" not in out["text"]


def test_correct_text_rejects_number_change(tmp_path):
    router = FakeRouter(correction=LETTER_FIXED.replace("15000", "16000"))
    ctx = _ctx(tmp_path, router)
    out = vd.correct_text(ctx, LETTER_RAW, source="test")
    assert out["corrected"] is False
    assert out["text"] == LETTER_RAW
    assert "number" in out["note"]


def test_correct_text_rejects_dropped_illegible(tmp_path):
    raw = "Name: Peace [illegible] total 500"
    router = FakeRouter(correction="Name: Peace total 500")
    ctx = _ctx(tmp_path, router)
    out = vd.correct_text(ctx, raw, source="test")
    assert out["corrected"] is False
    assert "[illegible]" in out["text"]
    assert "illegible" in out["note"]


def test_correct_text_no_router_returns_uncorrected(tmp_path):
    ctx = _ctx(tmp_path, FakeRouter())
    ctx.router = None
    out = vd.correct_text(ctx, LETTER_RAW, source="test")
    assert out["corrected"] is False
    assert out["text"] == LETTER_RAW
    assert "no model" in out["note"]


def test_correct_text_mock_returns_uncorrected(tmp_path):
    ctx = _ctx(tmp_path, FakeRouter(provider="mock"))
    out = vd.correct_text(ctx, LETTER_RAW, source="test")
    assert out["corrected"] is False
    assert out["text"] == LETTER_RAW
    assert "mock" in out["note"]


def test_correct_text_router_raises_returns_uncorrected(tmp_path):
    ctx = _ctx(tmp_path, RaisingRouter())
    out = vd.correct_text(ctx, LETTER_RAW, source="test")
    assert out["corrected"] is False
    assert out["text"] == LETTER_RAW


# ── signals + listing (no mapping) ───────────────────────────────────────────


def test_content_signals_has_no_format_field():
    signals = vd.content_signals(
        "Dear Sir,\n\nItem  Price\nRice  15000\nBeans  12000\n"
        "Total: ₦15000.00\n")
    assert "format" not in signals
    assert "recommend" not in signals
    assert "tool" not in signals
    assert signals["table_like_rows"] == 2
    assert signals["has_salutation"] is True
    assert signals["has_amounts"] is True


def test_document_tool_listing_is_live(tmp_path):
    ctx = _ctx(tmp_path)
    reg = _registry(ctx)
    listing = vd.document_tool_listing(reg)
    for name in vd.DOCUMENT_TOOL_NAMES:
        assert name in listing
    reg.unregister("document_pdf")
    assert "document_pdf" not in vd.document_tool_listing(reg)


def test_formats_maps_user_words_only():
    assert set(vd.FORMATS) == {
        "docx", "word", "html", "htm", "txt", "text", "pdf", "xlsx",
        "sheet", "spreadsheet",
    }
    assert "letter" not in vd.FORMATS and "receipt" not in vd.FORMATS


def test_no_content_kind_mapping_in_module():
    src = Path(vd.__file__).read_text()
    for kind in ('"letter":', '"receipt":', '"tabular":', '"invoice":',
                 '"notes":'):
        assert kind not in src


# ── builders ─────────────────────────────────────────────────────────────────


def test_word_builder_makes_real_table(tmp_path):
    ctx = _ctx(tmp_path)
    out = vd.build_document_file(
        ctx, "document_word", "Price list", "",
        json.dumps(TABLE_STRUCTURE["sections"]),
        json.dumps(TABLE_STRUCTURE["tables"]))
    assert out["format"] == "docx"
    path = Path(out["path"])
    assert path.exists()
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml").decode()
    assert "<w:tbl>" in xml
    assert "Item" in xml and "Rice" in xml and "15000" in xml


def test_html_builder_is_print_page(tmp_path):
    ctx = _ctx(tmp_path)
    out = vd.build_document_file(ctx, "document_html", "Price list", "",
                                 json.dumps(TABLE_STRUCTURE["sections"]),
                                 json.dumps(TABLE_STRUCTURE["tables"]))
    assert out["format"] == "html"
    html = Path(out["path"]).read_text()
    assert "<html" in html.lower()
    assert "Price list" in html
    assert "Item" in html and "Rice" in html


def test_text_builder_plain(tmp_path):
    ctx = _ctx(tmp_path)
    out = vd.build_document_file(ctx, "document_text", "Price list", "",
                                 json.dumps(TABLE_STRUCTURE["sections"]),
                                 json.dumps(TABLE_STRUCTURE["tables"]))
    text = Path(out["path"]).read_text()
    assert "Price list" in text
    assert "Item | Price" in text
    assert "Rice | 15000" in text


def test_pdf_builder_magic(tmp_path):
    ctx = _ctx(tmp_path)
    out = vd.build_document_file(ctx, "document_pdf", "Price list", "",
                                 json.dumps(TABLE_STRUCTURE["sections"]),
                                 json.dumps(TABLE_STRUCTURE["tables"]))
    assert Path(out["path"]).read_bytes()[:4] == b"%PDF"


def test_sheet_builder_valid_xlsx(tmp_path):
    ctx = _ctx(tmp_path)
    out = vd.build_document_file(ctx, "document_sheet", "Price list", "",
                                 json.dumps(TABLE_STRUCTURE["sections"]),
                                 json.dumps(TABLE_STRUCTURE["tables"]))
    assert out["format"] == "xlsx"
    with zipfile.ZipFile(Path(out["path"])) as zf:
        names = zf.namelist()
        assert any(n.startswith("xl/worksheets/") for n in names)
        blob = b"\n".join(zf.read(n) for n in names if n.endswith(".xml"))
    assert b"Item" in blob and b"Rice" in blob


def test_builder_needs_content(tmp_path):
    ctx = _ctx(tmp_path)
    with pytest.raises(ToolError, match="needs text or a table"):
        vd.build_document_file(ctx, "document_word")


def test_builder_bad_json(tmp_path):
    ctx = _ctx(tmp_path)
    with pytest.raises(ToolError, match="not valid JSON"):
        vd.build_document_file(ctx, "document_word", "T", "",
                               sections_json="{nope")


# ── one-step flow ────────────────────────────────────────────────────────────


def test_picture_to_document_end_to_end_word(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    reg = _registry(ctx)
    monkeypatch.setattr(vd, "read_text",
                        lambda c, data, **kw: _fake_ocr(LETTER_RAW))
    out = vd.picture_to_document(ctx, reg, images=[(_png_bytes(), "test.png")])
    assert out["format"] == "docx"
    assert out["tool"] == "document_word"
    assert out["corrected"] is True
    assert Path(out["path"]).exists()
    assert "Complaint_letter" in out["filename"]


def test_picture_to_document_explicit_format_skips_choice(tmp_path,
                                                         monkeypatch):
    ctx = _ctx(tmp_path)
    reg = _registry(ctx)
    monkeypatch.setattr(vd, "read_text",
                        lambda c, data, **kw: _fake_ocr(LETTER_RAW))
    out = vd.picture_to_document(ctx, reg, images=[(_png_bytes(), "test.png")],
                                 format="html")
    assert out["format"] == "html"
    assert out["tool"] == "document_html"
    assert not any("choosing how to turn" in p for p in ctx.router.prompts)
    assert any("shaping corrected text" in p for p in ctx.router.prompts)


def test_picture_to_document_rejects_invented_number(tmp_path, monkeypatch):
    bad_choice = dict(CHOICE_WORD)
    bad_choice["sections"] = [
        {"heading": "", "level": 1,
         "text": LETTER_FIXED.replace("15000", "99999"), "kind": "text"}]
    ctx = _ctx(tmp_path, FakeRouter(choice=bad_choice))
    reg = _registry(ctx)
    monkeypatch.setattr(vd, "read_text",
                        lambda c, data, **kw: _fake_ocr(LETTER_RAW))
    out = vd.picture_to_document(ctx, reg, images=[(_png_bytes(), "test.png")])
    assert out["format"] == "txt"
    assert out["fallback"] is True
    assert "invented" in out["reason"]
    assert "15000" in Path(out["path"]).read_text()


def test_picture_to_document_empty_sends_no_file(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    reg = _registry(ctx)
    monkeypatch.setattr(vd, "read_text",
                        lambda c, data, **kw: _fake_ocr(""))
    with pytest.raises(ToolError, match="no text found"):
        vd.picture_to_document(ctx, reg, images=[(_png_bytes(), "blank.png")])
    assert not list((tmp_path / "vision_docs").glob("*")) \
        if (tmp_path / "vision_docs").exists() else True


def test_picture_to_document_missing_rungs_names_both(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    reg = _registry(ctx)
    monkeypatch.setattr(
        vd, "read_text",
        lambda c, data, **kw: _fake_ocr("[vision unavailable: no router]"))
    with pytest.raises(ToolError) as excinfo:
        vd.picture_to_document(ctx, reg, images=[(_png_bytes(), "test.png")])
    assert "tesseract" in str(excinfo.value)
    assert "vision model" in str(excinfo.value)


def test_picture_to_document_multi_picture_one_document(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, FakeRouter(correction=None))
    reg = _registry(ctx)
    first = "Page one total 100."
    second = "Page two total 200."
    calls = {"n": 0}

    def _ocr(c, data, **kw):
        calls["n"] += 1
        return _fake_ocr(first if calls["n"] == 1 else second)

    monkeypatch.setattr(vd, "read_text", _ocr)
    out = vd.picture_to_document(
        ctx, reg,
        images=[(_png_bytes(), "p1.png"), (_png_bytes(), "p2.png")])
    assert Path(out["path"]).exists()
    assert "Page one total 100." in out["text"]
    assert "Page two total 200." in out["text"]
    assert out["text"].index("Page one") < out["text"].index("Page two")


# ── the tool itself (extract + correct only — no file) ───────────────────────


def test_vision_document_tool_returns_text_and_listing_no_file(tmp_path,
                                                               monkeypatch):
    ctx = _ctx(tmp_path)
    reg = _registry(ctx)
    (tmp_path / "letter.png").write_bytes(_png_bytes())
    monkeypatch.setattr(vision_mod, "read_text",
                        lambda c, data, **kw: _fake_ocr(LETTER_RAW))
    outcome = reg.call("vision_document", path="letter.png")
    assert outcome.ok, outcome.error
    value = outcome.value
    assert value["corrected_text"] == LETTER_FIXED
    assert value["corrected"] is True
    assert "path" not in value
    for name in vd.DOCUMENT_TOOL_NAMES:
        assert name in value["document_tools"]


# ── /vision handler branches ─────────────────────────────────────────────────


def _handler_fixture(tmp_path):
    ctx = _ctx(tmp_path)
    reg = _registry(ctx)
    ctx.tools = reg
    png = tmp_path / "shot.png"
    png.write_bytes(_png_bytes())
    sent = {}

    class Gateway:
        def send_file(self, platform, chat, path, **kw):
            sent.update(platform=platform, chat=chat, path=path, **kw)

    fake = SimpleNamespace(
        context=ctx,
        gateway=Gateway(),
        _VISION_SUBCOMMANDS=RuntimeMediaMixin._VISION_SUBCOMMANDS,
        _ref_from_key=lambda key: SimpleNamespace(platform="telegram-bot",
                                                  chat_id="123"),
        _send_long_checked=lambda platform, chat, text: text,
    )
    msg = SimpleNamespace(media=[SimpleNamespace(kind="image",
                                                 path=str(png))])
    return fake, msg, sent


def test_handler_document_branch_sends_file(tmp_path):
    fake, msg, sent = _handler_fixture(tmp_path)
    text = RuntimeMediaMixin._control_vision(
        fake, "document word", chat_key="telegram-bot:123", message=msg)
    assert sent["path"].endswith(".docx")
    assert sent.get("kind") == "document"
    assert "Complaint_letter" in text
    assert "document_word" not in text
    assert "docx" in text


def test_handler_read_text_delivers_document(tmp_path):
    fake, msg, sent = _handler_fixture(tmp_path)
    text = RuntimeMediaMixin._control_vision(
        fake, "read-text", chat_key="telegram-bot:123", message=msg)
    assert sent["path"].endswith(".docx")
    assert "📝" in text
    assert "writing to complain" in text
    assert "document:" in text

"""Gapfix R2-23 — vision fusion wired into extract().

Failing-first: before the wiring, extract() returns no document_kind,
no claims, no grounding block, and ``nomorals.vision`` does not export
fuse_image_text. After the wiring, a document image run through
extract() comes back classified AND with its figures checked against
deterministic (tesseract) OCR text; without tesseract the result stays
honest about grounding being unavailable instead of pretending.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any
from unittest import mock

from nomorals.llm.base import LLMProvider, LLMResponse, Message, SamplingParams, Usage
from nomorals.llm.router import LLMRouter
from nomorals.tools import vision as V
from nomorals.vision.seer import UNTRUSTED_VISION_PREFIX

# ── fakes (mirroring tests/test_vision.py — no cross-test imports) ──────────


class StubVisionProvider(LLMProvider):
    """Vision-capable provider with a scripted reply. No network."""

    name = "stub-vision"

    def __init__(self, text: str = "[stub] a red square", **kw: Any) -> None:
        super().__init__(**kw)
        self.text = text

    @property
    def capabilities(self) -> set[str]:
        return {"chat", "vision"}

    def chat(self, messages: Sequence[Message],
             params: SamplingParams | None = None, **kw: Any) -> LLMResponse:
        return LLMResponse(text="chat", model="stub")

    def describe_image(self, image: bytes, prompt: str = "",
                       params: SamplingParams | None = None,
                       **kw: Any) -> LLMResponse:
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        return LLMResponse(text=self.text, model="stub-vlm", provider=self.name,
                           usage=usage)


def _context(router: Any = None) -> SimpleNamespace:
    settings = SimpleNamespace(
        partner=SimpleNamespace(vision=SimpleNamespace(
            allow_screenshot=False, max_dimension=1568,
            max_image_bytes=25 * 1024 * 1024, log_calls=False, enabled=True)),
        workspace_dir=tempfile.mkdtemp())
    return SimpleNamespace(settings=settings, router=router,
                           extras={"attachments": []})


def _png(width: int = 64, height: int = 48,
         color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def _extract_json(summary: str, key_text: str = "",
                  elements: list[str] | None = None,
                  scene_type: str = "document") -> str:
    return json.dumps({
        "summary": summary,
        "key_text": key_text,
        "notable_elements": elements or [],
        "scene_type": scene_type,
    })


def _router_for(reply: str) -> LLMRouter:
    router = LLMRouter()
    router.add(StubVisionProvider(text=reply), primary=True, name="vlm")
    return router


INVOICE_OCR = ("INVOICE No 42 — Amount due $1,200. Subtotal $1,000. "
               "Tax $200.")


def _fake_ocr(text: str):
    def _read_text(context: Any, data: bytes, **kwargs: Any) -> dict[str, Any]:
        return {"text": UNTRUSTED_VISION_PREFIX + text,
                "method": "native-ocr", "provider": "native",
                "model": "tesseract"}
    return _read_text


class ExtractFusionTests(unittest.TestCase):
    def test_extract_invoice_is_classified_and_grounded(self) -> None:
        router = _router_for(_extract_json(
            "The total is $1,200.", key_text="INVOICE total $1,200",
            elements=["Total $1,200"]))
        ctx = _context(router)
        ocr = mock.MagicMock(side_effect=_fake_ocr(INVOICE_OCR))
        with mock.patch.object(V, "read_text", ocr):
            result = V.extract(ctx, _png())
        self.assertEqual(result["document_kind"], "invoice")
        self.assertEqual(result["ocr_method"], "native-ocr")
        self.assertIn("1,200", result["ocr_text"])
        self.assertTrue(result["grounding"]["available"])
        self.assertEqual(result["ungrounded"], [])
        self.assertTrue(result["claims"])
        self.assertTrue(all(c["grounded"] for c in result["claims"]))
        # grounding reads deterministic OCR only — never a second model call
        _, kwargs = ocr.call_args
        self.assertEqual(kwargs.get("strategy"), "native")

    def test_extract_hallucinated_number_is_flagged(self) -> None:
        router = _router_for(_extract_json(
            "The total is $9,999.", key_text="INVOICE total $9,999",
            elements=["Total $9,999"]))
        ctx = _context(router)
        with mock.patch.object(V, "read_text", _fake_ocr(INVOICE_OCR)):
            result = V.extract(ctx, _png())
        self.assertEqual(result["document_kind"], "invoice")
        missing = [n for c in result["ungrounded"]
                   for n in c["missing_numbers"]]
        self.assertIn("9,999", missing)
        chat = V.format_chat_summary(result)
        self.assertIn("9,999", chat)
        self.assertIn("nverified", chat)

    def test_extract_contract_is_classified(self) -> None:
        router = _router_for(_extract_json(
            "The parties agree to the terms.", key_text="",
            elements=["Agreement"], scene_type="document"))
        ctx = _context(router)
        ocr = ("AGREEMENT — Whereas the parties hereby agree to the "
               "terms and conditions herein. Effective date 2026-01-05.")
        with mock.patch.object(V, "read_text", _fake_ocr(ocr)):
            result = V.extract(ctx, _png())
        self.assertEqual(result["document_kind"], "contract")

    def test_extract_without_tesseract_is_honest(self) -> None:
        from nomorals.vision.native import NativeUnavailable

        router = _router_for(_extract_json(
            "The total is $1,200.", key_text="INVOICE total $1,200"))
        ctx = _context(router)

        def _no_ocr(context: Any, data: bytes, **kwargs: Any) -> dict[str, Any]:
            raise NativeUnavailable(
                "native OCR needs the tesseract binary (Termux: "
                "pkg install tesseract; or set NM_OCR_BINARY)")

        with mock.patch.object(V, "read_text", _no_ocr):
            result = V.extract(ctx, _png())
        # the model result still comes back — grounding is additive, never fatal
        self.assertEqual(result["summary"], "The total is $1,200.")
        self.assertFalse(result["grounding"]["available"])
        self.assertIn("tesseract", result["grounding"]["reason"])
        self.assertEqual(result["document_kind"], "unknown")
        self.assertEqual(result["claims"], [])
        self.assertEqual(result["ungrounded"], [])
        chat = V.format_chat_summary(result)
        self.assertIn("not checked against image text", chat)

    def test_extract_empty_ocr_flags_model_numbers(self) -> None:
        router = _router_for(_extract_json(
            "The total is $1,200.", key_text="Total $1,200"))
        ctx = _context(router)
        with mock.patch.object(V, "read_text", _fake_ocr("")):
            result = V.extract(ctx, _png())
        # an empty transcript is valid OCR, not "unavailable" — and an image
        # with no text cannot ground any figure the model emitted
        self.assertTrue(result["grounding"]["available"])
        self.assertEqual(result["document_kind"], "other")
        self.assertTrue(result["ungrounded"])
        missing = [n for c in result["ungrounded"]
                   for n in c["missing_numbers"]]
        self.assertIn("1,200", missing)

    def test_fusion_exported_from_vision_package(self) -> None:
        from nomorals.vision import fuse_image_text
        from nomorals.vision import fusion as fusion_mod

        self.assertIs(fuse_image_text, fusion_mod.fuse_image_text)

    def test_format_chat_summary_renders_document_kind(self) -> None:
        result = {
            "summary": "A receipt for coffee.",
            "notable_elements": [],
            "key_text": "",
            "scene_type": "document",
            "document_kind": "receipt",
            "grounding": {"available": True},
            "claims": [],
            "ungrounded": [],
        }
        self.assertIn("Document: receipt", V.format_chat_summary(result))


if __name__ == "__main__":
    unittest.main()

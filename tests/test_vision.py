"""Prompt 09 — vision tool tests.

Acceptance criteria covered:
- describe via a vision-capable routed model (mocked, no network)
- router with a text-only model raises the ModelError path cleanly
- read_text transcribes; uncertainty flagged, not invented
- locate returns 0-1000 coords + confidence + approximate disclaimer
- screenshot refused without allow_screenshot; still needs per-call confirm
- inbox image drops classify to the image intent and get described
- identity questions pass through to the provider untouched (no refusal layer)
- size guards, downscaling, tokens_used, attachment refs
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from nomorals.core.errors import ModelError, ToolError
from nomorals.llm.base import LLMProvider, LLMResponse, Message, SamplingParams, Usage
from nomorals.llm.providers.mock import MockProvider
from nomorals.llm.router import LLMRouter
from nomorals.tools import vision as V


# ── fakes ────────────────────────────────────────────────────────────────────


class StubVisionProvider(LLMProvider):
    """Vision-capable provider with a scripted reply. No network."""

    name = "stub-vision"

    def __init__(self, text: str = "[stub] a red square", **kw: Any) -> None:
        super().__init__(**kw)
        self.text = text
        self.seen: list[tuple[bytes, str]] = []

    @property
    def capabilities(self) -> set[str]:
        return {"chat", "vision"}

    def chat(self, messages: Sequence[Message],
             params: SamplingParams | None = None, **kw: Any) -> LLMResponse:
        return LLMResponse(text="chat", model="stub")

    def describe_image(self, image: bytes, prompt: str = "",
                       params: SamplingParams | None = None,
                       **kw: Any) -> LLMResponse:
        self.seen.append((image, prompt))
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        return LLMResponse(text=self.text, model="stub-vlm", provider=self.name,
                           usage=usage)


def _vision_settings(**over: Any) -> SimpleNamespace:
    base = dict(allow_screenshot=False, max_dimension=1568,
                max_image_bytes=25 * 1024 * 1024, log_calls=False,
                enabled=True)
    base.update(over)
    return SimpleNamespace(vision=SimpleNamespace(**base))


def _context(router: Any = None, workspace: str = "",
             attachments: list[dict[str, Any]] | None = None) -> SimpleNamespace:
    settings = SimpleNamespace(
        partner=_vision_settings(), workspace_dir=workspace or tempfile.mkdtemp())
    return SimpleNamespace(settings=settings, router=router,
                           extras={"attachments": attachments or []})


def _png(width: int = 64, height: int = 48,
         color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    from PIL import Image

    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── router ───────────────────────────────────────────────────────────────────


class RouterVisionTests(unittest.TestCase):
    def test_vision_routes_to_capable_provider(self) -> None:
        router = LLMRouter()
        router.add(MockProvider(model="text-only"), primary=True, name="text")
        stub = StubVisionProvider()
        router.add(stub, name="vlm")
        resp = router.describe_image(b"\x89PNG\r\n\x1a\n", "what?")
        self.assertTrue(resp.ok)
        # the vision-capable provider answered, not the text-only mock
        self.assertEqual(resp.text, "[stub] a red square")
        self.assertEqual(len(stub.seen), 1)

    def test_text_only_model_raises_model_error_cleanly(self) -> None:
        """Acceptance: no vision-capable provider → ModelError, no traceback leak."""
        router = LLMRouter()
        router.add(MockProvider(model="text-only"), primary=True, name="text")
        with self.assertRaises(ModelError) as cm:
            router.describe_image(b"\x89PNG\r\n\x1a\n", "what?")
        self.assertIn("vision", str(cm.exception).lower())
        # clean domain error, not a traceback dump
        self.assertNotIn("Traceback", str(cm.exception))


# ── describe ─────────────────────────────────────────────────────────────────


class DescribeTests(unittest.TestCase):
    def test_describe_returns_model_tokens_and_metadata(self) -> None:
        stub = StubVisionProvider()
        router = LLMRouter()
        router.add(stub, primary=True, name="vlm")
        ctx = _context(router)
        result = V.describe(ctx, _png(), "what's in this image?")
        self.assertEqual(result["description"], "[stub] a red square")
        self.assertEqual(result["model"], "stub-vlm")
        self.assertEqual(result["tokens_used"]["total_tokens"], 30)
        self.assertEqual(result["format"], "png")
        self.assertEqual((result["width"], result["height"]), (64, 48))
        self.assertEqual(result["source"], "bytes")

    def test_describe_tolerant_when_no_provider(self) -> None:
        ctx = _context(None)  # no router at all
        result = V.describe(ctx, _png())
        self.assertIn("[vision unavailable", result["description"])
        self.assertEqual(result["format"], "png")  # metadata still works

    def test_describe_strict_raises(self) -> None:
        ctx = _context(None)
        with self.assertRaises(ModelError):
            V.describe(ctx, _png(), strict=True)

    def test_size_guards(self) -> None:
        ctx = _context(None)
        with self.assertRaises(ToolError):
            V.describe(ctx, b"")

    def test_size_guard_honors_small_cap(self) -> None:
        ctx = _context(None)
        ctx.settings.partner = _vision_settings(max_image_bytes=10)
        with self.assertRaises(ToolError) as cm:
            V.describe(ctx, b"x" * 11)
        self.assertIn("over the 10-byte", str(cm.exception))

    def test_downscale_before_send(self) -> None:
        stub = StubVisionProvider()
        router = LLMRouter()
        router.add(stub, primary=True, name="vlm")
        ctx = _context(router)
        ctx.settings.partner = _vision_settings(max_dimension=32, log_calls=False)
        data = _png(400, 300)
        result = V.describe(ctx, data)
        sent = stub.seen[0][0]
        self.assertLess(len(sent), len(data))
        self.assertIn("downscaled 400x300", result["downscale_note"])

    def test_no_downscale_for_small_images(self) -> None:
        stub = StubVisionProvider()
        router = LLMRouter()
        router.add(stub, primary=True, name="vlm")
        ctx = _context(router)
        data = _png(64, 48)
        V.describe(ctx, data)
        self.assertEqual(stub.seen[0][0], data)

    def test_cache_avoids_second_model_call(self) -> None:
        stub = StubVisionProvider()
        router = LLMRouter()
        router.add(stub, primary=True, name="vlm")
        ctx = _context(router)
        cache: dict[str, dict[str, Any]] = {}
        data = _png()
        V.describe(ctx, data, cache=cache)
        second = V.describe(ctx, data, cache=cache)
        self.assertEqual(len(stub.seen), 1)
        self.assertTrue(second["cached"])


# ── identity passthrough (no refusal layer) ────────────────────────────────
# Devon adds no identity refusal of its own: the owner's prompt reaches the
# provider verbatim.  Whatever the provider answers is what the owner gets.


class IdentityPassthroughTests(unittest.TestCase):
    def _described(self, prompt: str) -> tuple[dict[str, Any], str]:
        stub = StubVisionProvider(text="[stub] a person in a blue jacket")
        router = LLMRouter()
        router.add(stub, primary=True, name="vlm")
        ctx = _context(router)
        result = V.describe(ctx, _png(), prompt)
        return result, stub.seen[0][1]

    def test_who_is_this_passes_through_untouched(self) -> None:
        result, sent_prompt = self._described("who is this person in the photo?")
        self.assertNotIn("identity_declined", result)
        self.assertNotIn("identity_note", result)
        self.assertEqual(sent_prompt, "who is this person in the photo?")

    def test_ordinary_prompt_untouched(self) -> None:
        result, sent_prompt = self._described("describe this chart")
        self.assertNotIn("identity_declined", result)
        self.assertEqual(sent_prompt, "describe this chart")


# ── read_text / locate / compare ─────────────────────────────────────────────


class ReadTextTests(unittest.TestCase):
    def test_read_text_returns_text_and_confidence_note(self) -> None:
        stub = StubVisionProvider(text="INVOICE #123\nTotal: $45.00")
        router = LLMRouter()
        router.add(stub, primary=True, name="vlm")
        ctx = _context(router)
        result = V.read_text(ctx, _png())
        self.assertEqual(result["text"], "INVOICE #123\nTotal: $45.00")
        self.assertIn("confidence_note", result)
        # fidelity prompt: never invent, flag uncertainty
        sent = stub.seen[0][1]
        self.assertIn("[illegible]", sent)
        self.assertIn("verbatim", sent.lower())


class LocateTests(unittest.TestCase):
    def _ctx(self, text: str) -> SimpleNamespace:
        stub = StubVisionProvider(text=text)
        router = LLMRouter()
        router.add(stub, primary=True, name="vlm")
        return _context(router)

    def test_locate_returns_bbox_confidence_disclaimer(self) -> None:
        ctx = self._ctx('{"x": 100, "y": 200, "w": 150, "h": 40, "confidence": 0.9}')
        result = V.locate(ctx, _png(), "the submit button")
        self.assertTrue(result["found"])
        self.assertEqual((result["x"], result["y"],
                          result["w"], result["h"]), (100, 200, 150, 40))
        self.assertAlmostEqual(result["confidence"], 0.9)
        self.assertTrue(result["approximate"])
        self.assertIn("approximate", result["disclaimer"].lower())

    def test_locate_clamps_out_of_range(self) -> None:
        ctx = self._ctx('{"x": -5, "y": 0, "w": 2000, "h": 40, "confidence": 9}')
        result = V.locate(ctx, _png(), "thing")
        self.assertEqual(result["x"], 0)
        self.assertEqual(result["w"], 1000)
        self.assertEqual(result["confidence"], 1.0)

    def test_locate_not_found(self) -> None:
        ctx = self._ctx('{"found": false}')
        result = V.locate(ctx, _png(), "a unicorn")
        self.assertFalse(result["found"])
        self.assertIn("disclaimer", result)

    def test_locate_needs_target(self) -> None:
        ctx = self._ctx('{}')
        with self.assertRaises(ToolError):
            V.locate(ctx, _png(), "  ")


class CompareTests(unittest.TestCase):
    def test_compare_side_by_side(self) -> None:
        stub = StubVisionProvider(text="the right image has an extra tree")
        router = LLMRouter()
        router.add(stub, primary=True, name="vlm")
        ctx = _context(router)
        result = V.compare(ctx, _png(color=(255, 0, 0)), _png(color=(0, 0, 255)))
        self.assertEqual(result["method"], "side-by-side")
        self.assertIn("extra tree", result["description"])
        self.assertEqual(len(stub.seen), 1)  # one call, not two
        sent = stub.seen[0][0]
        meta = V.image_metadata(sent)
        self.assertEqual(meta["width"], 64 + 64 + 8)  # stitched


# ── intake: resolve_image ────────────────────────────────────────────────────


class IntakeTests(unittest.TestCase):
    def test_exactly_one_source(self) -> None:
        ctx = _context()
        with self.assertRaises(ToolError):
            V.resolve_image(ctx)
        with self.assertRaises(ToolError):
            V.resolve_image(ctx, path="a.png", url="http://x")

    def test_bytes_source(self) -> None:
        ctx = _context()
        data, source = V.resolve_image(ctx, data=b"abc")
        self.assertEqual((data, source), (b"abc", "bytes"))

    def test_path_source_sandboxed(self) -> None:
        with tempfile.TemporaryDirectory() as ws:
            p = Path(ws) / "shot.png"
            p.write_bytes(_png())
            ctx = _context(workspace=ws)
            data, source = V.resolve_image(ctx, path="shot.png")
            self.assertEqual(source, "path:shot.png")
            self.assertTrue(data.startswith(b"\x89PNG"))
            with self.assertRaises(Exception):  # escape rejected
                V.resolve_image(ctx, path="../outside.png")

    def test_attachment_reference(self) -> None:
        with tempfile.TemporaryDirectory() as media:
            p = Path(media) / "photo.jpg"
            p.write_bytes(_png())
            ctx = _context(attachments=[{"path": str(p), "name": "photo.jpg",
                                         "mime": "image/png"}])
            data, source = V.resolve_image(ctx, reference="attachment:0")
            self.assertEqual(source, "attachment:0")
            self.assertTrue(data.startswith(b"\x89PNG"))

    def test_attachment_out_of_range(self) -> None:
        ctx = _context(attachments=[])
        with self.assertRaises(ToolError):
            V.resolve_image(ctx, reference="attachment:0")

    def test_unknown_reference(self) -> None:
        ctx = _context()
        with self.assertRaises(ToolError):
            V.resolve_image(ctx, reference="carrier-pigeon:7")


# ── screenshot gating ────────────────────────────────────────────────────────


class ScreenshotGateTests(unittest.TestCase):
    def _registry(self, **vision_knobs: Any) -> Any:
        from nomorals.tools.registry import ToolRegistry

        router = LLMRouter()
        router.add(StubVisionProvider(), primary=True, name="vlm")
        ctx = _context(router)
        ctx.settings.partner = _vision_settings(**vision_knobs)
        reg = ToolRegistry(context=ctx)
        V.register(reg)
        return reg

    def test_refused_without_setting(self) -> None:
        reg = self._registry(allow_screenshot=False)
        outcome = reg.call("vision_screenshot", actor="cli", confirm=True)
        self.assertFalse(outcome.ok)
        self.assertIn("disabled", str(outcome.error).lower())

    def test_refused_without_per_call_confirm(self) -> None:
        reg = self._registry(allow_screenshot=True)
        outcome = reg.call("vision_screenshot", actor="cli", confirm=False)
        self.assertFalse(outcome.ok)
        self.assertIn("confirm", str(outcome.error).lower())

    def test_allowed_with_both_gates(self) -> None:
        reg = self._registry(allow_screenshot=True)
        real = V._screenshot_capture
        V._screenshot_capture = lambda display=0: _png()  # noqa: E731
        try:
            outcome = reg.call("vision_screenshot", actor="cli", confirm=True)
        finally:
            V._screenshot_capture = real
        self.assertTrue(outcome.ok, outcome.error if not outcome.ok else "")
        self.assertEqual(outcome.value["source"], "screenshot")


# ── registered tools end-to-end ──────────────────────────────────────────────


class RegisteredToolTests(unittest.TestCase):
    def _registry(self) -> Any:
        from nomorals.tools.registry import ToolRegistry

        router = LLMRouter()
        router.add(StubVisionProvider(), primary=True, name="vlm")
        ctx = _context(router)
        reg = ToolRegistry(context=ctx)
        V.register(reg)
        return reg

    def test_all_expected_tools_registered(self) -> None:
        reg = self._registry()
        for name in ("vision_describe", "vision_read_text", "vision_locate",
                     "vision_compare", "vision_screenshot", "vision_metadata",
                     "vision_to_base64"):
            self.assertIn(name, reg.names(), name)

    def test_vision_read_text_tool(self) -> None:
        with tempfile.TemporaryDirectory() as ws:
            (Path(ws) / "doc.png").write_bytes(_png())
            reg = self._registry()
            # registry context has its own workspace; write there instead
            ctx = reg.context
            target = Path(ctx.settings.workspace_dir) / "doc.png"
            target.write_bytes(_png())
            outcome = reg.call("vision_read_text", actor="t", path="doc.png")
            self.assertTrue(outcome.ok, outcome.error if not outcome.ok else "")
            self.assertIn("text", outcome.value)
            self.assertIn("confidence_note", outcome.value)

    def test_tool_error_is_clean_not_traceback(self) -> None:
        reg = self._registry()
        outcome = reg.call("vision_describe", actor="t")  # no source at all
        self.assertFalse(outcome.ok)
        self.assertNotIn("Traceback", str(outcome.error))


# ── inbox image intent ───────────────────────────────────────────────────────


class InboxImageTests(unittest.TestCase):
    def test_images_classify_to_image_intent(self) -> None:
        from nomorals.workspace.inbox import _default_classify

        item = SimpleNamespace(kind="file", name="photo.PNG", mime="image/png")
        self.assertEqual(_default_classify(None, item), "image")
        video = SimpleNamespace(kind="file", name="clip.mp4", mime="video/mp4")
        self.assertEqual(_default_classify(None, video), "describe")

    def test_directives_map_to_image_intent(self) -> None:
        from nomorals.workspace.inbox import DIRECTIVE_INTENTS

        self.assertEqual(DIRECTIVE_INTENTS["read-text"], "image")
        self.assertEqual(DIRECTIVE_INTENTS["locate"], "image")

    def test_handle_image_describes_via_hook(self) -> None:
        from nomorals.workspace.inbox import Inbox, InboxItem, _handle_image

        calls: list[tuple[str, str]] = []

        def hook(data: bytes, action: str, prompt: str) -> dict[str, Any]:
            calls.append((action, prompt))
            return {"description": "[hook] a cat on a couch"}

        with tempfile.TemporaryDirectory() as root:
            inbox = Inbox(root, vision=hook)
            item_path = Path(root) / "inbox" / "cat.png"
            item_path.write_bytes(_png())
            item = InboxItem(id="1", name="cat.png", path=str(item_path))
            result = _handle_image(inbox, item)
            self.assertEqual(calls, [("describe", "")])
            self.assertIn("cat", result.summary)
            self.assertEqual(result.disposition, "processed")

    def test_handle_image_read_text_directive(self) -> None:
        from nomorals.workspace.inbox import Inbox, InboxItem, _handle_image

        def hook(data: bytes, action: str, prompt: str) -> dict[str, Any]:
            return {"text": "HELLO WORLD", "confidence_note": "n/a"}

        with tempfile.TemporaryDirectory() as root:
            inbox = Inbox(root, vision=hook)
            item_path = Path(root) / "inbox" / "doc.png"
            item_path.write_bytes(_png())
            item = InboxItem(id="2", name="doc.png", path=str(item_path),
                             directive="@read-text")
            result = _handle_image(inbox, item)
            self.assertIn("HELLO WORLD", result.summary)

    def test_handle_image_locate_directive(self) -> None:
        from nomorals.workspace.inbox import Inbox, InboxItem, _handle_image

        seen: list[str] = []

        def hook(data: bytes, action: str, prompt: str) -> dict[str, Any]:
            seen.append(prompt)
            return {"found": True, "x": 1, "y": 2, "w": 3, "h": 4,
                    "confidence": 0.8}

        with tempfile.TemporaryDirectory() as root:
            inbox = Inbox(root, vision=hook)
            item_path = Path(root) / "inbox" / "ui.png"
            item_path.write_bytes(_png())
            item = InboxItem(id="3", name="ui.png", path=str(item_path),
                             directive="@locate the submit button")
            result = _handle_image(inbox, item)
            self.assertEqual(seen, ["the submit button"])
            self.assertIn("submit button", result.summary)

    def test_handle_image_without_hook_is_probe_only_and_honest(self) -> None:
        from nomorals.workspace.inbox import Inbox, InboxItem, _handle_image

        with tempfile.TemporaryDirectory() as root:
            inbox = Inbox(root)  # no vision hook
            item_path = Path(root) / "inbox" / "cat.png"
            item_path.write_bytes(_png())
            item = InboxItem(id="4", name="cat.png", path=str(item_path))
            result = _handle_image(inbox, item)
            self.assertIn("vision not wired", result.summary)
            self.assertEqual(result.disposition, "processed")

    def test_vision_failure_parks_item(self) -> None:
        from nomorals.workspace.inbox import Inbox, InboxItem, _handle_image

        def hook(data: bytes, action: str, prompt: str) -> dict[str, Any]:
            raise ModelError("no registered provider supports vision")

        with tempfile.TemporaryDirectory() as root:
            inbox = Inbox(root, vision=hook)
            item_path = Path(root) / "inbox" / "cat.png"
            item_path.write_bytes(_png())
            item = InboxItem(id="5", name="cat.png", path=str(item_path))
            result = _handle_image(inbox, item)
            self.assertEqual(result.disposition, "stay")
            self.assertIn("vision", item.error)


if __name__ == "__main__":
    unittest.main()

"""Wave F2 — MEDIA/FILES acceptance tests.

Covers the three F2 deliverables:

1. download → compress → send pipeline (per-stage status, fail-fast,
   honest compress skips);
2. vision extraction with chat-friendly structured output (never an
   empty success when the model is missing);
3. zip/unzip + send wiring to the social adapters (mock gateway);
4. a presence guard so no media module, tool, or function goes missing.

Run with TMPDIR=/var/tmp.
"""

from __future__ import annotations

import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from PIL import Image

from nomorals.core.errors import MediaError, ModelError, ToolError
from nomorals.llm.base import LLMProvider, LLMResponse, Message, SamplingParams, Usage
from nomorals.llm.router import LLMRouter
from nomorals.tools import archive as A
from nomorals.tools import media_pipeline as P
from nomorals.tools import vision as V


# ── fixtures ─────────────────────────────────────────────────────────────────


def _context(**over: Any) -> SimpleNamespace:
    ctx = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=tempfile.mkdtemp()),
        extras={},
        router=None,
    )
    for key, value in over.items():
        setattr(ctx, key, value)
    return ctx


class FakeGateway:
    """Minimal stand-in for the live chat gateway's file-send path."""

    def __init__(self) -> None:
        self.adapters = {"telegram": object(), "whatsapp": object()}
        self.sent: list[dict[str, Any]] = []

    def send_file(self, platform: str, chat_id: str, path: str,
                  caption: str = "") -> SimpleNamespace:
        self.sent.append({"platform": platform, "chat": chat_id,
                          "path": path, "caption": caption})
        return SimpleNamespace(ok=True, message_id="m1")


def _noisy_jpeg(path: Path, width: int = 2000, height: int = 1500,
                quality: int = 95) -> Path:
    import random

    rng = random.Random(42)
    img = Image.new("RGB", (width, height))
    img.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256))
                 for _ in range(width * height)])
    img.save(path, "JPEG", quality=quality)
    return path


def _png_bytes(width: int = 64, height: int = 48) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def _make_downloader(fixture: Path):
    def fake(url: str, dest_dir: Any, **kw: Any) -> dict[str, Any]:
        dest = Path(dest_dir) / fixture.name
        dest.write_bytes(fixture.read_bytes())
        return {"url": url, "path": str(dest), "bytes": dest.stat().st_size,
                "title": fixture.name, "extractor": "fake"}
    return fake


def _make_sender(captured: list):
    def fake(context: Any, platform: str, chat_id: str, path: str,
             caption: str = "") -> dict[str, Any]:
        captured.append((platform, chat_id, path, caption))
        return {"sent": True, "platform": platform, "chat": chat_id,
                "path": path, "message_id": "m1",
                "bytes": Path(path).stat().st_size}
    return fake


# ── 1. pipeline ──────────────────────────────────────────────────────────────


class PipelineTests(unittest.TestCase):
    def test_pipeline_happy_path_image(self) -> None:
        ctx = _context()
        fixture = _noisy_jpeg(Path(ctx.settings.workspace_dir) / "big.jpg")
        captured: list = []
        report = P.run_pipeline(
            ctx, "http://example.invalid/big.jpg", "telegram", "123",
            image_max_width=800, image_quality=60,
            downloader=_make_downloader(fixture),
            sender=_make_sender(captured),
        )
        self.assertTrue(report["ok"])
        self.assertEqual([s["stage"] for s in report["stages"]],
                         ["download", "compress", "send"])
        self.assertTrue(all(s["ok"] for s in report["stages"]))
        download, compress, send = report["stages"]
        self.assertEqual(download["bytes"], fixture.stat().st_size)
        # compression actually shrunk the image and the sender got the new path
        self.assertTrue(compress["ok"])
        self.assertLess(compress["new_bytes"], compress["original_bytes"])
        self.assertTrue(report["compressed"])
        self.assertEqual(len(captured), 1)
        platform, chat, sent_path, _caption = captured[0]
        self.assertEqual((platform, chat), ("telegram", "123"))
        self.assertEqual(sent_path, compress["path"])
        self.assertTrue(sent_path.endswith(".send.jpg"))
        self.assertEqual(report["sent_path"], sent_path)

    def test_pipeline_download_failure_is_fail_fast(self) -> None:
        ctx = _context()

        def boom(url: str, dest_dir: Any, **kw: Any) -> dict[str, Any]:
            raise MediaError("network down")

        with self.assertRaises(MediaError) as cm:
            P.run_pipeline(ctx, "http://example.invalid/x.mp4", "telegram",
                           "123", downloader=boom, sender=_make_sender([]))
        self.assertIn("pipeline download failed", str(cm.exception))

    def test_pipeline_send_failure_is_fail_fast(self) -> None:
        ctx = _context()
        fixture = _noisy_jpeg(Path(ctx.settings.workspace_dir) / "pic.jpg")

        def bad_send(context: Any, platform: str, chat_id: str, path: str,
                     caption: str = "") -> dict[str, Any]:
            raise ToolError("gateway offline")

        with self.assertRaises(ToolError) as cm:
            P.run_pipeline(ctx, "http://example.invalid/pic.jpg", "telegram",
                           "123", compress=False,
                           downloader=_make_downloader(fixture),
                           sender=bad_send)
        self.assertIn("pipeline send failed", str(cm.exception))

    def test_pipeline_non_media_skips_compress_honestly(self) -> None:
        ctx = _context()
        root = Path(ctx.settings.workspace_dir)
        fixture = root / "notes.txt"
        fixture.write_text("hello world\n" * 100)
        captured: list = []
        report = P.run_pipeline(
            ctx, "http://example.invalid/notes.txt", "whatsapp", "456",
            downloader=_make_downloader(fixture),
            sender=_make_sender(captured),
        )
        self.assertTrue(report["ok"])
        compress = report["stages"][1]
        self.assertEqual(compress["stage"], "compress")
        self.assertFalse(compress["ok"])
        self.assertIn("no compression attempted", compress["reason"])
        # the original still goes out
        self.assertEqual(captured[0][2], str(report["downloaded_path"]))
        self.assertFalse(report["compressed"])

    def test_pipeline_video_without_ffmpeg_skips_honestly(self) -> None:
        import shutil

        import nomorals.tools.compress as C

        if shutil.which("ffmpeg"):
            self.skipTest("ffmpeg present — the honest-skip path needs it absent")
        ctx = _context()
        root = Path(ctx.settings.workspace_dir)
        fake_video = root / "clip.mp4"
        fake_video.write_bytes(b"\x00" * 1024)
        captured: list = []
        report = P.run_pipeline(
            ctx, "http://example.invalid/clip.mp4", "telegram", "123",
            downloader=_make_downloader(fake_video),
            sender=_make_sender(captured),
        )
        compress = report["stages"][1]
        self.assertFalse(compress["ok"])
        self.assertIn("ffmpeg", compress["reason"].lower())
        # original still sent — compression never blocks delivery
        self.assertEqual(captured[0][2], str(report["downloaded_path"]))

    def test_pipeline_rejects_empty_inputs(self) -> None:
        ctx = _context()
        with self.assertRaises(MediaError):
            P.run_pipeline(ctx, "", "telegram", "123")
        with self.assertRaises(ToolError):
            P.run_pipeline(ctx, "http://x/", "", "123")
        with self.assertRaises(ToolError):
            P.run_pipeline(ctx, "http://x/", "telegram", "")

    def test_compress_image_reports_when_not_smaller(self) -> None:
        ctx = _context()
        root = Path(ctx.settings.workspace_dir)
        tiny = root / "tiny.jpg"
        Image.new("RGB", (8, 8), (10, 20, 30)).save(tiny, "JPEG", quality=95)
        report = P.compress_image_for_send(tiny, max_width=1600, quality=95)
        self.assertFalse(report["ok"])
        self.assertIn("not smaller", report["reason"])
        self.assertEqual(report["path"], str(tiny))


# ── 2. vision extract ────────────────────────────────────────────────────────


class StubVisionProvider(LLMProvider):
    name = "stub-vision"

    def __init__(self, text: str, **kw: Any) -> None:
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
        return LLMResponse(text=self.text, model="stub-vlm",
                           provider=self.name,
                           usage=Usage(prompt_tokens=5, completion_tokens=10,
                                       total_tokens=15))


_EXTRACT_JSON = (
    '{"summary": "A red square on a white background.", '
    '"key_text": "HELLO", '
    '"notable_elements": ["red square", "white background", "the word HELLO"], '
    '"scene_type": "photo"}'
)


class VisionExtractTests(unittest.TestCase):
    def _router(self, text: str) -> LLMRouter:
        router = LLMRouter()
        router.add(StubVisionProvider(text), primary=True, name="vlm")
        return router

    def test_extract_returns_structured_chat_ready_result(self) -> None:
        ctx = _context(router=self._router(_EXTRACT_JSON))
        result = V.extract(ctx, _png_bytes())
        self.assertTrue(result["available"])
        self.assertTrue(result["parsed"])
        self.assertEqual(result["summary"],
                         "A red square on a white background.")
        self.assertEqual(result["key_text"], "HELLO")
        self.assertEqual(result["notable_elements"],
                         ["red square", "white background", "the word HELLO"])
        self.assertEqual(result["scene_type"], "photo")
        self.assertEqual(result["provider"], "vlm")
        self.assertEqual(result["format"], "png")

    def test_chat_summary_format(self) -> None:
        ctx = _context(router=self._router(_EXTRACT_JSON))
        result = V.extract(ctx, _png_bytes())
        text = V.format_chat_summary(result)
        self.assertTrue(text.startswith("🖼️ A red square"))
        self.assertIn("• red square", text)
        self.assertIn("📝 Text in image: HELLO", text)
        self.assertIn("png 64x48", text)
        self.assertIn("vlm", text)

    def test_chat_summary_omits_empty_sections(self) -> None:
        text = V.format_chat_summary({
            "summary": "Just a square.", "notable_elements": [],
            "key_text": "", "scene_type": "unknown",
            "format": "png", "width": 8, "height": 8,
            "provider": "", "model": "", "seconds": 0.1,
        })
        self.assertNotIn("📝", text)
        self.assertNotIn("•", text)
        self.assertIn("🖼️ Just a square.", text)

    def test_extract_without_router_raises_explicitly(self) -> None:
        """Acceptance: no vision model → explicit ModelError, never []."""
        ctx = _context(router=None)
        with self.assertRaises(ModelError) as cm:
            V.extract(ctx, _png_bytes())
        message = str(cm.exception).lower()
        self.assertIn("unavailable", message)
        self.assertIn("vision", message)

    def test_extract_non_json_model_reply_degrades_honestly(self) -> None:
        ctx = _context(router=self._router(
            "It's a nice red square, that's all I can say."))
        result = V.extract(ctx, _png_bytes())
        self.assertTrue(result["available"])
        self.assertFalse(result["parsed"])
        self.assertIn("nice red square", result["summary"])
        self.assertEqual(result["notable_elements"], [])
        self.assertIn("did not return structured JSON", result["note"])

    def test_extract_empty_model_reply_gets_placeholder(self) -> None:
        ctx = _context(router=self._router("   "))
        result = V.extract(ctx, _png_bytes())
        self.assertIn("no usable description", result["summary"])

    def test_vision_extract_tool_end_to_end(self) -> None:
        from nomorals.core.policy import CapabilitySet
        from nomorals.tools.registry import ToolRegistry

        ctx = _context(router=self._router(_EXTRACT_JSON))
        root = Path(ctx.settings.workspace_dir)
        (root / "shot.png").write_bytes(_png_bytes())
        reg = ToolRegistry(ctx).register_builtins()
        out = reg.call("vision_extract", path="shot.png",
                       capabilities=CapabilitySet.all())
        self.assertTrue(out.ok, f"tool failed: {out.error}")
        value = out.unwrap()
        self.assertEqual(value["summary"], "A red square on a white background.")
        self.assertIn("🖼️", value["chat_summary"])
        self.assertIn("HELLO", value["chat_summary"])


# ── 3. archive + send ────────────────────────────────────────────────────────


class ArchiveTests(unittest.TestCase):
    def _tree(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        (root / "a.txt").write_text("alpha")
        (root / "sub").mkdir(exist_ok=True)
        (root / "sub" / "b.txt").write_text("beta" * 100)

    def test_zip_round_trip(self) -> None:
        ctx = _context()
        root = Path(ctx.settings.workspace_dir)
        self._tree(root / "docs")
        created = A.zip_create(ctx, ["docs"], name="bundle")
        self.assertTrue(Path(created["path"]).is_file())
        self.assertEqual(created["files"], 2)
        self.assertLess(created["archive_bytes"], created["input_bytes"] * 2)
        extracted = A.zip_extract(ctx, created["path"])
        self.assertEqual(extracted["files"], 2)
        dest = Path(extracted["destination"])
        self.assertEqual((dest / "docs" / "a.txt").read_text(), "alpha")
        self.assertEqual((dest / "docs" / "sub" / "b.txt").read_text(),
                         "beta" * 100)

    def test_zip_create_needs_sources(self) -> None:
        ctx = _context()
        with self.assertRaises(ToolError):
            A.zip_create(ctx, [])
        with self.assertRaises(Exception):  # safe_path NotFound/ValidationError
            A.zip_create(ctx, ["nope/missing.txt"])

    def test_zip_extract_rejects_zip_slip(self) -> None:
        ctx = _context()
        root = Path(ctx.settings.workspace_dir)
        evil = root / "evil.zip"
        with zipfile.ZipFile(evil, "w") as zf:
            zf.writestr("../escape.txt", "pwned")
            zf.writestr("/absolute.txt", "pwned")
        with self.assertRaises(ToolError) as cm:
            A.zip_extract(ctx, "evil.zip")
        self.assertIn("zip-slip", str(cm.exception))
        self.assertFalse((root / "escape.txt").exists())

    def test_zip_extract_rejects_non_zip(self) -> None:
        ctx = _context()
        root = Path(ctx.settings.workspace_dir)
        (root / "plain.txt").write_text("not a zip")
        with self.assertRaises(ToolError):
            A.zip_extract(ctx, "plain.txt")

    def test_zip_send_end_to_end(self) -> None:
        gw = FakeGateway()
        ctx = _context(extras={"gateway": gw})
        root = Path(ctx.settings.workspace_dir)
        self._tree(root / "docs")
        out = A.zip_send(ctx, ["docs"], "telegram", "123", name="bundle")
        self.assertTrue(out["ok"])
        self.assertTrue(out["sent"])
        self.assertEqual(out["platform"], "telegram")
        self.assertEqual(out["chat"], "telegram:123")
        self.assertEqual(len(gw.sent), 1)
        self.assertTrue(gw.sent[0]["path"].endswith("bundle.zip"))
        self.assertTrue(Path(out["path"]).is_file())

    def test_zip_send_works_for_whatsapp_too(self) -> None:
        gw = FakeGateway()
        ctx = _context(extras={"gateway": gw})
        root = Path(ctx.settings.workspace_dir)
        (root / "f.txt").write_text("x")
        out = A.zip_send(ctx, ["f.txt"], "whatsapp", "15551234567")
        self.assertTrue(out["ok"])
        self.assertEqual(gw.sent[0]["platform"], "whatsapp")

    def test_zip_send_without_gateway_fails_loudly(self) -> None:
        ctx = _context()  # no gateway in extras
        root = Path(ctx.settings.workspace_dir)
        (root / "f.txt").write_text("x")
        with self.assertRaises(ToolError) as cm:
            A.zip_send(ctx, ["f.txt"], "telegram", "123")
        self.assertIn("gateway", str(cm.exception).lower())

    def test_unzip_send_sends_every_file(self) -> None:
        gw = FakeGateway()
        ctx = _context(extras={"gateway": gw})
        root = Path(ctx.settings.workspace_dir)
        src = root / "pack"
        src.mkdir()
        for i in range(3):
            (src / f"file{i}.txt").write_text(f"content {i}")
        created = A.zip_create(ctx, ["pack"], name="pack")
        out = A.unzip_send(ctx, created["path"], "telegram", "123")
        self.assertTrue(out["ok"])
        self.assertEqual(len(out["sent"]), 3)
        self.assertEqual(out["failed"], [])
        self.assertEqual(out["skipped"], [])
        self.assertEqual(len(gw.sent), 3)

    def test_unzip_send_caps_and_reports_skipped(self) -> None:
        gw = FakeGateway()
        ctx = _context(extras={"gateway": gw})
        root = Path(ctx.settings.workspace_dir)
        src = root / "pack"
        src.mkdir()
        for i in range(4):
            (src / f"file{i}.txt").write_text(f"content {i}")
        created = A.zip_create(ctx, ["pack"], name="pack")
        out = A.unzip_send(ctx, created["path"], "telegram", "123",
                           max_files=2)
        self.assertEqual(len(out["sent"]), 2)
        self.assertEqual(len(out["skipped"]), 2)
        self.assertIn("max_files=2", out["skipped_reason"])

    def test_archive_tools_registered(self) -> None:
        from nomorals.core.policy import CapabilitySet
        from nomorals.tools.registry import ToolRegistry

        ctx = _context()
        reg = ToolRegistry(ctx).register_builtins()
        for name in ("zip_create", "zip_extract", "zip_send", "unzip_send",
                     "media_pipeline"):
            self.assertIsNotNone(reg.get(name), f"{name} not registered")
        root = Path(ctx.settings.workspace_dir)
        (root / "f.txt").write_text("hi")
        out = reg.call("zip_create", sources=["f.txt"], name="t",
                       capabilities=CapabilitySet.all())
        self.assertTrue(out.ok, f"zip_create failed: {out.error}")
        self.assertEqual(out.unwrap()["files"], 1)


# ── 4. presence guard ────────────────────────────────────────────────────────


class MediaModulePresenceTests(unittest.TestCase):
    """Nothing in the media area may silently disappear (Wave F2 rule)."""

    def test_media_edit_package_intact(self) -> None:
        from nomorals.media_edit import (
            EditStudio, GenerativeBackend, apply_chain, batch_edit,
            edit_image, image_probe, load_image, save_image, validate_ops,
        )
        from nomorals.media_edit import generate, images, jobs, studio, videos

        for mod in (images, videos, generate, studio, jobs):
            self.assertTrue(mod.__name__.startswith("nomorals.media_edit"))
        self.assertTrue(callable(edit_image) and callable(batch_edit)
                        and callable(apply_chain) and callable(validate_ops)
                        and callable(image_probe) and callable(load_image)
                        and callable(save_image))
        self.assertTrue(callable(videos.transcode) and callable(videos.trim)
                        and callable(videos.concat))
        self.assertTrue(callable(generate.get_backend))
        self.assertTrue(hasattr(EditStudio, "__init__")
                        and hasattr(GenerativeBackend, "__init__"))

    def test_tool_modules_intact(self) -> None:
        from nomorals.tools import archive, compress, filesend, media, vision
        from nomorals.tools import media_pipeline

        self.assertTrue(callable(media.download) and callable(media.probe))
        self.assertTrue(callable(compress.compress_file))
        self.assertTrue(callable(filesend.create_file)
                        and callable(filesend.send_file)
                        and callable(filesend.publish_report))
        self.assertTrue(callable(vision.describe) and callable(vision.read_text)
                        and callable(vision.locate) and callable(vision.compare)
                        and callable(vision.extract)
                        and callable(vision.format_chat_summary))
        self.assertTrue(callable(media_pipeline.run_pipeline)
                        and callable(media_pipeline.compress_image_for_send))
        self.assertTrue(callable(archive.zip_create)
                        and callable(archive.zip_extract)
                        and callable(archive.zip_send)
                        and callable(archive.unzip_send))


if __name__ == "__main__":
    unittest.main()

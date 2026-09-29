"""Real vision capability: the router must actually send image bytes to a
model that can see, and OCR must read pixels when no model is available.

Coverage:
* the OCR provider — tesseract over image bytes (fake binary), missing
  binary → actionable install hint, health, no chat
* the router chain — API provider dead → failover to OCR; mock is never a
  vision provider (its fake "[mock vision]" text is not a real description)
* vision model selection — openai_compat / hf_serverless send the image to
  the dedicated vision model/endpoint when configured
* the vision tool — model answer first, OCR floor second, verbatim OCR in
  screen reads, cache keys that keep ocr and non-ocr results apart
* /look end-to-end through the runtime, offline, with a fake tesseract
"""

from __future__ import annotations

import base64
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from tests.test_partner_runtime import FakeRouter, _make_context

from nomorals.core.errors import ProviderError
from nomorals.llm.providers import build_provider
from nomorals.llm.providers.ocr import OCRProvider, ocr_binary, ocr_bytes
from nomorals.llm.router import LLMRouter
from nomorals.tools import vision as vision_module
from nomorals.core.config import load_settings

# A real 1x1 png — bytes the OCR/tesseract path can be handed
_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def make_fake_tesseract(dirpath: Path, text: str = "FAKE OCR OUTPUT") -> str:
    """A stand-in tesseract: prints fixed text no matter what file it gets."""
    script = dirpath / "tesseract"
    script.write_text(f"#!/bin/sh\necho {text!r}\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


# ── OCR provider ─────────────────────────────────────────────────────────────


class OCRProviderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-ocr-")
        self.exe = make_fake_tesseract(Path(self.tmp.name))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_describe_image_runs_tesseract(self) -> None:
        provider = OCRProvider(binary=self.exe)
        response = provider.describe_image(_PNG_1X1, "read this screen")
        self.assertTrue(response.ok)
        self.assertEqual(response.text, "FAKE OCR OUTPUT")
        self.assertEqual(response.provider, "ocr")
        self.assertEqual(response.model, "tesseract")

    def test_ocr_bytes_returns_text(self) -> None:
        self.assertEqual(ocr_bytes(_PNG_1X1, binary=self.exe), "FAKE OCR OUTPUT")

    def test_missing_binary_gives_install_hint(self) -> None:
        import unittest.mock as mock

        with mock.patch("shutil.which", return_value=None):
            with self.assertRaises(ProviderError) as ctx:
                ocr_bytes(_PNG_1X1, binary="/nonexistent/tesseract")
        self.assertIn("pkg install tesseract", str(ctx.exception))

    def test_provider_error_when_no_binary(self) -> None:
        import unittest.mock as mock

        with mock.patch("shutil.which", return_value=None):
            provider = OCRProvider(binary="/nonexistent")
            self.assertFalse(provider.health())
            response = provider.describe_image(_PNG_1X1)
            self.assertFalse(response.ok)
            self.assertIn("tesseract", response.error)

    def test_health_true_with_binary(self) -> None:
        self.assertTrue(OCRProvider(binary=self.exe).health())

    def test_chat_refused(self) -> None:
        from nomorals.llm.base import Message

        with self.assertRaises(ProviderError):
            OCRProvider(binary=self.exe).chat([Message.user("hi")])

    def test_empty_image_refused(self) -> None:
        with self.assertRaises(ProviderError):
            ocr_bytes(b"", binary=self.exe)

    def test_factory_builds_ocr(self) -> None:
        provider = build_provider("ocr", binary=self.exe)
        self.assertIsInstance(provider, OCRProvider)

    def test_ocr_binary_prefers_explicit_path(self) -> None:
        self.assertEqual(ocr_binary(self.exe), self.exe)


# ── router chain: real vision failover ──────────────────────────────────────


class RouterVisionChainTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-ocr-")
        self.exe = make_fake_tesseract(Path(self.tmp.name))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_api_dead_fails_over_to_ocr(self) -> None:
        from nomorals.llm.providers import build_provider

        router = LLMRouter()
        dead = build_provider(
            "openai_compat", base_url="http://127.0.0.1:1/v1", model="whatever",
            timeout=1.0, max_retries=1,
        )
        router.add(dead, primary=True, name="dead_api")
        router.add(OCRProvider(binary=self.exe), primary=False, name="ocr")

        response = router.describe_image(_PNG_1X1, "what is on screen?")
        self.assertTrue(response.ok, msg=response.error)
        self.assertEqual(response.text, "FAKE OCR OUTPUT")
        self.assertEqual(response.provider, "ocr")

    def test_mock_is_not_a_vision_provider(self) -> None:
        from nomorals.llm.providers import build_provider

        mock = build_provider("mock")
        self.assertNotIn("vision", mock.capabilities)
        self.assertIn("vision", build_provider("openai_compat").capabilities)

    def test_ocr_always_in_context_router(self) -> None:
        context, tmp = _make_context()
        try:
            self.assertIn("ocr", context.router.providers())
        finally:
            context.close()
            tmp.cleanup()


# ── vision model selection ──────────────────────────────────────────────────


class _CaptureHttp:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.fail = fail

    def post_json(self, url: str, payload: dict, **kw):  # noqa: ANN001
        self.calls.append((url, payload))
        if self.fail:
            raise RuntimeError("connection refused")
        return _FakeResp({"choices": [{"message": {"content": "a cat"}}], "model": "vision-1"})

    def get(self, url, **kw):  # noqa: ANN001
        raise RuntimeError("no models endpoint in test")


class _FakeResp:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.status = 200
        self.ok = True
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        pass


class VisionModelSelectionTest(unittest.TestCase):
    def test_openai_compat_sends_image_to_vision_model(self) -> None:
        import unittest.mock as mock

        from nomorals.llm.providers import openai_compat as oc

        http = _CaptureHttp()
        # patch before construction: the provider keeps the http client it was given
        with mock.patch.object(oc, "HttpClient", return_value=http):
            provider = oc.OpenAICompatProvider(
                base_url="http://chat.local/v1", model="text-model",
                vision_model="qwen-vl-7b",
            )
            response = provider.describe_image(_PNG_1X1, "describe")
        self.assertTrue(response.ok)
        self.assertEqual(response.text, "a cat")
        url, payload = http.calls[0]
        self.assertEqual(url, "http://chat.local/v1/chat/completions")
        self.assertEqual(payload["model"], "qwen-vl-7b")
        content = payload["messages"][0]["content"]
        self.assertEqual(content[1]["type"], "image_url")
        data_url = content[1]["image_url"]["url"]
        self.assertTrue(data_url.startswith("data:image/png;base64,"))
        # the actual pixels made it into the payload
        self.assertEqual(base64.b64decode(data_url.split(",", 1)[1]), _PNG_1X1)

    def test_vision_endpoint_override(self) -> None:
        import unittest.mock as mock

        from nomorals.llm.providers import openai_compat as oc

        provider = oc.OpenAICompatProvider(
            base_url="http://chat.local/v1", model="text-model",
            vision_model="vlm", vision_base_url="http://vlm.local/v1",
            vision_api_key="sk-test",
        )
        http = _CaptureHttp()
        with mock.patch.object(oc, "HttpClient", return_value=http):
            response = provider.describe_image(_PNG_1X1, "describe")
        self.assertTrue(response.ok)
        url, payload = http.calls[0]
        self.assertEqual(url, "http://vlm.local/v1/chat/completions")
        self.assertEqual(payload["model"], "vlm")

    def test_hf_vision_model_in_legacy_url(self) -> None:
        import unittest.mock as mock

        from nomorals.llm.providers import hf_serverless as hf

        provider = hf.HFServerlessProvider(
            token="hf_test", model="dolphin-text",
            vision_model="Qwen/Qwen2.5-VL-7B-Instruct",
        )
        http = _CaptureHttp()
        with mock.patch.object(provider, "http", http):
            response = provider.describe_image(_PNG_1X1, "describe")
        self.assertTrue(response.ok)
        url, payload = http.calls[0]
        self.assertIn("/models/Qwen/Qwen2.5-VL-7B-Instruct/", url)
        self.assertEqual(payload["model"], "Qwen/Qwen2.5-VL-7B-Instruct")


# ── the vision tool: model first, OCR floor ─────────────────────────────────


class VisionToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-ocr-")
        self.exe = make_fake_tesseract(Path(self.tmp.name))
        self.context, self.ctx_tmp = _make_context()
        self.context.settings.vision.ocr_binary = self.exe

    def tearDown(self) -> None:
        self.context.close()
        self.ctx_tmp.cleanup()
        self.tmp.cleanup()

    def test_ocr_floor_when_no_model(self) -> None:
        self.context.router = None
        result = vision_module.describe(self.context, _PNG_1X1)
        self.assertEqual(result["description"], "FAKE OCR OUTPUT")
        self.assertEqual(result["provider"], "ocr (eng)")
        self.assertEqual(result["ocr_text"], "FAKE OCR OUTPUT")

    def test_model_answer_wins_over_ocr(self) -> None:
        class VisionOKRouter:
            def describe_image(self, data, prompt, **kw):
                from nomorals.llm.base import LLMResponse

                return LLMResponse(text="a photo of a lagoon", model="qwen-vl",
                                   provider="hf")

        self.context.router = VisionOKRouter()
        result = vision_module.describe(self.context, _PNG_1X1)
        self.assertEqual(result["description"], "a photo of a lagoon")
        self.assertEqual(result["provider"], "hf")

    def test_mock_vision_text_is_not_real_vision(self) -> None:
        class FakeVisionRouter:
            def describe_image(self, data, prompt, **kw):
                from nomorals.llm.base import LLMResponse

                return LLMResponse(text="[mock vision] image sha256=abc size=1x1",
                                   model="mock", provider="mock")

        self.context.router = FakeVisionRouter()
        result = vision_module.describe(self.context, _PNG_1X1)
        # the placeholder must not be served as the description
        self.assertTrue(result["description"].startswith("FAKE OCR OUTPUT")
                        or result["description"].startswith("[vision unavailable"),
                        msg=result["description"])
        self.assertNotEqual(result["provider"], "mock")

    def test_no_model_no_ocr_clear_message(self) -> None:
        import unittest.mock as mock

        self.context.router = None
        self.context.settings.vision.ocr_binary = "/nonexistent"
        with mock.patch("shutil.which", return_value=None):
            result = vision_module.describe(self.context, _PNG_1X1)
        self.assertTrue(result["description"].startswith("[vision unavailable"),
                        msg=result["description"])
        self.assertIn("NM_VISION_MODEL", result["description"])

    def test_screen_read_includes_verbatim_ocr(self) -> None:
        self.context.router = None
        workspace = Path(self.context.settings.workspace_dir)
        workspace.mkdir(parents=True, exist_ok=True)
        shot = workspace / "shot.png"
        shot.write_bytes(_PNG_1X1)

        from nomorals.tools.registry import ToolRegistry

        reg = ToolRegistry()
        reg.context = self.context
        vision_module.register(reg)
        spec = reg.get("vision_screen")
        result = spec.fn(path=str(shot), focus="login form")
        self.assertEqual(result["analysis"], "screen")
        self.assertEqual(result["ocr_text"], "FAKE OCR OUTPUT")
        self.assertIn("FOCUS:", result["prompt"])

    def test_ocr_cache_key_distinct(self) -> None:
        self.context.router = None
        cache: dict[str, dict] = {}
        a = vision_module.describe(self.context, _PNG_1X1, cache=cache, ocr=False)
        b = vision_module.describe(self.context, _PNG_1X1, cache=cache, ocr=True)
        self.assertFalse(b.get("cached"))  # ocr variant must not serve the non-ocr cache
        self.assertIsNotNone(b["ocr_text"])
        c = vision_module.describe(self.context, _PNG_1X1, cache=cache, ocr=True)
        self.assertTrue(c.get("cached"))


# ── config: vision env vars ─────────────────────────────────────────────────


class VisionSettingsEnvTest(unittest.TestCase):
    def test_env_reaches_settings(self) -> None:
        s = load_settings(
            env={
                "NM_VISION_MODEL": "Qwen/Qwen2.5-VL-7B-Instruct",
                "NM_VISION_BASE_URL": "http://vlm.local/v1",
                "NM_VISION_API_KEY": "sk-test-key",
                "NM_VISION_OCR_LANGUAGE": "eng+fra",
            },
            use_env_file=False,
        )
        self.assertEqual(s.vision.model, "Qwen/Qwen2.5-VL-7B-Instruct")
        self.assertEqual(s.vision.base_url, "http://vlm.local/v1")
        self.assertEqual(s.vision.api_key, "sk-test-key")
        self.assertEqual(s.vision.ocr_language, "eng+fra")

    def test_defaults(self) -> None:
        s = load_settings(env={}, use_env_file=False)
        self.assertEqual(s.vision.model, "")
        self.assertEqual(s.vision.ocr_language, "eng")
        self.assertFalse(s.vision.ocr_binary)

    def test_api_key_redacted(self) -> None:
        s = load_settings(env={"NM_VISION_API_KEY": "sk-a-very-long-test-key"},
                          use_env_file=False)
        from nomorals.core.config import _asdict

        dumped = _asdict(s, redact=True)
        self.assertNotIn("sk-a-very-long-test-key", str(dumped.get("vision", {})))


# ── /look end-to-end, offline, real OCR ─────────────────────────────────────


class LookEndToEndTest(unittest.TestCase):
    def setUp(self) -> None:
        from tests.test_partner_runtime import FakeAdapter
        from nomorals.agents.partner_runtime import PartnerRuntime
        from nomorals.social.chat.base import ChatKind, ChatRef
        from nomorals.social.chat.gateway import ChatGateway

        self.tmp = tempfile.TemporaryDirectory(prefix="nm-ocr-")
        self.exe = make_fake_tesseract(Path(self.tmp.name), "HELLO FROM THE SCREEN")

        self.context, self.ctx_tmp = _make_context()
        self.context.router = FakeRouter()  # no describe_image → not a real vision
        self.context.settings.vision.ocr_binary = self.exe
        self.registry = None
        from nomorals.tools.registry import ToolRegistry

        reg = ToolRegistry()
        reg.context = self.context
        reg.register_builtins()
        self.context.tools = reg

        self.adapter = FakeAdapter("local")
        self.gateway = ChatGateway({"local": self.adapter}, db=self.context.db)
        self.chat = ChatRef(platform="local", chat_id="console",
                            kind=ChatKind.DM, peer="you")
        self.runtime = PartnerRuntime(self.context, gateway=self.gateway)

        self.workspace = Path(self.context.settings.workspace_dir)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.shot = self.workspace / "screen.png"
        self.shot.write_bytes(_PNG_1X1)

    def tearDown(self) -> None:
        try:
            self.runtime.stop()
        except Exception:
            pass
        self.context.close()
        self.ctx_tmp.cleanup()
        self.tmp.cleanup()

    def test_look_reads_screen_with_ocr_offline(self) -> None:
        self.runtime.handle_control(f"/look {self.shot}", self.chat.key)
        sent = "\n".join(self.adapter.sent)
        self.assertIn("screen read", sent)
        self.assertIn("HELLO FROM THE SCREEN", sent)
        self.assertIn("verbatim text from the pixels", sent)


if __name__ == "__main__":
    unittest.main(verbosity=2)

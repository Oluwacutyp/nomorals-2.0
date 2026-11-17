"""Gapfix R3-50: the web console adapter is built from the factory and
routes the console through it — one web path, no shadow.

Failing-first: on the unwired tree build_adapter("web") returns None,
"web" is in neither PLATFORM_CATALOGUE nor the default platforms, and
WebAdapter serves no HTTP at all.
"""

import json
import time
import types
import unittest
import urllib.error
import urllib.request

from nomorals.core.config import ChatSettings, PartnerSettings, load_settings
from nomorals.social.chat import PLATFORM_CATALOGUE, build_adapter
from nomorals.social.chat.base import ChatKind, ChatRef
from nomorals.social.chat.web import WebAdapter


def _settings(**chat_overrides):
    chat = ChatSettings(**chat_overrides)
    partner = PartnerSettings(platforms="web")
    return types.SimpleNamespace(chat=chat, partner=partner,
                                 resolve=lambda p: p)


def _wait_for_server(adapter, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if getattr(adapter, "_server", None) is not None and adapter.port:
            return True
        time.sleep(0.05)
    return False


def _http(method, url, body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json",
                                          **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        raw = exc.read() or b"{}"
        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, {"ok": False, "error": raw.decode("utf-8", "replace")}


class WebAdapterFactoryTests(unittest.TestCase):
    def test_build_adapter_web_returns_web_adapter(self):
        adapter = build_adapter(_settings(), "web")
        self.assertIsNotNone(adapter)
        assert adapter is not None
        self.assertIsInstance(adapter, WebAdapter)
        self.assertEqual(adapter.name, "web")

    def test_build_adapter_web_disabled_returns_none(self):
        adapter = build_adapter(_settings(web_console_enabled=False), "web")
        self.assertIsNone(adapter)

    def test_web_in_platform_catalogue(self):
        self.assertIn("web", PLATFORM_CATALOGUE)

    def test_web_in_default_platforms(self):
        settings = load_settings(env={}, use_env_file=False)
        self.assertIn("web", settings.partner.platforms.split(","))
        self.assertTrue(settings.chat.web_console_enabled)


class WebConsoleServerTests(unittest.TestCase):
    def _started(self, token=""):
        adapter = WebAdapter(host="127.0.0.1", port=0, token=token)
        seen = []

        def handler(message):
            seen.append(message)
            adapter.send(message.chat, "hello from partner")

        self.assertTrue(adapter.start(handler))
        self.assertTrue(_wait_for_server(adapter), "console server never bound")
        self.addCleanup(adapter.stop)
        return adapter, seen

    def _base(self, adapter):
        return f"http://127.0.0.1:{adapter.port}"

    def test_console_message_round_trip(self):
        adapter, seen = self._started()
        status, payload = _http("POST", self._base(adapter) + "/console/message",
                                {"text": "hello"})
        self.assertEqual(status, 200, payload)
        self.assertTrue(payload.get("ok"), payload)
        self.assertEqual(payload.get("chat"), "web:console")
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].text, "hello")
        self.assertEqual(seen[0].chat.key, "web:console")
        status, payload = _http("GET", self._base(adapter) + "/console/parts")
        self.assertEqual(status, 200, payload)
        self.assertEqual(payload.get("parts"), ["hello from partner"])
        # Drained: a second poll delivers nothing twice.
        status, payload = _http("GET", self._base(adapter) + "/console/parts")
        self.assertEqual(payload.get("parts"), [])

    def test_console_rejects_empty_text(self):
        adapter, _seen = self._started()
        status, payload = _http("POST", self._base(adapter) + "/console/message",
                                {"text": "   "})
        self.assertEqual(status, 400, payload)
        self.assertFalse(payload.get("ok", True))

    def test_console_typing_endpoint(self):
        adapter, _seen = self._started()
        chat = ChatRef(platform="web", chat_id="console", kind=ChatKind.DM)
        self.assertTrue(adapter.typing(chat, seconds=60.0))
        status, payload = _http("GET", self._base(adapter) + "/console/typing")
        self.assertEqual(status, 200, payload)
        self.assertTrue(payload.get("typing"))
        adapter._typing_until[chat.key] = time.time() - 1.0  # noqa: SLF001
        _status, payload = _http("GET", self._base(adapter) + "/console/typing")
        self.assertFalse(payload.get("typing"))

    def test_console_token_enforced(self):
        adapter, seen = self._started(token="s3cret")
        status, _payload = _http("POST", self._base(adapter) + "/console/message",
                                 {"text": "hello"})
        self.assertEqual(status, 403)
        self.assertEqual(seen, [])
        status, payload = _http(
            "POST", self._base(adapter) + "/console/message?token=s3cret",
            {"text": "hello"})
        self.assertEqual(status, 200, payload)
        self.assertEqual(len(seen), 1)

    def test_console_health(self):
        adapter, _seen = self._started()
        status, payload = _http("GET", self._base(adapter) + "/health")
        self.assertEqual(status, 200, payload)
        self.assertTrue(payload.get("ok"))
        self.assertEqual(payload.get("platform"), "web")

    def test_stop_closes_server(self):
        adapter, _seen = self._started()
        status, _payload = _http("GET", self._base(adapter) + "/health")
        self.assertEqual(status, 200)
        adapter.stop()
        deadline = time.time() + 5.0
        refused = False
        while time.time() < deadline:
            try:
                _http("GET", self._base(adapter) + "/health")
            except (urllib.error.URLError, ConnectionError):
                # Refused outright, or accepted-then-reset mid-shutdown:
                # either way the server is no longer answering.
                refused = True
                break
            time.sleep(0.1)
        self.assertTrue(refused, "console server still answering after stop()")


if __name__ == "__main__":
    unittest.main()

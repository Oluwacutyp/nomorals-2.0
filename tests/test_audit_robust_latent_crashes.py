"""Robustness audit: latent crashes from unhandled external data.

External payloads (Telegram updates, webhook bodies, HF responses, IMAP)
are untrusted. These tests pin the audit fixes:

- the TelegramBotAdapter poll loop survives malformed updates instead of
  going deaf (per-update isolation via ``_dispatch_update``);
- malformed entities / dates / media / callbacks are skipped, not raised;
- non-dict webhook bodies get an honest 400, not an AttributeError;
- IMAP connections carry a timeout;
- malformed HF chat / embedding payloads degrade (guards / ModelError);
- previously swallowed failures are now logged (openai_compat key
  rotation, gateway stop, connector 429 retry_after, entity aliases,
  Telethon get_input_entity fallthrough).

All offline: fake HTTP sessions / servers stand in for the network.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any
from unittest import mock

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors.base import ConnectorRateLimitError
from nomorals.connectors.telegram import TelegramConnector
from nomorals.core.errors import ModelError
from nomorals.integrations.email_integration import EmailIntegration
from nomorals.llm.providers.hf_serverless import HFServerlessProvider
from nomorals.llm.providers.openai_compat import OpenAICompatProvider
from nomorals.social.chat.base import (
    ChatAdapter,
    ChatKind,
    ChatMessage,
    ChatRef,
    SendResult,
)
from nomorals.social.chat.gateway import ChatGateway
from nomorals.social.chat.telegram import (
    TelegramAdapter,
    TelegramBotAdapter,
    _entity_aliases,
)
from nomorals.social.chat.webhook import WebhookAdapter, _HookHandler
from nomorals.storage.db import Database


# ── TelegramBotAdapter fakes ──────────────────────────────────────────────

class _FakeResp:
    def __init__(self, payload=None):
        self._payload = payload if payload is not None else {"ok": True, "result": True}

    def json(self):
        return self._payload


class _FakeSession:
    """Scripted Bot API session. ``getUpdates`` behaviour is set by the test."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.pending: list[Any] = []
        self.poll_batches: list[list[Any]] = []
        self.on_poll: Any = None

    def post_json(self, url, params, timeout=None):
        self.calls.append((url, dict(params)))
        if "getMe" in url:
            return _FakeResp({"ok": True, "result": {"id": 1, "username": "bot"}})
        if "getUpdates" in url:
            if params.get("offset") == -1:
                return _FakeResp({"ok": True, "result": self.pending})
            if self.on_poll is not None:
                self.on_poll()
            batch = self.poll_batches.pop(0) if self.poll_batches else []
            return _FakeResp({"ok": True, "result": batch})
        return _FakeResp({"ok": True, "result": True})

    def get(self, url, timeout=None):
        self.calls.append((url, {}))
        return _FakeResp()


def _bot_adapter(session=None, **kw):
    kw.setdefault("token", "test-token-123")
    kw["session"] = session or _FakeSession()
    return TelegramBotAdapter(**kw)


def _good_message(update_id=100, message_id=1, text="hello", chat_type="private"):
    return {
        "update_id": update_id,
        "message": {
            "message_id": message_id,
            "date": 1700000000,
            "chat": {"id": 42, "type": chat_type},
            "from": {"id": 7, "first_name": "Owner"},
            "text": text,
        },
    }


class PollLoopTests(unittest.TestCase):
    def test_poll_loop_survives_malformed_updates(self):
        """One malformed update must not kill the poll thread (bot deaf)."""
        session = _FakeSession()
        adapter = _bot_adapter(session)
        # Stale-skip batch: every entry malformed (bad / missing / non-int id,
        # non-dict entries) — must not raise.
        session.pending = [
            {"update_id": "bad"},
            {"no_id": True},
            "a-string",
            None,
            42,
        ]
        malformed_batch = [
            "not-a-dict",
            None,
            42,
            {"update_id": "bad"},
            {"callback_query": "not-a-dict"},
            {"callback_query": {"id": "q1", "from": {"id": 7},
                                "message": {"chat": {"id": 42, "type": "private"}},
                                "data": 123}},
            {"inline_query": [1, 2]},
            _good_message(update_id=100, message_id=1, text="hello"),
            # Group message: out-of-bounds entity + non-numeric date.
            {
                "update_id": 101,
                "message": {
                    "message_id": 2,
                    "date": "junk-date",
                    "chat": {"id": 42, "type": "group"},
                    "from": {"id": 7, "first_name": "Owner"},
                    "text": "hi",
                    "entities": [{"type": "mention", "offset": 999, "length": 5}],
                },
            },
        ]
        session.poll_batches = [malformed_batch]
        session.on_poll = lambda: adapter._stopped.set()
        seen: list[ChatMessage] = []
        # _run_locked runs until _stopped; on_poll sets it on the first poll.
        adapter._run_locked(seen.append)
        self.assertEqual([m.text for m in seen], ["hello", "hi"])
        # The non-numeric date fell back to ~now instead of raising.
        self.assertLess(abs(seen[1].ts - time.time()), 60)

    def test_dispatch_rejects_non_dict_shapes(self):
        adapter = _bot_adapter()
        seen: list[ChatMessage] = []
        for bad in ("not-a-dict", None, 42, ["list"],
                    {"callback_query": "not-a-dict"},
                    {"callback_query": None},
                    {"inline_query": [1, 2]},
                    {"inline_query": "not-a-dict"}):
            adapter._dispatch_update(bad, seen.append)  # must not raise
        self.assertEqual(seen, [])

    def test_convert_malformed_entities_skipped(self):
        adapter = _bot_adapter()
        adapter._bot_username = "bot"
        base = {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "date": 1700000000,
                "chat": {"id": 42, "type": "group"},
                "from": {"id": 7, "first_name": "Owner"},
                "text": "@bot hi",
            },
        }
        # Valid mention still detected alongside garbage entries.
        update = json.loads(json.dumps(base))
        update["message"]["entities"] = [
            {"type": "mention", "offset": 0, "length": 4},
            {"type": "mention", "offset": 999, "length": 5},
            {"type": "mention", "offset": -1, "length": 2},
            {"type": "mention", "offset": "x", "length": 1},
            {"type": "mention", "offset": True, "length": 1},
            "not-a-dict",
            {"type": "mention"},
            {"type": "mention", "offset": 0, "length": -3},
        ]
        msg = adapter._convert(update)
        self.assertIsNotNone(msg)
        assert msg is not None
        self.assertTrue(msg.mentioned)
        # Out-of-bounds-only entities: no mention, no raise.
        update2 = json.loads(json.dumps(base))
        update2["message"]["entities"] = [
            {"type": "mention", "offset": 999, "length": 5},
        ]
        msg2 = adapter._convert(update2)
        self.assertIsNotNone(msg2)
        assert msg2 is not None
        self.assertFalse(msg2.mentioned)
        # Non-list entities container: skipped, message survives.
        update3 = json.loads(json.dumps(base))
        update3["message"]["entities"] = "junk"
        self.assertIsNotNone(adapter._convert(update3))
        # Non-dict message / sender / chat: dropped safely, not raised.
        self.assertIsNone(adapter._convert({"message": "junk"}))
        self.assertIsNone(adapter._convert({"message": {"from": "junk"}}))

    def test_convert_non_numeric_date_falls_back_to_now(self):
        adapter = _bot_adapter()
        for bad_date in ("junk", None, [1], {"x": 1}):
            update = _good_message()
            update["message"]["date"] = bad_date
            msg = adapter._convert(update)
            self.assertIsNotNone(msg)
            assert msg is not None
            self.assertIsInstance(msg.ts, float)
            self.assertLess(abs(msg.ts - time.time()), 60)

    def test_inbound_media_malformed_skipped(self):
        adapter = _bot_adapter()
        # Photo entries: non-dict, missing file_id, non-string file_id.
        msg = {
            "photo": ["junk", {}, {"file_id": 123}, {"file_id": None}],
            "text": "hi",
        }
        self.assertEqual(adapter._inbound_media(msg, ChatKind.DM), [])
        # Every single-media slot malformed: no raise, no download.
        for key in ("document", "video", "voice", "audio", "video_note"):
            self.assertEqual(
                adapter._inbound_media({key: "junk"}, ChatKind.DM), [])
            self.assertEqual(
                adapter._inbound_media({key: {"file_id": 42}}, ChatKind.DM), [])
            self.assertEqual(
                adapter._inbound_media({key: {}}, ChatKind.DM), [])
        self.assertEqual(adapter._inbound_media("not-a-dict", ChatKind.DM), [])

    def test_callback_non_string_data_rejected(self):
        adapter = _bot_adapter()
        for bad in (123, None, ["x"], {"data": 1}, 4.5):
            self.assertIsNone(adapter._verify_callback(bad))
        # Non-string / non-dict callback shapes never raise, never deliver.
        seen: list[ChatMessage] = []
        adapter._handle_callback("not-a-dict", seen.append)
        adapter._handle_callback(None, seen.append)
        adapter._handle_callback(
            {"id": "q1", "from": {"id": 7}, "data": 123}, seen.append)
        # A non-dict "message" inside an otherwise valid callback must not
        # crash: it degrades to an empty chat and still delivers.
        adapter._handle_callback(
            {"id": "q1", "from": {"id": 7}, "message": "junk",
             "data": adapter._sign_callback("/inventory")}, seen.append)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].text, "/inventory")
        adapter._handle_inline_query("not-a-dict")
        adapter._handle_inline_query(None)
        adapter._handle_inline_query({"id": "q", "from": {}, "query": 123})
        # No further deliveries from the malformed inline queries.
        self.assertEqual(len(seen), 1)


# ── Webhook ───────────────────────────────────────────────────────────────

def _post(port, body_bytes, path="/hook/test"):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=body_bytes, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


class WebhookTests(unittest.TestCase):
    def _serve(self, adapter):
        handler_cls = type("_TestHook", (_HookHandler,), {"adapter": adapter})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        return server.server_address[1]

    def test_webhook_non_dict_body_400(self):
        adapter = WebhookAdapter(port=0)
        adapter._handler = lambda m: None
        port = self._serve(adapter)
        for body in (b"[1, 2]", b'"just a string"', b"42", b"null", b"true"):
            status, payload = _post(port, body)
            self.assertEqual(status, 400, body)
            self.assertIn("JSON body must be an object", payload.get("error", ""))

    def test_webhook_dict_body_still_works(self):
        adapter = WebhookAdapter(port=0)
        seen: list[ChatMessage] = []
        adapter._handler = seen.append
        port = self._serve(adapter)
        status, payload = _post(
            port, json.dumps({"text": "hello", "sender": "sentinel"}).encode())
        self.assertEqual(status, 200)
        self.assertTrue(payload.get("ok"))
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].text, "hello")


# ── IMAP timeout ──────────────────────────────────────────────────────────

class _FakeMail:
    def __init__(self, host, port, timeout=None):
        _FakeMail.calls.append((host, port, timeout))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def login(self, *args):
        pass

    def select(self, *args):
        return ("OK", [])

    def search(self, *args):
        return ("OK", [b"1"])

    def fetch(self, *args):
        return ("OK", [(None, b"From: a@b.com\r\nSubject: hi\r\n\r\nbody")])

    def uid(self, *args):
        return ("OK", [])

    def store(self, *args):
        return ("OK", [])

    def expunge(self):
        return ("OK", [])


_FakeMail.calls: list[tuple] = []


class _StubAccounts:
    def get_credential(self, *args):
        return SimpleNamespace(password="pw")


class ImapTimeoutTests(unittest.TestCase):
    def test_all_imap_connections_use_timeout_30(self):
        integration = EmailIntegration(_StubAccounts(), object())
        _FakeMail.calls.clear()
        with mock.patch(
            "nomorals.integrations.email_integration.imaplib.IMAP4_SSL",
            new=_FakeMail,
        ):
            asyncio.run(integration._read_imap("a@gmail.com", 10, "INBOX", False))
            asyncio.run(integration._mark_read_imap("1", "a@gmail.com"))
            asyncio.run(integration._delete_imap("1", "a@gmail.com"))
        self.assertEqual(len(_FakeMail.calls), 3)
        for host, port, timeout in _FakeMail.calls:
            self.assertEqual(host, "imap.gmail.com")
            self.assertEqual(port, 993)
            self.assertEqual(timeout, 30)


# ── HF provider parsing ───────────────────────────────────────────────────

class HFParseTests(unittest.TestCase):
    def _provider(self):
        return HFServerlessProvider(token="<redacted>", model="test-model")

    def test_parse_chat_malformed_shapes(self):
        provider = self._provider()
        # Non-dict choice + non-dict usage: used to raise AttributeError.
        resp = provider._parse_chat(
            {"choices": ["not-a-dict"], "usage": "junk", "model": "test-model"})
        self.assertEqual(resp.text, "")
        self.assertEqual(resp.usage.prompt_tokens, 0)
        # Non-dict message + non-numeric usage values: coerced, not raised.
        resp = provider._parse_chat({
            "choices": [{"message": "not-a-dict", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": "abc", "completion_tokens": [1],
                      "total_tokens": None},
        })
        self.assertEqual(resp.text, "")
        self.assertEqual(resp.usage.prompt_tokens, 0)
        self.assertEqual(resp.usage.completion_tokens, 0)
        self.assertEqual(resp.usage.total_tokens, 0)
        # Non-list choices: falls back to the text-generation shape.
        resp = provider._parse_chat({"choices": "not-a-list"})
        self.assertEqual(resp.text, "")
        # Happy path unchanged.
        resp = provider._parse_chat({
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "model": "test-model",
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        })
        self.assertEqual(resp.text, "hi")
        self.assertEqual(resp.usage.total_tokens, 8)

    def test_parse_embeddings_malformed_raise_model_error(self):
        parse = HFServerlessProvider._parse_embeddings
        # data["data"] assumed a list — a string raised raw TypeError.
        with self.assertRaises(ModelError):
            parse({"data": "not-a-list"}, 1)
        # float(x) on a non-numeric embedding raised raw ValueError.
        with self.assertRaises(ModelError):
            parse({"data": [{"embedding": ["x"]}]}, 1)
        with self.assertRaises(ModelError):
            parse([["x"]], 1)
        with self.assertRaises(ModelError):
            parse(42, 1)
        # Happy paths unchanged.
        self.assertEqual(
            parse({"data": [{"index": 0, "embedding": [1.0, 2.0]}]}, 1),
            [[1.0, 2.0]])
        self.assertEqual(parse([[0.5]], 1), [[0.5]])


# ── Swallow → log ─────────────────────────────────────────────────────────

class _RaisingStore(dict):
    def __setitem__(self, key, value):
        raise RuntimeError("store broken")


class _BadHttp:
    def __init__(self):
        self._headers = _RaisingStore()

    @property
    def headers(self):
        raise RuntimeError("headers broken")


class SwallowToLogTests(unittest.TestCase):
    def test_apply_key_logs_warning_on_broken_header_store(self):
        provider = OpenAICompatProvider(
            base_url="http://localhost:11434/v1", api_key="k", model="m")
        provider.http = _BadHttp()
        with self.assertLogs(
            "nomorals.llm.providers.openai_compat", level="WARNING"
        ) as logs:
            provider._apply_key("new-key")
        self.assertEqual(provider.api_key, "new-key")
        self.assertTrue(any("Authorization" in line for line in logs.output))

    def test_gateway_stop_logs_and_continues(self):
        class _Good(ChatAdapter):
            name = "good"

            def __init__(self):
                super().__init__()
                self.stopped_flag = False

            def run(self, handler):
                pass

            def send(self, chat, text, **kw):
                return SendResult(ok=True, platform=self.name)

            def stop(self):
                self.stopped_flag = True

        class _Bad(ChatAdapter):
            name = "bad"

            def run(self, handler):
                pass

            def send(self, chat, text, **kw):
                return SendResult(ok=True, platform=self.name)

            def stop(self):
                raise RuntimeError("boom")

        good = _Good()
        gateway = ChatGateway(adapters={"bad": _Bad(), "good": good})
        with self.assertLogs(
            "nomorals.social.chat.gateway", level="WARNING"
        ) as logs:
            gateway.stop()  # must not raise
        self.assertTrue(good.stopped_flag)
        self.assertTrue(any("bad" in line for line in logs.output))

    def test_connector_429_retry_after_parse_failure_logs_debug(self):
        import json as _json

        class _FakeResp:
            def __init__(self, status, payload):
                self.status = status
                self._payload = payload

            @property
            def ok(self):
                return 200 <= self.status < 300

            @property
            def text(self):
                return _json.dumps(self._payload)

            def json(self):
                return self._payload

        class _FakeHttp:
            def __init__(self):
                self.routes: list = []

            def route(self, method, path, response):
                self.routes.append((method.upper(), path, response))

            def post_json(self, url, payload, **kw):
                for rm, rp, resp in self.routes:
                    if rm == "POST" and rp in url:
                        return resp
                return _FakeResp(404, {"ok": False, "description": "not mocked"})

        http = _FakeHttp()
        http.route("POST", "/getMe", _FakeResp(200, {
            "ok": True, "result": {"id": 1, "is_bot": True,
                                   "first_name": "Bot", "username": "test_bot",
                                   "can_join_groups": True,
                                   "can_read_all_group_messages": False}}))
        vault = CredentialVault(Database(":memory:"), master_passphrase="test")
        conn = TelegramConnector(vault, http=http)
        self.assertTrue(conn.connect(token="tok123").ok)
        http.route("POST", "/sendMessage", _FakeResp(429, {
            "ok": False, "description": "Too Many Requests",
            "parameters": {"retry_after": "not-a-number"},
        }))
        with self.assertLogs(
            "nomorals.connectors.telegram", level="DEBUG"
        ) as logs:
            with self.assertRaises(ConnectorRateLimitError) as ctx:
                conn.send_message(1, "hi")
        # The 1s default stands.
        self.assertEqual(ctx.exception.retry_after, 1.0)
        self.assertTrue(any("retry_after" in line for line in logs.output))

    def test_entity_aliases_logs_debug_on_raising_entity(self):
        class _Boom:
            @property
            def id(self):
                raise RuntimeError("boom")

        with self.assertLogs(
            "nomorals.social.chat.telegram", level="DEBUG"
        ) as logs:
            self.assertEqual(_entity_aliases(_Boom()), [])
        self.assertTrue(any("alias" in line for line in logs.output))

    def test_resolve_get_input_entity_fallthrough_logs_debug(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = TelegramAdapter(
                api_id=1, api_hash="hash", media_dir=tmp)

            class _FakeClient:
                async def get_input_entity(self, peer):
                    raise ValueError("no peer")

                async def get_entity(self, peer):
                    raise ValueError("no entity")

            adapter._client = _FakeClient()
            with self.assertLogs(
                "nomorals.social.chat.telegram", level="DEBUG"
            ) as logs:
                result = asyncio.run(adapter._resolve(ChatRef(
                    platform="telegram", chat_id="123", kind=ChatKind.DM)))
            # Fallthrough: raw integer ID is the last resort.
            self.assertEqual(result, 123)
            self.assertTrue(
                any("get_input_entity" in line for line in logs.output))


if __name__ == "__main__":
    unittest.main()

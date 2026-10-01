"""Telegram Bot API adapter + webhook adapter."""

from __future__ import annotations

import json
import threading
import time
import unittest
import urllib.request

from nomorals.core.errors import ValidationError
from nomorals.social.chat.base import ChatKind, ChatRef, MediaRef
from nomorals.social.chat.telegram import TelegramBotAdapter, _chunk_text
from nomorals.social.chat.webhook import WebhookAdapter


class FakeResponse:
    def __init__(self, payload: dict, content: bytes = b""):
        self._payload = payload
        self.content = content

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        pass


class FakeSession:
    """Records Bot API calls; canned getMe/getUpdates/getFile."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.updates: list[dict] = []

    def post(self, url: str, **kwargs: object) -> FakeResponse:
        method = url.rsplit("/", 1)[-1]
        params = kwargs.get("json") or {}
        files = kwargs.get("files")
        self.calls.append((method, {"params": params, "files": bool(files)}))
        if method == "getMe":
            return FakeResponse({"ok": True,
                                 "result": {"id": 999, "username": "HerBot"}})
        if method == "getUpdates":
            return FakeResponse({"ok": True, "result": self.updates})
        if method == "getFile":
            return FakeResponse({"ok": True, "result": {
                "file_path": "photos/x.jpg", "file_size": 1234}})
        if method == "sendMessage":
            if params.get("text") == "FAIL":
                return FakeResponse({"ok": False, "description": "bad request"})
            return FakeResponse({"ok": True, "result": {"message_id": 42}})
        if method in ("sendPhoto", "sendDocument", "sendVideo", "sendVoice",
                      "sendAudio", "sendChatAction"):
            return FakeResponse({"ok": True, "result": {"message_id": 43}})
        return FakeResponse({"ok": True, "result": True})

    def get(self, url: str, **kwargs: object) -> FakeResponse:
        return FakeResponse({"ok": True}, content=b"fake-bytes")


def _bot(**kwargs: object) -> TelegramBotAdapter:
    kwargs.setdefault("session", FakeSession())
    return TelegramBotAdapter(token="test-token", **kwargs)  # type: ignore[arg-type]


class ChunkTests(unittest.TestCase):
    def test_short_passthrough(self) -> None:
        self.assertEqual(_chunk_text("hi", 4096), ["hi"])

    def test_splits_on_newlines(self) -> None:
        text = "a\n" * 3000  # 6000 chars
        chunks = _chunk_text(text, 4096)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(all(len(c) <= 4096 for c in chunks))
        self.assertEqual("".join(c + "\n" for c in chunks).replace("\n\n", "\n")[:10],
                         text[:10])

    def test_hard_split_without_newlines(self) -> None:
        chunks = _chunk_text("x" * 5000, 4096)
        self.assertEqual([len(c) for c in chunks], [4096, 904])


class BotApiTests(unittest.TestCase):
    def test_preflight_sets_identity(self) -> None:
        bot = _bot()
        bot.preflight()
        self.assertEqual(bot._bot_username, "HerBot")
        self.assertEqual(bot._bot_id, 999)

    def test_api_error_raises(self) -> None:
        bot = _bot()
        with self.assertRaises(ValidationError):
            bot._api("sendMessage", chat_id=1, text="FAIL")

    def test_requires_token(self) -> None:
        with self.assertRaises(ValidationError):
            TelegramBotAdapter(token="")

    def test_send_splits_long_text(self) -> None:
        sess = FakeSession()
        bot = _bot(session=sess)
        result = bot.send(ChatRef(platform="telegram-bot", chat_id="123"),
                          "y" * 5000)
        self.assertTrue(result.ok)
        sends = [c for c in sess.calls if c[0] == "sendMessage"]
        self.assertEqual(len(sends), 2)
        self.assertEqual(len(sends[0][1]["params"]["text"]), 4096)

    def test_send_reply_parameters(self) -> None:
        sess = FakeSession()
        bot = _bot(session=sess)
        bot.send(ChatRef(platform="telegram-bot", chat_id="123"), "hi",
                 reply_to="7")
        params = sess.calls[-1][1]["params"]
        self.assertEqual(params["reply_parameters"], {"message_id": 7})

    def test_send_media_picks_photo(self) -> None:
        import tempfile, os
        sess = FakeSession()
        bot = _bot(session=sess)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as fh:
            fh.write(b"img")
            path = fh.name
        try:
            result = bot.send_media(
                ChatRef(platform="telegram-bot", chat_id="123"),
                MediaRef(path=path, kind="image"))
        finally:
            os.unlink(path)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(sess.calls[-1][0], "sendPhoto")

    def test_send_media_missing_file(self) -> None:
        bot = _bot()
        result = bot.send_media(ChatRef(platform="telegram-bot", chat_id="1"),
                                MediaRef(path="/nope.jpg", kind="image"))
        self.assertFalse(result.ok)

    def test_typing(self) -> None:
        sess = FakeSession()
        bot = _bot(session=sess)
        self.assertTrue(bot.typing(ChatRef(platform="telegram-bot", chat_id="1")))
        self.assertEqual(sess.calls[-1][0], "sendChatAction")


class ConvertTests(unittest.TestCase):
    def _update(self, **msg: object) -> dict:
        base = {"update_id": 10, "message": {
            "message_id": 5, "date": 1700000000,
            "from": {"id": 111, "first_name": "Ada", "username": "ada"},
            "chat": {"id": 111, "type": "private"},
            "text": "hello"}}
        base["message"].update(msg)  # type: ignore[typeddict-item]
        return base

    def test_dm(self) -> None:
        bot = _bot()
        bot._bot_username = "HerBot"
        m = bot._convert(self._update())
        assert m is not None
        self.assertEqual(m.chat.key, "telegram-bot:111")
        self.assertEqual(m.chat.kind, ChatKind.DM)
        self.assertEqual(m.text, "hello")
        self.assertEqual(m.sender, "ada")
        self.assertTrue(m.incoming)

    def test_other_bots_skipped(self) -> None:
        bot = _bot()
        upd = self._update()
        upd["message"]["from"] = {"id": 222, "is_bot": True, "first_name": "Spambot"}
        self.assertIsNone(bot._convert(upd))

    def test_group_mention_flag(self) -> None:
        bot = _bot()
        bot._bot_username = "HerBot"
        upd = self._update(chat={"id": -5, "type": "supergroup", "title": "Crew"},
                           text="hey @HerBot what up",
                           entities=[{"type": "mention", "offset": 4, "length": 7}])
        m = bot._convert(upd)
        assert m is not None
        self.assertEqual(m.chat.kind, ChatKind.GROUP)
        self.assertTrue(m.mentioned)

    def test_allowlist(self) -> None:
        bot = _bot(chat_allow="111")
        self.assertIsNotNone(bot._convert(self._update()))
        upd = self._update(chat={"id": 999, "type": "private"})
        self.assertIsNone(bot._convert(upd))

    def test_photo_media_downloaded(self) -> None:
        import tempfile
        sess = FakeSession()
        with tempfile.TemporaryDirectory() as tmp:
            bot = _bot(session=sess, media_dir=tmp)
            upd = self._update(photo=[{"file_id": "abc", "file_size": 100},
                                      {"file_id": "def", "file_size": 9000}])
            m = bot._convert(upd)
            assert m is not None
            self.assertEqual(len(m.media), 1)
            self.assertEqual(m.media[0].kind, "image")
            # largest file_id chosen
            self.assertIn("def", m.media[0].path)


class WebhookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = WebhookAdapter(port=0, token="s3cret")
        self.received: list = []
        self.thread = threading.Thread(
            target=self.adapter.run, args=(self.received.append,), daemon=True)
        self.thread.start()
        for _ in range(100):
            if self.adapter.port:
                break
            time.sleep(0.05)
        self.base = f"http://127.0.0.1:{self.adapter.port}"

    def tearDown(self) -> None:
        self.adapter.stop()
        self.thread.join(timeout=5)

    def _post(self, path: str, body: dict, token: str | None = "s3cret") -> tuple[int, dict]:
        url = self.base + path
        if token:
            url += f"?token={token}"
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_hook_round_trip(self) -> None:
        code, data = self._post("/hook/sentinel",
                                {"text": "BTC wicked below 60k", "sender": "sentinel"})
        self.assertEqual(code, 200)
        self.assertTrue(data["ok"])
        self.assertEqual(len(self.received), 1)
        msg = self.received[0]
        self.assertEqual(msg.text, "BTC wicked below 60k")
        self.assertEqual(msg.chat.platform, "webhook")
        self.assertEqual(msg.sender, "sentinel")

    def test_bad_token_rejected(self) -> None:
        code, _ = self._post("/hook/sentinel", {"text": "hi"}, token="wrong")
        self.assertEqual(code, 403)
        self.assertEqual(self.received, [])

    def test_empty_text_rejected(self) -> None:
        code, _ = self._post("/hook/sentinel", {"text": "  "})
        self.assertEqual(code, 400)

    def test_send_captured_for_polling(self) -> None:
        chat = ChatRef(platform="webhook", chat_id="alerts")
        result = self.adapter.send(chat, "noted")
        self.assertTrue(result.ok)
        self.assertEqual(self.adapter.take_parts("webhook:alerts"), ["noted"])
        self.assertEqual(self.adapter.take_parts("webhook:alerts"), [])

    def test_health(self) -> None:
        with urllib.request.urlopen(self.base + "/health",
                                    timeout=5) as resp:
            data = json.loads(resp.read())
        self.assertEqual(resp.status, 200)
        self.assertTrue(data["ok"])


if __name__ == "__main__":
    unittest.main()

"""Gateway transmission integration tests: chunking, retry, durable, nm sent."""

import os
import tempfile
import unittest
from types import SimpleNamespace

from nomorals.cmdline.commands.sent import _cmd_sent
from nomorals.cmdline.parser import _parser
from nomorals.core.retry import BackoffPolicy, JitterStyle
from nomorals.social.chat.base import ChatAdapter, ChatRef, MediaRef, SendResult
from nomorals.social.chat.gateway import ChatGateway
from nomorals.social.chat.platform_limits import (
    PlatformLimits,
    register_limits,
)
from nomorals.storage.db import Database

_NO_SLEEP = BackoffPolicy(base=0.0, factor=2.0, cap=0.0,
                          jitter=False, jitter_style=JitterStyle.NONE)


class FakeAdapter(ChatAdapter):
    """Records every send; scripted failures by exact error string."""

    def __init__(self, name: str = "telegram") -> None:
        super().__init__(media_dir="/tmp/nm-transmission-media")
        self.name = name
        self.sent: list[tuple[str, dict]] = []  # (text, kwargs)
        self.media: list[tuple[str, str]] = []  # (kind, caption)
        self.typing_calls: list[str] = []
        self.fail_script: list[str] = []  # errors to return, in order

    def run(self, handler) -> None:  # pragma: no cover - never started
        raise NotImplementedError

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
             buttons=None, reply_keyboard=None,
             parse_mode: str = "") -> SendResult:
        if self.fail_script:
            return SendResult(ok=False, platform=self.name,
                              error=self.fail_script.pop(0))
        kwargs: dict = {"reply_to": reply_to}
        if buttons is not None:
            kwargs["buttons"] = buttons
        if reply_keyboard is not None:
            kwargs["reply_keyboard"] = reply_keyboard
        if parse_mode:
            kwargs["parse_mode"] = parse_mode
        self.sent.append((text, kwargs))
        return SendResult(ok=True, platform=self.name,
                          message_id=f"m{len(self.sent)}")

    def send_media(self, chat: ChatRef, media: MediaRef, *,
                   caption: str = "") -> SendResult:
        if self.fail_script:
            return SendResult(ok=False, platform=self.name,
                              error=self.fail_script.pop(0))
        self.media.append((media.kind, caption))
        return SendResult(ok=True, platform=self.name,
                          message_id=f"media-{len(self.media)}")

    def typing(self, chat: ChatRef, seconds: float = 3.0,
               action: str = "typing") -> bool:
        self.typing_calls.append(chat.key)
        return True


def make_db() -> Database:
    db = Database(":memory:")
    db.migrate()
    return db


class TestGatewayDirect(unittest.TestCase):
    def test_long_text_chunked_and_labelled(self) -> None:
        fake = FakeAdapter("telegram")
        gw = ChatGateway({"telegram": fake}, retry_policy=_NO_SLEEP)
        text = ("Lorem ipsum dolor sit amet. " * 700) + "\n\n" + ("Tail. " * 300)
        result = gw.send("telegram", ChatRef(platform="telegram", chat_id="1"),
                         text, buttons=[[("B", "cb")]])
        self.assertTrue(result.ok, result.error)
        self.assertGreater(len(fake.sent), 1)
        n = len(fake.sent)
        contents = []
        for i, (chunk, kwargs) in enumerate(fake.sent):
            self.assertLessEqual(len(chunk), 4096)
            self.assertTrue(chunk.endswith(f" ({i + 1}/{n})"),
                            f"chunk {i} missing label")
            contents.append(chunk[: -len(f" ({i + 1}/{n})")])
            if i < n - 1:
                self.assertNotIn("buttons", kwargs)
            else:
                self.assertIn("buttons", kwargs)
        self.assertEqual("".join(contents), text)

    def test_direct_transient_retried(self) -> None:
        fake = FakeAdapter("telegram")
        fake.fail_script = ["429 Too Many Requests", "503 Service Unavailable"]
        gw = ChatGateway({"telegram": fake}, retry_policy=_NO_SLEEP)
        result = gw.send("telegram", ChatRef(platform="telegram", chat_id="1"),
                         "hello")
        self.assertTrue(result.ok)
        self.assertEqual(len(fake.sent), 1)

    def test_direct_terminal_stops(self) -> None:
        fake = FakeAdapter("telegram")
        fake.fail_script = ["400 chat not found"]
        gw = ChatGateway({"telegram": fake}, retry_policy=_NO_SLEEP)
        result = gw.send("telegram", ChatRef(platform="telegram", chat_id="1"),
                         "hello")
        self.assertFalse(result.ok)
        self.assertEqual(len(fake.sent), 0)

    def test_durable_requested_without_db_falls_back_direct(self) -> None:
        fake = FakeAdapter("telegram")
        gw = ChatGateway({"telegram": fake}, retry_policy=_NO_SLEEP)
        result = gw.send("telegram", ChatRef(platform="telegram", chat_id="1"),
                         "hello", durable=True)
        self.assertTrue(result.ok)  # no silent drop
        self.assertEqual(len(fake.sent), 1)


class TestGatewayDurable(unittest.TestCase):
    def setUp(self) -> None:
        self.db = make_db()

    def test_durable_send_receipted(self) -> None:
        fake = FakeAdapter("telegram")
        gw = ChatGateway({"telegram": fake}, db=self.db, durable_outbox=True,
                         retry_policy=_NO_SLEEP)
        chat = ChatRef(platform="telegram", chat_id="1")
        r1 = gw.send("telegram", chat, "first")
        r2 = gw.send("telegram", chat, "second")
        self.assertTrue(r1.ok)
        self.assertTrue(r2.ok)
        rows = gw.receipts(chat_key="telegram:1")
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["state"] == "sent" for r in rows))
        self.assertEqual({r["text_preview"] for r in rows},
                         {"first", "second"})

    def test_durable_terminal_cancels_remaining_chunks(self) -> None:
        fake = FakeAdapter("telegram")
        fake.fail_script = ["400 chat not found"]
        gw = ChatGateway({"telegram": fake}, db=self.db, durable_outbox=True,
                         retry_policy=_NO_SLEEP)
        chat = ChatRef(platform="telegram", chat_id="1")
        result = gw.send("telegram", chat, "x" * 9000)
        self.assertFalse(result.ok)
        rows = gw.receipts(chat_key="telegram:1")
        self.assertEqual(len(rows), 3)
        states = sorted(r["state"] for r in rows)
        self.assertEqual(states, ["dropped", "dropped", "failed_terminal"])

    def test_typing_never_durable(self) -> None:
        fake = FakeAdapter("telegram")
        gw = ChatGateway({"telegram": fake}, db=self.db, durable_outbox=True,
                         retry_policy=_NO_SLEEP)
        chat = ChatRef(platform="telegram", chat_id="1")
        self.assertTrue(gw.typing("telegram", chat))
        self.assertEqual(fake.typing_calls, ["telegram:1"])
        self.assertEqual(gw.receipts(chat_key="telegram:1"), [])

    def test_nm_sent_command(self) -> None:
        fake = FakeAdapter("telegram")
        gw = ChatGateway({"telegram": fake}, db=self.db, durable_outbox=True,
                         retry_policy=_NO_SLEEP)
        chat = ChatRef(platform="telegram", chat_id="1")
        gw.send("telegram", chat, "hello")
        ctx = SimpleNamespace(db=self.db)
        args = SimpleNamespace(to="telegram:1", since="24h", limit=50,
                               json=False)
        self.assertEqual(_cmd_sent(args, ctx), 0)
        args_empty = SimpleNamespace(to="telegram:999", since="24h",
                                     limit=50, json=False)
        self.assertEqual(_cmd_sent(args_empty, ctx), 0)
        args_bad = SimpleNamespace(to="telegram:1", since="bogus", limit=50,
                                   json=False)
        self.assertEqual(_cmd_sent(args_bad, ctx), 2)

    def test_sent_parser(self) -> None:
        args = _parser().parse_args(
            ["sent", "--to", "telegram:1", "--since", "1h"])
        self.assertEqual(args.command, "sent")
        self.assertEqual(args.to, "telegram:1")
        self.assertEqual(args.since, "1h")


class TestGatewaySendFile(unittest.TestCase):
    def setUp(self) -> None:
        try:
            register_limits(PlatformLimits(
                platform="testcap", max_text=100, max_caption=10,
                max_file_mb=0.000001, supports_buttons=False,
                supports_html=False, supports_media_edit=False,
                rate_per_hour=0, cost_model="free",
            ))
        except Exception:
            pass
        fd, self.path = tempfile.mkstemp(suffix=".bin")
        with os.fdopen(fd, "wb") as fh:
            fh.write(b"x" * 100)

    def tearDown(self) -> None:
        try:
            os.unlink(self.path)
        except OSError:
            pass
        from nomorals.social.chat.platform_limits import _LIMITS

        _LIMITS.pop("testcap", None)

    def test_media_over_cap_goes_as_document_and_caption_chunks(self) -> None:
        fake = FakeAdapter("testcap")
        gw = ChatGateway({"testcap": fake}, retry_policy=_NO_SLEEP)
        chat = ChatRef(platform="testcap", chat_id="1")
        caption = "0123456789" * 4  # 40 chars > max_caption 10
        result = gw.send_file("testcap", chat, self.path, caption=caption)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(len(fake.media), 1)
        kind, media_caption = fake.media[0]
        self.assertEqual(kind, "document")
        self.assertLessEqual(len(media_caption), 10)
        text_pieces = [text for text, _ in fake.sent]
        self.assertEqual(media_caption + "".join(text_pieces), caption)


if __name__ == "__main__":
    unittest.main()

"""WisdomKeeper reading: inline Next / Previous buttons on the personal bot.

``/wisdom read`` pages carry book-style paging buttons (Previous on every
page but the first, Next on every page but the last). Tapping a button
re-runs the exact printed ``/wisdom read`` command for that page, so the
buttons are real routable commands — the printed commands stay as the
fallback. Failing-first for the owner's "like a real book" request.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.agents.partner.runtime_wisdom import RuntimeWisdomMixin
from nomorals.social.chat.telegram import TelegramBotAdapter
from nomorals.social.chat.tgbot_buttons import personal_buttons_for
from nomorals.wisdom import ManifestEntry, WisdomKeeper


def _ctx():
    tmp = tempfile.mkdtemp(prefix="wisdom-read-buttons-")
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(db=None, extras={}, settings=settings)


class FakeRuntime(RuntimeWisdomMixin):
    """Test double carrying the real read mixin method."""

    def __init__(self, ctx):
        self.context = ctx
        self.sent: list[str] = []

    def _ref_from_key(self, key):
        plat, _, cid = key.partition(":")
        return SimpleNamespace(platform=plat or "local", chat_id=cid, key=key)

    def _send_long_checked(self, platform, chat, text):
        self.sent.append(text)
        return ""


# Long multi-sentence text: >3500 chars per chapter so pagination has
# several pages; every sentence is unique so page identity is exact.
CH1 = "# Sayings\n\n" + " ".join(
    f"Saying {i}: the kingdom of heaven is within you and all around you."
    for i in range(1, 121))
CH2 = "# Parables\n\n" + " ".join(
    f"Parable {i}: a sower went out to sow his seed upon the good earth "
    f"and it bore fruit a hundredfold."
    for i in range(1, 121))
LONG_TEXT = CH1 + "\n\n" + CH2


def _callbacks(keyboard):
    return [data for row in (keyboard or []) for _, data in row]


def _labels(keyboard):
    return [label for row in (keyboard or []) for label, _ in row]


class ReadButtonsTests(unittest.TestCase):
    def setUp(self):
        self.ctx = _ctx()
        self.rt = FakeRuntime(self.ctx)
        k = WisdomKeeper(self.ctx)
        k.corpus.register(ManifestEntry.from_dict({
            "slug": "gth", "title": "Gospel of Thomas",
            "tradition": "christian-gnostic", "canon_status": "gnostic",
            "source_url": "https://example.com/gth"}))
        k.corpus.ingest_text("gth", LONG_TEXT)
        self.keeper = k

    def _read(self, tail):
        self.rt.sent.clear()
        self.rt._control_wisdom(f"read {tail}", chat_key="telegram:1")
        self.assertTrue(self.rt.sent, "read produced no sent text")
        return self.rt.sent[0]

    def test_page1_carries_next_only(self):
        text = self._read("gth 1")
        self.assertIn("page 1/", text)
        self.assertIn("next \u2192 /wisdom read gth 2", text)
        self.assertNotIn("\u2190 prev:", text)
        kb = personal_buttons_for(text, "/wisdom read gth 1")
        self.assertIsNotNone(kb)
        assert kb is not None
        self.assertEqual(_labels(kb), ["Next \u27a1\ufe0f"])
        self.assertEqual(_callbacks(kb), ["wisdom read gth 2"])

    def test_next_tap_shows_page2_with_both_buttons(self):
        page1 = self._read("gth 1")
        kb1 = personal_buttons_for(page1, "/wisdom read gth 1")
        assert kb1 is not None
        next_cb = _callbacks(kb1)[0]
        # A tap delivers "/<callback>" as a synthetic command.
        page2 = self._read("gth 2")
        self.assertIn("page 2/", page2)
        kb2 = personal_buttons_for(page2, f"/{next_cb}")
        self.assertIsNotNone(kb2)
        assert kb2 is not None
        self.assertEqual(_labels(kb2), ["\u2b05\ufe0f Previous", "Next \u27a1\ufe0f"])
        self.assertEqual(_callbacks(kb2),
                         ["wisdom read gth 1", "wisdom read gth 3"])

    def test_prev_tap_from_page2_returns_page1(self):
        page2 = self._read("gth 2")
        kb2 = personal_buttons_for(page2, "/wisdom read gth 2")
        assert kb2 is not None
        prev_cb = _callbacks(kb2)[0]
        self.assertEqual(prev_cb, "wisdom read gth 1")
        page1 = self._read("gth 1")
        kb1 = personal_buttons_for(page1, f"/{prev_cb}")
        assert kb1 is not None
        self.assertEqual(_labels(kb1), ["Next \u27a1\ufe0f"])
        self.assertEqual(_callbacks(kb1), ["wisdom read gth 2"])

    def test_last_page_carries_prev_only(self):
        total = self.keeper.corpus.read_work("gth", page=1)["total_pages"]
        self.assertGreater(total, 2)
        text = self._read(f"gth {total}")
        self.assertIn(f"page {total}/", text)
        self.assertNotIn("next \u2192", text)
        kb = personal_buttons_for(text, f"/wisdom read gth {total}")
        self.assertIsNotNone(kb)
        assert kb is not None
        self.assertEqual(_labels(kb), ["\u2b05\ufe0f Previous"])
        self.assertEqual(_callbacks(kb), [f"wisdom read gth {total - 1}"])

    def test_chapter_page_buttons_carry_chapter(self):
        text = self._read("gth 2 1")
        self.assertIn("chapter 2", text)
        kb = personal_buttons_for(text, "/wisdom read gth 2 1")
        self.assertIsNotNone(kb)
        assert kb is not None
        self.assertEqual(_callbacks(kb), ["wisdom read gth 2 2"])

    def test_no_nav_lines_means_no_buttons(self):
        self.assertIsNone(
            personal_buttons_for("plain reply text", "/wisdom read gth 1"))
        # Wrong origin: an ask reply never gets read paging buttons.
        self.assertIsNone(
            personal_buttons_for("page 1/3\nnext \u2192 /wisdom read gth 2",
                                 "/wisdom ask foo"))

    def test_over_budget_button_is_omitted_not_truncated(self):
        slug = "x" * 31
        # Prev (page 9) fits the 47-byte budget; Next (page 10) does not.
        text = ("page 9/99\n"
                f"\u2190 prev: /wisdom read {slug} 1 9\n"
                f"next \u2192 /wisdom read {slug} 1 10")
        kb = personal_buttons_for(text, f"/wisdom read {slug} 1 9")
        self.assertIsNotNone(kb)
        assert kb is not None
        for data in _callbacks(kb):
            self.assertLessEqual(len(data.encode()), 47)
        self.assertEqual(_callbacks(kb), [f"wisdom read {slug} 1 9"])

    def test_all_read_callbacks_within_budget(self):
        for page in (1, 2, 3):
            text = self._read(f"gth {page}")
            kb = personal_buttons_for(text, f"/wisdom read gth {page}")
            assert kb is not None
            for data in _callbacks(kb):
                self.assertLessEqual(len(data.encode()), 47)


class _FakeResp:
    def __init__(self, payload=None):
        self._payload = payload or {"ok": True, "result": True}

    def json(self):
        return self._payload


class _FakeSession:
    """Records Bot API calls; never touches the network."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def post_json(self, url, params, timeout=None):
        self.calls.append((url, dict(params)))
        return _FakeResp()

    def get(self, url, timeout=None):
        self.calls.append((url, {}))
        return _FakeResp()


def _personal_adapter(**kw):
    kw.setdefault("token", "test-token-123")
    kw.setdefault("session", _FakeSession())
    kw.setdefault("owner_only", True)
    kw.setdefault("owner_sender_ids", ("7",))
    return TelegramBotAdapter(**kw)


class ReadCallbackRoutingTests(unittest.TestCase):
    def test_next_callback_routes_as_read_command_on_personal_bot(self):
        adapter = _personal_adapter()
        received: list = []
        signed = adapter._sign_callback("wisdom read gth 2")
        query = {
            "id": "q1",
            "from": {"id": 7, "is_bot": False, "first_name": "Owner"},
            "message": {"chat": {"id": 7, "type": "private"}},
            "data": signed,
        }
        adapter._handle_callback(query, received.append)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].text, "/wisdom read gth 2")
        self.assertTrue(received[0].meta.get("callback_query"))
        self.assertEqual(received[0].meta.get("bot_role"), "personal")

    def test_non_owner_callback_denied_on_personal_bot(self):
        adapter = _personal_adapter()
        received: list = []
        signed = adapter._sign_callback("wisdom read gth 2")
        query = {
            "id": "q1",
            "from": {"id": 999, "is_bot": False, "first_name": "Stranger"},
            "message": {"chat": {"id": 999, "type": "private"}},
            "data": signed,
        }
        adapter._handle_callback(query, received.append)
        self.assertEqual(received, [])


if __name__ == "__main__":
    unittest.main()

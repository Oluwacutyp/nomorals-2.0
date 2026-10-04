"""R8: Telegram inline-button callback path (nomorals.social.chat.telegram).

Covers the R4 HMAC-signed inline buttons — the security-critical part of
the adapter (owner gating + anti-spoofing) — which had zero test coverage.
All offline: a fake HTTP session stands in for the Bot API.
"""

from __future__ import annotations

import unittest

from nomorals.social.chat.telegram import TelegramBotAdapter


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


def _adapter(**kw):
    kw.setdefault("token", "test-token-123")
    kw["session"] = _FakeSession()
    return TelegramBotAdapter(**kw)


def _query(adapter, data, chat_id="42", sender_id=7, is_bot=False):
    return {
        "id": "q1",
        "from": {"id": sender_id, "is_bot": is_bot, "first_name": "Owner"},
        "message": {"chat": {"id": int(chat_id), "type": "private"}},
        "data": data,
    }


class CallbackSignTests(unittest.TestCase):
    def test_sign_verify_roundtrip(self):
        a = _adapter()
        signed = a._sign_callback("/inventory")
        self.assertEqual(a._verify_callback(signed), "/inventory")

    def test_tampered_data_rejected(self):
        a = _adapter()
        signed = a._sign_callback("/inventory")
        data, sig = signed.rsplit("|", 1)
        self.assertIsNone(a._verify_callback(f"/equip sword|{sig}"))
        self.assertIsNone(a._verify_callback(data + "x|" + sig))

    def test_tampered_signature_rejected(self):
        a = _adapter()
        signed = a._sign_callback("/inventory")
        data, _ = signed.rsplit("|", 1)
        self.assertIsNone(a._verify_callback(f"{data}|{'0' * 16}"))

    def test_wrong_token_rejected(self):
        a = _adapter()
        b = _adapter(token="other-token")
        signed = a._sign_callback("/inventory")
        self.assertIsNone(b._verify_callback(signed))

    def test_malformed_rejected(self):
        a = _adapter()
        self.assertIsNone(a._verify_callback("no-separator-here"))
        self.assertIsNone(a._verify_callback(""))


class HandleCallbackTests(unittest.TestCase):
    def test_happy_path_delivers_synthetic_command(self):
        a = _adapter(chat_allow="42")
        received = []
        a._handle_callback(_query(a, a._sign_callback("inventory")), received.append)
        self.assertEqual(len(received), 1)
        msg = received[0]
        self.assertTrue(msg.incoming)
        self.assertEqual(msg.text, "/inventory")
        self.assertTrue(msg.meta.get("callback_query"))
        self.assertEqual(msg.chat.chat_id, "42")
        # the spinner was dismissed via answerCallbackQuery
        methods = [u.rsplit("/", 1)[-1] for u, _ in a._session.calls]
        self.assertIn("answerCallbackQuery", methods)

    def test_slash_data_kept_verbatim(self):
        a = _adapter(chat_allow="42")
        received = []
        a._handle_callback(_query(a, a._sign_callback("/equip sword")), received.append)
        self.assertEqual(received[0].text, "/equip sword")

    def test_non_allowlisted_chat_rejected(self):
        a = _adapter(chat_allow="42")
        received = []
        a._handle_callback(_query(a, a._sign_callback("inventory"), chat_id="99"),
                           received.append)
        self.assertEqual(received, [])
        toasts = [p.get("text", "") for u, p in a._session.calls
                  if u.endswith("answerCallbackQuery")]
        self.assertIn("Not authorized", toasts)

    def test_bad_signature_rejected(self):
        a = _adapter(chat_allow="42")
        received = []
        q = _query(a, "inventory|deadbeefdeadbeef")
        a._handle_callback(q, received.append)
        self.assertEqual(received, [])
        toasts = [p.get("text", "") for u, p in a._session.calls
                  if u.endswith("answerCallbackQuery")]
        self.assertIn("Invalid button", toasts)

    def test_bot_sender_ignored(self):
        a = _adapter(chat_allow="42")
        received = []
        a._handle_callback(
            _query(a, a._sign_callback("inventory"), is_bot=True), received.append)
        self.assertEqual(received, [])

    def test_no_allowlist_allows_all(self):
        a = _adapter()  # chat_allow="" → no restriction
        received = []
        a._handle_callback(_query(a, a._sign_callback("shop"), chat_id="777"),
                           received.append)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].text, "/shop")


if __name__ == "__main__":
    unittest.main()

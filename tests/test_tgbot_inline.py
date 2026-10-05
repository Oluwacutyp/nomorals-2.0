"""Tests for Telegram bot inline keyboards, callback signing, and inline queries."""

import pytest

from nomorals.social.chat.tgbot_buttons import (
    CALLBACK_PAYLOAD_LIMIT,
    buttons_for_text,
    check_callback_data,
    inline_results_for,
)
from nomorals.social.chat.telegram import TelegramBotAdapter
from nomorals.social.chat.base import ChatRef


# ── enricher ──────────────────────────────────────────────────────────

def test_game_menu_buttons():
    text = ("/game <name> — start · /game rematch — run it back\n"
            "modes & mastery unlocks\n"
            "duel moves: attack · focus")
    kb = buttons_for_text(text)
    assert kb is not None
    labels = [b[0] for row in kb for b in row]
    assert "🎮 Join Game" in labels
    assert "📊 Stats" in labels
    assert "🏆 Leaderboard" in labels
    datas = [b[1] for row in kb for b in row]
    assert "game join" in datas
    assert all(len(d.encode()) <= CALLBACK_PAYLOAD_LIMIT for d in datas)


def test_mafia_blurb_join_button():
    kb = buttons_for_text("Mafia is live — /game join to sit at the table (3/7)")
    assert kb == [[("🎮 Join", "game join")]]


def test_live_game_leave_button():
    kb = buttons_for_text("live game: wordchain — send your move (or /game quit)")
    assert kb is not None
    datas = [b[1] for row in kb for b in row]
    assert "game quit" in datas


def test_results_rematch_button():
    kb = buttons_for_text("The House wins this round. 120 points and coins credited.")
    assert kb is not None
    datas = [b[1] for row in kb for b in row]
    assert "game rematch" in datas


def test_plain_chat_no_buttons():
    assert buttons_for_text("hey, how's it going?") is None
    assert buttons_for_text("") is None


# ── callback signing ────────────────────────────────────────────────

def _adapter():
    return TelegramBotAdapter(token="dummy-token-for-tests")


def test_sign_verify_roundtrip():
    a = _adapter()
    signed = a._sign_callback("game join")
    assert a._verify_callback(signed) == "game join"


def test_verify_rejects_tampered():
    a = _adapter()
    signed = a._sign_callback("game join")
    assert a._verify_callback(signed + "x") is None
    assert a._verify_callback("nosig") is None


def test_sign_rejects_oversize_data():
    a = _adapter()
    with pytest.raises(ValueError):
        a._sign_callback("x" * (CALLBACK_PAYLOAD_LIMIT + 1))


def test_check_callback_data_ok():
    check_callback_data("game leaderboard")  # must not raise


# ── inline queries ──────────────────────────────────────────────────

def test_inline_empty_query():
    results = inline_results_for("")
    assert len(results) == 3
    assert all(r["type"] == "article" for r in results)
    texts = [r["input_message_content"]["message_text"] for r in results]
    assert "/game" in texts
    assert "/game stats" in texts


def test_inline_game_match():
    results = inline_results_for("mafia")
    assert len(results) >= 1
    assert results[0]["input_message_content"]["message_text"] == "/game mafia"


def test_inline_stats():
    results = inline_results_for("stats")
    assert results[0]["input_message_content"]["message_text"] == "/game stats"


def test_inline_fallback():
    results = inline_results_for("zzzunknown")
    assert len(results) == 1
    assert results[0]["input_message_content"]["message_text"] == "/game zzzunknown"


def test_handle_inline_query_answers():
    a = _adapter()
    calls = []

    def fake_api(method, **params):
        calls.append((method, params))
        return {"ok": True}

    a._api = fake_api
    a._handle_inline_query({
        "id": "iq123",
        "from": {"id": 42, "username": "tester", "is_bot": False},
        "query": "mafia",
    })
    assert len(calls) == 1
    method, params = calls[0]
    assert method == "answerInlineQuery"
    assert params["inline_query_id"] == "iq123"
    assert len(params["results"]) >= 1
    assert params["results"][0]["input_message_content"]["message_text"] == "/game mafia"


def test_handle_inline_query_ignores_bots():
    a = _adapter()
    calls = []
    a._api = lambda method, **p: calls.append(method) or {}
    a._handle_inline_query({
        "id": "iq1",
        "from": {"id": 1, "is_bot": True},
        "query": "",
    })
    assert calls == []


# ── send() button wiring ────────────────────────────────────────────

def test_send_attaches_enriched_buttons_last_chunk_only():
    a = _adapter()
    sent = []

    def fake_api(method, **params):
        sent.append(params)
        return {"message_id": 7}

    a._api = fake_api
    chat = ChatRef(platform="telegram-bot", chat_id="123", kind="dm")
    # Long enough to split into 2 chunks, game-menu text triggers buttons.
    body = ("x" * 8000) + "\n/game <name> — start\nmodes & mastery unlocks"
    result = a.send(chat, body)
    assert result.ok
    assert len(sent) == 2
    # First chunk: no keyboard. Last chunk: keyboard present.
    assert "reply_markup" not in sent[0]
    assert "reply_markup" in sent[1]
    kb = sent[1]["reply_markup"]["inline_keyboard"]
    flat = [b for row in kb for b in row]
    assert any(b["text"] == "🎮 Join Game" for b in flat)
    # Callback data is signed.
    assert "|" in flat[0]["callback_data"]


def test_send_explicit_buttons_win_over_enricher():
    a = _adapter()
    sent = []
    a._api = lambda method, **p: sent.append(p) or {"message_id": 1}
    chat = ChatRef(platform="telegram-bot", chat_id="123", kind="dm")
    a.send(chat, "plain text", buttons=[[("Hi", "hello")]])
    kb = sent[0]["reply_markup"]["inline_keyboard"]
    assert kb[0][0]["text"] == "Hi"
    assert a._verify_callback(kb[0][0]["callback_data"]) == "hello"


def test_send_plain_text_no_markup():
    a = _adapter()
    sent = []
    a._api = lambda method, **p: sent.append(p) or {"message_id": 1}
    chat = ChatRef(platform="telegram-bot", chat_id="123", kind="dm")
    a.send(chat, "just chatting")
    assert "reply_markup" not in sent[0]

"""Tests for tappable inline action buttons (Telegram Bot API).

Covers:
- actionable() helper: structure, slash normalization, budget enforcement
- buttons_for_text(): the 5 high-value auto-derived patterns
- WhatsApp fallback: adapter accepts buttons= without rendering/breaking
- never-raises on garbage input
"""

import pytest

from nomorals.social.chat.tgbot_buttons import (
    ActionableMessage,
    actionable,
    buttons_for_text,
    check_callback_data,
    CALLBACK_PAYLOAD_LIMIT,
)


# ── actionable() helper ───────────────────────────────────────────────

def test_actionable_builds_single_row_keyboard():
    msg = actionable("pool empty", [("🔄 Refresh pool", "/proxy refresh")])
    assert isinstance(msg, ActionableMessage)
    assert msg.text == "pool empty"
    assert msg.buttons == [[("🔄 Refresh pool", "proxy refresh")]]


def test_actionable_strips_leading_slash():
    msg = actionable("t", [("A", "/proxy refresh"), ("B", "proxy pool")])
    assert msg.buttons == [[("A", "proxy refresh"), ("B", "proxy pool")]]


def test_actionable_multiple_buttons_one_row():
    msg = actionable("t", [("✅ Approve", "research approve latest"),
                           ("❌ Deny", "research deny latest")])
    assert len(msg.buttons) == 1
    assert len(msg.buttons[0]) == 2


def test_actionable_empty_buttons():
    msg = actionable("plain text", [])
    assert msg.text == "plain text"
    assert msg.buttons == []


def test_actionable_rejects_overlong_callback():
    with pytest.raises(ValueError):
        actionable("t", [("X", "x" * (CALLBACK_PAYLOAD_LIMIT + 1))])


def test_check_callback_data_boundary():
    check_callback_data("x" * CALLBACK_PAYLOAD_LIMIT)  # exactly at limit: ok
    with pytest.raises(ValueError):
        check_callback_data("x" * (CALLBACK_PAYLOAD_LIMIT + 1))


# ── buttons_for_text: the 5 high-value patterns ───────────────────────

def test_proxy_pool_empty():
    kb = buttons_for_text("pool empty — /proxy refresh to scrape+test")
    assert kb == [[("🔄 Refresh pool", "proxy refresh")]]


def test_research_approve_deny():
    text = ("top proposals (2):\n"
            "· abc123  0.85  [pending] tech — something\n"
            "approve: /research approve <id|latest>   deny: /research deny <id|latest>")
    kb = buttons_for_text(text)
    assert kb == [[("✅ Approve latest", "research approve latest"),
                   ("❌ Deny latest", "research deny latest")]]


def test_build_failure_coding_bot():
    kb = buttons_for_text("❌ coding bot gave up after 3 iteration(s), 42s\nsome error\n"
                          "it's in the workspace if you want to take over.")
    assert kb == [[("📋 Checkpoints", "checkpoints")]]


def test_build_failure_builder():
    kb = buttons_for_text("❌ the builder gave up after 0 iteration(s): model returned no code block")
    assert kb == [[("📋 Checkpoints", "checkpoints")]]


def test_music_track_transport():
    kb = buttons_for_text("🎵 \u201cMidnight Drive\u201d  [uk-drill, A minor, 143 bpm]\nchorus: ...")
    assert kb == [[("⏸️ Pause", "play pause"), ("⏭️ Next", "play next")]]


def test_music_error_gets_no_buttons():
    kb = buttons_for_text("🎵 couldn't make the full song:\nsome error")
    assert kb is None


def test_tour_results():
    kb = buttons_for_text("🧪 tour — 213 commands probed\n"
                          "✅ 142 OK · ❌ 4 broken · ⏭️ 67 skipped\n"
                          "deep-probe one: /tour <command>")
    assert kb == [[("🔁 Re-run tour", "tour")]]


# ── no false positives ────────────────────────────────────────────────

def test_unrelated_text_gets_no_buttons():
    assert buttons_for_text("hello, how are you?") is None
    assert buttons_for_text("") is None


def test_partial_patterns_dont_match():
    # research text without the approve/deny footer → no buttons
    assert buttons_for_text("top proposals (0):\nno proposals yet — /research run") is None
    # tour mention without the probed marker → no buttons
    assert buttons_for_text("have you tried /tour?") is None


# ── never raises ──────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [None, "", "x" * 100_000, "🎵\u201c", "pool empty —"])
def test_buttons_for_text_never_raises(bad):
    assert buttons_for_text(bad) in (None, []) or isinstance(buttons_for_text(bad), list)


# ── callback budget: every new button fits ────────────────────────────

def test_all_new_callback_payloads_within_budget():
    payloads = [
        "proxy refresh",
        "research approve latest",
        "research deny latest",
        "checkpoints",
        "play pause",
        "play next",
        "tour",
    ]
    for p in payloads:
        assert len(p.encode("utf-8")) <= CALLBACK_PAYLOAD_LIMIT, p


# ── WhatsApp fallback: buttons accepted, ignored, never break ─────────

def test_whatsapp_send_accepts_buttons_kwarg():
    import inspect
    from nomorals.social.chat.whatsapp import WhatsAppAdapter
    sig = inspect.signature(WhatsAppAdapter.send)
    assert "buttons" in sig.parameters


def test_whatsapp_does_not_render_inline_keyboard():
    import inspect
    src = inspect.getsource(__import__(
        "nomorals.social.chat.whatsapp", fromlist=["WhatsAppAdapter"]).WhatsAppAdapter.send)
    assert "inline_keyboard" not in src


# ── Telegram adapter: buttons flow to sendMessage ─────────────────────

def test_telegram_bot_send_accepts_buttons_kwarg():
    import inspect
    from nomorals.social.chat.telegram import TelegramBotAdapter
    sig = inspect.signature(TelegramBotAdapter.send)
    assert "buttons" in sig.parameters

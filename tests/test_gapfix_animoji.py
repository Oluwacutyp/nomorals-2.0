"""Animated emoji in texts (owner, 2026-11-11, chat audit N1):
"Telegram emojis still whack you know telegram has a way where
emojis are animated" / "Still using still emoji's" / "Not sent as
single emojis bro. I'm talking about real emoji animation even in
texts".

What this lane proves, end to end:

A. the ACTION SCHEMA honors ``style`` + ``icon_custom_emoji_id``
   on the Telegram Bot API surfaces (Bot API 9.4, 2026-02-09:
   ``style`` is one of primary/success/danger; ``icon_custom_emoji_id``
   is the custom emoji shown before the button text, gated on the
   bot owner having Telegram Premium or the bot having bought
   Fragment usernames) — validated at construction, resolved as
   honored, and dropped by name everywhere else, including when
   the id is not one of the set Devon's registry actually holds.
B. the BOT ADAPTER emits them: ``_button_json`` carries both
   fields on callback AND url buttons, ``send``/``edit_message``
   accept Action items in the keyboard, and a refusal naming
   custom emoji retries the message with a PLAIN keyboard instead
   of eating it.
C. in-text weaving reaches the texts that actually go out: the
   gateway converts most replies to ``parse_mode="HTML"``, and the
   old weave hook only ran on plain chunks — so registered glyphs
   in an HTML reply never animated. The adapter now converts the
   HTML chunk to plain text + entities, merges the custom-emoji
   entities in, and sends the entities form; HTML with no
   registered glyph is byte-identical to before, and a weave
   refusal falls back to the HTML parse form (text still lands,
   formatting intact).
D. the card path carries the rich fields: ``render_card`` keeps
   its legacy tuple keyboard AND exposes the Actions as
   ``rich_keyboard`` when any action declares style/icon, so the
   CardHost -> gateway -> adapter chain can emit them.
"""

from __future__ import annotations

import json

import pytest

from nomorals.social.chat.action_schema import (
    BUTTON_STYLES,
    Action,
    resolve_action,
)
from nomorals.social.chat.base import ChatKind, ChatRef
from nomorals.social.chat.expression import CustomEmojiRegistry
from nomorals.social.chat.living_cards import Card, render_card
from nomorals.social.chat.telegram import (
    TelegramBotAdapter,
    telegram_html_to_entities,
)

# ── rigs (same shape as test_emoji_weaving.py) ───────────────────


def _bot():
    adapter = TelegramBotAdapter(
        token="test-token", owner_sender_ids={"7"})
    adapter._bot_username = "Devon_test_bot"
    return adapter


def _chat(kind=ChatKind.DM, chat_id="42"):
    return ChatRef(platform="telegram-bot", chat_id=chat_id,
                   kind=kind)


def _send_rig(adapter, *, refuse_icon=False, refuse_entities=None):
    """Fake the Bot API; returns the sendMessage call list.

    ``refuse_icon``: any call whose keyboard carries an
    ``icon_custom_emoji_id`` dies with Telegram's own refusal.
    ``refuse_entities``: any call carrying ``entities`` dies
    with that error text. The plain / unrich send always lands.
    """
    calls = []

    def fake_api(method, **params):
        assert method == "sendMessage"
        calls.append(params)
        if refuse_icon and "icon_custom_emoji_id" in json.dumps(
                params.get("reply_markup") or {}):
            raise RuntimeError(
                "Bad Request: custom emoji is not available")
        if refuse_entities is not None and "entities" in params:
            raise RuntimeError(refuse_entities)
        return {"message_id": 100 + len(calls)}

    adapter._api = fake_api
    return calls


def _wired(adapter):
    registry = CustomEmojiRegistry()
    registry.add("bag", unicode="💰", platform="telegram",
                 ref="M1", format="animated")
    adapter.custom_emoji_weave_source = registry.glyph_map
    return adapter


def _rich_action(**kw):
    base = dict(label="Go", intent="callback", data="help",
                style="success", icon_custom_emoji_id="12345")
    base.update(kw)
    return Action(**base)


# ══ A · the action schema honors style + icon ───────────────────


def test_styles_are_the_bot_api_closed_set():
    assert BUTTON_STYLES == ("primary", "success", "danger")


def test_telegram_bot_honors_style_and_icon():
    res = resolve_action(_rich_action(), "telegram-bot")
    assert res.form == "inline"
    assert "style" in res.honored
    assert "icon_custom_emoji_id" in res.honored
    assert res.dropped == ()


def test_telegram_personal_honors_style_and_icon():
    res = resolve_action(_rich_action(), "telegram-personal")
    assert res.form == "inline"
    assert "style" in res.honored
    assert "icon_custom_emoji_id" in res.honored
    assert res.dropped == ()


def test_other_surfaces_still_drop_both_by_name():
    for platform in ("whatsapp", "discord", "sms", "web", "local",
                     "telegram"):
        res = resolve_action(_rich_action(), platform)
        assert "style" in res.dropped, platform
        assert "icon_custom_emoji_id" in res.dropped, platform


def test_action_rejects_a_style_telegram_does_not_have():
    with pytest.raises(ValueError):
        _rich_action(style="glow")


def test_action_rejects_a_non_numeric_icon_id():
    with pytest.raises(ValueError):
        _rich_action(icon_custom_emoji_id="abc")
    with pytest.raises(ValueError):
        _rich_action(icon_custom_emoji_id="12 34")


def test_resolve_drops_an_icon_the_set_does_not_hold():
    known = {"111", "222"}
    res = resolve_action(_rich_action(icon_custom_emoji_id="999"),
                         "telegram-bot", known_custom_emoji_ids=known)
    assert "icon_custom_emoji_id" in res.dropped
    assert "style" in res.honored  # only the icon is unverified
    assert "unverified" in res.note or "999" in res.note
    res = resolve_action(_rich_action(icon_custom_emoji_id="111"),
                         "telegram-bot", known_custom_emoji_ids=known)
    assert "icon_custom_emoji_id" in res.honored
    assert res.dropped == ()


# ══ B · the adapter emits them, and never eats the message ──────


def test_button_json_emits_style_and_icon_on_callback_buttons():
    adapter = _bot()
    btn = adapter._button_json(
        "Go", "help", style="primary",
        icon_custom_emoji_id="5368324170671202286")
    assert btn["text"] == "Go"
    assert btn["callback_data"].startswith("help|")
    assert btn["style"] == "primary"
    assert btn["icon_custom_emoji_id"] == "5368324170671202286"


def test_button_json_emits_style_and_icon_on_url_buttons():
    adapter = _bot()
    btn = adapter._button_json(
        "Rules", "https://example.com/rules", style="danger",
        icon_custom_emoji_id="12345")
    assert btn == {"text": "Rules", "url": "https://example.com/rules",
                   "style": "danger", "icon_custom_emoji_id": "12345"}


def test_button_json_drops_invalid_fields_without_failing():
    adapter = _bot()
    btn = adapter._button_json("Go", "help", style="glow",
                               icon_custom_emoji_id="abc")
    assert "style" not in btn
    assert "icon_custom_emoji_id" not in btn
    assert btn["text"] == "Go"  # the button still goes out


def test_send_action_buttons_land_in_the_payload():
    adapter = _bot()
    calls = _send_rig(adapter)
    out = adapter.send(_chat(), "pick one",
                       buttons=[[_rich_action()]])
    assert out.ok and len(calls) == 1
    btn = calls[0]["reply_markup"]["inline_keyboard"][0][0]
    assert btn["text"] == "Go"
    assert btn["style"] == "success"
    assert btn["icon_custom_emoji_id"] == "12345"


def test_send_rich_button_refusal_retries_plain_and_lands():
    adapter = _bot()
    calls = _send_rig(adapter, refuse_icon=True)
    out = adapter.send(_chat(), "pick one",
                       buttons=[[_rich_action()]])
    assert out.ok  # the message still went out
    assert len(calls) == 2
    first = calls[0]["reply_markup"]["inline_keyboard"][0][0]
    assert first["icon_custom_emoji_id"] == "12345"
    second = calls[1]["reply_markup"]["inline_keyboard"][0][0]
    assert "icon_custom_emoji_id" not in second
    assert "style" not in second
    assert calls[1]["text"] == "pick one"


# ══ C · weaving reaches HTML replies ─────────────────────────────


def test_html_to_entities_covers_the_gateway_tags():
    plain, entities = telegram_html_to_entities(
        "<b>paid 💰</b> and <i>go</i>")
    assert plain == "paid 💰 and go"
    bold = next(e for e in entities if e["type"] == "bold")
    assert (bold["offset"], bold["length"]) == (0, 7)
    italic = next(e for e in entities if e["type"] == "italic")
    assert (italic["offset"], italic["length"]) == (12, 2)


def test_html_to_entities_nests_and_unescapes():
    plain, entities = telegram_html_to_entities(
        "<b>go <code>💰 &amp; co</code></b>")
    assert plain == "go 💰 & co"
    code = next(e for e in entities if e["type"] == "code")
    assert (code["offset"], code["length"]) == (3, 7)
    bold = next(e for e in entities if e["type"] == "bold")
    assert (bold["offset"], bold["length"]) == (0, 10)


def test_html_to_entities_tolerates_an_unclosed_tag():
    plain, entities = telegram_html_to_entities("<b>paid 💰")
    assert plain == "paid 💰"
    assert entities == [{"type": "bold", "offset": 0, "length": 7}]


def test_send_html_weaves_registered_glyphs_as_entities():
    adapter = _wired(_bot())
    calls = _send_rig(adapter)
    out = adapter.send(_chat(), "<b>paid 💰</b>", parse_mode="HTML")
    assert out.ok and len(calls) == 1
    # the entities form carries the formatting — parse_mode is
    # gone (the Bot API takes entities INSTEAD of parse_mode)
    assert "parse_mode" not in calls[0]
    assert calls[0]["text"] == "paid 💰"
    entities = json.loads(calls[0]["entities"])
    bold = next(e for e in entities if e["type"] == "bold")
    assert (bold["offset"], bold["length"]) == (0, 7)
    custom = next(e for e in entities if e["type"] == "custom_emoji")
    assert (custom["offset"], custom["length"]) == (5, 2)
    assert custom["custom_emoji_id"] == "M1"


def test_send_markdown_keeps_its_parse_path_even_with_a_glyph():
    # The reputation card still sends Markdown: weaving there
    # would print its markup literally, so the parse path wins
    # over the glyph.
    adapter = _wired(_bot())
    calls = _send_rig(adapter)
    out = adapter.send(_chat(), "*paid 💰*", parse_mode="Markdown")
    assert out.ok and len(calls) == 1
    assert calls[0]["parse_mode"] == "Markdown"
    assert calls[0]["text"] == "*paid 💰*"
    assert "entities" not in calls[0]


def test_send_html_without_registered_glyphs_keeps_parse_path():
    adapter = _wired(_bot())
    calls = _send_rig(adapter)
    out = adapter.send(_chat(), "<b>hello</b>", parse_mode="HTML")
    assert out.ok and len(calls) == 1
    assert calls[0]["parse_mode"] == "HTML"
    assert calls[0]["text"] == "<b>hello</b>"
    assert "entities" not in calls[0]


def test_send_html_weave_refusal_keeps_html_and_lands():
    adapter = _wired(_bot())
    calls = _send_rig(
        adapter,
        refuse_entities="telegram bot api: sendMessage failed: "
                        "Bad Request: custom emoji is not available")
    out = adapter.send(_chat(), "<b>paid 💰</b>", parse_mode="HTML")
    assert out.ok  # the message still went out
    assert len(calls) == 2
    assert "entities" in calls[0] and "entities" not in calls[1]
    # the fallback keeps the HTML parse form — formatting intact
    assert calls[1]["parse_mode"] == "HTML"
    assert calls[1]["text"] == "<b>paid 💰</b>"
    assert adapter._weave_refused.keys() == {"dm"}


# ══ D · the card path carries the rich fields ────────────────────


def _card(actions):
    return Card(card_id="c0ffee42", family="generic",
                chat_key="telegram-bot:42", owner_key="owner1",
                platform="telegram-bot", chat_id="42",
                title="T", body="pick one", actions=tuple(actions))


def test_render_card_carries_rich_keyboard_on_telegram():
    out = render_card(_card([_rich_action()]), "telegram-bot")
    assert out.form == "inline"
    # the legacy tuple keyboard is untouched …
    assert out.keyboard == [[("Go", "cardtap c0ffee42 1 1")]]
    # … and the rich rows carry the Action itself through
    assert len(out.rich_keyboard) == 1
    action = out.rich_keyboard[0][0]
    assert action.style == "success"
    assert action.icon_custom_emoji_id == "12345"


def test_render_card_rich_keyboard_stays_empty_without_rich_fields():
    plain = Action(label="Go", intent="callback", data="help")
    out = render_card(_card([plain]), "telegram-bot")
    assert out.rich_keyboard == []
    assert out.keyboard == [[("Go", "cardtap c0ffee42 1 1")]]


def test_render_card_rich_keyboard_stays_empty_off_telegram():
    out = render_card(_card([_rich_action()]), "whatsapp")
    assert out.form == "interactive"
    assert out.rich_keyboard == []


def test_send_dict_form_buttons_land_in_the_payload():
    # The gateway normalizes Actions to dicts before the adapter
    # ever sees them — the adapter must render the dict form
    # identically (this is also the form the durable outbox
    # replays).
    adapter = _bot()
    calls = _send_rig(adapter)
    out = adapter.send(
        _chat(), "pick one",
        buttons=[[{"label": "Go", "data": "help", "style": "primary",
                   "icon_custom_emoji_id": "12345"}]])
    assert out.ok and len(calls) == 1
    btn = calls[0]["reply_markup"]["inline_keyboard"][0][0]
    assert btn["style"] == "primary"
    assert btn["icon_custom_emoji_id"] == "12345"


# ══ E · the gateway normalizes rich buttons for every path ──────


def test_gateway_normalizes_action_buttons_to_json_safe_dicts():
    from nomorals.social.chat.gateway import _normalize_buttons

    rows = _normalize_buttons(
        [[_rich_action(), ("Help", "help")]])
    assert rows == [[
        {"label": "Go", "data": "help", "style": "success",
         "icon_custom_emoji_id": "12345"},
        ("Help", "help")]]
    # The durable outbox round-trips buttons through JSON — the
    # normalized form must survive it exactly (tuples come back
    # as lists, as they always did through that path).
    roundtripped = json.loads(json.dumps(rows))
    assert roundtripped[0][0] == {
        "label": "Go", "data": "help", "style": "success",
        "icon_custom_emoji_id": "12345"}
    assert roundtripped[0][1] == ["Help", "help"]


def test_gateway_normalization_leaves_plain_buttons_untouched():
    from nomorals.social.chat.gateway import _normalize_buttons

    rows = [[("Help", "help")]]
    assert _normalize_buttons(rows) == rows
    assert _normalize_buttons(None) is None


def test_card_host_sends_the_rich_rows():
    # End to end of the card path: publish() hands the transport
    # the Action rows (not the legacy tuples) when any action is
    # rich — the gateway then normalizes them for the adapter.
    from nomorals.social.chat.living_cards import CardHost, CardStore

    sent: list = []

    def _send(platform, chat, text, buttons=None, **kw):
        sent.append(buttons)
        return (True, "m1")

    host = CardHost(CardStore(), send=_send,
                    edit=lambda *a, **k: False)
    host.publish(_card([_rich_action()]), object())
    assert len(sent) == 1
    rows = sent[0]
    assert len(rows) == 1
    assert isinstance(rows[0][0], Action)
    assert rows[0][0].style == "success"
    assert rows[0][0].icon_custom_emoji_id == "12345"


def test_card_host_sends_tuples_when_nothing_is_rich():
    from nomorals.social.chat.living_cards import CardHost, CardStore

    sent: list = []

    def _send(platform, chat, text, buttons=None, **kw):
        sent.append(buttons)
        return (True, "m1")

    plain = Action(label="Go", intent="callback", data="help")
    host = CardHost(CardStore(), send=_send,
                    edit=lambda *a, **k: False)
    host.publish(_card([plain]), object())
    assert sent[0] == [[("Go", "cardtap c0ffee42 1 1")]]

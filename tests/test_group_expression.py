"""DESIGN §21 + §22 at the runtime seam — expression moments
under the §33 gate, the §31 quiet switch, per-group settings
grammar + console, and the expression commands.

The §33 PIN (the brief's own words): no welcome GIF fires in
an unclaimed room, while the welcome TEXT path behaves exactly
as the gate build left it (suppressed abroad, posted in a
claimed room).  Expression is bot-system output; the as-him
hat is untouched by everything here.
"""

from __future__ import annotations

from types import SimpleNamespace

from nomorals.agents.partner.runtime_expression import (
    RuntimeExpressionMixin,
)
from nomorals.agents.partner.runtime_games import RuntimeGamesMixin
from nomorals.agents.partner.runtime_group import RuntimeGroupMixin
from nomorals.social.chat.base import ChatKind

BOT_KEY = "telegram-bot:-100123"
BOT_KEY2 = "telegram-bot:-100777"
USER_KEY = "telegram:-100999"


class _Chat:
    def __init__(self, platform, key, title="Builders"):
        self.platform = platform
        self.key = key
        self.chat_id = key.split(":", 1)[1]
        self.title = title
        self.kind = ChatKind.GROUP


class _Msg:
    def __init__(self, chat, **kw):
        self.chat = chat
        self.meta = kw.get("meta") or {}
        self.sender_id = kw.get("sender_id", "7")
        self.sender = kw.get("sender", "Ada")
        self.sender_username = kw.get("sender_username", "ada")
        self.text = kw.get("text", "")
        self.message_id = kw.get("message_id", "55")
        self.reply_to = kw.get("reply_to", "")
        self.media = kw.get("media") or []


class _Adapter:
    """An adapter with the full expression surface: admin
    probes (for the claim), media sends, reactions, and the
    replied-media download the stealer rides."""

    def __init__(self, tmp_path, admin_ids=(),
                 bot_status="administrator"):
        self.calls = []
        self.media_sent = []
        self.reactions = []
        self.admin_ids = set(admin_ids)
        self._bot_id = 4242
        self.bot_status = bot_status
        self.tmp_path = tmp_path

    def admin_member(self, chat, user_id):
        if str(user_id) == str(self._bot_id):
            status = self.bot_status
        elif str(user_id) in self.admin_ids:
            status = "administrator"
        else:
            status = "member"
        return {"ok": True, "result": {"status": status}}

    def admin_delete(self, chat, message_id):
        self.calls.append(("delete", message_id))
        return {"ok": True}

    def admin_ban(self, chat, user_id):
        self.calls.append(("ban", user_id))
        return {"ok": True}

    def send_media(self, chat, ref):
        self.media_sent.append(ref)
        return SimpleNamespace(ok=True, message_id="9",
                               media_file_id="TGFILE1", error="")

    def react(self, chat, message_id, emoji):
        self.reactions.append((message_id, emoji))
        return SimpleNamespace(ok=True, message_id=message_id,
                               error="")

    def download_file(self, file_id, fname="media"):
        dest = self.tmp_path / f"dl_{file_id}_{fname}"
        dest.write_bytes(b"GIF89a stolen")
        return str(dest)


class _Gateway:
    def __init__(self, adapters):
        self.adapters = adapters
        self.sent = []

    def send(self, platform, chat, text, **kw):
        self.sent.append((platform, text))
        return SimpleNamespace(ok=True, message_id="1")


class _Runtime(RuntimeExpressionMixin, RuntimeGroupMixin,
               RuntimeGamesMixin):
    def __init__(self, tmp_path, adapters, operators=()):
        self.sent = []
        self.gateway = _Gateway(adapters)
        self.operators = set(operators)
        self.context = SimpleNamespace(settings=SimpleNamespace(
            workspace_dir=str(tmp_path)))

    def _send(self, chat, text, **kw):
        self.sent.append(text)
        return SimpleNamespace(ok=True, message_id="1")

    def _is_operator(self, message):
        return str(message.sender_id or "") in self.operators

    def turn(self, msg):
        return self._group_surface_turn(msg)


def _rig(tmp_path, platform="telegram-bot", key=BOT_KEY, **kw):
    adapter = _Adapter(tmp_path, **kw)
    rt = _Runtime(tmp_path, {platform: adapter}, operators={"99"})
    return rt, adapter, _Chat(platform, key)


def _join(chat, user_id="7", name="Ada", username="ada"):
    return _Msg(chat, meta={"service": "join", "new_members": [
        {"id": user_id, "username": username, "name": name}]},
        sender_id=user_id, sender=name, sender_username=username)


def _stock(rt, tmp_path, tags=("welcome",)):
    path = tmp_path / "party.gif"
    path.write_bytes(b"GIF89a")
    return rt._expression_library().add_item(
        kind="gif", path=str(path), tags=list(tags),
        source="test")


# ── the §33 pin ───────────────────────────────────────────────


def test_no_welcome_gif_in_unclaimed_room(tmp_path):
    rt, adapter, chat = _rig(tmp_path, platform="telegram",
                             key=USER_KEY)
    _stock(rt, tmp_path)
    assert rt.turn(_join(chat)) is True
    # the welcome TEXT path is exactly as the gate build left
    # it: nothing posts abroad — and no GIF rides it either
    assert rt.sent == []
    assert adapter.media_sent == []
    kinds = [a["kind"] for a in
             rt._group_surface().audit(USER_KEY)]
    assert "system_output_suppressed" in kinds


def test_welcome_gif_fires_in_claimed_bot_room(tmp_path):
    rt, adapter, chat = _rig(tmp_path)
    item = _stock(rt, tmp_path)
    assert rt.turn(_join(chat)) is True
    assert rt.sent and "Ada" in rt.sent[0]  # the text, as before
    assert len(adapter.media_sent) == 1
    ref = adapter.media_sent[0]
    assert ref.kind == "animation"  # a GIF rides sendAnimation
    # the first send's file_id was captured into the library
    assert item["file_ids"]["telegram-bot"] == "TGFILE1"
    kinds = [a["kind"] for a in
             rt._group_surface().audit(BOT_KEY)]
    assert "expression" in kinds


def test_second_room_resends_by_file_id(tmp_path):
    rt, adapter, chat = _rig(tmp_path)
    _stock(rt, tmp_path)
    rt.turn(_join(chat))
    chat2 = _Chat("telegram-bot", BOT_KEY2)
    rt.turn(_join(chat2, user_id="8", name="Bola",
                  username="bola"))
    assert len(adapter.media_sent) == 2
    assert adapter.media_sent[1].file_id == "TGFILE1"


def test_expression_off_keeps_text_suppresses_media(tmp_path):
    rt, adapter, chat = _rig(tmp_path)
    _stock(rt, tmp_path)
    surface = rt._group_surface()
    surface.update_settings(BOT_KEY, expression={
        "enabled": False, "gifs": "normal",
        "welcome_media": True, "ban_theatre": True})
    assert rt.turn(_join(chat)) is True
    assert rt.sent  # welcome text still posts
    assert adapter.media_sent == []
    entries = [a for a in surface.audit(BOT_KEY)
               if a["kind"] == "expression"]
    assert entries and entries[-1]["sent"] is False
    assert "off" in entries[-1]["reason"]


def test_quiet_autonomy_silences_moments(tmp_path):
    rt, adapter, chat = _rig(tmp_path)
    _stock(rt, tmp_path)
    surface = rt._group_surface()
    rt._defense_update(surface, BOT_KEY, "_root", autonomy=False)
    assert rt.turn(_join(chat)) is True
    assert adapter.media_sent == []
    entries = [a for a in surface.audit(BOT_KEY)
               if a["kind"] == "expression"]
    assert entries and "quiet" in entries[-1]["reason"]


def test_reaction_events_are_journaled_not_answered(tmp_path):
    rt, adapter, chat = _rig(tmp_path, platform="telegram",
                             key=USER_KEY)
    msg = _Msg(chat, meta={"service": "reaction",
                           "reaction_new": ["🔥"],
                           "reaction_old": [],
                           "reacted_message_id": "31"},
               message_id="31")
    assert rt.turn(msg) is True
    assert rt.sent == [] and adapter.media_sent == []
    entries = [a for a in rt._group_surface().audit(USER_KEY)
               if a["kind"] == "reaction"]
    assert entries and entries[-1]["emojis"] == "🔥"
    # a read is not accrual: no economy woke up over a reaction
    assert rt._group_surface().economy.balance(USER_KEY, "7") == 0


# ── buttons: one vocabulary on every surface ──────────────────


def _all_labels(buttons):
    return [label for row in buttons or [] for label, _cb in row]


def test_console_and_welcome_and_game_buttons_share_the_map(
        tmp_path):
    from nomorals.social.chat.expression import BUTTON_EMOJI

    rt, adapter, chat = _rig(tmp_path, admin_ids={"7"})
    surface = rt._group_surface()
    # a console card (defense section)
    out = rt._console_section(surface, BOT_KEY,
                              _Msg(chat, sender_id="99"), "tok",
                              "defense")
    labels = _all_labels(getattr(out, "buttons", None) or [])
    assert any(label.startswith(BUTTON_EMOJI["autonomy"])
               for label in labels)
    assert any(label.startswith(BUTTON_EMOJI["anticrash"])
               for label in labels)
    # the welcome keyboard
    labels = _all_labels(rt._welcome_buttons(surface, BOT_KEY))
    assert labels[0] == "📜 Rules"
    # a game board (slots buttons authored in the handler)
    from nomorals.social.chat.expression import buttons_from_spec

    rows = buttons_from_spec(
        [[("spin", "Spin again (10)", "slots 10"),
          ("balance", "My balance", "balance")]],
        settings=surface.store.settings(BOT_KEY))
    assert rows[0][0][0] == "🎰 Spin again (10)"
    assert rows[0][1][0] == "🪙 My balance"


# ── settings grammar + console ────────────────────────────────


def test_gsettings_expression_grammar(tmp_path):
    rt, adapter, chat = _rig(tmp_path, admin_ids={"7"})
    surface = rt._group_surface()
    msg = _Msg(chat, sender_id="7")
    out = rt._control_gsettings("expression gifs low", msg)
    assert "gifs low" in str(out)
    assert surface.store.settings(BOT_KEY)["expression"]["gifs"] \
        == "low"
    out = rt._control_gsettings("expression off", msg)
    assert surface.store.settings(BOT_KEY)["expression"]["enabled"] \
        is False
    out = rt._control_gsettings("expression bogus", msg)
    assert "usage:" in str(out)
    # the card reads the same state back, in its own block
    card = rt._control_gsettings("", _Msg(chat, sender_id="99"))
    text = getattr(card, "text", card)
    assert "Expression" in text and "gifs low" in text


def test_console_expression_section_and_apply(tmp_path):
    rt, adapter, chat = _rig(tmp_path, admin_ids={"7"})
    surface = rt._group_surface()
    msg = _Msg(chat, sender_id="99")
    out = rt._console_section(surface, BOT_KEY, msg, "tok",
                              "expression")
    text = getattr(out, "text", out)
    assert "Expression" in text and "gifs normal" in text
    labels = _all_labels(getattr(out, "buttons", None) or [])
    assert any(label.startswith("🎭") for label in labels)
    line = rt._console_apply(surface, BOT_KEY, msg, "Owner",
                             "expression", ["ban", "off"])
    assert "ban theatre off" in str(line)
    assert surface.store.settings(BOT_KEY)["expression"][
        "ban_theatre"] is False
    line = rt._console_apply(surface, BOT_KEY, msg, "Owner",
                             "expression", ["gifs", "low"])
    assert surface.store.settings(BOT_KEY)["expression"]["gifs"] \
        == "low"


# ── the commands ──────────────────────────────────────────────


def test_steal_claims_into_library_and_resends(tmp_path):
    rt, adapter, chat = _rig(tmp_path, admin_ids={"7"})
    msg = _Msg(chat, reply_to="42", meta={"replied_media": {
        "kind": "sticker", "file_id": "FID9",
        "file_unique_id": "U9", "name": "sticker.webp",
        "set_name": "", "emoji": ""}})
    out = rt._control_steal("funny dance", msg)
    assert out == ""  # the re-sent sticker IS the reply
    library = rt._expression_library()
    assert len(library) == 1
    item = library.items[0]
    assert item["source"] == "steal"
    assert "funny" in item["tags"] and "sticker" in item["tags"]
    assert item["file_ids"]["telegram-bot"] in ("FID9", "TGFILE1")
    assert adapter.media_sent  # re-sent into the room


def test_steal_in_unclaimed_room_collects_nothing(tmp_path):
    rt, adapter, chat = _rig(tmp_path, platform="telegram",
                             key=USER_KEY)
    msg = _Msg(chat, reply_to="42", meta={"replied_media": {
        "kind": "sticker", "file_id": "FID9",
        "file_unique_id": "U9", "name": "sticker.webp",
        "set_name": "", "emoji": ""}})
    out = rt._control_steal("", msg)
    assert "isn't claimed" in str(out)
    assert len(rt._expression_library()) == 0
    assert adapter.media_sent == []


def test_emoji_command_searches_the_whole_table(tmp_path):
    rt, adapter, chat = _rig(tmp_path)
    out = rt._control_emoji("melting face", _Msg(chat))
    assert "🫠" in str(out)
    out = rt._control_emoji("👨‍👩‍👧", _Msg(chat))
    assert "family" in str(out)


def test_react_command_places_and_respects_the_bot_set(tmp_path):
    rt, adapter, chat = _rig(tmp_path)
    msg = _Msg(chat, reply_to="42")
    out = rt._control_react("🔥", msg)
    assert out == "" and adapter.reactions == [("42", "🔥")]
    # 🫠 is not in the Bot API's standard set — honest refusal,
    # no adapter call
    out = rt._control_react("🫠", msg)
    assert "standard set" in str(out)
    assert adapter.reactions == [("42", "🔥")]


def test_gif_command_uses_the_library(tmp_path):
    rt, adapter, chat = _rig(tmp_path)
    _stock(rt, tmp_path, tags=("celebration", "dance"))
    out = rt._control_gif("dance", _Msg(chat))
    assert out == "" and len(adapter.media_sent) == 1
    out = rt._control_gif("underwater basketweaving", _Msg(chat))
    assert "nothing in my library" in str(out)


def test_customemoji_command_roundtrip(tmp_path):
    rt, adapter, chat = _rig(tmp_path, admin_ids={"7"})
    owner = _Msg(chat, sender_id="99")
    out = rt._control_customemoji("add devon 🥷", owner)
    assert "added" in str(out)
    out = rt._control_customemoji("list", _Msg(chat))
    assert "devon" in str(out) and "🥷" in str(out)
    # a stranger cannot curate her set
    out = rt._control_customemoji("add sneaky 😈",
                                  _Msg(chat, sender_id="1234"))
    assert "curated" in str(out)
    registry = rt._expression_library().custom_emoji
    assert registry.get("sneaky") is None
    assert registry.get("devon")["unicode"] == "🥷"

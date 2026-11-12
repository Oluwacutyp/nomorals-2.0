"""The /play pick-list ack-edit carries real buttons on WhatsApp.

The specific-query fall-through offer used to land as a keyboard-CLEARING
edit (``buttons=[]``) — the buttons were never sent, so the owner saw
plain text with "tap a number below" and no keyboard. The ack-edit now
derives the keyboard from the final text when the final is a pick-list
offer (the ListMessage rows on WhatsApp), and settled renders still
clear. The number reply stays as the fallback rung.
"""

from types import SimpleNamespace
from unittest import mock


class _FakeAdapter:
    """Records edits; derives buttons exactly like the real WhatsApp
    adapter (personal set first, then the adaptive content patterns)."""

    def __init__(self):
        self.edits = []  # (message_id, text, buttons)

    def _derive_buttons(self, chat, text):
        from nomorals.social.chat.adaptive_buttons import (
            adaptive_buttons_for,
        )
        from nomorals.social.chat.tgbot_buttons import personal_buttons_for

        return (personal_buttons_for(text, "")
                or adaptive_buttons_for(origin_command="", text=text))

    def edit_message(self, chat, message_id, text, *, buttons=None):
        self.edits.append((str(message_id), text, buttons))
        return {"ok": True}

    def edit_progress(self, chat, message_id, text):
        self.edits.append((str(message_id), text, None))
        return {"ok": True}


class _NoDeriveAdapter(_FakeAdapter):
    """An adapter with no derivation seam (Telethon shape): the edit
    must still carry the keyboard via the shared fallback."""

    def __init__(self):
        super().__init__()
        # Shadow the class method with a non-callable: getattr finds
        # None, so the edit derives via the shared fallback instead.
        self._derive_buttons = None


class _PlainAdapter:
    """An adapter whose edit verb takes no buttons kwarg at all."""

    def __init__(self):
        self.edits = []

    def edit_message(self, chat, message_id, text):
        self.edits.append((str(message_id), text))
        return {"ok": True}

    def edit_progress(self, chat, message_id, text):
        self.edits.append((str(message_id), text))
        return {"ok": True}


class _FakeGateway:
    """Records sends; plays the adapter for edit verbs."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.sent = []
        self._n = 0

    def send(self, platform, chat, text, **kw):
        self._n += 1
        self.sent.append((platform, chat, text))
        return SimpleNamespace(ok=True, platform=platform,
                               message_id=f"m{self._n}")

    def _adapter_for(self, platform):
        return self.adapter


def _handler(adapter):
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    rt = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
    rt.context = SimpleNamespace()
    rt.gateway = _FakeGateway(adapter)
    return rt


def _chat(chat_id="12345"):
    chat = mock.Mock()
    chat.platform = "whatsapp"
    chat.chat_id = chat_id
    chat.key = f"whatsapp:{chat_id}"
    return chat


def _picklist_text(token="d5a5bbeb"):
    from nomorals.media.picklist import format_picklist

    cands = [SimpleNamespace(title=f"Track {i}", artist="Corizo",
                             duration=200.0, kind="soundcloud",
                             source_name="soundcloud", downloadable=True)
             for i in range(1, 9)]
    return format_picklist(cands, "haza by corizo", token)


def _button_total(buttons):
    return sum(len(row) for row in buttons if row)


def _collect(chat_key, title="haza by corizo", timeout=10.0):
    from nomorals.core.jobs import shared_registry

    snaps = shared_registry().list(
        labels={"owner": "play", "chat": chat_key, "title": title})
    assert snaps, f"no play job for {chat_key}/{title}"
    job = shared_registry().collect(snaps[-1]["id"], timeout=timeout)
    assert job is not None and job.done(), \
        f"job never finished: {job.status if job else None}"
    return job


# ── 1: the ack-edit derives the pick-list keyboard ────────────────────


def test_ack_edit_derives_picklist_keyboard():
    rt = _handler(_FakeAdapter())
    chat = _chat("btn-1")
    text = _picklist_text()
    assert rt._edit_chat_message(chat, "m1", text, buttons=None) is True
    assert len(rt.gateway.adapter.edits) == 1
    _mid, _text, buttons = rt.gateway.adapter.edits[0]
    assert buttons is not None
    assert _button_total(buttons) == 8
    datas = [data for row in buttons for _label, data in row]
    assert datas == [f"play pick d5a5bbeb {n}" for n in range(1, 9)]


def test_ack_edit_derives_without_adapter_derive_seam():
    rt = _handler(_NoDeriveAdapter())
    chat = _chat("btn-2")
    text = _picklist_text()
    assert rt._edit_chat_message(chat, "m1", text, buttons=None) is True
    _mid, _text, buttons = rt.gateway.adapter.edits[0]
    assert buttons is not None
    assert _button_total(buttons) == 8


def test_ack_edit_derives_nothing_for_plain_text():
    rt = _handler(_FakeAdapter())
    chat = _chat("btn-3")
    assert rt._edit_chat_message(chat, "m1", "▶ playing “Song”",
                                 buttons=None) is True
    _mid, _text, buttons = rt.gateway.adapter.edits[0]
    assert buttons is None


# ── 2: settled renders still clear ────────────────────────────────────


def test_default_edit_still_clears():
    rt = _handler(_FakeAdapter())
    chat = _chat("btn-4")
    assert rt._edit_chat_message(chat, "m1", "▶ playing “Song”") is True
    _mid, _text, buttons = rt.gateway.adapter.edits[0]
    assert buttons == []


def test_explicit_empty_still_clears():
    rt = _handler(_FakeAdapter())
    chat = _chat("btn-5")
    assert rt._edit_chat_message(chat, "m1", _picklist_text(),
                                 buttons=[]) is True
    _mid, _text, buttons = rt.gateway.adapter.edits[0]
    assert buttons == []


def test_explicit_buttons_pass_through():
    rt = _handler(_FakeAdapter())
    chat = _chat("btn-6")
    kb = [[("1", "play pick d5a5bbeb 1")]]
    assert rt._edit_chat_message(chat, "m1", "pick one", buttons=kb) is True
    _mid, _text, buttons = rt.gateway.adapter.edits[0]
    assert buttons == kb


def test_edit_without_buttons_kwarg_still_lands():
    rt = _handler(_PlainAdapter())
    chat = _chat("btn-7")
    assert rt._edit_chat_message(chat, "m1", "▶ playing “Song”") is True
    assert rt.gateway.adapter.edits == [("m1", "▶ playing “Song”")]
    assert rt._edit_chat_message(chat, "m1", _picklist_text(),
                                 buttons=None) is True
    assert len(rt.gateway.adapter.edits) == 2


# ── 3: the copy names the button, and the derivation still matches ───


def test_picklist_footer_claims_buttons_and_still_derives():
    from nomorals.agents.partner.runtime import _claims_buttons
    from nomorals.social.chat.tgbot_buttons import buttons_for_text

    text = _picklist_text()
    assert "tap a button below 👇" in text
    assert _claims_buttons(text) is True
    kb = buttons_for_text(text)
    assert kb is not None
    assert _button_total(kb) == 8


# ── 4: the download-body final edit carries the keyboard ──────────────


def test_body_edits_picklist_final_with_buttons():
    rt = _handler(_FakeAdapter())
    chat = _chat("btn-8")
    offer = _picklist_text()
    out = rt._play_download_bg(
        chat, chat.key, "⏳ haza by corizo",
        lambda: (offer, False),
        title="haza by corizo")
    assert out == ""
    job = _collect(chat.key)
    assert job.result["delivered"] is True
    assert len(rt.gateway.adapter.edits) == 1
    _mid, _text, buttons = rt.gateway.adapter.edits[0]
    assert _text == offer
    assert buttons is not None
    assert _button_total(buttons) == 8


def test_body_edits_plain_final_without_buttons():
    rt = _handler(_FakeAdapter())
    chat = _chat("btn-9")
    final = "couldn't download “haza by corizo” — every match was blocked."
    out = rt._play_download_bg(
        chat, chat.key, "⏳ haza by corizo",
        lambda: (final, False),
        title="haza by corizo")
    assert out == ""
    job = _collect(chat.key)
    assert job.result["delivered"] is True
    assert len(rt.gateway.adapter.edits) == 1
    _mid, _text, buttons = rt.gateway.adapter.edits[0]
    assert _text == final
    assert buttons == []

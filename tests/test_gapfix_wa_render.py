"""WhatsApp table/diff → PNG render wiring (audit gap R1-2).

The renderer module (nomorals/social/chat/whatsapp_render.py) existed
with zero production callers: every table or diff Devon sent on
WhatsApp arrived as a mangled wall of reflowed monospace. These pins
cover the one call site in WhatsAppAdapter.send (plus the outbox
flush door, which duplicates the text path inline):

* a table / diff goes out as a PNG via send_media, with the original
  text as caption — never as plain text;
* plain text is byte-identical (same ``send`` cmd, same text);
* a reply or a mention cannot ride the media door, so those stay text;
* a render failure, or a definitive media refusal, falls back to
  plain text — the message is never dropped;
* an unconfirmed media dispatch is never double-sent as text;
* a wide keyboard rides the media follow-up exactly once (no loop);
* a table queued during an outage renders at flush time;
* the render path is stable per (chat, text), so the media send-once
  guard keys the retry instead of double-sending the image.
"""
from __future__ import annotations

from pathlib import Path

from nomorals.social.chat.base import ChatKind, ChatRef
from nomorals.social.chat.whatsapp import UNCONFIRMED_SEND_ERROR, WhatsAppAdapter

CHAT = ChatRef(platform="whatsapp", chat_id="234801@s.whatsapp.net",
               kind=ChatKind.DM)

TABLE = "| name | coins |\n|---|---|\n| ada | 50 |\n| bo | 30 |"
DIFF = "--- a/x\n+++ b/x\n-old line\n+new line\n context\n"


def _adapter(tmp_path, responder):
    """A connected WhatsAppAdapter whose bridge is a script."""
    adapter = WhatsAppAdapter(
        media_dir=str(tmp_path / "media"),
        outbox_dir=str(tmp_path / "outbox"))
    adapter.connected.set()
    calls: list = []

    def fake_send_cmd(cmd, timeout=25.0):
        calls.append(dict(cmd))
        return responder(cmd)

    adapter._send_cmd = fake_send_cmd  # type: ignore[method-assign]
    return adapter, calls


def _ok(cmd):
    return {"ok": True, "message_id": "M1"}


def _media_cmds(calls):
    return [c for c in calls if c.get("cmd") == "send_media"]


def _text_cmds(calls):
    return [c for c in calls if c.get("cmd") == "send"]


def test_table_send_goes_out_as_png_with_text_caption(tmp_path):
    adapter, calls = _adapter(tmp_path, _ok)
    out = adapter.send(CHAT, TABLE, buttons=[])
    assert out.ok
    assert not _text_cmds(calls), calls
    media = _media_cmds(calls)
    assert len(media) == 1, calls
    assert media[0]["path"].endswith(".png"), media[0]
    assert media[0]["caption"] == TABLE, media[0]
    assert media[0]["media_kind"] == "image", media[0]
    data = Path(media[0]["path"]).read_bytes()
    assert data.startswith(b"\x89PNG")
    assert len(data) > 100


def test_diff_send_goes_out_as_png(tmp_path):
    adapter, calls = _adapter(tmp_path, _ok)
    out = adapter.send(CHAT, DIFF, buttons=[])
    assert out.ok
    assert not _text_cmds(calls), calls
    media = _media_cmds(calls)
    assert len(media) == 1, calls
    assert Path(media[0]["path"]).read_bytes().startswith(b"\x89PNG")
    assert media[0]["caption"] == DIFF, media[0]


def test_plain_text_still_goes_out_as_text(tmp_path):
    adapter, calls = _adapter(tmp_path, _ok)
    out = adapter.send(CHAT, "hello there", buttons=[])
    assert out.ok
    assert not _media_cmds(calls), calls
    text = _text_cmds(calls)
    assert len(text) == 1, calls
    assert text[0]["text"] == "hello there", text[0]


def test_render_failure_falls_back_to_text(tmp_path, monkeypatch):
    import nomorals.social.chat.whatsapp_render as render_mod

    def _boom(*args, **kwargs):
        raise RuntimeError("no PIL today")

    monkeypatch.setattr(render_mod, "render_for_whatsapp", _boom)
    adapter, calls = _adapter(tmp_path, _ok)
    out = adapter.send(CHAT, TABLE, buttons=[])
    assert out.ok
    assert not _media_cmds(calls), calls
    text = _text_cmds(calls)
    assert len(text) == 1, calls
    assert text[0]["text"] == TABLE, text[0]


def test_table_with_reply_to_stays_text(tmp_path):
    adapter, calls = _adapter(tmp_path, _ok)
    out = adapter.send(CHAT, TABLE, reply_to="M9", buttons=[])
    assert out.ok
    assert not _media_cmds(calls), calls
    text = _text_cmds(calls)
    assert len(text) == 1, calls
    assert text[0]["reply_to"] == "M9", text[0]


def test_table_with_mentions_stays_text(tmp_path):
    adapter, calls = _adapter(tmp_path, _ok)
    out = adapter.send(CHAT, TABLE, mentions=["234802@c.us"], buttons=[])
    assert out.ok
    assert not _media_cmds(calls), calls
    text = _text_cmds(calls)
    assert len(text) == 1, calls
    assert text[0]["mentions"] == ["234802@c.us"], text[0]


def test_definitive_media_refusal_falls_back_to_text(tmp_path):
    def responder(cmd):
        if cmd.get("cmd") == "send_media":
            return {"ok": False, "error": "bridge rejected media"}
        return {"ok": True, "message_id": "M1"}

    adapter, calls = _adapter(tmp_path, responder)
    out = adapter.send(CHAT, TABLE, buttons=[])
    assert out.ok
    assert len(_media_cmds(calls)) == 1, calls
    text = _text_cmds(calls)
    assert len(text) == 1, calls
    assert text[0]["text"] == TABLE, text[0]


def test_unconfirmed_media_dispatch_is_never_double_sent(tmp_path):
    def responder(cmd):
        if cmd.get("cmd") == "send_media":
            return {"ok": False, "ambiguous": True}
        return {"ok": True, "message_id": "M1"}

    adapter, calls = _adapter(tmp_path, responder)
    out = adapter.send(CHAT, TABLE, buttons=[])
    assert not out.ok
    assert out.error == UNCONFIRMED_SEND_ERROR, out
    assert not _text_cmds(calls), calls
    assert len(_media_cmds(calls)) == 1, calls


def test_table_with_small_keyboard_rides_the_media(tmp_path):
    adapter, calls = _adapter(tmp_path, _ok)
    out = adapter.send(CHAT, TABLE, buttons=[[("A", "a")]])
    assert out.ok
    assert not _text_cmds(calls), calls
    media = _media_cmds(calls)
    assert len(media) == 1, calls
    assert media[0]["buttons"] == [[["A", "a"]]], media[0]


def test_table_with_wide_keyboard_follows_up_exactly_once(tmp_path):
    # Four rows exceeds the native quick_reply cap, so the keyboard
    # rides a follow-up text message — which re-enters send() with
    # the table as its body. Without the re-entrancy guard that
    # follow-up would render again and loop.
    wide = [[(f"B{i}", f"b{i}")] for i in range(4)]
    adapter, calls = _adapter(tmp_path, _ok)
    out = adapter.send(CHAT, TABLE, buttons=wide)
    assert out.ok
    assert len(_media_cmds(calls)) == 1, calls
    text = _text_cmds(calls)
    assert len(text) == 1, calls
    assert text[0]["text"] == TABLE, text[0]
    assert len(calls) == 2, calls


def test_table_queued_during_outage_renders_at_flush(tmp_path):
    adapter = WhatsAppAdapter(
        media_dir=str(tmp_path / "media"),
        outbox_dir=str(tmp_path / "outbox"))
    out = adapter.send(CHAT, TABLE, buttons=[])
    assert not out.ok and "queued" in (out.error or ""), out
    entries = adapter.outbox.pending()
    assert entries and entries[0]["kind"] == "text", entries
    calls: list = []
    adapter.connected.set()
    adapter._send_cmd = (  # type: ignore[method-assign]
        lambda cmd, timeout=25.0: (calls.append(dict(cmd)),
                                   {"ok": True,
                                    "message_id": "M9"})[1])
    assert adapter._deliver_entry(dict(entries[0])) is True
    assert len(_media_cmds(calls)) == 1, calls
    assert not _text_cmds(calls), calls


def test_same_table_twice_uses_the_same_png_path(tmp_path):
    adapter, calls = _adapter(tmp_path, _ok)
    assert adapter.send(CHAT, TABLE, buttons=[]).ok
    assert adapter.send(CHAT, TABLE, buttons=[]).ok
    media = _media_cmds(calls)
    assert len(media) == 2, calls
    assert media[0]["path"] == media[1]["path"], media

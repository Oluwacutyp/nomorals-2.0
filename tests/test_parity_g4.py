"""PARITY G4 — WhatsApp ENGINE-PARITY SWEEP (platform parity
census bundle #7, 2026-11-07).

Three PARTIALs where one bridge engine carried a verb and
the other answered an honest refusal:

  * message pin — real on Go (whatsmeow PinInChatMessage),
    refused on Baileys with "no message-pin verb".  The
    refusal was stale: Baileys 6.7.24's sendMessage builds
    the very same pinInChatMessage from a {pin: key, type}
    content (lib/Utils/messages.js; type 1 = PIN_FOR_ALL,
    2 = UNPIN_FOR_ALL in WAProto), relayed with the pin
    edit attribute (lib/Socket/messages-send.js).  The
    Baileys handlers now resolve the key from bridge
    memory exactly like message_delete and fire it.
  * channel rename/description — real on Baileys
    (newsletterUpdateName / newsletterUpdateDescription),
    refused on Go with "no verb in this whatsmeow pin".
    whatsmeow has no PUBLIC update method, but its
    newsletter.go carries the update mutation's query ID
    and its transport (sendMexIQ) is exposed as
    DangerousInternals().SendMexIQ.  The Go bridge now
    sends the mutation with Baileys' exact variables
    envelope ({newsletter_id, updates: {field,
    settings: null}}).
  * disappearing DMs — real on Go (SetDisappearingTimer,
    both scopes), groups-only on Baileys.  Baileys' group
    path is the groupToggleEphemeral IQ; a DM's timer is
    an EPHEMERAL_SETTING protocol message, which
    sendMessage builds for {disappearingMessagesInChat}
    on any non-group chat.  The Baileys verb now branches:
    groups keep the IQ, DMs take the message path.

The adapter verbs never changed — both engines always
received the same payloads; the refusals lived in the
bridges.  Bridge internals are pinned where they live:
the Go verb's Mex call shape has its own Go tests
(bridge/gobridge/newsletter_test.go — query ID, envelope,
settings key, error propagation), the Node verbs are
source-pinned against Baileys' exported shapes, the
adapter payloads are pinned through a recording fake,
and the rebuilt Go binaries carry the mutation's ID.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from nomorals.social.chat.base import ChatKind, ChatRef

ROOT = Path(__file__).resolve().parents[1]
GROUP = "120363001234567890@g.us"
DM = "2348012345678@s.whatsapp.net"
CHANNEL = "120207123456789012@newsletter"


# ── bridge source pins ─────────────────────────────────────

def test_go_bridge_newsletter_update_verbs():
    groups = (ROOT / "bridge" / "gobridge"
              / "groups.go").read_text()
    assert 'case "channel_set_name"' in groups
    assert 'case "channel_set_description"' in groups
    assert "setNewsletterName" in groups
    assert "setNewsletterDescription" in groups
    # the stale refusals are gone — the verbs are real now
    assert "no newsletter rename verb" not in groups
    assert "no newsletter description verb" not in groups
    src = (ROOT / "bridge" / "gobridge"
           / "newsletter.go").read_text()
    # the transport: whatsmeow's exposed internal Mex sender
    assert "DangerousInternalClient" in src
    assert "SendMexIQ" in src
    # the mutation: whatsmeow's own query ID for it
    assert '"7150902998257522"' in src
    # the envelope Baileys' working implementation sends
    assert '"newsletter_id"' in src
    assert '"settings"' in src


def test_baileys_pin_verbs():
    src = (ROOT / "bridge"
           / "whatsapp-groups-admin.mjs").read_text()
    # the pin content Baileys builds pinInChatMessage from:
    # the bridge-memory key + the WAProto type (1 pin, 2 unpin)
    assert "{ pin: cached.key, type: 1 }" in src
    assert "{ pin: cached.key, type: 2 }" in src
    # the key discipline is message_delete's: never guessed
    assert "message not in bridge memory" in src
    # the stale unsupported is gone
    assert "has no message-pin verb" not in src
    assert "Go bridge engine only" not in src


def test_baileys_disappear_covers_dms():
    src = (ROOT / "bridge" / "whatsapp-bridge.mjs").read_text()
    start = src.index("case 'chat_disappear'")
    region = src[start:src.index("case '", start + 1)]
    # groups keep the ephemeral IQ…
    assert "groupToggleEphemeral" in region
    # …and DMs take the ephemeral-setting message path
    assert "disappearingMessagesInChat" in region
    # the stale groups-only refusal is gone
    assert "DM timers ride the Go bridge engine" not in src
    assert "can only set disappearing timers in groups" not in src


def test_rebuilt_binaries_carry_the_newsletter_mutation():
    # The update mutation's query ID is unique to this
    # build's Go code and survives in the binaries' strings.
    for name in ("whatsapp-bridge-linux", "whatsapp-bridge.exe"):
        data = (ROOT / "bridge" / "bin" / name).read_bytes()
        assert b"7150902998257522" in data, name


# ── adapter payload pins (recording fake bridge) ──────────

@pytest.fixture()
def adapter(monkeypatch):
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    ad = WhatsAppAdapter(port=19999)
    calls: list[tuple] = []

    def fake_admin(cmd, payload, *, timeout=30.0):
        calls.append((cmd, payload))
        return {"ok": True}

    monkeypatch.setattr(ad, "_admin_cmd", fake_admin)
    ad.calls = calls  # type: ignore[attr-defined]
    return ad


def _chat(chat_id, kind):
    return ChatRef(platform="whatsapp", chat_id=chat_id,
                   kind=kind, peer=chat_id)


def test_adapter_channel_set_name_payload(adapter):
    res = adapter.channel_set_name(CHANNEL, "  Gold Room  ")
    assert res["ok"] is True
    assert adapter.calls == [(
        "channel_set_name",
        {"chat": CHANNEL, "name": "Gold Room"})]


def test_adapter_channel_set_description_payload(adapter):
    res = adapter.channel_set_description(CHANNEL, "Daily XAUUSD")
    assert res["ok"] is True
    assert adapter.calls == [(
        "channel_set_description",
        {"chat": CHANNEL, "description": "Daily XAUUSD"})]


def test_adapter_disappear_dm_payload(adapter):
    # a DM chat rides the same chat_disappear verb a group
    # does — the engines branch internally, the adapter
    # payload never carried the scope split
    res = adapter.set_disappearing(_chat(DM, ChatKind.DM), 86400)
    assert res["ok"] is True
    assert adapter.calls == [(
        "chat_disappear", {"chat": DM, "seconds": 86400})]


def test_adapter_pin_payloads(adapter):
    adapter.admin_pin(_chat(GROUP, ChatKind.GROUP), "MSG42")
    adapter.admin_unpin(_chat(GROUP, ChatKind.GROUP), "MSG42")
    assert adapter.calls == [
        ("message_pin", {"chat": GROUP, "message_id": "MSG42"}),
        ("message_unpin", {"chat": GROUP, "message_id": "MSG42"})]

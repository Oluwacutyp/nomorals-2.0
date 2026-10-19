"""PASS #7: partner runtime — verified bugs, failing-first.

Bug 1: ``PartnerRuntime._send_long`` split long texts at hard character
boundaries (``text[i:i+limit]``) — mid-sentence, mid-word — even though
the boundary-aware lossless chunker (``social.chat.chunking``) exists
and the gateway chunks that way on its own. Long reports/digests went
out with words sliced in half.
"""
from __future__ import annotations

from nomorals.agents.partner.runtime import PartnerRuntime
from nomorals.social.chat.base import ChatRef, SendResult


class _FakeGateway:
    dry_run = False

    def __init__(self):
        self.sent: list[str] = []

    def typing(self, platform, chat, seconds=0.0, **kw):
        return True

    def send(self, platform, chat, text, **kw):
        self.sent.append(text)
        return SendResult(ok=True, platform=platform, message_id="m1")


def _runtime():
    rt = PartnerRuntime.__new__(PartnerRuntime)
    rt.gateway = _FakeGateway()
    # typing simulation off: the test is about chunk BOUNDARIES, not the
    # indicator. (_typing_enabled's real logic is covered elsewhere.)
    rt._typing_enabled = lambda chat: False
    from types import SimpleNamespace
    rt.settings = SimpleNamespace(
        partner=SimpleNamespace(typing_in_groups=True))
    return rt


def _sentences(n: int) -> str:
    # ~60 chars each; boundaries every sentence.
    return " ".join(
        f"This is sentence number {i:04d} and it ends right here."
        for i in range(n))


def test_send_long_never_splits_mid_word():
    rt = _runtime()
    chat = ChatRef(platform="local", chat_id="1")
    text = _sentences(100)  # ~6k chars, sentence boundary every ~60
    sent = rt._send_long("local", chat, text, limit=3800)
    assert sent >= 2
    joined = "".join(rt.gateway.sent)
    assert joined == text, "chunking must be lossless"
    for piece in rt.gateway.sent[:-1]:
        # A non-final chunk must end at a word/sentence/paragraph
        # boundary — never in the middle of a word.
        assert piece[-1] in " \t\n.!?…", (
            f"chunk split mid-word: ...{piece[-40:]!r}")


def test_send_long_lossless_single_chunk():
    rt = _runtime()
    chat = ChatRef(platform="local", chat_id="1")
    text = "short enough to go whole."
    assert rt._send_long("local", chat, text, limit=3800) == 1
    assert rt.gateway.sent == [text]


# ── Bug 2: _send_voice_reply drops remaining parts when TTS throws ──

class _FakeVoiceAdapter:
    def __init__(self, gw):
        self._gw = gw

    def send_media(self, chat, media):
        self._gw.voice_sent.append(getattr(media, "path", ""))
        return SendResult(ok=True, platform="local", message_id="v1")


class _VoiceGateway(_FakeGateway):
    def __init__(self):
        super().__init__()
        self.voice_sent: list[str] = []

    def _adapter_for(self, platform):
        return _FakeVoiceAdapter(self)


def _voice_runtime(monkeypatch):
    import nomorals.voice.pingpong as _pp

    rt = PartnerRuntime.__new__(PartnerRuntime)
    gw = _VoiceGateway()
    rt.gateway = gw
    rt._typing_enabled = lambda chat: False
    rt.context = object()
    calls = {"n": 0}

    def _fake_synth(context, text):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("tts engine exploded")
        return f"/tmp/fake-{calls['n']}.ogg"

    monkeypatch.setattr(_pp, "synthesize_voice_reply", _fake_synth)
    return rt, gw


def _voice_message():
    from nomorals.social.chat.base import ChatMessage
    return ChatMessage(chat=ChatRef(platform="local", chat_id="1"),
                       incoming=True, text="", meta={"voice_in": True})


def test_voice_reply_mid_failure_falls_back_to_text(monkeypatch):
    # Part 1 sends as voice; TTS throws on part 2. Parts 2 and 3 must
    # still reach the user — as text — not vanish silently.
    rt, gw = _voice_runtime(monkeypatch)
    msg = _voice_message()
    ok = rt._send_voice_reply(msg, ["first part", "second part",
                                    "third part"])
    assert ok is True
    assert len(gw.voice_sent) == 1, "only part 1 went out as voice"
    assert gw.sent == ["second part", "third part"], (
        f"parts 2+ must fall back to text, got {gw.sent!r}")

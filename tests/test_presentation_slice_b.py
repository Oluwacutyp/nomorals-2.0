"""Presentation build, slice B (adoption sweep): the games
lobby and the WhatsApp contract.

Pins:
- /game list returns an ActionableMessage whose buttons are
  authored from the chat's live state: a live room adds its own
  rematch/quit row; the boards/lobby rows are always there
  because the list documents them.
- _relay_send carries buttons across the relay (the rematch
  invite's accept button rides it).
- WhatsApp: the adapter hands buttons to the bridge on the live
  path AND stores them on the offline outbox entry, so a queued
  send keeps its keyboard; the bridge renders/degrades them
  (bridge side is covered by its node syntax + protocol shape —
  the Python payload is the contract pinned here).
"""
from __future__ import annotations

from types import SimpleNamespace

from nomorals.social.chat.base import ChatKind, ChatRef


def _flat(msg) -> dict:
    return {data: label for row in (msg.buttons or [])
            for label, data in row}


class _FakeEngine:
    def __init__(self, live=None):
        self._live = live

    def live(self, chat_key):  # noqa: ARG002
        return self._live

    def list_games(self):
        return "  ttt — tic tac toe\n  chess — chess"


def _games_rt(live=None):
    from nomorals.agents.partner.runtime_games import RuntimeGamesMixin

    class _RT(RuntimeGamesMixin):
        def __init__(self):
            self.context = SimpleNamespace()
            self._eng = _FakeEngine(live)

        def _game_engine(self):  # type: ignore[no-untyped-def]
            return self._eng

    return _RT()


def test_game_list_buttons_match_the_display_no_live_room():
    rt = _games_rt()
    out = rt._control_game("list", "local:c1")
    datas = _flat(out)
    assert "game leaderboard" in datas
    assert "game stats" in datas
    assert "pvp" in datas
    # no room is live in this chat — no rematch/quit buttons,
    # because there is nothing on screen they would act on
    assert "game rematch" not in datas
    assert "game quit" not in datas
    assert "tic tac toe" in out.text  # the catalog itself


def test_game_list_buttons_gain_room_actions_when_live():
    rt = _games_rt(live=SimpleNamespace(game="ttt"))
    out = rt._control_game("", "local:c1")
    datas = _flat(out)
    assert "game rematch" in datas
    assert "game quit" in datas
    assert "live game: ttt" in out.text


def test_relay_send_carries_buttons():
    from nomorals.agents.partner.runtime_games import RuntimeGamesMixin

    rt = RuntimeGamesMixin()
    sent = {}

    class _GW:
        def send(self, platform, chat_key, text, **kw):
            sent.update(platform=platform, chat=chat_key,
                        text=text, **kw)
            return SimpleNamespace(ok=True)

    rt.gateway = _GW()
    ok = rt._relay_send("telegram-bot:-5", "duel over",
                        buttons=[[("🔁 Rematch — accept",
                                   "game accept XYZ")]])
    assert ok is True
    assert sent["buttons"] == [[("🔁 Rematch — accept",
                                 "game accept XYZ")]]


# ── WhatsApp contract ─────────────────────────────────────────

def _wa_adapter(monkeypatch):
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.connected = __import__("threading").Event()
    adapter.connected.set()
    adapter.stats = {"sent": 0, "send_errors": 0}
    adapter.name = "whatsapp"
    captured = {}
    monkeypatch.setattr(
        adapter, "_send_cmd",
        lambda req: captured.update(req) or {"ok": True,
                                              "message_id": "m1"})
    return adapter, captured


def test_whatsapp_send_hands_buttons_to_the_bridge(monkeypatch):
    adapter, captured = _wa_adapter(monkeypatch)
    chat = ChatRef(platform="whatsapp", chat_id="2348012345678",
                   kind=ChatKind.DM)
    result = adapter.send(chat, "drop is live",
                          buttons=[[("🎁 Claim 25 sparks", "claim")]])
    assert result.ok
    assert captured["buttons"] == [[["🎁 Claim 25 sparks", "claim"]]]


def test_whatsapp_send_without_buttons_sends_no_button_field(
        monkeypatch):
    adapter, captured = _wa_adapter(monkeypatch)
    chat = ChatRef(platform="whatsapp", chat_id="2348012345678",
                   kind=ChatKind.DM)
    assert adapter.send(chat, "plain").ok
    assert "buttons" not in captured


def test_whatsapp_offline_entry_keeps_its_buttons(monkeypatch):
    adapter, _captured = _wa_adapter(monkeypatch)
    adapter.connected.clear()
    entries = {}
    monkeypatch.setattr(
        adapter, "_enqueue",
        lambda entry: entries.update(entry) or
        SimpleNamespace(ok=False))
    chat = ChatRef(platform="whatsapp", chat_id="2348012345678",
                   kind=ChatKind.DM)
    adapter.send(chat, "queued drop",
                 buttons=[[("🎁 Claim", "claim")]])
    assert entries["buttons"] == [[["🎁 Claim", "claim"]]]

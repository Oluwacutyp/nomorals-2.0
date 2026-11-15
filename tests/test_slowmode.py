"""Slow mode (DESIGN §4/§16): /slowmode sets a Telegram group's
slow-mode delay through the MTProto userbot — the Bot API cannot
set it (slow_mode_delay is read-only there), and WhatsApp groups
don't have slow mode at all.

Pins: registration (CONTROL_COMMANDS, PUBLIC_COMMANDS,
WHATSAPP_UNSUPPORTED_KINDS, COMMAND_DETAILS), the snap table
(Telegram accepts only 0/5/10/30/60/300/900/3600 seconds),
admin gating, store-only-after-enforcement, the audit rows,
bare-status with live drift, the console defense section +
grammar, /gsettings grammar, and the /groups dashboard line.
Adapter-level: ToggleSlowModeRequest + GetFullChannelRequest
through a fake Telethon client (Telethon is not installed on
this box, so the tl.functions module is stubbed in sys.modules
for those two tests only).
"""
from __future__ import annotations

import asyncio
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.social.chat.control import (
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    PUBLIC_COMMANDS,
    WHATSAPP_UNSUPPORTED_KINDS,
)
from nomorals.social.chat.group_surface import GroupStore, GroupSurface

TG_KEY = "telegram:-1001"
WA_KEY = "whatsapp:1203630123456@g.us"


class _Adapter:
    """Fake group adapter: admin_member + the two slow-mode verbs."""

    def __init__(self, *, admin=True, refuse=False, detected=0,
                 verbs=True, fail_with=None):
        self.calls: list = []
        self._admin = admin
        self._refuse = refuse
        self.detected = detected
        self.fail_with = fail_with
        if not verbs:
            self.admin_set_slow_mode = None  # type: ignore[method-assign]
            self.chat_slow_mode_delay = None  # type: ignore[method-assign]

    def admin_member(self, chat, user_id):
        return {"status": "administrator" if self._admin
                else "member"}

    def admin_set_slow_mode(self, chat, seconds):
        self.calls.append(("slowmode", seconds))
        if self.fail_with is not None:
            return {"ok": False, "error": self.fail_with}
        return {"ok": not self._refuse,
                "error": "nope" if self._refuse else ""}

    def chat_slow_mode_delay(self, chat):
        self.calls.append(("detect",))
        return {"ok": True, "seconds": self.detected}


def _msg(*, key=TG_KEY, platform="telegram", kind="group",
         sender_id="admin1"):
    return SimpleNamespace(
        chat=SimpleNamespace(key=key, platform=platform, kind=kind,
                             title="Builders"),
        sender_id=sender_id, sender="Admin", sender_username="boss",
        text="", meta={}, reply_to="", message_id="900")


def _rig(tmp_path: Path, **adapter_kw):
    from nomorals.agents.partner.runtime_group import RuntimeGroupMixin

    surface = GroupSurface(GroupStore(tmp_path / "g.json").load())
    adapter = _Adapter(**adapter_kw)
    mixin = RuntimeGroupMixin()
    mixin._group_surface = lambda: surface  # type: ignore[method-assign]
    mixin._group_adapter = lambda platform: adapter  # type: ignore[method-assign]
    return mixin, surface, adapter


# ── registration ─────────────────────────────────────────────

def test_slowmode_registered_everywhere():
    assert CONTROL_COMMANDS["slowmode"] == (0, None)
    assert "slowmode" in PUBLIC_COMMANDS
    assert "slowmode" in WHATSAPP_UNSUPPORTED_KINDS
    details = COMMAND_DETAILS["slowmode"]
    assert "/slowmode 30s" in details["usage"]
    assert "Telegram" in details["what"]


def test_allowed_set_is_telegrams_steps():
    from nomorals.social.chat.telegram import SLOWMODE_ALLOWED_S

    assert SLOWMODE_ALLOWED_S == (0, 5, 10, 30, 60, 300, 900, 3600)


# ── the snap table ───────────────────────────────────────────

@pytest.mark.parametrize("requested,expected", [
    (0, 0), (5, 5), (10, 10), (30, 30), (45, 30), (59, 30),
    (60, 60), (119, 60), (300, 300), (599, 300), (900, 900),
    (3599, 900), (3600, 3600), (7200, 3600), (86400, 3600),
])
def test_slowmode_snap(requested, expected):
    from nomorals.agents.partner.runtime_group import _slowmode_snap

    assert _slowmode_snap(requested) == expected


@pytest.mark.parametrize("requested", [1, 2, 3, 4])
def test_slowmode_snap_too_short_is_none(requested):
    from nomorals.agents.partner.runtime_group import _slowmode_snap

    assert _slowmode_snap(requested) is None


# ── the command ──────────────────────────────────────────────

def test_admin_sets_30s_enforced_then_stored(tmp_path):
    mixin, surface, adapter = _rig(tmp_path)
    out = mixin._control_slowmode("30s", _msg())
    assert adapter.calls == [("slowmode", 30)]
    assert surface.slowmode_delay(TG_KEY) == 30
    assert "30s" in out
    audit = surface.audit(TG_KEY, limit=5)
    row = next(e for e in audit
               if e["kind"] == "admin_action"
               and e["action"] == "slowmode")
    assert row["seconds"] == 30 and row["previous"] == 0
    # Durable: a fresh surface over the same store sees it.
    surface2 = GroupSurface(GroupStore(tmp_path / "g.json").load())
    assert surface2.slowmode_delay(TG_KEY) == 30


def test_admin_turns_slowmode_off(tmp_path):
    mixin, surface, adapter = _rig(tmp_path)
    mixin._control_slowmode("30s", _msg())
    out = mixin._control_slowmode("off", _msg())
    assert adapter.calls[-1] == ("slowmode", 0)
    assert surface.slowmode_delay(TG_KEY) == 0
    assert "off" in out


def test_45s_snaps_to_30s_and_names_it(tmp_path):
    mixin, surface, adapter = _rig(tmp_path)
    out = mixin._control_slowmode("45s", _msg())
    assert adapter.calls == [("slowmode", 30)]
    assert surface.slowmode_delay(TG_KEY) == 30
    assert "30s" in out and "45s" in out


def test_2h_snaps_to_1h(tmp_path):
    mixin, surface, adapter = _rig(tmp_path)
    out = mixin._control_slowmode("2h", _msg())
    assert adapter.calls == [("slowmode", 3600)]
    assert "1h" in out and "2h" in out


def test_3s_refused_with_usage_and_no_call(tmp_path):
    mixin, surface, adapter = _rig(tmp_path)
    out = mixin._control_slowmode("3s", _msg())
    assert "5s" in out
    assert adapter.calls == []
    assert surface.slowmode_delay(TG_KEY) == 0


def test_garbage_duration_is_usage(tmp_path):
    mixin, surface, adapter = _rig(tmp_path)
    out = mixin._control_slowmode("banana", _msg())
    assert "usage:" in out
    assert adapter.calls == []


def test_non_admin_refused_without_adapter_calls(tmp_path):
    mixin, surface, adapter = _rig(tmp_path, admin=False)
    out = mixin._control_slowmode("30s", _msg())
    assert "admin" in out
    assert adapter.calls == []
    assert surface.slowmode_delay(TG_KEY) == 0


def test_dm_is_refused(tmp_path):
    mixin, surface, adapter = _rig(tmp_path)
    out = mixin._control_slowmode(
        "30s", _msg(kind="dm", key="telegram:123"))
    assert "group" in out
    assert adapter.calls == []


def test_whatsapp_group_gets_platform_truth(tmp_path):
    mixin, surface, adapter = _rig(tmp_path)
    out = mixin._control_slowmode(
        "30s", _msg(key=WA_KEY, platform="whatsapp"))
    assert "don't have slow mode" in out
    assert adapter.calls == []
    assert surface.slowmode_delay(WA_KEY) == 0


def test_bare_reports_stored_and_detected(tmp_path):
    mixin, surface, adapter = _rig(tmp_path, detected=0)
    out = mixin._control_slowmode("", _msg())
    assert "off" in out
    mixin._control_slowmode("30s", _msg())
    adapter.detected = 60
    out = mixin._control_slowmode("", _msg())
    assert "30s" in out and "1m" in out


def test_bare_matches_when_no_drift(tmp_path):
    mixin, surface, adapter = _rig(tmp_path, detected=30)
    mixin._control_slowmode("30s", _msg())
    out = mixin._control_slowmode("", _msg())
    assert "30s" in out
    assert "Telegram itself shows" not in out


def test_adapter_refusal_stores_nothing(tmp_path):
    mixin, surface, adapter = _rig(tmp_path, refuse=True)
    out = mixin._control_slowmode("30s", _msg())
    assert adapter.calls == [("slowmode", 30)]
    assert surface.slowmode_delay(TG_KEY) == 0
    assert "didn't change" in out
    audit = surface.audit(TG_KEY, limit=5)
    assert any(e["kind"] == "admin_action_failed"
               and e["action"] == "slowmode" for e in audit)


def test_adapter_without_the_verb_refuses_honestly(tmp_path):
    mixin, surface, adapter = _rig(tmp_path, verbs=False)
    out = mixin._control_slowmode("30s", _msg())
    assert "can't set it" in out
    assert surface.slowmode_delay(TG_KEY) == 0


def test_bot_api_adapter_names_the_real_reason(tmp_path):
    mixin, surface, adapter = _rig(
        tmp_path,
        fail_with=("the Bot API cannot set slow mode — it only "
                   "reads the delay; the userbot sets it"))
    out = mixin._control_slowmode("30s", _msg())
    assert "Bot API can't set it" in out
    assert "group itself" in out
    assert surface.slowmode_delay(TG_KEY) == 0


# ── console + gsettings ──────────────────────────────────────

def test_console_defense_section_shows_slowmode(tmp_path):
    mixin, surface, adapter = _rig(tmp_path)
    mixin._control_slowmode("30s", _msg())
    out = mixin._console_section(surface, TG_KEY, _msg(), "tok",
                                 "defense")
    assert "slow mode 30s" in out.text
    assert "slowmode 30s|off" in out.text
    buttons = " ".join(str(b) for b in (out.buttons or []))
    assert "Slow: 30s" in buttons and "Slow: off" in buttons


def test_console_defense_section_off_by_default(tmp_path):
    mixin, surface, adapter = _rig(tmp_path)
    out = mixin._console_section(surface, TG_KEY, _msg(), "tok",
                                 "defense")
    assert "slow mode off" in out.text


def test_console_defense_section_whatsapp_truth(tmp_path):
    mixin, surface, adapter = _rig(tmp_path)
    out = mixin._console_section(surface, WA_KEY,
                                 _msg(key=WA_KEY, platform="whatsapp"),
                                 "tok", "defense")
    assert "slow mode n/a on WhatsApp" in out.text
    buttons = " ".join(str(b) for b in (out.buttons or []))
    assert "Slow:" not in buttons


def test_console_apply_slowmode_enforces_and_stores(tmp_path):
    mixin, surface, adapter = _rig(tmp_path)
    out = mixin._console_apply(surface, TG_KEY, _msg(), "admin1",
                               "defense", ["slowmode", "5m"])
    assert "5m" in out
    assert adapter.calls == [("slowmode", 300)]
    assert surface.slowmode_delay(TG_KEY) == 300
    out = mixin._console_apply(surface, TG_KEY, _msg(), "admin1",
                               "defense", ["slowmode", "off"])
    assert "off" in out
    assert surface.slowmode_delay(TG_KEY) == 0


def test_console_apply_slowmode_refused_on_whatsapp(tmp_path):
    mixin, surface, adapter = _rig(tmp_path)
    out = mixin._console_apply(
        surface, WA_KEY, _msg(key=WA_KEY, platform="whatsapp"),
        "admin1", "defense", ["slowmode", "30s"])
    assert "don't have slow mode" in out
    assert adapter.calls == []


def test_gsettings_slowmode_enforces_and_stores(tmp_path):
    mixin, surface, adapter = _rig(tmp_path)
    out = mixin._control_gsettings("slowmode 1h", _msg())
    assert "1h" in out
    assert adapter.calls == [("slowmode", 3600)]
    assert surface.slowmode_delay(TG_KEY) == 3600


def test_gsettings_slowmode_refused_on_whatsapp(tmp_path):
    mixin, surface, adapter = _rig(tmp_path)
    out = mixin._control_gsettings(
        "slowmode 30s", _msg(key=WA_KEY, platform="whatsapp"))
    assert "don't have slow mode" in out
    assert adapter.calls == []


def test_gsettings_panel_shows_slowmode(tmp_path):
    mixin, surface, adapter = _rig(tmp_path)
    out = mixin._control_gsettings("", _msg())
    assert "slow mode off" in out.text
    assert "slowmode 30s|off" in out.text
    mixin._control_slowmode("30s", _msg())
    out = mixin._control_gsettings("", _msg())
    assert "slow mode 30s" in out.text


# ── /groups dashboard ────────────────────────────────────────

def test_groups_dashboard_line_shows_slowmode(tmp_path):
    mixin, surface, adapter = _rig(tmp_path)
    lines = mixin._groups_group_lines(surface, TG_KEY, "A")
    assert any("slow mode off" in line for line in lines)
    surface.set_slowmode_delay(TG_KEY, 300)
    lines = mixin._groups_group_lines(surface, TG_KEY, "A")
    assert any("slow mode 5m" in line for line in lines)


# ── adapter level (fake Telethon) ────────────────────────────

def _stub_telethon(monkeypatch):
    """Telethon is not installed on this box; stub just the two
    request classes the slow-mode coroutines build."""
    telethon = types.ModuleType("telethon")
    tl = types.ModuleType("telethon.tl")
    functions = types.ModuleType("telethon.tl.functions")
    channels = types.ModuleType("telethon.tl.functions.channels")

    class ToggleSlowModeRequest:
        def __init__(self, channel, seconds):
            self.channel = channel
            self.seconds = seconds

    class GetFullChannelRequest:
        def __init__(self, channel):
            self.channel = channel

    channels.ToggleSlowModeRequest = ToggleSlowModeRequest
    channels.GetFullChannelRequest = GetFullChannelRequest
    functions.channels = channels
    tl.functions = functions
    telethon.tl = tl
    for name, mod in (("telethon", telethon),
                      ("telethon.tl", tl),
                      ("telethon.tl.functions", functions),
                      ("telethon.tl.functions.channels", channels)):
        monkeypatch.setitem(sys.modules, name, mod)
    return ToggleSlowModeRequest, GetFullChannelRequest


class _FakeClient:
    def __init__(self, full_chat=None):
        self.requests: list = []
        self.full_chat = full_chat

    async def __call__(self, request):
        self.requests.append(request)
        return SimpleNamespace(full_chat=self.full_chat)


def _live_adapter(monkeypatch):
    """A TelegramAdapter with a real loop in a thread, a fake
    client, and _resolve stubbed — the _run_on_loop pipe is
    exercised for real."""
    from nomorals.social.chat.telegram import TelegramAdapter

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    adapter = TelegramAdapter.__new__(TelegramAdapter)
    adapter._loop = loop
    adapter._client = _FakeClient()
    adapter._thread = thread

    async def _resolve(chat):
        return "entity-1"

    adapter._resolve = _resolve  # type: ignore[method-assign]
    return adapter


def test_userbot_adapter_sets_slow_mode_via_toggle(monkeypatch,
                                                   tmp_path):
    ToggleSlowModeRequest, _ = _stub_telethon(monkeypatch)
    adapter = _live_adapter(monkeypatch)
    chat = SimpleNamespace(chat_id="-1001", platform="telegram",
                           kind="group")
    out = adapter.admin_set_slow_mode(chat, 30)
    assert out == {"ok": True}
    req = adapter._client.requests[0]
    assert isinstance(req, ToggleSlowModeRequest)
    assert req.channel == "entity-1" and req.seconds == 30
    adapter._loop.call_soon_threadsafe(adapter._loop.stop)


def test_userbot_adapter_refuses_unlisted_seconds(monkeypatch,
                                                  tmp_path):
    _stub_telethon(monkeypatch)
    adapter = _live_adapter(monkeypatch)
    chat = SimpleNamespace(chat_id="-1001", platform="telegram",
                           kind="group")
    out = adapter.admin_set_slow_mode(chat, 45)
    assert out["ok"] is False
    assert "45" in out["error"]
    assert adapter._client.requests == []
    adapter._loop.call_soon_threadsafe(adapter._loop.stop)


def test_userbot_adapter_reads_delay_via_full_channel(monkeypatch,
                                                      tmp_path):
    _, GetFullChannelRequest = _stub_telethon(monkeypatch)
    adapter = _live_adapter(monkeypatch)
    adapter._client.full_chat = SimpleNamespace(slowmode_seconds=60)
    chat = SimpleNamespace(chat_id="-1001", platform="telegram",
                           kind="group")
    out = adapter.chat_slow_mode_delay(chat)
    assert out == {"ok": True, "seconds": 60}
    assert isinstance(adapter._client.requests[0],
                      GetFullChannelRequest)
    # Absent field means off, not an error.
    adapter._client.full_chat = SimpleNamespace()
    out = adapter.chat_slow_mode_delay(chat)
    assert out == {"ok": True, "seconds": 0}
    adapter._loop.call_soon_threadsafe(adapter._loop.stop)


def test_bot_adapter_cannot_set_but_can_read():
    from nomorals.social.chat.telegram import TelegramBotAdapter

    adapter = TelegramBotAdapter.__new__(TelegramBotAdapter)
    chat = SimpleNamespace(chat_id="-1001", platform="telegram-bot",
                           kind="group")
    out = adapter.admin_set_slow_mode(chat, 30)
    assert out["ok"] is False
    assert "Bot API cannot set slow mode" in out["error"]
    adapter._admin = lambda method, chat, **kw: {  # type: ignore[method-assign]
        "ok": True, "result": {"slow_mode_delay": 60}}
    assert adapter.chat_slow_mode_delay(chat) == {
        "ok": True, "seconds": 60}
    adapter._admin = lambda method, chat, **kw: {  # type: ignore[method-assign]
        "ok": True, "result": {}}
    assert adapter.chat_slow_mode_delay(chat) == {
        "ok": True, "seconds": 0}

"""WhatsApp group help + menus (command census gaps #2, #4, #5).

The census found three surface defects on WhatsApp:

* bare /help inside a WhatsApp group served the generic DM menu —
  the help branch checked the platform before the group branch, so
  group help (live state, member/admin surfaces) was unreachable
  on WhatsApp;
* bare /start and /menu on WhatsApp served the Telegram public
  menu — the wrong command set, with "tap a button below"
  instructions on a surface that has no buttons;
* an unclaimed WhatsApp group was undiscoverable: no bot exists
  there, so the group systems stay off until the owner designates
  the room (§33), and nothing in the room said so — the economy
  sections of group help simply vanished and every refused
  command explained the wall only after the fact.

Pins:
- WhatsApp group /help serves GROUP help, with the room's claim
  state stated plainly (unclaimed + designated variants), in
  exactly one reply (claimed-once: never also a brain reply).
- WhatsApp /start and /menu serve the WhatsApp menu in DM and
  group contexts alike, never the Telegram public menu.
- Telegram /start, /menu and group /help are unchanged.

Scope expansion (owner, same day — replying to the moderation
seam): WhatsApp has no per-person mute, so the mute family must
not sit on the WhatsApp surface at all — absent from the
WhatsApp menu and from WhatsApp group help, a typed /mute or
/unmute on WhatsApp gets the platform truth at dispatch (never
_mod_run, never an admin_action_failed audit), and a WhatsApp
group's warn ladder cannot newly take a mute or restrict step
(the gsettings/console step menus don't offer them there;
stored ladders keep their data).  Telegram is untouched.
"""
from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import pytest

from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef
from nomorals.social.chat.control import (
    help_menu,
    help_text,
    public_menu_text,
    whatsapp_menu,
)

_TEST_TMPDIR = os.environ.get("NM_TEST_TMPDIR") or None

WA_MENU_MARKER = "text-first — no buttons"
TAP_A_BUTTON = "tap a button below"
GROUP_HELP_MARKER = "what I do here"
UNCLAIMED_MARKER = "not switched on yet"
CLAIMED_MARKER = "switched on — the owner designated this group"


@pytest.fixture()
def rig():
    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import load_settings

    tmp = tempfile.mkdtemp(prefix="nm-wamenu-", dir=_TEST_TMPDIR)
    settings = load_settings(overrides={
        "home": tmp,
        "partner.platforms": "local",
        "chat.local_enabled": "true",
    })
    context = build_context(settings, with_executor=False, with_tools=False)

    class _Router:
        def chat(self, *a, **k):
            raise AssertionError("the brain must not be called: "
                                 "the command path owns this message")

    context.router = _Router()

    sends: list[str] = []

    class _Gateway:
        adapters: dict = {}
        dry_run = False

        def status(self):
            return {"local": {"running_in_session": True}}

        def set_rate_limit(self, *a, **k):
            return None

        def typing(self, *a, **k):
            return True

        def send(self, platform, chat, text, **kw):
            sends.append(text)
            return SimpleNamespace(ok=True)

    runtime = PartnerRuntime(context, gateway=_Gateway())
    yield runtime, sends
    try:
        runtime.stop()
    except Exception:  # noqa: BLE001 - best-effort teardown
        pass
    try:
        context.close()
    except Exception:  # noqa: BLE001 - best-effort teardown
        pass


def _msg(text, platform, kind, owner=False, chat_id="777"):
    return ChatMessage(
        chat=ChatRef(platform=platform, chat_id=chat_id, kind=kind),
        incoming=True, text=text, sender="member" if not owner else "owner",
        sender_id="42" if not owner else "1",
        meta={"is_owner": True} if owner else {})


# ── routing: WhatsApp group /help reaches GROUP help ─────────

def test_whatsapp_group_help_is_group_help_unclaimed(rig):
    runtime, sends = rig
    runtime._process_inner(_msg("/help", "whatsapp", ChatKind.GROUP))
    assert len(sends) == 1  # claimed once: one reply, no brain line
    reply = sends[0]
    assert GROUP_HELP_MARKER in reply
    # the room's claim state is stated, before any refusal teaches it
    assert UNCLAIMED_MARKER in reply
    assert "what works here today" in reply
    assert "/gsettings system output on" in reply
    # and it is NOT the generic WhatsApp DM menu
    assert WA_MENU_MARKER not in reply
    assert TAP_A_BUTTON not in reply


def test_whatsapp_group_help_claimed_after_designation(rig):
    runtime, sends = rig
    runtime._group_surface().set_designated(
        "whatsapp:777", True, by="owner")
    runtime._process_inner(_msg("/help", "whatsapp", ChatKind.GROUP))
    assert len(sends) == 1
    reply = sends[0]
    assert GROUP_HELP_MARKER in reply
    assert CLAIMED_MARKER in reply
    assert UNCLAIMED_MARKER not in reply
    assert WA_MENU_MARKER not in reply


def test_telegram_group_help_has_no_whatsapp_claim_block(rig):
    runtime, sends = rig
    runtime._process_inner(_msg("/help", "telegram", ChatKind.GROUP))
    assert len(sends) == 1
    reply = sends[0]
    assert GROUP_HELP_MARKER in reply
    assert UNCLAIMED_MARKER not in reply
    assert CLAIMED_MARKER not in reply


def test_whatsapp_group_help_topic_still_detailed(rig):
    runtime, sends = rig
    runtime._process_inner(
        _msg("/help balance", "whatsapp", ChatKind.GROUP))
    assert len(sends) == 1
    assert sends[0].strip()
    assert GROUP_HELP_MARKER not in sends[0]


# ── dispatch: /start and /menu serve the WhatsApp menu ───────

def test_whatsapp_start_and_menu_dm(rig):
    runtime, _sends = rig
    for text in ("/start", "/menu"):
        out = runtime.handle_control(
            text, "whatsapp:15551234567",
            message=_msg(text, "whatsapp", ChatKind.DM, owner=True,
                         chat_id="15551234567"))
        assert out == help_menu("whatsapp")
        assert WA_MENU_MARKER in out
        assert TAP_A_BUTTON not in out


def test_whatsapp_start_and_menu_group_context(rig):
    runtime, _sends = rig
    for text in ("/start", "/menu"):
        out = runtime.handle_control(
            text, "whatsapp:777",
            message=_msg(text, "whatsapp", ChatKind.GROUP, owner=True))
        assert out == help_menu("whatsapp")
        assert TAP_A_BUTTON not in out


def test_telegram_start_and_menu_unchanged(rig):
    runtime, _sends = rig
    for text in ("/start", "/menu"):
        out = runtime.handle_control(
            text, "telegram:999",
            message=_msg(text, "telegram", ChatKind.DM, owner=True,
                         chat_id="999"))
        assert out == public_menu_text()
        assert TAP_A_BUTTON in out


def test_whatsapp_dm_help_still_whatsapp_menu(rig):
    runtime, _sends = rig
    out = runtime.handle_control(
        "/help", "whatsapp:15551234567",
        message=_msg("/help", "whatsapp", ChatKind.DM, owner=True,
                     chat_id="15551234567"))
    assert out == help_menu("whatsapp")


# ── unit: the claim block inside _control_group_help ─────────

def _group_help_rig(tmp_path, key, platform):
    from nomorals.agents.partner.runtime_group import RuntimeGroupMixin
    from nomorals.social.chat.group_surface import GroupStore, GroupSurface

    class _Gateway:
        def __init__(self):
            self.sent: list = []

        def send(self, platform, chat, text, **kw):
            self.sent.append({"platform": platform, "text": text, **kw})
            return {"ok": True}

    class _Adapter:
        def admin_member(self, chat, user_id):
            return {"status": "administrator"}

    class _RT(RuntimeGroupMixin):
        operator_ids = frozenset()

        def _group_adapter(self, platform):  # type: ignore[no-untyped-def]
            return _Adapter()

    rt = _RT()
    rt.gateway = _Gateway()
    surface = GroupSurface(GroupStore(tmp_path / "g.json").load())
    rt._group_surface_obj = surface
    msg = SimpleNamespace(
        chat=SimpleNamespace(key=key, platform=platform,
                             kind="group", title="Crew"),
        sender_id="m1", sender="Member", sender_username="",
        sender_role="", text="/help", meta={}, reply_to="",
        message_id="1")
    return rt, surface, msg


def test_group_help_whatsapp_unclaimed_block(tmp_path):
    rt, _surface, msg = _group_help_rig(
        tmp_path, "whatsapp:120363@g.us", "whatsapp")
    out = rt._control_group_help(msg)
    assert UNCLAIMED_MARKER in out.text
    assert "what works here today" in out.text
    assert "/gsettings system output on" in out.text


def test_group_help_whatsapp_claimed_block(tmp_path):
    rt, surface, msg = _group_help_rig(
        tmp_path, "whatsapp:120363@g.us", "whatsapp")
    surface.set_designated("whatsapp:120363@g.us", True, by="owner")
    out = rt._control_group_help(msg)
    assert CLAIMED_MARKER in out.text
    assert UNCLAIMED_MARKER not in out.text


def test_group_help_telegram_no_claim_block(tmp_path):
    rt, _surface, msg = _group_help_rig(
        tmp_path, "telegram:-1001", "telegram")
    out = rt._control_group_help(msg)
    assert UNCLAIMED_MARKER not in out.text
    assert CLAIMED_MARKER not in out.text


# ── scope expansion: the mute family is not on WhatsApp ──────

NO_MUTE_LINE = ("WhatsApp has no per-person mute — a member "
                "here can only be removed, or left as they are.")


def test_whatsapp_menu_omits_mute_family():
    menu = whatsapp_menu()
    assert "/mute" not in menu
    assert "/unmute" not in menu
    # the global catalog and the Telegram menu keep them
    assert "/mute" in help_text()
    assert "/mute" in public_menu_text()


def test_group_help_whatsapp_omits_mute_family(tmp_path):
    rt, _surface, msg = _group_help_rig(
        tmp_path, "whatsapp:120363@g.us", "whatsapp")
    out = rt._control_group_help(msg)
    assert "/mute" not in out.text
    assert "/unmute" not in out.text
    # the platform truth is stated in their place
    assert "no per-person mute" in out.text


def test_group_help_telegram_keeps_mute_family(tmp_path):
    rt, _surface, msg = _group_help_rig(
        tmp_path, "telegram:-1001", "telegram")
    out = rt._control_group_help(msg)
    assert "/mute" in out.text
    assert "/unmute" in out.text


def test_whatsapp_group_mute_gets_truth_no_mod_run(rig, monkeypatch):
    runtime, sends = rig
    calls: list = []
    monkeypatch.setattr(
        runtime, "_mod_run",
        lambda *a, **k: calls.append((a, k)) or True)
    runtime._process_inner(
        _msg("/mute @member 10m", "whatsapp", ChatKind.GROUP))
    # one reply, the platform truth — no moderation attempt,
    # so nothing for _mod_run to audit as failed
    assert sends == [NO_MUTE_LINE]
    assert calls == []


def test_whatsapp_group_unmute_gets_truth_no_mod_run(rig, monkeypatch):
    runtime, sends = rig
    calls: list = []
    monkeypatch.setattr(
        runtime, "_mod_run",
        lambda *a, **k: calls.append((a, k)) or True)
    runtime._process_inner(
        _msg("/unmute @member", "whatsapp", ChatKind.GROUP))
    assert sends == [NO_MUTE_LINE]
    assert calls == []


def test_whatsapp_dm_mute_gets_truth(rig):
    runtime, _sends = rig
    out = runtime.handle_control(
        "/mute @member 10m", "whatsapp:15551234567",
        message=_msg("/mute @member 10m", "whatsapp", ChatKind.DM,
                     owner=True, chat_id="15551234567"))
    assert out == NO_MUTE_LINE


def test_telegram_group_mute_is_not_the_truth_line(rig):
    runtime, sends = rig
    runtime._process_inner(
        _msg("/mute @member 10m", "telegram", ChatKind.GROUP))
    # a non-admin member meets the moderation gate, as before —
    # the WhatsApp platform answer must not leak onto Telegram
    assert len(sends) == 1
    assert sends[0] != NO_MUTE_LINE
    assert "per-person mute" not in sends[0]


# ── scope expansion: the warn ladder on WhatsApp ─────────────

def _esc_rig(tmp_path, key, platform):
    rt, surface, msg = _group_help_rig(tmp_path, key, platform)
    return rt, surface, msg


def test_ladder_whatsapp_cannot_design_mute_or_restrict(tmp_path):
    key = "whatsapp:120363@g.us"
    rt, surface, _msg_ = _esc_rig(tmp_path, key, "whatsapp")
    out = rt._apply_escalation_grammar(
        surface, key, ["at", "3", "mute", "10m"])
    assert "no per-person mute" in out
    out = rt._apply_escalation_grammar(
        surface, key, ["at", "4", "restrict"])
    assert "no per-person mute" in out
    cfg = surface.escalation_config(key)
    # the refusals designed nothing: no step exists at 4, and
    # the at-3 step is still the untouched house default
    # (60 minutes — the refused 10m design never landed)
    assert all(s["at"] != 4 for s in cfg.intervals)
    at3 = [s for s in cfg.intervals if s["at"] == 3]
    assert at3 == [{"at": 3, "action": "mute", "minutes": 60}]


def test_ladder_whatsapp_fine_step_still_designable(tmp_path):
    key = "whatsapp:120363@g.us"
    rt, surface, _msg_ = _esc_rig(tmp_path, key, "whatsapp")
    out = rt._apply_escalation_grammar(
        surface, key, ["at", "4", "fine", "100"])
    assert "step designed: fine at 4." in out
    stored = surface.store.settings(key)["escalation"]
    assert {"at": 4, "action": "fine", "amount": 100} in \
        stored["intervals"]


def test_ladder_whatsapp_reset_drops_restrict_bottomed_steps(tmp_path):
    key = "whatsapp:120363@g.us"
    rt, surface, _msg_ = _esc_rig(tmp_path, key, "whatsapp")
    out = rt._apply_escalation_grammar(surface, key, ["reset"])
    assert "house ladder" in out
    stored = surface.store.settings(key)["escalation"]
    actions = [s["action"] for s in stored["intervals"]]
    assert "mute" not in actions
    assert "restrict" not in actions
    assert "kick" in actions  # the rest of the house ladder lands


def test_ladder_telegram_mute_step_and_reset_unchanged(tmp_path):
    key = "telegram:-1001"
    rt, surface, _msg_ = _esc_rig(tmp_path, key, "telegram")
    out = rt._apply_escalation_grammar(
        surface, key, ["at", "3", "mute", "10m"])
    assert "step designed: mute at 3." in out
    out = rt._apply_escalation_grammar(surface, key, ["reset"])
    assert "house ladder" in out
    stored = surface.store.settings(key)["escalation"]
    actions = [s["action"] for s in stored["intervals"]]
    assert "mute" in actions  # the house ladder, intact


def test_ladder_whatsapp_stored_mute_step_data_left_alone(tmp_path):
    key = "whatsapp:120363@g.us"
    rt, surface, _msg_ = _esc_rig(tmp_path, key, "whatsapp")
    surface.update_settings(key, escalation={
        "warn_limit": 7, "final_action": "ban", "window_days": 7,
        "intervals": [{"at": 2, "action": "mute", "minutes": 30}]})
    out = rt._apply_escalation_grammar(surface, key, ["limit", "5"])
    assert "warn limit 5" in out
    stored = surface.store.settings(key)["escalation"]
    # an unrelated change does not strip the stored step
    assert {"at": 2, "action": "mute", "minutes": 30} in \
        stored["intervals"]
    assert stored["warn_limit"] == 5


def test_gsettings_step_offer_whatsapp_vs_telegram(tmp_path):
    rt, _surface, msg = _group_help_rig(
        tmp_path, "whatsapp:120363@g.us", "whatsapp")
    out = rt._control_gsettings("", msg)
    assert "escalation at N mute" not in out.text
    rt2, _surface2, msg2 = _group_help_rig(
        tmp_path, "telegram:-1001", "telegram")
    out2 = rt2._control_gsettings("", msg2)
    assert "escalation at N mute" in out2.text


def test_gsettings_grammar_fallback_offer_by_platform(tmp_path):
    rt, _surface, msg = _group_help_rig(
        tmp_path, "whatsapp:120363@g.us", "whatsapp")
    out = rt._control_gsettings("escalation bogus", msg)
    assert "at N mute" not in out
    assert "no mute or restrict step on WhatsApp" in out
    rt2, _surface2, msg2 = _group_help_rig(
        tmp_path, "telegram:-1001", "telegram")
    out2 = rt2._control_gsettings("escalation bogus", msg2)
    assert "at N mute" in out2


def test_console_grammar_step_offer_by_platform(tmp_path):
    rt, surface, msg = _group_help_rig(
        tmp_path, "whatsapp:120363@g.us", "whatsapp")
    out = rt._console_section(
        surface, "whatsapp:120363@g.us", msg, "tok", "defense")
    assert "escalation at N mute" not in out.text
    rt2, surface2, msg2 = _group_help_rig(
        tmp_path, "telegram:-1001", "telegram")
    out2 = rt2._console_section(
        surface2, "telegram:-1001", msg2, "tok", "defense")
    assert "escalation at N mute" in out2.text

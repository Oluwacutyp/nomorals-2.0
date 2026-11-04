"""Sensor-executor audit honesty (groups/channels census §6.1).

Every adapter in the codebase reports a platform refusal as an
``{"ok": False, "error": ...}`` dict, never as an exception.
The autonomy executor (`_execute_group_action`) must therefore
audit a dict refusal as an UNEXECUTION carrying the platform's
reason — a refused delete/ban/mute recorded as
``moderation_executed`` is a false success in the one record
that must never lie.  The same truth applies to the
service-message cleaner (`service_cleaned` only when the delete
actually reported ok) and to a bought pin grant (`pin_grant_used`
only when the pin actually landed; a refusal restores the
grant instead of spending it).

Rigs follow the house shapes from test_group_defense_system:
GroupSurface over a tmp_path store, a RuntimeGroupMixin stub,
and fake adapters at the admin seam.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from nomorals.agents.partner.runtime_group import RuntimeGroupMixin
from nomorals.social.chat.base import ChatKind
from nomorals.social.chat.group_surface import default_surface

KEY = "telegram-bot:-1001234567"
WA_KEY = "whatsapp:120363012345@g.us"

# The WhatsApp adapter's own admin_restrict refusal, verbatim
# (social/chat/whatsapp.py): WhatsApp has no per-member mute,
# so the verb refuses by design on every call.
WA_MUTE_ERROR = (
    "unsupported: WhatsApp has no per-member mute — a member "
    "can only be removed, or the whole group set to "
    "admins-only; neither is a mute, so nothing was applied")


class _Chat:
    def __init__(self, platform="telegram-bot", key=KEY):
        self.platform = platform
        self.chat_id = key.split(":", 1)[1]
        self.title = "Builders"
        self.key = key
        self.kind = ChatKind.GROUP


class _Msg:
    def __init__(self, chat=None, **kw):
        self.chat = chat or _Chat()
        self.meta = {}
        self.sender_id = kw.get("sender_id", "7")
        self.sender = kw.get("sender", "Ada")
        self.sender_username = kw.get("sender_username", "ada")
        self.text = ""
        self.message_id = kw.get("message_id", "55")
        self.reply_to = ""


class _RefusingAdapter:
    """Every admin verb refuses the way the real adapters do:
    an ok:False dict carrying the platform's reason."""

    def __init__(self, error="platform refused"):
        self.calls = []
        self.error = error

    def admin_delete(self, chat, message_id):
        self.calls.append(("delete", message_id))
        return {"ok": False, "error": self.error}

    def admin_ban(self, chat, user_id):
        self.calls.append(("ban", user_id))
        return {"ok": False, "error": self.error}

    def admin_unban(self, chat, user_id):
        self.calls.append(("unban", user_id))
        return {"ok": True}

    def admin_restrict(self, chat, user_id, *, minutes=0, **kw):
        self.calls.append(("restrict", user_id, minutes))
        return {"ok": False, "error": self.error}

    def admin_pin(self, chat, message_id):
        self.calls.append(("pin", message_id))
        return {"ok": False, "error": self.error}


class _OkAdapter(_RefusingAdapter):
    """The same seam, with the platform performing each verb."""

    def admin_delete(self, chat, message_id):
        self.calls.append(("delete", message_id))
        return {"ok": True}

    def admin_ban(self, chat, user_id):
        self.calls.append(("ban", user_id))
        return {"ok": True}

    def admin_restrict(self, chat, user_id, *, minutes=0, **kw):
        self.calls.append(("restrict", user_id, minutes))
        return {"ok": True}

    def admin_pin(self, chat, message_id):
        self.calls.append(("pin", message_id))
        return {"ok": True}


class _RaisingAdapter(_RefusingAdapter):
    """The seam failing by exception (the pre-existing path)."""

    def admin_delete(self, chat, message_id):
        raise RuntimeError("connection lost")

    def admin_ban(self, chat, user_id):
        raise RuntimeError("connection lost")

    def admin_restrict(self, chat, user_id, *, minutes=0, **kw):
        raise RuntimeError("connection lost")


class _Runtime(RuntimeGroupMixin):
    def __init__(self, tmp_path, adapter, platform="telegram-bot"):
        self.sent = []
        self.gateway = SimpleNamespace(adapters={platform: adapter})
        self.context = SimpleNamespace(settings=SimpleNamespace(
            workspace_dir=str(tmp_path)))
        self._group_surface_obj = default_surface(str(tmp_path))

    def _send(self, chat, text, **kw):
        self.sent.append(str(text))

    def _is_operator(self, message):
        return False


def _kinds(surface, key=KEY):
    return [a["kind"] for a in surface.audit(key)]


# ── _execute_group_action: the generic delete/ban/mute branch ──


def test_sensor_dict_refusals_audit_unexecuted_never_executed(
        tmp_path: Path):
    for i, kind in enumerate(("delete", "ban", "mute")):
        adapter = _RefusingAdapter(error=f"refused {kind} here")
        rt = _Runtime(tmp_path / str(i), adapter)
        surface = rt._group_surface_obj
        msg = _Msg()
        done = rt._execute_group_action(
            surface, KEY, adapter, msg,
            {"kind": kind, "user_id": "42", "rule": "flood"})
        assert done is False, kind
        kinds = _kinds(surface)
        assert "moderation_executed" not in kinds, kind
        unexecuted = [a for a in surface.audit(KEY)
                      if a["kind"] == "moderation_unexecuted"]
        assert len(unexecuted) == 1, kind
        assert unexecuted[0]["action"] == kind
        assert f"refused {kind} here" in unexecuted[0]["reason"]


def test_sensor_ok_dicts_still_audit_executed(tmp_path: Path):
    for i, kind in enumerate(("delete", "ban", "mute")):
        adapter = _OkAdapter()
        rt = _Runtime(tmp_path / str(i), adapter)
        surface = rt._group_surface_obj
        done = rt._execute_group_action(
            surface, KEY, adapter, _Msg(),
            {"kind": kind, "user_id": "42", "rule": "flood"})
        assert done is True, kind
        kinds = _kinds(surface)
        assert kinds.count("moderation_executed") == 1, kind
        assert "moderation_unexecuted" not in kinds, kind


def test_sensor_exception_path_unchanged(tmp_path: Path):
    for i, kind in enumerate(("delete", "ban", "mute")):
        adapter = _RaisingAdapter()
        rt = _Runtime(tmp_path / str(i), adapter)
        surface = rt._group_surface_obj
        done = rt._execute_group_action(
            surface, KEY, adapter, _Msg(),
            {"kind": kind, "user_id": "42", "rule": "flood"})
        assert done is False, kind
        kinds = _kinds(surface)
        assert "moderation_executed" not in kinds, kind
        unexecuted = [a for a in surface.audit(KEY)
                      if a["kind"] == "moderation_unexecuted"]
        assert len(unexecuted) == 1, kind
        assert "connection lost" in unexecuted[0]["reason"]


def test_sensor_whatsapp_mute_refusal_end_to_end(tmp_path: Path):
    """The census's live case: a defense mute dispatched on
    WhatsApp hits admin_restrict's by-design ok:False refusal.
    Through the executor it must land as an unexecution naming
    the platform's reason — not as a mute that happened."""
    adapter = _RefusingAdapter(error=WA_MUTE_ERROR)
    rt = _Runtime(tmp_path, adapter, platform="whatsapp")
    surface = rt._group_surface_obj
    chat = _Chat(platform="whatsapp", key=WA_KEY)
    msg = _Msg(chat=chat, sender_id="999")
    done = rt._execute_group_action(
        surface, WA_KEY, adapter, msg,
        {"kind": "mute", "user_id": "999", "rule": "flood",
         "minutes": 60})
    assert done is False
    assert ("restrict", "999", 60) in adapter.calls
    kinds = _kinds(surface, WA_KEY)
    assert "moderation_executed" not in kinds
    unexecuted = [a for a in surface.audit(WA_KEY)
                  if a["kind"] == "moderation_unexecuted"]
    assert len(unexecuted) == 1
    assert unexecuted[0]["action"] == "mute"
    assert "no per-member mute" in unexecuted[0]["reason"]


def test_sensor_kick_refusal_still_unexecuted(tmp_path: Path):
    """The kick branch's check, pinned so the generic branch's
    new check can't drift from it."""
    adapter = _RefusingAdapter(error="ban refused")
    rt = _Runtime(tmp_path, adapter)
    surface = rt._group_surface_obj
    done = rt._execute_group_action(
        surface, KEY, adapter, _Msg(),
        {"kind": "kick", "user_id": "42", "rule": "flood"})
    assert done is False
    kinds = _kinds(surface)
    assert "moderation_executed" not in kinds
    assert "moderation_unexecuted" in kinds
    # A refused ban means the unban lift is never attempted.
    assert ("unban", "42") not in adapter.calls


# ── _clean_service_message ──────────────────────────────────────


def test_service_clean_dict_refusal_audits_refusal(tmp_path: Path):
    adapter = _RefusingAdapter(error="message already deleted")
    rt = _Runtime(tmp_path, adapter)
    surface = rt._group_surface_obj
    rt._clean_service_message(surface, _Msg(message_id="77"))
    kinds = _kinds(surface)
    assert "service_cleaned" not in kinds
    refused = [a for a in surface.audit(KEY)
               if a["kind"] == "service_clean_refused"]
    assert len(refused) == 1
    assert refused[0]["message_id"] == "77"
    assert "already deleted" in refused[0]["reason"]


def test_service_clean_ok_still_audits_cleaned(tmp_path: Path):
    adapter = _OkAdapter()
    rt = _Runtime(tmp_path, adapter)
    surface = rt._group_surface_obj
    rt._clean_service_message(surface, _Msg(message_id="77"))
    kinds = _kinds(surface)
    assert kinds.count("service_cleaned") == 1
    assert "service_clean_refused" not in kinds


# ── _execute_pending_pin (third site, found by the sweep) ──────


def test_pin_grant_refusal_restores_grant_and_audits_refusal(
        tmp_path: Path):
    adapter = _RefusingAdapter(error="not enough rights to pin")
    rt = _Runtime(tmp_path, adapter)
    surface = rt._group_surface_obj
    surface.grant_pin(KEY, "7")
    rt._execute_pending_pin(surface, _Msg(message_id="88"))
    kinds = _kinds(surface)
    assert "pin_grant_used" not in kinds
    refused = [a for a in surface.audit(KEY)
               if a["kind"] == "pin_grant_refused"]
    assert len(refused) == 1
    assert "not enough rights" in refused[0]["reason"]
    # The grant was not spent on a pin that never happened.
    assert surface.take_pending_pin(KEY, "7") is True


def test_pin_grant_ok_audits_used_and_spends_grant(
        tmp_path: Path):
    adapter = _OkAdapter()
    rt = _Runtime(tmp_path, adapter)
    surface = rt._group_surface_obj
    surface.grant_pin(KEY, "7")
    rt._execute_pending_pin(surface, _Msg(message_id="88"))
    kinds = _kinds(surface)
    assert kinds.count("pin_grant_used") == 1
    assert "pin_grant_refused" not in kinds
    assert surface.take_pending_pin(KEY, "7") is False

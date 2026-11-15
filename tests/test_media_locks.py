"""Media locks (DESIGN §4): per-group toggles over which media kinds
members may send — stickers/gifs/voice/forwards/channel-posts (the
design's five) plus the rest of the family the envelope expresses
(photos, videos, audio, documents, video notes).

The sensor rides the stage-3 envelope sensors (antichannel/antitag
shape): it reads the normalized envelope, fires one SensorHit per
message, and the surface audits + deletes/warns + strikes on the
shared spine. Admins are exempt at the sensor (antitag shape); the
owner ladder is exempt at the surface (antichannel shape, §19).
All locks default OFF — an unconfigured group sees no change.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from nomorals.agents.partner.runtime_group import RuntimeGroupMixin
from nomorals.security.payload_inspection import normalize_envelope
from nomorals.social.chat.base import ChatKind
from nomorals.social.chat.group_defense import (
    MEDIA_LOCK_KINDS,
    MEDIA_LOCK_LABELS,
    DefenseConfig,
    check_media_locks,
    describe_media_locks,
)
from nomorals.social.chat.group_surface import (
    GroupStore,
    GroupSurface,
    _default_settings,
)

KEY = "telegram-bot:-1001234567"
WA_KEY = "whatsapp:120363012345@g.us"
NOW = 1_700_000_000.0


def _surface(tmp_path: Path) -> GroupSurface:
    return GroupSurface(GroupStore(tmp_path / "g.json").load())


def _cfg(**locks):
    block = {"enabled": True, "action": "delete_warn"}
    block.update(locks)
    return DefenseConfig.from_settings(
        {"defense": {"media_locks": block}})


def _env(**kw):
    return normalize_envelope(kw or None)


def _lock(surface, key=KEY, **locks):
    s = surface.store.settings(key)
    d = dict(s.get("defense") or {})
    ml = dict(d.get("media_locks") or {})
    ml.update(locks)
    d["media_locks"] = ml
    surface.update_settings(key, defense=d)


# ── sensor units ────────────────────────────────────────────────


def test_lock_kinds_cover_the_design_five():
    for kind in ("stickers", "gifs", "voice", "forwards",
                 "channel_posts"):
        assert kind in MEDIA_LOCK_KINDS


def test_defaults_lock_nothing():
    cfg = DefenseConfig.from_settings({})
    for env in ({"media": {"kind": "sticker"}},
                {"media": {"kind": "animation"}},
                {"media": {"kind": "voice"}},
                {"forward": {"is_forwarded": True,
                             "origin_kind": "user"}},
                {"forward": {"is_forwarded": True,
                             "origin_kind": "channel"}}):
        assert check_media_locks(_env(**env), cfg) is None
    assert "media_locks" in _default_settings()["defense"]
    assert describe_media_locks({}) == (
        "none locked — everything open")


def test_each_locked_media_kind_fires():
    cases = [
        ("stickers", {"media": {"kind": "sticker"}}),
        ("gifs", {"media": {"kind": "animation"}}),
        ("voice", {"media": {"kind": "voice"}}),
        ("video_notes", {"media": {"kind": "video_note"}}),
        ("photos", {"media": {"kind": "photo"}}),
        ("videos", {"media": {"kind": "video"}}),
        ("audio", {"media": {"kind": "audio"}}),
        ("documents", {"media": {"kind": "document"}}),
    ]
    for kind, raw in cases:
        cfg = _cfg(**{kind: True})
        hit = check_media_locks(_env(**raw), cfg)
        assert hit is not None, kind
        assert hit.rule == "media_locks"
        assert hit.signal["kind"] == kind
        assert MEDIA_LOCK_LABELS[kind] in hit.reason
        assert hit.action == "delete_warn"
        # unrelated kinds pass under the same lock
        other = {"media": {"kind": "photo"}} \
            if kind != "photos" else {"media": {"kind": "video"}}
        assert check_media_locks(_env(**other), cfg) is None


def test_master_switch_disables_the_sensor():
    cfg = _cfg(enabled=False, stickers=True, gifs=True,
               forwards=True, channel_posts=True)
    assert check_media_locks(
        _env(media={"kind": "sticker"}), cfg) is None
    assert check_media_locks(
        _env(forward={"is_forwarded": True,
                      "origin_kind": "channel"}), cfg) is None


def test_whatsapp_kind_aliases():
    cfg = _cfg(photos=True, voice=True, documents=True)
    assert check_media_locks(
        _env(media={"kind": "image"}), cfg).signal["kind"] \
        == "photos"
    assert check_media_locks(
        _env(media={"kind": "voice"}), cfg).signal["kind"] \
        == "voice"
    assert check_media_locks(
        _env(media={"kind": "document"}), cfg).signal["kind"] \
        == "documents"
    assert check_media_locks(
        _env(media={"kind": "video"}), cfg) is None


def test_sticker_envelope_fires_sticker_lock():
    cfg = _cfg(stickers=True)
    hit = check_media_locks(
        _env(sticker={"key": "abc", "pack": "p"}), cfg)
    assert hit is not None
    assert hit.signal["kind"] == "stickers"


def test_forward_lock_fires_on_any_forward():
    cfg = _cfg(forwards=True)
    for origin in ("user", "group", "channel"):
        hit = check_media_locks(
            _env(forward={"is_forwarded": True,
                          "origin_kind": origin}), cfg)
        assert hit is not None, origin
        assert hit.signal["kind"] == "forwards"
    assert check_media_locks(_env(), cfg) is None


def test_channel_post_lock():
    cfg = _cfg(channel_posts=True)
    hit = check_media_locks(
        _env(forward={"is_forwarded": True,
                      "origin_kind": "channel",
                      "origin_name": "Signals",
                      "sender_chat_id": "-100777"}), cfg)
    assert hit is not None
    assert hit.signal["kind"] == "channel_posts"
    # a channel speaking as itself, no forward header
    hit = check_media_locks(
        _env(forward={"is_forwarded": False,
                      "sender_chat_id": "-100777"}), cfg)
    assert hit is not None
    # a WhatsApp newsletter posting into the group
    hit = check_media_locks(
        _env(platform="whatsapp", chat_key=WA_KEY,
             sender_id="1234@newsletter"), cfg)
    assert hit is not None
    # a plain member post is not a channel post
    assert check_media_locks(
        _env(forward={"is_forwarded": True,
                      "origin_kind": "user"}), cfg) is None


def test_linked_channel_is_exempt():
    cfg = _cfg(channel_posts=True, forwards=True)
    auto = {"forward": {"is_forwarded": True,
                        "is_automatic": True,
                        "origin_kind": "channel",
                        "sender_chat_id": "-100777"}}
    # automatic forwards are the linked channel's own posts —
    # the feature, not an attack: neither lock fires
    assert check_media_locks(_env(**auto), cfg) is None
    # the explicit linked id exempts channel posts…
    cfg_ch = _cfg(channel_posts=True)
    linked = {"forward": {"is_forwarded": True,
                          "origin_kind": "channel",
                          "sender_chat_id": "-100777"}}
    assert check_media_locks(
        _env(**linked), cfg_ch, linked_chat_id="-100777") is None
    # …but a member manually forwarding the linked channel's
    # post is still a forward
    hit = check_media_locks(_env(**linked), cfg)
    assert hit is not None
    assert hit.signal["kind"] == "forwards"
    # …and a message speaking AS the linked channel passes
    # even a media lock
    as_channel = {"forward": {"is_forwarded": False,
                              "sender_chat_id": "-100777"},
                  "media": {"kind": "photo"}}
    cfg_ph = _cfg(photos=True, channel_posts=True)
    assert check_media_locks(
        _env(**as_channel), cfg_ph, linked_chat_id="-100777") \
        is None
    # …while a foreign channel's photo is locked media
    assert check_media_locks(_env(**as_channel), cfg_ph) \
        is not None


def test_action_variants():
    cfg = _cfg(stickers=True, action="warn")
    hit = check_media_locks(
        _env(media={"kind": "sticker"}), cfg)
    assert hit.action == "warn"
    cfg = _cfg(stickers=True, action="delete")
    assert check_media_locks(
        _env(media={"kind": "sticker"}), cfg).action == "delete"
    cfg = _cfg(stickers=True, action="explode")
    assert check_media_locks(
        _env(media={"kind": "sticker"}), cfg).action \
        == "delete_warn"


def test_mime_fallback_when_kind_absent():
    cfg = _cfg(gifs=True, photos=True, videos=True, audio=True)
    assert check_media_locks(
        _env(media={"mime": "image/gif"}), cfg).signal["kind"] \
        == "gifs"
    assert check_media_locks(
        _env(media={"mime": "image/jpeg"}), cfg).signal["kind"] \
        == "photos"
    assert check_media_locks(
        _env(media={"mime": "video/mp4"}), cfg).signal["kind"] \
        == "videos"
    assert check_media_locks(
        _env(media={"mime": "audio/ogg"}), cfg).signal["kind"] \
        == "audio"
    assert check_media_locks(
        _env(media={"mime": "application/pdf"}), cfg) is None


def test_garbage_envelope_is_safe():
    cfg = _cfg(stickers=True, forwards=True, channel_posts=True)
    assert check_media_locks(
        _env(media=42, forward="junk", sticker=[]), cfg) is None


def test_admin_exemption_at_the_sensor():
    cfg = _cfg(stickers=True)
    assert check_media_locks(
        _env(media={"kind": "sticker"}), cfg,
        sender_is_admin=True) is None


# ── surface: the pipeline stage ─────────────────────────────────


def test_locked_sticker_deleted_struck_and_audited(tmp_path):
    surface = _surface(tmp_path)
    _lock(surface, stickers=True)
    out = surface.handle_message(
        KEY, "7", name="Ada", text="", message_id="1", ts=NOW,
        envelope={"media": {"kind": "sticker"}})
    assert [a["kind"] for a in out["actions"]] == ["delete"]
    assert surface.strikes_for(KEY, "7", now=NOW)["count"] == 1
    kinds = [row.get("kind") for row in surface.audit(KEY)]
    assert "media_locks" in kinds
    warning = out["actions"][0].get("warning") or ""
    assert warning.startswith("⛔ Ada [7] — ")
    assert "strike 1" in warning


def test_locked_message_earns_no_economy(tmp_path):
    surface = _surface(tmp_path)
    surface.handle_message(KEY, "8", name="Bo", text="hello",
                           message_id="1", ts=NOW)
    assert surface.economy.is_verified(KEY, "8")
    _lock(surface, gifs=True)
    bad = surface.handle_message(
        KEY, "9", name="Cy", text="", message_id="2", ts=NOW,
        envelope={"media": {"kind": "animation"}})
    assert [a["kind"] for a in bad["actions"]] == ["delete"]
    assert not surface.economy.is_verified(KEY, "9")
    assert surface.strikes_for(KEY, "9", now=NOW)["count"] == 1


def test_admin_message_not_locked(tmp_path):
    surface = _surface(tmp_path)
    _lock(surface, stickers=True)
    out = surface.handle_message(
        KEY, "7", name="Ada", text="", message_id="1", ts=NOW,
        envelope={"media": {"kind": "sticker"}},
        sender_is_admin=True)
    assert out["actions"] == []
    assert surface.strikes_for(KEY, "7", now=NOW)["count"] == 0


def test_owner_ladder_never_lands(tmp_path):
    class _Registry:
        def role_of(self, platform, user_id, group_key=""):
            return "owner" if user_id == "1" else ""

        def owner_elsewhere(self, platform, user_id,
                            username, name):
            return ""

    surface = GroupSurface(
        GroupStore(tmp_path / "g.json").load(),
        owner_registry=_Registry())
    _lock(surface, stickers=True)
    out = surface.handle_message(
        KEY, "1", name="Boss", text="", message_id="1", ts=NOW,
        envelope={"media": {"kind": "sticker"}})
    assert out["actions"] == []
    assert surface.strikes_for(KEY, "1", now=NOW)["count"] == 0
    flagged = [row for row in surface.audit(KEY)
               if row.get("kind") == "media_locks"]
    assert flagged and flagged[0].get("role_exempt") == "owner"


def test_autonomy_off_suppresses_but_audits(tmp_path):
    surface = _surface(tmp_path)
    _lock(surface, stickers=True)
    s = surface.store.settings(KEY)
    d = dict(s.get("defense") or {})
    d["autonomy"] = False
    surface.update_settings(KEY, defense=d)
    out = surface.handle_message(
        KEY, "7", name="Ada", text="", message_id="1", ts=NOW,
        envelope={"media": {"kind": "sticker"}})
    assert out["actions"] == []
    assert out["suppressed"] == 1
    flagged = [row for row in surface.audit(KEY)
               if row.get("kind") == "media_locks"]
    assert flagged and flagged[0].get("suppressed") \
        == "autonomy_off"


def test_ladder_counts_media_lock_strikes(tmp_path):
    surface = _surface(tmp_path)
    _lock(surface, stickers=True)
    for i in (1, 2):
        out = surface.handle_message(
            KEY, "7", name="Ada", text="", message_id=f"{i}",
            ts=NOW + i,
            envelope={"media": {"kind": "sticker"}})
        assert [a["kind"] for a in out["actions"]] == ["delete"]
    third = surface.handle_message(
        KEY, "7", name="Ada", text="", message_id="3", ts=NOW + 3,
        envelope={"media": {"kind": "sticker"}})
    kinds = [a["kind"] for a in third["actions"]]
    # the lock's delete lands first; the shared spine
    # escalates the third strike to a mute
    assert kinds[0] == "delete"
    assert "mute" in kinds
    assert surface.strikes_for(KEY, "7", now=NOW + 3)["count"] == 3


def test_warn_action_posts_warn_not_delete(tmp_path):
    surface = _surface(tmp_path)
    _lock(surface, stickers=True, action="warn")
    out = surface.handle_message(
        KEY, "7", name="Ada", text="", message_id="1", ts=NOW,
        envelope={"media": {"kind": "sticker"}})
    assert [a["kind"] for a in out["actions"]] == ["warn"]
    assert surface.strikes_for(KEY, "7", now=NOW)["count"] == 1


# ── grammar + console + dashboard (runtime rig) ────────────────


class _Chat:
    def __init__(self, platform="telegram-bot", key=KEY):
        self.platform = platform
        self.chat_id = key.split(":")[1]
        self.title = "Builders"
        self.key = key
        self.kind = ChatKind.GROUP


class _Msg:
    def __init__(self, chat=None, **kw):
        self.chat = chat or _Chat()
        self.meta = kw.get("meta") or {}
        self.sender_id = kw.get("sender_id", "7")
        self.sender = kw.get("sender", "Ada")
        self.sender_username = kw.get("sender_username", "ada")
        self.text = kw.get("text", "")
        self.message_id = kw.get("message_id", "55")
        self.reply_to = ""


class _Adapter:
    def __init__(self, statuses=None, bare=False):
        self.calls = []
        self.statuses = statuses or {}
        self.bare = bare

    def admin_member(self, chat, user_id):
        return {"status": self.statuses.get(user_id, "member")}

    def admin_delete(self, chat, message_id):
        if self.bare:
            raise AttributeError("no admin_delete")
        self.calls.append(("delete", message_id))
        return {"ok": True}

    def admin_ban(self, chat, user_id):
        self.calls.append(("ban", user_id))
        return {"ok": True}

    def admin_unban(self, chat, user_id):
        self.calls.append(("unban", user_id))
        return {"ok": True}

    def admin_restrict(self, chat, user_id, *, minutes=0, **kw):
        self.calls.append(("restrict", user_id, minutes))
        return {"ok": True}


class _Runtime(RuntimeGroupMixin):
    def __init__(self, tmp_path, adapter, platform="telegram-bot",
                 operator=False):
        self.sent = []
        self.gateway = SimpleNamespace(
            adapters={platform: adapter})
        self._operator = operator
        self.context = SimpleNamespace(settings=SimpleNamespace(
            workspace_dir=str(tmp_path)))
        from nomorals.social.chat.group_surface import default_surface

        self._group_surface_obj = default_surface(str(tmp_path))

    def _send(self, chat, text, **kw):
        self.sent.append(str(text))

    def _is_operator(self, message):
        return self._operator


def _text(out) -> str:
    return getattr(out, "text", str(out))


def _admin_rt(tmp_path):
    adapter = _Adapter(statuses={"7": "administrator"})
    return _Runtime(tmp_path, adapter)


def test_gsettings_medialock_toggles_persist(tmp_path):
    rt = _admin_rt(tmp_path)
    admin = _Msg(sender_id="7")
    rt._control_gsettings("medialock stickers on", admin)
    rt._control_gsettings("medialock forwards on", admin)
    rt._control_gsettings("medialock action warn", admin)
    ml = rt._group_surface_obj.store.settings(KEY)["defense"][
        "media_locks"]
    assert ml["stickers"] is True
    assert ml["forwards"] is True
    assert ml["gifs"] is False
    assert ml["action"] == "warn"
    rt._control_gsettings("medialock all on", admin)
    ml = rt._group_surface_obj.store.settings(KEY)["defense"][
        "media_locks"]
    assert all(ml[k] is True for k in MEDIA_LOCK_KINDS)
    rt._control_gsettings("medialock all off", admin)
    ml = rt._group_surface_obj.store.settings(KEY)["defense"][
        "media_locks"]
    assert all(ml[k] is False for k in MEDIA_LOCK_KINDS)


def test_gsettings_medialock_rejects_unknown_kind(tmp_path):
    rt = _admin_rt(tmp_path)
    admin = _Msg(sender_id="7")
    out = rt._control_gsettings("medialock lasers on", admin)
    assert "stickers" in str(out) and "lasers" in str(out)
    ml = rt._group_surface_obj.store.settings(KEY)["defense"][
        "media_locks"]
    assert all(ml[k] is False for k in MEDIA_LOCK_KINDS)


def test_gsettings_panel_shows_media_locks(tmp_path):
    rt = _admin_rt(tmp_path)
    admin = _Msg(sender_id="7")
    rt._control_gsettings("medialock stickers on", admin)
    rt._control_gsettings("medialock forwards on", admin)
    out = _text(rt._control_gsettings("", admin))
    assert "media locks" in out
    assert "stickers" in out and "forwards" in out


def test_console_medialock_toggle(tmp_path):
    rt = _admin_rt(tmp_path)
    surface = rt._group_surface_obj
    out = rt._console_apply(
        surface, KEY, _Msg(sender_id="7"), "7", "defense",
        ["medialock", "gifs", "on"])
    assert "GIFs" in str(out)
    ml = surface.store.settings(KEY)["defense"]["media_locks"]
    assert ml["gifs"] is True
    section = rt._console_section(
        surface, KEY, _Msg(sender_id="7"), "g1", "defense")
    assert "media locks" in section.text
    assert "GIFs locked" in section.text
    buttons = " ".join(str(b) for b in (section.buttons or []))
    assert "GIFs: ON" in buttons


def test_groups_dashboard_media_locks_line(tmp_path):
    rt = _admin_rt(tmp_path)
    surface = rt._group_surface_obj
    surface.set_designated(KEY, True, by="owner")
    rt._control_gsettings("medialock stickers on",
                          _Msg(sender_id="7"))
    lines = rt._groups_group_lines(surface, KEY, "g1")
    assert any(line.startswith("media locks ")
               and "stickers" in line for line in lines)


def test_runtime_executes_media_lock_delete(tmp_path):
    rt = _admin_rt(tmp_path)
    adapter = rt.gateway.adapters["telegram-bot"]
    rt._control_gsettings("medialock stickers on",
                          _Msg(sender_id="7"))
    # the sticker arrives from a MEMBER (sender 8): the admin
    # who set the lock is exempt from it
    msg = _Msg(text="", message_id="99", sender_id="8")
    msg.meta["envelope"] = {"media": {"kind": "sticker"}}
    assert rt._group_surface_turn(msg) is True
    assert ("delete", "99") in adapter.calls
    warnings = [s for s in rt.sent if s.startswith("⛔")]
    assert len(warnings) == 1
    assert "Ada" in warnings[0] and "strike 1" in warnings[0]

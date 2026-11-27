"""Log Channels (pull-only) + Locks — the Rose depth build.

Log Channels: /setlog designates a channel; NOTHING is ever
posted to it.  An admin asking in that channel with /glog pulls
the held audit log, filtered by the group's enabled categories
(/log and /nolog flip them).  /glog in the group is unchanged.
Locks: /lock /unlock /locks over the media-locks block — the new
kinds (links, games, polls, rtl, emails, phones) plus per-kind
actions (delete/warn/delete_warn/mute/kick/ban) and silent
variants that run as their loud equivalent until Silent Actions
lands.  Every lock hit rides stage 3c and the shared strike
spine; every designation/category/lock change is audited.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from nomorals.agents.partner.runtime_group import RuntimeGroupMixin
from nomorals.security.payload_inspection import normalize_envelope
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
CHANNEL = "telegram-bot:-100999888777"
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


# ── sensor units: the new kinds ─────────────────────────────────


def test_new_lock_kinds_exist_with_labels():
    for kind in ("links", "games", "polls", "rtl", "emails",
                 "phones"):
        assert kind in MEDIA_LOCK_KINDS
        assert MEDIA_LOCK_LABELS[kind]
    # No bot lock: the adapters drop bot senders at intake, so
    # the surface never sees a bot message for one to fire on.
    assert "bot" not in MEDIA_LOCK_KINDS


def test_defaults_lock_nothing_new():
    cfg = DefenseConfig.from_settings({})
    assert check_media_locks(
        _env(text="see https://example.com"), cfg) is None
    assert check_media_locks(_env(game=True), cfg) is None
    assert check_media_locks(_env(poll=True), cfg) is None
    assert check_media_locks(_env(text="שלום"), cfg) is None
    assert check_media_locks(
        _env(text="mail me at a@b.com"), cfg) is None
    assert check_media_locks(
        _env(text="call +234 803 123 4567"), cfg) is None


def test_links_lock_fires_on_entity_and_bare_url():
    cfg = _cfg(links=True)
    hit = check_media_locks(
        _env(entities=[{"type": "url", "offset": 4, "length": 11}],
             text="see example.com here"), cfg)
    assert hit is not None and hit.signal["kind"] == "links"
    hit = check_media_locks(
        _env(text="see https://example.com for details"), cfg)
    assert hit is not None and hit.signal["kind"] == "links"
    assert check_media_locks(_env(text="no links here"), cfg) \
        is None


def test_games_and_polls_locks_fire_on_envelope_flags():
    assert check_media_locks(_env(game=True), _cfg(games=True)) \
        is not None
    assert check_media_locks(_env(game=True), _cfg(polls=True)) \
        is None
    hit = check_media_locks(_env(poll=True), _cfg(polls=True))
    assert hit is not None and hit.signal["kind"] == "polls"
    assert check_media_locks(_env(poll=True), _cfg(games=True)) \
        is None


def test_rtl_lock_fires_on_hebrew_and_arabic_only():
    cfg = _cfg(rtl=True)
    assert check_media_locks(_env(text="שלום חברים"), cfg) \
        is not None
    assert check_media_locks(_env(text="مرحبا"), cfg) is not None
    assert check_media_locks(_env(text="hello friends"), cfg) \
        is None


def test_emails_lock_fires():
    cfg = _cfg(emails=True)
    hit = check_media_locks(
        _env(text="reach me at ada@example.com"), cfg)
    assert hit is not None and hit.signal["kind"] == "emails"
    assert check_media_locks(_env(text="no address here"), cfg) \
        is None


def test_phones_lock_needs_nine_digits():
    cfg = _cfg(phones=True)
    hit = check_media_locks(
        _env(text="call +234 803 123 4567"), cfg)
    assert hit is not None and hit.signal["kind"] == "phones"
    # A date, a year and a short number never trip it.
    assert check_media_locks(_env(text="on 2026-11-27"), cfg) \
        is None
    assert check_media_locks(_env(text="room 2026"), cfg) \
        is None
    assert check_media_locks(_env(text="call 1234"), cfg) is None


def test_per_kind_action_overrides_the_block():
    cfg = _cfg(stickers=True, action="delete_warn",
               actions={"stickers": "mute"})
    hit = check_media_locks(
        _env(media={"kind": "sticker"}), cfg)
    assert hit is not None and hit.action == "mute"
    # A kind without an override rides the block action.
    cfg2 = _cfg(stickers=True, photos=True, action="warn",
                actions={"stickers": "ban"})
    assert check_media_locks(
        _env(media={"kind": "sticker"}), cfg2).action == "ban"
    assert check_media_locks(
        _env(media={"kind": "photo"}), cfg2).action == "warn"


def test_silent_action_maps_to_loud_and_signals_it():
    cfg = _cfg(stickers=True, actions={"stickers": "sban"})
    hit = check_media_locks(
        _env(media={"kind": "sticker"}), cfg)
    assert hit is not None
    assert hit.action == "ban"
    assert hit.signal["silent_action"] == "sban"
    cfg2 = _cfg(links=True, action="smute")
    hit2 = check_media_locks(
        _env(text="see https://example.com"), cfg2)
    assert hit2.action == "mute"
    assert hit2.signal["silent_action"] == "smute"


def test_unknown_action_degrades_to_delete_warn():
    cfg = _cfg(stickers=True, actions={"stickers": "explode"})
    hit = check_media_locks(
        _env(media={"kind": "sticker"}), cfg)
    assert hit is not None and hit.action == "delete_warn"
    cfg2 = _cfg(links=True, action="explode")
    assert check_media_locks(
        _env(text="see https://example.com"), cfg2).action \
        == "delete_warn"


def test_describe_shows_per_kind_actions_and_silent_note():
    assert describe_media_locks({}) == (
        "none locked — everything open")
    line = describe_media_locks(
        {"stickers": True, "links": True, "action": "delete_warn",
         "actions": {"stickers": "mute"}})
    assert "stickers (mute)" in line
    assert "links locked (delete_warn)" in line
    silent = describe_media_locks(
        {"stickers": True, "actions": {"stickers": "sban"}})
    assert "stickers (ban)" in silent
    assert "run as their loud equivalent" in silent


def test_lock_action_vocabulary():
    # Imported here, not at module top: the name does not exist
    # at the base commit, and a top-level import would make the
    # whole file a collection error on pristine instead of
    # per-test failures.
    from nomorals.social.chat.group_defense import (
        LOCK_ALL_ACTIONS,
    )

    for action in ("delete", "warn", "delete_warn", "mute",
                   "kick", "ban", "sban", "smute", "skick",
                   "swarn"):
        assert action in LOCK_ALL_ACTIONS


# ── log-channel categories ──────────────────────────────────────


def test_log_category_of_spot_checks():
    # Imported here, not at module top: the name does not exist
    # at the base commit (see test_lock_action_vocabulary).
    from nomorals.social.chat.group_surface import (
        log_category_of,
    )

    assert log_category_of("media_locks") == "defense"
    assert log_category_of("flood") == "defense"
    assert log_category_of("admin_action") == "moderation"
    assert log_category_of("moderation_executed") == "moderation"
    assert log_category_of("join") == "joins"
    assert log_category_of("member_joined") == "joins"
    assert log_category_of("treasury_grant") == "economy"
    assert log_category_of("report") == "reports"
    assert log_category_of("log_channel_set") == "settings"
    assert log_category_of("lock_set") == "settings"
    # Unfiled kinds are always shown — nothing hides by accident.
    assert log_category_of("something_new_tomorrow") is None
    assert log_category_of("") is None


def test_default_settings_has_log_channel_block():
    # Imported here, not at module top: the name does not exist
    # at the base commit (see test_lock_action_vocabulary).
    from nomorals.social.chat.group_surface import (
        LOG_CHANNEL_CATEGORIES,
    )

    block = _default_settings()["log_channel"]
    assert block["channel"] == ""
    assert block["categories"] == {
        c: True for c in LOG_CHANNEL_CATEGORIES}
    assert LOG_CHANNEL_CATEGORIES == (
        "moderation", "defense", "joins", "economy", "reports",
        "settings")


# ── surface stage 3c ────────────────────────────────────────────


def test_links_lock_deletes_and_strikes_once(tmp_path):
    surface = _surface(tmp_path)
    _lock(surface, links=True)
    hit = surface.handle_message(
        KEY, "u1", text="see https://example.com now",
        envelope={"entities": [{"type": "url", "offset": 4,
                                "length": 19}]},
        message_id="m1")
    assert any(a["kind"] == "delete"
               and a["rule"] == "media_locks"
               for a in hit["actions"])
    member = surface.member(KEY, "u1")
    assert len(member.get("strikes") or []) == 1
    audit = surface.audit(KEY, limit=5)
    assert any(e["kind"] == "media_locks"
               and e["action"] == "delete_warn" for e in audit)


def test_lock_mute_action_appends_mute_with_minutes(tmp_path):
    surface = _surface(tmp_path)
    _lock(surface, stickers=True, actions={"stickers": "mute"})
    hit = surface.handle_message(
        KEY, "u1", envelope={"media": {"kind": "sticker"}},
        message_id="m2")
    mute = [a for a in hit["actions"] if a["kind"] == "mute"]
    assert len(mute) == 1
    assert mute[0]["minutes"] == 60
    member = surface.member(KEY, "u1")
    assert (member.get("strikes") or [{}])[0]["action"] == "mute"


def test_lock_ban_action_appends_ban(tmp_path):
    surface = _surface(tmp_path)
    _lock(surface, links=True, actions={"links": "ban"})
    hit = surface.handle_message(
        KEY, "u1", text="see https://example.com",
        message_id="m3")
    assert any(a["kind"] == "ban" for a in hit["actions"])


def test_games_lock_fires_through_the_surface(tmp_path):
    surface = _surface(tmp_path)
    _lock(surface, games=True)
    hit = surface.handle_message(
        KEY, "u1", envelope={"game": True}, message_id="m4")
    assert any(a["rule"] == "media_locks"
               for a in hit["actions"])


def test_admin_sender_is_exempt_at_the_sensor(tmp_path):
    surface = _surface(tmp_path)
    _lock(surface, stickers=True)
    hit = surface.handle_message(
        KEY, "admin1", envelope={"media": {"kind": "sticker"}},
        message_id="m5", sender_is_admin=True)
    assert hit["actions"] == []
    member = surface.member(KEY, "admin1")
    assert not (member.get("strikes") or [])
    audit = surface.audit(KEY, limit=5)
    assert not any(e["kind"] == "media_locks" for e in audit)


# ── handlers ────────────────────────────────────────────────────


class _Adapter:
    def __init__(self, *, admin: bool = True):
        self.calls: list = []
        self._admin = admin

    def admin_member(self, chat, user_id):
        return {"status": "administrator" if self._admin
                else "member"}

    def admin_delete(self, chat, message_id):
        self.calls.append(("delete", message_id))
        return {"ok": True}

    def admin_restrict(self, chat, user_id, *, minutes=60):
        self.calls.append(("restrict", user_id, minutes))
        return {"ok": True}

    def admin_ban(self, chat, user_id):
        self.calls.append(("ban", user_id))
        return {"ok": True}

    def admin_unban(self, chat, user_id):
        self.calls.append(("unban", user_id))
        return {"ok": True}


def _msg(*, sender_id="admin1", kind="group", key=KEY):
    return SimpleNamespace(
        chat=SimpleNamespace(key=key, platform="telegram-bot",
                             kind=kind, title="Builders"),
        sender_id=sender_id, sender="Admin",
        sender_username="boss", text="", meta={}, reply_to="",
        message_id="900")


def _rig(tmp_path: Path, **adapter_kw):
    surface = GroupSurface(GroupStore(tmp_path / "g.json").load())
    adapter = _Adapter(**adapter_kw)
    mixin = RuntimeGroupMixin()
    mixin._group_surface = lambda: surface  # type: ignore[method-assign]
    mixin._group_adapter = lambda platform: adapter  # type: ignore[method-assign]
    return mixin, surface, adapter


def test_setlog_designates_and_audits(tmp_path):
    mixin, surface, _ = _rig(tmp_path)
    out = mixin._control_setlog("-100999888777", _msg())
    assert CHANNEL in out
    assert "never" in out.lower() or "posted" in out
    block = surface.store.settings(KEY)["log_channel"]
    assert block["channel"] == CHANNEL
    audit = surface.audit(KEY, limit=5)
    assert any(e["kind"] == "log_channel_set"
               and e["new"] == CHANNEL for e in audit)
    # Bare /setlog reports the designation.
    assert CHANNEL in mixin._control_setlog("", _msg())
    # A @name alone does not resolve to a channel.
    assert "id" in mixin._control_setlog("@signals", _msg())
    # The group itself is not a channel.
    assert "this group" in mixin._control_setlog(
        "-1001234567", _msg())
    # /unsetlog removes it and audits.
    out = mixin._control_unsetlog("", _msg())
    assert "removed" in out
    assert surface.store.settings(KEY)["log_channel"][
        "channel"] == ""
    assert any(e["kind"] == "log_channel_unset"
               for e in surface.audit(KEY, limit=5))


def test_setlog_accepts_platform_id(tmp_path):
    mixin, surface, _ = _rig(tmp_path)
    mixin._control_setlog("telegram-bot:-100999888777", _msg())
    assert surface.store.settings(KEY)["log_channel"][
        "channel"] == CHANNEL


def test_log_nolog_flip_categories_and_audit(tmp_path):
    mixin, surface, _ = _rig(tmp_path)
    out = mixin._control_nolog("economy", _msg())
    assert "hidden" in out
    assert surface.store.settings(KEY)["log_channel"][
        "categories"]["economy"] is False
    audit = surface.audit(KEY, limit=5)
    assert any(e["kind"] == "log_category"
               and e["rule"] == "economy"
               and e["action"] == "nolog" for e in audit)
    out = mixin._control_log("economy", _msg())
    assert "shown" in out
    # Unknown categories are named, not silently applied.
    assert "moderation" in mixin._control_log("nope", _msg())
    # Bare /log reports the states.
    assert "economy: on" in mixin._control_log("", _msg())


def test_lock_unlock_roundtrip(tmp_path):
    mixin, surface, _ = _rig(tmp_path)
    out = mixin._control_lock("links delete", _msg())
    assert "links locked (delete)" in out
    block = surface.store.settings(KEY)["defense"]["media_locks"]
    assert block["links"] is True
    assert any(e["kind"] == "lock_set" and e["rule"] == "links"
               for e in surface.audit(KEY, limit=5))
    # A silent action is accepted and says it runs loud.
    out = mixin._control_lock("stickers sban", _msg())
    assert "stickers locked (ban)" in out
    assert "Silent" in out
    block = surface.store.settings(KEY)["defense"]["media_locks"]
    assert block["actions"]["stickers"] == "sban"
    # /locks lists the state.
    out = mixin._control_locks("", _msg())
    assert "links" in out and "stickers" in out
    # /unlock opens it and clears the override.
    out = mixin._control_unlock("stickers", _msg())
    assert "open" in out
    block = surface.store.settings(KEY)["defense"]["media_locks"]
    assert block["stickers"] is False
    assert "stickers" not in block["actions"]
    # Unknown kinds and actions are refused plainly.
    assert "don't lock" in mixin._control_lock("nope", _msg())
    assert "isn't a lock action" in mixin._control_lock(
        "links explode", _msg())


def test_lock_plain_form_keeps_the_block_action(tmp_path):
    mixin, surface, _ = _rig(tmp_path)
    mixin._control_lock("links", _msg())
    block = surface.store.settings(KEY)["defense"]["media_locks"]
    assert block["links"] is True
    assert block.get("actions") == {}


def test_non_admin_is_refused(tmp_path):
    mixin, surface, _ = _rig(tmp_path, admin=False)
    assert "admin" in mixin._control_setlog("-1001", _msg())
    assert "admin" in mixin._control_lock("links", _msg())
    assert "admin" in mixin._control_locks("", _msg())
    assert surface.store.settings(KEY)["log_channel"][
        "channel"] == ""


def test_medialock_grammar_per_kind_action(tmp_path):
    mixin, surface, _ = _rig(tmp_path)
    out = mixin._apply_media_lock_grammar(
        surface, KEY, ["links", "mute"])
    assert "links locked (mute)" in out
    block = surface.store.settings(KEY)["defense"]["media_locks"]
    assert block["links"] is True
    assert block["actions"]["links"] == "mute"
    out = mixin._apply_media_lock_grammar(
        surface, KEY, ["action", "kick"])
    assert "kick" in out
    out = mixin._apply_media_lock_grammar(
        surface, KEY, ["all", "off"])
    assert "all media locks off" in out
    block = surface.store.settings(KEY)["defense"]["media_locks"]
    assert block["actions"] == {}
    out = mixin._apply_media_lock_grammar(
        surface, KEY, ["status"])
    assert "media locks for" in out
    assert "none locked" in out


# ── the pull view ───────────────────────────────────────────────


def _seed_log(surface):
    surface._audit(KEY, "admin_action", action="mute",
                   user_id="u1")
    surface._audit(KEY, "treasury_grant", user_id="u1")
    surface._audit(KEY, "media_locks", action="delete_warn",
                   user_id="u1")
    surface._audit(KEY, "join", user_id="u2")
    surface._audit(KEY, "something_new_tomorrow", user_id="u1")


def _channel_msg():
    return _msg(kind="channel", key=CHANNEL)


def test_glog_pull_filters_categories(tmp_path):
    mixin, surface, _ = _rig(tmp_path)
    mixin._control_setlog("-100999888777", _msg())
    mixin._control_nolog("economy", _msg())
    _seed_log(surface)
    out = mixin._control_glog("10", _channel_msg())
    assert "admin_action" in out
    assert "media_locks" in out
    assert "join" in out
    # The hidden category stays out of the pull…
    assert "treasury_grant" not in out
    # …but an unfiled kind is always shown.
    assert "something_new_tomorrow" in out
    assert "pulled from" in out


def test_glog_pull_refuses_undesignated_channel(tmp_path):
    mixin, surface, _ = _rig(tmp_path)
    _seed_log(surface)
    out = mixin._control_glog("10", _channel_msg())
    assert "isn't any group's log channel" in out


def test_glog_pull_refuses_non_admin(tmp_path):
    # Imported here, not at module top: the name does not exist
    # at the base commit (see test_lock_action_vocabulary).
    from nomorals.social.chat.group_surface import (
        LOG_CHANNEL_CATEGORIES,
    )

    mixin, surface, _ = _rig(tmp_path, admin=False)
    # Designate directly: /setlog itself needs admin, and this
    # rig's adapter refuses it — the pull refusal is what's
    # under test.
    surface.update_settings(KEY, log_channel={
        "channel": CHANNEL,
        "categories": {c: True
                       for c in LOG_CHANNEL_CATEGORIES}})
    out = mixin._control_glog("10", _channel_msg())
    assert "admin" in out


def test_glog_in_group_is_unchanged(tmp_path):
    mixin, surface, _ = _rig(tmp_path)
    mixin._control_setlog("-100999888777", _msg())
    mixin._control_nolog("economy", _msg())
    _seed_log(surface)
    out = mixin._control_glog("10", _msg())
    # In the group, /glog shows the whole held log.
    assert "treasury_grant" in out
    assert "admin_action" in out
    assert "pulled from" not in out


# ── envelope flags ──────────────────────────────────────────────


def test_envelope_game_and_poll_flags():
    env = normalize_envelope({"game": True})
    assert env.game is True and env.poll is False
    env = normalize_envelope({"poll": {"question": "x"}})
    assert env.poll is True and env.game is False
    env = normalize_envelope({})
    assert env.game is False and env.poll is False
    assert env.to_dict()["game"] is False


def test_bot_envelope_stamps_game_and_poll():
    from nomorals.social.chat.telegram import (
        _bot_message_envelope,
    )

    env = _bot_message_envelope(
        {"game": {"title": "Chess"}, "text": "play"}, "play")
    assert env["game"] is True and env["poll"] is False
    env = _bot_message_envelope(
        {"poll": {"question": "lunch?"}}, "")
    assert env["poll"] is True and env["game"] is False
    env = _bot_message_envelope({"dice": {"value": 5}}, "")
    assert env["game"] is True
    env = _bot_message_envelope({"text": "hi"}, "hi")
    assert env["game"] is False and env["poll"] is False


def test_whatsapp_envelope_stamps_poll():
    from nomorals.social.chat.whatsapp import _wa_envelope

    assert _wa_envelope({"pollMessage": {"a": 1}}, "")["poll"] \
        is True
    assert _wa_envelope({}, "")["poll"] is False

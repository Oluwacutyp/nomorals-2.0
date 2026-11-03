"""Escalation configuration + the notice layer + the DM admin
console (DESIGN §30 + §31).

§30: the warn ladder is per-group configuration — a warn limit
(the final-action count), designed interval steps (mute /
restrict / fine / kick), the final action (kick|ban), and the
warn expiry window — read by EVERY strike consumer (defense
sensors, manual warns, acceptance enforcement).  Defaults ARE
the shipped ladder (mute 3 · kick 5 · ban 7, 7-day window).

§31: every automated action posts ONE notice — mention +
[numeric id], the trigger in plain words, the action with its
ladder state, expiry when time-bound — with an Undo that
reverses exactly that action, editing the notice in place
where the platform supports it.

§30 console: an admin configures a group from Devon's DM —
only groups they actually admin are listed; changes land in
the group's settings and in its /glog naming the admin;
owner-only switches keep the owner gate, judged in the target
group.

Rigs follow the house shapes: GroupSurface over a tmp_path
GroupStore; RuntimeGroupMixin stubs like the defense/economy
suites.
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

from nomorals.social.chat.group_defense import EscalationConfig
from nomorals.social.chat.group_surface import GroupStore, GroupSurface

KEY = "telegram-bot:-1001234567"
WA_KEY = "whatsapp:120363012345@g.us"
NOW = 1_700_000_000.0


def _surface(tmp_path: Path) -> GroupSurface:
    return GroupSurface(GroupStore(tmp_path / "g.json").load())


def _seed_filter(surface: GroupSurface, key: str = KEY) -> None:
    surface.update_settings(key, moderation={
        "flood_max": 6, "flood_window_s": 10.0,
        "repeat_threshold": 3, "link_allowlist": [],
        "word_filters": ["badword"], "filter_action": "delete"})


def _offense(surface: GroupSurface, user: str, i: int, ts: float,
             key: str = KEY):
    return surface.handle_message(
        key, user, name="Ada", text=f"badword post {i}",
        message_id=f"o{i}", ts=ts)


# ── §30: the ladder — defaults, configuration, expiry ──────────


def test_default_ladder_is_the_shipped_ladder(tmp_path):
    surface = _surface(tmp_path)
    esc = surface.escalation_config(KEY)
    assert esc.warn_limit == 7
    assert esc.final_action == "ban"
    assert esc.window_days == 7
    assert [(s["at"], s["action"]) for s in esc.intervals] == [
        (3, "mute"), (5, "kick")]
    assert esc.interval_at(3)["minutes"] == 60
    _seed_filter(surface)
    kinds_at = {}
    for i in range(1, 8):
        out = _offense(surface, "7", i, NOW + i * 30)
        kinds_at[i] = [a["kind"] for a in out["actions"]]
    assert kinds_at[1] == ["delete"]
    assert "mute" in kinds_at[3]
    assert "kick" in kinds_at[5]
    assert "ban" in kinds_at[7]


def test_legacy_settings_derive_the_same_ladder():
    """Raw settings dicts from before the escalation block
    existed (defense fields + warn_threshold) resolve to the
    identical ladder — no group's behavior changes."""
    esc = EscalationConfig.from_settings({
        "warn_threshold": 3,
        "defense": {"escalation": {"window_days": 7,
                                    "mute_minutes": 60,
                                    "kick_at": 5, "ban_at": 7}}})
    assert esc.warn_limit == 7 and esc.final_action == "ban"
    assert [(s["at"], s["action"]) for s in esc.intervals] == [
        (3, "mute"), (5, "kick")]


def test_configured_ladder_changes_outcomes(tmp_path):
    surface = _surface(tmp_path)
    _seed_filter(surface)
    surface.update_settings(KEY, escalation={
        "warn_limit": 3, "final_action": "kick", "window_days": 7,
        "intervals": [{"at": 2, "action": "mute", "minutes": 10}]})
    out1 = _offense(surface, "7", 1, NOW)
    assert [a["kind"] for a in out1["actions"]] == ["delete"]
    out2 = _offense(surface, "7", 2, NOW + 30)
    mute = next(a for a in out2["actions"] if a["kind"] == "mute")
    assert mute["minutes"] == 10
    out3 = _offense(surface, "7", 3, NOW + 60)
    assert "kick" in [a["kind"] for a in out3["actions"]]
    assert "ban" not in [a["kind"] for a in out3["actions"]]


def test_configured_fine_and_restrict_steps(tmp_path):
    surface = _surface(tmp_path)
    _seed_filter(surface)
    surface.update_settings(KEY, escalation={
        "warn_limit": 4, "final_action": "ban", "window_days": 7,
        "intervals": [{"at": 1, "action": "restrict"},
                      {"at": 2, "action": "fine", "amount": 50}]})
    # fund the member so the fine can actually collect
    surface.economy.verify_member(KEY, "7")
    surface.economy.award(KEY, "7", 200, "seed", now=NOW)
    out1 = _offense(surface, "7", 1, NOW)
    assert "acceptance_restrict" in [a["kind"]
                                     for a in out1["actions"]]
    out2 = _offense(surface, "7", 2, NOW + 30)
    warning = next(a["warning"] for a in out2["actions"]
                   if a.get("warning"))
    assert "fined 50 sparks" in warning
    assert surface.economy.balance(KEY, "7") == 150


def test_warn_expiry_prunes_with_configured_window(tmp_path):
    surface = _surface(tmp_path)
    _seed_filter(surface)
    surface.update_settings(KEY, escalation={
        "warn_limit": 7, "final_action": "ban", "window_days": 1,
        "intervals": [{"at": 3, "action": "mute", "minutes": 60},
                      {"at": 5, "action": "kick"}]})
    _offense(surface, "7", 1, NOW)
    _offense(surface, "7", 2, NOW + 30)
    later = NOW + 2 * 86400
    out = _offense(surface, "7", 3, later)
    assert surface.strikes_for(KEY, "7", now=later)["count"] == 1
    assert "mute" not in [a["kind"] for a in out["actions"]]


def test_warning_text_carries_position_next_and_expiry(tmp_path):
    surface = _surface(tmp_path)
    _seed_filter(surface)
    out = _offense(surface, "7", 1, NOW)
    warning = next(a["warning"] for a in out["actions"]
                   if a.get("warning"))
    assert warning.startswith("⛔ Ada [7] — ")
    assert "that word is filtered here" in warning
    assert "strike 1 of 7" in warning
    assert "next: mute for 60 min at 3" in warning
    assert "oldest warn clears" in warning


def test_strikes_read_model_renders_the_groups_ladder(tmp_path):
    surface = _surface(tmp_path)
    _seed_filter(surface)
    surface.update_settings(KEY, escalation={
        "warn_limit": 4, "final_action": "kick", "window_days": 7,
        "intervals": [{"at": 2, "action": "mute", "minutes": 30}]})
    _offense(surface, "7", 1, NOW)
    info = surface.strikes_for(KEY, "7", now=NOW)
    assert info["limit"] == 4 and info["final_action"] == "kick"
    assert info["next_at"] == 2
    assert info["next_label"] == "mute for 30 min"
    assert info["oldest_ts"] == NOW


# ── §31: notices — shape, quiet mode, undo ──────────────────────


def test_notice_record_carries_the_reversal_state(tmp_path):
    surface = _surface(tmp_path)
    _seed_filter(surface)
    out = None
    for i in range(1, 4):
        out = _offense(surface, "7", i, NOW + i * 30)
    top = next(a for a in out["actions"] if a.get("notice_id"))
    record = surface.get_notice(KEY, top["notice_id"])
    assert record is not None
    assert record["user_id"] == "7"
    assert record["action"] == "mute"
    assert record["count"] == 3 and record["limit"] == 7
    assert record["strike_rule"] == "word_filter"
    assert record["strike_ts"] > 0
    assert "Ada [7]" in record["text"]
    assert "strike 3 of 7" in record["text"]
    assert record["undone"] is False


def test_quiet_mode_holds_notices(tmp_path):
    """Autonomy OFF (the defense master switch — the one quiet
    mode) holds every notice: no actions, no records, and the
    audits carry the suppression marker."""
    surface = _surface(tmp_path)
    _seed_filter(surface)
    surface.update_settings(KEY, defense={
        "autonomy": False,
        "anticrash": {"enabled": True, "action": "delete"},
        "antichannel": {"enabled": True, "action": "delete_warn",
                        "exempt_linked": True},
        "antitag": {"enabled": True, "max_mentions": 10,
                    "max_roster_pct": 20, "action": "delete_warn"},
        "antidelete": {"enabled": True, "mode": "log",
                       "callout_to": "admins"},
        "antiedit": {"enabled": True, "mode": "log",
                     "callout_to": "admins"},
        "escalation": {"window_days": 7, "mute_minutes": 60,
                       "kick_at": 5, "ban_at": 7}})
    out = _offense(surface, "7", 1, NOW)
    assert out["actions"] == []
    assert out["suppressed"] >= 0
    entry = surface.store.group(KEY)
    assert not entry.get("notices")
    assert surface.strikes_for(KEY, "7", now=NOW)["count"] == 0


def test_remove_strike_removes_exactly_one(tmp_path):
    surface = _surface(tmp_path)
    _seed_filter(surface)
    _offense(surface, "7", 1, NOW)
    _offense(surface, "7", 2, NOW + 30)
    member = surface.store.group(KEY)["members"]["7"]
    newest = member["strikes"][-1]
    removed = surface.remove_strike(KEY, "7", rule="word_filter",
                                    ts=float(newest["ts"]))
    assert removed is not None
    assert surface.strikes_for(KEY, "7", now=NOW + 30)["count"] == 1
    # the other strike is untouched
    assert surface.remove_strike(KEY, "7", rule="word_filter",
                                 ts=float(newest["ts"])) is None


# ── runtime rigs ────────────────────────────────────────────────


class _Chat:
    def __init__(self, key=KEY, platform="telegram-bot",
                 kind="group", title="Builders"):
        self.key = key
        self.platform = platform
        self.kind = kind
        self.title = title


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
    """admin_member answers from a per-id status map; admin_*
    calls are recorded; edit_message records and succeeds."""

    def __init__(self, statuses=None):
        self.calls = []
        self.edits = []
        self.statuses = statuses or {}

    def admin_member(self, chat, user_id):
        return {"status": self.statuses.get(str(user_id),
                                            "member")}

    def admin_delete(self, chat, message_id):
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

    def admin_unrestrict(self, chat, user_id):
        self.calls.append(("unrestrict", user_id))
        return {"ok": True}

    def edit_message(self, chat, message_id, text, *, buttons=None,
                     **kw):
        self.edits.append((message_id, text, buttons))
        return {"ok": True}


from nomorals.agents.partner.runtime_group import RuntimeGroupMixin
from nomorals.social.chat.group_surface import default_surface


class _Runtime(RuntimeGroupMixin):
    """The mixin over a fake gateway whose send returns a real
    message id (so notice message-id capture + edits pin)."""

    def __init__(self, tmp_path, adapter,
                 platform="telegram-bot", operator=False):
        self.sent = []          # (text, buttons)
        self.gateway = SimpleNamespace(
            adapters={platform: adapter})
        self._operator = operator
        self.context = SimpleNamespace(settings=SimpleNamespace(
            workspace_dir=str(tmp_path)))
        self._group_surface_obj = default_surface(str(tmp_path))

    def _send(self, chat, text, **kw):
        self.sent.append((str(text), kw.get("buttons")))
        return SimpleNamespace(ok=True, platform=chat.platform,
                               message_id="4242", error="")

    def _is_operator(self, message):
        return self._operator


def _runtime(tmp_path, adapter, **kw):
    rt = _Runtime(tmp_path, adapter, **kw)
    return rt, rt._group_surface_obj


def test_notice_posts_with_undo_button_on_telegram(tmp_path):
    adapter = _Adapter()
    mixin, surface = _runtime(tmp_path, adapter)
    _seed_filter(surface)
    for i in range(1, 4):
        mixin._group_surface_turn(_Msg(
            text=f"badword post {i}", message_id=str(90 + i),
            sender_id="7"))
    assert ("restrict", "7", 60) in adapter.calls
    # exactly one send carries buttons per offense, and the mute
    # offense's send is the notice: text + its Undo button,
    # authored from the same record
    noticed = [(t, b) for t, b in mixin.sent if b]
    assert noticed
    text, buttons = noticed[-1]
    assert "Ada [7]" in text and "strike 3 of 7" in text
    assert buttons[0][0][0] == "↩️ Undo"
    assert buttons[0][0][1].startswith("undonotice ")
    nid = buttons[0][0][1].split()[1]
    assert surface.get_notice(KEY, nid)["action"] == "mute"


def test_undo_reverses_mute_and_strike_and_edits(tmp_path):
    adapter = _Adapter(statuses={"99": "administrator"})
    mixin, surface = _runtime(tmp_path, adapter)
    _seed_filter(surface)
    for i in range(1, 4):
        mixin._group_surface_turn(_Msg(
            text=f"badword post {i}", message_id=str(90 + i),
            sender_id="7"))
    assert ("restrict", "7", 60) in adapter.calls
    assert surface.strikes_for(KEY, "7")["count"] == 3
    # the notice for the mute: find it in the registry
    notices = surface.store.group(KEY).get("notices") or {}
    mute_notice = next(n for n in notices.values()
                       if n["action"] == "mute")
    assert mute_notice["message_id"] == "4242"
    out = mixin._control_undonotice(
        mute_notice["id"],
        _Msg(sender_id="99", sender="Boss",
             sender_username="boss"))
    assert out == ""  # edited in place: no stacked message
    assert ("unrestrict", "7") in adapter.calls
    assert surface.strikes_for(KEY, "7")["count"] == 2
    assert len(adapter.edits) == 1
    edited_id, edited_text, edited_buttons = adapter.edits[0]
    assert edited_id == "4242"
    assert "Undone" in edited_text
    assert "the mute is lifted" in edited_text
    record = surface.get_notice(KEY, mute_notice["id"])
    assert record["undone"] is True
    assert record["undone_by"] == "99"
    audits = [a for a in surface.audit(KEY, limit=50)
              if a["kind"] == "notice_undo"]
    assert audits and audits[-1]["by"] == "99"
    # a second undo is refused plainly
    out2 = mixin._control_undonotice(
        mute_notice["id"],
        _Msg(sender_id="99", sender="Boss"))
    assert "already undone" in out2


def test_notice_text_fallback_on_buttonless_platform(tmp_path):
    from nomorals.social.chat.group_notices import (
        notice_keyboard, notice_text, platform_has_buttons)

    record = {"id": "n1-0", "text": "⛔ Ada [7] — warned.",
              "undone": False, "extra": {}}
    assert platform_has_buttons("telegram-bot")
    assert not platform_has_buttons("whatsapp")
    assert notice_keyboard(record) == [[("↩️ Undo",
                                         "undonotice n1-0")]]
    wa_text = notice_text(record, "whatsapp")
    assert wa_text.endswith("↩️ undo: /undonotice n1-0")
    tg_text = notice_text(record, "telegram-bot")
    assert "undonotice" not in tg_text


def test_undo_requires_admin(tmp_path):
    adapter = _Adapter(statuses={"99": "administrator"})
    mixin, surface = _runtime(tmp_path, adapter)
    nid = surface.record_notice(
        KEY, user_id="7", mention="Ada", rule="words",
        action="warn", text="⛔ Ada [7] — warned.")
    out = mixin._control_undonotice(nid, _Msg(sender_id="7"))
    assert "admins" in out
    assert surface.get_notice(KEY, nid)["undone"] is False


# ── /gsettings grammar for the new blocks ───────────────────────


def test_gsettings_escalation_and_estate_grammar(tmp_path):
    adapter = _Adapter(statuses={"99": "administrator"})
    mixin, surface = _runtime(tmp_path, adapter)
    msg = _Msg(sender_id="99", sender="Boss")
    out = mixin._control_gsettings("escalation limit 4", msg)
    assert "warn limit 4" in out
    assert surface.escalation_config(KEY).warn_limit == 4
    out = mixin._control_gsettings("escalation at 2 fine 75", msg)
    assert "fine" in out
    esc = surface.escalation_config(KEY)
    assert esc.interval_at(2) == {"at": 2, "action": "fine",
                                  "amount": 75}
    out = mixin._control_gsettings("escalation final kick", msg)
    assert surface.escalation_config(KEY).final_action == "kick"
    out = mixin._control_gsettings("estate ltv 45", msg)
    assert "estate ltv: 45" in out
    settings = surface.store.settings(KEY)
    assert settings["estate"]["ltv_pct"] == 45
    # the bare panel shows both blocks
    panel = mixin._control_gsettings("", msg)
    text = getattr(panel, "text", panel)
    assert "Escalation" in text and "Estate" in text


# ── §30: the DM console ─────────────────────────────────────────


def _dm_msg(sender_id="99", sender="Boss", platform="telegram-bot"):
    chat = _Chat(key=f"{platform}:99", platform=platform,
                 kind="dm", title="")
    return _Msg(chat=chat, sender_id=sender_id, sender=sender,
                sender_username="boss")


def test_console_lists_only_admined_groups(tmp_path):
    adapter = _Adapter(statuses={"99": "administrator"})
    mixin, surface = _runtime(tmp_path, adapter)
    surface.store.group(KEY)
    surface.store.group(WA_KEY)
    # telegram adapter serves both keys in this rig; 99 admins
    out = mixin._control_console("", _dm_msg())
    text = getattr(out, "text", out)
    assert KEY in text
    # a plain member gets the plain refusal, no settings leak
    out2 = mixin._control_console("", _dm_msg(sender_id="7",
                                              sender="Ada"))
    assert out2 == ("you don't admin any group I serve — the "
                    "console opens for group admins.")


def test_console_change_lands_and_journals(tmp_path):
    adapter = _Adapter(statuses={"99": "administrator"})
    mixin, surface = _runtime(tmp_path, adapter)
    surface.store.group(KEY)
    mixin._control_console("", _dm_msg())
    out = mixin._control_console("g1 estate ltv 33", _dm_msg())
    text = getattr(out, "text", out)
    assert "✓" in text and "33" in text
    settings = surface.store.settings(KEY)
    assert settings["estate"]["ltv_pct"] == 33
    audits = [a for a in surface.audit(KEY, limit=50)
              if a["kind"] == "console_change"]
    assert audits and audits[-1]["by"] == "99"
    assert audits[-1]["user_id"] == "99"
    # and it shows in the group's /glog with the admin named
    log = mixin._control_glog("10", _Msg(sender_id="99"))
    log_text = getattr(log, "text", log)
    assert "console_change" in log_text and "99" in log_text


def test_console_section_render_and_escalation_change(tmp_path):
    adapter = _Adapter(statuses={"99": "administrator"})
    mixin, surface = _runtime(tmp_path, adapter)
    surface.store.group(KEY)
    mixin._control_console("", _dm_msg())
    out = mixin._control_console("g1 defense", _dm_msg())
    text = getattr(out, "text", out)
    assert "ladder:" in text and "Grammar" in text
    out = mixin._control_console(
        "g1 defense escalation limit 5", _dm_msg())
    assert surface.escalation_config(KEY).warn_limit == 5


def test_console_owner_gate_holds_for_autonomy(tmp_path):
    # 99 is an admin but NOT the owner; 55 is the creator.
    adapter = _Adapter(statuses={"99": "administrator",
                                 "55": "creator"})
    mixin, surface = _runtime(tmp_path, adapter)
    surface.store.group(KEY)
    mixin._control_console("", _dm_msg(sender_id="99"))
    out = mixin._control_console(
        "g1 defense autonomy off", _dm_msg(sender_id="99"))
    text = getattr(out, "text", out)
    assert "owner" in text
    settings = surface.store.settings(KEY)
    assert settings["defense"]["autonomy"] is True
    # the creator's console session flips it
    mixin._control_console("", _dm_msg(sender_id="55",
                                       sender="Owner"))
    out = mixin._control_console(
        "g1 defense autonomy off",
        _dm_msg(sender_id="55", sender="Owner"))
    settings = surface.store.settings(KEY)
    assert settings["defense"]["autonomy"] is False


def test_console_in_group_points_to_dm(tmp_path):
    adapter = _Adapter(statuses={"99": "administrator"})
    mixin, surface = _runtime(tmp_path, adapter)
    out = mixin._control_console("", _Msg(sender_id="99"))
    assert "DM" in out


def test_manual_warn_quotes_the_next_step(tmp_path):
    """Manual warns read the same ladder: with the default
    design the denominators are the mute step (3) until it
    fires, exactly as the shipped ladder quoted them."""
    adapter = _Adapter(statuses={"99": "administrator"})
    mixin, surface = _runtime(tmp_path, adapter)
    surface.handle_message(KEY, "7", username="loud", name="Lou",
                           text="hi", ts=time.time())
    msg = _Msg(sender_id="99", sender="Boss")
    out1 = mixin._control_warn("@loud caps", msg)
    assert "1/3" in out1
    out2 = mixin._control_warn("@loud caps", msg)
    assert "2/3" in out2
    out3 = mixin._control_warn("@loud caps", msg)
    assert "muted for an hour" in out3
    assert ("restrict", "7", 60) in adapter.calls
    member = surface.member(KEY, "7")
    assert member["warn_count"] == 0
    # the spine kept the history the ladder climbs on
    assert surface.strikes_for(KEY, "7")["count"] == 3

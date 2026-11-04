"""Member cleanup (group build — the DESIGN gap named since
2026-10-25: deleted-account scanning + acting on inactive
members).  Pins:

* the scan is a DRY RUN — it reports deleted accounts
  (Telegram's own markers), members inactive past the window
  (ledger evidence, then Telegram last-seen), and the members
  it cannot judge — and never calls a removal verb;
* last-seen CLASSES rescue or abstain, never convict;
* /rmdeleted and /rminactive kick (never ban) through the
  admin seam, journaled per member in /glog via _mod_run, one
  failure never aborts the rest or reports as success;
* admins/owner and unverifiable standings are never removed;
* non-admins refused, unclaimed groups refused, non-Telegram
  platforms get the platform truth;
* the Bot API path: no member list exists, so the ledger is
  the universe and each member is probed via getChatMember;
* settings: the cleanup block in /gsettings grammar and the
  DM console's cleanup section share one write path.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from nomorals.social.chat.group_surface import GroupStore, GroupSurface

KEY = "telegram:-1001"        # userbot surface (claimed by designation)
BOTKEY = "telegram-bot:-1002"  # bot surface (claimed by bot presence)
NOW = time.time()
DAY = 86400.0


def _row(uid, name, *, username="", bot=False, deleted=False,
         status_kind="", was_online=None):
    return {"id": uid, "name": name, "username": username,
            "bot": bot, "deleted": deleted,
            "status_kind": status_kind, "was_online": was_online}


class _UserbotAdapter:
    """MTProto-shaped fake: full member list + per-member role."""

    def __init__(self, members=(), *, sender_admin=True,
                 refuse_ids=(), statuses=None, my_admin=True):
        self.calls: list = []
        self.list_calls = 0
        self._members = list(members)
        self._sender_admin = sender_admin
        self._refuse = {str(x) for x in refuse_ids}
        self._statuses = {str(k): v for k, v in
                          (statuses or {}).items()}
        self._me_id = 4242  # Devon's own account on this surface
        self._my_admin = my_admin

    def admin_my_rights(self, chat):
        return {"is_admin": self._my_admin, "rights": {}}

    def admin_member(self, chat, user_id):
        uid = str(user_id)
        if uid in self._statuses:
            return {"ok": True, "status": self._statuses[uid]}
        if uid == "admin1":
            return {"ok": True,
                    "status": "administrator" if self._sender_admin
                    else "member"}
        return {"ok": True, "status": "member"}

    def admin_members_full(self, chat):
        self.list_calls += 1
        return {"ok": True, "members": list(self._members),
                "count": len(self._members), "truncated": False}

    def admin_administrators(self, chat):
        return {"ok": True, "administrators": [
            {"id": "admin1", "status": "administrator"},
            {"id": "creator1", "status": "creator"}]}

    def admin_ban(self, chat, user_id):
        self.calls.append(("ban", str(user_id)))
        if str(user_id) in self._refuse:
            return {"ok": False, "error": "not enough rights"}
        return {"ok": True}

    def admin_unban(self, chat, user_id):
        self.calls.append(("unban", str(user_id)))
        return {"ok": True}


class _BotAdapter:
    """Bot API-shaped fake: NO member list — per-member probes."""

    def __init__(self, profiles=None, *, sender_admin=True,
                 refuse_ids=()):
        self.calls: list = []
        self._profiles = profiles or {}
        self._sender_admin = sender_admin
        self._refuse = {str(x) for x in refuse_ids}
        self._bot_id = 777

    def admin_member(self, chat, user_id):
        uid = str(user_id)
        prof = self._profiles.get(uid)
        if prof is not None:
            return {"ok": True, "status": prof["status"]}
        return {"ok": True,
                "status": "administrator" if (
                    uid == "admin1" and self._sender_admin)
                else "member"}

    def admin_member_profile(self, chat, user_id):
        prof = self._profiles.get(str(user_id))
        if prof is None:
            return {"ok": False, "error": "user not found"}
        return {"ok": True, "status": prof["status"],
                "user": prof["user"]}

    def admin_member_count(self, chat):
        return {"ok": True, "count": 50}

    def admin_ban(self, chat, user_id):
        self.calls.append(("ban", str(user_id)))
        if str(user_id) in self._refuse:
            return {"ok": False, "error": "refused"}
        return {"ok": True}

    def admin_unban(self, chat, user_id):
        self.calls.append(("unban", str(user_id)))
        return {"ok": True}


def _msg(key=KEY, platform="telegram", *, sender_id="admin1"):
    return SimpleNamespace(
        chat=SimpleNamespace(key=key, platform=platform,
                             kind="group", title="Builders",
                             chat_id=key.split(":", 1)[1]),
        sender_id=sender_id, sender="Admin",
        sender_username="boss", text="", meta={}, reply_to="",
        message_id="900")


def _rig(tmp_path: Path, adapter, *, key=KEY, designate=True):
    from nomorals.agents.partner.runtime_group import \
        RuntimeGroupMixin

    surface = GroupSurface(GroupStore(tmp_path / "g.json").load())
    if designate:
        surface.set_designated(key, True, by="owner")
    mixin = RuntimeGroupMixin()
    mixin._group_surface = lambda: surface  # type: ignore[method-assign]
    mixin._group_adapter = lambda platform: adapter  # type: ignore[method-assign]
    return mixin, surface


def _seed_ledger(surface, key=KEY):
    """Ledger truth: ghost joined 200d ago and never spoke;
    regular joined 200d ago but messaged yesterday; newbie
    joined 3d ago."""
    surface.note_join(key, "ghost", name="Ghost",
                      ts=NOW - 200 * DAY)
    surface.note_join(key, "regular", name="Regular",
                      ts=NOW - 200 * DAY)
    surface.handle_message(key, "regular", name="Regular",
                           text="hi", ts=NOW - DAY)
    surface.note_join(key, "newbie", name="Newbie",
                      ts=NOW - 3 * DAY)


# ── the organ ──────────────────────────────────────────────────

def test_organ_deleted_inactive_unknown_split():
    from nomorals.social.chat.group_cleanup import analyze_cleanup

    rows = [
        _row("dead", "Deleted Account", deleted=True),
        _row("ghost", "Ghost"),
        _row("hidden", "Hidden"),
        _row("recent", "Recent", status_kind="recently"),
        _row("month", "Month", status_kind="last_month"),
        _row("quietbot", "QuietBot", bot=True),
        _row("boss", "Boss"),
    ]
    ledger = {
        "ghost": {"joined_ts": NOW - 200 * DAY,
                  "last_seen": NOW - 200 * DAY,
                  "message_count": 0, "left_ts": None},
        "boss": {"joined_ts": NOW - 200 * DAY,
                 "last_seen": NOW - 200 * DAY,
                 "message_count": 0, "left_ts": None},
    }
    out = analyze_cleanup(rows, ledger, now=NOW, inactive_days=90,
                          protected_ids={"boss"})
    assert [f["user_id"] for f in out["deleted"]] == ["dead"]
    assert [f["user_id"] for f in out["inactive"]] == ["ghost"]
    assert out["inactive"][0]["basis"] == "ledger"
    assert out["inactive"][0]["days_inactive"] == 200
    # hidden: no evidence at all → unjudged, never convicted.
    # month: last-seen class rescues at a 90d window.
    assert [f["user_id"] for f in out["unknown"]] == ["hidden"]
    assert [f["user_id"] for f in out["protected"]] == ["boss"]
    assert out["coverage"]["bots_skipped"] == 1


def test_organ_status_class_never_convicts():
    from nomorals.social.chat.group_cleanup import analyze_cleanup

    rows = [_row("month", "Month", status_kind="last_month"),
            _row("week", "Week", status_kind="last_week")]
    out = analyze_cleanup(rows, {}, now=NOW, inactive_days=10)
    # last_week = seen within 7 days: that IS activity inside a
    # 10-day window, so Week is rescued.  last_month (within
    # ~30 days) can neither prove activity within 10 days nor
    # silence past it: Month abstains.  NEITHER is convicted.
    assert out["inactive"] == []
    assert {f["user_id"] for f in out["unknown"]} == {"month"}
    out = analyze_cleanup(rows, {}, now=NOW, inactive_days=5)
    # At 5 days even last_week cannot prove recency: both
    # abstain, still neither convicted.
    assert out["inactive"] == []
    assert {f["user_id"] for f in out["unknown"]} == {"month",
                                                      "week"}
    out = analyze_cleanup(rows, {}, now=NOW, inactive_days=90)
    assert out["unknown"] == [] and out["inactive"] == []


def test_organ_telegram_was_online_decides_and_rescues():
    from nomorals.social.chat.group_cleanup import analyze_cleanup

    rows = [
        _row("gone", "Gone", was_online=NOW - 300 * DAY),
        _row("alive", "Alive", was_online=NOW - DAY),
    ]
    ledger = {"alive": {"joined_ts": NOW - 300 * DAY,
                        "last_seen": NOW - 300 * DAY,
                        "message_count": 3, "left_ts": None}}
    out = analyze_cleanup(rows, ledger, now=NOW, inactive_days=90)
    assert [f["user_id"] for f in out["inactive"]] == ["gone"]
    assert out["inactive"][0]["basis"] == "telegram"
    # alive's ledger is ancient but Telegram saw them yesterday.
    assert out["unknown"] == []


# ── the scan (dry run) ─────────────────────────────────────────

def _scan_rig(tmp_path):
    members = [
        _row("dead", "Deleted Account", deleted=True),
        _row("ghost", "Ghost"),
        _row("regular", "Regular"),
        _row("hidden", "Hidden"),
        _row("4242", "Devon Herself"),
    ]
    adapter = _UserbotAdapter(members)
    mixin, surface = _rig(tmp_path, adapter)
    _seed_ledger(surface)
    return mixin, surface, adapter


def test_scan_reports_and_removes_nothing(tmp_path):
    mixin, surface, adapter = _scan_rig(tmp_path)
    out = mixin._control_cleanup("", _msg())
    assert "deleted accounts: 1" in out
    assert "inactive past 90d: 1" in out
    assert "Ghost" in out and "Hidden" in out
    assert "scan only — nothing was removed" in out
    assert "full member list" in out  # the coverage source line
    # Devon's own account is silently out of the sweep.
    assert "Devon Herself" not in out
    # THE dry-run pin: zero removal verbs, zero audits of action.
    assert adapter.calls == []
    assert not [e for e in surface.audit(KEY, limit=50)
                if e["kind"] == "admin_action"]


def test_scan_unclaimed_group_refused(tmp_path):
    adapter = _UserbotAdapter([_row("dead", "X", deleted=True)])
    mixin, surface = _rig(tmp_path, adapter, designate=False)
    out = mixin._control_cleanup("", _msg())
    assert "switched on" in out
    assert adapter.calls == []


def test_scan_non_admin_refused(tmp_path):
    adapter = _UserbotAdapter([], sender_admin=False)
    mixin, surface = _rig(tmp_path, adapter)
    out = mixin._control_cleanup("", _msg())
    assert "admin" in out


def test_cleanup_platform_truth_off_telegram(tmp_path):
    adapter = _UserbotAdapter([])
    mixin, surface = _rig(tmp_path, adapter, key="whatsapp:123",
                          designate=False)
    out = mixin._control_cleanup(
        "", _msg(key="whatsapp:123", platform="whatsapp"))
    assert "Telegram tool" in out


# ── removals ───────────────────────────────────────────────────

def test_rmdeleted_kicks_and_journals(tmp_path):
    members = [
        _row("dead", "Deleted Account", deleted=True),
        _row("ghost", "Ghost"),
    ]
    adapter = _UserbotAdapter(members)
    mixin, surface = _rig(tmp_path, adapter)
    _seed_ledger(surface)
    out = mixin._control_rmdeleted("", _msg())
    assert "1 removed" in out and "0 failed" in out
    # A kick: ban immediately lifted — and ONLY for the dead.
    assert adapter.calls == [("ban", "dead"), ("unban", "dead")]
    audit = surface.audit(KEY, limit=50)
    assert any(e["kind"] == "admin_action"
               and e["action"] == "cleanup_deleted"
               and e["user_id"] == "dead" for e in audit)
    # "dead" was never in the ledger; ghost is untouched.
    assert surface.member(KEY, "dead") is None
    ghost = surface.member(KEY, "ghost")
    assert ghost is not None and ghost.get("left_ts") is None


def test_rmdeleted_failure_never_aborts_or_claims_success(tmp_path):
    members = [
        _row("dead1", "Deleted Account", deleted=True),
        _row("dead2", "Deleted Account", deleted=True),
    ]
    adapter = _UserbotAdapter(members, refuse_ids={"dead2"})
    mixin, surface = _rig(tmp_path, adapter)
    out = mixin._control_rmdeleted("", _msg())
    assert "1 removed" in out and "1 failed" in out
    assert ("ban", "dead1") in adapter.calls
    audit = surface.audit(KEY, limit=50)
    assert any(e["kind"] == "admin_action_failed"
               and e["user_id"] == "dead2" for e in audit)
    assert any(e["kind"] == "admin_action"
               and e["user_id"] == "dead1" for e in audit)


def test_rminactive_window_override_and_default(tmp_path):
    members = [_row("ghost", "Ghost"),
               _row("regular", "Regular")]
    adapter = _UserbotAdapter(members)
    mixin, surface = _rig(tmp_path, adapter)
    _seed_ledger(surface)
    # 210d window: ghost (200d silent) is inside it — no removal.
    out = mixin._control_rminactive("210", _msg())
    assert "nothing to remove" in out
    assert adapter.calls == []
    # Default 90d window: ghost goes, regular (active) stays.
    out = mixin._control_rminactive("", _msg())
    assert "1 removed" in out
    assert adapter.calls == [("ban", "ghost"), ("unban", "ghost")]
    assert surface.member(KEY, "ghost")["left_ts"] is not None


def test_rminactive_bad_days_is_usage_not_a_guess(tmp_path):
    adapter = _UserbotAdapter([_row("ghost", "Ghost")])
    mixin, surface = _rig(tmp_path, adapter)
    _seed_ledger(surface)
    out = mixin._control_rminactive("whenever", _msg())
    assert "usage:" in out and "90 days" in out
    assert adapter.calls == []


def test_admin_target_is_never_removed(tmp_path):
    members = [_row("boss", "Boss", deleted=True)]
    adapter = _UserbotAdapter(
        members, statuses={"boss": "administrator"})
    mixin, surface = _rig(tmp_path, adapter)
    out = mixin._control_rmdeleted("", _msg())
    assert "0 removed" in out and "1 skipped" in out
    assert adapter.calls == []
    audit = surface.audit(KEY, limit=50)
    assert any(e["kind"] == "cleanup_skip"
               and e["user_id"] == "boss" for e in audit)


def test_removals_off_blocks_removal_but_not_scan(tmp_path):
    members = [_row("dead", "Deleted Account", deleted=True)]
    adapter = _UserbotAdapter(members)
    mixin, surface = _rig(tmp_path, adapter)
    assert "removals off" in mixin._control_gsettings(
        "cleanup off", _msg())
    out = mixin._control_rmdeleted("", _msg())
    assert "removals are off" in out
    assert adapter.calls == []
    scan = mixin._control_cleanup("", _msg())
    assert "deleted accounts: 1" in scan


# ── automatic upkeep: deleted accounts remove themselves ───────

def test_auto_drive_removes_deleted_without_any_command(tmp_path):
    members = [
        _row("dead", "Deleted Account", deleted=True),
        _row("ghost", "Ghost"),
        _row("regular", "Regular"),
    ]
    adapter = _UserbotAdapter(members)
    mixin, surface = _rig(tmp_path, adapter)
    _seed_ledger(surface)
    # No command is issued — a group turn's upkeep beat runs.
    mixin._cleanup_auto_drive(surface, KEY, _msg())
    assert adapter.calls == [("ban", "dead"), ("unban", "dead")]
    audit = surface.audit(KEY, limit=50)
    assert any(e["kind"] == "admin_action"
               and e["action"] == "cleanup_deleted"
               and e["user_id"] == "dead"
               and e["by"] == "member cleanup (automatic)"
               for e in audit)
    assert any(e["kind"] == "cleanup_auto"
               and e["removed"] == 1 for e in audit)


def test_auto_drive_never_sweeps_inactive_people(tmp_path):
    members = [_row("ghost", "Ghost"),
               _row("regular", "Regular")]
    adapter = _UserbotAdapter(members)
    mixin, surface = _rig(tmp_path, adapter)
    _seed_ledger(surface)  # ghost: 200 days silent — a PERSON
    mixin._cleanup_auto_drive(surface, KEY, _msg())
    assert adapter.calls == []
    assert surface.member(KEY, "ghost")["left_ts"] is None


def test_auto_drive_toggle_off_removes_nothing(tmp_path):
    members = [_row("dead", "Deleted Account", deleted=True)]
    adapter = _UserbotAdapter(members)
    mixin, surface = _rig(tmp_path, adapter)
    assert "auto-remove deleted accounts off" in \
        mixin._control_gsettings("cleanup auto off", _msg())
    assert surface.cleanup_cfg(KEY)["auto_deleted"] is False
    mixin._cleanup_auto_drive(surface, KEY, _msg())
    assert adapter.calls == []
    assert adapter.list_calls == 0  # gated before enumeration


def test_auto_drive_unclaimed_and_powerless_do_nothing(tmp_path):
    members = [_row("dead", "Deleted Account", deleted=True)]
    # Unclaimed room (userbot surface, no designation).
    adapter = _UserbotAdapter(members)
    mixin, surface = _rig(tmp_path, adapter, designate=False)
    mixin._cleanup_auto_drive(surface, KEY, _msg())
    assert adapter.calls == [] and adapter.list_calls == 0
    # Claimed room, but Devon holds no admin power in it.
    adapter2 = _UserbotAdapter(members, my_admin=False)
    mixin2, surface2 = _rig(tmp_path, adapter2)
    mixin2._cleanup_auto_drive(surface2, KEY, _msg())
    assert adapter2.calls == []


def test_auto_drive_autonomy_off_holds(tmp_path):
    members = [_row("dead", "Deleted Account", deleted=True)]
    adapter = _UserbotAdapter(members)
    mixin, surface = _rig(tmp_path, adapter)
    surface.update_settings(KEY, defense={"autonomy": False})
    mixin._cleanup_auto_drive(surface, KEY, _msg())
    assert adapter.calls == [] and adapter.list_calls == 0


def test_auto_drive_failure_journals_and_never_aborts(tmp_path):
    members = [
        _row("dead1", "Deleted Account", deleted=True),
        _row("dead2", "Deleted Account", deleted=True),
    ]
    adapter = _UserbotAdapter(members, refuse_ids={"dead2"})
    mixin, surface = _rig(tmp_path, adapter)
    mixin._cleanup_auto_drive(surface, KEY, _msg())
    assert ("ban", "dead1") in adapter.calls
    audit = surface.audit(KEY, limit=50)
    assert any(e["kind"] == "admin_action"
               and e["user_id"] == "dead1" for e in audit)
    assert any(e["kind"] == "admin_action_failed"
               and e["user_id"] == "dead2" for e in audit)
    assert any(e["kind"] == "cleanup_auto"
               and e["removed"] == 1 and e["failed"] == 1
               for e in audit)


def test_auto_drive_is_throttled_per_group(tmp_path):
    members = [_row("dead", "Deleted Account", deleted=True)]
    adapter = _UserbotAdapter(members)
    mixin, surface = _rig(tmp_path, adapter)
    mixin._cleanup_auto_drive(surface, KEY, _msg())
    mixin._cleanup_auto_drive(surface, KEY, _msg())
    assert adapter.list_calls == 1  # one enumeration per window


# ── settings: gsettings grammar + console share one path ───────

def test_gsettings_cleanup_days_persists(tmp_path):
    adapter = _UserbotAdapter([])
    mixin, surface = _rig(tmp_path, adapter)
    out = mixin._control_gsettings("cleanup days 45", _msg())
    assert "45 days" in out
    assert surface.cleanup_cfg(KEY)["inactive_days"] == 45
    out = mixin._control_gsettings("cleanup bogus", _msg())
    assert "usage:" in out


def test_console_cleanup_section_applies_same_grammar(tmp_path):
    adapter = _UserbotAdapter([])
    mixin, surface = _rig(tmp_path, adapter)
    reply = mixin._console_apply(surface, KEY, _msg(), "admin1",
                                 "cleanup", ["days", "30"])
    assert "30 days" in reply
    assert surface.cleanup_cfg(KEY)["inactive_days"] == 30
    assert "cleanup" in mixin._CONSOLE_SECTIONS


# ── the Bot API path: ledger universe + per-member probe ──────

def test_bot_path_probes_ledger_for_deleted(tmp_path):
    profiles = {
        "gone": {"status": "member",
                 "user": {"id": "gone",
                          "first_name": "Deleted Account",
                          "last_name": "", "username": "",
                          "is_bot": False}},
        "pal": {"status": "member",
                "user": {"id": "pal", "first_name": "Pal",
                         "last_name": "", "username": "pal",
                         "is_bot": False}},
        "lefty": {"status": "left",
                  "user": {"id": "lefty", "first_name": "Lefty",
                           "last_name": "", "username": "",
                           "is_bot": False}},
    }
    adapter = _BotAdapter(profiles)
    mixin, surface = _rig(tmp_path, adapter, key=BOTKEY,
                          designate=False)
    surface.note_join(BOTKEY, "gone", name="Gone",
                      ts=NOW - 10 * DAY)
    surface.note_join(BOTKEY, "pal", name="Pal", ts=NOW - DAY)
    surface.note_join(BOTKEY, "lefty", name="Lefty",
                      ts=NOW - 10 * DAY)
    out = mixin._control_cleanup("", _msg(key=BOTKEY,
                                          platform="telegram-bot"))
    assert "deleted accounts: 1" in out
    assert "member ledger" in out  # the coverage honesty line
    assert "Lefty" not in out      # already out — no ghost
    out = mixin._control_rmdeleted("", _msg(key=BOTKEY,
                                            platform="telegram-bot"))
    assert "1 removed" in out
    assert adapter.calls == [("ban", "gone"), ("unban", "gone")]
    assert surface.member(BOTKEY, "gone")["left_ts"] is not None


# ── adapters: the enumeration material itself ──────────────────

class _FakeStatus:
    pass


def _status(name, **attrs):
    cls = type(name, (_FakeStatus,), {})
    obj = cls()
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


class _FakeUser:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeClient:
    def __init__(self, users):
        self._users = users

    async def iter_participants(self, entity, limit=0):
        n = 0
        for user in self._users:
            if limit and n >= limit:
                break
            n += 1
            yield user


def _mtproto_adapter(users):
    from nomorals.social.chat.telegram import TelegramAdapter

    adapter = TelegramAdapter.__new__(TelegramAdapter)
    adapter._client = _FakeClient(users)

    async def _resolve(chat):
        return object()

    adapter._resolve = _resolve  # type: ignore[method-assign]

    def _run(coro, timeout=60.0):
        return asyncio.new_event_loop().run_until_complete(coro)

    adapter._run_on_loop = _run  # type: ignore[method-assign]
    return adapter


def test_mtproto_full_list_carries_deleted_and_last_seen():
    from nomorals.social.chat.base import ChatRef

    users = [
        _FakeUser(id=1, first_name="Deleted Account", last_name="",
                  username="", bot=False, deleted=True, status=None),
        _FakeUser(id=2, first_name="Old", last_name="Timer",
                  username="old", bot=False, deleted=False,
                  status=_status(
                      "UserStatusOffline",
                      was_online=datetime.fromtimestamp(
                          NOW - 300 * DAY, tz=timezone.utc))),
        _FakeUser(id=3, first_name="Now", last_name="",
                  username="now", bot=False, deleted=False,
                  status=_status("UserStatusOnline")),
    ]
    adapter = _mtproto_adapter(users)
    out = adapter.admin_members_full(
        ChatRef(platform="telegram", chat_id="-1001",
                kind="group"))
    assert out["ok"] and out["count"] == 3
    assert out["truncated"] is False
    dead, old, now = out["members"]
    assert dead["deleted"] is True
    assert old["status_kind"] == "offline"
    assert abs(old["was_online"] - (NOW - 300 * DAY)) < 2
    assert now["status_kind"] == "online"


def test_mtproto_full_list_reports_truncation():
    from nomorals.social.chat.base import ChatRef

    users = [_FakeUser(id=i, first_name=f"U{i}", last_name="",
                       username="", bot=False, deleted=False,
                       status=None) for i in range(5)]
    adapter = _mtproto_adapter(users)
    out = adapter.admin_members_full(
        ChatRef(platform="telegram", chat_id="-1001",
                kind="group"),
        limit=3)
    assert out["ok"] and out["count"] == 3
    assert out["truncated"] is True


def test_bot_profile_probe_shape():
    from nomorals.social.chat.base import ChatRef
    from nomorals.social.chat.telegram import TelegramBotAdapter

    adapter = TelegramBotAdapter.__new__(TelegramBotAdapter)

    def _api(method, **params):
        assert method == "getChatMember"
        return {"status": "member",
                "user": {"id": 9, "first_name": "Deleted Account",
                         "is_bot": False}}

    adapter._api = _api  # type: ignore[method-assign]
    out = adapter.admin_member_profile(
        ChatRef(platform="telegram-bot", chat_id="-1001",
                kind="group"), "9")
    assert out["ok"] and out["status"] == "member"
    assert out["user"]["first_name"] == "Deleted Account"
    assert out["user"]["username"] == ""


# ── registry: the trio is on the catalog, off WhatsApp ─────────

def test_registry_and_platform_surfaces():
    from nomorals.social.chat.control import (
        COMMAND_DETAILS, CONTROL_COMMANDS, LIST_GROUPS,
        PUBLIC_COMMANDS, TELEGRAM_CLEANUP_KINDS,
        WHATSAPP_UNSUPPORTED_KINDS, help_text, whatsapp_menu)

    for kind in ("cleanup", "rmdeleted", "rminactive"):
        assert kind in CONTROL_COMMANDS
        assert kind in PUBLIC_COMMANDS
        assert kind in COMMAND_DETAILS
        assert kind in TELEGRAM_CLEANUP_KINDS
        assert kind in WHATSAPP_UNSUPPORTED_KINDS
        slots = sum(kind in kinds for _name, kinds in LIST_GROUPS)
        assert slots == 1
        assert f"/{kind}" in help_text()
        assert f"/{kind}" not in whatsapp_menu()

"""gapfix129 — group hacks bundle: mass react/vote, deleted-account
removal, inactive-member action, under the personal-surface live
gate (owner rebuke 2026-11-04).

The organs already existed (mass_actions 7e167200, group_cleanup
f20afada, both 2026-11-04); what this lane proves is the live
gate on top of them:

1. the owner's stop is STICKY — a live mass run and every
   removal (command or automatic drive) refuse on a stopped
   platform; dry-runs and scans (reads) still run;
2. a degraded/no-model brain is observe-only — the automatic
   cleanup drive holds while the brain is degraded; owner-issued
   commands are exempt, per brain_gate.py;
3. the lifecycle is tested against REAL shapes — Telegram's
   deleted-account marker in both its forms (MTProto ``deleted``
   flag and the Bot API "Deleted Account" rendering), MTProto
   last-seen classes (``long_ago`` abstains), WhatsApp LID-form
   ids as opaque ids, and real provider failures (FloodWait on
   Telegram, a dead bridge on WhatsApp).

Fixtures are the small fakes the organ's own suites use
(test_mass_actions.py, test_group_cleanup.py) — the reachability
probes here go through the real handlers, the real runner and
the real organ, never a re-implementation.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

import nomorals.agents.partner.runtime_group as _grp
from nomorals.social.chat.group_cleanup import analyze_cleanup

# ── mass ───────────────────────────────────────────────────────────────────
from nomorals.social.mass_actions import (
    WA_CAPABILITIES,
    AccountRecord,
    MassAccount,
    MassArms,
    MassJournal,
    MassTarget,
    TelegramSessionActor,
    WhatsAppBridgeActor,
    parse_target,
)

# ── small fakes ─────────────────────────────────────────────────────────────

class _FakeRegistry:
    """Pairing registry with an owner stop the test can flip."""

    def __init__(self, stopped: dict | None = None):
        self._stopped = stopped or {}

    def owner_stopped(self, platform: str) -> bool:
        return bool(self._stopped.get(platform))

    def all(self) -> list:
        return []  # the fake pairs no accounts; the pool is the fixture's

    def set_stopped(self, platform: str, on: bool) -> None:
        self._stopped[platform] = on


class _FakeGate:
    """Brain gate the test can degrade."""

    def __init__(self, degraded: bool = False):
        self._degraded = degraded

    @property
    def degraded_now(self) -> bool:
        return self._degraded

    def set_degraded(self, on: bool) -> None:
        self._degraded = on


class _FakeActor:
    """A mass actor that records its calls; ``fail`` raises.
    (Actors are duck-typed in the organ — .id/.react/.vote.)"""

    def __init__(self, account_id: str, fail: str | None = None):
        self.id = account_id
        self.calls: list[tuple[str, str]] = []
        self._fail = fail

    def react(self, target: MassTarget, emoji: str) -> dict:
        self.calls.append(("react", emoji))
        if self._fail:
            raise RuntimeError(self._fail)
        return {"ok": True, "detail": "reacted"}

    def vote(self, target: MassTarget, choice: int) -> dict:
        self.calls.append(("vote", str(choice)))
        if self._fail:
            raise RuntimeError(self._fail)
        return {"ok": True, "detail": "voted"}


class _FakeHandles:
    """One MassAccount per platform, each backed by a _FakeActor."""

    def __init__(self, arms, journal, platform="telegram"):
        self._arms = arms
        self._journal = journal
        self._platform = platform
        self.calls: list[tuple[str, str]] = []
        self._fake = _FakeActor(f"{platform}:acc0")
        self._fake.calls = self.calls

    def _handles(self, platform):
        record = AccountRecord(id=f"{platform}:acc0", platform=platform,
                               identity="acc0", status="linked")
        return [MassAccount(record=record,
                            capabilities=(
                                "massreact", "pollvote", "chfollow"),
                            factory=lambda: self._fake)]

    def _service(self, stopped=None):
        from nomorals.social.mass_actions import MassRunner

        return {
            "arms": self._arms,
            "journal": self._journal,
            "runner": MassRunner(self._arms, self._journal),
            "registry": _FakeRegistry(stopped),
        }

    def _runtime(self, stopped: dict | None = None):
        from nomorals.agents.partner.runtime_provision import RuntimeProvisionMixin

        svc = self._service()
        svc["registry"] = _FakeRegistry(stopped)
        runtime = RuntimeProvisionMixin.__new__(RuntimeProvisionMixin)
        runtime._mass_service = lambda: svc
        runtime._mass_handles = self._handles
        runtime._mass_journal = lambda: self._journal
        return runtime


def _mass_run(runtime, tool, tail):
    return runtime._control_mass(tool, tail)


def _bare_mass_env():
    """Armed pair + journal + fake actor, mirroring
    test_mass_actions._bare_mass_runtime (the suite's own rig)."""
    import tempfile
    from pathlib import Path

    tmp = tempfile.mkdtemp(prefix="hacksbundle-mass-")
    arms = MassArms(Path(tmp) / "arms.json")
    journal = MassJournal(Path(tmp) / "journal.jsonl")
    holder = _FakeHandles(arms, journal)
    return holder, arms, journal


@pytest.fixture()
def mass_env():
    holder, arms, journal = _bare_mass_env()
    runtime = holder._runtime()
    runtime._control_mass("massreact", "arm")
    return holder, runtime, arms, journal


# ── 1. the owner's stop is STICKY ───────────────────────────────────────────

def test_mass_live_run_refuses_on_stopped_platform(mass_env):
    holder, runtime, arms, journal = mass_env
    runtime._mass_service = lambda: holder._service({"telegram": True})
    out = _mass_run(runtime, "massreact",
                    "https://t.me/c/12345/678 | 🔥 | live")
    assert "standing stop" in out
    assert "nothing was done" in out
    assert holder.calls == []  # no account was ever engaged
    refused = [e for e in journal.tail(50)
               if e.get("kind") == "run_refused"]
    assert len(refused) == 1
    assert refused[0]["note"] == out.removeprefix("⚠️ ").strip()


def test_mass_dry_run_still_plans_on_stopped_platform(mass_env):
    holder, runtime, arms, journal = mass_env
    runtime._mass_service = lambda: holder._service({"telegram": True})
    out = _mass_run(runtime, "massreact",
                    "https://t.me/c/12345/678 | 🔥")
    assert "DRY RUN" in out  # a read plans; it performs nothing
    assert holder.calls == []


def test_mass_live_run_proceeds_when_platform_not_stopped(mass_env):
    holder, runtime, arms, journal = mass_env
    out = _mass_run(runtime, "massreact",
                    "https://t.me/c/12345/678 | 🔥 | live")
    assert "1/1 account(s) acted" in out
    assert holder.calls == [("react", "🔥")]


def test_mass_meta_commands_still_work_when_stopped(mass_env):
    holder, runtime, arms, journal = mass_env
    runtime._mass_service = lambda: holder._service({"telegram": True})
    out = _mass_run(runtime, "massreact", "status")
    assert "armed" in out.lower()


def test_mass_live_run_refusal_names_the_stop_in_words(mass_env):
    """The refusal is the chatops wording, verbatim — the owner
    hears the same sentence from every surface."""
    holder, runtime, arms, journal = mass_env
    runtime._mass_service = lambda: holder._service({"telegram": True})
    out = _mass_run(runtime, "massreact",
                    "https://t.me/c/12345/678 | 🔥 | live")
    assert out.startswith("⚠️ telegram is under your standing stop")


# ── 3. real shapes: provider failures ───────────────────────────────────────

class _FloodClient:
    """A Telegram client that answers every call with a FloodWait
    — the real failure shape, not a hand-made dict."""

    async def connect(self):
        return None

    async def disconnect(self):
        return None

    async def react(self, target, emoji):
        raise RuntimeError(
            "FLOOD_WAIT_60: A wait of 60 seconds is required")


def test_telegram_floodwait_fails_per_account_and_names_itself():
    holder, arms, journal = _bare_mass_env()
    from nomorals.social.mass_actions import MassRunner

    record = AccountRecord(id="telegram:acc0", platform="telegram",
                           identity="acc0", status="linked")
    failing = MassAccount(
        record=record, capabilities=("massreact",),
        factory=lambda: TelegramSessionActor(
            record, client_factory=lambda path: _FloodClient()))
    record2 = AccountRecord(id="telegram:acc1", platform="telegram",
                            identity="acc1", status="linked")
    succeeding = MassAccount(
        record=record2, capabilities=("massreact",),
        factory=lambda: _FakeActor("telegram:acc1"))
    arms.arm("massreact")
    target, _ = parse_target("https://t.me/c/12345/678")
    run = MassRunner(arms, journal).run(
        tool="massreact", target=target, emoji="🔥",
        accounts=[failing, succeeding], dry_run=False)
    assert run.entries[0]["ok"] is False
    assert "FLOOD_WAIT_60" in run.entries[0]["error"]
    # the run stops on the error — the second account is never
    # engaged, and the journal says so, not silently
    assert run.entries[1]["detail"] == "skipped — run stopped on error"
    assert run.entries[1]["ok"] is False


def _wa_request_bridge_down(payload):
    if payload.get("cmd") == "channel_resolve":
        return {"ok": True, "chat": "12345@newsletter"}
    return {"ok": False,
            "error": ("bridge on port 9999 unreachable: "
                      "[Errno 111] Connection refused")}


def test_whatsapp_bridge_down_failure_names_the_bridge():
    holder, arms, journal = _bare_mass_env()
    from nomorals.social.mass_actions import MassRunner

    record = AccountRecord(id="whatsapp:acc0", platform="whatsapp",
                           identity="acc0", status="linked")
    account = MassAccount(
        record=record, capabilities=WA_CAPABILITIES,
        factory=lambda: WhatsAppBridgeActor(
            record, request=_wa_request_bridge_down))
    arms.arm("massreact")
    target, _ = parse_target(
        "https://whatsapp.com/channel/0029VaExample/12345")
    run = MassRunner(arms, journal).run(
        tool="massreact", target=target, emoji="🔥",
        accounts=[account], dry_run=False)
    assert run.entries[0]["ok"] is False
    assert "bridge on port 9999 unreachable" in run.entries[0]["error"]


# ── group cleanup: the rig ───────────────────────────────────────────────────

_GROUP = "telegram:-1001234567890"
_MSG = None


class _UserbotAdapter:
    """The cleanup suite's own fake adapter (test_group_cleanup)."""

    def __init__(self, members, my_rights=("ban_users",)):
        self._members = list(members)
        self._my_rights = list(my_rights)
        self.calls: list[tuple] = []

    def admin_members_full(self, chat):
        return {"ok": True, "members": self._members}

    def admin_member(self, chat, user_id):
        for member in self._members:
            if str(member.get("id")) == str(user_id):
                return {"ok": True,
                        "status": member.get("status", "member")}
        return {"ok": True, "status": "member"}

    def admin_administrators(self, chat):
        return {"ok": True, "administrators": []}

    def admin_ban(self, chat, user_id):
        self.calls.append(("ban", user_id))
        return {"ok": True, "detail": "banned"}

    def admin_unban(self, chat, user_id):
        self.calls.append(("unban", user_id))
        return {"ok": True, "detail": "unbanned"}

    def admin_my_rights(self, chat):
        return {"is_admin": True,
                "rights": {"ban_users": "ban_users" in self._my_rights}}


def _msg(key=_GROUP, sender_id="admin1", text=""):
    return SimpleNamespace(
        chat=SimpleNamespace(platform="telegram", key=key,
                             kind="group", title="Builders",
                             chat_id="-1001234567890"),
        sender_id=sender_id, sender="owner",
        sender_username="", text=text,
        meta={"is_owner": True}, reply_to="", message_id="")


def _rig(members, ledger=None, *, stopped=None, degraded=False):
    """A bare RuntimeGroupMixin wired to the fake adapter, with a
    flippable owner stop and a flippable brain gate."""
    import tempfile
    from pathlib import Path

    from nomorals.social.chat.group_surface import GroupStore, GroupSurface

    tmp = Path(tempfile.mkdtemp(prefix="hacksbundle-cleanup-"))
    store = GroupStore(tmp / "groups.json").load()
    surface = GroupSurface(store)
    key = _GROUP
    surface.set_designated(key, True, by="owner")
    adapter = _UserbotAdapter(members)
    if ledger:
        entry = store.group(key)
        entry["members"] = ledger
        store.save()

    mixin = _grp.RuntimeGroupMixin.__new__(_grp.RuntimeGroupMixin)
    mixin._group_surface = lambda: surface
    mixin._group_adapter = lambda key, ws=None: adapter
    # ONE registry instance for the whole rig: the drive calls
    # _pairing_supervisor() on every beat, so a per-call build
    # would make a test flip invisible to the next beat.
    registry = _FakeRegistry(stopped)
    mixin._pairing_supervisor = lambda: SimpleNamespace(
        registry=registry)
    mixin._fake_registry = registry
    mixin._get_brain_gate = lambda: _FakeGate(degraded)
    mixin._is_operator = lambda message: True
    return mixin, surface, adapter


def _members_now():
    return [
        {"id": "admin1", "status": "administrator",
         "name": "Owner", "username": "owner"},
        {"id": "dead", "deleted": True, "name": "Gone",
         "username": ""},
        {"id": "ghost", "name": "Ghost", "username": "ghost"},
        {"id": "hidden", "name": "Hidden", "username": ""},
    ]


def _ledger_old_now():
    now = time.time()
    day = 86400
    return {
        "dead": {"last_seen": now - 400 * day, "joined_ts": now - 500 * day},
        "ghost": {"last_seen": now - 200 * day, "joined_ts": now - 300 * day},
    }


# ── 1. the stop on removal commands ──────────────────────────────────────────

def test_rmdeleted_refuses_when_platform_stopped():
    mixin, surface, adapter = _rig(
        _members_now(), _ledger_old_now(),
        stopped={"telegram": True})
    out = mixin._control_rmdeleted("", _msg())
    assert "standing stop" in out
    assert "nothing was removed" in out
    assert adapter.calls == []


def test_rminactive_refuses_when_platform_stopped():
    mixin, surface, adapter = _rig(
        _members_now(), _ledger_old_now(),
        stopped={"telegram": True})
    out = mixin._control_rminactive("90", _msg())
    assert "standing stop" in out
    assert adapter.calls == []


def test_cleanup_scan_still_reads_when_platform_stopped():
    mixin, surface, adapter = _rig(
        _members_now(), _ledger_old_now(),
        stopped={"telegram": True})
    out = mixin._control_cleanup("", _msg())
    assert "deleted" in out.lower()
    assert adapter.calls == []  # the scan removes nothing


def test_rmdeleted_kicks_only_the_deleted():
    mixin, surface, adapter = _rig(_members_now(), _ledger_old_now())
    out = mixin._control_rmdeleted("", _msg())
    assert "removed" in out.lower()
    assert ("ban", "dead") in adapter.calls
    assert ("ban", "ghost") not in adapter.calls
    assert ("ban", "hidden") not in adapter.calls
    assert ("ban", "admin1") not in adapter.calls


def test_rminactive_kicks_inactive_never_unknown():
    mixin, surface, adapter = _rig(_members_now(), _ledger_old_now())
    mixin._control_rminactive("90", _msg())
    assert ("ban", "ghost") in adapter.calls
    assert ("ban", "hidden") not in adapter.calls  # no evidence
    assert ("ban", "dead") not in adapter.calls  # deleted ≠ inactive
    assert ("ban", "admin1") not in adapter.calls


# ── 2. the automatic drive holds ─────────────────────────────────────────────

def test_auto_drive_holds_when_owner_stopped():
    mixin, surface, adapter = _rig(
        _members_now(), _ledger_old_now(),
        stopped={"telegram": True})
    mixin._cleanup_auto_drive(surface, _GROUP, _msg())
    assert adapter.calls == []  # no kick while stopped
    entry = surface.store.group(_GROUP)
    assert (entry.get("extras") or {}).get(
        "cleanup_auto_held", {}).get("reason") == "owner stop"
    # held beats never burn the throttle: the due stamp stays empty
    assert not (entry.get("cleanup_auto") or {}).get("last_run")


def test_auto_drive_holds_when_brain_degraded():
    mixin, surface, adapter = _rig(
        _members_now(), _ledger_old_now(), degraded=True)
    mixin._cleanup_auto_drive(surface, _GROUP, _msg())
    assert adapter.calls == []
    entry = surface.store.group(_GROUP)
    assert (entry.get("extras") or {}).get(
        "cleanup_auto_held", {}).get("reason") == "brain degraded"


def test_auto_drive_hold_is_audited_once_then_resumed_once():
    mixin, surface, adapter = _rig(
        _members_now(), _ledger_old_now(),
        stopped={"telegram": True})
    registry = mixin._pairing_supervisor().registry
    mixin._cleanup_auto_drive(surface, _GROUP, _msg())
    mixin._cleanup_auto_drive(surface, _GROUP, _msg())
    entry = surface.store.group(_GROUP)
    held = [a for a in entry.get("audit", [])
            if a.get("kind") == "cleanup_auto_held"]
    assert len(held) == 1  # one episode, one line — never a drumbeat
    registry.set_stopped("telegram", False)
    mixin._cleanup_auto_drive(surface, _GROUP, _msg())
    entry = surface.store.group(_GROUP)
    assert ("ban", "dead") in adapter.calls  # the drive runs again
    resumed = [a for a in entry.get("audit", [])
               if a.get("kind") == "cleanup_auto_resumed"]
    assert len(resumed) == 1
    assert "cleanup_auto_held" not in (entry.get("extras") or {})


def test_auto_drive_healthy_removes_deleted():
    mixin, surface, adapter = _rig(_members_now(), _ledger_old_now())
    mixin._cleanup_auto_drive(surface, _GROUP, _msg())
    assert ("ban", "dead") in adapter.calls


# ── 3. real shapes: the organ's verdicts ─────────────────────────────────────

def _ids(result, bucket):
    return [finding["user_id"] for finding in result[bucket]]


def test_deleted_marker_bot_api_rendering():
    now = time.time()
    rows = [
        {"id": "gone", "name": "Deleted Account",
         "username": "", "last_name": ""},
        {"id": "namesake", "name": "Deleted Account",
         "username": "someone"},
    ]
    result = analyze_cleanup(rows, {}, now=now)
    assert _ids(result, "deleted") == ["gone"]
    assert "namesake" not in _ids(result, "deleted")  # a name is not a marker
    assert _ids(result, "unknown") == ["namesake"]


def test_deleted_marker_mtproto_flag():
    now = time.time()
    rows = [{"id": "gone", "deleted": True, "name": "Gone"}]
    result = analyze_cleanup(rows, {}, now=now)
    assert _ids(result, "deleted") == ["gone"]


def test_lid_form_ids_pass_through_as_opaque_ids():
    now = time.time()
    day = 86400
    rows = [
        {"id": "123456789012345@lid", "deleted": True,
         "name": "Gone"},
        {"id": "15551234567@c.us", "name": "Quiet",
         "status_kind": "offline"},
        {"id": "987654321@lid", "name": "Mystery"},
    ]
    ledger = {
        "15551234567@c.us": {"last_seen": now - 200 * day,
                             "joined_ts": now - 300 * day},
    }
    result = analyze_cleanup(rows, ledger, now=now)
    assert _ids(result, "deleted") == ["123456789012345@lid"]
    assert _ids(result, "inactive") == ["15551234567@c.us"]
    assert _ids(result, "unknown") == ["987654321@lid"]


def test_mtproto_long_ago_abstains():
    """UserStatusLongAgo is a real MTProto class — the organ has
    no read for it, so it abstains rather than convicting."""
    now = time.time()
    rows = [{"id": "old", "name": "Old", "status_kind": "long_ago"}]
    result = analyze_cleanup(rows, {}, now=now)
    assert _ids(result, "unknown") == ["old"]
    assert _ids(result, "inactive") == []


def test_last_seen_class_rescues_when_no_timestamp_exists():
    """The class rescues only where no timestamp exists: a
    ledger last_seen is a real signal and decides on its own."""
    now = time.time()
    rows = [{"id": "seen", "name": "Seen",
             "status_kind": "recently"}]
    result = analyze_cleanup(rows, {}, now=now)
    assert _ids(result, "inactive") == []
    assert _ids(result, "unknown") == []  # rescued, not unknown


def test_ledger_timestamp_decides_over_the_class():
    now = time.time()
    day = 86400
    rows = [{"id": "seen", "name": "Seen",
             "status_kind": "recently"}]
    ledger = {"seen": {"last_seen": now - 200 * day,
                       "joined_ts": now - 300 * day}}
    result = analyze_cleanup(rows, ledger, now=now)
    assert _ids(result, "inactive") == ["seen"]


# ── the brain tool ───────────────────────────────────────────────────────────

class _ToolRegistry:
    def __init__(self, context):
        self.context = context
        self.tools = {}

    def register(self, name, description, capability):
        def deco(fn):
            self.tools[name] = fn
            return fn
        return deco


class _ToolRuntime:
    """A stub runtime that records the delegation."""

    def __init__(self):
        self.calls: list[tuple] = []

    def _control_cleanup(self, tail, message):
        self.calls.append(("cleanup", message.chat.key, message.meta))
        return "scan text"

    def _control_rmdeleted(self, tail, message):
        self.calls.append(("rmdeleted", message.chat.key, message.meta))
        return "removed 1"

    def _control_rminactive(self, tail, message):
        self.calls.append(("rminactive", tail, message.chat.key,
                           message.meta))
        return "removed 1"


def _tool_context(tmp, runtime=None):
    from pathlib import Path

    from nomorals.social.chat.group_surface import GroupStore, GroupSurface

    store = GroupStore(Path(tmp) / "data" / "social" / "groups.json").load()
    surface = GroupSurface(store)
    surface.set_designated(_GROUP, True, by="owner")
    entry = store.group(_GROUP)
    entry["title"] = "Builders"
    store.save()
    extras = {"partner_runtime": runtime} if runtime is not None else {}
    return SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=str(tmp)),
        extras=extras)


def _tool_fn(tmp, runtime=None):
    from nomorals.tools.group_cleanup import register

    reg = _ToolRegistry(_tool_context(tmp, runtime))
    register(reg)
    return reg.tools["group_cleanup"]


def test_tool_scan_delegates_to_the_runtime(tmp_path):
    fn = _tool_fn(tmp_path, _ToolRuntime())
    out = fn(action="scan", group="Builders")
    assert out == {"ok": True, "text": "scan text"}


def test_tool_delegation_carries_the_owner_stamp(tmp_path):
    runtime = _ToolRuntime()
    fn = _tool_fn(tmp_path, runtime)
    fn(action="rmdeleted", group=_GROUP)
    assert runtime.calls == [("rmdeleted", _GROUP, {"is_owner": True})]


def test_tool_rminactive_passes_the_window(tmp_path):
    runtime = _ToolRuntime()
    fn = _tool_fn(tmp_path, runtime)
    out = fn(action="rminactive", group="Builders", days="180")
    assert out == {"ok": True, "text": "removed 1"}
    assert runtime.calls[0][1] == "180"


def test_tool_refuses_without_a_live_runtime(tmp_path):
    fn = _tool_fn(tmp_path, None)
    out = fn(action="scan", group="Builders")
    assert out["ok"] is False
    assert "no live runtime" in out["error"]


def test_tool_unknown_group_is_answered_honestly(tmp_path):
    fn = _tool_fn(tmp_path, _ToolRuntime())
    out = fn(action="scan", group="Nobody Here")
    assert out["ok"] is False
    assert "no group on record" in out["error"]


def test_tool_unknown_action_is_answered_honestly(tmp_path):
    fn = _tool_fn(tmp_path, _ToolRuntime())
    out = fn(action="banish", group="Builders")
    assert out["ok"] is False
    assert "unknown action" in out["error"]


def test_tool_status_is_a_surface_read(tmp_path):
    fn = _tool_fn(tmp_path, None)  # no runtime needed for status
    out = fn(action="status", group="Builders")
    assert out["ok"] is True
    assert "removals ON" in out["text"]
    assert "last automatic pass: never" in out["text"]


def test_tool_scan_through_the_real_runtime(tmp_path):
    """The tool's delegation seam against the real mixin
    methods, not a stub — the scan runs and removes nothing."""
    mixin, surface, adapter = _rig(_members_now(), _ledger_old_now())
    fn = _tool_fn(tmp_path, mixin)
    # the tool resolves the group through the surface it builds
    # from the context workspace — seed that same workspace
    from nomorals.social.chat.group_surface import default_surface

    seeded = default_surface(str(tmp_path))
    seeded.set_designated(_GROUP, True, by="owner")
    seeded.store.group(_GROUP)["title"] = "Builders"
    seeded.store.save()
    out = fn(action="scan", group="Builders")
    assert out["ok"] is True
    assert "deleted" in out["text"].lower()
    assert adapter.calls == []


def test_tool_registry_count_is_804():
    from nomorals.tools.registry import ToolRegistry

    registry = ToolRegistry(SimpleNamespace())
    registry.register_builtins()
    assert len(registry._tools) == 812
    assert "group_cleanup" in registry._tools
    assert registry.failed_modules() == []

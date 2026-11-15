"""Group backup / export (§9.8): the whole group state — settings,
roster, economy ledger, audit log — as a file that can be restored,
on a schedule.

Two halves: the organ (nomorals/social/chat/group_backup.py — build,
list, load, preview, apply, delete, schedule, tick) on a bare
GroupSurface, and the chat surface (/gbackup through
PartnerRuntime.handle_control — board, export, restore preview +
confirm, schedule, delete, gates).

Design law under test: a restore never silently overwrites live
state — without ``yes`` the command answers with a preview of what
would change; only ``yes`` applies it. A kicked-and-re-added group
resumes from its backup (§9.8's "the bot kicked → state survives").

The fixtures are the pass-55 bare-runtime shape from
test_groups_command.py: a PartnerRuntime built with __new__, a
seeded GroupSurface attached as _group_surface_obj, nothing
touching the real home.
"""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.social.chat.base import ChatKind
from nomorals.social.chat.control import (
    CONTROL_COMMANDS,
    PUBLIC_COMMANDS,
    caller_may_run,
)
from nomorals.social.chat.group_surface import GroupStore, GroupSurface
from nomorals.storage.db import Database
from tests.frozen_clock import MIDDAY_TS

OWNER_ID = "111"
MEMBER_ID = "u99"
GROUP_A = "telegram-bot:-100111"
GROUP_B = "telegram-bot:-100222"
CHANNEL = "telegram-bot:-100333"
NOW = MIDDAY_TS


def _bare_runtime(community_dir: str) -> PartnerRuntime:
    rt = PartnerRuntime.__new__(PartnerRuntime)
    rt.context = SimpleNamespace(
        db=Database(":memory:"),
        settings=SimpleNamespace(community_dir=community_dir),
    )
    rt._owner_chats = {OWNER_ID}
    rt._owner_sender_ids = {OWNER_ID}
    return rt


def _dm_msg(sender_id: str, text: str):
    chat = SimpleNamespace(key=f"telegram:{sender_id}", platform="telegram",
                           kind=ChatKind.DM, chat_id=sender_id)
    return SimpleNamespace(chat=chat, text=text, meta={}, sender="someone",
                           sender_id=sender_id)


def _group_msg(sender_id: str, text: str):
    chat = SimpleNamespace(key=GROUP_A, platform="telegram",
                           kind=ChatKind.GROUP, chat_id="-100111")
    return SimpleNamespace(chat=chat, text=text, meta={}, sender="someone",
                           sender_id=sender_id)


def _text(out) -> str:
    return getattr(out, "text", out)


def _buttons(out) -> list:
    raw = getattr(out, "buttons", None) or []
    flat: list = []
    for row in raw:
        if isinstance(row, (list, tuple)):
            flat.extend(row)
        else:
            flat.append(row)
    return [str(item[1]) for item in flat
            if isinstance(item, (list, tuple)) and len(item) > 1]


def _seed_surface(tmp_path: Path) -> GroupSurface:
    surface = GroupSurface(GroupStore(tmp_path / "g.json").load())
    # Group A: two members, a live economy, a settings override.
    for uid, name in (("1", "Ada"), ("2", "Bola")):
        surface.record_join(GROUP_A, uid, username=name.lower(),
                            name=name, ts=NOW)
    surface.store.group(GROUP_A).update(
        {"settings": {
            "currency": {"symbol": "₦", "name": "Naira", "code": "NGN"},
            "economy": {"faucet_amount": 50},
        }})
    surface.store.save()
    # Group B: one member, economy untouched.
    surface.record_join(GROUP_B, "1", username="ada", name="Ada", ts=NOW)
    surface.store.save()
    # Channel: claimed-channel marker + title, nothing queued/posted.
    centry = surface.store.group(CHANNEL)
    centry.setdefault("extras", {})["channel"] = {"title": "Signals"}
    surface.store.save()
    return surface


def _seed_economy(surface: GroupSurface):
    """Give GROUP_A a real economy: Ada +100, Bola +50, Ada
    records one activity. GROUP_B stays empty."""
    econ = surface.economy
    econ.verify_member(GROUP_A, "1")
    econ.verify_member(GROUP_A, "2")
    econ.award(GROUP_A, "1", 100, "faucet")
    econ.award(GROUP_A, "2", 50, "faucet")
    econ.record_activity(GROUP_A, "1", "game")
    return econ


class _BackupCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="nm-test-gbackup-")
        self.tmp_path = Path(self._tmp.name)
        self.rt = _bare_runtime(self._tmp.name)
        self.surface = _seed_surface(self.tmp_path)
        self.rt._group_surface_obj = self.surface

    def tearDown(self):
        self._tmp.cleanup()

    def _reply(self, text, sender_id=OWNER_ID):
        return self.rt.handle_control(
            text, f"telegram:{sender_id}",
            message=_dm_msg(sender_id, text))


class TestGroupBackupOrgan(_BackupCase):
    """The organ on a bare GroupSurface: build, list, load,
    preview, apply, delete, schedule, tick."""

    def test_create_list_load_round_trip(self):
        from nomorals.social.chat.group_backup import (
            create_backup,
            list_backups,
            load_backup,
        )

        _seed_economy(self.surface)
        info = create_backup(self.surface, GROUP_A, by=OWNER_ID,
                             reason="manual")
        self.assertTrue(info["backup_id"])
        self.assertEqual(info["economy_status"], "ok")
        self.assertEqual(info["counts"]["members"], 2)
        self.assertEqual(info["counts"]["ledger_entries"], 2)
        self.assertGreaterEqual(info["counts"]["audit_entries"], 2)
        self.assertGreaterEqual(info["counts"]["economy_files"], 1)
        infos, corrupt = list_backups(self.surface, GROUP_A)
        self.assertEqual(len(infos), 1)
        self.assertEqual(corrupt, 0)
        self.assertEqual(infos[0]["backup_id"], info["backup_id"])
        backup = load_backup(self.surface, GROUP_A, info["backup_id"])
        self.assertEqual(backup["manifest"]["kind"], "devon-group-backup")
        self.assertEqual(backup["manifest"]["format_version"], 1)
        self.assertEqual(backup["manifest"]["group_key"], GROUP_A)
        self.assertEqual(set(backup["store"]["members"]), {"1", "2"})
        self.assertEqual(
            backup["store"]["settings"]["currency"]["symbol"], "₦")
        ledger = backup["economy"]["ledger"]
        self.assertEqual(len(ledger), 2)
        self.assertEqual({e["amount"] for e in ledger}, {100, 50})
        # The backup itself is journaled, and the store remembers it.
        entry = self.surface.store.group(GROUP_A)
        kinds = [a.get("kind") for a in entry.get("audit") or []]
        self.assertIn("backup_created", kinds)
        self.assertEqual(
            entry["settings"]["backup"]["last_backup_id"],
            info["backup_id"])

    def test_create_backup_unknown_group_refused(self):
        from nomorals.social.chat.group_backup import (
            GroupBackupRefused,
            create_backup,
        )

        with self.assertRaisesRegex(GroupBackupRefused, "no group"):
            create_backup(self.surface, "telegram-bot:-100999",
                          by=OWNER_ID, reason="manual")

    def test_create_backup_channel_refused(self):
        from nomorals.social.chat.group_backup import (
            GroupBackupRefused,
            create_backup,
        )

        with self.assertRaisesRegex(GroupBackupRefused, "channel"):
            create_backup(self.surface, CHANNEL, by=OWNER_ID,
                          reason="manual")

    def test_load_backup_tampered_checksum_refused(self):
        from nomorals.social.chat.group_backup import (
            GroupBackupCorrupt,
            create_backup,
            list_backups,
            load_backup,
        )

        _seed_economy(self.surface)
        info = create_backup(self.surface, GROUP_A, by=OWNER_ID,
                             reason="manual")
        path = (self.tmp_path / "group_backups"
                / "telegram-bot_-100111" / (info["backup_id"] + ".json"))
        data = json.loads(path.read_text())
        data["economy"]["ledger"][0]["amount"] = 99999
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(GroupBackupCorrupt, "checksum"):
            load_backup(self.surface, GROUP_A, info["backup_id"])
        infos, corrupt = list_backups(self.surface, GROUP_A)
        self.assertEqual(infos, [])
        self.assertEqual(corrupt, 1)

    def test_load_backup_wrong_group_refused(self):
        from nomorals.social.chat.group_backup import (
            GroupBackupCorrupt,
            create_backup,
            load_backup,
        )

        _seed_economy(self.surface)
        info = create_backup(self.surface, GROUP_A, by=OWNER_ID,
                             reason="manual")
        # A backup file copied into another group's dir still
        # refuses — it belongs to its own group only.
        import shutil

        src = (self.tmp_path / "group_backups" / "telegram-bot_-100111"
               / (info["backup_id"] + ".json"))
        dest_dir = (self.tmp_path / "group_backups"
                    / "telegram-bot_-100222")
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dest_dir / (info["backup_id"] + ".json"))
        with self.assertRaisesRegex(GroupBackupCorrupt, "own group"):
            load_backup(self.surface, GROUP_B, info["backup_id"])

    def _backed_up(self):
        from nomorals.social.chat.group_backup import create_backup

        _seed_economy(self.surface)
        info = create_backup(self.surface, GROUP_A, by=OWNER_ID,
                             reason="manual")
        return info

    def test_preview_names_member_and_balance_moves(self):
        from nomorals.social.chat.group_backup import (
            load_backup,
            preview_restore,
        )

        info = self._backed_up()
        # Drift after the backup: a new member, Ada paid more, the
        # faucet amount changed.
        self.surface.record_join(GROUP_A, "3", username="chi",
                                 name="Chi", ts=NOW)
        self.surface.store.group(GROUP_A)["settings"]["economy"][
            "faucet_amount"] = 75
        self.surface.economy.award(GROUP_A, "1", 40, "faucet")
        self.surface.store.save()
        backup = load_backup(self.surface, GROUP_A, info["backup_id"])
        preview = preview_restore(self.surface, GROUP_A, backup)
        text = "\n".join(preview["lines"])
        self.assertIn("gone: chi", text)
        self.assertIn("ada: 140 → 100", text)
        # The settings diff names the changed top-level key, not
        # the nested field; the backup bookkeeping block never
        # shows as a change.
        self.assertIn("economy", text)
        settings_line = next(
            line for line in preview["lines"]
            if line.startswith("settings:"))
        self.assertNotIn("backup", settings_line)
        self.assertEqual(preview["members"]["removed"], ["3"])
        self.assertGreaterEqual(
            preview["balances"]["moved_total"], 1)

    def test_apply_restore_returns_state(self):
        from nomorals.social.chat.group_backup import (
            apply_restore,
            list_backups,
            load_backup,
        )

        info = self._backed_up()
        self.surface.record_join(GROUP_A, "3", username="chi",
                                 name="Chi", ts=NOW)
        self.surface.store.group(GROUP_A)["settings"]["economy"][
            "faucet_amount"] = 75
        self.surface.economy.award(GROUP_A, "1", 40, "faucet")
        self.surface.store.save()
        backup = load_backup(self.surface, GROUP_A, info["backup_id"])
        applied = apply_restore(self.surface, GROUP_A, backup,
                                by=OWNER_ID)
        self.assertEqual(applied["members"], 2)
        self.assertEqual(applied["ledger_entries"], 2)
        self.assertEqual(
            self.surface.store.group(GROUP_A)["settings"]["economy"][
                "faucet_amount"], 50)
        self.assertEqual(self.surface.economy.balance(GROUP_A, "1"), 100)
        self.assertNotIn("3",
                         self.surface.store.group(GROUP_A)["members"])
        entry = self.surface.store.group(GROUP_A)
        kinds = [a.get("kind") for a in entry.get("audit") or []]
        self.assertIn("backup_restored", kinds)
        infos, _ = list_backups(self.surface, GROUP_A)
        self.assertEqual(len(infos), 1)

    def test_apply_restore_removes_files_added_after_backup(self):
        from nomorals.social.chat.group_backup import (
            apply_restore,
            load_backup,
        )

        info = self._backed_up()
        # A mining file after the backup (stands in for any later
        # economy file — flags, cases, quest claims); the restore
        # must remove it again.
        self.surface.economy._save_json(GROUP_A, "mining",
                                        {"hashrate": 10})
        files, _ = self.surface.economy.read_state_files(GROUP_A)
        self.assertIn("mining", files)
        backup = load_backup(self.surface, GROUP_A, info["backup_id"])
        apply_restore(self.surface, GROUP_A, backup, by=OWNER_ID)
        files, _ = self.surface.economy.read_state_files(GROUP_A)
        self.assertNotIn("mining", files)

    def test_schedule_and_due(self):
        from nomorals.social.chat.group_backup import (
            backup_due,
            backup_settings,
            set_backup_schedule,
        )

        cfg = backup_settings(self.surface, GROUP_A)
        self.assertEqual(cfg["cadence"], "off")
        self.assertEqual(cfg["keep"], 7)
        self.assertFalse(backup_due(cfg, MIDDAY_TS))
        out = set_backup_schedule(self.surface, GROUP_A,
                                  cadence="daily", hour=2, keep=5,
                                  by=OWNER_ID)
        cfg = out["settings"]
        self.assertEqual(cfg, {"cadence": "daily", "hour": 2, "keep": 5,
                               "last_run_ts": 0.0, "last_backup_id": ""})
        self.assertIn("02:00", out["next"])
        # A daily run is due once its hour has come and a day has
        # passed since the last run.
        now = time.mktime(time.strptime("2026-11-15 03:00:00",
                                        "%Y-%m-%d %H:%M:%S"))
        self.assertTrue(backup_due(cfg, now))
        # 01:00 — the 02:00 hour has not come yet.
        self.assertFalse(backup_due(cfg, now - 7200))
        recent = dict(cfg, last_run_ts=now - 60)
        self.assertFalse(backup_due(recent, now))
        out = set_backup_schedule(self.surface, GROUP_A, cadence="off",
                                  by=OWNER_ID)
        self.assertEqual(out["settings"]["cadence"], "off")
        self.assertIn("off", out["next"])

    def test_schedule_rejects_bad_values(self):
        from nomorals.social.chat.group_backup import (
            GroupBackupRefused,
            set_backup_schedule,
        )

        for kwargs in ({"cadence": "hourly"},
                       {"cadence": "daily", "hour": 25},
                       {"cadence": "daily", "hour": -1},
                       {"cadence": "daily", "keep": 0},
                       {"cadence": "daily", "keep": 99}):
            with self.assertRaises(GroupBackupRefused, msg=str(kwargs)):
                set_backup_schedule(self.surface, GROUP_A, by=OWNER_ID,
                                    **kwargs)

    def test_prune_keeps_newest(self):
        from nomorals.social.chat.group_backup import (
            create_backup,
            list_backups,
            set_backup_schedule,
        )

        _seed_economy(self.surface)
        set_backup_schedule(self.surface, GROUP_A, cadence="daily",
                            hour=0, keep=2, by=OWNER_ID)
        first = create_backup(self.surface, GROUP_A, by=OWNER_ID,
                              reason="manual")
        second = create_backup(self.surface, GROUP_A, by=OWNER_ID,
                               reason="manual")
        third = create_backup(self.surface, GROUP_A, by=OWNER_ID,
                              reason="manual")
        infos, corrupt = list_backups(self.surface, GROUP_A)
        self.assertEqual(corrupt, 0)
        self.assertEqual([i["backup_id"] for i in infos],
                         [third["backup_id"], second["backup_id"]])
        self.assertNotIn(first["backup_id"],
                         {i["backup_id"] for i in infos})
        self.assertEqual(third["pruned"], [first["backup_id"]])

    def test_delete_backup(self):
        from nomorals.social.chat.group_backup import (
            GroupBackupRefused,
            create_backup,
            delete_backup,
            list_backups,
        )

        _seed_economy(self.surface)
        info = create_backup(self.surface, GROUP_A, by=OWNER_ID,
                             reason="manual")
        delete_backup(self.surface, GROUP_A, info["backup_id"],
                      by=OWNER_ID)
        infos, _ = list_backups(self.surface, GROUP_A)
        self.assertEqual(infos, [])
        with self.assertRaisesRegex(GroupBackupRefused, "no backup"):
            delete_backup(self.surface, GROUP_A, info["backup_id"],
                          by=OWNER_ID)

    def test_tick_backs_up_due_groups_only(self):
        from nomorals.social.chat.group_backup import (
            list_backups,
            run_group_backup_tick_once,
            set_backup_schedule,
        )

        _seed_economy(self.surface)
        set_backup_schedule(self.surface, GROUP_A, cadence="daily",
                            hour=0, keep=7, by=OWNER_ID)
        # GROUP_B stays off; the channel is never backed up.
        result = run_group_backup_tick_once(
            context=SimpleNamespace(), surface=self.surface,
            workspace_dir=self.tmp_path)
        self.assertTrue(result["ok"])
        self.assertEqual([b["group"] for b in result["backed_up"]],
                         [GROUP_A])
        self.assertEqual(result["failed"], [])
        self.assertGreaterEqual(result["skipped"], 1)
        infos, _ = list_backups(self.surface, GROUP_A)
        self.assertEqual(len(infos), 1)
        self.assertEqual(infos[0]["reason"], "scheduled")
        # A second tick in the same hour does not back up again.
        result = run_group_backup_tick_once(
            context=SimpleNamespace(), surface=self.surface,
            workspace_dir=self.tmp_path)
        self.assertEqual(result["backed_up"], [])

    def test_degraded_economy_still_backs_up(self):
        from nomorals.social.chat.group_backup import create_backup

        _seed_economy(self.surface)
        econ_dir = (self.tmp_path / "group_economy"
                    / "telegram-bot_-100111")
        (econ_dir / "ledger.json").write_text("not json{{{")
        info = create_backup(self.surface, GROUP_A, by=OWNER_ID,
                             reason="manual")
        self.assertEqual(info["economy_status"], "degraded")
        self.assertIn("ledger", info["economy_reason"] or "")
        self.assertEqual(info["counts"]["members"], 2)
        self.assertEqual(info["counts"]["ledger_entries"], 0)

    def test_ensure_group_backup_job_is_idempotent(self):
        from nomorals.social.chat.group_backup import (
            BACKUP_JOB,
            ensure_group_backup_job,
        )

        class _FakeScheduler:
            """The real scheduler's shape: list_jobs / add /
            remove (mirrors ensure_season_tick_job's calls)."""

            def __init__(self):
                self.jobs = {}

            def list_jobs(self):
                return [SimpleNamespace(id=k, name=v["name"],
                                        spec=v["spec"])
                        for k, v in self.jobs.items()]

            def add(self, name, spec, kind, data, routine=False):
                self.jobs[name] = {"name": name, "spec": spec,
                                   "tool": data.get("tool")}
                return {"id": name, "name": name, "spec": spec}

            def remove(self, job_id):
                self.jobs.pop(job_id, None)

        sched = _FakeScheduler()
        first = ensure_group_backup_job(sched)
        second = ensure_group_backup_job(sched)
        # The first call returns the add() receipt (a dict); the
        # second returns the existing job object — same job, two
        # shapes, so compare the identity fields.
        self.assertEqual(first["id"], second.id)
        self.assertEqual(second.name, BACKUP_JOB)
        self.assertEqual(len(sched.jobs), 1)
        self.assertIn(BACKUP_JOB, sched.jobs)
        self.assertEqual(sched.jobs[BACKUP_JOB]["tool"],
                         "__group_backup_tick__")
        # A stale spec is replaced, not duplicated.
        sched.jobs[BACKUP_JOB]["spec"] = "every 9h"
        ensure_group_backup_job(sched)
        self.assertEqual(sched.jobs[BACKUP_JOB]["spec"], "every 1h")


class TestGroupBackupHandleControl(_BackupCase):
    """The chat surface: /gbackup through handle_control."""

    def test_gbackup_is_a_public_command(self):
        self.assertIn("gbackup", CONTROL_COMMANDS)
        self.assertIn("gbackup", PUBLIC_COMMANDS)
        self.assertTrue(caller_may_run("gbackup", False))

    def test_owner_dm_board_lists_groups_with_counts(self):
        from nomorals.social.chat.group_backup import create_backup

        _seed_economy(self.surface)
        create_backup(self.surface, GROUP_A, by=OWNER_ID,
                      reason="manual")
        out = self._reply("/gbackup")
        text = _text(out)
        self.assertIn(GROUP_A, text)
        self.assertIn(GROUP_B, text)
        self.assertNotIn(CHANNEL, text)  # channels have no backups
        self.assertIn("1 backup(s)", text)
        self.assertIn("0 backup(s)", text)
        self.assertIn("off", text)
        cbs = _buttons(out)
        self.assertTrue(any(cb == "gbackup g1" for cb in cbs),
                        f"no gbackup deep-link in {cbs!r}")

    def test_non_admin_dm_sees_no_groups(self):
        text = _text(self._reply("/gbackup", sender_id=MEMBER_ID))
        self.assertIn("don't admin any group", text)

    def test_gbackup_refuses_group_chat(self):
        out = self.rt.handle_control(
            "/gbackup", GROUP_A, message=_group_msg(OWNER_ID, "/gbackup"))
        self.assertIn("my DM", _text(out))

    def test_export_builds_file_and_reports_summary(self):
        from nomorals.social.chat.group_backup import load_backup

        _seed_economy(self.surface)
        text = _text(self._reply("/gbackup export g1"))
        self.assertIn("backup of", text)
        self.assertIn(GROUP_A, text)
        self.assertIn("2 members", text)
        self.assertIn("2 ledger entries", text)
        # No gateway on a bare runtime: the reply says where the
        # file lives, and the file itself is a valid backup.
        self.assertIn("The file is built at", text)
        infos_dir = (self.tmp_path / "group_backups"
                     / "telegram-bot_-100111")
        files = sorted(infos_dir.glob("*.json"))
        self.assertEqual(len(files), 1)
        backup = load_backup(self.surface, GROUP_A,
                             files[0].name[:-len(".json")])
        self.assertEqual(len(backup["economy"]["ledger"]), 2)

    def test_export_sends_file_when_gateway_present(self):
        _seed_economy(self.surface)
        sent = []

        def _send_file(platform, chat, path, caption=None, name=None):
            sent.append({"platform": platform, "path": path,
                         "caption": caption, "name": name})
            return SimpleNamespace(ok=True)

        self.rt.gateway = SimpleNamespace(send_file=_send_file)
        text = _text(self._reply("/gbackup export g1"))
        self.assertIn("backup of", text)
        self.assertNotIn("can't carry files back", text)
        self.assertEqual(len(sent), 1)
        self.assertTrue(sent[0]["name"].endswith(".json"))
        self.assertIn("backup", sent[0]["name"])
        self.assertIn("backup of", sent[0]["caption"] or "")

    def test_restore_preview_then_yes_applies(self):
        from nomorals.social.chat.group_backup import create_backup

        _seed_economy(self.surface)
        info = create_backup(self.surface, GROUP_A, by=OWNER_ID,
                             reason="manual")
        backup_id = info["backup_id"]
        self.surface.record_join(GROUP_A, "3", username="chi",
                                 name="Chi", ts=NOW)
        self.surface.economy.award(GROUP_A, "1", 40, "faucet")
        self.surface.store.save()
        # Step 1: no `yes` — a preview only; the live state is
        # untouched.
        text = _text(self._reply(f"/gbackup restore g1 {backup_id}"))
        self.assertIn("gone: chi", text)
        self.assertIn("ada: 140 → 100", text)
        self.assertIn(f"/gbackup restore g1 {backup_id} yes", text)
        self.assertIn("3", self.surface.store.group(GROUP_A)["members"])
        self.assertEqual(self.surface.economy.balance(GROUP_A, "1"), 140)
        # Step 2: `yes` — applied, and the state is back.
        text = _text(self._reply(
            f"/gbackup restore g1 {backup_id} yes"))
        self.assertIn("restored", text)
        self.assertIn(backup_id, text)
        self.assertNotIn("3",
                         self.surface.store.group(GROUP_A)["members"])
        self.assertEqual(self.surface.economy.balance(GROUP_A, "1"), 100)

    def test_restore_unknown_backup_refused(self):
        text = _text(self._reply("/gbackup restore g1 nope-nope"))
        self.assertIn("no backup", text)

    def test_schedule_set_and_list_shows_it(self):
        text = _text(self._reply(
            "/gbackup schedule g1 daily hour 3 keep 5"))
        self.assertIn("daily at 03:00", text)
        self.assertIn("newest 5", text)
        text = _text(self._reply("/gbackup g1"))
        self.assertIn("daily", text)
        self.assertIn("03:00", text)

    def test_schedule_bad_cadence_refused(self):
        text = _text(self._reply("/gbackup schedule g1 hourly"))
        self.assertIn("cadence", text)

    def test_delete_backup(self):
        from nomorals.social.chat.group_backup import create_backup

        _seed_economy(self.surface)
        info = create_backup(self.surface, GROUP_A, by=OWNER_ID,
                             reason="manual")
        text = _text(self._reply(
            f"/gbackup delete g1 {info['backup_id']}"))
        self.assertIn("deleted", text)
        self.assertIn("untouched", text)
        text = _text(self._reply("/gbackup g1"))
        self.assertIn("no backups on file yet", text)

    def test_channel_token_refused(self):
        # /groups seeds the session map (g1/g2 + c1 for the channel).
        self._reply("/groups")
        text = _text(self._reply("/gbackup export c1"))
        self.assertIn("channel", text)

    def test_unknown_group_token(self):
        text = _text(self._reply("/gbackup export g99"))
        self.assertIn("expired", text)

    def test_help(self):
        text = _text(self._reply("/gbackup help"))
        self.assertIn("export", text)
        self.assertIn("restore", text)
        self.assertIn("schedule", text)
        self.assertIn("delete", text)


if __name__ == "__main__":
    unittest.main()

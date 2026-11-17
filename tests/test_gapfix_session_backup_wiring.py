"""Encrypted WhatsApp session backup, wired into a real surface.

The seal/restore functions in ``nomorals.social.chat.baileys_guard``
existed with zero production callers (only the round-trip pin in
``tests/test_step9_bots.py``): if the Baileys auth dir was lost, the
number's session was gone with no sealed copy anywhere. These pins
drive the wiring:

* the daily backup run writes a sealed copy per linked WhatsApp
  account (vault passphrase, never a chat argument) and the report
  carries a "WhatsApp session" line naming the destination;
* ``/pair session backup`` writes a sealed copy now and reports the
  path; ``/pair session restore`` restores the newest sealed copy
  into the account's auth dir, refusing while the bridge is running;
* a run with no Baileys auth dir present skips the session line
  instead of failing the whole backup.

Tier: unit (fully offline — tmp dirs, fake vault / fake notifier /
fake pairing supervisor; no network, no real bridge).
"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.social.pairing import AccountRecord, PairingRegistry

ACC = "whatsapp:2348012345678"
DIGITS = "2348012345678"
PASSPHRASE = "session-seal-passphrase"


class FakeVault:
    """Minimal vault: store/get by (service, username)."""

    def __init__(self) -> None:
        self.creds: dict[tuple[str, str], str] = {}

    def store(self, service, username, password, **kwargs):
        self.creds[(service, username)] = password
        return SimpleNamespace(password=password)

    def get(self, service, username, *, mark_used=True):
        try:
            return SimpleNamespace(password=self.creds[(service, username)])
        except KeyError as exc:
            raise KeyError(f"no credential {service}/{username}") from exc


class FakeNotifier:
    """Captures Notifier.publish calls."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    def publish(self, kind, title, body="", *, critical=False, **kwargs):
        self.sent.append({"kind": kind, "title": title, "body": body,
                          "critical": critical})
        return {"delivered": True, "id": "n1"}


def _make_root(tmp: Path) -> Path:
    """Minimal home the snapshot accepts: the required database."""
    root = tmp / "root"
    db = root / "data" / "nomorals.db"
    db.parent.mkdir(parents=True)
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, v TEXT)")
        conn.execute("INSERT INTO items (v) VALUES ('hello')")
        conn.commit()
    finally:
        conn.close()
    return root


def _seed_workspace(tmp: Path, *, with_creds: bool = True) -> Path:
    """A workspace with one linked WhatsApp account in the registry.

    Returns the workspace dir. When ``with_creds`` is False the
    record exists but its creds dir is absent from disk (the
    no-auth-dir case: the run must skip, not fail).
    """
    ws = tmp / "workspace"
    registry = PairingRegistry(
        ws / "data" / "social" / "paired_accounts.json")
    creds = ws / "data" / "whatsapp_sessions" / f"wa-{DIGITS}"
    registry.upsert(AccountRecord(
        id=ACC, platform="whatsapp", identity=DIGITS,
        creds_dir=str(creds), port=8787, status="linked",
        created_at=1.0))
    if with_creds:
        creds.mkdir(parents=True)
        (creds / "whatsmeow.db").write_bytes(b"fake-session-bytes")
        (creds / "auth.json").write_text('{"key": "secret"}')
    return ws


def _vault_with_session_passphrase() -> FakeVault:
    from nomorals.backup import generate_backup_key, store_backup_key
    from nomorals.backup.protect import store_session_passphrase

    vault = FakeVault()
    store_backup_key(vault, generate_backup_key())
    store_session_passphrase(vault, PASSPHRASE)
    return vault


def _coordinator(root: Path, work: Path, ws: Path, vault: FakeVault,
                 notifier: FakeNotifier):
    from nomorals.backup import BackupCoordinator

    return BackupCoordinator(
        root=root, work_dir=work, vault=vault, connector=None,
        notifier=notifier, session_workspace_dir=ws)


def _runtime(tmp: Path, *, running: bool):
    """A RuntimeProvisionMixin shell over a stubbed supervisor."""
    from nomorals.agents.partner.runtime_provision import RuntimeProvisionMixin

    ws = _seed_workspace(tmp)
    registry = PairingRegistry(
        ws / "data" / "social" / "paired_accounts.json")
    rt = object.__new__(RuntimeProvisionMixin)
    rt._pair_supervisor = SimpleNamespace(
        registry=registry, is_running=lambda account_id: running)
    rt.context = SimpleNamespace(
        db=None,
        settings=SimpleNamespace(workspace_dir=str(ws)))
    return rt, ws


class TestDailyBackupSessionCopy(unittest.TestCase):
    def test_daily_backup_writes_sealed_copy_that_round_trips(self):
        from nomorals.social.chat.baileys_guard import restore_session_encrypted

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            root = _make_root(tmp)
            ws = _seed_workspace(tmp)
            vault = _vault_with_session_passphrase()
            notifier = FakeNotifier()
            coord = _coordinator(root, tmp / "work", ws, vault, notifier)
            result = coord.run_backup(label="test")
            sealed = result["whatsapp_session"]["written"][0]["path"]
            sealed_path = Path(sealed)
            self.assertTrue(sealed_path.is_file())
            self.assertEqual(sealed_path.stat().st_mode & 0o777, 0o600)
            self.assertNotIn(b"secret", sealed_path.read_bytes())
            dest = restore_session_encrypted(
                sealed_path, tmp / "restored", PASSPHRASE)
            self.assertEqual((dest / "auth.json").read_text(),
                             '{"key": "secret"}')
            bodies = [s["body"] for s in notifier.sent]
            self.assertTrue(any("WhatsApp session" in b for b in bodies))
            self.assertTrue(any(sealed in b for b in bodies))
            self.assertFalse(any(PASSPHRASE in b for b in bodies))

    def test_wrong_passphrase_refuses_restore(self):
        from nomorals.social.chat.baileys_guard import (
            backup_session_encrypted,
            restore_session_encrypted,
        )

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = tmp / "session"
            src.mkdir()
            (src / "auth.json").write_text('{"key": "secret"}')
            backup = backup_session_encrypted(
                src, tmp / "bak.json", PASSPHRASE)
            with self.assertRaises(ValueError):
                restore_session_encrypted(backup, tmp / "bad", "wrong-pw")

    def test_daily_backup_skips_session_line_without_auth_dir(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            root = _make_root(tmp)
            ws = _seed_workspace(tmp, with_creds=False)
            vault = _vault_with_session_passphrase()
            notifier = FakeNotifier()
            coord = _coordinator(root, tmp / "work", ws, vault, notifier)
            result = coord.run_backup(label="test")
            self.assertIn("backup_id", result)
            self.assertTrue(
                result["whatsapp_session"]["skipped"],
                "no auth dir on disk must skip, not fail")
            bodies = [s["body"] for s in notifier.sent]
            self.assertFalse(any("WhatsApp session" in b for b in bodies))

    def test_daily_backup_skips_session_without_vault_passphrase(self):
        from nomorals.backup import generate_backup_key, store_backup_key

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            root = _make_root(tmp)
            ws = _seed_workspace(tmp)
            vault = FakeVault()
            store_backup_key(vault, generate_backup_key())
            notifier = FakeNotifier()
            coord = _coordinator(root, tmp / "work", ws, vault, notifier)
            result = coord.run_backup(label="test")
            self.assertIn("backup_id", result)
            self.assertTrue(
                result["whatsapp_session"]["skipped"],
                "no session passphrase in the vault must skip, not fail")
            bodies = [s["body"] for s in notifier.sent]
            self.assertFalse(any("WhatsApp session" in b for b in bodies))


class TestPairSessionVerbs(unittest.TestCase):
    def _patch_vault(self, vault):
        patcher = mock.patch(
            "nomorals.accounts.vault.open_vault", return_value=vault)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_pair_session_backup_reports_path(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            rt, ws = _runtime(tmp, running=False)
            self._patch_vault(_vault_with_session_passphrase())
            out = rt._control_pair("session backup")
            sealed = (ws / "data" / "whatsapp_sessions"
                      / f"wa-{DIGITS}-session-backup.json")
            self.assertTrue(sealed.is_file(), "sealed copy must exist")
            self.assertIn(str(sealed), out)
            self.assertNotIn(PASSPHRASE, out)

    def test_pair_session_restore_refuses_while_bridge_running(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            rt, ws = _runtime(tmp, running=True)
            self._patch_vault(_vault_with_session_passphrase())
            # Seal first (bridge stopped for the backup), then the
            # bridge comes up and a restore must refuse.
            rt._pair_supervisor.is_running = lambda account_id: False
            rt._control_pair("session backup")
            rt._pair_supervisor.is_running = lambda account_id: True
            creds = (ws / "data" / "whatsapp_sessions" / f"wa-{DIGITS}")
            before = (creds / "auth.json").read_bytes()
            out = rt._control_pair("session restore")
            self.assertIn("running", out.lower())
            self.assertIn("stop", out.lower())
            self.assertEqual((creds / "auth.json").read_bytes(), before,
                             "a refused restore must not touch the session")

    def test_pair_session_restore_succeeds_when_stopped(self):
        from nomorals.social.chat.baileys_guard import restore_session_encrypted

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            rt, ws = _runtime(tmp, running=False)
            self._patch_vault(_vault_with_session_passphrase())
            rt._control_pair("session backup")
            creds = (ws / "data" / "whatsapp_sessions" / f"wa-{DIGITS}")
            # The auth dir is lost: wipe it, then restore.
            (creds / "auth.json").unlink()
            (creds / "whatsmeow.db").unlink()
            out = rt._control_pair("session restore")
            self.assertIn("restored", out.lower())
            self.assertEqual((creds / "auth.json").read_text(),
                             '{"key": "secret"}')
            # The restored copy round-trips through the same function.
            sealed = (ws / "data" / "whatsapp_sessions"
                      / f"wa-{DIGITS}-session-backup.json")
            dest = restore_session_encrypted(
                sealed, tmp / "check", PASSPHRASE)
            self.assertEqual((dest / "auth.json").read_text(),
                             '{"key": "secret"}')

    def test_pair_session_status_names_sealed_state(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            rt, ws = _runtime(tmp, running=False)
            self._patch_vault(_vault_with_session_passphrase())
            out = rt._control_pair("session")
            self.assertIn("no sealed copy", out.lower())
            rt._control_pair("session backup")
            out = rt._control_pair("session")
            self.assertIn("sealed copy", out.lower())
            self.assertIn(f"wa-{DIGITS}-session-backup.json", out)


if __name__ == "__main__":
    unittest.main()

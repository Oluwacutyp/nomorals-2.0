"""Tests for the GitHub backup coordinator (nomorals/backup).

Tier: unit (fully offline — tmp dirs, real local SQLite files, fake
GitHub connector / fake vault / fake notifier; no network, no real
GitHub, no real tokens).
"""

from __future__ import annotations

import base64
import contextlib
import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.backup import (
    BackupCoordinator,
    BackupKeyError,
    ChecksumMismatchError,
    CorruptBackupError,
    TokenExpiredError,
    TokenScopeError,
    build_manifest,
    build_plan,
    check_token_scopes,
    ensure_repo,
    generate_backup_key,
    load_backup_key,
    load_manifest,
    select_keepers,
    store_backup_key,
    take_snapshot,
)
from nomorals.backup.errors import BackupError
from nomorals.backup.manifest import verify_manifest_files, write_manifest
from nomorals.backup.policy import prune_github_releases
from nomorals.backup.protect import (
    decrypt_bytes,
    decrypt_file,
    encrypt_bytes,
    join_parts,
    protect_snapshot,
    split_encrypted,
)
from nomorals.backup.ship import Outbox, ship_snapshot
from nomorals.backup.verify import drill


class FakeGHError(Exception):
    def __init__(self, message: str, *, status_code: int = 0) -> None:
        super().__init__(message)
        self.status_code = status_code


class FakeResp:
    def __init__(self, status: int, body: bytes = b"",
                 headers: dict | None = None) -> None:
        self.status = status
        self.body = body
        self.headers = headers or {}
        self.text = body.decode("utf-8", errors="replace")

    def json(self):
        return json.loads(self.body.decode("utf-8"))


class FakeHttp:
    """Stands in for HttpClient for release-asset upload/download."""

    def __init__(self, connector: "FakeGitHub") -> None:
        self.connector = connector

    def request(self, method: str, url: str, **kwargs):
        if method == "POST" and "uploads.github.com" in url:
            data = kwargs.get("data", b"")
            if self.connector.fail_upload == "network":
                raise ConnectionError("Connection timed out")
            if self.connector.fail_upload == "partial":
                # Server claims a different size than we sent.
                return FakeResp(201, json.dumps(
                    {"id": 7, "size": len(data) + 1}).encode())
            return FakeResp(201, json.dumps(
                {"id": 7, "size": len(data)}).encode())
        if method == "GET":
            return FakeResp(200, b"asset-bytes")
        raise AssertionError(f"unexpected http {method} {url}")


class FakeGitHub:
    """Duck-typed GitHubConnector double (no network)."""

    def __init__(self) -> None:
        self.repos: dict[str, dict] = {}
        self.created: list[str] = []
        self.releases: list[dict] = []
        self.deleted_releases: list[int] = []
        self.pushed: list[dict] = []
        self.fail_upload: str = ""
        self.fail_create: FakeGHError | None = None
        self.fail_get_repo: FakeGHError | None = None
        self.fail_release: FakeGHError | None = None
        self.http = FakeHttp(self)
        self.credential = SimpleNamespace(
            username="owner", password="tok",
            metadata={"scopes": ["fine-grained (see token settings)"]},
        )

    # — connector surface used by ship.py —
    def _api(self, method: str, path: str, payload=None, *, params=None):
        if path == "/user":
            return {"login": "owner", "id": 1}
        if method == "GET" and path.endswith("/releases"):
            return list(self.releases)
        if method == "GET" and "/releases/tags/" in path:
            tag = path.rsplit("/", 1)[-1]
            for rel in self.releases:
                if rel.get("tag_name") == tag:
                    return rel
            raise FakeGHError("not found", status_code=404)
        if method == "DELETE" and "/releases/" in path:
            rid = int(path.rsplit("/", 1)[-1])
            self.deleted_releases.append(rid)
            return {}
        raise AssertionError(f"unexpected _api {method} {path}")

    def _require_credential(self):
        return self.credential

    def _load_credential(self):
        return self.credential

    def _headers(self, token: str):
        return {"Authorization": f"Bearer {token}"}

    @contextlib.contextmanager
    def _askpass_env(self, token):
        yield {"GIT_ASKPASS": "fake"}

    def get_repo(self, full_name: str):
        if self.fail_get_repo is not None:
            raise self.fail_get_repo
        if full_name in self.repos:
            return self.repos[full_name]
        raise FakeGHError("Not Found", status_code=404)

    def create_repo(self, name: str, *, private=True, description="",
                    auto_init=False):
        if self.fail_create is not None:
            raise self.fail_create
        full = f"owner/{name}"
        repo = {"full_name": full, "private": private}
        self.repos[full] = repo
        self.created.append(full)
        return repo

    def create_release(self, full_name: str, tag: str, *, name="",
                       body="", draft=False, prerelease=False):
        if self.fail_release is not None:
            raise self.fail_release
        for rel in self.releases:
            if rel.get("tag_name") == tag:
                raise FakeGHError("already_exists", status_code=422)
        release = {"id": len(self.releases) + 1, "tag_name": tag,
                   "upload_url": "https://uploads.github.com/repos/o/n"
                   "/releases/1/assets{?name,label}",
                   "created_at": time.strftime(
                       "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        self.releases.append(release)
        return release

    def push_code(self, local_dir, full_name, *, branch="main",
                  remote_name="origin", timeout=300.0):
        files = sorted(str(p.relative_to(local_dir))
                       for p in Path(local_dir).rglob("*") if p.is_file())
        self.pushed.append({"dir": str(local_dir), "repo": full_name,
                            "files": files})
        return {"pushed": True, "branch": branch}


class FakeVault:
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
    def __init__(self) -> None:
        self.sent: list[dict] = []

    def publish(self, kind, title, body="", *, critical=False, **kwargs):
        self.sent.append({"kind": kind, "title": title, "body": body,
                          "critical": critical})
        return {"delivered": True, "id": "n1"}


def _make_db(path: Path, *, rows: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, v TEXT)")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER)")
        conn.execute("INSERT INTO schema_migrations VALUES (86)")
        for i in range(rows):
            conn.execute("INSERT INTO items VALUES (?, ?)", (i, f"v{i}"))
        conn.commit()
    finally:
        conn.close()


def _make_root(tmp: Path) -> Path:
    root = tmp / "root"
    _make_db(root / "data" / "nomorals.db")
    (root / "config").mkdir(parents=True)
    (root / "config" / "settings.json").write_text('{"a": 1}')
    (root / "memory").mkdir(parents=True)
    (root / "memory" / "notes.md").write_text("hello")
    (root / "cache").mkdir(parents=True)
    (root / "cache" / "junk.tmp").write_text("x")
    (root / "llm").mkdir(parents=True)
    (root / "llm" / "model.gguf").write_bytes(b"fake-weights")
    (root / "vault.json").write_text('{"secret": true}')
    wal = root / "data" / "nomorals.db-wal"
    wal.write_bytes(b"wal-sidecar")
    return root


class TestEnsureRepo(unittest.TestCase):
    def test_create_when_missing_then_reuse_idempotent(self):
        gh = FakeGitHub()
        first = ensure_repo(gh)
        self.assertTrue(first["created"])
        second = ensure_repo(gh)
        self.assertFalse(second["created"])
        self.assertEqual(gh.created, ["owner/nomorals-backup"])

    def test_public_existing_repo_refused(self):
        gh = FakeGitHub()
        gh.repos["owner/nomorals-backup"] = {"full_name": "owner/nomorals-backup",
                                             "private": False}
        with self.assertRaises(BackupError):
            ensure_repo(gh)

    def test_401_maps_to_token_expired(self):
        gh = FakeGitHub()
        gh.fail_get_repo = FakeGHError("bad creds", status_code=401)
        with self.assertRaises(TokenExpiredError):
            ensure_repo(gh)

    def test_403_create_maps_to_scope_error_naming_administration(self):
        gh = FakeGitHub()
        gh.fail_create = FakeGHError("Resource not accessible", status_code=403)
        with self.assertRaises(TokenScopeError) as ctx:
            ensure_repo(gh)
        self.assertIn("Administration", str(ctx.exception))

    def test_scopes_honest_about_fine_grained_unknown(self):
        gh = FakeGitHub()
        info = check_token_scopes(gh)
        self.assertTrue(info["fine_grained"])
        self.assertIsNone(info["can_create_repo"])


class TestProtect(unittest.TestCase):
    def test_encrypt_round_trip_and_wrong_key_fails(self):
        key = generate_backup_key()
        blob = encrypt_bytes(b"session-cookie-secret", key)
        self.assertEqual(decrypt_bytes(blob, key), b"session-cookie-secret")
        with self.assertRaises(Exception):  # noqa: BLE001 - CipherError
            decrypt_bytes(blob, generate_backup_key())

    def test_key_vault_round_trip_and_missing_key(self):
        vault = FakeVault()
        with self.assertRaises(BackupKeyError) as ctx:
            load_backup_key(vault)
        self.assertIn("lose this key", str(ctx.exception))
        key = generate_backup_key()
        store_backup_key(vault, key)
        self.assertEqual(load_backup_key(vault), key)

    def test_split_and_join(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "blob.enc"
            path.write_bytes(b"x" * 2500)
            parts = split_encrypted(path, max_bytes=1000)
            self.assertEqual(len(parts), 3)
            out = Path(td) / "joined"
            join_parts(parts, out)
            self.assertEqual(out.read_bytes(), b"x" * 2500)
            small = Path(td) / "small.enc"
            small.write_bytes(b"y" * 10)
            self.assertEqual(split_encrypted(small, max_bytes=1000), [small])


class TestPlanManifest(unittest.TestCase):
    def test_plan_excludes_sidecars_weights_and_marks_vault(self):
        with tempfile.TemporaryDirectory() as td:
            root = _make_root(Path(td))
            plan = build_plan(root)
            rels = [f["rel"] for f in plan.files]
            self.assertNotIn("data/nomorals.db-wal", rels)
            self.assertNotIn("cache/junk.tmp", rels)
            self.assertNotIn("llm/model.gguf", rels)
            self.assertEqual(
                [v["rel"] for v in plan.vault_files], ["vault.json"])
            self.assertEqual(
                plan.llm_manifest["models"][0]["name"], "llm/model.gguf")
            self.assertTrue(any(
                d["rel"] == "data/nomorals.db" and d["status"] == "present"
                for d in plan.databases))

    def test_manifest_round_trip_and_schema_reject(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = build_manifest(
                backup_id="backup-20260101-000000", protected=[])
            path = write_manifest(Path(td), manifest)
            self.assertEqual(load_manifest(path)["backup_id"],
                             "backup-20260101-000000")
            bad = dict(manifest, schema_version=99)
            bad_path = Path(td) / "bad.json"
            bad_path.write_text(json.dumps(bad))
            with self.assertRaises(BackupError):
                load_manifest(bad_path)


class TestSnapshotDrill(unittest.TestCase):
    def test_snapshot_drill_pass_then_tamper_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = _make_root(Path(td))
            work = Path(td) / "work"
            snap = take_snapshot(root, work)
            db_entries = [e for e in snap.entries if e.kind == "database"]
            self.assertEqual(db_entries[0].tables, 2)  # items + schema_migrations
            self.assertEqual(db_entries[0].schema_version, 86)
            key = generate_backup_key()
            protected = protect_snapshot(snap, work / "protected", key)
            manifest = build_manifest(
                backup_id=snap.backup_id, created_at=snap.created_at,
                protected=protected, plan=snap.plan)
            mpath = write_manifest(work / snap.backup_id, manifest)
            report = drill(mpath, work / "protected", key)
            self.assertTrue(report.ok, report.summary())
            # Tamper: flip one ciphertext byte -> hash step fails.
            part = Path(protected[0].parts[0])
            data = bytearray(part.read_bytes())
            data[10] ^= 0xFF
            part.write_bytes(bytes(data))
            failed = drill(mpath, work / "protected", key)
            self.assertFalse(failed.ok)
            self.assertTrue(any(
                "hash" in s.name or "decrypt" in s.name
                for s in failed.steps if not s.ok))

    def test_corrupt_db_snapshot_refused(self):
        from nomorals.backup.snapshot import snapshot_sqlite_file

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "garbage.db"
            src.write_bytes(b"this is not sqlite at all")
            with self.assertRaises(CorruptBackupError):
                snapshot_sqlite_file(src, Path(td) / "out.db")


class TestPolicy(unittest.TestCase):
    def test_keep_last_daily_monthly_union(self):
        now = time.time()
        entries = []
        for days_ago in range(45):
            entries.append({"backup_id": f"backup-d{days_ago}",
                            "created_at": now - days_ago * 86400})
        # Two in the same day: only the newest survives the daily rule.
        entries.append({"backup_id": "backup-dup-old",
                        "created_at": now - 10 * 86400 - 3600})
        # Newest in its month, 200 days back: monthly rule keeps it.
        entries.append({"backup_id": "backup-old-month",
                        "created_at": now - 200 * 86400})
        # 400 days back: outside every window.
        entries.append({"backup_id": "backup-ancient",
                        "created_at": now - 400 * 86400})
        keepers = select_keepers(entries, now=now)
        for i in range(7):
            self.assertIn(f"backup-d{i}", keepers)
        self.assertIn("backup-old-month", keepers)
        self.assertNotIn("backup-ancient", keepers)
        self.assertNotIn("backup-dup-old", keepers)

    def test_prune_github_releases_deletes_only_pruned(self):
        gh = FakeGitHub()
        now = time.time()
        import datetime

        def _rel(tag: str, days: float) -> dict:
            return {"id": len(gh.releases) + 1, "tag_name": tag,
                    "created_at": datetime.datetime.fromtimestamp(
                        now - days * 86400,
                        tz=datetime.timezone.utc).strftime(
                            "%Y-%m-%dT%H:%M:%SZ")}

        gh.releases = [_rel(f"backup-fresh-{i}", 0)
                       for i in range(7)]
        gh.releases += [_rel("backup-ancient", 400),
                        _rel("not-a-backup", 400)]
        # Re-number ids sequentially so the delete assertion is exact.
        for i, rel in enumerate(gh.releases, start=1):
            rel["id"] = i
        result = prune_github_releases(gh, "owner/nomorals-backup", now=now)
        self.assertIn("backup-ancient", result["deleted"])
        self.assertNotIn("not-a-backup", result["deleted"])
        self.assertEqual(gh.deleted_releases, [8])


class TestShipFailureModes(unittest.TestCase):
    def test_network_down_queues_never_silent(self):
        gh = FakeGitHub()
        gh.fail_release = ConnectionError("Connection timed out")
        manifest = build_manifest(backup_id="backup-20260101-000000",
                                  protected=[])
        with tempfile.TemporaryDirectory() as td:
            result = ship_snapshot(gh, "owner/nomorals-backup", manifest, [],
                                   outbox_dir=Path(td) / "outbox")
            self.assertFalse(result["shipped"])
            self.assertTrue(result["queued"])
            self.assertEqual(Outbox(Path(td) / "outbox").depth(), 1)

    def test_partial_upload_is_failure_not_partial(self):
        gh = FakeGitHub()
        gh.fail_upload = "partial"
        manifest = build_manifest(backup_id="backup-20260101-000000",
                                  protected=[])
        with tempfile.TemporaryDirectory() as td:
            blob = Path(td) / "blob.gz.enc"
            blob.write_bytes(b"ciphertext")
            with self.assertRaises(ChecksumMismatchError):
                ship_snapshot(gh, "owner/nomorals-backup", manifest, [blob])

    def test_release_reuse_on_422_retry(self):
        gh = FakeGitHub()
        manifest = build_manifest(backup_id="backup-20260101-000000",
                                  protected=[])
        # First attempt created the release; second attempt reuses it.
        gh.create_release("owner/nomorals-backup", "backup-20260101-000000")
        with tempfile.TemporaryDirectory() as td:
            blob = Path(td) / "blob.gz.enc"
            blob.write_bytes(b"ciphertext")
            result = ship_snapshot(gh, "owner/nomorals-backup", manifest,
                                   [blob])
            self.assertTrue(result["shipped"])
            self.assertEqual(len(gh.releases), 1)


class TestCoordinatorEndToEnd(unittest.TestCase):
    def test_run_backup_ships_manifest_only_and_drill_passes(self):
        with tempfile.TemporaryDirectory() as td:
            root = _make_root(Path(td))
            work = Path(td) / "work"
            vault = FakeVault()
            store_backup_key(vault, generate_backup_key())
            gh = FakeGitHub()
            notifier = FakeNotifier()
            coord = BackupCoordinator(
                root=root, work_dir=work, vault=vault, connector=gh,
                notifier=notifier,
            )
            result = coord.run_backup()
            self.assertTrue(result["shipped"])
            self.assertTrue(result["manifest_pushed"])
            self.assertEqual(len(gh.pushed), 1)
            pushed_files = gh.pushed[0]["files"]
            self.assertTrue(all(
                not f.endswith((".enc", ".gz", ".db", ".db-wal"))
                for f in pushed_files), pushed_files)
            self.assertTrue(any("manifest.json" in f for f in pushed_files))
            self.assertTrue(any("RESTORE.md" in f for f in pushed_files))
            self.assertTrue(any(
                n.title.startswith("backup shipped") for n in
                (SimpleNamespace(**s) for s in notifier.sent)))
            drill_report = coord.run_drill()
            self.assertTrue(drill_report.ok, drill_report.summary())

    def test_no_connector_queues_locally(self):
        with tempfile.TemporaryDirectory() as td:
            root = _make_root(Path(td))
            vault = FakeVault()
            store_backup_key(vault, generate_backup_key())
            coord = BackupCoordinator(root=root, work_dir=Path(td) / "work",
                                      vault=vault, connector=None,
                                      notifier=FakeNotifier())
            result = coord.run_backup()
            self.assertTrue(result["queued"])
            self.assertFalse(result["shipped"])

    def test_schedule_specs_register(self):
        from nomorals.backup.coordinator import (
            register_jobs,
            schedule_specs,
        )
        from nomorals.scheduler.unified_scheduler import UnifiedJob

        specs = schedule_specs()
        self.assertEqual(
            [s["job_id"] for s in specs],
            ["github-backup-daily", "github-backup-drill-monthly"])
        for spec in specs:  # recurrence format validates eagerly
            UnifiedJob(job_id=spec["job_id"], kind="cron",
                       recurrence=spec["recurrence"],
                       payload=spec["payload"], name=spec["name"])

        class FakeFacade:
            def __init__(self) -> None:
                self.jobs: list = []

            def add_job(self, job):
                self.jobs.append(job)
                return job

        facade = FakeFacade()
        installed = register_jobs(facade)
        self.assertTrue(all(i["installed"] for i in installed))
        self.assertEqual(len(facade.jobs), 2)


if __name__ == "__main__":
    unittest.main()

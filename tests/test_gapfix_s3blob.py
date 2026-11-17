"""S3 offsite blob store — the 4th backup copy, wired in (audit R3-58).

Failing-first: the ship-side machinery already exists
(`BackupCoordinator.s3_store` + `ship_to_blob_store` in
`run_backup`), but no production builder ever sets `s3_store` and
`open_blob_store` has zero production callers — so the 4th copy
never uploads. This file pins the missing link: `open_s3_store()`
in the coordinator, both coordinator builders passing it through,
the scheduled tick surfacing the outcome, and the shipped report
naming the copy. No live S3 endpoint is ever touched: construction
is network-free, uploads go to fake stores.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.backup import generate_backup_key, store_backup_key
from nomorals.backup.coordinator import BackupCoordinator
from nomorals.storage.s3blob import S3BlobStore

_S3_KEYS = (
    "S3_ENDPOINT", "S3_BUCKET", "S3_ACCESS_KEY", "S3_SECRET_KEY",
    "S3_REGION", "S3_PREFIX", "S3_PATH_STYLE", "S3_TIMEOUT",
)

_FULL_ENV = {
    "S3_ENDPOINT": "https://s3.example.test:9000",
    "S3_BUCKET": "devon-backups",
    "S3_ACCESS_KEY": "test-access",
    "S3_SECRET_KEY": "test-secret",
}


@contextlib.contextmanager
def _s3_env(**overrides):
    """S3 env with exactly the given keys set (all others removed)."""
    saved = {k: os.environ.pop(k) for k in _S3_KEYS if k in os.environ}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key in list(overrides):
            os.environ.pop(key, None)
        os.environ.update(saved)


class _FakeS3Store:
    """Records put_file calls; returns the file's real SHA-256."""

    def __init__(self) -> None:
        self.puts: list[dict] = []

    def put_file(self, path, mime="application/octet-stream"):
        data = Path(path).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        self.puts.append({"file": Path(path).name, "mime": mime,
                          "sha256": digest, "size": len(data)})
        return SimpleNamespace(sha256=digest, size=len(data))


class _FailingStore:
    def put_file(self, path, mime="application/octet-stream"):
        raise ConnectionError("boom: endpoint unreachable")


class _FakeVault:
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


class _FakeNotifier:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    def publish(self, kind, title, body="", *, critical=False, **kwargs):
        self.sent.append({"kind": kind, "title": title, "body": body,
                          "critical": critical})
        return {"delivered": True, "id": "n1"}


class _FakeGHError(Exception):
    def __init__(self, message, *, status_code=0):
        super().__init__(message)
        self.status_code = status_code


class _FakeGitHub:
    """Minimal duck-typed GitHubConnector for run_backup (no network)."""

    def __init__(self) -> None:
        self.repos: dict[str, dict] = {}
        self.releases: list[dict] = []
        self.pushed: list[dict] = []
        self.credential = SimpleNamespace(
            username="owner", password="fake-token",
            metadata={"scopes": ["fine-grained (see token settings)"]},
        )

    def _api(self, method, path, payload=None, *, params=None):
        if path == "/user":
            return {"login": "owner", "id": 1}
        if method == "GET" and path.endswith("/releases"):
            return list(self.releases)
        if method == "GET" and "/releases/tags/" in path:
            tag = path.rsplit("/", 1)[-1]
            for rel in self.releases:
                if rel.get("tag_name") == tag:
                    return rel
            raise _FakeGHError("not found", status_code=404)
        raise AssertionError(f"unexpected _api {method} {path}")

    def _require_credential(self):
        return self.credential

    def _load_credential(self):
        return self.credential

    def _headers(self, token):
        return {"Authorization": f"Bearer {token}"}

    @contextlib.contextmanager
    def _askpass_env(self, token):
        yield {"GIT_ASKPASS": "fake"}

    def get_repo(self, full_name):
        if full_name in self.repos:
            return self.repos[full_name]
        raise _FakeGHError("Not Found", status_code=404)

    def create_repo(self, name, *, private=True, description="",
                    auto_init=False):
        full = f"owner/{name}"
        repo = {"full_name": full, "private": private}
        self.repos[full] = repo
        return repo

    def create_release(self, full_name, tag, *, name="", body="",
                       draft=False, prerelease=False):
        release = {"id": len(self.releases) + 1, "tag_name": tag,
                   "upload_url": "https://uploads.github.com/repos/o/n"
                   "/releases/1/assets{?name,label}"}
        self.releases.append(release)
        return release

    def upload_release_asset(self, release, path, *, label=""):
        return {"id": 7, "size": Path(path).stat().st_size}

    def push_code(self, local_dir, full_name, *, branch="main",
                  remote_name="origin", timeout=300.0):
        files = sorted(str(p.relative_to(local_dir))
                       for p in Path(local_dir).rglob("*") if p.is_file())
        self.pushed.append({"dir": str(local_dir), "repo": full_name,
                            "files": files})
        return {"pushed": True, "branch": branch}


def _make_root(tmp: Path) -> Path:
    root = tmp / "root"
    db_path = root / "data" / "nomorals.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, v TEXT)")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER)")
        conn.execute("INSERT INTO schema_migrations VALUES (86)")
        for i in range(3):
            conn.execute("INSERT INTO items VALUES (?, ?)", (i, f"v{i}"))
        conn.commit()
    finally:
        conn.close()
    (root / "config").mkdir(parents=True)
    (root / "config" / "settings.json").write_text('{"a": 1}')
    (root / "memory").mkdir(parents=True)
    (root / "memory" / "notes.md").write_text("hello")
    return root


class OpenS3StoreTests(unittest.TestCase):
    def test_no_env_returns_none(self):
        from nomorals.backup.coordinator import open_s3_store

        with _s3_env():
            self.assertIsNone(open_s3_store())

    def test_full_env_returns_s3_store(self):
        from nomorals.backup.coordinator import open_s3_store

        with _s3_env(**_FULL_ENV):
            store = open_s3_store()
        self.assertIsInstance(store, S3BlobStore)
        self.assertEqual(store.config.endpoint,
                         "https://s3.example.test:9000")
        self.assertEqual(store.config.bucket, "devon-backups")

    def test_partial_env_returns_none(self):
        from nomorals.backup.coordinator import open_s3_store

        partial = {k: v for k, v in _FULL_ENV.items()
                   if k != "S3_SECRET_KEY"}
        with _s3_env(**partial):
            self.assertIsNone(open_s3_store())

    def test_bad_timeout_returns_none_not_raise(self):
        from nomorals.backup.coordinator import open_s3_store

        with _s3_env(**_FULL_ENV, S3_TIMEOUT="not-a-number"):
            self.assertIsNone(open_s3_store())


class BuilderTests(unittest.TestCase):
    def _fake_ctx(self, tmp: Path):
        return SimpleNamespace(
            settings=SimpleNamespace(
                home_path=str(tmp / "root"),
                backup_dir=str(tmp / "work")),
            db=object(),
        )

    def _patch_vault_and_github(self):
        vault = _FakeVault()
        store_backup_key(vault, generate_backup_key())

        class _FakeConnector:
            def __init__(self, vault_arg):
                self.vault = vault_arg

            def _load_credential(self):
                return None

        patches = [
            mock.patch("nomorals.accounts.vault.open_vault",
                       return_value=vault),
            mock.patch("nomorals.connectors.github.GitHubConnector",
                       _FakeConnector),
            mock.patch("nomorals.agents.notifier.Notifier",
                       return_value=_FakeNotifier()),
        ]
        return patches

    def test_tick_builder_passes_s3_store_when_configured(self):
        from nomorals.tools.backup import _build_coordinator

        with (
            tempfile.TemporaryDirectory() as td,
            _s3_env(**_FULL_ENV),
            contextlib.ExitStack() as stack,
        ):
            tmp = Path(td)
            ctx = self._fake_ctx(tmp)
            for patch in self._patch_vault_and_github():
                stack.enter_context(patch)
            coord = _build_coordinator(ctx)
            self.assertIsInstance(coord.s3_store, S3BlobStore)
            self.assertEqual(coord.s3_store.config.bucket, "devon-backups")

    def test_tick_builder_s3_store_none_when_unconfigured(self):
        from nomorals.tools.backup import _build_coordinator

        with (
            tempfile.TemporaryDirectory() as td,
            _s3_env(),
            contextlib.ExitStack() as stack,
        ):
            tmp = Path(td)
            ctx = self._fake_ctx(tmp)
            for patch in self._patch_vault_and_github():
                stack.enter_context(patch)
            coord = _build_coordinator(ctx)
            self.assertIsNone(coord.s3_store)

    def test_cli_builder_passes_s3_store_when_configured(self):
        from nomorals.backup.__main__ import _build_cli_coordinator

        with (
            tempfile.TemporaryDirectory() as td,
            _s3_env(**_FULL_ENV),
            contextlib.ExitStack() as stack,
        ):
            tmp = Path(td)
            root = _make_root(tmp)
            for patch in self._patch_vault_and_github():
                stack.enter_context(patch)
            coord = _build_cli_coordinator(root, str(tmp / "work"))
            self.assertIsInstance(coord.s3_store, S3BlobStore)
            self.assertEqual(coord.s3_store.config.bucket, "devon-backups")

    def test_cli_builder_s3_store_none_when_unconfigured(self):
        from nomorals.backup.__main__ import _build_cli_coordinator

        with (
            tempfile.TemporaryDirectory() as td,
            _s3_env(),
            contextlib.ExitStack() as stack,
        ):
            tmp = Path(td)
            root = _make_root(tmp)
            for patch in self._patch_vault_and_github():
                stack.enter_context(patch)
            coord = _build_cli_coordinator(root, str(tmp / "work"))
            self.assertIsNone(coord.s3_store)


class RunBackupS3Tests(unittest.TestCase):
    def _coordinator(self, tmp: Path, s3_store):
        root = _make_root(tmp)
        vault = _FakeVault()
        store_backup_key(vault, generate_backup_key())
        return BackupCoordinator(
            root=root, work_dir=tmp / "work", vault=vault,
            connector=_FakeGitHub(), notifier=_FakeNotifier(),
            s3_store=s3_store,
        ), root

    def test_run_backup_uploads_fourth_copy(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            store = _FakeS3Store()
            coord, _root = self._coordinator(tmp, store)
            result = coord.run_backup()
            self.assertTrue(result["shipped"])
            self.assertIn("s3", result)
            self.assertTrue(result["s3"]["shipped"])
            self.assertEqual(len(store.puts), len(result["s3"]["objects"]))
            self.assertTrue(all(
                p["file"].endswith(".enc") for p in store.puts))
            self.assertNotIn("s3_error", result)

    def test_s3_failure_is_honest_not_silent(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            coord, _root = self._coordinator(tmp, _FailingStore())
            result = coord.run_backup()
            self.assertTrue(result["shipped"])
            self.assertIn("s3_error", result)
            self.assertIn("boom", result["s3_error"])
            self.assertNotIn("s3", result)

    def test_no_s3_store_leaves_result_untouched(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            coord, _root = self._coordinator(tmp, None)
            result = coord.run_backup()
            self.assertTrue(result["shipped"])
            self.assertNotIn("s3", result)
            self.assertNotIn("s3_error", result)


class TickReturnTests(unittest.TestCase):
    def _fake_registry(self, ctx):
        fns: dict = {}

        class _Registry:
            context = ctx

            def register(self, name, **kwargs):
                def deco(fn):
                    fns[name] = fn
                    return fn
                return deco

        return _Registry(), fns

    def test_tick_return_surfaces_s3_outcome(self):
        from nomorals.tools.backup import register

        canned = {"backup_id": "backup-20260101-000000", "shipped": True,
                  "queued": False, "manifest_pushed": True,
                  "total_encrypted_bytes": 42,
                  "s3": {"shipped": True, "objects": [{"file": "a.enc"}]}}

        class _Coord:
            def run_backup(self, *, label=""):
                return dict(canned)

        registry, fns = self._fake_registry(SimpleNamespace())
        with mock.patch("nomorals.tools.backup._build_coordinator",
                        return_value=_Coord()):
            register(registry)
            out = fns["__backup_tick__"]()
        self.assertTrue(out["ok"])
        self.assertTrue(out["s3"])
        self.assertEqual(out["s3_error"], "")

    def test_tick_return_surfaces_s3_failure(self):
        from nomorals.tools.backup import register

        canned = {"backup_id": "backup-20260101-000000", "shipped": True,
                  "queued": False, "manifest_pushed": True,
                  "total_encrypted_bytes": 42,
                  "s3_error": "boom: endpoint unreachable"}

        class _Coord:
            def run_backup(self, *, label=""):
                return dict(canned)

        registry, fns = self._fake_registry(SimpleNamespace())
        with mock.patch("nomorals.tools.backup._build_coordinator",
                        return_value=_Coord()):
            register(registry)
            out = fns["__backup_tick__"]()
        self.assertTrue(out["ok"])
        self.assertFalse(out["s3"])
        self.assertIn("boom", out["s3_error"])


class ReportS3Tests(unittest.TestCase):
    def _rows(self, doc):
        rows = []
        for section in doc.sections:
            rows.extend(getattr(section, "rows", ()))
        return rows

    def test_result_doc_shows_s3_row_when_uploaded(self):
        from nomorals.agents.alert_templates import backup_result_doc

        doc = backup_result_doc({
            "backup_id": "nightly",
            "s3": {"shipped": True,
                   "objects": [{"file": "a.enc"}, {"file": "b.enc"}]},
        })
        rows = self._rows(doc)
        s3 = [r for r in rows if r.label == "S3 copy"]
        self.assertEqual(len(s3), 1)
        self.assertIn("2 objects", str(s3[0].value))

    def test_result_doc_shows_s3_row_when_failed(self):
        from nomorals.agents.alert_templates import backup_result_doc

        doc = backup_result_doc({
            "backup_id": "nightly",
            "s3_error": "boom: endpoint unreachable",
        })
        rows = self._rows(doc)
        s3 = [r for r in rows if r.label == "S3 copy"]
        self.assertEqual(len(s3), 1)
        self.assertIn("failed", str(s3[0].value))
        self.assertIn("boom", str(s3[0].value))

    def test_result_doc_has_no_s3_row_when_unconfigured(self):
        from nomorals.agents.alert_templates import backup_result_doc
        from nomorals.social.chat.present import body_text

        doc = backup_result_doc({"backup_id": "nightly"})
        rows = self._rows(doc)
        self.assertEqual([r for r in rows if r.label == "S3 copy"], [])
        self.assertNotIn("S3", body_text(doc))

    def test_report_body_names_s3_copy_when_uploaded(self):
        from nomorals.backup.report import report_backup_result

        notifier = _FakeNotifier()
        report_backup_result(notifier, {
            "backup_id": "nightly", "assets": [1, 2],
            "total_encrypted_bytes": 10, "tag": "nightly",
            "manifest_pushed": True,
            "s3": {"shipped": True, "objects": [{"file": "a.enc"}]},
        })
        self.assertEqual(len(notifier.sent), 1)
        self.assertIn("S3 copy", notifier.sent[0]["body"])


if __name__ == "__main__":
    unittest.main()

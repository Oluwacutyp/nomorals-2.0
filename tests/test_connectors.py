"""Wave L1: connector framework + GitHub reference connector.

Offline by design — the GitHub API is faked at the HttpClient boundary,
git operations run against local file:// repos. No network, no secrets.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors import (
    Connector,
    ConnectorError,
    ConnectorStatus,
    ConnectResult,
    GitHubConnector,
    GitHubError,
    create_connector,
    device_flow_token,
    get_connector,
    list_connectors,
    pick_scopes,
    prompt_secret,
    register_connector,
)
from nomorals.storage.db import Database


def _vault() -> CredentialVault:
    return CredentialVault(Database(":memory:"), master_passphrase="test")


# ── fake HTTP ────────────────────────────────────────────────────────────


class FakeResponse:
    def __init__(
        self,
        status: int = 200,
        payload: Any = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status = status
        self.headers = headers or {}
        self._payload = payload if payload is not None else {}

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    @property
    def text(self) -> str:
        return json.dumps(self._payload)

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> FakeResponse:
        if not self.ok:
            raise AssertionError(f"unexpected {self.status}")
        return self


class FakeHttp:
    """Scripted stand-in for HttpClient. No network."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any]] = []
        self.routes: list[tuple[str, str, FakeResponse]] = []

    def route(self, method: str, path: str, response: FakeResponse) -> None:
        self.routes.append((method.upper(), path, response))

    def _dispatch(self, method: str, url: str, payload: Any = None) -> FakeResponse:
        self.calls.append((method.upper(), url, payload))
        # Longest path first: "/user" must not shadow "/user/repos".
        for rm, rp, resp in sorted(self.routes, key=lambda r: -len(r[1])):
            if rm == method.upper() and rp in url:
                return resp
        return FakeResponse(404, {"message": "not mocked"})

    def get(self, url: str, **kw: Any) -> FakeResponse:
        return self._dispatch("GET", url)

    def post_json(self, url: str, payload: Any, **kw: Any) -> FakeResponse:
        return self._dispatch("POST", url, payload)

    def post_form(self, url: str, form: Any, **kw: Any) -> FakeResponse:
        return self._dispatch("POST", url, form)

    def request(self, method: str, url: str, **kw: Any) -> FakeResponse:
        return self._dispatch(method, url)


def _github(http: FakeHttp | None = None) -> tuple[GitHubConnector, FakeHttp]:
    http = http or FakeHttp()
    return GitHubConnector(_vault(), http=http), http


def _route_user(http: FakeHttp, login: str = "octocat") -> None:
    http.route(
        "GET", "/user",
        FakeResponse(200, {"login": login, "id": 1},
                     {"x-oauth-scopes": "repo, read:user"}),
    )


# ── registry ─────────────────────────────────────────────────────────────


class RegistryTests(unittest.TestCase):
    def test_github_is_registered(self) -> None:
        cls = get_connector("github")
        self.assertIs(cls, GitHubConnector)

    def test_unknown_connector_raises_with_known_list(self) -> None:
        with self.assertRaises(ConnectorError) as ctx:
            get_connector("nope")
        self.assertIn("github", str(ctx.exception))

    def test_list_includes_github_shape(self) -> None:
        infos = list_connectors()
        gh = next(i for i in infos if i["id"] == "github")
        self.assertEqual(gh["name"], "GitHub")
        self.assertIn("pat", gh["auth_methods"])
        self.assertIn("repo", gh["provisionable"])

    def test_duplicate_registration_rejected(self) -> None:
        with self.assertRaises(ConnectorError):
            register_connector(GitHubConnector)

    def test_empty_id_rejected(self) -> None:
        class Bad(Connector):
            id = ""

            def connect(self, **kw: Any) -> ConnectResult:  # pragma: no cover
                raise AssertionError

            def disconnect(self) -> None:  # pragma: no cover
                raise AssertionError

            def status(self) -> ConnectorStatus:  # pragma: no cover
                raise AssertionError

            def test_connection(self) -> bool:  # pragma: no cover
                raise AssertionError

        with self.assertRaises(ConnectorError):
            register_connector(Bad)

    def test_create_connector_builds_instance(self) -> None:
        conn = create_connector("github", _vault())
        self.assertIsInstance(conn, GitHubConnector)


# ── base credential lifecycle ────────────────────────────────────────────


class CredentialLifecycleTests(unittest.TestCase):
    def test_store_load_clear_roundtrip(self) -> None:
        conn, _ = _github()
        self.assertIsNone(conn._load_credential())
        cred = conn._store_credential(
            "octocat", "s3cret", credential_type="pat", scopes=["repo"]
        )
        self.assertGreater(cred.id, 0)
        loaded = conn._load_credential()
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.password, "s3cret")  # decrypted
        self.assertEqual(loaded.metadata["scopes"], ["repo"])
        conn._clear_credential()
        self.assertIsNone(conn._load_credential())

    def test_status_to_dict_has_no_secrets(self) -> None:
        st = ConnectorStatus(connected=True, account="octocat",
                             scopes=["repo"], detail="ok")
        d = st.to_dict()
        self.assertNotIn("s3cret", json.dumps(d))
        self.assertTrue(d["connected"])


# ── auth UX ──────────────────────────────────────────────────────────────


class PromptSecretTests(unittest.TestCase):
    def test_env_var_wins(self) -> None:
        with mock.patch.dict(os.environ, {"NM_TEST_SECRET": "from-env"}):
            self.assertEqual(
                prompt_secret("tok", env_var="NM_TEST_SECRET"), "from-env"
            )

    def test_non_tty_fails_fast_with_guidance(self) -> None:
        env = {"NM_TEST_SECRET_MISSING": ""}
        with mock.patch.dict(os.environ, env, clear=False), mock.patch.object(
            sys.stdin, "isatty", return_value=False
        ), self.assertRaises(ConnectorError) as ctx:
            prompt_secret("tok", env_var="NM_TEST_SECRET_MISSING")
        self.assertIn("NM_TEST_SECRET_MISSING", str(ctx.exception))

    def test_tty_prompt_uses_getpass(self) -> None:
        with mock.patch.object(sys.stdin, "isatty", return_value=True), mock.patch(
            "getpass.getpass", return_value="typed"
        ) as gp:
            self.assertEqual(prompt_secret("tok"), "typed")
        gp.assert_called_once()


class PickScopesTests(unittest.TestCase):
    def test_non_tty_returns_defaults(self) -> None:
        with mock.patch.object(sys.stdin, "isatty", return_value=False):
            self.assertEqual(pick_scopes(["a", "b"], ["a"]), ["a"])

    def test_empty_answer_returns_defaults(self) -> None:
        self.assertEqual(pick_scopes(["a", "b"], ["a"], input_fn=lambda _: ""),
                         ["a"])

    def test_all_keyword(self) -> None:
        self.assertEqual(pick_scopes(["a", "b"], ["a"],
                                     input_fn=lambda _: "all"), ["a", "b"])

    def test_number_selection(self) -> None:
        self.assertEqual(pick_scopes(["a", "b", "c"], ["a"],
                                     input_fn=lambda _: "2,3"), ["b", "c"])


class DeviceFlowTests(unittest.TestCase):
    def test_happy_path(self) -> None:
        http = FakeHttp()
        seen: list[dict[str, Any]] = []

        class Scripted(FakeHttp):
            polls = 0

            def post_form(self, url: str, form: Any, **kw: Any) -> FakeResponse:
                if "device" in url:
                    return FakeResponse(200, {
                        "device_code": "dc", "user_code": "UC-1",
                        "verification_uri": "https://x/activate",
                        "expires_in": 900, "interval": 1,
                    })
                self.polls += 1
                if self.polls < 2:
                    return FakeResponse(200, {"error": "authorization_pending"})
                return FakeResponse(200, {"access_token": "tok123"})

        http = Scripted()
        out = device_flow_token(
            client_id="cid",
            device_code_url="https://x/device",
            token_url="https://x/token",
            http=http,
            on_code=seen.append,
        )
        self.assertEqual(out["access_token"], "tok123")
        self.assertEqual(seen[0]["user_code"], "UC-1")

    def test_access_denied_raises(self) -> None:
        class Denied(FakeHttp):
            def post_form(self, url: str, form: Any, **kw: Any) -> FakeResponse:
                if "device" in url:
                    return FakeResponse(200, {
                        "device_code": "dc", "user_code": "UC-1",
                        "verification_uri": "https://x/activate",
                        "expires_in": 900, "interval": 1,
                    })
                return FakeResponse(200, {"error": "access_denied"})

        with self.assertRaises(ConnectorError) as ctx:
            device_flow_token(
                client_id="cid",
                device_code_url="https://x/device",
                token_url="https://x/token",
                http=Denied(),
                on_code=lambda _: None,
            )
        self.assertIn("access denied", str(ctx.exception))


# ── GitHub lifecycle ─────────────────────────────────────────────────────


class GitHubConnectTests(unittest.TestCase):
    def test_connect_validates_and_vault_stores(self) -> None:
        conn, http = _github()
        _route_user(http)
        with mock.patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_fake"}):
            result = conn.connect()
        self.assertTrue(result.ok)
        self.assertEqual(result.account, "octocat")
        self.assertIn("repo", result.scopes)
        cred = conn.vault.get("connector:github", "octocat")
        self.assertEqual(cred.password, "ghp_fake")
        self.assertEqual(cred.credential_type, "pat")

    def test_connect_rejects_bad_token(self) -> None:
        conn, http = _github()
        http.route("GET", "/user",
                   FakeResponse(401, {"message": "Bad credentials"}))
        with self.assertRaises(GitHubError) as ctx:
            conn.connect(token="ghp_bad")
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIsNone(conn._load_credential())

    def test_disconnect_is_idempotent(self) -> None:
        conn, http = _github()
        conn.disconnect()  # nothing stored: no crash
        _route_user(http)
        conn.connect(token="ghp_fake")
        conn.disconnect()
        self.assertIsNone(conn._load_credential())

    def test_status_flows(self) -> None:
        conn, http = _github()
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertIn("not connected", st.detail)
        _route_user(http)
        conn.connect(token="ghp_fake")
        st = conn.status()
        self.assertTrue(st.connected)
        self.assertEqual(st.account, "octocat")

    def test_status_reports_rejected_token(self) -> None:
        conn, http = _github()
        _route_user(http)
        conn.connect(token="ghp_fake")
        http.routes.clear()
        http.route("GET", "/user",
                   FakeResponse(401, {"message": "Bad credentials"}))
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertIn("401", st.detail)

    def test_test_connection(self) -> None:
        conn, http = _github()
        self.assertFalse(conn.test_connection())
        _route_user(http)
        conn.connect(token="ghp_fake")
        self.assertTrue(conn.test_connection())

    def test_rate_limit_message(self) -> None:
        conn, http = _github()
        _route_user(http)
        conn.connect(token="ghp_fake")
        http.routes.clear()
        http.route("GET", "/user",
                   FakeResponse(403, {"message": "API rate limit exceeded"},
                                {"x-ratelimit-reset": "123"}))
        with self.assertRaises(GitHubError) as ctx:
            conn._api("GET", "/user")
        self.assertIn("rate limit", str(ctx.exception).lower())


# ── GitHub API surface ───────────────────────────────────────────────────


class GitHubApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn, self.http = _github()
        _route_user(self.http)
        self.conn.connect(token="ghp_fake")

    def test_create_repo_payload(self) -> None:
        self.http.route("POST", "/user/repos",
                        FakeResponse(201, {"full_name": "octocat/nw",
                                           "private": True}))
        repo = self.conn.create_repo("nw", private=True,
                                     description="d")
        self.assertEqual(repo["full_name"], "octocat/nw")
        method, url, payload = self.http.calls[-1]
        self.assertEqual(method, "POST")
        self.assertTrue(payload["private"])
        self.assertEqual(payload["name"], "nw")

    def test_list_repos(self) -> None:
        self.http.route("GET", "/user/repos",
                        FakeResponse(200, [{"full_name": "octocat/a"}]))
        repos = self.conn.list_repos(limit=5)
        self.assertEqual(len(repos), 1)

    def test_create_branch_resolves_base_sha(self) -> None:
        self.http.route("GET", "/git/ref/heads/main",
                        FakeResponse(200, {"object": {"sha": "abc123"}}))
        self.http.route("POST", "/git/refs",
                        FakeResponse(201, {"ref": "refs/heads/feat"}))
        out = self.conn.create_branch("octocat/nw", "feat")
        self.assertEqual(out["ref"], "refs/heads/feat")
        _, _, payload = self.http.calls[-1]
        self.assertEqual(payload["sha"], "abc123")

    def test_create_release_payload(self) -> None:
        self.http.route("POST", "/releases",
                        FakeResponse(201, {"tag_name": "v1"}))
        out = self.conn.create_release("octocat/nw", "v1", body="notes")
        self.assertEqual(out["tag_name"], "v1")
        _, _, payload = self.http.calls[-1]
        self.assertEqual(payload["tag_name"], "v1")

    def test_create_webhook_payload(self) -> None:
        self.http.route("POST", "/hooks",
                        FakeResponse(201, {"id": 7}))
        out = self.conn.create_webhook("octocat/nw", "https://x/hook",
                                       events=["push", "pull_request"])
        self.assertEqual(out["id"], 7)
        _, _, payload = self.http.calls[-1]
        self.assertEqual(payload["config"]["url"], "https://x/hook")

    def test_add_deploy_key_payload(self) -> None:
        self.http.route("POST", "/keys",
                        FakeResponse(201, {"id": 9, "read_only": True}))
        out = self.conn.add_deploy_key("octocat/nw", "ci",
                                       "ssh-ed25519 AAAA")
        self.assertTrue(out["read_only"])
        _, _, payload = self.http.calls[-1]
        self.assertEqual(payload["title"], "ci")

    def test_provision_dispatch_and_unknown_kind(self) -> None:
        self.http.route("POST", "/user/repos",
                        FakeResponse(201, {"full_name": "octocat/p"}))
        self.assertTrue(self.conn.can_provision("repo"))
        out = self.conn.provision("repo", name="p")
        self.assertEqual(out["full_name"], "octocat/p")
        with self.assertRaises(ConnectorError):
            self.conn.provision("nope")

    def test_not_found_surfaces(self) -> None:
        self.http.route("GET", "/repos/octocat/missing",
                        FakeResponse(404, {"message": "Not Found"}))
        with self.assertRaises(GitHubError) as ctx:
            self.conn.get_repo("octocat/missing")
        self.assertEqual(ctx.exception.status_code, 404)


# ── git-backed behavior (offline, file://) ───────────────────────────────


def _git(*args: str, cwd: Path, env: dict[str, str] | None = None) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), env=env,
                   capture_output=True, check=True, timeout=60)


class GitBackedTests(unittest.TestCase):
    def setUp(self) -> None:
        if not shutil.which("git"):
            self.skipTest("git not available")
        self.tmp = Path(tempfile.mkdtemp(prefix="nm-gh-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        # source repo with one commit
        self.src = self.tmp / "src"
        self.src.mkdir()
        _git("init", "-b", "main", cwd=self.src)
        _git("config", "user.email", "t@t", cwd=self.src)
        _git("config", "user.name", "t", cwd=self.src)
        (self.src / "a.txt").write_text("hello\n")
        _git("add", ".", cwd=self.src)
        _git("commit", "-qm", "init", cwd=self.src)
        self.conn, self.http = _github()
        _route_user(self.http)
        self.conn.connect(token="ghp_fake")

    def test_backup_repo_mirror_and_manifest(self) -> None:
        url = f"file://{self.src}"
        manifest = self.conn.backup_repo(url, self.tmp / "backups")
        target = Path(manifest["path"])
        self.assertTrue(target.is_dir())
        self.assertTrue(manifest["verified"])
        self.assertEqual(manifest["repo"], url)
        manifest_file = self.tmp / "backups" / (
            url.replace("://", "_").replace("/", "__") + ".manifest.json")
        self.assertTrue(manifest_file.is_file())
        # the mirror really holds the commit
        out = subprocess.run(
            ["git", "--git-dir", str(target), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(out.stdout.strip(), manifest["head"])
        # second run updates instead of failing
        manifest2 = self.conn.backup_repo(url, self.tmp / "backups")
        self.assertEqual(manifest2["head"], manifest["head"])

    def test_push_code_to_file_remote(self) -> None:
        bare = self.tmp / "remote.git"
        subprocess.run(["git", "init", "--bare", "-b", "main", str(bare)],
                       capture_output=True, check=True, timeout=30)
        _git("remote", "add", "origin", f"file://{bare}", cwd=self.src)
        result = self.conn.push_code(self.src, "octocat/nw", branch="main")
        # file:// needs no auth; push should succeed (or fail only for a
        # real git reason, reported honestly in the payload)
        self.assertIn("pushed", result)
        if result["pushed"]:
            out = subprocess.run(
                ["git", "--git-dir", str(bare), "rev-parse", "main"],
                capture_output=True, text=True, timeout=30)
            src_head = subprocess.run(
                ["git", "-C", str(self.src), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=30).stdout.strip()
            self.assertEqual(out.stdout.strip(), src_head)
        else:
            self.assertTrue(result["output"].strip())

    def test_push_code_rejects_non_repo(self) -> None:
        notrepo = self.tmp / "notrepo"
        notrepo.mkdir()
        with self.assertRaises(GitHubError):
            self.conn.push_code(notrepo, "octocat/nw")


# ── CLI ──────────────────────────────────────────────────────────────────


class CliTests(unittest.TestCase):
    def test_connectors_list_shows_github(self) -> None:
        import argparse

        from nomorals.cmdline.commands.meta import _cmd_connectors

        args = argparse.Namespace(action="list", name="", json=False)
        with mock.patch.dict(os.environ, {}, clear=False):
            rc = _cmd_connectors(args, context=None)
        self.assertEqual(rc, 0)

    def test_connectors_unknown_name_fails_cleanly(self) -> None:
        import argparse
        from types import SimpleNamespace

        from nomorals.cmdline.commands.meta import _cmd_connectors

        args = argparse.Namespace(action="status", name="nope", json=True)
        ctx = SimpleNamespace(db=Database(":memory:"))
        with mock.patch.dict(os.environ,
                             {"NM_VAULT_PASSPHRASE": "test"}), \
             mock.patch("nomorals.cmdline.commands.meta._emit") as emit:
            rc = _cmd_connectors(args, context=ctx)
        self.assertEqual(rc, 1)
        payload = emit.call_args[0][1]
        self.assertIn("unknown connector", payload["error"])


# ── human-in-the-loop checkpoints ──────────────────────────────────────


def _checkpoint_db() -> Database:
    db = Database(":memory:")
    db.execute(
        """
        CREATE TABLE notifications (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL DEFAULT '',
            title TEXT NOT NULL DEFAULT '',
            body TEXT NOT NULL DEFAULT '',
            delivered INTEGER NOT NULL DEFAULT 0,
            delivery_state TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL DEFAULT 0
        )
        """
    )
    return db


def _notty() -> mock._patch:
    """Force the non-interactive path regardless of the real stdin."""
    return mock.patch.object(sys.stdin, "isatty", return_value=False)


class CheckpointStoreTests(unittest.TestCase):
    def test_create_get_roundtrip(self) -> None:
        from nomorals.connectors import (
            CheckpointKind,
            CheckpointState,
            CheckpointStore,
        )

        store = CheckpointStore(_checkpoint_db())
        cp = store.create(
            "github", CheckpointKind.CAPTCHA, "Solve the CAPTCHA",
            "do it yourself", resume_state={"stage": "signup"},
        )
        self.assertEqual(cp.state, CheckpointState.PENDING)
        got = store.get(cp.id)
        self.assertEqual(got.id, cp.id)
        self.assertEqual(got.kind, CheckpointKind.CAPTCHA)
        self.assertEqual(got.resume_state["stage"], "signup")

    def test_get_unknown_raises(self) -> None:
        from nomorals.connectors import CheckpointStore
        from nomorals.core.errors import NotFound

        store = CheckpointStore(_checkpoint_db())
        with self.assertRaises(NotFound):
            store.get("chk-nope")

    def test_list_pending_filters_and_resolve(self) -> None:
        from nomorals.connectors import (
            CheckpointKind,
            CheckpointState,
            CheckpointStore,
        )

        store = CheckpointStore(_checkpoint_db())
        a = store.create("github", CheckpointKind.MANUAL_STEP, "A", "a")
        b = store.create("github", CheckpointKind.MANUAL_STEP, "B", "b")
        resolved = store.resolve(a.id, note="done")
        self.assertEqual(resolved.state, CheckpointState.RESOLVED)
        self.assertEqual(resolved.result_note, "done")
        pending = store.list_pending("github")
        self.assertEqual([c.id for c in pending], [b.id])
        self.assertEqual(pending[0].title, "B")

    def test_resolve_twice_rejected(self) -> None:
        from nomorals.connectors import (
            CheckpointKind,
            CheckpointStore,
            HumanCheckpointPending,
        )

        store = CheckpointStore(_checkpoint_db())
        cp = store.create("github", CheckpointKind.MANUAL_STEP, "A", "a")
        store.resolve(cp.id)
        with self.assertRaises(ConnectorError):
            store.resolve(cp.id)
        # HumanCheckpointPending is a ConnectorError (clean CLI surfacing)
        self.assertTrue(issubclass(HumanCheckpointPending, ConnectorError))

    def test_cancel(self) -> None:
        from nomorals.connectors import (
            CheckpointKind,
            CheckpointState,
            CheckpointStore,
        )

        store = CheckpointStore(_checkpoint_db())
        cp = store.create("github", CheckpointKind.MANUAL_STEP, "A", "a")
        cancelled = store.cancel(cp.id, note="changed mind")
        self.assertEqual(cancelled.state, CheckpointState.CANCELLED)
        self.assertEqual(store.list_pending(), [])

    def test_expire_stale(self) -> None:
        from nomorals.connectors import (
            CheckpointKind,
            CheckpointState,
            CheckpointStore,
        )

        store = CheckpointStore(_checkpoint_db())
        fresh = store.create("github", CheckpointKind.MANUAL_STEP, "F", "f")
        stale = store.create("github", CheckpointKind.MANUAL_STEP, "S", "s",
                             ttl_seconds=-1)
        self.assertEqual(store.expire_stale(), 1)
        self.assertEqual(store.get(stale.id).state, CheckpointState.EXPIRED)
        self.assertEqual(store.get(fresh.id).state, CheckpointState.PENDING)


class RequestHumanActionTests(unittest.TestCase):
    def test_noninteractive_persists_pings_and_pauses(self) -> None:
        from types import SimpleNamespace

        from nomorals.connectors import (
            CheckpointKind,
            CheckpointState,
            CheckpointStore,
            HumanCheckpointPending,
            request_human_action,
        )

        db = _checkpoint_db()
        ctx = SimpleNamespace(db=db)  # no gateway: persist-only delivery
        with _notty(), self.assertRaises(HumanCheckpointPending) as raised:
            request_human_action(
                    "github", CheckpointKind.CAPTCHA, "Solve the CAPTCHA",
                    "you do it", db=db, context=ctx,
                    resume_state={"stage": "x"},
                )
        cp = raised.exception.checkpoint
        self.assertIn("nm connectors checkpoint resolve --id", str(raised.exception))
        stored = CheckpointStore(db).get(cp.id)
        self.assertEqual(stored.state, CheckpointState.PENDING)
        # owner was pinged through the owner-only channel (persisted)
        row = db.query_one(
            "SELECT kind, title, delivery_state FROM notifications "
            "ORDER BY created_at DESC LIMIT 1"
        )
        self.assertEqual(row["kind"], "checkpoint")
        self.assertIn("Solve the CAPTCHA", row["title"])

    def test_interactive_waits_then_resolves(self) -> None:
        from nomorals.connectors import (
            CheckpointKind,
            CheckpointState,
            CheckpointStore,
            request_human_action,
        )

        db = _checkpoint_db()
        with mock.patch.object(sys.stdin, "isatty", return_value=True), \
             mock.patch("builtins.input", return_value=""):
            cp = request_human_action(
                "github", CheckpointKind.EMAIL_VERIFY, "Click the link",
                "check your inbox", db=db, context=None,
            )
        self.assertEqual(cp.state, CheckpointState.RESOLVED)
        self.assertEqual(CheckpointStore(db).get(cp.id).state,
                         CheckpointState.RESOLVED)


class _MinimalConnector(Connector):
    id = "mini"
    name = "Mini"
    description = "test double"
    auth_methods = []
    provisionable = ()

    def connect(self, **kw: Any) -> ConnectResult:
        raise NotImplementedError

    def disconnect(self) -> None:
        raise NotImplementedError

    def status(self) -> ConnectorStatus:
        raise NotImplementedError

    def test_connection(self) -> bool:
        raise NotImplementedError


class BaseCheckpointTests(unittest.TestCase):
    def test_resume_checkpoint_default_unsupported(self) -> None:
        from nomorals.connectors import (
            CheckpointKind,
            CheckpointState,
            HumanCheckpoint,
        )

        conn = _MinimalConnector(_vault())
        cp = HumanCheckpoint(
            id="chk-x", connector_id="mini",
            kind=CheckpointKind.MANUAL_STEP,
            title="t", instructions="i", state=CheckpointState.RESOLVED,
        )
        with self.assertRaises(ConnectorError):
            conn.resume_checkpoint(cp, db=Database(":memory:"))

    def test_request_human_helper_routes_to_store(self) -> None:
        from nomorals.connectors import CheckpointKind

        conn = _MinimalConnector(_vault())
        db = _checkpoint_db()
        with _notty(), self.assertRaises(ConnectorError):
            conn.request_human(
                    CheckpointKind.TOS_ACCEPT, "Accept the terms",
                    "read them first", db=db, context=None,
                )
        from nomorals.connectors import CheckpointStore

        self.assertEqual(len(CheckpointStore(db).list_pending("mini")), 1)


class GitHubAccountFlowTests(unittest.TestCase):
    def _account_db(self) -> Database:
        return _checkpoint_db()

    def test_invalid_username_rejected(self) -> None:
        conn, _http = _github()
        db = self._account_db()
        with self.assertRaises(ConnectorError) as ctx:
            conn.provision("github_account", username="-bad-", db=db)
        self.assertIn("valid GitHub username", str(ctx.exception))

    def test_taken_username_rejected(self) -> None:
        conn, http = _github()
        http.route("GET", "/users/takenuser",
                   FakeResponse(200, {"login": "takenuser"}))
        db = self._account_db()
        with self.assertRaises(ConnectorError) as ctx:
            conn.provision("github_account", username="takenuser", db=db)
        self.assertIn("taken", str(ctx.exception))

    def test_one_account_per_service_enforced(self) -> None:
        conn, _http = _github()
        conn._store_credential("existinguser", "tok",
                               credential_type="pat")
        db = self._account_db()
        with self.assertRaises(ConnectorError) as ctx:
            conn.provision("github_account", username="freshuser", db=db)
        self.assertIn("one account per service", str(ctx.exception))

    def test_signup_stage_creates_checkpoint_and_pauses(self) -> None:
        from nomorals.connectors import (
            CheckpointKind,
            CheckpointState,
            CheckpointStore,
            HumanCheckpointPending,
        )

        conn, _http = _github()  # unmocked /users/* -> 404 = available
        db = self._account_db()
        with _notty(), self.assertRaises(HumanCheckpointPending) as raised:
            conn.provision("github_account", username="freshuser", db=db)
        cp = raised.exception.checkpoint
        self.assertEqual(cp.kind, CheckpointKind.MANUAL_STEP)
        self.assertEqual(cp.resume_state["stage"], "signup")
        self.assertEqual(cp.resume_state["username"], "freshuser")
        self.assertEqual(CheckpointStore(db).get(cp.id).state,
                         CheckpointState.PENDING)

    def test_resume_signup_moves_to_email_verify(self) -> None:
        from nomorals.connectors import (
            CheckpointKind,
            CheckpointStore,
            HumanCheckpointPending,
        )

        conn, http = _github()
        http.route("GET", "/users/freshuser",
                   FakeResponse(200, {"login": "freshuser"}))
        db = self._account_db()
        store = CheckpointStore(db)
        cp = store.create(
            "github", CheckpointKind.MANUAL_STEP, "signup", "do it",
            resume_state={"stage": "signup", "username": "freshuser"},
        )
        store.resolve(cp.id, note="owner signed up")
        with _notty(), self.assertRaises(HumanCheckpointPending) as raised:
            conn.resume_checkpoint(store.get(cp.id), db=db)
        nxt = raised.exception.checkpoint
        self.assertEqual(nxt.kind, CheckpointKind.EMAIL_VERIFY)
        self.assertEqual(nxt.resume_state["stage"], "verify_email")

    def test_resume_signup_account_missing_rejected(self) -> None:
        from nomorals.connectors import (
            CheckpointKind,
            CheckpointStore,
        )

        conn, _http = _github()  # /users/ghost -> 404
        db = self._account_db()
        store = CheckpointStore(db)
        cp = store.create(
            "github", CheckpointKind.MANUAL_STEP, "signup", "do it",
            resume_state={"stage": "signup", "username": "ghost"},
        )
        store.resolve(cp.id)
        with self.assertRaises(ConnectorError) as ctx:
            conn.resume_checkpoint(store.get(cp.id), db=db)
        self.assertIn("does not exist", str(ctx.exception))

    def test_resume_verify_email_completes_handover(self) -> None:
        from nomorals.connectors import (
            CheckpointKind,
            CheckpointStore,
        )

        conn, _http = _github()
        db = self._account_db()
        store = CheckpointStore(db)
        cp = store.create(
            "github", CheckpointKind.EMAIL_VERIFY, "verify", "click",
            resume_state={"stage": "verify_email", "username": "freshuser"},
        )
        store.resolve(cp.id, note="clicked")
        result = conn.resume_checkpoint(store.get(cp.id), db=db)
        self.assertTrue(result["done"])
        self.assertEqual(result["account"], "freshuser")
        self.assertEqual(result["identity"], "owner")
        self.assertIn("connect --name github", result["message"])

    def test_resume_requires_resolved_checkpoint(self) -> None:
        from nomorals.connectors import (
            CheckpointKind,
            CheckpointStore,
        )

        conn, _http = _github()
        db = self._account_db()
        store = CheckpointStore(db)
        cp = store.create(
            "github", CheckpointKind.MANUAL_STEP, "signup", "do it",
            resume_state={"stage": "signup", "username": "freshuser"},
        )
        with self.assertRaises(ConnectorError) as ctx:
            conn.resume_checkpoint(cp, db=db)
        self.assertIn("not resolved", str(ctx.exception))

    def test_resume_unknown_stage_rejected(self) -> None:
        from nomorals.connectors import (
            CheckpointKind,
            CheckpointStore,
        )

        conn, _http = _github()
        db = self._account_db()
        store = CheckpointStore(db)
        cp = store.create(
            "github", CheckpointKind.MANUAL_STEP, "weird", "do it",
            resume_state={"stage": "bogus"},
        )
        store.resolve(cp.id)
        with self.assertRaises(ConnectorError):
            conn.resume_checkpoint(store.get(cp.id), db=db)

    def test_github_account_is_provisionable(self) -> None:
        conn, _http = _github()
        self.assertTrue(conn.can_provision("github_account"))


class CaptchaBoundaryTests(unittest.TestCase):
    def test_no_captcha_solving_machinery_in_package(self) -> None:
        """Structural enforcement of the hard boundary: the connectors
        package must contain no CAPTCHA-solving services, helpers, or
        bypass code. The human checkpoint is the only path."""
        import nomorals.connectors as pkg

        banned = (
            "2captcha", "anti-captcha", "anticaptcha", "capmonster",
            "capsolver", "deathbycaptcha", "solve_captcha", "captcha_solver",
            "captcha_bypass", "bypass_captcha", "recaptcha_solver",
        )
        hits = []
        root = Path(pkg.__file__).parent
        for path in sorted(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8").lower()
            for pat in banned:
                if pat in text:
                    hits.append(f"{path.name}: {pat}")
        self.assertEqual(hits, [])


class CheckpointCliTests(unittest.TestCase):
    def test_checkpoint_list_and_resolve_flow(self) -> None:
        import argparse
        from types import SimpleNamespace

        from nomorals.cmdline.commands.meta import _cmd_connectors
        from nomorals.connectors import (
            CheckpointKind,
            CheckpointStore,
        )

        db = _checkpoint_db()
        store = CheckpointStore(db)
        cp = store.create("github", CheckpointKind.MANUAL_STEP, "T", "i",
                          resume_state={"stage": "verify_email",
                                        "username": "freshuser"})
        ctx = SimpleNamespace(db=db)
        env = {"NM_VAULT_PASSPHRASE": "test"}

        with mock.patch.dict(os.environ, env), \
             mock.patch("nomorals.cmdline.commands.meta._emit") as emit:
            rc = _cmd_connectors(
                argparse.Namespace(action="checkpoint", name="",
                                   cop="list", json=True),
                ctx,
            )
        self.assertEqual(rc, 0)
        payload = emit.call_args[0][1]
        self.assertEqual(payload["count"], 1)

        # resolve: no network needed for verify_email stage
        with mock.patch.dict(os.environ, env), \
             mock.patch("nomorals.cmdline.commands.meta._emit") as emit:
            rc = _cmd_connectors(
                argparse.Namespace(action="checkpoint", name="",
                                   cop="resolve", id=cp.id, note="done",
                                   json=True),
                ctx,
            )
        self.assertEqual(rc, 0)
        payload = emit.call_args[0][1]
        self.assertEqual(payload["resolved"], cp.id)
        self.assertTrue(payload["resumed"])
        self.assertEqual(payload["result"]["account"], "freshuser")

    def test_checkpoint_cancel(self) -> None:
        import argparse
        from types import SimpleNamespace

        from nomorals.cmdline.commands.meta import _cmd_connectors
        from nomorals.connectors import (
            CheckpointKind,
            CheckpointStore,
        )

        db = _checkpoint_db()
        store = CheckpointStore(db)
        cp = store.create("github", CheckpointKind.MANUAL_STEP, "T", "i")
        ctx = SimpleNamespace(db=db)
        with mock.patch.dict(os.environ, {"NM_VAULT_PASSPHRASE": "test"}), \
             mock.patch("nomorals.cmdline.commands.meta._emit"):
            rc = _cmd_connectors(
                argparse.Namespace(action="checkpoint", name="",
                                   cop="cancel", id=cp.id, note="",
                                   json=True),
                ctx,
            )
        self.assertEqual(rc, 0)
        self.assertEqual(store.list_pending(), [])


if __name__ == "__main__":
    unittest.main()

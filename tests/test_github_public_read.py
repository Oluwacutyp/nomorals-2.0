"""GitHub public reads — no token needed.

GitHub serves public repos to unauthenticated callers (60 req/hr), but
the connector routed every read through the token path, so public data
was unreachable without a PAT — an artificial limitation. These tests
pin the public path: real API shapes via a scripted HttpClient, the
no-Authorization proof, the service_call gate skip, and (gated) one
live call against api.github.com.

Recorded fixtures below are real responses fetched from the live API
(2026-11-14): /repos/Oluwacutyp/goldenswing and the first page of
/users/Oluwacutyp/repos (sort=updated). Offline by default; the live
test runs only with NM_LIVE_GITHUB=1.
"""

from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors import GitHubConnector, GitHubError
from nomorals.storage.db import Database
from nomorals.tools import services
from nomorals.tools.registry import ToolRegistry

#: Real response for GET /repos/Oluwacutyp/goldenswing (2026-11-14).
GOLDENSWING_REPO = {
    "full_name": "Oluwacutyp/goldenswing",
    "name": "goldenswing",
    "description": (
        "GoldenSwing - all-timeframe trading engine (D1 bias, H4 zones, "
        "H1 trigger, M15/M30 entries)"
    ),
    "private": False,
    "html_url": "https://github.com/Oluwacutyp/goldenswing",
    "language": "Python",
    "stargazers_count": 0,
    "forks_count": 0,
    "open_issues_count": 0,
    "default_branch": "main",
    "created_at": "2026-10-14T06:42:55Z",
    "updated_at": "2026-11-12T01:59:01Z",
    "pushed_at": "2026-11-14T12:20:21Z",
    "size": 16486,
    "license": None,
    "topics": [],
    "visibility": "public",
}

#: Real response for GET /users/Oluwacutyp/repos (first 3, sort=updated).
OLUWACUTYP_REPOS = [
    {
        "full_name": "Oluwacutyp/nomorals-2.0",
        "name": "nomorals-2.0",
        "private": False,
        "language": "Python",
        "stargazers_count": 3,
    },
    {
        "full_name": "Oluwacutyp/goldenswing",
        "name": "goldenswing",
        "private": False,
        "language": "Python",
        "stargazers_count": 0,
    },
    {
        "full_name": "Oluwacutyp/voice-changer",
        "name": "voice-changer",
        "private": False,
        "language": None,
        "stargazers_count": 0,
    },
]


def _vault() -> CredentialVault:
    return CredentialVault(Database(":memory:"), master_passphrase="test")


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


class FakeHttp:
    """Scripted stand-in for HttpClient; records headers and params."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.routes: list[tuple[str, str, FakeResponse]] = []

    def route(self, method: str, path: str, response: FakeResponse) -> None:
        self.routes.append((method.upper(), path, response))

    def get(self, url: str, **kw: Any) -> FakeResponse:
        self.calls.append(
            {"method": "GET", "url": url,
             "headers": dict(kw.get("headers") or {}),
             "params": kw.get("params")}
        )
        for rm, rp, resp in sorted(self.routes, key=lambda r: -len(r[1])):
            if rm == "GET" and rp in url:
                return resp
        return FakeResponse(404, {"message": "not mocked"})


def _github(http: FakeHttp | None = None) -> tuple[GitHubConnector, FakeHttp]:
    http = http or FakeHttp()
    return GitHubConnector(_vault(), http=http), http


def _route_repo(http: FakeHttp) -> None:
    http.route("GET", "/repos/Oluwacutyp/goldenswing",
               FakeResponse(200, GOLDENSWING_REPO))


def _route_user_repos(http: FakeHttp) -> None:
    http.route("GET", "/users/Oluwacutyp/repos",
               FakeResponse(200, OLUWACUTYP_REPOS))


class PublicRepoTests(unittest.TestCase):
    def test_get_public_repo_returns_real_shape(self) -> None:
        conn, http = _github()
        _route_repo(http)
        repo = conn.get_public_repo("Oluwacutyp/goldenswing")
        self.assertEqual(repo["full_name"], "Oluwacutyp/goldenswing")
        self.assertFalse(repo["private"])
        self.assertEqual(repo["language"], "Python")
        self.assertEqual(repo["default_branch"], "main")

    def test_get_public_repo_sends_no_authorization(self) -> None:
        conn, http = _github()
        _route_repo(http)
        conn.get_public_repo("Oluwacutyp/goldenswing")
        self.assertEqual(len(http.calls), 1)
        headers = http.calls[0]["headers"]
        self.assertNotIn("Authorization", headers)
        self.assertNotIn("authorization", headers)

    def test_get_public_repo_needs_no_credential(self) -> None:
        conn, http = _github()
        _route_repo(http)
        self.assertIsNone(conn._load_credential())
        repo = conn.get_public_repo("Oluwacutyp/goldenswing")
        self.assertEqual(repo["full_name"], "Oluwacutyp/goldenswing")

    def test_get_public_repo_404_says_private_or_missing(self) -> None:
        conn, http = _github()
        http.route("GET", "/repos/someone/nope",
                   FakeResponse(404, {"message": "Not Found"}))
        with self.assertRaises(GitHubError) as ctx:
            conn.get_public_repo("someone/nope")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertIn("private", str(ctx.exception))

    def test_get_public_repo_rate_limit_is_honest(self) -> None:
        conn, http = _github()
        http.route(
            "GET", "/repos/Oluwacutyp/goldenswing",
            FakeResponse(403, {"message": "API rate limit exceeded"}),
        )
        with self.assertRaises(GitHubError) as ctx:
            conn.get_public_repo("Oluwacutyp/goldenswing")
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertIn("rate limit", str(ctx.exception).lower())

    def test_get_public_repo_bad_name_never_calls_api(self) -> None:
        conn, http = _github()
        for bad in ("not-a-repo", "a/b/c", "/goldenswing", "Oluwacutyp/"):
            with self.assertRaises(GitHubError):
                conn.get_public_repo(bad)
        self.assertEqual(http.calls, [])

    def test_list_public_repos_returns_list(self) -> None:
        conn, http = _github()
        _route_user_repos(http)
        repos = conn.list_public_repos("Oluwacutyp", limit=3)
        self.assertEqual(len(repos), 3)
        self.assertEqual(repos[0]["full_name"], "Oluwacutyp/nomorals-2.0")
        self.assertTrue(all(r["private"] is False for r in repos))

    def test_list_public_repos_passes_sort_and_limit(self) -> None:
        conn, http = _github()
        _route_user_repos(http)
        conn.list_public_repos("Oluwacutyp", limit=3)
        params = http.calls[0]["params"]
        self.assertEqual(params["sort"], "updated")
        self.assertEqual(params["per_page"], 3)

    def test_list_public_repos_limit_clamped(self) -> None:
        conn, http = _github()
        _route_user_repos(http)
        conn.list_public_repos("Oluwacutyp", limit=500)
        self.assertEqual(http.calls[0]["params"]["per_page"], 100)
        conn.list_public_repos("Oluwacutyp", limit=0)
        self.assertEqual(http.calls[1]["params"]["per_page"], 1)

    def test_list_public_repos_bad_username_never_calls_api(self) -> None:
        conn, http = _github()
        with self.assertRaises(GitHubError):
            conn.list_public_repos("bad user!")
        self.assertEqual(http.calls, [])

    def test_list_public_repos_404(self) -> None:
        conn, http = _github()
        http.route("GET", "/users/ghost123/repos",
                   FakeResponse(404, {"message": "Not Found"}))
        with self.assertRaises(GitHubError) as ctx:
            conn.list_public_repos("ghost123")
        self.assertEqual(ctx.exception.status_code, 404)


class PublicActionsTests(unittest.TestCase):
    def test_public_actions_declared(self) -> None:
        self.assertIn("get_public_repo", GitHubConnector.PUBLIC_ACTIONS)
        self.assertIn("list_public_repos", GitHubConnector.PUBLIC_ACTIONS)
        self.assertNotIn("get_repo", GitHubConnector.PUBLIC_ACTIONS)
        self.assertNotIn("create_repo", GitHubConnector.PUBLIC_ACTIONS)

    def test_service_list_discovers_public_actions(self) -> None:
        registry = ToolRegistry()
        services.register(registry)
        out = registry.call("service_list", context=SimpleNamespace())
        self.assertTrue(out.ok, out)
        gh = next(s for s in out.value["services"] if s["id"] == "github")
        actions = {a["action"]: a for a in gh["actions"]}
        self.assertIn("get_public_repo", actions)
        self.assertIn("list_public_repos", actions)
        self.assertIn("full_name", actions["get_public_repo"]["params"])
        self.assertFalse(actions["get_public_repo"]["sensitive"])

    def test_service_call_runs_public_action_without_vault(self) -> None:
        conn, http = _github()
        _route_repo(http)
        registry = ToolRegistry()
        services.register(registry)
        ctx = SimpleNamespace(capabilities=None, vault=None, db=None)
        with mock.patch(
            "nomorals.connectors.registry.create_connector",
            return_value=conn,
        ):
            out = registry.call(
                "service_call", context=ctx, connector="github",
                action="get_public_repo",
                params={"full_name": "Oluwacutyp/goldenswing"},
            )
        self.assertTrue(out.ok, out)
        self.assertTrue(out.value["ok"], out.value)
        self.assertEqual(
            out.value["result"]["full_name"], "Oluwacutyp/goldenswing"
        )

    def test_service_call_still_gates_token_actions(self) -> None:
        conn, http = _github()
        _route_user(http)
        registry = ToolRegistry()
        services.register(registry)
        ctx = SimpleNamespace(capabilities=None, vault=None, db=None)
        with mock.patch(
            "nomorals.connectors.registry.create_connector",
            return_value=conn,
        ):
            out = registry.call(
                "service_call", context=ctx, connector="github",
                action="get_repo",
                params={"full_name": "Oluwacutyp/goldenswing"},
            )
        self.assertTrue(out.ok, out)
        self.assertFalse(out.value["ok"], out.value)
        # The token action never reached the API: no vault, no connect.
        self.assertEqual(http.calls, [])


def _route_user(http: FakeHttp) -> None:
    http.route("GET", "/user",
               FakeResponse(200, {"login": "octocat", "id": 1}))


@unittest.skipUnless(
    os.environ.get("NM_LIVE_GITHUB") == "1",
    "live GitHub test — set NM_LIVE_GITHUB=1 to run",
)
class LivePublicReadTests(unittest.TestCase):
    """One real call against api.github.com (no token, no vault).

    Two properties of THIS box (not of the connector) need relaxing
    here, and only here — production code is unchanged:

    * The box resolves api.github.com to a CGNAT address (198.18.x.x),
      which Python's ipaddress flags as private, so the SSRF guard
      refuses the call — ``allow_private_ips=True`` relaxes that.
    * Devon's HttpClient does not read the standard proxy env vars;
      direct egress from this box fails, so the box's egress proxy is
      passed explicitly from the environment (never embedded — the URL
      carries a credential). On EC2 (real public DNS, direct egress)
      the default client works and neither relaxation is needed.
    """

    def _live_connector(self) -> GitHubConnector:
        from nomorals.core.http import HttpClient

        return GitHubConnector(
            _vault(),
            http=HttpClient(
                allow_private_ips=True,
                proxy_url=os.environ.get("HTTPS_PROXY", "")
                or os.environ.get("https_proxy", ""),
            ),
        )

    def test_live_get_public_repo(self) -> None:
        conn = self._live_connector()
        repo = conn.get_public_repo("Oluwacutyp/goldenswing")
        self.assertEqual(repo["full_name"], "Oluwacutyp/goldenswing")
        self.assertFalse(repo["private"])

    def test_live_list_public_repos(self) -> None:
        conn = self._live_connector()
        repos = conn.list_public_repos("Oluwacutyp", limit=5)
        self.assertIsInstance(repos, list)
        self.assertGreater(len(repos), 0)
        self.assertTrue(all(
            isinstance(r, dict) and r.get("private") is False
            for r in repos
        ))


if __name__ == "__main__":
    unittest.main()

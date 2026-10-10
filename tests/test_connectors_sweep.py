"""Sweep tests: framework spine, webhooks, presentation, and the new
connector features (GitHub issues/PRs/files, Telegram media, proxy
strategies, Stripe idempotency). All HTTP is mocked — no network."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import unittest
import urllib.parse
from types import SimpleNamespace
from typing import Any

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors import (
    WebhookDeduper,
    describe_connector,
    health_snapshot,
    list_connectors,
    new_state,
    parse_event,
    pkce_pair,
    render_capabilities,
    render_checkpoint_list,
    render_connect_guide,
    render_connector_table,
    render_health_snapshot,
    render_status,
    search_connectors,
    verify_signature,
)
from nomorals.connectors.auth import device_flow_token  # noqa: F401 (import surface)
from nomorals.connectors.base import (
    ConnectorAuthError,
    ConnectorError,
    ConnectorNetworkError,
    ConnectorNotFoundError,
    ConnectorRateLimitError,
    ConnectorStatus,
    ConnectorValidationError,
    paginate,
    request_with_retry,
)
from nomorals.connectors.checkpoints import CheckpointKind, CheckpointStore
from nomorals.connectors.github import GitHubConnector
from nomorals.connectors.proxypool import ProxyPoolConnector
from nomorals.connectors.stripe import StripeConnector
from nomorals.connectors.telegram import TelegramConnector
from nomorals.core.errors import NotFound
from nomorals.storage.db import Database


def _vault() -> CredentialVault:
    return CredentialVault(Database(":memory:"), master_passphrase="test")


class FakeResponse:
    def __init__(self, status: int = 200, payload: Any = None,
                 headers: dict[str, str] | None = None) -> None:
        self.status = status
        self._payload = payload
        self.headers = dict(headers or {})

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    @property
    def text(self) -> str:
        return json.dumps(self._payload)

    def json(self) -> Any:
        return self._payload


class FakeHttp:
    """Scripted HttpClient stand-in. Records calls; no network."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.routes: list[tuple[str, str, FakeResponse]] = []

    def route(self, method: str, path: str, response: FakeResponse) -> None:
        self.routes.append((method.upper(), path, response))

    def _dispatch(self, method: str, url: str, payload: Any = None,
                  **kw: Any) -> FakeResponse:
        self.calls.append({"method": method.upper(), "url": url,
                           "payload": payload, "kw": kw})
        for rm, rp, resp in sorted(self.routes, key=lambda r: -len(r[1])):
            if rm == method.upper() and rp in url:
                return resp
        return FakeResponse(404, {"message": "not mocked"})

    def get(self, url: str, **kw: Any) -> FakeResponse:
        params = kw.pop("params", None) or {}
        if params:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{urllib.parse.urlencode(params)}"
        return self._dispatch("GET", url, None, **kw)

    def post_json(self, url: str, payload: Any, **kw: Any) -> FakeResponse:
        return self._dispatch("POST", url, payload, **kw)

    def put_json(self, url: str, payload: Any, **kw: Any) -> FakeResponse:
        return self._dispatch("PUT", url, payload, **kw)

    def post_form(self, url: str, form: Any, **kw: Any) -> FakeResponse:
        return self._dispatch("POST", url, form, **kw)

    def post_multipart(self, url: str, fields: Any = None,
                       files: Any = None, **kw: Any) -> FakeResponse:
        return self._dispatch("MULTIPART", url,
                              {"fields": fields, "files": files}, **kw)

    def request(self, method: str, url: str, **kw: Any) -> FakeResponse:
        return self._dispatch(method, url, kw.get("data"), **kw)


def _store(conn: Any, username: str = "owner", secret: str = "secret",
           ctype: str = "api_key") -> None:
    conn._store_credential(username, secret, credential_type=ctype)


# ── base: error taxonomy ──────────────────────────────────────────

class TestErrorTaxonomy(unittest.TestCase):
    def test_hierarchy(self) -> None:
        for cls in (ConnectorAuthError, ConnectorRateLimitError,
                    ConnectorNetworkError, ConnectorNotFoundError,
                    ConnectorValidationError):
            self.assertTrue(issubclass(cls, ConnectorError), cls)
        # NotFound interop: `except NotFound` keeps catching it.
        self.assertTrue(issubclass(ConnectorNotFoundError, NotFound))

    def test_rate_limit_carries_wait(self) -> None:
        err = ConnectorRateLimitError("slow down", retry_after=7.5)
        self.assertEqual(err.retry_after, 7.5)
        self.assertEqual(ConnectorRateLimitError("x").retry_after, 0.0)
        self.assertEqual(ConnectorRateLimitError("x", retry_after=-3).retry_after,
                         0.0)


# ── base: request_with_retry ──────────────────────────────────────

class TestRequestWithRetry(unittest.TestCase):
    def test_eventual_success(self) -> None:
        attempts = []

        def do() -> str:
            attempts.append(1)
            if len(attempts) < 3:
                raise ConnectorRateLimitError("429", retry_after=0.01)
            return "ok"

        self.assertEqual(
            request_with_retry(do, op="t", base_delay=0.01, max_delay=0.02),
            "ok")
        self.assertEqual(len(attempts), 3)

    def test_auth_fails_fast(self) -> None:
        calls = []

        def do() -> None:
            calls.append(1)
            raise ConnectorAuthError("bad token")

        with self.assertRaises(ConnectorAuthError):
            request_with_retry(do, op="t", base_delay=0.01)
        self.assertEqual(len(calls), 1)

    def test_validation_fails_fast(self) -> None:
        with self.assertRaises(ConnectorValidationError):
            request_with_retry(
                lambda: (_ for _ in ()).throw(
                    ConnectorValidationError("bad args")),
                op="t", base_delay=0.01)

    def test_retryable_status_then_give_up(self) -> None:
        class E(Exception):
            status = 503

        with self.assertRaises(E):
            request_with_retry(lambda: (_ for _ in ()).throw(E("down")),
                               op="t", max_attempts=2, base_delay=0.01,
                               max_delay=0.02)

    def test_non_retryable_status_raises_immediately(self) -> None:
        class E(Exception):
            status = 400

        with self.assertRaises(E):
            request_with_retry(lambda: (_ for _ in ()).throw(E("bad")),
                               op="t", base_delay=0.01)


# ── base: paginate ────────────────────────────────────────────────

class TestPaginate(unittest.TestCase):
    def test_link_style_two_pages(self) -> None:
        pages = {
            "": ([1, 2], {"link": '<https://x/p2>; rel="next"'}),
            "https://x/p2": ([3], {}),
        }

        def fetch(url: str = "") -> tuple[list[int], dict[str, str]]:
            return pages[url]

        self.assertEqual(list(paginate(fetch, style="link")), [1, 2, 3])

    def test_link_style_bounded(self) -> None:
        def fetch(url: str = "") -> tuple[list[int], dict[str, str]]:
            nxt = "https://x/more"
            return [1], {"link": f'<{nxt}>; rel="next"'}

        out = list(paginate(fetch, style="link", max_pages=5))
        self.assertEqual(out, [1] * 5)

    def test_page_style_short_page_stops(self) -> None:
        calls = []

        def fetch(page: int, per_page: int) -> list[int]:
            calls.append(page)
            return [1, 2] if page == 1 else []

        out = list(paginate(fetch, style="page", per_page=5))
        self.assertEqual(out, [1, 2])
        self.assertEqual(calls, [1])

    def test_page_style_dict_items_key(self) -> None:
        def fetch(page: int, per_page: int) -> dict[str, Any]:
            return {"items": [{"n": page}]} if page == 1 else {"items": []}

        out = list(paginate(fetch, style="page", per_page=10,
                            items_key="items"))
        self.assertEqual(out, [{"n": 1}])


# ── registry ──────────────────────────────────────────────────────

class TestRegistry(unittest.TestCase):
    def test_search(self) -> None:
        ids = [i["id"] for i in search_connectors("pay")]
        self.assertIn("stripe", ids)
        self.assertIn("paystack", ids)
        self.assertEqual(len(search_connectors("")),
                         len(list_connectors()))
        self.assertEqual(search_connectors("zzz-no-such"), [])

    def test_search_matches_category(self) -> None:
        ids = [i["id"] for i in search_connectors("messaging")]
        self.assertIn("telegram", ids)

    def test_describe(self) -> None:
        d = describe_connector("github")
        self.assertEqual(d["id"], "github")
        self.assertEqual(d["category"], "dev")
        self.assertIn("create_issue", d["features"])
        self.assertIn("list_pull_requests", d["features"])
        with self.assertRaises(ConnectorError):
            describe_connector("nope")

    def test_health_snapshot(self) -> None:
        ok_status = ConnectorStatus(connected=True, account="a@b.c",
                                    detail="fine")

        def factory(cid: str, vault: Any) -> Any:
            if cid == "boom":
                raise RuntimeError("kaput")
            m = SimpleNamespace()
            m.name = cid.title()
            m.status = lambda: ok_status  # noqa: E731
            return m

        snap = health_snapshot(None, ids=["github", "boom"],
                               factory=factory)
        self.assertEqual(snap["total"], 2)
        self.assertEqual(snap["connected"], 1)
        self.assertEqual(snap["disconnected"], 1)
        by_id = {d["id"]: d for d in snap["details"]}
        self.assertTrue(by_id["github"]["connected"])
        self.assertIn("kaput", by_id["boom"]["detail"])

    def test_health_snapshot_empty(self) -> None:
        snap = health_snapshot(None, ids=[], factory=lambda c, v: None)
        self.assertEqual(snap["total"], 0)


# ── auth: PKCE + state ────────────────────────────────────────────

class TestPkce(unittest.TestCase):
    def test_pair_shape(self) -> None:
        verifier, challenge = pkce_pair()
        self.assertTrue(43 <= len(verifier) <= 128)
        expect = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        self.assertEqual(challenge, expect)

    def test_pair_unique(self) -> None:
        self.assertNotEqual(pkce_pair()[0], pkce_pair()[0])

    def test_new_state(self) -> None:
        s1, s2 = new_state(), new_state()
        self.assertTrue(len(s1) >= 32)
        self.assertNotEqual(s1, s2)

    def test_google_authorize_url_pkce(self) -> None:
        from nomorals.connectors._google_oauth import GoogleOAuth
        mix = GoogleOAuth()
        verifier, challenge = pkce_pair()
        state = new_state()
        url = mix.google_authorize_url(
            "cid", ["scope1"], redirect_uri="http://127.0.0.1:9/",
            code_challenge=challenge, state=state)
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        self.assertEqual(q["code_challenge"][0], challenge)
        self.assertEqual(q["code_challenge_method"][0], "S256")
        self.assertEqual(q["state"][0], state)
        # Without PKCE the params are absent (backward compatible).
        url2 = mix.google_authorize_url("cid", ["scope1"])
        self.assertNotIn("code_challenge", url2)


# ── webhooks ──────────────────────────────────────────────────────

class TestWebhooks(unittest.TestCase):
    def _stripe_sig(self, payload: bytes, secret: str,
                    ts: float | None = None) -> str:
        ts = time.time() if ts is None else ts
        mac = hmac.new(secret.encode(), f"{ts:.0f}.".encode() + payload,
                       hashlib.sha256).hexdigest()
        return f"t={ts:.0f},v1={mac}"

    def test_stripe_ok(self) -> None:
        body = b'{"id":"evt_1","type":"charge.succeeded"}'
        sig = self._stripe_sig(body, "whsec_x")
        verify_signature("stripe", body, sig, "whsec_x")  # no raise

    def test_stripe_tampered(self) -> None:
        body = b'{"id":"evt_1"}'
        sig = self._stripe_sig(b'{"id":"evt_2"}', "whsec_x")
        with self.assertRaises(ConnectorValidationError):
            verify_signature("stripe", body, sig, "whsec_x")

    def test_stripe_stale_rejected(self) -> None:
        body = b'{"id":"evt_1"}'
        sig = self._stripe_sig(body, "whsec_x",
                               ts=time.time() - 3600)
        with self.assertRaises(ConnectorValidationError):
            verify_signature("stripe", body, sig, "whsec_x")

    def test_stripe_malformed(self) -> None:
        with self.assertRaises(ConnectorValidationError):
            verify_signature("stripe", b"{}", "garbage", "whsec_x")

    def test_paystack_ok_and_bad(self) -> None:
        body = b'{"event":"charge.success"}'
        good = hmac.new(b"s", body, hashlib.sha512).hexdigest()
        verify_signature("paystack", body, good, "s")
        with self.assertRaises(ConnectorValidationError):
            verify_signature("paystack", body, "0" * 128, "s")

    def test_github_ok_and_malformed(self) -> None:
        body = b'{"zen":"hi"}'
        good = "sha256=" + hmac.new(b"s", body, hashlib.sha256).hexdigest()
        verify_signature("github", body, good, "s")
        with self.assertRaises(ConnectorValidationError):
            verify_signature("github", body, "md5=abc", "s")
        with self.assertRaises(ConnectorValidationError):
            verify_signature("github", body, "sha256=" + "0" * 64, "s")

    def test_unknown_provider(self) -> None:
        with self.assertRaises(ConnectorError):
            verify_signature("acme", b"{}", "x", "s")

    def test_no_secret(self) -> None:
        with self.assertRaises(ConnectorError):
            verify_signature("stripe", b"{}", "t=1,v1=2", "")

    def test_parse_event(self) -> None:
        ev = parse_event("stripe", {"id": "evt_9", "type": "x.y",
                                    "created": 123,
                                    "data": {"object": {"a": 1}}})
        self.assertEqual(ev.id, "evt_9")
        self.assertEqual(ev.type, "x.y")
        self.assertEqual(ev.data, {"a": 1})
        ev2 = parse_event("paystack", b'{"event":"charge.success",'
                                     b'"data":{"reference":"r1"}}')
        self.assertEqual(ev2.type, "charge.success")
        with self.assertRaises(ConnectorValidationError):
            parse_event("stripe", b"not json")

    def test_deduper(self) -> None:
        d = WebhookDeduper(ttl_seconds=100)
        self.assertFalse(d.check_and_mark("evt_1"))
        self.assertTrue(d.check_and_mark("evt_1"))  # duplicate
        self.assertTrue(d.seen("evt_1"))
        self.assertFalse(d.seen("evt_2"))
        self.assertEqual(len(d), 1)
        # TTL expiry
        d2 = WebhookDeduper(ttl_seconds=100)
        d2.check_and_mark("e", now=1000.0)
        self.assertEqual(d2.purge(now=1000.0 + 200.0), 1)
        self.assertFalse(d2.seen("e"))


# ── presentation ──────────────────────────────────────────────────

class TestPresent(unittest.TestCase):
    def test_render_status(self) -> None:
        st = ConnectorStatus(connected=True, account="bot@example",
                             scopes=["send"], detail="responding")
        out = render_status("Telegram", st)
        self.assertIn("Telegram", out)
        self.assertIn("connected", out)
        self.assertIn("bot@example", out)
        st2 = ConnectorStatus(connected=False, detail="no token")
        self.assertIn("disconnected", render_status("X", st2))

    def test_render_health_snapshot(self) -> None:
        snap = {"total": 3, "connected": 2, "disconnected": 1,
                "checked_at": 0.0,
                "details": [
                    {"id": "a", "connected": True, "account": "x",
                     "detail": "", "latency_ms": 12.0},
                    {"id": "b", "connected": False, "account": None,
                     "detail": "token dead", "latency_ms": 3.0},
                    {"id": "c", "connected": True, "account": "y",
                     "detail": "", "latency_ms": 5.0},
                ]}
        out = render_health_snapshot(snap)
        self.assertIn("2/3", out)
        self.assertIn("token dead", out)

    def test_render_connector_table(self) -> None:
        infos = [{"id": "github", "name": "GitHub", "category": "dev",
                  "auth_methods": ["pat"], "description": "code"}]
        out = render_connector_table(infos)
        self.assertIn("github", out)
        self.assertIn("GitHub", out)
        statuses = {"github": {"connected": True, "account": "octocat",
                               "detail": ""}}
        out2 = render_connector_table(infos, statuses)
        self.assertIn("octocat", out2)

    def test_render_capabilities(self) -> None:
        d = describe_connector("telegram")
        out = render_capabilities(d)
        self.assertIn("Telegram", out)
        self.assertIn("send_photo", out)

    def test_render_connect_guide(self) -> None:
        out = render_connect_guide("GitHub", ["Get a token", "Run connect"],
                                   note="tokens stay vaulted")
        self.assertIn("1.", out)
        self.assertIn("Get a token", out)
        self.assertIn("tokens stay vaulted", out)

    def test_render_checkpoint_list(self) -> None:
        cp = SimpleNamespace(id="chk_1", connector_id="github",
                             kind=CheckpointKind.CAPTCHA, title="Solve me",
                             created_at=time.time() - 90)
        out = render_checkpoint_list([cp])
        self.assertIn("chk_1", out)
        self.assertIn("Solve me", out)
        self.assertIn("nothing waiting", render_checkpoint_list([]))

    def test_plain_when_not_tty(self) -> None:
        # Test runner stdout is not a TTY → no ANSI escapes.
        out = render_status("T", ConnectorStatus(connected=True))
        self.assertNotIn("\033[", out)


# ── GitHub: issues / PRs / files ──────────────────────────────────

ISSUES_PAGE_1 = [
    {"number": 3, "title": "Bug", "state": "open",
     "labels": [{"name": "bug"}], "user": {"login": "a"},
     "comments": 1, "created_at": "t", "updated_at": "t",
     "html_url": "u3", "body": "b3"},
    {"number": 2, "title": "PR thing", "state": "open",
     "labels": [], "user": {"login": "b"}, "comments": 0,
     "created_at": "t", "updated_at": "t", "html_url": "u2",
     "body": "b2", "pull_request": {"url": "x"}},
]
ISSUES_PAGE_2 = [
    {"number": 1, "title": "Old", "state": "open", "labels": [],
     "user": {"login": "c"}, "comments": 0, "created_at": "t",
     "updated_at": "t", "html_url": "u1", "body": "b1"},
]


def _github() -> tuple[GitHubConnector, FakeHttp]:
    http = FakeHttp()
    conn = GitHubConnector(_vault(), http=http)
    _store(conn, "octocat", "tok123", "pat")
    return conn, http


class TestGitHubSweep(unittest.TestCase):
    def test_list_issues_filters_prs_and_paginates(self) -> None:
        conn, http = _github()
        http.route("GET", "/repos/o/r/issues",
                   FakeResponse(200, ISSUES_PAGE_1, headers={
                       "link": '<https://api.github.com/repos/o/r/issues?page=2>; rel="next"'}))
        http.route("GET", "https://api.github.com/repos/o/r/issues?page=2",
                   FakeResponse(200, ISSUES_PAGE_2, headers={}))
        issues = conn.list_issues("o/r", limit=10)
        numbers = [i["number"] for i in issues]
        self.assertEqual(numbers, [3, 1])  # PR #2 filtered out
        self.assertFalse(any(i["is_pull_request"] for i in issues))

    def test_create_and_close_issue(self) -> None:
        conn, http = _github()
        http.route("POST", "/repos/o/r/issues",
                   FakeResponse(201, {"number": 9, "title": "Hi",
                                      "state": "open", "labels": [],
                                      "user": {"login": "a"}, "comments": 0,
                                      "created_at": "t", "updated_at": "t",
                                      "html_url": "u", "body": ""}))
        created = conn.create_issue("o/r", "Hi", labels=["bug"])
        self.assertEqual(created["number"], 9)
        posted = http.calls[-1]["payload"]
        self.assertEqual(posted["title"], "Hi")
        self.assertEqual(posted["labels"], ["bug"])
        http.route("PATCH", "/repos/o/r/issues/9",
                   FakeResponse(200, {"number": 9, "title": "Hi",
                                      "state": "closed"}))
        closed = conn.close_issue("o/r", 9)
        self.assertEqual(closed["state"], "closed")

    def test_comment_issue_rejects_empty(self) -> None:
        conn, _ = _github()
        with self.assertRaises(ConnectorError):
            conn.comment_issue("o/r", 1, "   ")

    def test_pull_requests(self) -> None:
        conn, http = _github()
        pr = {"number": 7, "title": "Feat", "state": "open",
              "user": {"login": "dev"}, "head": {"ref": "feat"},
              "base": {"ref": "main"}, "mergeable": True, "merged": False,
              "additions": 10, "deletions": 2, "changed_files": 1,
              "created_at": "t", "html_url": "u", "body": "b"}
        http.route("GET", "/repos/o/r/pulls",
                   FakeResponse(200, [pr], headers={}))
        prs = conn.list_pull_requests("o/r")
        self.assertEqual(prs[0]["head"], "feat")
        http.route("GET", "/repos/o/r/pulls/7", FakeResponse(200, pr))
        one = conn.get_pull_request("o/r", 7)
        self.assertEqual(one["number"], 7)
        http.route("POST", "/repos/o/r/pulls",
                   FakeResponse(201, pr))
        created = conn.create_pull_request("o/r", "feat", "main", "Feat")
        self.assertEqual(created["number"], 7)
        http.route("PUT", "/repos/o/r/pulls/7/merge",
                   FakeResponse(200, {"merged": True}))
        merged = conn.merge_pull_request("o/r", 7, method="squash")
        self.assertTrue(merged["merged"])
        self.assertEqual(http.calls[-1]["payload"]["merge_method"], "squash")
        with self.assertRaises(ConnectorError):
            conn.merge_pull_request("o/r", 7, method="nope")

    def test_file_content_roundtrip(self) -> None:
        conn, http = _github()
        raw_text = "hello world"
        http.route("GET", "/repos/o/r/contents/README.md",
                   FakeResponse(200, {
                       "type": "file", "path": "README.md", "sha": "abc",
                       "size": 11, "encoding": "base64",
                       "content": base64.b64encode(
                           raw_text.encode()).decode(),
                       "html_url": "u"}))
        got = conn.get_file_content("o/r", "README.md")
        self.assertEqual(got["content"], raw_text)
        self.assertEqual(got["sha"], "abc")
        http.route("PUT", "/repos/o/r/contents/README.md",
                   FakeResponse(200, {"content": {"sha": "def"}}))
        updated = conn.create_or_update_file(
            "o/r", "README.md", "hello v2", "update readme", sha="abc")
        self.assertEqual(updated["content"]["sha"], "def")
        sent = http.calls[-1]["payload"]
        self.assertEqual(sent["sha"], "abc")
        self.assertEqual(base64.b64decode(sent["content"]).decode(),
                         "hello v2")

    def test_search(self) -> None:
        conn, http = _github()
        http.route("GET", "/search/repositories",
                   FakeResponse(200, {"items": [
                       {"full_name": "a/b", "description": "d",
                        "stargazers_count": 5, "language": "Python",
                        "updated_at": "t", "html_url": "u"}]}))
        repos = conn.search_repositories("topic:cli")
        self.assertEqual(repos[0]["full_name"], "a/b")
        self.assertEqual(repos[0]["stars"], 5)

    def test_rate_limit_taxonomy(self) -> None:
        conn, http = _github()
        http.route("GET", "/repos/o/r",
                   FakeResponse(429, {"message": "slow"},
                                headers={"retry-after": "4"}))
        with self.assertRaises(ConnectorRateLimitError) as ctx:
            conn.get_repo("o/r")
        self.assertEqual(ctx.exception.retry_after, 4.0)

    def test_verify_webhook_signature(self) -> None:
        conn, _ = _github()
        body = b'{"action":"opened"}'
        good = "sha256=" + hmac.new(b"s", body, hashlib.sha256).hexdigest()
        conn.verify_webhook_signature(body, good, "s")  # no raise
        with self.assertRaises(ConnectorValidationError):
            conn.verify_webhook_signature(body, "sha256=" + "0" * 64, "s")


# ── Telegram: media ───────────────────────────────────────────────

class TestTelegramSweep(unittest.TestCase):
    def _tg(self) -> tuple[TelegramConnector, FakeHttp]:
        http = FakeHttp()
        conn = TelegramConnector(_vault(), http=http)
        _store(conn, "@devon_test_bot", "tok123")
        return conn, http

    def test_send_photo_url(self) -> None:
        conn, http = self._tg()
        http.route("POST", "/sendPhoto",
                   FakeResponse(200, {"ok": True,
                                      "result": {"message_id": 1}}))
        out = conn.send_photo(123, "https://x/y.png", caption="hi")
        self.assertEqual(out["message_id"], 1)
        call = http.calls[-1]
        self.assertEqual(call["payload"]["photo"], "https://x/y.png")

    def test_send_photo_upload(self) -> None:
        import tempfile, os
        conn, http = self._tg()
        http.route("MULTIPART", "/sendPhoto",
                   FakeResponse(200, {"ok": True,
                                      "result": {"message_id": 2}}))
        with tempfile.NamedTemporaryFile(suffix=".png",
                                         delete=False) as fh:
            fh.write(b"\x89PNG fake")
            path = fh.name
        try:
            out = conn.send_photo(123, path, caption="up")
        finally:
            os.unlink(path)
        self.assertEqual(out["message_id"], 2)
        call = http.calls[-1]
        self.assertEqual(call["method"], "MULTIPART")
        uploaded = call["payload"]["files"]
        self.assertEqual(uploaded[0][0], "photo")
        self.assertTrue(str(uploaded[0][1]).endswith(".png"))

    def test_send_document_and_video(self) -> None:
        conn, http = self._tg()
        http.route("POST", "/sendDocument",
                   FakeResponse(200, {"ok": True, "result": {"message_id": 3}}))
        conn.send_document(123, "file_id:ABC")
        self.assertEqual(http.calls[-1]["payload"]["document"], "file_id:ABC")
        http.route("POST", "/sendVideo",
                   FakeResponse(200, {"ok": True, "result": {"message_id": 4}}))
        conn.send_video(123, "https://x/v.mp4")
        self.assertEqual(http.calls[-1]["payload"]["video"],
                         "https://x/v.mp4")

    def test_caption_limit(self) -> None:
        conn, _ = self._tg()
        with self.assertRaises(ConnectorError):
            conn.send_photo(123, "https://x/y.png", caption="x" * 1025)

    def test_edit_delete_answer(self) -> None:
        conn, http = self._tg()
        http.route("POST", "/editMessageText",
                   FakeResponse(200, {"ok": True, "result": {"message_id": 5}}))
        conn.edit_message_text(123, 5, "new text")
        self.assertEqual(http.calls[-1]["payload"]["text"], "new text")
        with self.assertRaises(ConnectorError):
            conn.edit_message_text(123, 5, "")
        http.route("POST", "/deleteMessage",
                   FakeResponse(200, {"ok": True, "result": True}))
        self.assertTrue(conn.delete_message(123, 5))
        http.route("POST", "/answerCallbackQuery",
                   FakeResponse(200, {"ok": True, "result": True}))
        self.assertTrue(conn.answer_callback_query("cq1", text="done"))

    def test_webhook_management(self) -> None:
        conn, http = self._tg()
        http.route("POST", "/setWebhook",
                   FakeResponse(200, {"ok": True, "result": True}))
        self.assertTrue(conn.set_webhook("https://x/hook",
                                         secret_token="s3cret"))
        self.assertEqual(http.calls[-1]["payload"]["secret_token"], "s3cret")
        http.route("POST", "/getWebhookInfo",
                   FakeResponse(200, {"ok": True,
                                      "result": {"url": "https://x/hook"}}))
        info = conn.get_webhook_info()
        self.assertEqual(info["url"], "https://x/hook")

    def test_rate_limit_taxonomy(self) -> None:
        conn, http = self._tg()
        http.route("POST", "/sendMessage",
                   FakeResponse(429, {"ok": False, "error_code": 429,
                                      "description": "Too Many Requests",
                                      "parameters": {"retry_after": 3}}))
        with self.assertRaises(ConnectorRateLimitError) as ctx:
            conn.send_message(123, "hi")
        self.assertEqual(ctx.exception.retry_after, 3.0)


# ── Proxy pool: strategies / sticky / quarantine / stats ──────────

class TestProxyPoolSweep(unittest.TestCase):
    def _pool(self) -> ProxyPoolConnector:
        return ProxyPoolConnector(_vault())

    def _seed(self, pool: ProxyPoolConnector) -> list[str]:
        ids = []
        for host, country, latency in (("1.1.1.1", "US", 80.0),
                                      ("2.2.2.2", "DE", 400.0),
                                      ("3.3.3.3", "US", 1500.0)):
            view = pool.add_proxy(host, 8080, protocol="http",
                                  country=country)
            pid = view["id"]
            pool._record_health(pid, ok=True, latency_ms=latency,
                                status_code=200, error="")
            ids.append(pid)
        return ids

    def test_geo_and_score(self) -> None:
        pool = self._pool()
        ids = self._seed(pool)
        views = {v["id"]: v for v in pool.list_proxies()}
        self.assertEqual(views[ids[0]]["country"], "US")
        # Fast proxy scores higher than slow one.
        self.assertGreater(views[ids[0]]["health"]["score"],
                           views[ids[2]]["health"]["score"])
        # EWMA tracked.
        self.assertAlmostEqual(views[ids[0]]["health"]["latency_ewma_ms"],
                               80.0)

    def test_select_strategies(self) -> None:
        pool = self._pool()
        self._seed(pool)
        least = pool.select("least_latency")
        self.assertEqual(least["host"], "1.1.1.1")
        de = pool.select("random", country="DE")
        self.assertEqual(de["host"], "2.2.2.2")
        fast = pool.select("p2c", max_latency_ms=500.0)
        self.assertIn(fast["host"], ("1.1.1.1", "2.2.2.2"))
        with self.assertRaises(ConnectorError):
            pool.select("nope")
        with self.assertRaises(ConnectorError):
            pool.select("random", country="XX")

    def test_round_robin_cycles(self) -> None:
        pool = self._pool()
        self._seed(pool)
        seen = [pool.select("round_robin")["host"] for _ in range(4)]
        self.assertEqual(seen[:3], ["1.1.1.1", "2.2.2.2", "3.3.3.3"])
        self.assertEqual(seen[3], "1.1.1.1")

    def test_sticky_pins(self) -> None:
        pool = self._pool()
        self._seed(pool)
        first = pool.sticky_acquire("sess-1")
        second = pool.sticky_acquire("sess-1")
        self.assertEqual(first["id"], second["id"])
        self.assertTrue(second["sticky"])
        self.assertTrue(pool.sticky_release("sess-1"))
        self.assertFalse(pool.sticky_release("sess-1"))
        with self.assertRaises(ConnectorError):
            pool.sticky_acquire("")

    def test_sticky_repins_when_dead(self) -> None:
        pool = self._pool()
        ids = self._seed(pool)
        pinned = pool.sticky_acquire("sess-9", strategy="least_latency")
        self.assertEqual(pinned["host"], "1.1.1.1")
        pool.quarantine(ids[0], seconds=600, reason="blocked")
        repinned = pool.sticky_acquire("sess-9", strategy="least_latency")
        self.assertNotEqual(repinned["id"], ids[0])

    def test_quarantine_skipped_by_select(self) -> None:
        pool = self._pool()
        ids = self._seed(pool)
        pool.quarantine(ids[0], seconds=600)
        pool.quarantine(ids[1], seconds=600)
        only = pool.select("least_latency")
        self.assertEqual(only["id"], ids[2])
        pool.quarantine(ids[2], seconds=600)
        with self.assertRaises(ConnectorError):
            pool.select("least_latency")

    def test_auto_quarantine_on_repeated_failure(self) -> None:
        pool = self._pool()
        ids = self._seed(pool)
        for _ in range(3):
            pool.record_failure(ids[0], "timeout")
        views = {v["id"]: v for v in pool.list_proxies()}
        self.assertGreater(views[ids[0]]["health"]["quarantined_until"],
                           time.time())
        # ...but a success clears it.
        pool.record_success(ids[0], latency_ms=50.0)
        views = {v["id"]: v for v in pool.list_proxies()}
        self.assertEqual(views[ids[0]]["health"]["quarantined_until"], 0.0)

    def test_stats(self) -> None:
        pool = self._pool()
        self._seed(pool)
        s = pool.stats()
        self.assertEqual(s["total"], 3)
        self.assertEqual(s["healthy"], 3)
        self.assertEqual(s["countries"], {"DE": 1, "US": 2})
        self.assertEqual(s["protocols"], {"http": 3})
        self.assertIsNotNone(s["avg_latency_ms"])
        self.assertIsNotNone(s["avg_score"])


# ── Stripe: idempotency + webhooks ─────────────────────────────────

class TestStripeSweep(unittest.TestCase):
    def _stripe(self) -> tuple[StripeConnector, FakeHttp]:
        http = FakeHttp()
        conn = StripeConnector(_vault(), http=http)
        _store(conn, "acct", "sk_test_123")
        return conn, http

    def test_idempotency_key_sent(self) -> None:
        conn, http = self._stripe()
        http.route("POST", "/v1/customers",
                   FakeResponse(200, {"id": "cus_1"}))
        conn.create_customer(name="Ada")
        headers = http.calls[-1]["kw"].get("headers", {})
        self.assertIn("Idempotency-Key", headers)
        key1 = headers["Idempotency-Key"]
        # Same logical operation retried → same key (safe replay).
        conn.create_customer(name="Ada")
        key2 = http.calls[-1]["kw"].get("headers", {})["Idempotency-Key"]
        self.assertEqual(key1, key2)
        # Different operation → different key.
        conn.create_customer(name="Bob")
        key3 = http.calls[-1]["kw"].get("headers", {})["Idempotency-Key"]
        self.assertNotEqual(key1, key3)

    def test_explicit_key_overrides(self) -> None:
        conn, http = self._stripe()
        http.route("POST", "/v1/customers",
                   FakeResponse(200, {"id": "cus_9"}))
        conn._api("POST", "/v1/customers", params={"name": "Z"},
                  idempotency_key="my-key-1")
        headers = http.calls[-1]["kw"].get("headers", {})
        self.assertEqual(headers["Idempotency-Key"], "my-key-1")

    def test_no_key_on_get(self) -> None:
        conn, http = self._stripe()
        http.route("GET", "/v1/balance", FakeResponse(200, {}))
        conn.get_balance()
        headers = http.calls[-1]["kw"].get("headers", {})
        self.assertNotIn("Idempotency-Key", headers)

    def test_idempotency_key_for_stable(self) -> None:
        k1 = StripeConnector.idempotency_key_for("order-1", "cus_2")
        k2 = StripeConnector.idempotency_key_for("order-1", "cus_2")
        k3 = StripeConnector.idempotency_key_for("order-2", "cus_2")
        self.assertEqual(k1, k2)
        self.assertNotEqual(k1, k3)
        self.assertTrue(k1.startswith("nm-"))

    def test_webhook_verify_and_parse(self) -> None:
        conn, _ = self._stripe()
        body = b'{"id":"evt_1","type":"payment_intent.succeeded",' \
               b'"created":1700000000,"data":{"object":{"id":"pi_1"}}}'
        ts = f"{int(time.time())}"
        mac = hmac.new(b"whsec", f"{ts}.".encode() + body,
                       hashlib.sha256).hexdigest()
        conn.verify_webhook_signature(body, f"t={ts},v1={mac}", "whsec")
        ev = conn.parse_webhook_event(body)
        self.assertEqual(ev.id, "evt_1")
        self.assertEqual(ev.type, "payment_intent.succeeded")
        self.assertEqual(ev.data["id"], "pi_1")
        with self.assertRaises(ConnectorValidationError):
            conn.verify_webhook_signature(body, f"t={ts},v1={'0' * 64}",
                                          "whsec")

    def test_capabilities_flags(self) -> None:
        conn, _ = self._stripe()
        caps = conn.capabilities()
        self.assertTrue(caps["webhooks"]["inbound"])
        self.assertIn("Idempotency-Key", caps["idempotency"])


# ── checkpoints history ───────────────────────────────────────────

class TestCheckpointHistory(unittest.TestCase):
    def test_history(self) -> None:
        store = CheckpointStore(Database(":memory:"))
        cp = store.create("github", CheckpointKind.MANUAL_STEP, "T",
                          "do it", ttl_seconds=3600)
        self.assertEqual(len(store.list_pending()), 1)
        self.assertEqual(store.history(), [])
        store.resolve(cp.id, note="done")
        self.assertEqual(store.list_pending(), [])
        hist = store.history()
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist[0].id, cp.id)
        self.assertEqual(hist[0].result_note, "done")
        self.assertEqual(len(store.history("github")), 1)
        self.assertEqual(store.history("stripe"), [])


# ── duffel format_order ───────────────────────────────────────────

class TestDuffelFormatOrder(unittest.TestCase):
    def test_format_order(self) -> None:
        from nomorals.connectors.duffel import DuffelConnector, Order
        conn = DuffelConnector.__new__(DuffelConnector)
        order = Order(id="ord_1", booking_reference="ABC123",
                      total_amount="250.00", total_currency="USD",
                      origin="LOS", destination="JFK",
                      created_at="2026-01-01",
                      raw={"passengers": [
                          {"given_name": "Ada", "family_name": "Lovelace"}]})
        out = DuffelConnector.format_order(conn, order)
        self.assertIn("ABC123", out)
        self.assertIn("LOS → JFK", out)
        self.assertIn("Ada Lovelace", out)
        self.assertIn("USD 250.00", out)


if __name__ == "__main__":
    unittest.main()

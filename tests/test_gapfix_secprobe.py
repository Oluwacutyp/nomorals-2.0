"""Secprobe wire-up: the owner's 14 security files + the attacker glue.

Every tool's happy path + refusals, all offline:

* a local HTTP fixture server on 127.0.0.1 — the real HttpClient talks
  to it, so the probes run end to end without any live host;
* monkeypatched leak adapters / dossier sources / ban sessions for the
  parts that have no local shape;
* a fake browser driver and a fake signer for the two L4 seams
  (capture / browser signing), which refuse honestly without one.

The registry is built with a throwaway home (SimpleNamespace(home=tmp))
so case dirs, ledgers and dossiers land in the test tmp dir, never in
the repo. The tool count pin is 774 (749 + 25 secprobe tools). 775 after this lane (B.19) added scene_spawn. 776 after this lane (B.20) added cloudcam. 777 after this lane (B.22) added show.
"""
from __future__ import annotations

import base64
import hashlib
import hmac as hmac_mod
import json
import os
import re
import shutil
import tempfile
import threading
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from nomorals.core.http import HttpClient as _RealHttpClient
from nomorals.security import bans as bans_mod
from nomorals.security import case as case_mod
from nomorals.security import cors_csrf as cors_mod
from nomorals.security import dossier as dossier_mod
from nomorals.security import graphql as graphql_mod
from nomorals.security import inject as inject_mod
from nomorals.security import leaks as leaks_mod
from nomorals.security import params as params_mod
from nomorals.security import race as race_mod
from nomorals.security import replay as replay_mod
from nomorals.security import report_amp as report_amp_mod
from nomorals.security.diff import compare, fingerprint, login_delta
from nomorals.security.scope import Governor, Scope
from nomorals.tools.attacker import run_attack
from nomorals.tools.registry import ToolRegistry


# ── fixture server ────────────────────────────────────────────────────
class _FixtureHandler(BaseHTTPRequestHandler):
    """One scoped target for every probe: search, vuln, graphql, race,
    cors, login, replay, diag. Shared state lives on the class."""

    counter = 0
    diag_seen: set[str] = set()
    last_bodies: list[dict] = []
    seen_passwords: set[str] = set()
    lock = threading.Lock()

    def log_message(self, *args):  # noqa: D102 - silence the server
        pass

    def _body_bytes(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def _send(self, status: int, body: bytes, ctype: str = "text/plain",
              extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload).encode(),
                   ctype="application/json")

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler name
        parts = urllib.parse.urlsplit(self.path)
        qs = urllib.parse.parse_qs(parts.query)
        if parts.path == "/search":
            body = "results: none\n"
            if "q" in qs:
                body += "echo: " + qs["q"][0] + "\n"
            if "debug" in qs:
                body += "debug on\n"
            self._send(200, body.encode())
            return
        if parts.path == "/vuln":
            x = qs.get("x", ["1"])[0]
            if "<vns8471>" in x:
                self._send(200, ("ok " + x).encode())
            elif "{{7*7}}" in x:
                self._send(200, b"result: 49")
            elif "../" in x or "/etc/passwd" in x:
                self._send(200, b"root:x:0:0:root:/root:/bin/bash")
            elif x.startswith(("https://", "//")):
                self._send(302, b"", extra={"Location": x})
            elif x == "1 AND 1=2":
                # Long + different: the length axis (64/20%) fires alongside
                # the digest axis, so compare() calls it interesting — the
                # boolean pair only lands when the false branch departs.
                self._send(200, b"different: no rows at all in this table "
                                b"for that query, please check the value "
                                b"and try again")
            elif "'" in x or '"' in x or "\\" in x:
                self._send(200, ("SQL syntax error near '" + x + "'").encode())
            else:
                self._send(200, b"ok")
            return
        if parts.path == "/cors":
            origin = self.headers.get("Origin", "")
            self._send(200, b"ok", extra={
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Credentials": "true"})
            return
        self._send(404, b"not found")

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler name
        parts = urllib.parse.urlsplit(self.path)
        raw = self._body_bytes()
        if parts.path == "/graphql":
            try:
                query = json.loads(raw or b"{}").get("query", "")
            except ValueError:
                query = ""
            if "__schema" in query:
                self._json(200, {"data": {"__schema": {
                    "types": [
                        {"name": "Query", "fields": [
                            {"name": "user"}, {"name": "password"}]},
                        {"name": "User", "fields": [
                            {"name": "email"}, {"name": "passwordHash"}]}],
                    "mutationType": {"name": "Mutation"}}}})
            else:
                aliases = re.findall(r"\ba(\d+):", query)
                self._json(200, {"data": {f"a{i}": True for i in aliases}})
            return
        if parts.path == "/race":
            self._send(200, b"accepted")
            return
        if parts.path == "/login":
            form = urllib.parse.parse_qs(raw.decode(errors="replace"))
            user = (form.get("username") or [""])[0]
            pw = (form.get("password") or [""])[0]
            self.seen_passwords.add(pw)
            if user == "alice" and pw == "s3cret":
                self._send(302, b"", extra={"Location": "/dashboard"})
            else:
                self._send(200, b"Invalid credentials, try again.")
            return
        if parts.path in ("/replay", "/replay-fixed", "/blocked", "/diag"):
            try:
                posted = json.loads(raw or b"{}")
            except ValueError:
                posted = {}
            with self.lock:
                self.last_bodies.append(posted)
                if parts.path == "/replay":
                    # NOTE: counter is a class attribute — `self.counter += 1`
                    # would shadow it on the per-request handler instance
                    # and every request would read rep-1.
                    type(self).counter += 1
                    self._json(200, {"status": "ok",
                                     "id": f"rep-{type(self).counter}"})
                elif parts.path == "/replay-fixed":
                    self._json(200, {"status": "ok", "id": "fixed-1"})
                elif parts.path == "/blocked":
                    self._send(403, b"forbidden")
                else:  # /diag — a nonce counts once, then it is a dupe
                    nonce = str(posted.get("nonce", ""))
                    if nonce in self.diag_seen:
                        self._json(200, {"status": "dupe", "id": nonce})
                    else:
                        self.diag_seen.add(nonce)
                        self._json(200, {"status": "ok", "id": nonce})
            return
        self._send(404, b"not found")


_SERVER: ThreadingHTTPServer | None = None
_PORT = 0


class _LoopbackClient(_RealHttpClient):
    """HttpClient that may talk to the loopback fixture (tests only).

    The production guard (allow_private_ips=False) stays the default
    everywhere else; the probe modules construct HttpClient internally,
    so the fixture tests point their module-level name at this subclass.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("allow_private_ips", True)
        super().__init__(*args, **kwargs)


_PATCHED_HTTP: list[tuple] = []


def setUpModule() -> None:
    global _SERVER, _PORT
    _SERVER = ThreadingHTTPServer(("127.0.0.1", 0), _FixtureHandler)
    _PORT = _SERVER.server_address[1]
    threading.Thread(target=_SERVER.serve_forever, daemon=True).start()
    for mod in (params_mod, inject_mod, graphql_mod, race_mod, cors_mod,
                replay_mod):
        _PATCHED_HTTP.append((mod, mod.HttpClient))
        mod.HttpClient = _LoopbackClient


def tearDownModule() -> None:
    for mod, orig in _PATCHED_HTTP:
        mod.HttpClient = orig
    _PATCHED_HTTP.clear()
    if _SERVER is not None:
        _SERVER.shutdown()
        _SERVER.server_close()


def _url(path: str) -> str:
    return f"http://127.0.0.1:{_PORT}{path}"


# ── registry + case helpers ───────────────────────────────────────────
class _SecBase(unittest.TestCase):
    """One registry + one throwaway home + one cased scope per class."""

    home: str = ""
    registry: ToolRegistry

    @classmethod
    def setUpClass(cls) -> None:
        cls.home = tempfile.mkdtemp(prefix="secprobe-")
        cls.registry = ToolRegistry(SimpleNamespace(home=cls.home))
        cls.registry.register_builtins()
        cls._make_case("shop")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.home, ignore_errors=True)

    @classmethod
    def _make_case(cls, name: str) -> Path:
        made = case_mod.init_case(cls.home, name)
        scope = Path(made["scope"])
        scope.write_text("include: 127.0.0.1\n", encoding="utf-8")
        return Path(made["dir"])

    def _call(self, tool: str, **kw):
        outcome = self.registry.call(tool, **kw)
        if not getattr(outcome, "ok", False):
            raise AssertionError(
                f"{tool} failed: {getattr(outcome, 'error', outcome)}")
        return outcome.value

    def _raw(self, tool: str, **kw):
        return self.registry.call(tool, **kw)

    @staticmethod
    def _err(outcome) -> str:
        return str(getattr(outcome.error, "message", outcome.error))


# ── scope ─────────────────────────────────────────────────────────────
class ScopeTests(_SecBase):
    def test_load_and_check(self):
        loaded = self._call("scope_load", case="shop")
        self.assertIn("127.0.0.1", loaded["include"])
        self.assertFalse(loaded["allow_all"])
        self.assertTrue(self._call(
            "scope_check", url=_url("/login"), case="shop")["in_scope"])
        self.assertFalse(self._call(
            "scope_check", url="https://example.com/login",
            case="shop")["in_scope"])

    def test_empty_include_refuses(self):
        case_dir = self._make_case("empty")
        (case_dir / "scope.txt").write_text("include:\n", encoding="utf-8")
        out = self._raw("scope_load", case="empty")
        self.assertFalse(out.ok)
        self.assertIn("lists nothing to include", self._err(out))

    def test_no_scope_file_refuses(self):
        # No case, no default scope file under this home: default-deny.
        out = self._raw("scope_load")
        self.assertFalse(out.ok)
        self.assertIn("scope", self._err(out).lower())


# ── params ────────────────────────────────────────────────────────────
class ParamTests(_SecBase):
    def test_mine_reads_seed_json(self):
        case_dir = Path(self.home) / "data" / "evidence" / "shop"
        (case_dir / "seed.json").write_text(json.dumps(
            {"url": [{"value": _url("/search?q=test&page=2")}]}),
            encoding="utf-8")
        found = self._call("param_mine", case="shop")
        self.assertEqual(found["hosts"], 1)
        self.assertIn("127.0.0.1", found["candidates"])
        # mined names come back through the tool only as counts; check
        # the library directly for the names themselves.
        from nomorals.security.params import mine
        mined = mine(case_dir)
        self.assertIn("q", mined["127.0.0.1"])
        self.assertIn("page", mined["127.0.0.1"])
        self.assertIn("search", mined["127.0.0.1"])

    def test_confirm_keeps_only_reacting_names(self):
        out = self._call("param_confirm", url=_url("/search"),
                         names="q,zzz", case="shop")
        confirmed = [p["name"] for p in out["confirmed"]]
        self.assertIn("q", confirmed)
        self.assertNotIn("zzz", confirmed)
        self.assertTrue(out["confirmed"][0]["reflected"])

    def test_confirm_out_of_scope_refuses(self):
        out = self._raw("param_confirm", url="https://example.com/search",
                        names="q", case="shop")
        self.assertFalse(out.ok)
        self.assertIn("scope", self._err(out).lower())

    def test_confirm_without_scope_refuses(self):
        out = self._raw("param_confirm", url=_url("/search"), names="q")
        self.assertFalse(out.ok)


# ── inject ────────────────────────────────────────────────────────────
class InjectTests(_SecBase):
    def test_probe_finds_each_class(self):
        out = self._call("inject_probe", url=_url("/vuln?x=1"), param="x",
                         case="shop")
        by_class = {f["class"]: f for f in out["findings"]}
        for cls in ("sqli", "sqli_boolean", "xss", "ssti", "lfi",
                    "redirect"):
            self.assertIn(cls, by_class, f"missing {cls}: {out}")
            self.assertEqual(by_class[cls]["verdict"], "confirmed")
        self.assertIn("SQL syntax", by_class["sqli"]["evidence"])
        self.assertEqual(by_class["ssti"]["evidence"],
                         "arithmetic evaluated")
        self.assertIn("root:", by_class["lfi"]["evidence"])

    def test_probe_clean_param_finds_nothing(self):
        # zzz is never echoed and never changes the page: every payload
        # reads exactly like the baseline.
        out = self._call("inject_probe", url=_url("/search?zzz=1"),
                         param="zzz", case="shop")
        self.assertEqual(out["findings"], [])

    def test_probe_out_of_scope_refuses(self):
        out = self._raw("inject_probe", url="https://example.com/vuln?x=1",
                        param="x", case="shop")
        self.assertFalse(out.ok)


# ── jwt ───────────────────────────────────────────────────────────────
def _jwt_token(secret: str, alg: str = "HS256") -> str:
    header = base64.urlsafe_b64encode(
        json.dumps({"alg": alg, "typ": "JWT"}).encode()).decode().rstrip("=")
    payload = base64.urlsafe_b64encode(
        json.dumps({"sub": "alice"}).encode()).decode().rstrip("=")
    sig = hmac_mod.new(secret.encode(), f"{header}.{payload}".encode(),
                       hashlib.sha256).digest()
    return (f"{header}.{payload}."
            f"{base64.urlsafe_b64encode(sig).decode().rstrip('=')}")


class JwtTests(_SecBase):
    def test_bad_token_refuses(self):
        out = self._raw("jwt_attack", token="notajwt")
        self.assertFalse(out.ok)
        self.assertIn("not a JWT", self._err(out))

    def test_weak_hmac_confirmed(self):
        out = self._call("jwt_attack", token=_jwt_token("secret"))
        confirmed = [r for r in out["confirmed"]
                     if r["test"] == "weak_hmac"]
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0]["secret"], "secret")

    def test_strong_secret_not_confirmed(self):
        out = self._call("jwt_attack",
                         token=_jwt_token("n0t-in-the-dictionary-xyz"))
        self.assertEqual(
            [r for r in out["confirmed"] if r["test"] == "weak_hmac"], [])

    def test_alg_none_is_a_candidate(self):
        header = base64.urlsafe_b64encode(
            json.dumps({"alg": "none", "typ": "JWT"}).encode()
        ).decode().rstrip("=")
        payload = base64.urlsafe_b64encode(
            json.dumps({"sub": "alice"}).encode()).decode().rstrip("=")
        out = self._call("jwt_attack", token=f"{header}.{payload}.")
        self.assertTrue(any(r["test"] == "alg_none"
                            for r in out["results"]))


# ── graphql ───────────────────────────────────────────────────────────
class GraphqlTests(_SecBase):
    def test_probe_reads_schema(self):
        out = self._call("graphql_probe", url=_url("/graphql"), case="shop")
        self.assertTrue(out["introspection"])
        self.assertEqual(out["types"], 2)
        self.assertIn("Query.password", out["sensitive"])
        self.assertIn("User.passwordHash", out["sensitive"])
        self.assertIn("mutations exposed", out["notes"])

    def test_amplify_counts_answers(self):
        out = self._call("graphql_amplify", url=_url("/graphql"),
                         count="10", case="shop")
        self.assertTrue(out["ok"])
        self.assertEqual(out["answered"], 10)
        self.assertEqual(out["requested"], 10)

    def test_probe_out_of_scope_refuses(self):
        out = self._raw("graphql_probe", url="https://example.com/graphql",
                        case="shop")
        self.assertFalse(out.ok)


# ── race ──────────────────────────────────────────────────────────────
class RaceTests(_SecBase):
    def test_wave_all_accepted(self):
        out = self._call("race_wave", url=_url("/race"), count="5",
                         case="shop")
        self.assertEqual(out["sent"], 5)
        self.assertEqual(out["answered"], 5)
        self.assertEqual(out["verdict"], "all_accepted")

    def test_wave_out_of_scope_refuses(self):
        out = self._raw("race_wave", url="https://example.com/race",
                        count="5", case="shop")
        self.assertFalse(out.ok)


# ── cors + csrf ───────────────────────────────────────────────────────
class CorsTests(_SecBase):
    def test_probe_exploitable_on_echo(self):
        out = self._call("cors_probe", url=_url("/cors"), case="shop")
        self.assertEqual(out["verdict"], "exploitable")
        self.assertTrue(out["findings"])
        self.assertTrue(all(f["severity"] == "critical"
                            for f in out["findings"]))

    def test_probe_out_of_scope_refuses(self):
        out = self._raw("cors_probe", url="https://example.com/",
                        case="shop")
        self.assertFalse(out.ok)

    def test_csrf_poc_renders_fields(self):
        out = self._call("csrf_poc", action="https://example.com/transfer",
                         fields={"to": "attacker", "amount": "100"})
        self.assertGreater(out["bytes"], 0)
        self.assertIsNone(out["file"])
        from nomorals.security.cors_csrf import csrf_poc
        page = csrf_poc("https://example.com/transfer",
                        {"to": "attacker", "amount": "100"})
        self.assertIn("https://example.com/transfer", page)
        self.assertIn('name="to"', page)
        self.assertIn("attacker", page)


# ── planner ───────────────────────────────────────────────────────────
class PlannerTests(_SecBase):
    def test_add_next_resolve(self):
        added = self._call("plan_add", case="shop", cls="login_form",
                           target=_url("/login"),
                           param="user=username pass=password path=/login "
                                 "method=POST")
        self.assertEqual(added["class"], "login_form")
        nxt = self._call("plan_next", case="shop")
        self.assertEqual(len(nxt["next"]), 1)
        self.assertEqual(nxt["next"][0]["id"], added["id"])
        resolved = self._call("plan_resolve", case="shop", id=added["id"],
                              confirmed="true")
        self.assertEqual(resolved["state"], "confirmed")
        self.assertEqual(self._call("plan_next", case="shop")["next"], [])

    def test_empty_case_refuses(self):
        out = self._raw("plan_next", case="")
        self.assertFalse(out.ok)

    def test_resolve_unknown_id(self):
        out = self._call("plan_resolve", case="shop", id="nope",
                         confirmed="true")
        self.assertEqual(out.get("error"), "unknown id")


# ── leaks ─────────────────────────────────────────────────────────────
class LeaksTests(_SecBase):
    def setUp(self):
        for key in ("NM_HIBP_KEY", "NM_DEHASHED_KEY", "NM_LEAKCHECK_KEY",
                    "NM_INTELX_KEY", "NM_SNUSBASE_KEY"):
            os.environ.pop(key, None)

    def test_keyless_sources_merge(self):
        from nomorals.security.leaks import Exposure
        orig_cav, orig_prox = leaks_mod._cavalier, leaks_mod._proxynova
        leaks_mod._cavalier = lambda email="", username="", domain="", \
            timeout=25.0: Exposure(
                subject=email or username or domain,
                identities={email or username or domain},
                passwords={"hunter2"})
        leaks_mod._proxynova = lambda email, timeout=25.0: Exposure(
            subject=email, sessions=[{"service": "example", "id": "sess-1"}])
        try:
            out = self._call("leaks_harvest",
                             email="victim@example.invalid")
        finally:
            leaks_mod._cavalier, leaks_mod._proxynova = orig_cav, orig_prox
        self.assertTrue(out["exposed"])
        self.assertIn("hunter2", out["passwords"])
        self.assertEqual(out["sessions"][0]["id"], "sess-1")
        self.assertEqual(out["ran"], ["cavalier", "proxynova"])
        self.assertEqual(out["sources_skipped_no_key"],
                         ["hibp", "dehashed", "leakcheck", "intelx",
                          "snusbase"])

    def test_keyed_source_named_when_key_present(self):
        from nomorals.security.leaks import Exposure
        os.environ["NM_HIBP_KEY"] = "test-key"
        orig = (leaks_mod._cavalier, leaks_mod._proxynova, leaks_mod._hibp)
        leaks_mod._cavalier = lambda email="", username="", domain="", \
            timeout=25.0: Exposure(subject=email)
        leaks_mod._proxynova = lambda email, timeout=25.0: Exposure(
            subject=email)
        leaks_mod._hibp = lambda email, key, timeout=25.0: Exposure(
            subject=email, identities={email})
        try:
            out = self._call("leaks_harvest",
                             email="victim@example.invalid")
        finally:
            (leaks_mod._cavalier, leaks_mod._proxynova,
             leaks_mod._hibp) = orig
            os.environ.pop("NM_HIBP_KEY", None)
        self.assertIn("hibp", out["ran"])
        self.assertNotIn("hibp", out["sources_skipped_no_key"])
        self.assertIn("dehashed", out["sources_skipped_no_key"])

    def test_no_subject_refuses(self):
        out = self._call("leaks_harvest")
        self.assertIn("email", out["error"])


# ── dossier ───────────────────────────────────────────────────────────
class DossierTests(_SecBase):
    def test_build_offline_with_stubbed_sources(self):
        orig_enum = dossier_mod.social_enum.enum_username
        orig_harvest = dossier_mod.leaks.harvest_exposure
        dossier_mod.social_enum.enum_username = lambda *a, **k: []
        dossier_mod.leaks.harvest_exposure = lambda **k: {
            "exposed": False, "passwords": [], "sessions": [],
            "ran": ["cavalier"], "sources_skipped_no_key": ["hibp"],
            "identities": []}
        try:
            out = self._call("dossier_build", subject="someone")
        finally:
            dossier_mod.social_enum.enum_username = orig_enum
            dossier_mod.leaks.harvest_exposure = orig_harvest
        self.assertEqual(out["subject"], "someone")
        self.assertEqual(out["score"], 0)
        self.assertEqual(out["tier"], "low")
        self.assertTrue(any("hibp" in n for n in out["notes"]))

    def test_record_credentials_scores_and_hides_passwords(self):
        from nomorals.security.dossier import Dossier, save_dossier
        d = Dossier(subject="alice", kind="username")
        d.record_credentials("127.0.0.1",
                             [{"login": "alice", "password": "s3cret"}])
        self.assertEqual(d.score, 5)
        found = d.exposure["found_credentials"]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["username"], "alice")
        self.assertNotIn("password", found[0])
        self.assertNotIn("s3cret", json.dumps(d.as_dict()))
        root = Path(self.home) / "data" / "evidence" / "shop"
        saved = save_dossier(d, root)
        self.assertTrue(Path(saved["json"]).is_file())
        stored = json.loads(Path(saved["json"]).read_text())
        self.assertEqual(stored["subject"], "alice")
        self.assertEqual(stored["exposure"]["found_credentials"][0]
                         ["username"], "alice")


# ── bans ──────────────────────────────────────────────────────────────
class BansTests(_SecBase):
    def test_state_round_trip(self):
        self.assertEqual(self._call("ban_state", subject="nobody")["states"],
                         {})
        self._call("ban_set_state", subject="spammer", platform="tiktok",
                   state="reported")
        states = self._call("ban_state", subject="spammer")["states"]
        self.assertEqual(states.get("tiktok"), "reported")

    def test_appeal_writes_drafts(self):
        self._call("ban_set_state", subject="spammer", platform="tiktok",
                   state="banned")
        out = self._call("ban_appeal", handle="spammer")
        self.assertTrue(out["files"])
        for path in out["files"]:
            self.assertTrue(Path(path).is_file())
            self.assertIn("spammer", Path(path).read_text())

    def test_report_unknown_platform_refuses(self):
        out = self._raw("ban_report", platform="nope", target="x",
                        session_account="me")
        self.assertFalse(out.ok)
        self.assertIn("unknown platform", self._err(out))

    def test_report_without_session_fails_honestly(self):
        out = self._call("ban_report", platform="tiktok", target="spammer",
                         session_account="nobody")
        self.assertFalse(out["ok"])
        self.assertIn("no stored session", out["reason"])

    def test_report_with_stubbed_session(self):
        class _Resp:
            ok = True
            status = 200
            text = ""

        class _Client:
            def post(self, url, data=None, **kw):
                return _Resp()

            def get(self, url, **kw):
                return _Resp()

        orig_load, orig_client = (bans_mod.load_session,
                                  bans_mod._client_from)
        bans_mod.load_session = lambda *a, **k: {"session": {"x": "1"}}
        bans_mod._client_from = lambda *a, **k: _Client()
        try:
            out = self._call("ban_report", platform="tiktok",
                             target="spammer", session_account="me")
        finally:
            bans_mod.load_session, bans_mod._client_from = \
                orig_load, orig_client
        self.assertTrue(out["ok"])
        self.assertEqual(out["status"], 200)


# ── capture ───────────────────────────────────────────────────────────
class CaptureTests(_SecBase):
    def test_capture_without_driver_refuses(self):
        report_amp_mod.set_capture_driver(None)
        out = self._raw("report_capture", platform="telegram")
        self.assertFalse(out.ok)
        self.assertIn("Playwright", self._err(out))

    def test_capture_with_fake_driver_saves_template(self):
        def _driver(platform, url_hint, timeout):
            return {"url": "https://telegram.org/support",
                    "method": "POST",
                    "body": json.dumps({"message": "spam",
                                        "email": "a@b.c"}),
                    "headers": {"Content-Type":
                                "application/x-www-form-urlencoded"}}
        report_amp_mod.set_capture_driver(_driver)
        try:
            out = self._call("report_capture", platform="telegram")
        finally:
            report_amp_mod.set_capture_driver(None)
        path = Path(out["saved"])
        self.assertTrue(path.is_file())
        self.assertEqual(out["template"]["platform"], "telegram")
        self.assertEqual(out["template"]["url"],
                         "https://telegram.org/support")
        stored = json.loads(path.read_text())
        self.assertEqual(stored["body"]["message"], "spam")

    def test_capture_unknown_platform_refuses(self):
        out = self._raw("report_capture", platform="nope")
        self.assertFalse(out.ok)
        self.assertIn("unknown platform", self._err(out))


# ── amplify ───────────────────────────────────────────────────────────
class AmplifyTests(_SecBase):
    def _template(self) -> str:
        path = Path(self.home) / "template.json"
        path.write_text(json.dumps({
            "platform": "telegram", "url": _url("/replay"),
            "method": "POST",
            "body": {"device_id": "d1", "ts": "1", "message": "spam"},
            "headers": {}, "volatile": ["device_id", "ts"],
            "signer": "none", "success_marker": "", "dupe_marker": "",
            "report_id_field": "id"}), encoding="utf-8")
        return str(path)

    def test_amplify_counts_rows_with_device(self):
        out = self._call("report_amplify", template=self._template(),
                         count="3", case="shop")
        self.assertEqual(out["attempted"], 3)
        self.assertEqual(out["counted"], 3)
        self.assertEqual(out["platform"], "telegram")
        row = out["results"][0]
        self.assertIn("device", row)
        self.assertIn("proxy", row)

    def test_amplify_corrupt_template_refuses(self):
        bad = Path(self.home) / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        out = self._raw("report_amplify", template=str(bad), case="shop")
        self.assertFalse(out.ok)
        missing = Path(self.home) / "missing-keys.json"
        missing.write_text(json.dumps({"platform": "telegram"}),
                           encoding="utf-8")
        out = self._raw("report_amplify", template=str(missing),
                        case="shop")
        self.assertFalse(out.ok)

    def test_amplify_out_of_scope_refuses(self):
        path = Path(self.home) / "remote.json"
        path.write_text(json.dumps({
            "platform": "telegram", "url": "https://example.com/report",
            "method": "POST", "body": {}, "headers": {}, "volatile": [],
            "signer": "none"}), encoding="utf-8")
        out = self._raw("report_amplify", template=str(path), case="shop")
        self.assertFalse(out.ok)


# ── replay ────────────────────────────────────────────────────────────
class _FakeSigner:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def sign(self, field_name, body):
        return "fakesig"


class ReplayTests(_SecBase):
    def test_run_counts_unique_ids(self):
        out = self._call("replay_run", url=_url("/replay"),
                         body={"nonce": "n0", "message": "hi"},
                         count="3", kinds={"nonce": "nonce"},
                         id_field="id", case="shop")
        self.assertEqual(out["attempted"], 3)
        self.assertEqual(out["counted"], 3)
        self.assertEqual(out["dupes"], 0)

    def test_run_fixed_id_dupes(self):
        out = self._call("replay_run", url=_url("/replay-fixed"),
                         body={"nonce": "n0", "message": "hi"},
                         count="3", kinds={"nonce": "nonce"},
                         id_field="id", case="shop")
        self.assertEqual(out["counted"], 1)
        self.assertEqual(out["dupes"], 2)

    def test_run_blocked_status(self):
        out = self._call("replay_run", url=_url("/blocked"),
                         body={"nonce": "n0"}, count="3",
                         kinds={"nonce": "nonce"}, case="shop")
        self.assertEqual(out["blocked"], 3)
        self.assertEqual(out["counted"], 0)

    def test_browser_signer_without_factory_refuses(self):
        replay_mod.set_signer_factory(None)
        out = self._raw("replay_run", url=_url("/replay"),
                        body={"sig": "", "message": "hi"},
                        kinds={"sig": "signature"}, signer="browser",
                        signer_url=_url("/sign"), case="shop")
        self.assertFalse(out.ok)
        self.assertIn("Playwright", self._err(out))

    def test_browser_signer_with_fake_factory_signs(self):
        replay_mod.set_signer_factory(lambda url: _FakeSigner())
        try:
            _FixtureHandler.last_bodies.clear()
            out = self._call("replay_run", url=_url("/replay"),
                             body={"sig": "", "message": "hi"},
                             count="3", kinds={"sig": "signature"},
                             signer="browser", signer_url=_url("/sign"),
                             id_field="id", case="shop")
        finally:
            replay_mod.set_signer_factory(None)
        self.assertEqual(out["counted"], 3)
        posted = [b for b in _FixtureHandler.last_bodies
                  if "sig" in b]
        self.assertTrue(posted)
        self.assertTrue(all(b["sig"] == "fakesig" for b in posted))

    def test_diagnose_names_the_held_field(self):
        # The captured nonce is already spent on the fixture: holding it
        # static never counts, holding the note static (minting a fresh
        # nonce) counts every probe — so the note is the suspect by the
        # tool's own rule (a held field that restores acceptance).
        _FixtureHandler.diag_seen.add("seeded-old-nonce-001")
        out = self._call("replay_diagnose", url=_url("/diag"),
                         body={"nonce": "seeded-old-nonce-001",
                               "note": "hello"},
                         kinds={"nonce": "nonce", "note": "identity"},
                         dupe="dupe", case="shop")
        self.assertEqual(out["suspects"], ["note"])


# ── attacker (scope, budget, one fingerprint) ─────────────────────────
class AttackerTests(_SecBase):
    def test_run_without_governor_refuses(self):
        with self.assertRaises(Exception) as ctx:
            run_attack("127.0.0.1", port=_PORT, protocol="http_form",
                       usernames="alice", passwords="wrong1",
                       fail_string="Invalid credentials", path="/login",
                       quiet=True)
        self.assertIn("scope", str(ctx.exception).lower())

    def test_run_out_of_scope_refuses(self):
        gov = Governor(Scope(include=["example.com"]), max_total=10,
                       max_per_host=1000.0)
        with self.assertRaises(Exception) as ctx:
            run_attack("127.0.0.1", port=_PORT, protocol="http_form",
                       usernames="alice", passwords="wrong1",
                       fail_string="Invalid credentials", path="/login",
                       quiet=True, governor=gov)
        self.assertIn("scope", str(ctx.exception).lower())

    def test_budget_exhausted_note(self):
        # One request of budget, two passwords: the second password's
        # spend raises and the run stops with the budget named. Note
        # `attempts` counts dequeued items (the second was dequeued
        # before its spend raised), so the server-side proof is that
        # wrong2 never arrived — the baseline dummy and wrong1 did.
        _FixtureHandler.seen_passwords.clear()
        gov = Governor(Scope(include=["127.0.0.1"]), max_total=1,
                       max_per_host=1000.0)
        res = run_attack("127.0.0.1", port=_PORT, protocol="http_form",
                         usernames="alice", passwords="wrong1,wrong2",
                         fail_string="Invalid credentials", path="/login",
                         workers=1, quiet=True, governor=gov)
        self.assertIn("budget exhausted", res["note"])
        self.assertIn("wrong1", _FixtureHandler.seen_passwords)
        self.assertNotIn("wrong2", _FixtureHandler.seen_passwords)

    def test_login_delta_unified_verdict(self):
        base = fingerprint(200, "Invalid credentials, try again.", {},
                           length=32, markers=["Invalid credentials"])
        good = fingerprint(302, "welcome to your dashboard", {},
                           length=25, location="/dashboard",
                           markers=["Invalid credentials"])
        bad = fingerprint(200, "Invalid credentials, try again.", {},
                          length=32, markers=["Invalid credentials"])
        self.assertGreaterEqual(login_delta(base, good), 2)
        self.assertEqual(login_delta(base, bad), 0)
        # compare() agrees the success probe is interesting.
        self.assertTrue(compare(base, good).interesting)


# ── chain ─────────────────────────────────────────────────────────────
class ChainTests(_SecBase):
    def test_chain_end_to_end(self):
        case_dir = self._make_case("chain1")
        self._call("plan_add", case="chain1", cls="login_form",
                   target=_url("/login"),
                   param="user=username pass=password path=/login "
                         "method=POST")
        out = self._call("sec_chain", case="chain1", usernames="alice",
                         passwords="wrong1,s3cret",
                         fail_string="Invalid credentials")
        self.assertEqual(len(out["runs"]), 1)
        run = out["runs"][0]
        self.assertEqual(len(run["found"]), 1)
        self.assertEqual(run["found"][0]["login"], "alice")
        self.assertEqual(run["attempts"], 2)
        # the hypothesis is resolved confirmed in the ledger …
        nxt = self._call("plan_next", case="chain1")
        self.assertEqual(nxt["next"], [])
        # … and the credential lands in the case dossier, password-free.
        # save_dossier writes <slug>.json (the subject slugged), not
        # dossier-<slug>.json; the case dir holds no other .json.
        dossiers = list(case_dir.glob("*.json"))
        self.assertEqual(len(dossiers), 1)
        stored = json.loads(dossiers[0].read_text())
        found = stored["exposure"]["found_credentials"]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["username"], "alice")
        self.assertNotIn("s3cret", json.dumps(stored))

    def test_chain_skips_non_login_hypothesis(self):
        self._make_case("chain2")
        self._call("plan_add", case="chain2", cls="sqli",
                   target=_url("/vuln?x=1"), param="x")
        out = self._call("sec_chain", case="chain2", usernames="alice",
                         passwords="wrong1")
        self.assertTrue(out["runs"][0]["skipped"])
        self.assertIn("not a login form", out["runs"][0]["note"])

    def test_chain_empty_ledger(self):
        self._make_case("chain3")
        out = self._call("sec_chain", case="chain3", usernames="alice")
        self.assertEqual(out["runs"], [])


# ── tool count ────────────────────────────────────────────────────────
class ToolCountTests(_SecBase):
    def test_tool_count_is_786_after_secprobe(self):
        # 749 after the mocap port (lane 125) + 25 secprobe tools. 775 after B.19 added scene_spawn; 776 after B.20 added cloudcam; 777 after B.22 added show; 779 after this lane (B.23) added facemcap. 780 after this lane (B.25) added facemesh.
        self.assertEqual(len(self.registry._tools), 800)
        for name in ("scope_load", "scope_check", "param_mine",
                     "param_confirm", "inject_probe", "jwt_attack",
                     "graphql_probe", "graphql_amplify", "race_wave",
                     "cors_probe", "csrf_poc", "plan_next", "plan_add",
                     "plan_resolve", "leaks_harvest", "dossier_build",
                     "ban_report", "ban_state", "ban_set_state",
                     "ban_appeal", "report_capture", "report_amplify",
                     "replay_run", "replay_diagnose", "sec_chain"):
            self.assertIn(name, self.registry._tools)


if __name__ == "__main__":
    unittest.main()

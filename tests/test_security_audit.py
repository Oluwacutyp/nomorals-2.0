"""Security audit class (owner, 2026-11-03): exposure scanner,
repo auditor, disclosure desk, wiring.

Pins:
- the web battery proves findings against a LOCAL fixture: an
  open PostgREST-like endpoint (rows with PII-shaped fields +
  the source's own total in Content-Range), an exposed /.env,
  a JS bundle with a planted live key and a service_role JWT;
- evidence packs are complete: schema carries every field,
  the count is the source's own, samples are real rows —
  masked at rest in the journal for every target, unmasked in
  the delivered report only for owner-declared assets;
- controls stay clean: a locked endpoint, a 404-everything
  target, and a catch-all SPA (200 + HTML for every path)
  produce ZERO findings — content shape is the evidence,
  never a status code;
- the repo auditor finds tree secrets with file:line, a key
  that only survives in git history, a seed-feed vulnerable
  dependency, and a string-built SQL call — and the report
  redacts every secret it names;
- the disclosure desk drafts (never sends) using the contact
  the target itself publishes;
- wiring: /audit + /scan <url> are owner-only, DM-only (a
  group gets a pointer, never findings), queued as background
  jobs, and the drive delivers card + report file to the DM.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.security.audit_evidence import mask_value
from nomorals.security.audit_report import (
    AuditBook, default_book, discover_contacts_web,
    draft_disclosure)
from nomorals.security.audit_repo import audit_repo
from nomorals.security.audit_service import AuditService
from nomorals.security.audit_web import ExposureScanner, Fetcher
from nomorals.social.chat.base import ChatKind
from nomorals.social.chat.control import caller_may_run, parse_control

PASSPORT = "A12345678"
ROWS = [
    {"id": i, "full_name": f"Person {i}",
     "email": f"person{i}@example.com",
     "date_of_birth": "1990-04-12",
     "passport_number": PASSPORT,
     "phone": "+2348012345678"}
    for i in range(1, 6)
]
TOTAL = 80000

# Planted fixture secrets, assembled from parts: the runtime values
# are byte-identical to the leaked-key formats the audit scanner
# must detect, but the source text never contains a full-format
# literal — a scanner-shaped string sitting in source is exactly
# what this class exists to keep out of repos (and repository
# secret scanning blocks pushes that contain one).
STRIPE_LIVE_KEY = "sk_" + "live_" + "4eC39HqLyjWDarjtT1zdp7dc"
AWS_ACCESS_KEY_ID = "AKIA" + "IOSFODNN7EXAMPLE"

ENV_BODY = (
    "APP_NAME=fixture\n"
    "DATABASE_URL=postgres://user:pass@db.internal:5432/app\n"
    "APP_SECRET=supersecretvalue123\n"
    "MODE=production\n"
)


def _jwt(role: str) -> str:
    def b64(obj) -> str:
        raw = json.dumps(obj).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return (f"{b64({'alg': 'HS256', 'typ': 'JWT'})}."
            f"{b64({'role': role, 'iss': 'fixture'})}."
            "fakesignature")


APP_JS = (
    f'const STRIPE = "{STRIPE_LIVE_KEY}";\n'
    f'const SERVICE_KEY = "{_jwt("service_role")}";\n'
    'async function load() { return fetch("/rest/v1/profiles"); }\n'
)

INDEX_HTML = (
    "<html><head><title>fixture app</title>"
    '<script src="/app.js"></script></head>'
    "<body>hello</body></html>"
)


class _Handler(BaseHTTPRequestHandler):
    mode = "vuln"

    def log_message(self, *args):  # silence
        pass

    def _send(self, code, body, ctype="text/plain", extra=None):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802 - http.server's name
        path = self.path.split("?")[0]
        if self.mode == "spa":
            return self._send(200, INDEX_HTML, "text/html")
        if self.mode == "locked":
            if path == "/":
                return self._send(200, "<html>plain</html>",
                                  "text/html")
            if path == "/locked":
                return self._send(401, "unauthorized")
            return self._send(404, "<html>not found</html>",
                              "text/html")
        # vuln mode
        if path == "/":
            return self._send(200, INDEX_HTML, "text/html")
        if path == "/app.js":
            return self._send(200, APP_JS, "application/javascript")
        if path == "/.env":
            return self._send(200, ENV_BODY)
        if path in ("/rest/v1/", "/rest/v1"):
            return self._send(200, json.dumps(
                {"paths": {"/profiles": {}}}), "application/json")
        if path == "/rest/v1/profiles":
            return self._send(
                200, json.dumps(ROWS), "application/json",
                {"Content-Range": f"0-4/{TOTAL}"})
        if path == "/locked":
            return self._send(401, "unauthorized")
        if path == "/.well-known/security.txt":
            return self._send(
                200, "Contact: mailto:security@fixture.example\n")
        return self._send(404, "<html>not found</html>", "text/html")


@pytest.fixture()
def fixture_server():
    servers = []

    def make(mode="vuln"):
        handler = type(f"H_{mode}", (_Handler,), {"mode": mode})
        srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=srv.serve_forever,
                         daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}"

    yield make
    for srv in servers:
        srv.shutdown()


def _scanner() -> ExposureScanner:
    return ExposureScanner(Fetcher(allow_private=True,
                                   min_interval_s=0.0))


# ── the web battery ──────────────────────────────────────────

def test_web_scan_finds_the_planted_exposures(fixture_server):
    base = fixture_server("vuln")
    result = _scanner().scan(base)
    kinds = {f["kind"] for f in result["findings"]}
    assert "exposed_file" in kinds            # /.env
    assert "client_master_key" in kinds      # service_role JWT
    assert "client_secret" in kinds          # stripe key
    assert "unauthenticated_api" in kinds    # profiles endpoint
    # the locked route produced nothing
    assert not any("locked" in f["location"]
                   for f in result["findings"])


def test_evidence_pack_is_complete_and_masked(fixture_server):
    base = fixture_server("vuln")
    result = _scanner().scan(base)
    api = next(f for f in result["findings"]
               if f["kind"] == "unauthenticated_api")
    pack = api["evidence"]
    # the source's own count, not a guess from the sample size
    assert pack["total_count"] == TOTAL
    # the FULL schema — every field the source serves
    assert set(pack["schema"]) == {
        "id", "full_name", "email", "date_of_birth",
        "passport_number", "phone"}
    # real rows, bounded
    assert len(pack["sample"]) == 5
    assert pack["sample"][0]["passport_number"] == PASSPORT
    # …and the masked twin proves shape without carrying value
    masked = pack["sample_masked"][0]["passport_number"]
    assert masked != PASSPORT and masked.startswith("A123")
    assert pack["sample_masked"][0]["date_of_birth"] == "1990-••-12"
    assert pack["request"]["url"].endswith("/rest/v1/profiles")
    assert pack["response"]["status"] == 200


def test_postgrest_root_check_reads_tables(fixture_server):
    base = fixture_server("vuln")
    findings = _scanner().check_postgrest(
        [base + "/rest/v1"], own_asset=False, sample_cap=5)
    assert len(findings) == 1
    f = findings[0]
    assert f["kind"] == "public_backend"
    assert f["meta"]["total_count"] == TOTAL
    assert f["meta"]["door"] == "no key at all"
    assert f["evidence"]["total_count"] == TOTAL


def test_locked_target_is_clean(fixture_server):
    base = fixture_server("locked")
    result = _scanner().scan(base)
    assert result["findings"] == []


def test_spa_catch_all_is_clean(fixture_server):
    # a SPA answering 200+HTML for every path — including /.env —
    # must not produce a single finding: content shape decides
    base = fixture_server("spa")
    result = _scanner().scan(base)
    assert result["findings"] == []


def test_unreachable_target_reports_honestly():
    result = _scanner().scan("http://127.0.0.1:1/nope")
    assert result["findings"] == []
    assert any("unreachable" in n for n in result["notes"])


# ── journal + own-asset law ──────────────────────────────────

def _service(tmp_path) -> AuditService:
    book = default_book(tmp_path)
    return AuditService(book, allow_private=True,
                        min_interval_s=0.0, osv_enabled=False)


def test_journal_masks_at_rest_and_report_masks_third_party(
        fixture_server, tmp_path):
    base = fixture_server("vuln")
    outcome = _service(tmp_path).run_web_scan(base)
    assert outcome["summary"]["total"] >= 4
    journal = (tmp_path / "data" / "social"
               / "secaudit.json").read_text()
    assert PASSPORT not in journal          # masked at rest
    report = Path(outcome["report_path"]).read_text()
    assert PASSPORT not in report           # third-party: masked
    assert "A123" in report                 # shape still proven


def test_own_asset_report_unmasked_journal_still_masked(
        fixture_server, tmp_path):
    base = fixture_server("vuln")
    service = _service(tmp_path)
    service.book.add_own_asset("127.0.0.1")
    outcome = service.run_web_scan(base)
    assert outcome["own_asset"] is True
    report = Path(outcome["report_path"]).read_text()
    assert PASSPORT in report               # his asset: unmasked
    journal = (tmp_path / "data" / "social"
               / "secaudit.json").read_text()
    assert PASSPORT not in journal          # at rest: always masked


def test_disclosure_draft_written_from_published_contact(
        fixture_server, tmp_path):
    base = fixture_server("vuln")
    outcome = _service(tmp_path).run_web_scan(base)
    assert outcome["draft_path"]
    draft = Path(outcome["draft_path"]).read_text()
    assert "security@fixture.example" in draft
    assert PASSPORT not in draft            # drafts never carry data
    contacts = discover_contacts_web(
        Fetcher(allow_private=True, min_interval_s=0.0).get, base)
    assert contacts == ["mailto:security@fixture.example"]


def test_mask_value_shapes():
    assert mask_value("1990-04-12") == "1990-••-12"
    assert mask_value(PASSPORT) == "A123••••78"
    assert mask_value(42) == 42


# ── the repo auditor ─────────────────────────────────────────

def _git(root: Path, *args: str) -> None:
    env = {**os.environ,
           "GIT_AUTHOR_NAME": "Fixture", "GIT_AUTHOR_EMAIL": "f@x.io",
           "GIT_COMMITTER_NAME": "Fixture",
           "GIT_COMMITTER_EMAIL": "f@x.io"}
    subprocess.run(["git", "-C", str(root), *args], check=True,
                   env=env, capture_output=True)


@pytest.fixture()
def repo_fixture(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "leaked.txt").write_text(
        f'OLD = "{STRIPE_LIVE_KEY}"\n')
    (root / "settings.py").write_text(
        f'AWS_KEY = "{AWS_ACCESS_KEY_ID}"\nDEBUG = True\n')
    (root / "app.py").write_text(
        "import sqlite3\n\n\n"
        "def find(conn, name):\n"
        "    return conn.execute(\"SELECT * FROM t WHERE x = "
        "'%s'\" % name)\n")
    (root / "package-lock.json").write_text(json.dumps({
        "name": "fixture", "lockfileVersion": 3,
        "packages": {"": {"name": "fixture", "version": "1.0.0"},
                     "node_modules/lodash": {"version": "4.17.20"}}}))
    (root / "SECURITY.md").write_text(
        "Report issues to security@fixture.example\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "first")
    _git(root, "rm", "-q", "leaked.txt")
    _git(root, "commit", "-q", "-m", "remove the leaked file")
    return root


def test_repo_audit_finds_tree_history_deps_and_patterns(
        repo_fixture):
    result = audit_repo(str(repo_fixture), osv_enabled=False)
    findings = result["findings"]
    tree = [f for f in findings
            if f["kind"] == "repo_secret"
            and not f["meta"].get("in_history")]
    history = [f for f in findings
               if f["kind"] == "repo_secret"
               and f["meta"].get("in_history")]
    assert any(f["meta"]["pattern"] == "aws_access_key"
               for f in tree)
    assert any("settings.py:1" in f["location"] for f in tree)
    # the stripe key is gone from the tree but alive in history
    assert any(f["meta"]["pattern"] == "stripe_live_key"
               for f in history)
    deps = [f for f in findings
            if f["kind"] == "vulnerable_dependency"]
    assert any(f["meta"]["advisory"] == "CVE-2020-8203"
               for f in deps)
    sql = [f for f in findings if f["kind"] == "sql_string_build"]
    assert sql and "app.py:5" in sql[0]["location"]
    assert result["advisory_source"] == "seed"
    assert result["deps_checked"] == 1


def test_repo_report_redacts_secrets(repo_fixture, tmp_path):
    book = default_book(tmp_path)
    service = AuditService(book, osv_enabled=False)
    outcome = service.run_repo_audit(str(repo_fixture))
    report = Path(outcome["report_path"]).read_text()
    assert STRIPE_LIVE_KEY not in report
    assert AWS_ACCESS_KEY_ID not in report
    assert "sk_l" in report and "AKIA" in report  # redacted forms
    # the fixture's own SECURITY.md gave the desk its contact
    assert outcome["draft_path"]
    assert "security@fixture.example" in Path(
        outcome["draft_path"]).read_text()


def test_draft_disclosure_carries_no_sample_data():
    findings = [{"kind": "public_backend",
                 "severity": "critical",
                 "title": "table readable",
                 "location": "https://x.example/rest/v1/profiles",
                 "meta": {}, "evidence": None, "detail": "",
                 "check": "postgrest"}]
    draft = draft_disclosure(target="https://x.example",
                             findings=findings,
                             contacts=["owner@x.example"])
    assert "owner@x.example" in draft
    assert "did not download your database" in draft


# ── own-asset registry matching ──────────────────────────────

def test_own_asset_matching(tmp_path):
    book = AuditBook(tmp_path).load()
    assert not book.is_own_asset("https://api.example.com/x")
    book.add_own_asset("example.com")
    assert book.is_own_asset("https://api.example.com/x")
    assert book.is_own_asset("https://example.com/")
    assert not book.is_own_asset("https://notexample.com/")
    book.add_own_asset("/home/owner/projects")
    assert book.is_own_asset("/home/owner/projects/app")
    assert not book.is_own_asset("/home/owner/other")
    assert book.remove_own_asset("example.com")
    assert not book.is_own_asset("https://example.com/")


# ── wiring ───────────────────────────────────────────────────

class _Gateway:
    def __init__(self):
        self.sent = []
        self.files = []

    def send(self, platform, chat, text, buttons=None,
             parse_mode=""):
        self.sent.append((platform, chat, text))
        return SimpleNamespace(ok=True)

    def send_file(self, platform, chat_key, path, caption=""):
        self.files.append((platform, chat_key, path, caption))
        return SimpleNamespace(ok=True)


def _rig(tmp_path):
    from nomorals.agents.partner.runtime_group import \
        RuntimeGroupMixin

    rig = RuntimeGroupMixin()
    rig.context = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=str(tmp_path)),
        db=None)
    rig.gateway = _Gateway()
    rig._owner_sender_ids = {"42"}
    rig._is_operator = lambda message: str(
        message.sender_id) == "42"
    return rig


def _msg(*, kind=ChatKind.DM, sender_id="42"):
    return SimpleNamespace(
        chat=SimpleNamespace(platform="telegram",
                             key="telegram:555",
                             chat_id="555", kind=kind, title=""),
        sender_id=sender_id, sender="Sender",
        sender_username="owner", text="", meta={},
        message_id="m1", reply_to="")


def test_audit_is_owner_only_and_registered():
    assert parse_control("/audit https://github.com/x/y").kind \
        == "audit"
    assert caller_may_run("audit", False) is False
    assert caller_may_run("audit", True) is True


def test_group_gets_a_pointer_never_findings(tmp_path):
    rig = _rig(tmp_path)
    group_msg = _msg(kind=ChatKind.GROUP)
    out = rig._control_audit("https://github.com/x/y", group_msg)
    assert "DM" in out
    out2 = rig._control_audit_scan("https://example.com", group_msg)
    assert "DM" in out2
    assert rig._secaudit_book().data["jobs"] == []


def test_scan_queues_and_drive_delivers_report(
        fixture_server, tmp_path):
    base = fixture_server("vuln")
    rig = _rig(tmp_path)
    msg = _msg()
    out = rig._control_audit_scan(base, msg)
    assert "queued" in out
    rig._secaudit_last = 0.0
    rig._drive_security_audits()
    assert rig.gateway.sent, "the card was delivered"
    card_text = rig.gateway.sent[0][2]
    assert "Security audit" in card_text
    assert "CRITICAL" in card_text.upper()
    assert rig.gateway.files, "the report file was delivered"
    report_path = rig.gateway.files[0][2]
    assert Path(report_path).exists()
    job = rig._secaudit_book().data["jobs"][0]
    assert job["status"] == "done"


def test_audit_own_registry_verbs(tmp_path):
    rig = _rig(tmp_path)
    msg = _msg()
    out = rig._control_audit("own add api.example.com", msg)
    assert "declared yours" in out
    out = rig._control_audit("own list", msg)
    assert "api.example.com" in out
    out = rig._control_audit("own remove api.example.com", msg)
    assert "removed" in out

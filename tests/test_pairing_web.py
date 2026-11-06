"""Pairing web contract (§15 polyglot rung 1): the machine seam
the PHP pairing page drives, the surface-rung registry, and the
PHP runtime ladder (resolve / provision / launch).

The contract tests drive the REAL registry + supervisor with a
fake bridge process (the test_pairing_pool shapes): a pairing
started through the contract is a pairing in the registry, a
toggle flipped through the contract is flipped for chat too —
one pool, every surface.
"""
from __future__ import annotations

import http.client
import io
import json
import tarfile
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.social import pairing_web
from nomorals.social.pairing import (
    AccountRecord,
    BridgeSupervisor,
    PairingRegistry,
    PairingUnavailable,
)
from nomorals.social.pairing_web import (
    PhpProvisionFailed,
    launch_pairing_page,
    pairing_action,
    pairing_state,
    pairing_surface_rungs,
    provision_php,
    resolve_php,
)
from nomorals.social.personal import PersonalSurfaces, PersonalVault
from nomorals.social.telegram_pairing import TelegramPairer

CODE_LINE = "PAIRING CODE for +2348012345678: WXYZ-2345\n"


class FakeProc:
    def __init__(self, lines=()):
        self.stdout = iter(list(lines))
        self.terminated = False
        self._poll = None

    def poll(self):
        return self._poll

    def terminate(self):
        self.terminated = True
        self._poll = 0


def make_svc(tmp_path, lines=(CODE_LINE,)):
    registry = PairingRegistry(tmp_path / "paired.json")
    spawns = []

    def fake_spawn(argv, env):
        proc = FakeProc(lines)
        spawns.append((argv, env, proc))
        return proc

    sup = BridgeSupervisor(
        registry, bridge_path="/repo/bridge/whatsapp-bridge.mjs",
        sessions_root=tmp_path / "sessions", spawn=fake_spawn)
    pairer = TelegramPairer(
        registry, sessions_root=tmp_path / "tg_sessions",
        api_id="", api_hash="")
    personal = PersonalSurfaces(
        registry, PersonalVault(tmp_path / "personal"))
    svc = {"supervisor": sup, "registry": registry,
           "pairer": pairer, "personal": personal,
           "workspace": tmp_path}
    return svc, registry, spawns


def context_for(tmp_path):
    return SimpleNamespace(settings=SimpleNamespace(
        workspace_dir=str(tmp_path),
        chat=SimpleNamespace(telegram_api_id="",
                             telegram_api_hash=""),
        api=SimpleNamespace(port=8787, token="")))


def wait_code(svc, account_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        got = svc["supervisor"].pairing_code(account_id)
        if got:
            return got
        time.sleep(0.02)
    return None


# ── the contract: state + actions over the real registry ────────────

def test_pair_action_and_state_round_trip(tmp_path):
    svc, registry, _ = make_svc(tmp_path)
    ctx = context_for(tmp_path)
    out = pairing_action(ctx, {"action": "pair_whatsapp",
                               "number": "2348012345678",
                               "label": "main line"}, svc=svc)
    assert out["ok"] is True
    assert out["account"] == "whatsapp:2348012345678"
    # The fresh state rides the action result (one round trip).
    assert out["state"]["counts"]["registered"] == 1
    code, _age = wait_code(svc, "whatsapp:2348012345678")
    assert code == "WXYZ-2345"
    state = pairing_state(ctx, svc=svc)
    row = state["accounts"][0]
    assert row["label"] == "main line"
    assert row["pairing_code"]["code"] == "WXYZ-2345"
    # Real toggle states: the full set, all off by default.
    assert set(row["toggles"]) == set(state["toggles"])
    assert not any(row["toggles"].values())
    # A fresh PairingRegistry over the same file (chat's view)
    # sees the same account — one pool.
    assert [r.id for r in registry.all()] == [row["id"]]


def test_toggle_use_stop_start_remove_actions(tmp_path):
    svc, registry, _ = make_svc(tmp_path)
    ctx = context_for(tmp_path)
    pairing_action(ctx, {"action": "pair_whatsapp",
                         "number": "2348012345678"}, svc=svc)
    aid = "whatsapp:2348012345678"

    out = pairing_action(ctx, {"action": "toggle", "id": aid,
                               "name": "status_saving", "on": True},
                         svc=svc)
    assert out["ok"] is True
    assert registry.get(aid).toggle("status_saving") is True

    bad = pairing_action(ctx, {"action": "toggle", "id": aid,
                               "name": "not_a_toggle", "on": True},
                         svc=svc)
    assert bad["ok"] is False
    assert "unknown personal toggle" in bad["error"]

    out = pairing_action(ctx, {"action": "use", "id": aid}, svc=svc)
    assert out["ok"] is True
    assert registry.get(aid).primary is True

    out = pairing_action(ctx, {"action": "stop", "id": aid}, svc=svc)
    assert out["ok"] is True
    assert registry.get(aid).status == "stopped"
    # The owner-stop intent is recorded (chat parity): no
    # reconcile may resurrect the surface behind the owner.
    assert registry.owner_stopped("whatsapp") is True

    out = pairing_action(ctx, {"action": "start", "id": aid}, svc=svc)
    assert out["ok"] is True
    assert registry.owner_stopped("whatsapp") is False

    out = pairing_action(ctx, {"action": "remove", "id": aid},
                         svc=svc)
    assert out["ok"] is True
    assert registry.get(aid) is None
    assert registry.owner_stopped("whatsapp") is False


def test_telegram_lifecycle_actions(tmp_path):
    svc, registry, _ = make_svc(tmp_path)
    ctx = context_for(tmp_path)
    registry.upsert(AccountRecord(
        id="telegram:55501", platform="telegram", identity="55501",
        status="linked", session_path=str(tmp_path / "nope")))
    out = pairing_action(ctx, {"action": "stop",
                               "id": "telegram:55501"}, svc=svc)
    assert out["ok"] is True
    assert registry.get("telegram:55501").status == "stopped"
    # start on Telegram is an authorization probe: no session
    # file → down, honestly, and the action says so.
    out = pairing_action(ctx, {"action": "start",
                               "id": "telegram:55501"}, svc=svc)
    assert out["ok"] is False
    assert "no longer authorizes" in out["error"]
    assert registry.get("telegram:55501").status == "down"
    out = pairing_action(ctx, {"action": "remove",
                               "id": "telegram:55501"}, svc=svc)
    assert out["ok"] is True
    assert registry.get("telegram:55501") is None


def test_action_failures_are_plain_words(tmp_path):
    svc, _registry, _ = make_svc(tmp_path)
    ctx = context_for(tmp_path)
    out = pairing_action(ctx, {"action": "explode"}, svc=svc)
    assert out["ok"] is False and "unknown pairing action" in out["error"]
    out = pairing_action(ctx, {"action": "pair_whatsapp",
                               "number": "123"}, svc=svc)
    assert out["ok"] is False
    assert "international format" in out["error"]
    out = pairing_action(ctx, {"action": "toggle", "id": "whatsapp:9",
                               "name": "view_once", "on": True}, svc=svc)
    assert out["ok"] is False and "which account" in out["error"]


def test_bundle_for_context_uses_the_real_registry_file(tmp_path):
    ctx = context_for(tmp_path)
    reg_path = tmp_path / "data" / "social" / "paired_accounts.json"
    reg_path.parent.mkdir(parents=True)
    seeded = PairingRegistry(reg_path)
    seeded.upsert(AccountRecord(
        id="whatsapp:2348012345678", platform="whatsapp",
        identity="2348012345678", status="linked"))
    state = pairing_state(ctx)  # no svc: builds the bundle
    assert state["counts"]["registered"] == 1
    assert state["accounts"][0]["status"] == "linked"
    # Cached on the context: a second call shares the supervisor.
    assert pairing_state(ctx)["counts"]["registered"] == 1
    assert ctx._pairing_web_svc is not None


# ── the API routes ride the same contract, owner-gated ──────────────

def _api_server(tmp_path):
    from nomorals.api.server import APIServer, _make_handler
    from nomorals.core.policy import Capability, CapabilitySet
    from nomorals.storage.db import Database
    from nomorals.tools.registry import ToolRegistry

    owner = "owner-token-for-pairing-web-tests"
    ro = "ro-token-for-pairing-web-tests"
    settings = SimpleNamespace(
        api=SimpleNamespace(token="", max_body_mb=0),
        backup_dir=str(tmp_path), workspace_dir=str(tmp_path),
        chat=SimpleNamespace(telegram_api_id="",
                             telegram_api_hash=""))
    context = SimpleNamespace(
        db=Database(":memory:"), settings=settings,
        tools=ToolRegistry(), bus=None, memory=None)
    api = APIServer(
        context, token=owner,
        principals={ro: ("ro", CapabilitySet.of(Capability.DB_READ))})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(api))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, httpd.server_address[1], owner, ro


def _http(port, method, path, token="", payload=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    conn.request(method, path,
                 body=json.dumps(payload) if payload is not None
                 else None,
                 headers=headers)
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    return resp.status, json.loads(raw or b"{}")


def test_api_routes_owner_gate_and_round_trip(tmp_path):
    reg_path = tmp_path / "data" / "social" / "paired_accounts.json"
    reg_path.parent.mkdir(parents=True)
    seeded = PairingRegistry(reg_path)
    seeded.upsert(AccountRecord(
        id="whatsapp:2348012345678", platform="whatsapp",
        identity="2348012345678", status="linked"))
    httpd, port, owner, ro = _api_server(tmp_path)
    try:
        status, body = _http(port, "GET", "/pair/state", token=owner)
        assert status == 200
        assert body["counts"]["registered"] == 1
        assert body["accounts"][0]["id"] == "whatsapp:2348012345678"
        # The bounded grant (no sys.config) is refused — the
        # contract is owner-only, like the HTML page.
        status, _ = _http(port, "GET", "/pair/state", token=ro)
        assert status == 403
        status, _ = _http(port, "POST", "/pair/action", token=ro,
                          payload={"action": "toggle",
                                   "id": "whatsapp:2348012345678",
                                   "name": "view_once", "on": True})
        assert status == 403
        # The owner's action round-trips into the real registry.
        status, body = _http(
            port, "POST", "/pair/action", token=owner,
            payload={"action": "toggle",
                     "id": "whatsapp:2348012345678",
                     "name": "view_once", "on": True})
        assert status == 200 and body["ok"] is True
        reread = PairingRegistry(reg_path)
        assert reread.get("whatsapp:2348012345678") \
            .toggle("view_once") is True
    finally:
        httpd.shutdown()
        httpd.server_close()


# ── the rung registry ───────────────────────────────────────────────

def test_rungs_report_php_availability_honestly(tmp_path):
    fake_php = tmp_path / "php"
    fake_php.write_text("#!/bin/sh\necho 'PHP 8.3.13 (cli)'\n")
    fake_php.chmod(0o755)
    env = {"PATH": str(tmp_path), "NM_PHP_BINARY": str(fake_php)}
    rungs = pairing_surface_rungs(env=env, workspace=tmp_path)
    by_name = {r["name"]: r for r in rungs}
    assert by_name["php-web"]["available"] is True
    assert "PHP 8.3.13" in by_name["php-web"]["detail"]
    assert by_name["api-page"]["available"] is True
    assert by_name["in-chat"]["available"] is True

    # php="" pins the probe's verdict (no runtime resolved) so
    # the unavailable branch is exercised on any machine.
    nowhere = pairing_surface_rungs(php="", workspace=tmp_path)
    php_rung = {r["name"]: r for r in nowhere}["php-web"]
    assert php_rung["available"] is False
    assert "provision" in php_rung["detail"]


def test_resolve_php_ladder(tmp_path):
    fake = tmp_path / "php"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    env = {"NM_PHP_BINARY": str(fake), "PATH": ""}
    assert resolve_php(env=env) == str(fake)
    # A configured php that does not exist fails honestly (None),
    # never a silent fall-back (resolve_node's contract).
    assert resolve_php(env={"NM_PHP_BINARY": "/nope/php",
                            "PATH": ""}) is None
    # Devon's managed copy for the workspace resolves.
    managed = pairing_web.php_managed_binary(tmp_path, "posix")
    managed.parent.mkdir(parents=True)
    managed.write_text("#!/bin/sh\n")
    managed.chmod(0o755)
    assert resolve_php(env={"PATH": ""}, platform="posix",
                       home=tmp_path / "h",
                       workspace=tmp_path, ladder=[]) == str(managed)


# ── PHP provisioning (staged machines only — no real installs) ──────

def _fake_php_tarball() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        data = b"#!/bin/sh\necho 'PHP 8.3.13 (cli)'\n"
        info = tarfile.TarInfo("php")
        info.size = len(data)
        info.mode = 0o755
        tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_provision_php_download_route(tmp_path):
    tarball = _fake_php_tarball()

    def downloader(url, dest):
        assert "static-php" in url and "8.3.13" in url
        dest.write_bytes(tarball)

    php = provision_php(platform="posix", machine="x86_64",
                        env={"PATH": ""}, home=tmp_path / "h",
                        workspace=tmp_path, downloader=downloader,
                        euid=1000)
    assert php == str(pairing_web.php_managed_binary(tmp_path,
                                                     "posix"))
    assert Path(php).is_file()


def test_provision_php_records_failed_attempts(tmp_path):
    def downloader(url, dest):
        raise OSError("network down")

    def runner(argv):
        return 1, "winget not here"

    with pytest.raises(PhpProvisionFailed) as err:
        provision_php(platform="posix", machine="x86_64",
                      env={"PATH": ""}, home=tmp_path / "h",
                      workspace=tmp_path, downloader=downloader,
                      runner=runner, euid=1000)
    attempts = " | ".join(err.value.attempts)
    assert "pinned download route failed" in attempts
    assert "apt-get route skipped" in attempts  # not root: recorded


# ── launching the page ──────────────────────────────────────────────

def test_launch_pairing_page_spawns_php_with_the_contract_env(tmp_path):
    fake_php = tmp_path / "php"
    fake_php.write_text("#!/bin/sh\n")
    fake_php.chmod(0o755)
    spawned = {}

    class FakeServerProc:
        pid = 4242

        def poll(self):
            return None

    def fake_spawn(argv, **kw):
        spawned["argv"] = argv
        spawned["env"] = kw.get("env") or {}
        return FakeServerProc()

    ctx = SimpleNamespace(settings=SimpleNamespace(
        workspace_dir=str(tmp_path),
        chat=SimpleNamespace(telegram_api_id="",
                             telegram_api_hash=""),
        api=SimpleNamespace(port=9999, token="owner-api-token")))
    out = launch_pairing_page(
        ctx, spawn=fake_spawn,
        env={"NM_PHP_BINARY": str(fake_php), "PATH": ""})
    assert out["url"].startswith("http://127.0.0.1:")
    assert out["token"]
    argv = spawned["argv"]
    assert argv[0] == str(fake_php) and "-S" in argv
    assert str(pairing_web.PAGE_FILE) in argv
    # The session dir is named explicitly (bare runtimes have no
    # session.save_path) and lives under the workspace.
    joined = " ".join(argv)
    assert "session.save_path=" in joined
    assert str(tmp_path) in joined
    env = spawned["env"]
    assert env["DEVON_API_BASE"] == "http://127.0.0.1:9999"
    assert env["DEVON_API_TOKEN"] == "owner-api-token"
    assert env["DEVON_PAIR_PAGE_TOKEN"] == out["token"]
    # The launch is recorded; the token persists across launches.
    record = json.loads(
        (tmp_path / "data" / "social"
         / "pairing_page_server.json").read_text())
    assert record["url"] == out["url"]
    token_file = tmp_path / "data" / "social" / "pairing_page_token"
    assert token_file.read_text().strip() == out["token"]


def test_launch_without_php_is_an_honest_refusal(tmp_path):
    ctx = context_for(tmp_path)
    with pytest.raises(PairingUnavailable) as err:
        launch_pairing_page(
            ctx, env={"PATH": str(tmp_path / "empty")},
            platform="posix", home=tmp_path / "nohome",
            provision=False)
    assert "no PHP runtime" in str(err.value)

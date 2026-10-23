"""Pump desk resilience pass — the 2026-10-23 EC2 all-feeds-down
autopsy, fixed and pinned:

1. per-source hard time budgets + degrade-and-continue (one
   stalled source can no longer stall the desk);
2. stuck fetch workers are abandoned, counted, and replaced —
   the worker ledger returns to baseline;
3. the dRPC http-400: the desk's request is well-formed (proven
   shape); the provider's JSON-RPC error body is surfaced with
   its code instead of a bare ``http-400``;
4. DexScreener's dead-with-zero-attempts registry state: a source
   with an empty attempts ledger presents ``unprobed``, never
   ``dead`` — and the desk now records DexScreener's real reads;
5. the pump.fun frontend ``PermissionError``: a transport denial
   (the OS refusing the connection) is wrapped and recorded as
   ``transport-denied`` — never under the raw class name, which
   the house taxonomy maps to a POLICY denial;
6. one user-agent from the config surface for every desk fetch;
7. desk stand-down: named, journaled, surfaced with per-source
   reasons, and lifted automatically by the first success;
8. PumpPortal key wiring through the provisioning/vault surface
   (env bridge, presence-only reporting, dummy values in tests).
"""

from __future__ import annotations

import io
import json
import threading
import time
import urllib.error

import pytest

from nomorals.agents.pumpdesk import chain, firehose, outcomes, sources
from nomorals.agents.pumpdesk.chain import RPCError, RoutedRPC, SolanaRPC
from nomorals.agents.pumpdesk.desk import PumpDesk
from nomorals.agents.pumpdesk.firehose import Firehose
from nomorals.agents.pumpdesk.wsclient import (
    TRANSPORT_DENIED_CLASS, TransportDeniedError)
from nomorals.media.source_registry import SourceRegistry


# ── helpers ───────────────────────────────────────────────────────

def _seeded_registry() -> SourceRegistry:
    reg = SourceRegistry()
    sources.ensure_pump_sources(reg)
    return reg


def _stand_down(reg: SourceRegistry, domain: str) -> None:
    desc = reg.for_domain(domain)
    for _ in range(3):
        reg.record_outcome(desc, False, error_class="TimeoutError")


class _Resp:
    """A urlopen context-manager double carrying a JSON body."""

    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._body


# ── 1. per-source budgets + degrade-and-continue ──────────────────

def test_rpc_budget_exhaustion_fails_fast_with_its_class(monkeypatch):
    seen = []
    calls = {"n": 0}

    def slow_urlopen(req, timeout=None):
        calls["n"] += 1
        time.sleep(0.35)
        return _Resp({"jsonrpc": "2.0", "id": 1, "result": 7})

    monkeypatch.setattr(chain.urllib.request, "urlopen", slow_urlopen)
    rpc = SolanaRPC("https://slow.example", timeout=5.0, budget_s=0.3,
                    on_outcome=lambda ok, lat, cls="": seen.append(
                        (ok, cls)))
    assert rpc.get_slot() == 7          # first call fits the budget
    with pytest.raises(RPCError) as ei:
        rpc.get_slot()                  # budget spent: no network try
    assert "budget" in str(ei.value)
    assert calls["n"] == 1              # the second call never dialed
    assert seen[-1] == (False, chain.SOURCE_BUDGET_CLASS)


def test_routed_rpc_degrades_to_the_healthy_route(monkeypatch):
    """Route A stalls past its budget; the read still answers
    from route B — the desk degrades, it does not stall."""
    def urlopen(req, timeout=None):
        if "stall" in req.full_url:
            time.sleep(0.3)
            return _Resp({"jsonrpc": "2.0", "id": 1, "result": 1})
        return _Resp({"jsonrpc": "2.0", "id": 1, "result": 99})

    monkeypatch.setattr(chain.urllib.request, "urlopen", urlopen)
    a = SolanaRPC("https://stall.example", timeout=5.0, budget_s=0.2)
    b = SolanaRPC("https://fast.example", timeout=5.0, budget_s=60.0)
    routed = RoutedRPC([a, b])
    assert routed.get_slot() == 1       # A answers once, slowly
    t0 = time.time()
    assert routed.get_slot() == 99      # A's budget is spent → B
    assert time.time() - t0 < 0.25      # no second stall was paid
    routed.reset_budgets()              # a new cycle re-arms A
    assert a.budget_left() == pytest.approx(0.2)


def test_frontend_failure_degrades_one_leg_only(tmp_path, monkeypatch):
    """A dead frontend leg names itself in the cycle report; the
    cycle still completes and the desk state stays honest."""
    reg = _seeded_registry()
    desk = PumpDesk(context=None, desk_dir=str(tmp_path), registry=reg)

    def dead_frontend(http_get, limit=50):
        raise TimeoutError("frontend stalled")

    monkeypatch.setattr(firehose, "fetch_frontend", dead_frontend)
    out = desk.watch_cycle()
    assert out["frontend"]["error"].startswith("TimeoutError")
    assert "frontend-api-v3.pump.fun" in out["degraded_sources"]
    assert out["stand_down"] is None    # other sources unattempted,
    # not down — no fake total outage is declared
    stats = reg.stats_for(reg.for_domain("frontend-api-v3.pump.fun"))
    assert stats["fail"] == 1
    assert stats["last_error_class"] == "TimeoutError"


def test_budgeted_http_get_caps_total_spend(tmp_path, monkeypatch):
    """The frontend source's budget is a TOTAL across the whole
    multi-page scan, not a per-page allowance: once spent, the
    next page fails fast instead of re-arming a stalled source."""
    reg = _seeded_registry()
    desk = PumpDesk(context=None, desk_dir=str(tmp_path), registry=reg)
    monkeypatch.setattr(sources, "desk_source_budget_s", lambda: 0.3)
    dials = {"n": 0}

    def slow_get(url, headers, timeout=20.0):
        dials["n"] += 1
        time.sleep(0.35)
        return 200, "[]"

    monkeypatch.setattr(firehose, "_urllib_get", slow_get)
    get = desk._budgeted_http_get()
    assert get("https://x/1", {}) == (200, "[]")
    with pytest.raises(TimeoutError):
        get("https://x/2", {})
    assert dials["n"] == 1              # page 2 never dialed


# ── 2. stuck-worker cleanup ───────────────────────────────────────

class _StuckBackend:
    name = "stuck"
    domain = "pumpportal.fun"

    def __init__(self):
        self.release = threading.Event()

    def serve(self, emit, ctl):
        self.release.wait(timeout=30.0)


def test_stuck_slice_worker_is_abandoned_and_replaced(monkeypatch):
    monkeypatch.setattr(Firehose, "SLICE_GRACE_S", 0.2)
    fh = Firehose(lambda k, p: None, registry=None)
    stuck = _StuckBackend()
    monkeypatch.setattr(fh, "_backends", lambda: [stuck])
    baseline = fh.live_workers()
    t0 = time.time()
    rep = fh.run_slice(0.2)
    elapsed = time.time() - t0
    assert rep["ok"] is False and rep.get("worker_stuck") is True
    assert elapsed < 5.0                # the slice did NOT wait 30s
    assert fh.live_workers() == baseline + 1   # visible, not hidden
    stuck.release.set()                 # the wedged serve now exits
    deadline = time.time() + 5.0
    while fh.live_workers() != baseline and time.time() < deadline:
        time.sleep(0.05)
    assert fh.live_workers() == baseline       # ledger back to base

    class _GoodBackend:
        name = "good"
        domain = "pumpportal.fun"

        def serve(self, emit, ctl):
            emit("launch", {"mint": "M"})

    monkeypatch.setattr(fh, "_backends", lambda: [_GoodBackend()])
    got = []
    fh.emit = lambda k, p: got.append((k, p))
    rep2 = fh.run_slice(0.5)
    assert rep2["ok"] is True and rep2["events"] == 1
    assert fh.live_workers() == baseline


# ── 3. dRPC: request shape + provider-error surfacing ─────────────

#: The response dRPC's free public endpoint actually returned to
#: the desk's exact request on 2026-10-23 (recorded live): HTTP
#: 400 carrying a JSON-RPC error body — the Solana chain is not
#: served on the free plan.  Identical requests succeed on the
#: other public routes, so the request was never malformed; what
#: the desk got wrong was discarding this body and recording a
#: bare ``http-400``.
DRPC_400_BODY = json.dumps({
    "id": 1, "jsonrpc": "2.0",
    "error": {"message": "chain is not available on free plan, "
                         "please upgrade to paid plan",
              "code": 35}}).encode()


def test_rpc_request_shape_is_well_formed(monkeypatch):
    captured = {}

    def urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        captured["headers"] = {k.lower(): v for k, v in
                               req.headers.items()}
        return _Resp({"jsonrpc": "2.0", "id": 1, "result": 460580366})

    monkeypatch.setattr(chain.urllib.request, "urlopen", urlopen)
    rpc = SolanaRPC("https://solana.drpc.org")
    assert rpc.get_slot() == 460580366
    assert captured["body"] == {
        "jsonrpc": "2.0", "id": 1, "method": "getSlot",
        "params": [{"commitment": "confirmed"}]}
    assert captured["headers"]["content-type"] == "application/json"
    assert captured["headers"]["user-agent"] == sources.desk_user_agent()


def test_http_error_body_is_surfaced_with_provider_code(monkeypatch):
    seen = []

    def urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 400, "Bad Request", {}, io.BytesIO(DRPC_400_BODY))

    monkeypatch.setattr(chain.urllib.request, "urlopen", urlopen)
    rpc = SolanaRPC("https://solana.drpc.org",
                    on_outcome=lambda ok, lat, cls="": seen.append(cls))
    with pytest.raises(RPCError) as ei:
        rpc.get_slot()
    assert "chain is not available on free plan" in str(ei.value)
    assert seen == ["rpc-error-35"]     # NOT a bare http-400


def test_200_with_error_body_is_surfaced_with_provider_code(monkeypatch):
    seen = []
    monkeypatch.setattr(
        chain.urllib.request, "urlopen",
        lambda req, timeout=None: _Resp(json.loads(DRPC_400_BODY)))
    rpc = SolanaRPC("https://solana.drpc.org",
                    on_outcome=lambda ok, lat, cls="": seen.append(cls))
    with pytest.raises(RPCError) as ei:
        rpc.get_slot()
    assert "free plan" in str(ei.value)
    assert seen == ["rpc-error-35"]


# ── 4. zero-attempt sources never present as dead ─────────────────

def test_unattempted_source_reads_unprobed_not_dead():
    reg = _seeded_registry()
    desc = reg.for_domain("api.dexscreener.com")
    # the health loop probes the API host's root, which 404s —
    # the exact EC2 evidence (health dead, stats all zero)
    summary = reg.verify_health(fetcher=lambda url: (404, b"gone"))
    assert summary["checked"] >= 7
    row = [s for s in sources.pump_status(reg)["sources"]
           if s["domain"] == "api.dexscreener.com"][0]
    assert row["stats"]["ok"] == 0 and row["stats"]["fail"] == 0
    assert row["stats"]["last_seen"] == 0.0
    assert row["health"] == "unprobed"          # NOT "dead"
    assert reg.is_active(desc) is True
    # once the desk has REALLY attempted the source and failed,
    # dead-after-failures is a legitimate presentation again
    reg.record_outcome(desc, False, error_class="http-500")
    assert reg.health_for(desc) == "dead"


def test_dex_reads_are_recorded_in_the_registry(tmp_path, monkeypatch):
    """The desk's DexScreener grading reads land as outcomes —
    the attempts ledger can no longer sit at zero while the
    grader works."""
    reg = _seeded_registry()
    desk = PumpDesk(context=None, desk_dir=str(tmp_path), registry=reg)
    monkeypatch.setattr(
        outcomes, "default_http_get",
        lambda url, headers, timeout=20.0: (
            200, json.dumps({"pairs": [{
                "chainId": "solana", "dexId": "pumpswap",
                "pairAddress": "P", "priceNative": "0.001",
                "liquidity": {"usd": 10}}]})))
    snap = desk._dex_snapshot("MintX")
    assert snap is not None and snap["dex"] == "pumpswap"
    stats = reg.stats_for(reg.for_domain("api.dexscreener.com"))
    assert stats["ok"] == 1 and stats["last_seen"] > 0
    monkeypatch.setattr(
        outcomes, "default_http_get",
        lambda url, headers, timeout=20.0: (503, "down"))
    assert desk._dex_snapshot("MintX") is None
    stats = reg.stats_for(reg.for_domain("api.dexscreener.com"))
    assert stats["fail"] == 1
    assert stats["last_error_class"] == "http-503"


# ── 5. transport denial is never recorded as PermissionError ──────

def test_urllib_get_wraps_permission_error():
    def denied(req, timeout=None):
        raise PermissionError(13, "Permission denied")

    orig = firehose.urllib.request.urlopen
    firehose.urllib.request.urlopen = denied
    try:
        with pytest.raises(TransportDeniedError):
            firehose._urllib_get("https://frontend-api-v3.pump.fun/x", {})
    finally:
        firehose.urllib.request.urlopen = orig


def test_urllib_get_wraps_urlerror_with_permission_reason():
    def denied(req, timeout=None):
        raise urllib.error.URLError(PermissionError(10013, "WSAEACCES"))

    orig = firehose.urllib.request.urlopen
    firehose.urllib.request.urlopen = denied
    try:
        with pytest.raises(TransportDeniedError):
            firehose._urllib_get("https://frontend-api-v3.pump.fun/x", {})
    finally:
        firehose.urllib.request.urlopen = orig


def test_rpc_call_records_transport_denied_not_permission_error(
        monkeypatch):
    seen = []

    def denied(req, timeout=None):
        raise PermissionError(10013, "access forbidden by OS")

    monkeypatch.setattr(chain.urllib.request, "urlopen", denied)
    rpc = SolanaRPC("https://solana-rpc.publicnode.com",
                    on_outcome=lambda ok, lat, cls="": seen.append(cls))
    with pytest.raises(RPCError):
        rpc.get_slot()
    assert seen == [TRANSPORT_DENIED_CLASS]


def test_frontend_transport_denial_records_its_real_class(
        tmp_path, monkeypatch):
    """The original EC2 raise path, reproduced: the frontend
    transport refuses at the OS layer; the registry must record
    transport-denied, never 'PermissionError' (the taxonomy's
    policy-denial name)."""
    reg = _seeded_registry()
    desk = PumpDesk(context=None, desk_dir=str(tmp_path), registry=reg)

    def denied_frontend(http_get, limit=50):
        raise TransportDeniedError("transport denied: WSAEACCES")

    monkeypatch.setattr(firehose, "fetch_frontend", denied_frontend)
    desk.watch_cycle()
    stats = reg.stats_for(reg.for_domain("frontend-api-v3.pump.fun"))
    assert stats["fail"] == 1
    assert stats["last_error_class"] == TRANSPORT_DENIED_CLASS


def test_ws_connect_wraps_permission_error(monkeypatch):
    import socket as _socket

    from nomorals.agents.pumpdesk.wsclient import WebSocketClient

    def denied(addr, timeout=None):
        raise PermissionError(10013, "access forbidden by OS")

    monkeypatch.setattr(_socket, "create_connection", denied)
    ws = WebSocketClient("wss://pumpportal.fun/api/data")
    with pytest.raises(TransportDeniedError):
        ws.connect()


# ── 6. one user-agent from the config surface ─────────────────────

def test_user_agent_comes_from_config_and_falls_back(monkeypatch):
    from nomorals.core.config import Settings, set_settings
    try:
        st = Settings()
        st.tools.user_agent = "DeskProbe/9.9"
        set_settings(st)
        assert sources.desk_user_agent() == "DeskProbe/9.9"
        st.tools.user_agent = ""
        assert sources.desk_user_agent() == sources.DEFAULT_USER_AGENT
    finally:
        from nomorals.core.config import reset_settings
        reset_settings()
    monkeypatch.setattr(sources, "_tools_settings", lambda: None)
    assert sources.desk_user_agent() == "nomorals-pumpdesk"
    # and the transports actually send the resolved value
    sent = {}
    firehose.fetch_frontend(
        lambda url, headers: (sent.update(headers) or (200, "[]")))
    assert sent["User-Agent"] == "nomorals-pumpdesk"


# ── 7. stand-down: named, journaled, self-recovering ──────────────

_RPC_DOMAINS = ("api.mainnet-beta.solana.com",
                "solana-rpc.publicnode.com", "solana.drpc.org")


def test_stand_down_enters_journals_and_recovers(tmp_path, monkeypatch):
    reg = _seeded_registry()
    desk = PumpDesk(context=None, desk_dir=str(tmp_path), registry=reg)
    for d in _RPC_DOMAINS + ("frontend-api-v3.pump.fun",):
        _stand_down(reg, d)
    monkeypatch.setattr(
        firehose, "fetch_frontend",
        lambda http_get, limit=50: (_ for _ in ()).throw(
            TimeoutError("still down")))
    out = desk.watch_cycle()
    assert out["stand_down"] is not None
    assert out["stand_down"]["state"] == "stand-down"
    assert "solana.drpc.org" in out["stand_down"]["reasons"]
    assert set(out["degraded_sources"]) >= set(_RPC_DOMAINS)
    events = desk.store.health_events()
    assert events and events[-1]["state"] == "stand-down"
    assert any(s["domain"] == "solana.drpc.org"
               for s in events[-1]["sources"])
    st = desk.status()
    assert st["desk_state"]["state"] == "stand-down"
    from nomorals.agents.pumpdesk.desk import render_status
    assert "STAND-DOWN" in render_status(st)
    # recovery is automatic: one source answers, the state lifts
    desc = reg.for_domain("frontend-api-v3.pump.fun")
    reg.record_outcome(desc, True)
    monkeypatch.setattr(firehose, "fetch_frontend",
                        lambda http_get, limit=50: [])
    out2 = desk.watch_cycle()
    assert out2["stand_down"] is None
    events2 = desk.store.health_events()
    assert events2[-1]["state"] in ("ok", "degraded")
    assert events2[-1].get("from") == "stand-down"


def test_partial_outage_is_degraded_not_stand_down(tmp_path):
    reg = _seeded_registry()
    desk = PumpDesk(context=None, desk_dir=str(tmp_path), registry=reg)
    _stand_down(reg, "solana.drpc.org")
    desc = reg.for_domain("api.mainnet-beta.solana.com")
    reg.record_outcome(desc, True)
    states = desk._source_states()
    ds = desk._desk_state(states)
    assert ds["state"] == "degraded"
    assert [s["domain"] for s in ds["down"]] == ["solana.drpc.org"]


# ── 8. PumpPortal key: vault surface, env bridge, socket ──────────

def test_pumpportal_provisioning_registration_and_bridge(
        tmp_path, monkeypatch):
    from nomorals.accounts.provisioning import (
        bridge_vault_credentials, get_provision_service,
        validate_service_token)
    ids = sources.ensure_provisioning()
    assert "pumpportal" in ids and "helius" in ids
    svc = get_provision_service("pumpportal")
    assert svc is not None
    assert svc.vault_service == "pumpportal"
    assert svc.env_keys == ("PUMPPORTAL_API_KEY",)
    # no GET-only key check exists at PumpPortal: validation is
    # honestly inconclusive, never a fabricated ok
    res = validate_service_token(
        svc, "dummy-key", http_get=lambda url, headers: (200, "{}"))
    assert res.status == "inconclusive"

    # a REAL vault (the provisioning test pattern): the key is
    # stored under the pumpportal service as an api_key credential
    # and the bridge — not any desk code — moves it into the env.
    import os as _os

    from nomorals.accounts.vault import CredentialVault
    from nomorals.storage.db import Database
    db = Database(_os.path.join(str(tmp_path), "vault.db"))
    db.migrate()
    vault = CredentialVault(db, master_passphrase="test-pass")
    vault.store("pumpportal", "owner", "dummy-key-123",
                credential_type="api_key")
    environ: dict = {}
    bridged = bridge_vault_credentials(vault, environ=environ)
    assert "PUMPPORTAL_API_KEY" in bridged
    assert environ["PUMPPORTAL_API_KEY"] == "dummy-key-123"
    monkeypatch.setenv("PUMPPORTAL_API_KEY", "dummy-key-123")
    assert sources.pumpportal_api_key() == "dummy-key-123"
    # the key reaches the socket URL (PumpPortal's own convention)
    backend = firehose.PumpPortalBackend(
        api_key=sources.pumpportal_api_key())
    assert backend.ws_url.endswith("?api-key=dummy-key-123")
    # keyless stays exactly the historical free socket
    monkeypatch.delenv("PUMPPORTAL_API_KEY")
    assert sources.pumpportal_api_key() == ""
    assert firehose.PumpPortalBackend().ws_url == firehose.PUMPPORTAL_WS
    # and the key VALUE appears in no status surface
    reg = _seeded_registry()
    assert "dummy-key-123" not in json.dumps(sources.pump_status(reg))

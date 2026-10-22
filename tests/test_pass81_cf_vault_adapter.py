"""Pass #81: the Cloudflare cookie rung is armed in production.

Pass #79 gave the download kit's Cloudflare state machine its first
consumers (browser/cloudflare_solver.py, media/cloudflare_gate.py)
but named the gap itself: "no production vault adapter exists for
the CF cookie store — both consumers accept the duck-typed store,
but nothing in production wires one yet, so the cookie rung is
armed only where a caller injects a store." In production nobody
injected one. This suite pins the fix:

(a) a production adapter over the ACCOUNTS vault (the #74 door,
    open_vault) round-trips a clearance in the kit's exact shape,
    and the #79-proven machine transitions run through it —
    stored cookie → inject → PASSED; a cookie that does not clear
    → COOKIE_EXPIRED (terminal); an expired entry → the gate's
    "expired" verdict — with NO parallel expiry logic anywhere;
(b) a locked vault (no seal / power key / passphrase at hand) makes
    the rung report UNAVAILABLE with the reason, escapes no
    exception, and writes NO plaintext cookie file anywhere;
(c) the production composition points — MediaDownloader (the
    runtime_media / tools construction shape: context, no explicit
    store) and the browser kind/backend entries (constructed bare,
    exactly as the flows build them) — receive a working vault
    store BY DEFAULT;
(d) an explicitly injected store still wins over the default.
"""

from __future__ import annotations

import http.cookiejar
import json
import os
import time

import pytest

from nomorals.accounts.cf_cookie_store import (
    CookieStoreUnavailable,
    VaultCookieStore,
    default_cf_cookie_store,
)
from nomorals.browser import cloudflare_solver
from nomorals.browser.cloudflare_solver import (
    CloudflareSolver,
    LocalCloudflareBackend,
    solve_cloudflare_kind,
)
from nomorals.media.cloudflare_gate import CloudflareGate
from nomorals.media.download_kit import (
    get_or_request_cookie,
    store_owner_cookie,
)
from nomorals.storage.db import Database

PASSPHRASE = "pass81-test-vault-passphrase"
CLEARANCE = "SECRET-CLEARANCE-VALUE-81"

CF_HTML = """<html><head><title>Just a moment...</title></head><body>
<div id="challenge-running">Checking your browser before accessing
x.test.</div><p>challenges.cloudflare.com</p></body></html>"""

CLEAN_HTML = "<html><body><h1>the article</h1></body></html>"


# ── fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture()
def home(tmp_path):
    """A tmp home holding a REAL accounts vault db file."""
    return tmp_path


@pytest.fixture()
def vault_env(monkeypatch):
    """The owner's passphrase is at hand (open_vault's env door)."""
    monkeypatch.setenv("NM_VAULT_PASSPHRASE", PASSPHRASE)
    yield
    monkeypatch.delenv("NM_VAULT_PASSPHRASE", raising=False)


@pytest.fixture()
def locked_env(monkeypatch):
    """No unlock material anywhere: no env passphrase, context-less."""
    monkeypatch.delenv("NM_VAULT_PASSPHRASE", raising=False)
    yield


def _db(home) -> Database:
    return Database(str(home / "vault.db"))


def _adapter(home, context=None) -> VaultCookieStore:
    return VaultCookieStore(_db(home), context=context,
                            actor="pass81-test")


class FakeCloudflareTab:
    """Same shape as the #79 suite's tab: challenged until the right
    cf_clearance cookie lands (or the wait clears it)."""

    def __init__(self, *, clears_on_wait: bool = False,
                 cookie_value: str = "") -> None:
        self.challenged = True
        self._clears_on_wait = clears_on_wait
        self._cookie_value = cookie_value
        self.added: list[dict] = []
        self.reloads = 0
        self.html_calls = 0
        self.fresh_value = ""

    @property
    def url(self) -> str:
        return "https://x.test/article"

    def html(self) -> str:
        self.html_calls += 1
        if self._clears_on_wait and self.html_calls > 1:
            self.challenged = False
        return CF_HTML if self.challenged else CLEAN_HTML

    def cookies(self) -> list[dict]:
        if self.challenged:
            return []
        if self.fresh_value:
            return [{"name": "cf_clearance", "value": self.fresh_value,
                     "domain": ".x.test"}]
        return []

    def add_cookies(self, cookies: list[dict]) -> None:
        self.added.extend(cookies)
        for c in cookies:
            if c.get("name") == "cf_clearance" and \
                    c.get("value") == self._cookie_value:
                self.challenged = False

    def reload(self) -> None:
        self.reloads += 1


def _solver(store, **kw) -> CloudflareSolver:
    kw.setdefault("wait_s", 0.0)
    kw.setdefault("sleeper", lambda _s: None)
    kw.setdefault("clock", lambda: 0.0)
    return CloudflareSolver(cookie_store=store, **kw)


class FakeStore:
    """The kit's duck-typed cookie store: get/set (as #79)."""

    def __init__(self) -> None:
        self.data: dict = {}

    def get(self, key: str):
        return self.data.get(key)

    def set(self, key: str, value) -> None:
        self.data[key] = value


# ── (a) adapter round-trip through a REAL vault ──────────────────────────

def test_adapter_round_trip_kit_shape(home, vault_env):
    store = _adapter(home)
    assert store.available() is True
    assert store.unavailable_reason == ""
    assert store_owner_cookie(store, "EXAMPLE.com", CLEARANCE,
                              ttl_s=3600.0) is True
    entry = store.get("cloudflare_cookies:example.com")
    assert isinstance(entry, dict)
    assert entry["value"] == CLEARANCE
    assert entry["domain"] == "example.com"
    assert entry["expires_at"] > time.time()
    # the kit reads it as a live cookie, in its own words
    status, payload = get_or_request_cookie(store, "example.com")
    assert status == "have" and payload == CLEARANCE
    # a second adapter instance over the same db sees the same entry:
    # the vault is the store, not the object
    again = _adapter(home)
    assert again.get("cloudflare_cookies:example.com")["value"] == \
        CLEARANCE
    # absent domain → None, never an exception, never a fabrication
    assert store.get("cloudflare_cookies:absent.test") is None
    status, _ = get_or_request_cookie(store, "absent.test")
    assert status == "ask"


def test_adapter_clearance_is_encrypted_at_rest(home, vault_env):
    store = _adapter(home)
    assert store_owner_cookie(store, "x.test", CLEARANCE) is True
    raw = (home / "vault.db").read_bytes()
    assert CLEARANCE.encode() not in raw


def test_real_adapter_drives_solver_inject_to_passed(home, vault_env):
    store = _adapter(home)
    assert store_owner_cookie(store, "x.test", "STORED-COOKIE") is True
    tab = FakeCloudflareTab(cookie_value="STORED-COOKIE")
    out = _solver(store).solve_tab(tab)
    assert out["ok"], out
    assert out["state"] == "passed"
    assert tab.added and tab.added[0]["value"] == "STORED-COOKIE"


def test_real_adapter_cookie_that_fails_ends_cookie_expired(home,
                                                            vault_env):
    store = _adapter(home)
    assert store_owner_cookie(store, "x.test", "STALE-COOKIE") is True
    # the tab never clears: the injected cookie did not pass the wall
    tab = FakeCloudflareTab(cookie_value="SOMETHING-ELSE")
    solver = _solver(store)
    out = solver.solve_tab(tab)
    assert out["ok"] is False
    assert out["state"] == "cookie_expired"      # the machine's terminal
    assert solver.machine.terminal is True


def test_real_adapter_expired_entry_is_the_kits_verdict(home, vault_env):
    store = _adapter(home)
    assert store_owner_cookie(store, "x.test", "OLD-COOKIE",
                              ttl_s=-60.0) is True
    # the adapter returns the entry untouched — expiry is the KIT's
    # law (get_or_request_cookie / the gate), not a parallel check
    entry = store.get("cloudflare_cookies:x.test")
    assert entry["value"] == "OLD-COOKIE"
    status, _ = get_or_request_cookie(store, "x.test")
    assert status == "expired"
    gate = CloudflareGate(store)
    assert gate.inspect(CF_HTML) is True
    decision = gate.cookie_decision("https://x.test/article")
    assert decision["action"] == "expired"
    assert decision["cookie"] == ""


def test_real_adapter_capture_on_managed_clear(home, vault_env):
    store = _adapter(home)
    tab = FakeCloudflareTab(clears_on_wait=True)
    tab.fresh_value = "FRESH-CLEARANCE"
    out = _solver(store).solve_tab(tab)
    assert out["ok"], out
    assert "stored in the vault" in out["detail"]
    entry = store.get("cloudflare_cookies:x.test")
    assert entry["value"] == "FRESH-CLEARANCE"


# ── (b) locked vault: unavailable truth, no crash, no plaintext ─────────

def test_locked_vault_adapter_reports_unavailable(home, locked_env):
    store = _adapter(home)
    assert store.available() is False
    reason = store.unavailable_reason
    assert reason, "a locked vault must name why it is unavailable"
    # get/set behave for the kit: get → None (no entry, no raise),
    # set → a clear failure the kit turns into False
    assert store.get("cloudflare_cookies:x.test") is None
    with pytest.raises(CookieStoreUnavailable):
        store.set("cloudflare_cookies:x.test",
                  {"value": "X", "expires_at": time.time() + 60,
                   "domain": "x.test"})
    assert store_owner_cookie(store, "x.test", CLEARANCE) is False
    status, _ = get_or_request_cookie(store, "x.test")
    assert status == "ask"


def test_locked_vault_flows_proceed_without_crash(home, locked_env):
    store = _adapter(home)
    tab = FakeCloudflareTab()
    out = _solver(store).solve_tab(tab)
    assert out["ok"] is False
    assert "unavailable" in out["detail"].lower(), out["detail"]
    gate = CloudflareGate(store)
    assert gate.inspect(CF_HTML) is True
    decision = gate.cookie_decision("https://x.test/article")
    assert decision["action"] == "ask"
    assert "unavailable" in decision["message"].lower(), decision


def test_locked_vault_writes_no_plaintext_anywhere(home, vault_env):
    # even a previously-unlocked home: once the material is gone the
    # rung must not fall back to a plaintext jar beside the vault
    store = _adapter(home)
    assert store_owner_cookie(store, "x.test", CLEARANCE) is True
    os.environ.pop("NM_VAULT_PASSPHRASE", None)
    locked = _adapter(home)
    assert locked.available() is False
    assert store_owner_cookie(locked, "x.test", "SECOND-VALUE") is False
    for path in home.rglob("*"):
        if path.is_file():
            assert CLEARANCE.encode() not in path.read_bytes()
            assert b"SECOND-VALUE" not in path.read_bytes()


def test_no_db_means_no_default_store():
    assert default_cf_cookie_store(None) is None
    assert default_cf_cookie_store(object()) is None  # no .db at all


# ── (c) production composition receives the default ──────────────────────

def test_downloader_default_store_from_context(home, vault_env):
    from nomorals.media.downloader import MediaDownloader
    context = type("Ctx", (), {"db": _db(home), "settings": None})()
    dl = MediaDownloader(context)          # runtime_media's shape
    assert dl._cookie_store is None        # nothing injected
    store = dl._cf_cookie_store()          # the lazy default (pass #76
    #                                        accessor pattern)
    assert isinstance(store, VaultCookieStore)
    assert store_owner_cookie(store, "x.test", "DL-COOKIE") is True
    gate_store_entry = store.get("cloudflare_cookies:x.test")
    assert gate_store_entry["value"] == "DL-COOKIE"


def test_downloader_gate_clears_with_default_store(home, vault_env):
    from nomorals.media.downloader import DownloadReport, MediaDownloader
    context = type("Ctx", (), {"db": _db(home), "settings": None})()
    dl = MediaDownloader(context)
    store = dl._cf_cookie_store()
    assert store_owner_cookie(store, "x.test", "GATE-COOKIE") is True

    class Session:
        def __init__(self) -> None:
            self.cookie_jar = http.cookiejar.CookieJar()
            self.cleared = False

        def html(self):
            return {"html": CLEAN_HTML if self.cleared else CF_HTML}

        def open(self, url):
            names = [c.name for c in self.cookie_jar]
            if "cf_clearance" in names:
                self.cleared = True
            return {"ok": True}

    report = DownloadReport(url="https://x.test/article")
    dl._cloudflare_gate(report, Session(), "https://x.test/article")
    notes = " ".join(s.note for s in report.stages)
    assert "vault clearance injected" in notes, notes
    assert "wall cleared" in notes, notes


def test_downloader_without_context_keeps_detect_only(home, locked_env):
    from nomorals.media.downloader import MediaDownloader
    dl = MediaDownloader()                  # sceneintel's shape
    assert dl._cf_cookie_store() is None
    gate = CloudflareGate(dl._cf_cookie_store())
    assert gate.inspect(CF_HTML) is True
    decision = gate.cookie_decision("https://x.test/article")
    assert decision["action"] == "ask"
    assert decision["cookie"] == ""
    # with no store the gate never invents a cookie: the message is
    # the kit's owner ask, exactly the pre-#81 detect-only behavior
    assert "cf_clearance" in decision["message"]


def test_browser_kind_entry_uses_default_store(home, vault_env,
                                               monkeypatch):
    store = _adapter(home)
    assert store_owner_cookie(store, "x.test", "KIND-COOKIE") is True
    monkeypatch.setattr(cloudflare_solver, "_default_cookie_store",
                        lambda: store)
    tab = FakeCloudflareTab(cookie_value="KIND-COOKIE")
    out = solve_cloudflare_kind(
        tab, {"kind": "cloudflare", "page_url": "https://x.test/article"})
    assert out["ok"], out
    assert out["state"] == "passed"


def test_browser_backend_uses_default_store(home, vault_env, monkeypatch):
    store = _adapter(home)
    assert store_owner_cookie(store, "x.test", "BACKEND-COOKIE") is True
    monkeypatch.setattr(cloudflare_solver, "_default_cookie_store",
                        lambda: store)
    tab = FakeCloudflareTab(cookie_value="BACKEND-COOKIE")
    backend = LocalCloudflareBackend()
    challenge = type("Ch", (), {
        "kind": "cloudflare", "metadata": {"session": tab}})()
    out = backend.solve(challenge)
    assert out.ok, out.detail


def test_register_seam_installs_vault_default(home, vault_env):
    """tools/browser.py::register (the #75 seam) binds the registry
    context's vault as the solver's default store."""
    from nomorals.tools import browser as browser_tools
    from nomorals.tools.registry import ToolRegistry
    context = type("Ctx", (), {"db": _db(home), "settings": None,
                               "extras": {}})()
    registry = ToolRegistry(context)
    browser_tools.register(registry)
    store = cloudflare_solver._default_cookie_store()
    assert isinstance(store, VaultCookieStore)
    assert store_owner_cookie(store, "x.test", "SEAM-COOKIE") is True
    assert store.get("cloudflare_cookies:x.test")["value"] == \
        "SEAM-COOKIE"


# ── (d) an injected store still wins ─────────────────────────────────────

def test_downloader_injected_store_wins(home, vault_env):
    from nomorals.media.downloader import MediaDownloader
    fake = FakeStore()
    context = type("Ctx", (), {"db": _db(home), "settings": None})()
    dl = MediaDownloader(context, cookie_store=fake)
    assert dl._cf_cookie_store() is fake


def test_solver_injected_store_wins_over_default(home, vault_env,
                                                 monkeypatch):
    default = _adapter(home)
    monkeypatch.setattr(cloudflare_solver, "_default_cookie_store",
                        lambda: default)
    fake = FakeStore()
    assert store_owner_cookie(fake, "x.test", "FAKE-COOKIE") is True
    tab = FakeCloudflareTab(cookie_value="FAKE-COOKIE")
    tab.fresh_value = ""
    out = _solver(fake).solve_tab(tab)
    assert out["ok"], out
    # the capture write-back went to the injected store, not the vault
    assert fake.get("cloudflare_cookies:x.test")["value"] == \
        "FAKE-COOKIE"
    assert default.get("cloudflare_cookies:x.test") is None


def test_solver_metadata_store_wins_over_default(home, vault_env,
                                                 monkeypatch):
    default = _adapter(home)
    monkeypatch.setattr(cloudflare_solver, "_default_cookie_store",
                        lambda: default)
    fake = FakeStore()
    assert store_owner_cookie(fake, "x.test", "META-COOKIE") is True
    solver = CloudflareSolver(wait_s=0.0, sleeper=lambda _s: None,
                              clock=lambda: 0.0)
    tab = FakeCloudflareTab(cookie_value="META-COOKIE")
    out = solver.solve_tab(tab, {"metadata": {"cookie_store": fake}})
    assert out["ok"], out
    assert default.get("cloudflare_cookies:x.test") is None


def test_estate_status_names_the_vault_wiring():
    from nomorals.browser.captcha_estate import estate_status
    status = estate_status()
    row = status["kinds"]["cloudflare"]
    assert "vault" in row["detail"].lower()
    assert "pass #81" in row["detail"].lower()

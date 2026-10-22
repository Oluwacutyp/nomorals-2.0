"""Pass 8 (directed) — bet desk activation.

The desk shipped inert: provisioning entries with ``flow=""`` (the
driver died at flow lookup), a registry the health loop never saw
(the desk held the UNBOUND singleton), and collectors that recorded
every failure as a plain fetch error.  These tests pin the rungs
this pass built:

* the odds-api / football-data page flows complete end-to-end
  through the REAL provisioning driver (login -> account page ->
  key read -> live validation -> vault -> env bridge), with the
  value in NO message/audit/log;
* betting sources get honest health from the ONE registry loop
  (ESPN probed through its real endpoint; a walled book reads
  bot-blocked, never dead; keyed APIs root-downgraded to
  unverifiable; 429 reads rate-limited);
* collector cycles land real rows (sports fixtures + results in
  the analyst's BetStore, deduped) and record registry outcomes in
  the registry's own vocabulary (blocked / rate-limited / outcome);
* the bustabit door is the tested file import (Pass 8 verified
  live: bustabit has NO keyless HTTP history endpoint — root and
  /api/* all answer the SPA shell; history rides its websocket
  app), in the bustabit export shape;
* status surfaces show vault evidence, not just env presence.

No network anywhere: tabs and HTTP are faked at the documented
seams, exactly like tests/test_provisioning.py.
"""

from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace
from unittest import mock

import pytest

from nomorals.accounts import provisioning
from nomorals.accounts.provisioning import (
    ProvisionDriver, ProvisionOutcome, bridge_vault_credentials,
    get_provision_service)
from nomorals.accounts.vault import CredentialVault
from nomorals.storage.db import Database

ODDS_KEY = "0123456789abcdef0123456789abcdef"
FD_KEY = "fedcba9876543210fedcba9876543210"
WRONG_KEY = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
OWNER_PW = "owner-login-password-123"


# ── fakes ─────────────────────────────────────────────────────────────

def _vault(tmp: str) -> CredentialVault:
    db = Database(os.path.join(tmp, "test.db"))
    db.migrate()
    return CredentialVault(db, master_passphrase="test-pass")


class DeskTab:
    """Scripted tab modelling the two desk services' pages.

    Login page (until submit) -> account page carrying the account's
    one key, as text (``key_mode="text"``) or in a readonly input
    (``key_mode="input"``, read through evaluate).  ``reveal=True``
    hides the key until a reveal control is clicked; ``captcha``
    challenges appear on the account page only.
    """

    def __init__(self, *, key: str = "", key_mode: str = "text",
                 captcha: list | None = None, reveal: bool = False,
                 label: str = "API Key") -> None:
        self.url = ""
        self.key = key
        self.key_mode = key_mode
        self.captcha = captcha or []
        self.reveal = reveal
        self.label = label
        self.revealed = False
        self.logged_in = False
        self.fills: dict[str, str] = {}
        self.clicks: list[str] = []
        self.closed = False

    def navigate(self, url: str):
        self.url = url
        return {"ok": True}

    def fill(self, name: str, value: str):
        self.fills[name] = value
        return {"ok": True}

    def submit(self, target: str = ""):
        self.logged_in = True
        return {"ok": True}

    def click(self, target: str):
        label = (target or "").strip().lower()
        self.clicks.append(target)
        if any(w in label for w in ("show", "reveal", "view")):
            self.revealed = True
            return {"ok": True}
        raise Exception(f"no clickable {target!r}")

    def _key_visible(self) -> bool:
        return bool(self.key) and (not self.reveal or self.revealed)

    def text(self, max_chars: int = 20000):
        if "/login" in self.url and not self.logged_in:
            return {"text": "Log in — email and password"}
        if "/account" in self.url or "/client/home" in self.url:
            body = "Your account"
            if self._key_visible():
                if self.key_mode == "text":
                    body += f" — Your {self.label}: {self.key}"
                else:
                    body += f" — Your {self.label} is shown below"
            return {"text": body}
        return {"text": "Welcome back — your account dashboard"}

    def evaluate(self, js: str):
        # the key-extraction script reads input/code/textarea values
        if "input, code, textarea" in js:
            if self.key_mode == "input" and self._key_visible():
                return self.key
            return None
        return False  # click-by-JS fallback: nothing matched

    def check_captcha(self):
        if "/account" in self.url or "/client/home" in self.url:
            return {"challenges": list(self.captcha)}
        return {"challenges": []}

    def cookies(self):
        return [{"name": "session", "value": "abc"}]

    def close(self):
        self.closed = True


def _driver(vault, *, tab: DeskTab, http_get, environ=None,
            captcha_solver=None):
    return ProvisionDriver(
        vault, open_tab=lambda service: tab, http_get=http_get,
        environ={} if environ is None else environ,
        captcha_solver=captcha_solver)


def _no_sleep():
    return mock.patch(
        "nomorals.accounts.browser_login.time.sleep", lambda s: None)


def _seed_login(vault, vault_service: str, login_url: str,
                username: str = "owner@example.com") -> None:
    vault.store(vault_service, username, OWNER_PW,
                credential_type="password",
                metadata={"login_url": login_url})


def _odds_http(url: str, headers: dict):
    assert "api.the-odds-api.com" in url
    return (200, "{}") if f"apiKey={ODDS_KEY}" in url else (401, "")


def _fd_http(url: str, headers: dict):
    assert "api.football-data.org" in url
    return (200, "{}") if headers.get("X-Auth-Token") == FD_KEY \
        else (403, "")


@pytest.fixture(autouse=True)
def _ensure():
    from nomorals.agents.betdesk.sources import ensure_provisioning
    ensure_provisioning()
    yield


# ── registration ──────────────────────────────────────────────────────

def test_flows_and_validators_registered():
    odds = get_provision_service("odds-api")
    fd = get_provision_service("football-data")
    assert odds is not None and fd is not None
    assert odds.flow == "odds_api_key_v1"
    assert fd.flow == "football_data_key_v1"
    assert provisioning._PAGE_FLOWS["odds_api_key_v1"] is not None
    assert provisioning._PAGE_FLOWS["football_data_key_v1"] is not None
    assert "odds_api_key" in provisioning._VALIDATORS
    assert "football_data_key" in provisioning._VALIDATORS


# ── full drives ───────────────────────────────────────────────────────

def test_odds_api_drive_end_to_end(tmp_path):
    vault = _vault(str(tmp_path))
    _seed_login(vault, "the-odds-api", "https://the-odds-api.com/login/")
    environ: dict[str, str] = {}
    driver = _driver(vault, tab=DeskTab(key=ODDS_KEY),
                     http_get=_odds_http, environ=environ)
    with _no_sleep():
        result = driver.provision("odds-api")
    assert result.outcome == ProvisionOutcome.PROVISIONED, result.message
    creds = vault.list_all(service="the-odds-api")
    key = next(c for c in creds if c.credential_type == "api_key")
    assert key.username == "api-key"
    assert vault.get("the-odds-api", "api-key").password == ODDS_KEY
    # the driver's live-validation evidence rode into the metadata
    assert "accepted" in (key.metadata.get("validation") or "")
    assert key.metadata.get("provisioned_at")
    # env bridge applied by the drive itself
    assert environ.get("ODDS_API_KEY") == ODDS_KEY
    # hygiene: the key and the password are in NO surface
    blob = result.message + " ".join(result.warnings) \
        + str(driver.audit_trail)
    assert ODDS_KEY not in blob and OWNER_PW not in blob


def test_football_data_drive_end_to_end(tmp_path):
    vault = _vault(str(tmp_path))
    _seed_login(vault, "football-data",
                "https://www.football-data.org/client/login")
    environ: dict[str, str] = {}
    tab = DeskTab(key=FD_KEY, key_mode="input", label="API Token")
    driver = _driver(vault, tab=tab, http_get=_fd_http, environ=environ)
    with _no_sleep():
        result = driver.provision("football-data")
    assert result.outcome == ProvisionOutcome.PROVISIONED, result.message
    assert vault.get("football-data", "api-key").password == FD_KEY
    assert environ.get("FOOTBALL_DATA_API_KEY") == FD_KEY
    blob = result.message + " ".join(result.warnings) \
        + str(driver.audit_trail)
    assert FD_KEY not in blob and OWNER_PW not in blob


def test_drive_reveal_click_finds_hidden_key(tmp_path):
    vault = _vault(str(tmp_path))
    _seed_login(vault, "the-odds-api", "https://the-odds-api.com/login/")
    tab = DeskTab(key=ODDS_KEY, reveal=True)
    driver = _driver(vault, tab=tab, http_get=_odds_http)
    with _no_sleep():
        result = driver.provision("odds-api")
    assert result.outcome == ProvisionOutcome.PROVISIONED, result.message
    assert any("show" in c.lower() for c in tab.clicks)


def test_drive_key_absent_is_page_changed_at_the_read_step(tmp_path):
    vault = _vault(str(tmp_path))
    _seed_login(vault, "the-odds-api", "https://the-odds-api.com/login/")
    driver = _driver(vault, tab=DeskTab(key=""), http_get=_odds_http)
    with _no_sleep():
        result = driver.provision("odds-api")
    assert result.outcome == ProvisionOutcome.PAGE_CHANGED, result.message
    assert result.step == "read_key_value"
    assert not any(c.credential_type == "api_key"
                   for c in vault.list_all(service="the-odds-api"))


def test_drive_captcha_blocks_without_solver(tmp_path):
    vault = _vault(str(tmp_path))
    _seed_login(vault, "the-odds-api", "https://the-odds-api.com/login/")
    tab = DeskTab(key=ODDS_KEY,
                  captcha=[{"kind": "hcaptcha", "sitekey": "x"}])
    driver = _driver(vault, tab=tab, http_get=_odds_http)
    with _no_sleep():
        result = driver.provision("odds-api")
    assert result.outcome == ProvisionOutcome.VERIFICATION_BLOCKED, \
        result.message
    assert not any(c.credential_type == "api_key"
                   for c in vault.list_all(service="the-odds-api"))


def test_drive_captcha_solver_proceeds(tmp_path):
    vault = _vault(str(tmp_path))
    _seed_login(vault, "the-odds-api", "https://the-odds-api.com/login/")
    seen: list = []

    def solver(challenge):
        seen.append(challenge)
        return {"ok": True, "detail": "solved"}

    tab = DeskTab(key=ODDS_KEY,
                  captcha=[{"kind": "hcaptcha", "sitekey": "x"}])
    driver = _driver(vault, tab=tab, http_get=_odds_http,
                     captcha_solver=solver)
    with _no_sleep():
        result = driver.provision("odds-api")
    assert result.outcome == ProvisionOutcome.PROVISIONED, result.message
    assert seen and seen[0].get("kind") == "hcaptcha"


def test_drive_validator_rejection_stores_nothing(tmp_path):
    vault = _vault(str(tmp_path))
    _seed_login(vault, "the-odds-api", "https://the-odds-api.com/login/")
    environ: dict[str, str] = {}
    driver = _driver(vault, tab=DeskTab(key=WRONG_KEY),
                     http_get=_odds_http, environ=environ)
    with _no_sleep():
        result = driver.provision("odds-api")
    assert result.outcome == ProvisionOutcome.KEY_REJECTED, result.message
    assert not any(c.credential_type == "api_key"
                   for c in vault.list_all(service="the-odds-api"))
    assert "ODDS_API_KEY" not in environ


def test_bridge_covers_desk_services_after_ensure(tmp_path):
    vault = _vault(str(tmp_path))
    vault.store("the-odds-api", "api-key", ODDS_KEY,
                credential_type="api_key")
    vault.store("football-data", "api-key", FD_KEY,
                credential_type="api_key")
    environ: dict[str, str] = {}
    applied = bridge_vault_credentials(vault, environ=environ)
    assert set(applied) == {"ODDS_API_KEY", "FOOTBALL_DATA_API_KEY"}
    assert environ["ODDS_API_KEY"] == ODDS_KEY
    assert environ["FOOTBALL_DATA_API_KEY"] == FD_KEY


# ── registry health: betting joins the ONE loop ───────────────────────

from nomorals.agents.betdesk import BettingDesk, render_collect, \
    render_status
from nomorals.agents.betdesk.collectors import (
    ProviderSpec, collect_provider, collect_sports,
    fetch_football_data_results, import_rounds_file)
from nomorals.agents.betdesk.crash import CrashStore
from nomorals.agents.betdesk.sources import (
    ESPN_PROBE_URL, betting_registry_for_context, ensure_betting_sources)
from nomorals.agents.sports_bet import BetStore, Fixture
from nomorals.media.source_registry import (
    Licence, SourceDescriptor, SourceRegistry, TrustTier,
    render_status_text)
from nomorals.storage.kv import KVStore


def _reg_with_seeds() -> SourceRegistry:
    reg = SourceRegistry()
    ensure_betting_sources(reg)
    return reg


def test_health_loop_classifies_betting_sources():
    reg = _reg_with_seeds()
    pages = {
        ESPN_PROBE_URL: (200, b'{"events": []}'),
        "https://stake.com/": (200, b"<html>stake</html>"),
        "https://sportybet.com/": (403, b"Attention Required! Cloudflare"),
        "https://msport.com/": (429, b""),
        "https://football.com/": (200, b"<html>football</html>"),
        "https://bustabit.com/": (200, b"<html>bustabit</html>"),
        "https://the-odds-api.com/": (500, b"server error"),
        "https://football-data.org/": (200, b"<html>football-data</html>"),
    }
    seen: list[str] = []

    def fetcher(url: str):
        seen.append(url)
        return pages[url]

    reg.verify_health(fetcher=fetcher)
    # ESPN was probed through its real scoreboard endpoint, not a root
    assert ESPN_PROBE_URL in seen
    assert "https://site.api.espn.com/" not in seen

    def health(domain: str) -> str:
        return reg.health_for(reg.for_domain(domain))

    assert health("site.api.espn.com") == "verified-live"
    assert health("stake.com") == "verified-live"
    assert health("bustabit.com") == "verified-live"
    # a walled book reads bot-blocked — never dead
    assert health("sportybet.com") == "bot-blocked"
    # a throttled book reads rate-limited
    assert health("msport.com") == "rate-limited"
    # a keyed API's root 500 proves nothing about the data route
    assert health("the-odds-api.com") == "unverifiable"
    # ...but a live root stays live
    assert health("football-data.org") == "verified-live"
    # the class rides the registry's one status surface
    st = reg.status()
    assert len(st["betting"]["sources"]) == 8
    assert "betting sources" in render_status_text(st)


def test_seed_update_path_replaces_stale_descriptor():
    reg = SourceRegistry()
    stale = SourceDescriptor(
        nation="GLOBAL", language="en", domain="site.api.espn.com",
        name="ESPN scoreboard", kinds=("betting",),
        licence=Licence.UNKNOWN, trust_tier=TrustTier.UNVERIFIED,
        auth_needed=False, notes="pre-pass-8 shape")
    reg.register(stale)
    out = ensure_betting_sources(reg)
    row = next(r for r in out if r["domain"] == "site.api.espn.com")
    assert row.get("updated") is True
    desc = reg.for_domain("site.api.espn.com")
    assert desc.search_url_template == ESPN_PROBE_URL
    out2 = ensure_betting_sources(reg)
    assert all(r.get("already_registered") for r in out2)


def test_keyed_api_seeds_carry_the_api_kind():
    reg = _reg_with_seeds()
    for domain in ("the-odds-api.com", "football-data.org"):
        assert "api" in reg.for_domain(domain).kinds


def test_bound_registry_persists_desk_seeds(tmp_path):
    """The seam that was broken: the desk's registry is the
    context-bound one, so its seeds persist into the state a fresh
    registry (the health loop's binding) loads."""
    db = Database(os.path.join(str(tmp_path), "ctx.db"))
    db.migrate()
    ctx = SimpleNamespace(db=db)
    desk = BettingDesk(ctx, desk_dir=str(tmp_path / "desk"))
    reg = desk.registry
    from nomorals.media.source_registry import DEFAULT_REGISTRY
    assert reg is DEFAULT_REGISTRY
    assert betting_registry_for_context(ctx) is DEFAULT_REGISTRY
    fresh = SourceRegistry(KVStore(db))
    assert fresh.for_domain("stake.com") is not None
    assert len(fresh.by_kind("betting")) == 8


# ── collector outcome vocabulary ──────────────────────────────────────

def test_collector_wall_records_blocked_not_plain_failure():
    reg = _reg_with_seeds()
    spec = ProviderSpec(
        id="stake", display="Stake", game="crash", domain="stake.com",
        history_url="https://stake.com/history")
    rep = collect_provider(
        spec, http_get=lambda u, h: (403, b"Attention Required!"),
        registry=reg)
    assert rep["ok"] is False
    desc = reg.for_domain("stake.com")
    st = reg.stats_for(desc)
    assert st["blocked"] == 1
    assert reg.health_for(desc) == "bot-blocked"


def test_collector_throttle_records_rate_limited():
    reg = _reg_with_seeds()
    spec = ProviderSpec(
        id="msport", display="MSport", game="aviator",
        domain="msport.com", history_url="https://msport.com/history")
    rep = collect_provider(spec, http_get=lambda u, h: (429, b""),
                           registry=reg)
    assert rep["ok"] is False
    desc = reg.for_domain("msport.com")
    assert reg.stats_for(desc)["rate_limited"] == 1
    assert reg.health_for(desc) == "rate-limited"


def test_collector_success_records_outcome():
    reg = _reg_with_seeds()
    spec = ProviderSpec(
        id="bustabit", display="Bustabit", game="crash",
        domain="bustabit.com",
        history_url="https://bustabit.com/history?limit={limit}",
        rounds_path="data.rounds",
        field_map=(("round_id", "id"), ("crash", "busted"),
                   ("ts", "ended_at"), ("game_hash", "hash")))
    payload = {"data": {"rounds": [
        {"id": 7, "busted": 2.34, "ended_at": 1700000000, "hash": "x"}]}}
    rep = collect_provider(
        spec, http_get=lambda u, h: (200, json.dumps(payload).encode()),
        registry=reg)
    assert rep["ok"] and len(rep["rounds"]) == 1
    desc = reg.for_domain("bustabit.com")
    assert reg.stats_for(desc)["ok"] == 1
    assert reg.health_for(desc) == "verified-live"


# ── bustabit: the import door, in the export shape ────────────────────

def test_bustabit_export_shape_imports(tmp_path):
    path = tmp_path / "bustabit-export.csv"
    path.write_text(
        "game_id,busted_at,hash,date\n"
        "7001,2.50,hh1,1700000000\n"
        "7002,1.00,hh2,1700000060\n"
        "7003,12.75,hh3,1700000120\n", encoding="utf-8")
    store = CrashStore(str(tmp_path / "desk"))
    res = import_rounds_file(store, str(path), "bustabit", "crash")
    assert res["ok"] and res["added"] == 3
    rounds = store.rounds("bustabit", "crash")
    assert [r.crash for r in rounds] == [2.50, 1.00, 12.75]
    assert rounds[0].round_id == "7001"


# ── sports collectors ─────────────────────────────────────────────────

class _FakeFetcher:
    name = "espn"

    def __init__(self, entries):
        self._entries = entries

    def fetch(self, league: str = "", limit: int = 20):
        return list(self._entries)


class _BoomFetcher:
    name = "espn"

    def fetch(self, league: str = "", limit: int = 20):
        raise RuntimeError("feed down")


def test_collect_sports_lands_dedupes_and_reports_skips(tmp_path):
    store = BetStore(bet_dir=str(tmp_path / "bet"))
    fx = [Fixture(home="Arsenal", away="Chelsea", league="eng.1",
                  date="2026-10-25T16:30:00Z"),
          Fixture(home="Liverpool", away="Everton", league="eng.1",
                  date="2026-10-25T12:30:00Z")]
    reg = _reg_with_seeds()
    rep = collect_sports(store, fetchers=[_FakeFetcher([(f, []) for f in fx])],
                         registry=reg, environ={})
    assert rep["landed"] == 2 and rep["history_total"] == 2
    feeds = {f["feed"]: f for f in rep["feeds"]}
    assert feeds["espn"]["ok"] and feeds["espn"]["landed"] == 2
    # keyed feeds without keys are skipped WITH the reason, not silent
    assert feeds["the-odds-api"]["skipped"]
    assert "ODDS_API_KEY" in feeds["the-odds-api"]["detail"]
    assert feeds["football-data"]["skipped"]
    # a second cycle lands nothing new
    rep2 = collect_sports(store,
                          fetchers=[_FakeFetcher([(f, []) for f in fx])],
                          registry=reg, environ={})
    assert rep2["landed"] == 0
    assert rep2["feeds"][0].get("skipped") == 2 or \
        [f for f in rep2["feeds"] if f["feed"] == "espn"][0]["skipped"] == 2
    # both cycles recorded outcomes on the registry
    desc = reg.for_domain("site.api.espn.com")
    assert reg.stats_for(desc)["ok"] == 2


def test_collect_sports_feed_failure_is_reported_and_recorded(tmp_path):
    store = BetStore(bet_dir=str(tmp_path / "bet"))
    reg = _reg_with_seeds()
    rep = collect_sports(store, fetchers=[_BoomFetcher()], registry=reg,
                         environ={})
    feeds = {f["feed"]: f for f in rep["feeds"]}
    assert feeds["espn"]["ok"] is False
    assert "feed down" in feeds["espn"]["detail"]
    desc = reg.for_domain("site.api.espn.com")
    assert reg.stats_for(desc)["fail"] == 1


def test_football_data_results_parse(tmp_path):
    payload = {"matches": [{
        "homeTeam": {"name": "Arsenal"}, "awayTeam": {"name": "Chelsea"},
        "utcDate": "2026-10-20T19:00:00Z",
        "score": {"fullTime": {"home": 2, "away": 1}}}]}

    def fake_get(url, headers=None):
        assert "status=FINISHED" in url
        assert headers["X-Auth-Token"] == FD_KEY
        return payload

    played = fetch_football_data_results(FD_KEY, http_get=fake_get)
    assert len(played) == 1
    assert played[0].played and played[0].home_goals == 2
    assert played[0].away_goals == 1


def test_collect_sports_result_updates_upcoming_fixture(tmp_path,
                                                        monkeypatch):
    store = BetStore(bet_dir=str(tmp_path / "bet"))
    store.record(Fixture(home="Arsenal", away="Chelsea", league="PL",
                         date="2026-10-20T19:00:00Z"), [])
    played = Fixture(home="Arsenal", away="Chelsea", league="PL",
                     date="2026-10-20T19:00:00Z",
                     home_goals=2, away_goals=1)
    monkeypatch.setattr(
        "nomorals.agents.betdesk.collectors.fetch_football_data_results",
        lambda key, league="PL": [played])
    rep = collect_sports(store, fetchers=[_FakeFetcher([])],
                         registry=None,
                         environ={"FOOTBALL_DATA_API_KEY": FD_KEY})
    assert rep["updated"] == 1
    got = store.fixtures()
    assert len(got) == 1 and got[0].played and got[0].home_goals == 2


# ── desk status / collect surfaces ────────────────────────────────────

def test_desk_status_shows_vault_evidence(tmp_path, monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.delenv("FOOTBALL_DATA_API_KEY", raising=False)
    monkeypatch.setenv("NOMORALS_BET_DIR", str(tmp_path / "bet"))
    vault = _vault(str(tmp_path))
    vault.store("the-odds-api", "api-key", ODDS_KEY,
                credential_type="api_key",
                metadata={"validation": "odds api key accepted",
                          "provisioned_at": 1700000000.0})
    desk = BettingDesk(None, desk_dir=str(tmp_path / "desk"), vault=vault)
    st = desk.status()
    rows = {p["service"]: p for p in st["provisioning"]}
    assert rows["odds-api"]["key_present"] is True
    assert rows["odds-api"]["source"] == "vault"
    assert "accepted" in rows["odds-api"]["validation"]
    assert rows["football-data"]["key_present"] is False
    assert st["sports_fixtures"] == 0
    text = render_status(st)
    assert "vaulted" in text
    assert "validated: odds api key accepted" in text
    assert "FOOTBALL_DATA_API_KEY: MISSING" in text


def test_desk_collect_cycle_and_render(tmp_path, monkeypatch):
    monkeypatch.setenv("NOMORALS_BET_DIR", str(tmp_path / "bet"))
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.delenv("FOOTBALL_DATA_API_KEY", raising=False)
    monkeypatch.setattr(
        "nomorals.agents.betdesk.desk.collect_all",
        lambda store, registry=None: {"providers": [], "added": 0,
                                      "at": 0})
    import nomorals.agents.sports_bet as sports_bet
    fx = Fixture(home="Arsenal", away="Chelsea", league="eng.1",
                 date="2026-10-25T16:30:00Z")
    monkeypatch.setattr(
        sports_bet.EspnFetcher, "fetch",
        lambda self, league="eng.1", limit=20: [(fx, [])])
    desk = BettingDesk(None, desk_dir=str(tmp_path / "desk"))
    rep = desk.collect()
    assert rep["sports"]["landed"] == 1
    text = render_collect(rep)
    assert "sports espn" in text
    assert "1 new fixtures landed" in text
    assert "ODDS_API_KEY not set" in text
    rep2 = desk.collect()
    assert rep2["sports"]["landed"] == 0

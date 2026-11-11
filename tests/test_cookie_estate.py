"""Navigator A6 pins: the unified cookie estate.

One store for the webnav profile path and the browser-service
path — no second copy, no sync job.  Every test opens its own
estate over a tmp database (never the home DB), with all four
legacy roots pointed at tmp dirs so the one-time migration is
hermetic too.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import stat
import time
from pathlib import Path
from typing import Any

import pytest

from nomorals.storage.cookie_estate import (
    COOKIE_ESTATE_DDL,
    SOURCE_IMPORTED,
    SOURCE_SERVICE,
    SOURCE_SOLVED,
    SOURCE_WEBNAV,
    CookieEstate,
    home_estate,
    migrate_legacy,
)
from nomorals.storage.db import open_database

#: A session value long enough to be a real credential (the
#: recipe guard only scans values of this length — shorter
#: values cannot be told apart from ordinary typed words).
SECRET = "sk-live-7f3a9c2e5b8d41f6a0c3e9b2d7f5a1c4"
OTHER_SECRET = "cf-clearance-9d2e7b4a6f1c8e3a5b7d9f0a2c4e6b8"


def _chrome_ts(unix: float) -> int:
    """Unix seconds -> Chromium microseconds since 1601."""
    return int((unix + 11644473600) * 1_000_000)


def _estate(tmp_path: Path, ensure: bool = True, **roots: Any
            ) -> CookieEstate:
    """A fresh estate over a tmp DB, all legacy roots tmp."""
    base = {
        "profile_dir": tmp_path / "profile",
        "service_data_dir": tmp_path / "service",
        "sessions_dir": tmp_path / "sessions",
        "solved_dir": tmp_path / "solved",
    }
    base.update(roots)
    for path in base.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    db, _, _ = open_database(str(tmp_path / "estate.sqlite3"))
    estate = CookieEstate(db, legacy_roots=base)
    if ensure:
        estate.ensure()
    else:
        # Tables only — the test drives the migration itself.
        estate.db.executescript(COOKIE_ESTATE_DDL)
    return estate


def _rec(name: str, value: str, domain: str, **kw: Any) -> dict[str, Any]:
    rec = {"name": name, "value": value, "domain": domain,
           "path": "/", "expires": -1.0, "secure": False,
           "httpOnly": False, "sameSite": ""}
    rec.update(kw)
    return rec


def _jar_triples(session: Any) -> set[tuple[str, str, str]]:
    return {(c.name, c.domain, c.value) for c in session.cookie_jar}


# ── estate semantics ──────────────────────────────────────────


def test_upsert_newest_wins_older_loses(tmp_path):
    estate = _estate(tmp_path)
    first = estate.upsert([_rec("SID", "old-value", ".example.com")],
                          source=SOURCE_WEBNAV)
    assert first["added"] == 1
    stale = estate.upsert(
        [_rec("SID", "older-value", ".example.com",
              updated_at=time.time() - 1000)],
        source=SOURCE_SERVICE)
    assert stale["kept_newer"] == 1
    assert stale["added"] == 0 and stale["replaced"] == 0
    assert estate.cookies_for("example.com")[0]["value"] == "old-value"
    fresh = estate.upsert(
        [_rec("SID", "new-value", ".example.com",
              updated_at=time.time() + 1000)],
        source=SOURCE_SERVICE)
    assert fresh["replaced"] == 1
    assert estate.cookies_for("example.com")[0]["value"] == "new-value"


def test_expired_dropped_at_write_and_read(tmp_path):
    estate = _estate(tmp_path)
    now = time.time()
    result = estate.upsert([
        _rec("dead", "x", ".example.com", expires=now - 10),
        _rec("live", "y", ".example.com", expires=now + 3600),
        _rec("session", "z", ".example.com", expires=-1.0),
    ], source=SOURCE_SERVICE)
    assert result["skipped"] == 1
    names = {r["name"] for r in estate.cookies_for("example.com")}
    assert names == {"live", "session"}


def test_domain_covering_read_vs_clear(tmp_path):
    estate = _estate(tmp_path)
    estate.upsert([
        _rec("wide", "w", ".google.com"),
        _rec("host", "h", "mail.google.com"),
        _rec("other", "o", ".x.com"),
    ], source=SOURCE_SERVICE)
    assert {r["name"] for r in estate.cookies_for("mail.google.com")} \
        == {"wide", "host"}
    assert {r["name"] for r in estate.cookies_for("google.com")} == {"wide"}
    removed, _ = estate.clear("google.com")
    assert removed == 2
    assert {r["name"] for r in estate.cookies_for()} == {"other"}


def test_imported_flag_is_sticky_and_clear_all_keeps_earned(tmp_path):
    estate = _estate(tmp_path)
    estate.upsert([_rec("SID", "owner-handed", ".example.com")],
                  source=SOURCE_IMPORTED)
    estate.upsert([_rec("SID", "refreshed-by-a-drive", ".example.com")],
                  source=SOURCE_WEBNAV)
    estate.upsert([_rec("earned", "e", ".example.com")],
                  source=SOURCE_WEBNAV)
    record = estate.cookies_for("example.com")
    by_name = {r["name"]: r for r in record}
    assert by_name["SID"]["imported"] is True
    assert by_name["earned"]["imported"] is False
    removed, _ = estate.clear("all", only_imported=True)
    assert removed == 1
    assert {r["name"] for r in estate.cookies_for()} == {"earned"}


def test_clear_source_scopes_by_writer(tmp_path):
    estate = _estate(tmp_path)
    estate.upsert([_rec("cf", "clearance", ".example.com")],
                  source=SOURCE_SOLVED)
    estate.upsert([_rec("SID", "login", ".example.com")],
                  source=SOURCE_WEBNAV)
    assert estate.clear_source("example.com", SOURCE_SOLVED) == 1
    assert {r["name"] for r in estate.cookies_for()} == {"SID"}


def test_export_netscape_is_0600_and_skips_bad_values(tmp_path):
    estate = _estate(tmp_path)
    estate.upsert([
        _rec("SID", "good-value", ".example.com", expires=1999999999),
        _rec("bad", "has\ttab", ".example.com"),
        _rec("bad2", "has\nnewline", ".example.com"),
        _rec("sess", "s", ".example.com", expires=-1.0),
    ], source=SOURCE_SERVICE)
    dest = tmp_path / "cookies.txt"
    result = estate.export_netscape(dest)
    assert result["count"] == 2
    assert result["skipped"] == 2
    assert stat.S_IMODE(dest.stat().st_mode) == 0o600
    text = dest.read_text(encoding="utf-8")
    assert "good-value" in text and "\tsess\ts" in text
    assert "has\ttab" not in text and "has\nnewline" not in text


def test_status_domains_has_domain(tmp_path):
    estate = _estate(tmp_path)
    assert estate.domains() == []
    assert estate.has_domain("example.com") is False
    estate.upsert([_rec("SID", "v", ".example.com",
                        expires=time.time() + 600)],
                  source=SOURCE_SERVICE)
    assert estate.has_domain("www.example.com") is True
    assert estate.has_domain("other.com") is False
    assert estate.domains() == [".example.com"]
    rows = estate.status()
    assert len(rows) == 1 and rows[0]["count"] == 1
    assert rows[0]["nearest_expiry"] is not None


def test_cookie_values_never_reach_the_log(tmp_path, caplog):
    estate = _estate(tmp_path)
    with caplog.at_level(logging.INFO,
                         logger="nomorals.storage.cookie_estate"):
        estate.upsert([_rec("SID", SECRET, ".example.com")],
                      source=SOURCE_WEBNAV)
        estate.upsert([_rec("SID", SECRET + "-new", ".example.com")],
                      source=SOURCE_SERVICE)
    blob = "\n".join(r.getMessage() for r in caplog.records)
    assert SECRET not in blob
    assert SECRET + "-new" not in blob


# ── migration: two estates import once, deduped ───────────────


def _write_profile_db(profile: Path, rows: list[dict[str, Any]]) -> None:
    conn = sqlite3.connect(str(profile / "Cookies"))
    conn.execute(
        "CREATE TABLE cookies (host_key TEXT, name TEXT, value TEXT, "
        "path TEXT, expires_utc INTEGER, is_secure INTEGER, "
        "is_httponly INTEGER, samesite INTEGER, last_access_utc INTEGER)")
    for row in rows:
        conn.execute(
            "INSERT INTO cookies VALUES (?,?,?,?,?,?,?,?,?)",
            (row["domain"], row["name"], row["value"],
             row.get("path", "/"), _chrome_ts(row.get("expires", -1))
             if row.get("expires", -1) > 0 else 0,
             int(bool(row.get("secure"))), int(bool(row.get("httpOnly"))),
             -1, _chrome_ts(row.get("updated_at", time.time()))))
    conn.commit()
    conn.close()


def test_migration_imports_both_estates_once_deduped(tmp_path):
    now = time.time()
    sessions = tmp_path / "sessions"
    service = tmp_path / "service"
    profile = tmp_path / "profile"
    solved = tmp_path / "solved"
    for d in (sessions, service, profile, solved):
        d.mkdir(parents=True, exist_ok=True)
    # Service jar: SID (older) + a service-only cookie.
    (sessions / "default.json").write_text(json.dumps([
        {"name": "SID", "value": "jar-value", "domain": ".example.com",
         "path": "/", "expires": now + 3600, "secure": False,
         "domain_specified": True, "domain_initial_dot": True,
         "path_specified": True, "discard": False, "rest": {}},
        {"name": "svc-only", "value": "s", "domain": ".example.com",
         "path": "/", "expires": now + 3600, "secure": False,
         "domain_specified": True, "domain_initial_dot": True,
         "path_specified": True, "discard": False, "rest": {}},
    ]), encoding="utf-8")
    # Webnav sidecar: the SAME SID, newer (owner-handed wins).
    (profile / "imported_cookies.json").write_text(json.dumps([
        {"name": "SID", "value": "sidecar-value", "domain": ".example.com",
         "path": "/", "expires": now + 7200, "updated_at": now + 100},
    ]), encoding="utf-8")
    # Chromium profile DB: a webnav-earned login.
    _write_profile_db(profile, [
        {"name": "nav-login", "value": "nav-value", "domain": ".example.com",
         "expires": now + 3600, "updated_at": now},
    ])
    # Solved clearance file.
    (solved / "example.com.json").write_text(json.dumps({
        "site": "example.com", "saved_at": now,
        "cookies": [{"name": "cf_clearance", "value": "clear",
                     "domain": ".example.com", "path": "/",
                     "expires": now + 3600}],
    }), encoding="utf-8")
    estate = _estate(tmp_path, ensure=False)
    result = estate._migrate_pending()
    by_name = {r["name"]: r for r in estate.cookies_for("example.com")}
    assert set(by_name) == {"SID", "svc-only", "nav-login", "cf_clearance"}
    assert by_name["SID"]["value"] == "sidecar-value"
    assert by_name["SID"]["imported"] is True
    assert result["sources"]["service"]["files"] >= 1
    assert result["sources"]["imported"]["files"] == 1
    assert result["sources"]["solved"]["files"] == 1
    # A second ensure imports nothing again.
    before = {r["name"]: r["value"]
              for r in estate.cookies_for("example.com")}
    again = estate._migrate_pending()
    assert all(v.get("already") for v in again["sources"].values())
    assert {r["name"]: r["value"]
            for r in estate.cookies_for("example.com")} == before


def test_migration_force_reruns(tmp_path):
    estate = _estate(tmp_path)
    first = migrate_legacy(estate, force=True, **{
        k: v for k, v in estate._roots.items()})
    assert all("already" not in v for v in first["sources"].values())


# ── cross-path visibility: one login, every surface ────────


class _FakeContext:
    """A Playwright context stand-in: cookies() + add_cookies()."""

    def __init__(self, cookies: list[dict[str, Any]] | None = None):
        self._cookies = list(cookies or [])
        self.added: list[dict[str, Any]] = []
        self.closed = False

    def cookies(self) -> list[dict[str, Any]]:
        return [dict(c) for c in self._cookies]

    def add_cookies(self, cookies: list[dict[str, Any]]) -> None:
        self.added.extend(dict(c) for c in cookies)

    def storage_state(self, path: str | None = None) -> dict[str, Any]:
        return {"cookies": [dict(c) for c in self._cookies],
                "origins": [{"origin": "https://example.com",
                             "localStorage": [{"name": "k", "value": "v"}]}]}

    def close(self) -> None:
        self.closed = True


def test_webnav_login_is_visible_to_the_browser_service(tmp_path):
    from nomorals.social.chat import webnav
    from nomorals.tools.browser import BrowserSession

    estate = _estate(tmp_path)
    profile = tmp_path / "profile"
    profile.mkdir(parents=True, exist_ok=True)
    session = webnav.NavSession(profile_dir=profile, cookie_estate=estate)
    session._context = _FakeContext([_rec(
        "SID", "webnav-login", ".example.com",
        expires=time.time() + 3600)])
    session._harvest_cookies()
    assert {r["name"] for r in estate.cookies_for("www.example.com")} \
        == {"SID"}
    # The browser-service path sees it — and writes no jar file.
    sessions_dir = tmp_path / "sessions"
    browser = BrowserSession(name="svc", session_dir=str(sessions_dir),
                             cookie_estate=estate)
    assert ("SID", ".example.com", "webnav-login") in _jar_triples(browser)
    assert list(sessions_dir.glob("*.json")) == []
    # A later webnav start applies it to the live context.
    second = webnav.NavSession(profile_dir=profile, cookie_estate=estate)
    second._context = _FakeContext()
    second._apply_imported_cookies()
    assert [c["name"] for c in second._context.added] == ["SID"]
    assert second._context.added[0]["value"] == "webnav-login"


def test_browser_service_save_is_visible_to_webnav_and_writes_no_jar(
        tmp_path):
    import http.cookiejar

    from nomorals.social.chat import webnav
    from nomorals.tools.browser import BrowserSession

    estate = _estate(tmp_path)
    sessions_dir = tmp_path / "sessions"
    browser = BrowserSession(name="svc", session_dir=str(sessions_dir),
                             cookie_estate=estate)
    cookie = http.cookiejar.Cookie(
        version=0, name="SID", value="service-login", port=None,
        port_specified=False, domain=".example.com",
        domain_specified=True, domain_initial_dot=True,
        path="/", path_specified=True, secure=False,
        expires=time.time() + 3600, discard=False,
        comment=None, comment_url=None, rest={"HttpOnly": None},
        rfc2109=False)
    browser.cookie_jar.set_cookie(cookie)
    browser._save_cookies()
    assert list(sessions_dir.glob("*.json")) == []
    assert {r["name"] for r in estate.cookies_for("example.com")} == {"SID"}
    profile = tmp_path / "profile"
    profile.mkdir(parents=True, exist_ok=True)
    session = webnav.NavSession(profile_dir=profile, cookie_estate=estate)
    session._context = _FakeContext()
    session._apply_imported_cookies()
    assert [c["value"] for c in session._context.added] == ["service-login"]


def test_rendered_tab_persist_harvests_and_strips_the_file(tmp_path):
    from nomorals.browser.service import RenderedTab

    estate = _estate(tmp_path)
    state_path = tmp_path / "cookies" / "sess" / "playwright-storage.json"
    tab = RenderedTab(
        tab_id="tab1", session_name="sess",
        storage_state_path=str(state_path),
        playwright=None, cookie_estate=estate)
    tab._context = _FakeContext([_rec(
        "SID", "rendered-login", ".example.com",
        expires=time.time() + 3600)])
    out = tab.persist()
    assert out["harvested"] is True
    assert out["cookies"] == 1
    assert {r["name"] for r in estate.cookies_for("example.com")} == {"SID"}
    stored = json.loads(state_path.read_text(encoding="utf-8"))
    assert stored["cookies"] == []
    assert stored["origins"][0]["origin"] == "https://example.com"
    # The next tab seeds from the estate, not the file.
    tab2 = RenderedTab(
        tab_id="tab2", session_name="sess",
        storage_state_path=str(state_path),
        playwright=None, cookie_estate=estate)
    tab2._context = _FakeContext()
    tab2._seed_estate_cookies()
    assert [c["value"] for c in tab2._context.added] == ["rendered-login"]
    # The loader strips a legacy file's cookies at open.
    assert tab2._origins_only_state() == {
        "cookies": [], "origins": stored["origins"]}


def test_solved_session_rides_the_estate_not_the_file(tmp_path):
    from nomorals.browser.solved_sessions import SolvedSessionStore

    estate = _estate(tmp_path)
    store = SolvedSessionStore(tmp_path, cookie_estate=estate)
    record = store.save(
        "https://example.com/wall", "https://example.com/wall",
        [_rec("cf_clearance", "clear", ".example.com",
              expires=time.time() + 3600)],
        origins=[{"origin": "https://example.com", "localStorage": []}],
        cleared_via="test")
    assert len(record["cookies"]) == 1
    on_disk = json.loads(
        (tmp_path / "solved_sessions" / "example.com.json")
        .read_text(encoding="utf-8"))
    assert on_disk["cookies"] == []
    assert on_disk["origins"][0]["origin"] == "https://example.com"
    assert {r["name"] for r in estate.cookies_for("example.com")} \
        == {"cf_clearance"}
    loaded = store.load("https://example.com/wall")
    assert loaded is not None
    assert [c["name"] for c in loaded["cookies"]] == ["cf_clearance"]
    assert store.has("https://example.com/wall") is True
    assert store.clear("https://example.com/wall") is True
    assert estate.cookies_for("example.com") == []
    assert store.load("https://example.com/wall") is None


def test_media_export_comes_from_the_estate(tmp_path):
    from nomorals.media.cookies import export_browser_cookies

    estate = _estate(tmp_path)
    estate.upsert([_rec("SID", "yt-login", ".youtube.com",
                        expires=time.time() + 3600)],
                  source=SOURCE_WEBNAV)
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    path = export_browser_cookies(str(sessions_dir), cookie_estate=estate)
    assert path is not None
    text = Path(path).read_text(encoding="utf-8")
    assert "yt-login" in text
    assert ".youtube.com" in text


def test_media_export_falls_back_to_jars_when_no_estate(tmp_path,
                                                        monkeypatch):
    from nomorals.media.cookies import export_browser_cookies

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / "default.json").write_text(json.dumps([
        {"name": "SID", "value": "jar-value", "domain": ".youtube.com",
         "path": "/", "secure": True, "expires": 1999999999,
         "domain_initial_dot": True},
    ]), encoding="utf-8")
    # The estate is unreachable (patched out AND the home DB
    # cannot open), so the legacy jar merge must still produce
    # the export.
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    monkeypatch.setattr(
        "nomorals.media.cookies._estate", lambda *a, **k: None)
    path = export_browser_cookies(str(sessions_dir))
    assert path is not None
    assert "jar-value" in Path(path).read_text(encoding="utf-8")


def test_imported_bundle_never_writes_the_sidecar(tmp_path):
    from nomorals.social.chat.cookie_bundle import ImportedCookiesStore

    estate = _estate(tmp_path)
    profile = tmp_path / "profile"
    profile.mkdir(parents=True, exist_ok=True)
    store = ImportedCookiesStore(profile, estate=estate)
    added, _ = store.merge([_rec("SID", "owner-bundle", ".example.com")])
    assert added == 1
    assert not (profile / "imported_cookies.json").exists()
    assert [r["name"] for r in store.load()] == ["SID"]
    # An earned login survives a bundle save (only imported clears).
    estate.upsert([_rec("earned", "e", ".example.com")],
                  source=SOURCE_WEBNAV)
    store.save([_rec("SID", "owner-bundle-2", ".example.com")])
    assert {r["name"] for r in store.load()} == {"SID", "earned"}
    removed, _ = store.clear("all")
    assert {r["name"] for r in store.load()} == {"earned"}


def test_home_estate_opens_over_an_injected_db(tmp_path):
    db, _, _ = open_database(str(tmp_path / "home.sqlite3"))
    estate = home_estate(db=db)
    estate.upsert([_rec("SID", "v", ".example.com")],
                  source=SOURCE_SERVICE)
    assert estate.has_domain("example.com") is True


# ── recipes: no secrets in, session declared, replay sees it ──


def _trace(url: str = "https://example.com/home") -> list[dict[str, Any]]:
    return [
        {"before_url": url, "after_url": url,
         "action": {"kind": "click", "target": "search-btn",
                    "label": "Search", "payload": {}},
         "result": {"ok": True}, "verified": True},
    ]


def test_compile_refuses_a_recipe_carrying_a_live_cookie_value(tmp_path):
    from nomorals.browser.recipes import compile_recipe

    estate = _estate(tmp_path)
    estate.upsert([_rec("SID", SECRET, ".example.com")],
                  source=SOURCE_WEBNAV)
    trace = [{
        "before_url": "https://example.com/callback?session=" + SECRET,
        "after_url": "https://example.com/home",
        "action": {"kind": "goto",
                   "target": "https://example.com/callback?session="
                   + SECRET, "label": "", "payload": {}},
        "result": {"ok": True}, "verified": True}]
    with pytest.raises(ValueError) as excinfo:
        compile_recipe(name="example:leak", goal="open the callback",
                       trace=trace, start_url="https://example.com/",
                       cookie_estate=estate)
    assert SECRET not in str(excinfo.value)
    assert "session value" in str(excinfo.value)


def test_compile_without_an_estate_skips_the_secret_scan(tmp_path):
    from nomorals.browser.recipes import compile_recipe

    recipe = compile_recipe(
        name="example:plain", goal="open the callback",
        trace=[{
            "before_url": "https://example.com/callback?session=" + SECRET,
            "after_url": "https://example.com/home",
            "action": {"kind": "goto",
                       "target": "https://example.com/callback?session="
                       + SECRET, "label": "", "payload": {}},
            "result": {"ok": True}, "verified": True}],
        start_url="https://example.com/")
    assert recipe.steps[0].kind == "goto"


def test_needs_session_round_trips_and_recipe_json_holds_no_secret(
        tmp_path):
    import json as _json

    from nomorals.browser.recipes import Recipe, compile_recipe

    estate = _estate(tmp_path)
    estate.upsert([_rec("SID", SECRET, ".example.com")],
                  source=SOURCE_WEBNAV)
    recipe = compile_recipe(
        name="example:search", goal="search the site",
        trace=_trace(), start_url="https://example.com/home",
        needs_session=True, cookie_estate=estate)
    assert recipe.needs_session is True
    assert recipe.site == "example.com"
    blob = _json.dumps(recipe.to_dict())
    assert SECRET not in blob
    restored = Recipe.from_dict(recipe.to_dict())
    assert restored.needs_session is True
    assert restored.site == "example.com"


class _FakeDriver:
    """A replay driver over canned pages."""

    def __init__(self, page: Any):
        self.page = page
        self.fetch_calls = 0

    def fetch(self, url: str) -> Any:
        self.fetch_calls += 1
        return self.page

    def act(self, action: Any, page: Any) -> Any:
        return self.page


def test_replay_needs_session_fails_before_fetch_without_one(tmp_path):
    from nomorals.browser.agent import Page
    from nomorals.browser.recipe_replay import RecipeReplayError, replay_recipe
    from nomorals.browser.recipes import compile_recipe

    estate = _estate(tmp_path)
    recipe = compile_recipe(
        name="example:search", goal="search the site",
        trace=_trace(), start_url="https://example.com/home",
        needs_session=True, cookie_estate=estate)
    page = Page(url="https://example.com/home", title="Home",
                elements=[{"id": "search-btn", "label": "Search"}])
    driver = _FakeDriver(page)
    with pytest.raises(RecipeReplayError) as excinfo:
        replay_recipe(recipe, driver=driver,
                      approve=lambda d, o: True,
                      cookie_estate=estate)
    assert driver.fetch_calls == 0
    assert "logged-in session" in str(excinfo.value)
    assert excinfo.value.stale is False


def test_replay_sees_a_webnav_login_through_the_estate(tmp_path):
    from nomorals.browser.agent import Page
    from nomorals.browser.recipe_replay import replay_recipe
    from nomorals.browser.recipes import compile_recipe
    from nomorals.social.chat import webnav

    estate = _estate(tmp_path)
    profile = tmp_path / "profile"
    profile.mkdir(parents=True, exist_ok=True)
    session = webnav.NavSession(profile_dir=profile, cookie_estate=estate)
    session._context = _FakeContext([_rec(
        "SID", "webnav-login", ".example.com",
        expires=time.time() + 3600)])
    session._harvest_cookies()
    recipe = compile_recipe(
        name="example:search", goal="search the site",
        trace=_trace(), start_url="https://example.com/home",
        needs_session=True, cookie_estate=estate)
    page = Page(url="https://example.com/home", title="Home",
                elements=[{"id": "search-btn", "label": "Search"}])
    driver = _FakeDriver(page)
    out = replay_recipe(recipe, driver=driver,
                        approve=lambda d, o: True,
                        cookie_estate=estate)
    assert driver.fetch_calls == 1
    assert out["replayed"] is True
    assert out["steps"] == 2

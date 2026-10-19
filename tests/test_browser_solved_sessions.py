"""Failing-first tests: save cookies/storage when a challenge clears.

When a wall clears — Devon self-solved it, or the owner solved it via
takeover — the tab's cookies + storage must persist per-site, so the
next visit reuses the solved session instead of facing the same wall.
Still-blocked challenges save NOTHING. Different sites never leak
cookies to each other.
"""
from __future__ import annotations

import json
import threading
import time

from nomorals.browser.challenge import ChallengeSolver
from nomorals.browser.solved_sessions import (
    SolvedSessionStore,
    restore_solved_session,
    save_solved_session,
)
from nomorals.browser.takeover import TakeoverConfig, TakeoverManager

CF_HTML = (
    "<html><head><title>Just a moment...</title></head><body>"
    "Verify you are human — checking your browser</body></html>"
)
CLEAN_HTML = "<html><head><title>Home</title></head><body>hello</body></html>"

SITE = "https://example.test/login"
OTHER_SITE = "https://other.test/login"
CLEARANCE = {
    "name": "cf_clearance",
    "value": "SECRET-CLEARANCE-VALUE",
    "domain": "example.test",
    "path": "/",
    "secure": True,
    "expires": 9999999999,
    "http_only": True,
}


class FakeTab:
    """A tab with canned HTML + an in-memory cookie jar."""

    def __init__(self, html: str = "", url: str = SITE) -> None:
        self._html = html
        self._url = url
        self._title = "Example"
        self.jar: list[dict] = []
        self.reloads = 0

    @property
    def url(self) -> str:
        return self._url

    @property
    def title(self) -> str:
        import re

        match = re.search(r"<title[^>]*>(.*?)</title>", self._html,
                          re.DOTALL | re.IGNORECASE)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()
        return self._title

    def html(self) -> str:
        return self._html

    def set_html(self, html: str) -> None:
        self._html = html

    def reload(self) -> None:
        self.reloads += 1

    def screenshot_png(self) -> bytes:
        return b"\x89PNG\r\n\x1a\nfake-png"

    def click_norm(self, x: float, y: float) -> None:
        pass

    def type_text(self, text: str) -> None:
        pass

    def press_key(self, key: str) -> None:
        pass

    def scroll_norm(self, dx: float, dy: float) -> None:
        pass

    # -- cookie jar ------------------------------------------------------
    def cookies(self) -> list[dict]:
        return [dict(c) for c in self.jar]

    def add_cookies(self, cookies) -> dict:
        for cookie in cookies or []:
            self.jar = [c for c in self.jar
                        if not (c.get("name") == cookie.get("name")
                                and c.get("domain") == cookie.get("domain"))]
            self.jar.append(dict(cookie))
        # A clearance cookie clears this fake wall, like the real one.
        if any(c.get("name") == "cf_clearance" for c in self.jar):
            self._html = CLEAN_HTML
        return {"ok": True, "added": len(cookies or [])}

    def storage_state(self) -> dict:
        return {"cookies": [dict(c) for c in self.jar],
                "origins": [{"origin": "https://example.test",
                             "localStorage": [{"name": "k", "value": "v"}]}]}


# -- store -------------------------------------------------------------------

def test_store_save_load_roundtrip(tmp_path):
    store = SolvedSessionStore(tmp_path)
    store.save(SITE, SITE, [CLEARANCE], origins=[])
    loaded = store.load(SITE)
    assert loaded is not None
    assert loaded["cookies"] == [CLEARANCE]
    assert loaded["site"] == "example.test"
    files = list((tmp_path / "solved_sessions").glob("*.json"))
    assert len(files) == 1
    assert "SECRET-CLEARANCE-VALUE" in files[0].read_text(encoding="utf-8")


def test_store_drops_expired_cookies_on_load(tmp_path):
    store = SolvedSessionStore(tmp_path)
    stale = dict(CLEARANCE, expires=1)  # long expired
    store.save(SITE, SITE, [stale, CLEARANCE])
    loaded = store.load(SITE)
    assert loaded is not None
    assert [c["name"] for c in loaded["cookies"]] == ["cf_clearance"]
    store.save(SITE, SITE, [stale])
    assert store.load(SITE) is None


def test_store_never_leaks_across_sites(tmp_path):
    store = SolvedSessionStore(tmp_path)
    store.save(SITE, SITE, [CLEARANCE])
    assert store.load(OTHER_SITE) is None
    assert store.has(SITE) is True
    assert store.has(OTHER_SITE) is False


# -- save on solve -----------------------------------------------------------

def test_solver_saves_cookies_on_self_solve(tmp_path):
    store = SolvedSessionStore(tmp_path)
    tab = FakeTab(CF_HTML)
    tab.jar = [dict(CLEARANCE)]

    def reload_clears() -> None:
        tab.reloads += 1
        tab.set_html(CLEAN_HTML)

    tab.reload = reload_clears  # type: ignore[method-assign]
    solver = ChallengeSolver(wait_s=0, session_store=store)
    outcome = solver.solve(tab=tab, resume_state={"step": 1})
    assert outcome.cleared is True
    assert outcome.via == "retry"
    assert outcome.session_saved == 1
    loaded = store.load(SITE)
    assert loaded is not None
    assert loaded["cookies"][0]["name"] == "cf_clearance"
    assert loaded["cleared_via"] == "retry"


def test_solver_saves_nothing_when_still_blocked(tmp_path):
    store = SolvedSessionStore(tmp_path)
    tab = FakeTab(CF_HTML)
    tab.jar = [dict(CLEARANCE)]
    solver = ChallengeSolver(wait_s=0, session_store=store)
    outcome = solver.solve(tab=tab, resume_state={"step": 1})
    assert outcome.cleared is False
    assert outcome.via == "takeover_required"
    assert outcome.session_saved == 0
    assert store.load(SITE) is None


def test_solver_restores_saved_session_before_ladder(tmp_path):
    store = SolvedSessionStore(tmp_path)
    store.save(SITE, SITE, [CLEARANCE])
    tab = FakeTab(CF_HTML)  # fresh tab, empty jar, wall up
    solver = ChallengeSolver(wait_s=0, session_store=store)
    outcome = solver.solve(tab=tab, resume_state={"step": 1})
    assert outcome.cleared is True
    assert outcome.via == "saved_session"
    assert [a.strategy for a in outcome.attempts] == ["saved_session"]
    # The clearance cookie reached THIS tab; the ladder never ran.
    assert any(c["name"] == "cf_clearance" for c in tab.jar)
    assert tab.reloads == 0


def test_solver_does_not_restore_other_site_cookies(tmp_path):
    store = SolvedSessionStore(tmp_path)
    store.save(SITE, SITE, [CLEARANCE])
    tab = FakeTab(CF_HTML, url=OTHER_SITE)
    solver = ChallengeSolver(wait_s=0, session_store=store)
    outcome = solver.solve(tab=tab, resume_state={"step": 1})
    assert outcome.cleared is False
    assert outcome.via == "takeover_required"
    assert tab.jar == []


def test_save_helper_prefers_storage_state_with_origins(tmp_path):
    store = SolvedSessionStore(tmp_path)
    tab = FakeTab(CLEAN_HTML)
    tab.jar = [dict(CLEARANCE)]
    saved = save_solved_session(tab, SITE, store, cleared_via="retry")
    assert saved == 1
    loaded = store.load(SITE)
    assert loaded is not None
    assert loaded["origins"][0]["origin"] == "https://example.test"


def test_restore_helper_returns_zero_without_adder(tmp_path):
    store = SolvedSessionStore(tmp_path)
    store.save(SITE, SITE, [CLEARANCE])

    class BareTab:
        url = SITE

        def html(self) -> str:
            return CF_HTML

    assert restore_solved_session(BareTab(), SITE, store) == 0


# -- takeover path: save on clear, nothing on still-blocked ------------------

def test_takeover_resolution_saves_and_audits_without_values(tmp_path):
    store = SolvedSessionStore(tmp_path)
    mgr = TakeoverManager(TakeoverConfig(wait_timeout_s=30),
                          data_dir=tmp_path, session_store=store)
    tab = FakeTab(CF_HTML)
    tab.jar = [dict(CLEARANCE)]
    solver = ChallengeSolver(wait_s=0, takeover_manager=mgr,
                             session_store=store)
    handler = solver.wall_handler_for(tab, owner_chat="owner-1")

    # Owner solves it shortly after the wait starts (auto-detect path).
    threading.Timer(0.2, lambda: tab.set_html(CLEAN_HTML)).start()

    class _Obs:
        class page:
            html = CF_HTML
            url = SITE
            title = "Just a moment..."

    assert handler(_Obs(), "cloudflare-challenge") == "continue"
    loaded = store.load(SITE)
    assert loaded is not None
    assert loaded["cookies"][0]["name"] == "cf_clearance"
    assert loaded["cleared_via"].startswith("takeover:")
    audit = (tmp_path / "takeover-audit.jsonl").read_text(encoding="utf-8")
    assert "session_saved" in audit
    assert "SECRET-CLEARANCE-VALUE" not in audit
    assert "example.test" in audit


def test_takeover_still_blocked_saves_nothing(tmp_path):
    store = SolvedSessionStore(tmp_path)
    mgr = TakeoverManager(TakeoverConfig(wait_timeout_s=1),
                          data_dir=tmp_path, session_store=store)
    tab = FakeTab(CF_HTML)
    tab.jar = [dict(CLEARANCE)]
    solver = ChallengeSolver(wait_s=0, takeover_manager=mgr,
                             session_store=store)
    handler = solver.wall_handler_for(tab, owner_chat="owner-1")

    class _Obs:
        class page:
            html = CF_HTML
            url = SITE
            title = "Just a moment..."

    started = time.time()
    assert handler(_Obs(), "cloudflare-challenge") == "stop"
    assert time.time() - started < 30
    assert store.load(SITE) is None
    audit = (tmp_path / "takeover-audit.jsonl").read_text(encoding="utf-8")
    assert "session_saved" not in audit


def test_browser_service_solved_session_store_factory(tmp_path):
    from nomorals.browser import BrowserService

    svc = BrowserService(data_dir=tmp_path / "svc")
    store = svc.solved_session_store()
    assert isinstance(store, SolvedSessionStore)
    assert store._dir == tmp_path / "svc" / "takeover" / "solved_sessions"
    mgr = svc.takeover_manager(session_store=store)
    assert mgr._session_store is store


def test_save_is_best_effort_never_breaks_resume(tmp_path):
    store = SolvedSessionStore(tmp_path)

    class BrokenTab:
        url = SITE

        def html(self) -> str:
            return CLEAN_HTML

        def storage_state(self) -> dict:
            raise RuntimeError("boom")

        def cookies(self) -> list:
            raise RuntimeError("boom")

    assert save_solved_session(BrokenTab(), SITE, store) == 0
    assert store.load(SITE) is None

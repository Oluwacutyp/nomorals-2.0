"""Failing-first tests for the human-in-the-loop browser takeover.

Covers: challenge detection, self-solve-before-escalation, escalation
message listing attempts, owner-only/expiring/single-use takeover links,
hand-back (explicit + auto-detect) resuming at the saved step,
still-blocked honesty, and challenge-memory learning.
"""
from __future__ import annotations

import time

import pytest

from nomorals.browser.challenge import (
    ChallengeSolver,
    detect_challenge_signal,
    format_escalation_message,
)
from nomorals.browser.challenge_memory import ChallengeMemory
from nomorals.browser.takeover import (
    TakeoverConfig,
    TakeoverManager,
    TakeoverTabAdapter,
)


class FakeTab(TakeoverTabAdapter):
    """A tab with canned HTML, recording every input."""

    def __init__(self, html: str = "", url: str = "https://example.test/") -> None:
        self._html = html
        self._url = url
        self._title = "Example"
        self.clicks: list[tuple[float, float]] = []
        self.typed: list[str] = []
        self.keys: list[str] = []
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
        self.clicks.append((x, y))

    def type_text(self, text: str) -> None:
        self.typed.append(text)

    def press_key(self, key: str) -> None:
        self.keys.append(key)

    def scroll_norm(self, dx: float, dy: float) -> None:
        pass


CF_HTML = (
    "<html><head><title>Just a moment...</title></head><body>"
    "Verify you are human — checking your browser</body></html>"
)
CLEAN_HTML = "<html><head><title>Home</title></head><body>hello</body></html>"
CAPTCHA_HTML = (
    "<html><body><form>"
    '<div class="g-recaptcha" data-sitekey="6Ltestkey"></div>'
    "</form></body></html>"
)


# -- detection ---------------------------------------------------------------

def test_detect_cloudflare_title():
    sig = detect_challenge_signal(CF_HTML, "https://example.test/")
    assert sig is not None
    assert sig.vendor == "cloudflare"
    assert sig.detection == "cloudflare-challenge"


def test_detect_captcha_kind():
    sig = detect_challenge_signal(CAPTCHA_HTML, "https://example.test/")
    assert sig is not None
    assert "recaptcha" in sig.kind.lower()


def test_detect_clean_page_is_none():
    assert detect_challenge_signal(CLEAN_HTML, "https://example.test/") is None


# -- solver: self-solve before escalation ------------------------------------

def test_solver_self_solves_on_retry_before_any_escalation():
    tab = FakeTab(CF_HTML)

    def _clear_on_reload() -> None:
        tab.set_html(CLEAN_HTML)

    orig_reload = tab.reload

    def reload_clears() -> None:
        orig_reload()
        _clear_on_reload()

    tab.reload = reload_clears  # type: ignore[method-assign]
    solver = ChallengeSolver(wait_s=0)
    outcome = solver.solve(tab=tab, resume_state={"step": 3, "url": tab.url})
    assert outcome.cleared is True
    assert outcome.via == "retry"
    assert outcome.takeover_url is None
    assert [a.strategy for a in outcome.attempts] == ["retry"]


def test_solver_tries_strategies_in_order_then_takeover():
    tab = FakeTab(CF_HTML)
    sent: list[str] = []

    def notifier(message: str, url: str) -> None:
        sent.append(message)

    mgr = TakeoverManager(TakeoverConfig(), notifier=notifier)
    solver = ChallengeSolver(wait_s=0, takeover_manager=mgr, notifier=notifier)
    outcome = solver.solve(
        tab=tab,
        resume_state={"step": 7, "url": tab.url},
        owner_chat="owner-1",
    )
    assert outcome.cleared is False
    assert outcome.via == "takeover_pending"
    assert outcome.takeover_url
    tried = [a.strategy for a in outcome.attempts]
    # Self-solve strategies run BEFORE the takeover strategy.
    assert tried[-1] == "owner_takeover"
    assert set(tried) >= {"retry", "wait_proof_of_work", "captcha_service",
                          "fresh_session"}
    assert len(sent) == 1
    # The escalation message must list what was already attempted.
    for strategy in tried[:-1]:
        assert strategy in sent[0]


def test_solver_without_manager_reports_takeover_required():
    tab = FakeTab(CF_HTML)
    solver = ChallengeSolver(wait_s=0)
    outcome = solver.solve(tab=tab, resume_state={"step": 1})
    assert outcome.cleared is False
    assert outcome.via == "takeover_required"
    assert outcome.takeover_url is None


def test_escalation_message_lists_attempts():
    msg = format_escalation_message(
        signal_kind="cloudflare-challenge",
        url="https://example.test/login",
        attempts=[("retry", False, "still walled after reload"),
                  ("captcha_service", False, "no solving service configured")],
        takeover_url="https://host/takeover/TOKEN",
        task_label="login",
    )
    assert "retry" in msg and "captcha_service" in msg
    assert "https://host/takeover/TOKEN" in msg
    assert "tried" in msg.lower()


# -- takeover links: owner-only / expiring / single-use ----------------------

def test_takeover_link_requires_token():
    mgr = TakeoverManager(TakeoverConfig())
    tab = FakeTab(CF_HTML)
    grant = mgr.request_takeover(
        session_id="s1", url=tab.url, task_label="t",
        signal_kind="cloudflare-challenge", attempts=[],
        resume_state={"step": 2}, tab=tab, owner_chat="owner-1",
    )
    assert grant.url.startswith("http")
    assert mgr.validate_token("bogus-token") is None
    assert mgr.validate_token(grant.token) is not None


def test_takeover_token_expires_and_closes_on_done():
    cfg = TakeoverConfig(link_ttl_s=60)
    mgr = TakeoverManager(cfg)
    tab = FakeTab(CF_HTML)
    grant = mgr.request_takeover(
        session_id="s2", url=tab.url, task_label="t",
        signal_kind="cloudflare-challenge", attempts=[],
        resume_state={"step": 2}, tab=tab, owner_chat="owner-1",
    )
    mgr.mark_done(grant, note="solved it")
    assert mgr.validate_token(grant.token) is None
    # A fresh grant works; the old token is dead (single-use grant).
    grant2 = mgr.request_takeover(
        session_id="s2", url=tab.url, task_label="t",
        signal_kind="cloudflare-challenge", attempts=[],
        resume_state={"step": 2}, tab=tab, owner_chat="owner-1",
    )
    assert grant2.token != grant.token
    assert mgr.validate_token(grant2.token) is not None


def test_takeover_supersedes_prior_grant_for_session():
    mgr = TakeoverManager(TakeoverConfig())
    tab = FakeTab(CF_HTML)
    g1 = mgr.request_takeover(
        session_id="s3", url=tab.url, task_label="t",
        signal_kind="cloudflare-challenge", attempts=[],
        resume_state={"step": 2}, tab=tab, owner_chat="owner-1",
    )
    g2 = mgr.request_takeover(
        session_id="s3", url=tab.url, task_label="t",
        signal_kind="cloudflare-challenge", attempts=[],
        resume_state={"step": 2}, tab=tab, owner_chat="owner-1",
    )
    assert mgr.validate_token(g1.token) is None
    assert mgr.validate_token(g2.token) is not None


# -- hand-back + resume ------------------------------------------------------

def test_explicit_done_resumes_at_saved_step():
    mgr = TakeoverManager(TakeoverConfig())
    tab = FakeTab(CF_HTML)
    grant = mgr.request_takeover(
        session_id="s4", url=tab.url, task_label="t",
        signal_kind="cloudflare-challenge", attempts=[],
        resume_state={"step": 5, "url": tab.url}, tab=tab,
        owner_chat="owner-1",
    )
    mgr.mark_done(grant, note="clicked the checkbox")
    tab.set_html(CLEAN_HTML)
    res = mgr.wait_for_resolution(
        grant, cleared_check=lambda: "captcha" not in tab.html().lower()
        and "just a moment" not in tab.html().lower(),
        timeout_s=5, poll_s=0.05,
    )
    assert res.cleared is True
    assert res.via == "owner_done"
    assert res.resume_state == {"step": 5, "url": tab.url}


def test_auto_detect_resumes_even_without_done():
    mgr = TakeoverManager(TakeoverConfig())
    tab = FakeTab(CF_HTML)
    grant = mgr.request_takeover(
        session_id="s5", url=tab.url, task_label="t",
        signal_kind="cloudflare-challenge", attempts=[],
        resume_state={"step": 9}, tab=tab, owner_chat="owner-1",
    )
    # Owner solves it and just closes the page — no Done tap.
    tab.set_html(CLEAN_HTML)
    res = mgr.wait_for_resolution(
        grant, cleared_check=lambda: "just a moment" not in tab.html().lower(),
        timeout_s=5, poll_s=0.05,
    )
    assert res.cleared is True
    assert res.via == "auto_detected"


def test_still_blocked_is_honest_no_blind_continue():
    mgr = TakeoverManager(TakeoverConfig())
    tab = FakeTab(CF_HTML)
    grant = mgr.request_takeover(
        session_id="s6", url=tab.url, task_label="t",
        signal_kind="cloudflare-challenge", attempts=[],
        resume_state={"step": 1}, tab=tab, owner_chat="owner-1",
    )
    mgr.mark_done(grant, note="thought I solved it")
    res = mgr.wait_for_resolution(
        grant, cleared_check=lambda: False,
        timeout_s=0.3, poll_s=0.05,
    )
    assert res.cleared is False
    assert res.via == "still_blocked"
    assert res.resume_state is None


def test_server_endpoints_need_token_and_forward_input():
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError
    import json

    mgr = TakeoverManager(TakeoverConfig(bind_host="127.0.0.1", port=0))
    tab = FakeTab(CF_HTML)
    grant = mgr.request_takeover(
        session_id="s7", url=tab.url, task_label="t",
        signal_kind="cloudflare-challenge", attempts=[],
        resume_state={"step": 1}, tab=tab, owner_chat="owner-1",
    )
    base = mgr.server_url
    assert base
    # No token -> 403.
    with pytest.raises(HTTPError) as excinfo:
        urlopen(base + "/takeover/nope/frame", timeout=5)
    assert excinfo.value.code == 403
    # Frame with token -> PNG bytes.
    with urlopen(grant.url + "/frame", timeout=5) as resp:
        assert resp.status == 200
        assert resp.read().startswith(b"\x89PNG")
    # Click forwards normalized coords into the tab.
    req = Request(grant.url + "/click", data=json.dumps(
        {"x": 0.5, "y": 0.25}).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=5) as resp:
        assert resp.status == 200
    assert tab.clicks == [(0.5, 0.25)]
    # Done marks the grant done.
    req = Request(grant.url + "/done", data=json.dumps(
        {"note": "done"}).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=5) as resp:
        assert resp.status == 200
    assert grant.done is True
    mgr.shutdown()


# -- learning ----------------------------------------------------------------

def test_memory_ranks_past_winner_first(tmp_path):
    mem = ChallengeMemory(tmp_path / "mem.db")
    mem.record(site="example.test", kind="cloudflare-challenge",
               vendor="cloudflare",
               strategies=[("retry", False, 100), ("fresh_session", True, 200)],
               cleared=True, via="fresh_session", takeover_needed=False, ms=300)
    mem.record(site="example.test", kind="cloudflare-challenge",
               vendor="cloudflare",
               strategies=[("retry", False, 100), ("fresh_session", True, 200)],
               cleared=True, via="fresh_session", takeover_needed=False, ms=300)
    ranked = mem.rank_strategies(
        "example.test", "cloudflare-challenge",
        ["retry", "wait_proof_of_work", "captcha_service", "fresh_session"])
    assert ranked[0] == "fresh_session"
    mem.close()


def test_memory_unknown_site_keeps_default_order(tmp_path):
    mem = ChallengeMemory(tmp_path / "mem.db")
    ranked = mem.rank_strategies(
        "never-seen.test", "cloudflare-challenge",
        ["retry", "wait_proof_of_work", "captcha_service", "fresh_session"])
    assert ranked == ["retry", "wait_proof_of_work", "captcha_service",
                      "fresh_session"]
    mem.close()


def test_solver_uses_memory_ordering(tmp_path):
    mem = ChallengeMemory(tmp_path / "mem.db")
    sig = detect_challenge_signal(CAPTCHA_HTML, "https://example.test/")
    assert sig is not None
    mem.record(site="example.test", kind=sig.kind,
               vendor="captcha",
               strategies=[("captcha_service", True, 100)],
               cleared=True, via="captcha_service", takeover_needed=False,
               ms=100)
    tab = FakeTab(CAPTCHA_HTML)
    solved: list[str] = []

    def fake_service(challenges):
        solved.append("called")
        tab.set_html(CLEAN_HTML)

        class _R:
            ok = True
        return _R()

    solver = ChallengeSolver(memory=mem, service_solve=fake_service, wait_s=0)
    outcome = solver.solve(tab=tab, resume_state={"step": 1})
    assert outcome.cleared is True
    assert solved == ["called"]
    # captcha_service ran before retry (memory-ranked first).
    assert outcome.attempts[0].strategy == "captcha_service"
    mem.close()


def test_audit_log_written(tmp_path):
    mgr = TakeoverManager(TakeoverConfig(),
                          data_dir=tmp_path)
    tab = FakeTab(CF_HTML)
    grant = mgr.request_takeover(
        session_id="s8", url=tab.url, task_label="t",
        signal_kind="cloudflare-challenge", attempts=[],
        resume_state={"step": 1}, tab=tab, owner_chat="owner-1",
    )
    mgr.mark_done(grant, note="ok")
    log = tmp_path / "takeover-audit.jsonl"
    assert log.is_file()
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2
    # No raw token in the audit log — only a hash prefix.
    assert grant.token not in log.read_text(encoding="utf-8")


def test_expired_grant_rejected(monkeypatch):
    mgr = TakeoverManager(TakeoverConfig(link_ttl_s=60))
    tab = FakeTab(CF_HTML)
    grant = mgr.request_takeover(
        session_id="s9", url=tab.url, task_label="t",
        signal_kind="cloudflare-challenge", attempts=[],
        resume_state={"step": 1}, tab=tab, owner_chat="owner-1",
    )
    # Pretend 2 hours passed.
    grant.expires_at = time.time() - 1
    assert mgr.validate_token(grant.token) is None


# -- AgentLoop integration ---------------------------------------------------

def test_agent_loop_wall_handler_can_continue():
    from nomorals.browser.agent import AgentLoop, Page

    class FakeDriver:
        def fetch(self, url: str) -> Page:
            return Page(url=url, html=CF_HTML, title="Just a moment...")

        def act(self, action, page: Page) -> Page:
            return page

    calls: list[str] = []

    def handler(obs, detection: str) -> str:
        calls.append(detection)
        return "continue"

    loop = AgentLoop(FakeDriver(), on_blocked=handler, max_steps=2)
    result = loop.run("https://example.test/")
    assert calls == ["cloudflare-challenge"]
    # Wall handled -> loop proceeded to decide (nothing safe to do).
    assert result["stopped_reason"] == "nothing safe to do"


def test_agent_loop_wall_without_handler_stops():
    from nomorals.browser.agent import AgentLoop, Page

    class FakeDriver:
        def fetch(self, url: str) -> Page:
            return Page(url=url, html=CF_HTML, title="Just a moment...")

        def act(self, action, page: Page) -> Page:
            return page

    loop = AgentLoop(FakeDriver(), max_steps=2)
    result = loop.run("https://example.test/")
    assert result["stopped_reason"].startswith("blocked: cloudflare")


def test_browser_service_takeover_manager_factory(tmp_path):
    from nomorals.browser import BrowserService

    svc = BrowserService(data_dir=tmp_path / "svc")
    mgr = svc.takeover_manager()
    assert isinstance(mgr, TakeoverManager)
    assert mgr._data_dir == tmp_path / "svc" / "takeover"

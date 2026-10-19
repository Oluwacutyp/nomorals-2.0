"""Auto-escalation: the browser agent loop wired to ChallengeSolver.

Until this wiring, ``AgentLoop.on_blocked`` had NO production caller:
the takeover library, the Telegram notifier, and the ``/browser
takeover`` verb all existed, but when Devon herself hit a wall
mid-task she never auto-escalated — only the operator-driven verb
worked. These tests pin the seam:

- (a) loop hits a wall -> ladder runs, self-solve attempted BEFORE any
  escalation;
- (b) escalation reaches the OWNER chat only, with attempts + link;
- (c) takeover clear -> loop resumes at the saved step, on the
  post-clear page (not the stale wall HTML);
- (d) still-blocked -> loop stops honestly, nothing saved;
- save-on-solve cookies apply on the loop path;
- ``/browser run`` is the production entry point (owner-only).
"""
from __future__ import annotations

import re
import tempfile
from typing import Any

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.browser.agent import AgentLoop, Decision, Observation
from nomorals.browser.challenge import ChallengeSolver
from nomorals.browser.loop_wiring import (
    RenderedTabDriver,
    build_wired_agent_loop,
)
from nomorals.browser.service import RenderedTabTakeoverAdapter
from nomorals.browser.solved_sessions import SolvedSessionStore
from nomorals.browser.takeover import TakeoverConfig, TakeoverManager
from nomorals.core.config import load_settings
from nomorals.social.chat.base import ChatAdapter, ChatKind, ChatMessage, ChatRef, SendResult
from nomorals.social.chat.gateway import ChatGateway

CF_HTML = (
    "<html><head><title>Just a moment...</title></head><body>"
    "Verify you are human — checking your browser</body></html>"
)
CLEAN_HTML = (
    "<html><head><title>Home</title></head><body>hello "
    '<button>Claim reward</button></body></html>'
)
URL = "https://example.test/"


class FakeLiveTab:
    """RenderedTab-shaped: html() -> dict, navigate/click/fill/snapshot."""

    def __init__(self, html: str = CLEAN_HTML, *,
                 clear_on_reload: bool = False,
                 cookies_to_save: list[dict[str, Any]] | None = None) -> None:
        self._html = html
        self.url = URL
        self.clear_on_reload = clear_on_reload
        self._cookies = list(cookies_to_save or [
            {"name": "cf_clearance", "value": "abc",
             "domain": "example.test", "path": "/"}])
        self.added_cookies: list[dict[str, Any]] = []
        self.clicks: list[str] = []
        self.fills: list[tuple[str, str]] = []
        self.reloads = 0
        self.typed_focused: list[str] = []

    @property
    def title(self) -> str:
        import re

        match = re.search(r"<title[^>]*>(.*?)</title>", self._html,
                          re.DOTALL | re.IGNORECASE)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()
        return "Home"

    def navigate(self, url: str) -> dict[str, Any]:
        self.url = url
        return {"ok": True, "url": url}

    def html(self) -> dict[str, Any]:
        return {"url": self.url, "title": self.title,
                "chars": len(self._html), "html": self._html,
                "truncated": False}

    def reload(self) -> None:
        self.reloads += 1
        if self.clear_on_reload:
            self._html = CLEAN_HTML

    def click(self, target: str) -> dict[str, Any]:
        self.clicks.append(target)
        return {"ok": True, "target": target}

    def fill(self, target: str, value: str) -> dict[str, Any]:
        self.fills.append((target, value))
        return {"ok": True}

    def snapshot(self) -> dict[str, Any]:
        return {"snapshot": '- button "Claim reward" [ref=e1]',
                "refs": {"e1": "aria-ref=e1"},
                "source": "aria", "truncated": False,
                "url": self.url, "tab_id": "t1"}

    def back(self) -> dict[str, Any]:
        return {"ok": True}

    def cookies(self) -> list[dict[str, Any]]:
        return list(self._cookies)

    def storage_state(self) -> dict[str, Any]:
        return {"cookies": list(self._cookies), "origins": []}

    def add_cookies(self, cookies: list[dict[str, Any]]) -> dict[str, Any]:
        self.added_cookies.extend(cookies)
        return {"ok": True}

    def type_into_focused(self, text: str) -> None:
        self.typed_focused.append(text)


class FakeGrant:
    def __init__(self, url: str = "http://host:9999/takeover/tok") -> None:
        self.url = url
        self.needs_public_base = True
        self.done = False
        self.resume_state: dict[str, Any] = {}


class FakeTakeoverManager:
    """Mimics TakeoverManager: owner solves in the live view on request."""

    def __init__(self, *, resolve_cleared: bool = True) -> None:
        self.resolve_cleared = resolve_cleared
        self.grants: list[FakeGrant] = []
        self.saved: list[dict[str, Any]] = []
        self.wait_timeout_s = 60

    def request_takeover(self, **kwargs: Any) -> FakeGrant:
        grant = FakeGrant()
        grant.resume_state = dict(kwargs.get("resume_state") or {})
        self.grants.append(grant)
        # The owner solves the challenge in the live view: flip the
        # underlying tab to clean, like a real hand-back would.
        tab = kwargs.get("tab")
        inner = getattr(tab, "_tab", tab)
        if self.resolve_cleared and inner is not None:
            inner._html = CLEAN_HTML
        return grant

    def wait_for_resolution(self, grant: FakeGrant, *,
                            cleared_check: Any, **_kw: Any) -> Any:
        from types import SimpleNamespace

        if self.resolve_cleared and cleared_check():
            return SimpleNamespace(cleared=True, via="owner_done",
                                   resume_state=dict(grant.resume_state),
                                   waited_s=0.0)
        return SimpleNamespace(cleared=False, via="still_blocked",
                               resume_state=None, waited_s=0.0)

    def record_session_saved(self, grant: FakeGrant, *,
                             site: str, cookie_count: int,
                             via: str) -> None:
        self.saved.append({"site": site, "cookies": cookie_count,
                           "via": via})


class FakeNotifier:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def __call__(self, message: str, url: str) -> None:
        self.sent.append((message, url))


def _store(tmp: str) -> SolvedSessionStore:
    return SolvedSessionStore(f"{tmp}/takeover")


def _approve_benign(decision: Any, _obs: Any) -> bool:
    """The production approver: benign acts proceed, consequential
    ones (spend/post/delete/irreversible) are denied — no human
    approver is attached, so they stop the loop instead of running."""
    from nomorals.browser.agent import CONSEQUENTIAL_KINDS

    return (not decision.irreversible
            and decision.action.kind not in CONSEQUENTIAL_KINDS)


def _loop(tab: FakeLiveTab, *, manager: Any = None,
          notifier: Any = None, store: Any = None,
          max_steps: int = 2) -> AgentLoop:
    mgr = manager if manager is not None else FakeTakeoverManager()
    note = notifier if notifier is not None else FakeNotifier()
    return build_wired_agent_loop(
        tab=tab, takeover_manager=mgr, notifier=note,
        session_store=store, owner_chat="telegram:111",
        task_label="browse", max_steps=max_steps,
        approve=_approve_benign)


# -- (a) ladder runs, self-solve BEFORE any escalation ----------------------

def test_loop_wall_selfsolve_before_escalation():
    tab = FakeLiveTab(CF_HTML, clear_on_reload=True)
    manager = FakeTakeoverManager()
    notifier = FakeNotifier()
    loop = _loop(tab, manager=manager, notifier=notifier)
    result = loop.run(URL, goal="browse")
    assert "blocked" not in result["stopped_reason"]
    assert result["steps"] == 2  # loop proceeded past the cleared wall
    assert manager.grants == []  # never escalated
    assert notifier.sent == []
    assert tab.reloads >= 1  # the retry rung actually ran


# -- (b) escalation reaches the owner only, with attempts + link ------------

def test_loop_wall_escalates_to_owner_with_attempts_and_link():
    tab = FakeLiveTab(CF_HTML)  # ladder cannot clear it
    manager = FakeTakeoverManager(resolve_cleared=True)
    notifier = FakeNotifier()
    loop = _loop(tab, manager=manager, notifier=notifier)
    result = loop.run(URL, goal="browse")
    assert "blocked" not in result["stopped_reason"]
    assert len(manager.grants) == 1
    assert len(notifier.sent) == 1
    message, url = notifier.sent[0]
    assert "retry" in message  # attempt receipt
    assert manager.grants[0].url in message  # the takeover link
    assert "Chrome" in message  # webview + open-in-Chrome hint
    # Resume state carries where the loop left off.
    assert manager.grants[0].resume_state.get("step") == 0


# -- (c) takeover clear resumes at the saved step, on the fresh page --------

def test_loop_takeover_clear_resumes_on_fresh_page():
    tab = FakeLiveTab(CF_HTML)
    manager = FakeTakeoverManager(resolve_cleared=True)
    seen_pages: list[str] = []

    def _decide(obs: Observation) -> Decision | None:
        seen_pages.append(obs.page.html)
        return None  # stop after observing the post-clear page

    loop = _loop(tab, manager=manager, notifier=FakeNotifier())
    loop.decide_llm = _decide  # type: ignore[attr-defined]
    result = loop.run(URL, goal="browse")
    assert "blocked" not in result["stopped_reason"]
    assert seen_pages, "loop never decided after the wall cleared"
    assert "Just a moment" not in seen_pages[0]
    assert "hello" in seen_pages[0]


# -- (d) still-blocked stops honestly, saves nothing -------------------------

def test_loop_still_blocked_stops_and_saves_nothing():
    with tempfile.TemporaryDirectory(prefix="nm-autoloop-") as tmp:
        tab = FakeLiveTab(CF_HTML)
        manager = FakeTakeoverManager(resolve_cleared=False)
        notifier = FakeNotifier()
        store = _store(tmp)
        loop = _loop(tab, manager=manager, notifier=notifier, store=store)
        result = loop.run(URL, goal="browse")
        assert "blocked" in result["stopped_reason"]
        assert result["steps"] == 0
        assert len(notifier.sent) == 1  # escalation happened
        assert store.load(URL) is None  # ...but nothing was saved
        assert manager.saved == []


# -- save-on-solve on the loop path ------------------------------------------

def test_loop_save_on_solve_persists_cookies():
    with tempfile.TemporaryDirectory(prefix="nm-autoloop-") as tmp:
        tab = FakeLiveTab(CF_HTML, clear_on_reload=True)
        store = _store(tmp)
        loop = _loop(tab, notifier=FakeNotifier(), store=store)
        result = loop.run(URL, goal="browse")
        assert "blocked" not in result["stopped_reason"]
        saved = store.load(URL)
        assert saved is not None
        names = [c["name"] for c in saved["cookies"]]
        assert "cf_clearance" in names


def test_adapter_delegates_cookie_methods():
    tab = FakeLiveTab(CLEAN_HTML)
    adapter = RenderedTabTakeoverAdapter(tab)
    assert adapter.storage_state()["cookies"][0]["name"] == "cf_clearance"
    assert adapter.cookies()[0]["name"] == "cf_clearance"
    adapter.add_cookies([{"name": "x", "value": "1",
                          "domain": "example.test", "path": "/"}])
    assert tab.added_cookies[0]["name"] == "x"


def test_adapter_type_text_uses_focused_typing():
    tab = FakeLiveTab(CLEAN_HTML)
    adapter = RenderedTabTakeoverAdapter(tab)
    adapter.type_text("hello")
    assert tab.typed_focused == ["hello"]


def test_adapter_type_text_falls_back_to_legacy_tab():
    class LegacyTab:
        def __init__(self) -> None:
            self.typed: list[str] = []
            self.url = URL
            self.title = "Home"

        def html(self) -> str:
            return CLEAN_HTML

        def reload(self) -> None:
            pass

        def type_text(self, text: str) -> None:
            self.typed.append(text)

    legacy = LegacyTab()
    adapter = RenderedTabTakeoverAdapter(legacy)
    adapter.type_text("hi")
    assert legacy.typed == ["hi"]


# -- /browser run: production entry point ------------------------------------

class _NoLlmBrain:
    """Brain with no usable LLM: every decider call falls back."""

    def complete(self, prompt: str, **_kw: Any) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(ok=False, text="")


_JOB_RE = re.compile(r"br-[0-9a-f]{12}")


def _job_id(reply: str) -> str:
    match = _JOB_RE.search(reply)
    assert match, f"no job id in reply: {reply!r}"
    return match.group(0)

class FakeTelegramAdapter(ChatAdapter):
    def __init__(self, name: str = "telegram") -> None:
        super().__init__(media_dir="/tmp/nm-test-media")
        self.name = name
        self.sent: list[tuple[str, str]] = []

    def run(self, handler: Any) -> None:
        pass

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
             buttons: Any = None, parse_mode: str = "") -> SendResult:
        self.sent.append((chat.chat_id, text))
        return SendResult(ok=True, platform=self.name,
                          message_id=f"m{len(self.sent)}")


def _make_runtime() -> tuple[PartnerRuntime, FakeTelegramAdapter,
                             tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory(prefix="nm-autoloop-verb-")
    settings = load_settings(overrides={
        "home": tmp.name,
        "partner.platforms": "telegram",
        "partner.owner_chats": "telegram:111",
        "partner.enabled": "false",
    })
    context = build_context(settings, with_executor=False, with_tools=False)
    adapter = FakeTelegramAdapter("telegram")
    gateway = ChatGateway({"telegram": adapter}, db=context.db)
    runtime = PartnerRuntime(context, gateway=gateway)
    return runtime, adapter, tmp


def _owner_msg() -> ChatMessage:
    return ChatMessage(
        chat=ChatRef(platform="telegram", chat_id="111", kind=ChatKind.DM),
        incoming=True, text="/browser run", sender="owner",
        sender_id="111", meta={"is_owner": True})


def _stranger_msg() -> ChatMessage:
    return ChatMessage(
        chat=ChatRef(platform="telegram", chat_id="999", kind=ChatKind.DM),
        incoming=True, text="/browser run", sender="stranger",
        sender_id="999", meta={})


def _teardown(runtime: PartnerRuntime, tmp: tempfile.TemporaryDirectory) -> None:
    try:
        runtime.stop()
    except Exception:
        pass
    try:
        runtime.context.close()
    except Exception:
        pass
    tmp.cleanup()


class FakeBrowserService:
    def __init__(self, tab: Any, store: Any = None) -> None:
        self._tab = tab
        self._store = store
        self.opened: list[tuple[str, str]] = []

    def open_rendered_tab(self, session_name: str, url: str = "") -> Any:
        self.opened.append((session_name, url))
        return self._tab

    def solved_session_store(self) -> Any:
        return self._store

    def takeover_manager(self, *, config: Any = None,
                         notifier: Any = None,
                         session_store: Any = None) -> Any:
        return TakeoverManager(TakeoverConfig(),
                               data_dir="/tmp/nm-autoloop-unused",
                               notifier=notifier)


def test_browser_run_verb_refuses_non_owner():
    runtime, adapter, tmp = _make_runtime()
    try:
        reply = runtime._control_browser(
            "run https://example.test/", chat_key="telegram:999",
            message=_stranger_msg())
        assert "owner-only" in reply.lower()
        assert adapter.sent == []
    finally:
        _teardown(runtime, tmp)


def test_browser_run_verb_runs_loop_and_reports(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "nomorals.llm.brain.brain_for",
        lambda ctx: _NoLlmBrain())
    runtime, adapter, tmp = _make_runtime()
    try:
        runtime._browser_service_for_run_job = (  # type: ignore[method-assign]
            lambda job_id: FakeBrowserService(FakeLiveTab(CLEAN_HTML)))
        # The verb returns immediately with a job id — the loop runs
        # detached. Poll the job for the outcome it used to return.
        reply = runtime._control_browser(
            "run https://example.test/ browse", chat_key="telegram:111",
            message=_owner_msg())
        assert "started" in reply.lower()
        job_id = _job_id(reply)
        assert runtime._browser_run_jobs().get(job_id).wait(15)
        status = runtime._control_browser(
            f"run status {job_id}", chat_key="telegram:111",
            message=_owner_msg())
        assert "steps" in status.lower()
        assert "example.test" in status
        # No wall, no escalation — only the job-done update.
        assert len(adapter.sent) == 1
        assert "done" in adapter.sent[0][1].lower()
    finally:
        _teardown(runtime, tmp)


def test_browser_run_verb_wall_escalates_to_owner(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "nomorals.llm.brain.brain_for",
        lambda ctx: _NoLlmBrain())
    runtime, adapter, tmp = _make_runtime()
    try:
        tab = FakeLiveTab(CF_HTML)
        runtime._browser_service_for_run_job = (  # type: ignore[method-assign]
            lambda job_id: FakeBrowserService(tab))
        # Short-circuit the real manager: the fake resolves like an
        # owner solving in the live view.
        runtime._takeover_manager_inst = FakeTakeoverManager(
            resolve_cleared=True)
        reply = runtime._control_browser(
            "run https://example.test/ browse", chat_key="telegram:111",
            message=_owner_msg())
        job_id = _job_id(reply)
        assert runtime._browser_run_jobs().get(job_id).wait(15)
        status = runtime._control_browser(
            f"run status {job_id}", chat_key="telegram:111",
            message=_owner_msg())
        assert "blocked" not in status.lower()
        assert "cleared via takeover" in status
        assert len(adapter.sent) >= 1
        assert adapter.sent[0][0] == "111"
        assert any("retry" in text for _, text in adapter.sent)
    finally:
        _teardown(runtime, tmp)


def test_takeover_verb_now_passes_session_store():
    runtime, adapter, tmp = _make_runtime()
    try:
        tab = FakeLiveTab(CF_HTML, clear_on_reload=True)
        store = _store(tmp.name)
        runtime._browser_service_for_takeover = (  # type: ignore[method-assign]
            lambda: FakeBrowserService(tab, store=store))
        reply = runtime._control_browser(
            "takeover https://example.test/", chat_key="telegram:111",
            message=_owner_msg())
        assert "wall cleared via retry" in reply.lower()
        saved = store.load(URL)
        assert saved is not None
        assert saved["cookies"][0]["name"] == "cf_clearance"
    finally:
        _teardown(runtime, tmp)

"""Background jobs for ``/browser run`` + owner Telegram updates.

(a) ``/browser run`` returns immediately with a job id while the loop
    runs detached on a worker thread;
(b) the owner gets Telegram updates on wall hit -> escalation,
    cleared -> resumed, blocked -> stopped, done;
(c) ``/browser run status <job>`` reports steps + wall outcome +
    final URL;
- no duplicate runs for the same URL/goal unless ``--new`` starts one
  explicitly.
"""
from __future__ import annotations

import re
import tempfile
import threading
from types import SimpleNamespace
from typing import Any

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.browser.agent import AgentLoop
from nomorals.browser.jobs import BrowserJobManager
from nomorals.browser.loop_wiring import build_wired_agent_loop
from nomorals.browser.solved_sessions import SolvedSessionStore
from nomorals.browser.takeover import TakeoverConfig, TakeoverManager
from nomorals.core.config import load_settings
from nomorals.social.chat.base import (
    ChatAdapter,
    ChatKind,
    ChatMessage,
    ChatRef,
    SendResult,
)
from nomorals.social.chat.gateway import ChatGateway

CF_HTML = (
    "<html><head><title>Just a moment...</title></head><body>"
    "Verify you are human — checking your browser</body></html>"
)
CLEAN_HTML = (
    "<html><head><title>Home</title></head><body>hello "
    '<button>Claim reward</button></body></html>'
)
DONE_HTML = "<html><head><title>Done</title></head><body>all done</body></html>"
URL = "https://example.test/"


# -- fakes ---------------------------------------------------------------------

class FakeLiveTab:
    def __init__(self, html: str = CLEAN_HTML, *,
                 clear_on_reload: bool = False) -> None:
        self._html = html
        self.url = URL
        self.clear_on_reload = clear_on_reload
        self._cookies = [{"name": "cf_clearance", "value": "abc",
                          "domain": "example.test", "path": "/"}]
        self.added_cookies: list[dict[str, Any]] = []
        self.clicks: list[str] = []
        self.reloads = 0
        self.typed_focused: list[str] = []

    @property
    def title(self) -> str:
        match = re.search(r"<title[^>]*>(.*?)</title>", self._html,
                          re.DOTALL | re.IGNORECASE)
        return re.sub(r"\s+", " ", match.group(1)).strip() if match else "Home"

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


class BlockingTab(FakeLiveTab):
    """First click blocks until the test releases it — proves the verb
    returns while the loop is still running."""

    def __init__(self, release: threading.Event) -> None:
        super().__init__(CLEAN_HTML)
        self._release = release
        self._blocked_once = False

    def click(self, target: str) -> dict[str, Any]:
        if not self._blocked_once:
            self._blocked_once = True
            assert self._release.wait(15), "test never released the click"
            self._html = DONE_HTML
        return super().click(target)

    def snapshot(self) -> dict[str, Any]:
        if self._blocked_once:
            return {"snapshot": "", "refs": {}, "source": "aria",
                    "truncated": False, "url": self.url, "tab_id": "t1"}
        return super().snapshot()


class FakeGrant:
    def __init__(self, url: str = "http://host:9999/takeover/tok") -> None:
        self.url = url
        self.needs_public_base = True
        self.done = False
        self.resume_state: dict[str, Any] = {}


class FakeTakeoverManager:
    def __init__(self, *, resolve_cleared: bool = True) -> None:
        self.resolve_cleared = resolve_cleared
        self.grants: list[FakeGrant] = []
        self.saved: list[dict[str, Any]] = []
        self.wait_timeout_s = 60

    def request_takeover(self, **kwargs: Any) -> FakeGrant:
        grant = FakeGrant()
        grant.resume_state = dict(kwargs.get("resume_state") or {})
        self.grants.append(grant)
        tab = kwargs.get("tab")
        inner = getattr(tab, "_tab", tab)
        if self.resolve_cleared and inner is not None:
            inner._html = CLEAN_HTML
        return grant

    def wait_for_resolution(self, grant: FakeGrant, *,
                            cleared_check: Any, **_kw: Any) -> Any:
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


class FakeBrain:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt: str, **_kw: Any) -> Any:
        self.prompts.append(prompt)
        return SimpleNamespace(ok=False, text="")


def _make_runtime() -> tuple[PartnerRuntime, FakeTelegramAdapter,
                             tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory(prefix="nm-run-jobs-")
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


_JOB_RE = re.compile(r"br-[0-9a-f]{12}")


def _job_id(reply: str) -> str:
    match = _JOB_RE.search(reply)
    assert match, f"no job id in reply: {reply!r}"
    return match.group(0)


def _no_llm(monkeypatch: Any) -> None:
    monkeypatch.setattr("nomorals.llm.brain.brain_for",
                        lambda ctx: FakeBrain())


def _approve_benign(decision: Any, _obs: Any) -> bool:
    from nomorals.browser.agent import CONSEQUENTIAL_KINDS

    return (not decision.irreversible
            and decision.action.kind not in CONSEQUENTIAL_KINDS)


# -- manager: dedupe -----------------------------------------------------------

def test_manager_dedupes_same_url_and_goal():
    manager = BrowserJobManager()
    started = threading.Event()
    release = threading.Event()
    calls = {"n": 0}

    def _runner(job: Any) -> dict[str, Any]:
        calls["n"] += 1
        started.set()
        assert release.wait(10)
        return {"goal": "browse", "steps": 0, "stopped_reason": "done",
                "final_url": URL, "trace": []}

    job1, dup1 = manager.start(URL, "browse", _runner)
    assert not dup1
    assert started.wait(5)
    job2, dup2 = manager.start(URL, "browse", _runner)
    assert dup2 and job2.job_id == job1.job_id
    assert calls["n"] == 1
    release.set()
    assert job1.wait(10)
    assert job1.status == "done"


def test_manager_force_starts_a_second_job():
    manager = BrowserJobManager()
    release = threading.Event()

    def _runner(job: Any) -> dict[str, Any]:
        assert release.wait(10)
        return {"goal": "browse", "steps": 0, "stopped_reason": "done",
                "final_url": URL, "trace": []}

    job1, _ = manager.start(URL, "browse", _runner)
    job2, dup2 = manager.start(URL, "browse", _runner, force=True)
    assert not dup2 and job2.job_id != job1.job_id
    release.set()
    assert job1.wait(10) and job2.wait(10)


def test_manager_records_wall_outcome_from_loop():
    tab = FakeLiveTab(CF_HTML)
    manager = FakeTakeoverManager(resolve_cleared=True)
    jobs = BrowserJobManager()
    job_holder: dict[str, Any] = {}

    def _runner(job: Any) -> dict[str, Any]:
        loop = build_wired_agent_loop(
            tab=tab, takeover_manager=manager,
            notifier=FakeNotifier(), owner_chat="telegram:111",
            task_label="browse", max_steps=1,
            approve=_approve_benign,
            on_wall_outcome=job.wall_event)
        return loop.run(URL, goal="browse")

    job, _ = jobs.start(URL, "browse", _runner)
    assert job.wait(10)
    assert job.wall_outcome == "cleared via takeover:owner_done"
    kinds = [e["kind"] for e in job.events]
    assert "wall_hit" in kinds and "wall_cleared" in kinds


# -- (a) verb returns immediately with a job id ---------------------------------

def test_browser_run_returns_immediately_while_loop_runs(monkeypatch: Any):
    _no_llm(monkeypatch)
    runtime, adapter, tmp = _make_runtime()
    try:
        release = threading.Event()
        tab = BlockingTab(release)
        svc = FakeBrowserService(tab)
        runtime._browser_service_for_run_job = (  # type: ignore[method-assign]
            lambda job_id: svc)
        reply = runtime._control_browser(
            "run https://example.test/ browse", chat_key="telegram:111",
            message=_owner_msg())
        job_id = _job_id(reply)
        # The reply came back while the loop is still blocked on click.
        assert "started" in reply.lower()
        status = runtime._control_browser(
            f"run status {job_id}", chat_key="telegram:111",
            message=_owner_msg())
        assert "running" in status.lower()
        release.set()
        jobs = runtime._browser_run_jobs()
        assert jobs.get(job_id).wait(10)
        done = runtime._control_browser(
            f"run status {job_id}", chat_key="telegram:111",
            message=_owner_msg())
        assert "steps" in done.lower() and "example.test" in done
        assert any("done" in text.lower() and job_id in text
                   for _, text in adapter.sent)
    finally:
        _teardown(runtime, tmp)


def test_browser_run_verb_dedupes_and_new_forces(monkeypatch: Any):
    _no_llm(monkeypatch)
    runtime, adapter, tmp = _make_runtime()
    try:
        release = threading.Event()
        tab = BlockingTab(release)
        svc = FakeBrowserService(tab)
        runtime._browser_service_for_run_job = (  # type: ignore[method-assign]
            lambda job_id: svc)
        first = runtime._control_browser(
            "run https://example.test/ browse", chat_key="telegram:111",
            message=_owner_msg())
        first_id = _job_id(first)
        second = runtime._control_browser(
            "run https://example.test/ browse", chat_key="telegram:111",
            message=_owner_msg())
        assert _job_id(second) == first_id
        assert "already running" in second.lower()
        assert len(svc.opened) == 1
        third = runtime._control_browser(
            "run https://example.test/ browse --new",
            chat_key="telegram:111", message=_owner_msg())
        third_id = _job_id(third)
        assert third_id != first_id
        assert len(svc.opened) == 2
        release.set()
        jobs = runtime._browser_run_jobs()
        assert jobs.get(first_id).wait(10)
        assert jobs.get(third_id).wait(10)
    finally:
        _teardown(runtime, tmp)


# -- (b) Telegram updates on wall / cleared / blocked / done --------------------

def test_browser_run_wall_updates_owner(monkeypatch: Any):
    _no_llm(monkeypatch)
    runtime, adapter, tmp = _make_runtime()
    try:
        tab = FakeLiveTab(CF_HTML)
        runtime._browser_service_for_run_job = (  # type: ignore[method-assign]
            lambda job_id: FakeBrowserService(tab))
        runtime._takeover_manager_inst = FakeTakeoverManager(
            resolve_cleared=True)
        reply = runtime._control_browser(
            "run https://example.test/ browse", chat_key="telegram:111",
            message=_owner_msg())
        job_id = _job_id(reply)
        assert runtime._browser_run_jobs().get(job_id).wait(15)
        bodies = [text for _, text in adapter.sent]
        assert any("retry" in b for b in bodies)  # escalation receipt
        assert any("cleared via takeover" in b for b in bodies)
        assert any("done" in b.lower() and job_id in b for b in bodies)
        status = runtime._control_browser(
            f"run status {job_id}", chat_key="telegram:111",
            message=_owner_msg())
        assert "cleared via takeover:owner_done" in status
    finally:
        _teardown(runtime, tmp)


def test_browser_run_blocked_updates_owner_and_status(monkeypatch: Any):
    _no_llm(monkeypatch)
    runtime, adapter, tmp = _make_runtime()
    try:
        tab = FakeLiveTab(CF_HTML)
        runtime._browser_service_for_run_job = (  # type: ignore[method-assign]
            lambda job_id: FakeBrowserService(tab))
        runtime._takeover_manager_inst = FakeTakeoverManager(
            resolve_cleared=False)
        reply = runtime._control_browser(
            "run https://example.test/ browse", chat_key="telegram:111",
            message=_owner_msg())
        job_id = _job_id(reply)
        job = runtime._browser_run_jobs().get(job_id)
        assert job.wait(15)
        assert job.status == "blocked"
        bodies = [text for _, text in adapter.sent]
        assert any("stopped" in b.lower() and "blocked" in b.lower()
                   for b in bodies)
        status = runtime._control_browser(
            f"run status {job_id}", chat_key="telegram:111",
            message=_owner_msg())
        assert "blocked" in status.lower()
        assert "final url" in status.lower()
    finally:
        _teardown(runtime, tmp)


# -- (c) status command ----------------------------------------------------------

def test_browser_run_status_lists_and_unknown(monkeypatch: Any):
    _no_llm(monkeypatch)
    runtime, adapter, tmp = _make_runtime()
    try:
        runtime._browser_service_for_run_job = (  # type: ignore[method-assign]
            lambda job_id: FakeBrowserService(FakeLiveTab(CLEAN_HTML)))
        reply = runtime._control_browser(
            "run https://example.test/ browse", chat_key="telegram:111",
            message=_owner_msg())
        job_id = _job_id(reply)
        assert runtime._browser_run_jobs().get(job_id).wait(10)
        listed = runtime._control_browser(
            "run status", chat_key="telegram:111", message=_owner_msg())
        assert job_id in listed
        unknown = runtime._control_browser(
            "run status br-deadbeef0000", chat_key="telegram:111",
            message=_owner_msg())
        assert "no browser run job" in unknown.lower()
        prefix = runtime._control_browser(
            f"run status {job_id[:8]}", chat_key="telegram:111",
            message=_owner_msg())
        assert job_id in prefix
    finally:
        _teardown(runtime, tmp)


def test_browser_run_status_reports_steps_wall_and_final_url(monkeypatch: Any):
    _no_llm(monkeypatch)
    runtime, adapter, tmp = _make_runtime()
    try:
        runtime._browser_service_for_run_job = (  # type: ignore[method-assign]
            lambda job_id: FakeBrowserService(FakeLiveTab(CLEAN_HTML)))
        reply = runtime._control_browser(
            "run https://example.test/ browse", chat_key="telegram:111",
            message=_owner_msg())
        job_id = _job_id(reply)
        assert runtime._browser_run_jobs().get(job_id).wait(10)
        status = runtime._control_browser(
            f"run status {job_id}", chat_key="telegram:111",
            message=_owner_msg())
        assert "steps" in status.lower()
        assert "wall" in status.lower()
        assert "final url" in status.lower()
        assert "example.test" in status
    finally:
        _teardown(runtime, tmp)

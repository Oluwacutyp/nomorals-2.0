"""Viewable browser (gapfix124): watch + takeover over Devon's own browser.

Covers: the shared live-session registry (rendered + HTTP tabs, weak-ref
drop, find variants, driveable check, JSON-safe live view), the run-job
pause (job + manager + loop boundary wait), /browser watch/unwatch,
/browser takeover <session> (grant, park, watcher resume, hand_back),
the browser_watch brain tool (registry 787), and nm browse live.
All offline — fake tabs/services, a real TakeoverManager on a temp dir.
"""
from __future__ import annotations

import gc
import json
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from nomorals.agents.partner.runtime_provision import RuntimeProvisionMixin
from nomorals.agents.partner.runtime_reach import RuntimeReachMixin
from nomorals.browser import sessions as sess_mod
from nomorals.browser.agent import AgentLoop
from nomorals.browser.jobs import BrowserJobManager, BrowserRunJob
from nomorals.browser.liveview import LiveView
from nomorals.browser.sessions import (
    describe_sessions,
    find_live_tab,
    is_driveable_tab,
    live_sessions,
    live_view,
    register_service,
)
from nomorals.browser.takeover import TakeoverConfig, TakeoverManager
from nomorals.core.errors import ToolError
from nomorals.social.chat.base import ChatRef, SendResult

# -- fakes --------------------------------------------------------------------

class FakeRenderedTab:
    """A rendered tab: screenshot source + click_norm = driveable."""

    def __init__(self, session="browser-run-br-1",
                 url="https://example.test/shop",
                 title="Example Shop"):
        self.session_name = session
        self.tab_id = "tab-" + session
        self.url = url
        self.title = title
        self.clicks = []
        self.on_action = None

    def screenshot_bytes(self):
        return b"\x89PNG fake"

    def screenshot(self):
        return {"path": "/tmp/fake-shot.png"}

    def click_norm(self, x, y):
        self.clicks.append((x, y))

    def html(self):
        return {"html": "<html><title>Example Shop</title></html>"}

    def reload(self):
        pass


class FakeHttpTab:
    """A plain-HTTP tab: text only, no live page to drive."""

    def __init__(self, url="https://example.test/page", title="Example"):
        self.url = url
        self.title = title
        self.tab_id = "http-1"


class FakeHandle:
    """SessionHandle shape: active_tab is a @property, never a method."""

    def __init__(self, tab):
        self._tab = tab
        self._tabs = [tab]

    @property
    def active_tab(self):
        return self._tab


class FakeService:
    """A BrowserService shape: rendered tabs + HTTP sessions."""

    def __init__(self, label="fake-svc", rendered=None, sessions=None):
        self._mission_id = label
        self._rendered_tabs = dict(rendered or {})
        self._sessions = dict(sessions or {})


@contextmanager
def registered(service, **kw):
    """Register a fake service for one test, then drop it."""
    register_service(service, **kw)
    try:
        yield service
    finally:
        del service
        gc.collect()


def _rendered_service(name="browser-run-br-1", **tab_kw):
    tab = FakeRenderedTab(session=name, **tab_kw)
    return FakeService(rendered={tab.tab_id: tab}), tab


# -- registry -----------------------------------------------------------------

def test_registry_finds_rendered_tab():
    svc, tab = _rendered_service()
    with registered(svc):
        recs = live_sessions()
    mine = [r for r in recs if r["session"] == "browser-run-br-1"]
    assert len(mine) == 1
    assert mine[0]["kind"] == "rendered"
    assert mine[0]["url"] == "https://example.test/shop"
    assert mine[0]["tab"] is tab


def test_registry_reads_http_tab_via_property_not_method():
    # Regression: the old probe called handle.active_tab() as a method.
    # It is a @property; calling its value raises TypeError and the
    # outer except swallowed the whole takeover-service half.
    tab = FakeHttpTab()
    svc = FakeService(sessions={"default": FakeHandle(tab)})
    with registered(svc):
        recs = live_sessions()
    mine = [r for r in recs if r["tab"] is tab]
    assert len(mine) == 1
    assert mine[0]["kind"] == "http"


def test_registry_drops_dead_service():
    svc, _tab = _rendered_service(name="browser-run-br-dead")
    register_service(svc)
    assert any(r["session"] == "browser-run-br-dead"
               for r in live_sessions())
    del svc
    gc.collect()
    assert not any(r["session"] == "browser-run-br-dead"
                   for r in live_sessions())


def test_find_live_tab_variants():
    svc, tab = _rendered_service(name="browser-run-br-9",
                                 url="https://example.test/shoes",
                                 title="Shoe Store")
    with registered(svc):
        assert find_live_tab("browser-run-br-9")["tab"] is tab
        assert find_live_tab("fake-svc:browser-run-br-9")["tab"] is tab
        assert find_live_tab("tab-browser-run-br-9")["tab"] is tab
        assert find_live_tab("example.test/shoes")["tab"] is tab
        assert find_live_tab("shoe store")["tab"] is tab
        assert find_live_tab("no-such-session") is None
        assert find_live_tab("") is None


def test_is_driveable_tab():
    assert is_driveable_tab(FakeRenderedTab()) is True
    assert is_driveable_tab(FakeHttpTab()) is False
    tab = FakeRenderedTab()
    tab.click_norm = None
    assert is_driveable_tab(tab) is False
    assert is_driveable_tab(object()) is False


def test_describe_sessions_lists_display_url_kind():
    svc, _tab = _rendered_service()
    with registered(svc):
        text = describe_sessions(live_sessions())
    assert "browser-run-br-1" in text
    assert "https://example.test/shop" in text
    assert "(rendered)" in text
    assert describe_sessions([]) == "no live browser sessions right now."


def test_live_view_is_json_safe():
    svc, _tab = _rendered_service()
    with registered(svc):
        view = live_view()
    assert json.dumps(view)  # no live objects leak into the record
    assert view["sessions"]
    rec = view["sessions"][0]
    assert set(rec) == {"service", "session", "display", "tab_id",
                        "url", "title", "kind"}
    assert "live browser sessions" in view["live"]


def test_register_service_carries_pause_resume_hooks():
    calls = []
    svc, _tab = _rendered_service(name="browser-run-br-hook")
    with registered(svc, pause=lambda: calls.append("pause") or True,
                    resume=lambda: calls.append("resume") or True):
        rec = find_live_tab("browser-run-br-hook")
    assert rec is not None
    assert rec["pause"]() is True
    assert rec["resume"]() is True
    assert calls == ["pause", "resume"]


def test_registry_never_raises_on_broken_service():
    class BrokenService:
        _mission_id = "broken"

        @property
        def _rendered_tabs(self):
            raise RuntimeError("driver died")

        @property
        def _sessions(self):
            raise RuntimeError("driver died")

    svc = BrokenService()
    register_service(svc)
    try:
        live_sessions()  # must not raise
    finally:
        del svc
        gc.collect()


# -- run-job pause ------------------------------------------------------------

def _job(job_id="br-test"):
    return BrowserRunJob(job_id=job_id, url="https://example.test/",
                         goal="test")


def test_job_pause_resume_cycle():
    job = _job()
    assert job.paused() is False
    assert job.pause() is True
    assert job.paused() is True
    assert job.resume() is True
    assert job.paused() is False
    assert job.resume() is False  # never paused now


def test_job_pause_refused_when_terminal():
    job = _job()
    job.status = "done"
    assert job.pause() is False


def test_wait_while_paused_blocks_until_resume():
    job = _job()
    job.pause()
    out = []

    def _wait():
        out.append(job.wait_while_paused())

    thread = threading.Thread(target=_wait, daemon=True)
    thread.start()
    time.sleep(0.05)
    assert thread.is_alive()  # still parked
    job.resume()
    thread.join(timeout=2)
    assert out == [True]


def test_wait_while_paused_false_when_cancelled():
    job = _job()
    job.pause()
    out = []

    def _wait():
        out.append(job.wait_while_paused())

    thread = threading.Thread(target=_wait, daemon=True)
    thread.start()
    time.sleep(0.05)
    job._cancel_requested.set()
    thread.join(timeout=2)
    assert out == [False]


def test_manager_pause_resume_mirror_cancel():
    mgr = BrowserJobManager()
    assert mgr.pause("br-nope")["state"] == "not_found"
    assert mgr.resume("br-nope")["state"] == "not_found"
    job = _job("br-live")
    mgr._jobs[job.job_id] = job
    assert mgr.pause("br-live")["state"] == "paused"
    assert job.paused() is True
    assert mgr.resume("br-live")["state"] == "resumed"
    assert mgr.resume("br-live")["state"] == "not_paused"
    job.status = "done"
    assert mgr.pause("br-live")["state"] == "already_terminal"


# -- loop boundary wait -------------------------------------------------------

class _Driver:
    pass


def test_pause_wait_true_when_no_hook():
    loop = AgentLoop(_Driver())
    assert loop._pause_wait() is True


def test_pause_wait_false_when_hook_says_stop():
    loop = AgentLoop(_Driver(), wait_while_paused=lambda: False)
    assert loop._pause_wait() is False


def test_pause_wait_fail_open_on_raising_hook():
    def _boom():
        raise RuntimeError("pause read broke")

    loop = AgentLoop(_Driver(), wait_while_paused=_boom)
    assert loop._pause_wait() is True
    assert loop.wait_while_paused is None  # disarmed for the run
    assert loop._pause_wait() is True  # still fine without the hook


# -- /browser watch / unwatch -------------------------------------------------

class FakeAdapter:
    name = "fake"

    def __init__(self):
        self.sends = []

    def send(self, chat, text, **kw):
        self.sends.append(("text", text))
        return SendResult(ok=True, platform=self.name, message_id="11")

    def send_media(self, chat, media, *, caption=""):
        self.sends.append(("media", caption, media.path))
        return SendResult(ok=True, platform=self.name, message_id="22")

    def edit_media(self, chat, message_id, media, *, caption=""):
        return SendResult(ok=True, platform=self.name,
                          message_id=message_id)


def _watch_self(operator=True):
    state = {"watch_tabs": {}, "watch_views": {}}
    adapter = FakeAdapter()
    message = SimpleNamespace(
        chat=ChatRef(platform="fake", chat_id="1", kind="dm"))
    self = SimpleNamespace(
        _is_operator=lambda msg: operator,
        _provision_state=lambda: state,
        _adapter_for=lambda platform: adapter,
        _state=state, _adapter=adapter, _message=message)
    # The verb calls the mixin's own probe on self.
    self._watchable_sessions = (
        lambda: RuntimeProvisionMixin._watchable_sessions(self))
    return self


def test_watchable_sessions_sees_per_job_service():
    svc, _tab = _rendered_service(name="browser-run-br-watch")
    self = _watch_self()
    with registered(svc):
        got = RuntimeProvisionMixin._watchable_sessions(self)
    assert "browser-run-br-watch" in got


def test_watch_attaches_live_view(tmp_path):
    png = tmp_path / "shot.png"
    png.write_bytes(b"\x89PNG fake")
    svc, tab = _rendered_service(name="browser-run-br-watch2")
    tab.screenshot = lambda: {"path": str(png)}
    self = _watch_self()
    with registered(svc):
        out = RuntimeProvisionMixin._control_browser_watch(
            self, "browser-run-br-watch2", chat_key="c1",
            message=self._message)
    assert "live view attached" in out
    assert "browser-run-br-watch2" in out
    view = self._state["watch_views"]["c1"]
    assert isinstance(view, LiveView)
    assert self._adapter.sends  # first frame posted
    view.detach()


def test_watch_text_only_for_http_tab():
    tab = FakeHttpTab()
    svc = FakeService(sessions={"default": FakeHandle(tab)})
    self = _watch_self()
    with registered(svc):
        out = RuntimeProvisionMixin._control_browser_watch(
            self, "default", chat_key="c1", message=self._message)
    assert "live view attached" in out
    assert "Text updates only" in out
    self._state["watch_views"]["c1"].detach()


def test_watch_unknown_session_lists_live():
    svc, _tab = _rendered_service(name="browser-run-br-watch3")
    svc2, _tab2 = _rendered_service(name="browser-run-br-watch4")
    self = _watch_self()
    with registered(svc), registered(svc2):
        out = RuntimeProvisionMixin._control_browser_watch(
            self, "nope", chat_key="c1", message=self._message)
    assert "which session?" in out
    assert "browser-run-br-watch3" in out
    assert "browser-run-br-watch4" in out


def test_watch_no_sessions_honest_empty():
    self = _watch_self()
    out = RuntimeProvisionMixin._control_browser_watch(
        self, "", chat_key="c1", message=self._message)
    assert "no live browser session to watch right now" in out


def test_watch_non_operator_refused():
    self = _watch_self(operator=False)
    out = RuntimeProvisionMixin._control_browser_watch(
        self, "", chat_key="c1", message=self._message)
    assert "owner-only" in out


def test_unwatch_detaches_and_second_call_empty(tmp_path):
    png = tmp_path / "shot.png"
    png.write_bytes(b"\x89PNG fake")
    svc, tab = _rendered_service(name="browser-run-br-watch4")
    tab.screenshot = lambda: {"path": str(png)}
    self = _watch_self()
    with registered(svc):
        RuntimeProvisionMixin._control_browser_watch(
            self, "browser-run-br-watch4", chat_key="c1",
            message=self._message)
        out = RuntimeProvisionMixin._control_browser_unwatch(
            self, chat_key="c1", message=self._message)
    assert "live view detached" in out
    assert "c1" not in self._state["watch_views"]
    out2 = RuntimeProvisionMixin._control_browser_unwatch(
        self, chat_key="c1", message=self._message)
    assert "no live view attached" in out2


# -- /browser takeover <session> ----------------------------------------------

def _takeover_self(mgr, notifies):
    self = SimpleNamespace(
        _is_operator=lambda msg: True,
        _takeover_manager=lambda: mgr,
        _takeover_notifier=lambda: (lambda text, url: notifies.append(
            (text, url))))
    # The verb starts the grant watcher through the mixin.
    self._watch_takeover_grant = (
        lambda m, g, resume=None:
        RuntimeReachMixin._watch_takeover_grant(self, m, g, resume=resume))
    return self


def _takeover_verb(self, ref, **kw):
    return RuntimeReachMixin._control_browser_takeover_session(
        self, ref, chat_key="owner-chat",
        message=SimpleNamespace(), **kw)


def test_takeover_session_unknown_lists_live():
    svc, _tab = _rendered_service(name="browser-run-br-tk1")
    mgr = TakeoverManager(TakeoverConfig(), data_dir=None)
    self = _takeover_self(mgr, [])
    with registered(svc):
        out = _takeover_verb(self, "nope")
    assert "no live session 'nope'" in out
    assert "browser-run-br-tk1" in out


def test_takeover_session_http_tab_refused():
    tab = FakeHttpTab(url="https://example.test/page")
    svc = FakeService(sessions={"default": FakeHandle(tab)})
    mgr = TakeoverManager(TakeoverConfig(), data_dir=None)
    self = _takeover_self(mgr, [])
    with registered(svc):
        out = _takeover_verb(self, "default")
    assert "no live page to drive" in out
    assert mgr.active_grants() == []


def test_takeover_session_issues_grant_and_parks_job(tmp_path):
    calls = []
    svc, tab = _rendered_service(name="browser-run-br-tk2")
    cfg = TakeoverConfig(poll_s=0.05, wait_timeout_s=30)
    mgr = TakeoverManager(cfg, data_dir=tmp_path / "takeover")
    notifies = []
    self = _takeover_self(mgr, notifies)
    with registered(svc, pause=lambda: calls.append("pause") or True,
                    resume=lambda: calls.append("resume") or True):
        out = _takeover_verb(self, "browser-run-br-tk2")
    assert out.startswith("takeover open: http://127.0.0.1:")
    assert "Devon parked the run at its next step" in out
    assert calls == ["pause"]
    assert len(mgr.active_grants()) == 1
    assert len(notifies) == 1  # link also sent to Telegram
    assert notifies[0][1].startswith("http://127.0.0.1:")
    # Owner taps Done -> watcher resumes the parked run.
    grant = next(iter(mgr._grants.values()))
    mgr.mark_done(grant, note="done driving")
    deadline = time.time() + 5
    while "resume" not in calls and time.time() < deadline:
        time.sleep(0.05)
    assert "resume" in calls


def test_takeover_session_already_active(tmp_path):
    svc, _tab = _rendered_service(name="browser-run-br-tk3")
    mgr = TakeoverManager(TakeoverConfig(poll_s=0.05),
                          data_dir=tmp_path / "takeover")
    self = _takeover_self(mgr, [])
    with registered(svc):
        first = _takeover_verb(self, "browser-run-br-tk3")
        second = _takeover_verb(self, "browser-run-br-tk3")
    assert first.startswith("takeover open: http://")
    assert "already active" in second
    assert "expires in" in second


def test_hand_back_closes_grant_and_is_idempotent(tmp_path):
    mgr = TakeoverManager(TakeoverConfig(), data_dir=tmp_path / "takeover")
    tab = FakeRenderedTab()
    grant = mgr.request_takeover(
        session_id="s-handback", url=tab.url, task_label="t",
        signal_kind="owner_takeover", attempts=[],
        resume_state={"step": 0, "url": tab.url},
        tab=tab, owner_chat="")
    assert mgr.hand_back(grant, note="expired") is True
    assert grant.closed is True
    assert mgr.hand_back(grant, note="again") is True  # no-op


# -- brain tool ---------------------------------------------------------------

def test_tool_count_is_787_after_browser_watch():
    from types import SimpleNamespace as NS

    from nomorals.tools.registry import ToolRegistry

    registry = ToolRegistry(NS())
    registry.register_builtins()
    # 786 after audit N4 added build_swarm; 801 after gapfix124 added
    # browser_watch.
    assert len(registry._tools) == 810
    assert registry.get("browser_watch") is not None


def test_browser_watch_tool_list_takeover_status(tmp_path):
    from types import SimpleNamespace as NS

    from nomorals.tools.registry import ToolRegistry

    svc, _tab = _rendered_service(name="browser-run-br-tool")
    settings = NS(resolve=lambda p: str(tmp_path / p))
    registry = ToolRegistry(NS(settings=settings))
    registry.register_builtins()
    tool = registry.get("browser_watch")
    assert tool is not None
    with registered(svc):
        listed = tool.fn(action="list")
        assert "browser-run-br-tool" in listed["sessions"]
        status = tool.fn(action="status")
        assert status["active_takeovers"] == []
        taken = tool.fn(action="takeover", session="browser-run-br-tool")
    assert taken["ok"] is True
    assert taken["takeover_url"].startswith("http://127.0.0.1:")
    assert taken["parked_run"] is False  # no job drives this fake tab
    assert "Hand this link to the owner" in taken["note"]
    with pytest.raises(ToolError):
        tool.fn(action="bogus")


# -- CLI: nm browse live ------------------------------------------------------

def test_daemon_live_op_lists_registry_json_safe():
    from nomorals.browser.daemon import _handle_op

    svc, _tab = _rendered_service(name="browser-run-br-cli")
    with registered(svc):
        result = _handle_op(svc, "live", {})
    assert json.dumps(result)  # JSON-safe for --json
    assert "browser-run-br-cli" in result["live"]
    assert result["sessions"][0]["kind"] == "rendered"


def test_browse_live_command_prints_live_view(monkeypatch, capsys):
    import nomorals.cmdline.commands.browse as browse_mod

    fake_backend = SimpleNamespace(
        live=lambda: {"live": "live browser sessions (1):\n  x — y (rendered)",
                      "sessions": [{"session": "x"}]})
    monkeypatch.setattr(browse_mod, "_backend", lambda a, c: fake_backend)
    args = SimpleNamespace(json=False)
    assert browse_mod._browse_live(args, None) == 0
    out = capsys.readouterr().out
    assert "live browser sessions (1)" in out


def test_browse_live_command_json(monkeypatch, capsys):
    import nomorals.cmdline.commands.browse as browse_mod

    payload = {"live": "no live browser sessions right now.",
               "sessions": []}
    fake_backend = SimpleNamespace(live=lambda: payload)
    monkeypatch.setattr(browse_mod, "_backend", lambda a, c: fake_backend)
    args = SimpleNamespace(json=True)
    assert browse_mod._browse_live(args, None) == 0
    out = capsys.readouterr().out
    assert json.loads(out) == payload


# -- layering -----------------------------------------------------------------

def test_sessions_module_is_stdlib_only():
    import ast
    from pathlib import Path

    tree = ast.parse(Path(sess_mod.__file__).read_text())
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    # stdlib + the one core logging import the browser layer already
    # uses; nothing from agents/ or tools/ (browser/ is L4).
    assert imports <= {"__future__", "threading", "weakref", "typing",
                       "core"}, imports
    text = Path(sess_mod.__file__).read_text()
    assert "from ..tools" not in text
    assert "from ...agents" not in text

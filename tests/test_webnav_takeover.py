"""The browser handoff link (owner's standing design,
2026-10-19/21): when Devon hits a wall she cannot clear,
she sends the owner a LINK into the SAME live session —
he watches it, taps/types into it, hands it back, and
she re-reads the page and resumes.

Pinned here, at the seams this box can fake (no
Playwright installed — the probe/page seam is faked
exactly the way test_webnav.py fakes it):

- the wall pass ends its hold on the owner's explicit
  hand-back: state re-read first — clear means resume,
  still-walled means an honest WallBlocked sentence
  (Done alone is not a clear);
- the takeover bridge registers the LIVE NavSession
  with the takeover manager (the grant's tab drives
  that session's real page) and delivers the
  plain-language message carrying the handoff link;
- owner input through the tab adapter lands on the
  session's page (normalized click scaled to the
  viewport, typing, keys, scroll, frames);
- a full NavSession.open over a walled page: the grant
  is issued mid-open, the owner's tap in the live view
  clears the wall, the open completes on the real page;
- the REAL TakeoverManager's token discipline holds
  through the bridge: wrong token refused, the link's
  token resolves, Done kills the link.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from nomorals.browser.takeover import TakeoverConfig, TakeoverManager
from nomorals.social.chat import webnav
from nomorals.social.chat.webnav import (
    NavTabAdapter,
    TakeoverAsk,
    TakeoverBridge,
    WallBlocked,
    format_takeover_message,
    run_wall_pass,
)

GRID_PAGE = """<html><head><title>Security Check</title>
</head><body><h1>Verify you are human</h1>
<div class="rc-imageselect"><p>Select all images with
buses</p><div class="rc-image-tile-wrapper"></div></div>
<div class="g-recaptcha" data-sitekey="x"></div>
</body></html>"""

CONTENT_PAGE = """<html><head><title>Real Page</title></head>
<body><h1>The actual content</h1>
<p>behind the wall.</p></body></html>"""

URL = "https://mgamemax.example/article"


# ── fakes ─────────────────────────────────────────────────


class FakeProbe:
    """test_webnav's scripted probe shape, with a wait
    hook so a test can act as the owner mid-hold."""

    def __init__(self, html, *, url=URL, title="Real Page",
                 on_wait=None):
        self._html = html
        self._url = url
        self._title = title
        self._on_wait = on_wait
        self.waits = []

    def url(self):
        return self._url

    def title(self):
        return self._title

    def html(self):
        return self._html

    def reload(self):
        pass

    def wait(self, seconds):
        self.waits.append(seconds)
        if self._on_wait is not None:
            cb, self._on_wait = self._on_wait, None
            cb()

    def click_gate(self, gate):
        return False

    def click_checkbox(self):
        return False


class FakeGrant:
    def __init__(self, url="https://host.test/takeover/TOK3N"):
        self.url = url
        self.done = False
        self.needs_public_base = False


class FakeManager:
    """The TakeoverManager seam: records every grant
    request; mark_done mirrors the real manager."""

    def __init__(self):
        self.requests = []
        self.grant = None

    def request_takeover(self, **kw):
        self.requests.append(kw)
        self.grant = FakeGrant()
        return self.grant

    def mark_done(self, grant, note=""):
        grant.done = True


class _Mouse:
    def __init__(self, page):
        self._page = page

    def click(self, x, y):
        self._page.clicks.append((x, y))
        # The owner's tap lands on the challenge control:
        # the wall comes down.
        self._page.walled = False

    def wheel(self, dx, dy):
        self._page.wheels.append((dx, dy))


class _Keyboard:
    def __init__(self, page):
        self._page = page

    def type(self, text):
        self._page.typed.append(text)

    def press(self, key):
        self._page.keys.append(key)


class FakePage:
    """A Playwright page double: walled until the owner
    clicks; stable geometry for settle + adapter math."""

    def __init__(self):
        self.walled = True
        self.url = URL
        self.clicks = []
        self.typed = []
        self.keys = []
        self.wheels = []
        self.mouse = _Mouse(self)
        self.keyboard = _Keyboard(self)

    def goto(self, url, **kw):
        self.url = url

    def wait_for_load_state(self, *a, **kw):
        pass

    def wait_for_timeout(self, ms):
        pass

    def evaluate(self, script):
        if "innerWidth" in script:
            return {"w": 1000, "h": 800}
        return [100, 100]

    def screenshot(self, **kw):
        return b"\x89PNG-fake-frame"


class SharedProbe:
    """A PageProbe reading the FakePage's live state —
    the wall the pipeline sees is the page the owner
    is driving."""

    def __init__(self, page, on_wait=None):
        self._page = page
        self._on_wait = on_wait

    def url(self):
        return self._page.url

    def title(self):
        return "Security Check" if self._page.walled \
            else "Real Page"

    def html(self):
        return GRID_PAGE if self._page.walled \
            else CONTENT_PAGE

    def reload(self):
        pass

    def wait(self, seconds):
        if self._on_wait is not None:
            cb, self._on_wait = self._on_wait, None
            cb()

    def click_gate(self, gate):
        return False

    def click_checkbox(self):
        return False


def _ask():
    return TakeoverAsk(url=URL, wall="hard",
                       attempts=["managed wait-out",
                                 "checkbox click"])


# ── the wall pass: hand-back ends the hold ────────────────


def test_wall_pass_handback_after_clear_resumes():
    state = {"t": 0.0, "done": False}
    probe = FakeProbe(GRID_PAGE, title="Security Check")

    def wait(seconds):
        state["t"] += seconds
        probe._html = CONTENT_PAGE
        probe._title = "Real Page"
        state["done"] = True

    probe.wait = wait
    via = run_wall_pass(
        probe, url=URL, takeover_hook=lambda ask: True,
        takeover_done=lambda: state["done"],
        takeover_budget_s=30.0, clock=lambda: state["t"])
    assert via == "owner_takeover"


def test_wall_pass_handback_with_wall_up_is_honest():
    state = {"t": 0.0}
    probe = FakeProbe(GRID_PAGE, title="Security Check")
    probe.wait = lambda s: state.__setitem__(
        "t", state["t"] + s)
    with pytest.raises(WallBlocked) as err:
        run_wall_pass(
            probe, url=URL, takeover_hook=lambda ask: True,
            takeover_done=lambda: state["t"] >= 2.0,
            takeover_budget_s=30.0,
            clock=lambda: state["t"])
    assert "handed the session back" in str(err.value)
    assert "still up" in str(err.value)


# ── the tab adapter: owner input lands on the page ───────


def _adapter():
    page = FakePage()
    session = SimpleNamespace(
        probe=FakeProbe(CONTENT_PAGE),
        page=page, capture_still=lambda: b"PNG-BYTES")
    return NavTabAdapter(session), page


def test_adapter_click_scales_normalized_coords():
    adapter, page = _adapter()
    adapter.click_norm(0.5, 0.25)
    assert page.clicks == [(500, 200)]
    adapter.click_norm(1.5, -0.5)  # clamped to the viewport
    assert page.clicks[-1] == (1000, 0)


def test_adapter_type_key_scroll_and_frame():
    adapter, page = _adapter()
    adapter.type_text("hello")
    adapter.press_key("Enter")
    adapter.scroll_norm(0.0, 640.0)
    assert page.typed == ["hello"]
    assert page.keys == ["Enter"]
    assert page.wheels == [(0.0, 640.0)]
    assert adapter.screenshot_png() == b"PNG-BYTES"
    assert adapter.url == URL
    assert "actual content" in adapter.html()


# ── the bridge: grant + link + hand-back ──────────────────


def test_bridge_registers_live_session_and_sends_link():
    manager = FakeManager()
    page = FakePage()
    session = SimpleNamespace(probe=FakeProbe(CONTENT_PAGE),
                              page=page,
                              capture_still=lambda: b"PNG")
    sent = []
    bridge = TakeoverBridge(
        manager, session,
        notify=lambda text: sent.append(text) or True,
        owner_chat="chat-1", task_label="browse")
    assert bridge.hook(_ask()) is True
    # the grant was requested for THIS session: its tab
    # adapter drives this session's page
    assert len(manager.requests) == 1
    req = manager.requests[0]
    assert req["owner_chat"] == "chat-1"
    assert req["task_label"] == "browse"
    assert req["signal_kind"] == "hard"
    tab = req["tab"]
    assert isinstance(tab, NavTabAdapter)
    tab.click_norm(0.5, 0.5)
    assert page.clicks == [(500, 400)]
    # the delivered message carries the link, in plain
    # words, naming the site
    assert len(sent) == 1
    text = sent[0]
    assert manager.grant.url in text
    assert "take over my browser" in text
    assert "hand it back" in text
    assert "mgamemax.example" in text
    assert "{'ok':" not in text
    # hand-back state flows from the grant
    assert bridge.done() is False
    manager.mark_done(manager.grant)
    assert bridge.done() is True


def test_bridge_undelivered_notice_reports_false():
    manager = FakeManager()
    session = SimpleNamespace(probe=FakeProbe(CONTENT_PAGE),
                              page=FakePage(),
                              capture_still=lambda: b"PNG")
    bridge = TakeoverBridge(manager, session,
                            notify=lambda text: False)
    assert bridge.hook(_ask()) is False
    assert bridge.grant is not None  # the grant exists;
    # only the notice failed — the pipeline decides


def test_bridge_grant_failure_reports_false_not_raise():
    class BoomManager:
        def request_takeover(self, **kw):
            raise RuntimeError("bind failed")

    session = SimpleNamespace(probe=FakeProbe(CONTENT_PAGE),
                              page=FakePage(),
                              capture_still=lambda: b"PNG")
    bridge = TakeoverBridge(BoomManager(), session,
                            notify=lambda text: True)
    assert bridge.hook(_ask()) is False
    assert "bind failed" in bridge.error


def test_format_takeover_message_names_wall_and_link():
    grant = FakeGrant()
    text = format_takeover_message(_ask(), grant)
    assert grant.url in text
    assert "wall" in text and "hard" in text
    assert "managed wait-out" in text  # what was tried
    grant.needs_public_base = True
    text = format_takeover_message(_ask(), grant)
    assert "NM_TAKEOVER_BASE_URL" in text


# ── the full open: grant mid-flight, owner clears, resume ──


def _walled_session(on_wait):
    session = webnav.NavSession()
    page = FakePage()
    session._page = page
    session.probe = SharedProbe(page, on_wait=on_wait)
    return session, page


def test_session_open_takeover_owner_clears_and_resumes():
    manager = FakeManager()
    sent = []
    holder = {}

    def owner_acts():
        # The owner opens the link and taps the challenge
        # in the live view — through the grant's own tab.
        tab = manager.requests[0]["tab"]
        tab.click_norm(0.5, 0.5)

    session, page = _walled_session(owner_acts)
    bridge = TakeoverBridge(
        manager, session,
        notify=lambda text: sent.append(text) or True,
        task_label="browse")
    outcome = session.open(URL, takeover_bridge=bridge,
                           takeover_budget_s=30.0)
    assert outcome.wall == "owner_takeover"
    # the session re-read the page after the takeover:
    # the content behind the wall is what the open holds
    assert session.probe.html() == CONTENT_PAGE
    assert outcome.title == "Real Page"
    # the grant's tab was bound to THIS session
    assert manager.requests[0]["tab"]._session is session
    assert page.clicks == [(500, 400)]
    assert manager.grant.url in sent[0]


def test_session_open_takeover_handback_wall_up_is_honest():
    manager = FakeManager()
    sent = []

    def owner_acts():
        # The owner looks, does not clear it, hands back.
        manager.mark_done(manager.grant)

    session, page = _walled_session(owner_acts)
    bridge = TakeoverBridge(
        manager, session,
        notify=lambda text: sent.append(text) or True,
        task_label="browse")
    with pytest.raises(WallBlocked) as err:
        session.open(URL, takeover_bridge=bridge,
                     takeover_budget_s=30.0)
    assert "handed the session back" in str(err.value)
    assert manager.grant.url in sent[0]


# ── the race: parallel rungs, bounded sessions ────────────


class _RaceSession:
    def __init__(self, name):
        self.name = name
        self.closed = False

    def __exit__(self, *a):
        self.closed = True


def _pres(name):
    return {"name": name}


def test_race_runs_rungs_concurrently():
    # Rung A cannot finish until rung B has STARTED:
    # on the old serial walk B starts only after A
    # ends, the overlap flag never sets, and this
    # test's winner/timing assertions fail.
    import threading

    started_b = threading.Event()
    overlap = []

    def open_one(pres, hook):
        if pres["name"] == "a-slow":
            overlap.append(started_b.wait(timeout=10))
            raise webnav.WallBlocked(
                "wall on a", wall="hard", attempts=[])
        started_b.set()
        return _RaceSession("b"), webnav.NavOutcome(
            final_url=URL, title="Real Page")

    _session, outcome = webnav.open_with_ladder(
        URL, [_pres("a-slow"), _pres("b-fast")], open_one)
    assert overlap == [True]
    assert outcome.presentation == "b-fast"
    assert outcome.rung_outcomes["a-slow"] == "wall_blocked"


def test_race_first_clearer_wins_and_loser_closes():
    import threading
    import time as _time

    sessions = {}

    def open_one(pres, hook):
        sess = _RaceSession(pres["name"])
        sessions[pres["name"]] = sess
        if pres["name"] == "slow-clearer":
            _time.sleep(0.8)
        return sess, webnav.NavOutcome(
            final_url=URL, title="Real Page")

    session, outcome = webnav.open_with_ladder(
        URL, [_pres("slow-clearer"), _pres("fast-clearer")],
        open_one)
    assert outcome.presentation == "fast-clearer"
    assert session is sessions["fast-clearer"]
    assert sessions["fast-clearer"].closed is False
    # the late clearer was superseded and closed
    deadline = _time.time() + 5
    while not sessions["slow-clearer"].closed \
            and _time.time() < deadline:
        _time.sleep(0.02)
    assert sessions["slow-clearer"].closed is True
    assert outcome.rung_outcomes["slow-clearer"] in (
        "cleared_late", "superseded")


def test_held_takeover_does_not_block_second_job():
    # Job A parks on an owner takeover; job B — a
    # separate capture — completes while A still holds;
    # A then resolves on hand-back alone.
    import threading

    done_a = threading.Event()
    a_holding = threading.Event()
    results = {}

    def opener_a(pres, hook):
        probe = FakeProbe(GRID_PAGE,
                          title="Security Check")
        wall = run_wall_pass(
            probe, url=URL, takeover_hook=hook,
            takeover_done=done_a.is_set,
            takeover_budget_s=60.0)
        return _RaceSession("a"), webnav.NavOutcome(
            final_url=URL, title="Real Page", wall=wall)

    def opener_b(pres, hook):
        return _RaceSession("b"), webnav.NavOutcome(
            final_url=URL, title="Real Page")

    def job_a():
        try:
            results["a"] = webnav.open_with_ladder(
                URL, [_pres("only-a")], opener_a,
                takeover_hook=lambda ask:
                    a_holding.set() or True)
        except Exception as exc:  # noqa: BLE001 - the
            results["a_error"] = exc  # resolution IS data

    thread = threading.Thread(target=job_a, daemon=True)
    thread.start()
    assert a_holding.wait(timeout=10), \
        "job A never reached its takeover hold"
    _sess_b, outcome_b = webnav.open_with_ladder(
        URL, [_pres("only-b")], opener_b)
    assert outcome_b.presentation == "only-b"
    assert "a" not in results and "a_error" not in results, \
        "job A resolved before its owner handed back"
    # The owner hands back with the wall still up: A
    # resolves on its own — honestly (the wall verdict),
    # and it never blocked B either way.
    done_a.set()
    thread.join(timeout=15)
    assert not thread.is_alive()
    assert isinstance(results.get("a_error"),
                      webnav.WallBlocked)
    assert "handed the session back" in str(
        results["a_error"])


def test_session_slots_are_bounded(monkeypatch):
    import threading

    monkeypatch.setenv("NM_WEBNAV_MAX_SESSIONS", "7")
    assert webnav.max_nav_sessions() == 7
    monkeypatch.delenv("NM_WEBNAV_MAX_SESSIONS")
    monkeypatch.setattr(webnav, "max_nav_sessions",
                        lambda: 1)
    webnav._nav_slot_semaphores.pop(1, None)
    first = webnav.NavSession()
    second = webnav.NavSession()
    first._acquire_slot()
    acquired = threading.Event()

    def grab():
        second._acquire_slot()
        acquired.set()

    thread = threading.Thread(target=grab, daemon=True)
    thread.start()
    assert not acquired.wait(timeout=1.0), \
        "the second session took a slot past the bound"
    first._release_slot()
    assert acquired.wait(timeout=5.0)
    second._release_slot()
    webnav._nav_slot_semaphores.pop(1, None)


# ── the real manager: token discipline through the bridge ──


def test_real_manager_token_discipline():
    now = [1000.0]
    manager = TakeoverManager(
        TakeoverConfig(link_ttl_s=60), clock=lambda: now[0])
    try:
        sent = []
        page = FakePage()
        session = SimpleNamespace(
            probe=FakeProbe(CONTENT_PAGE), page=page,
            capture_still=lambda: b"PNG")
        bridge = TakeoverBridge(
            manager, session,
            notify=lambda text: sent.append(text) or True,
            task_label="browse")
        assert bridge.hook(_ask()) is True
        grant = bridge.grant
        assert grant is not None
        token = grant.url.rsplit("/", 1)[-1]
        assert token and token not in sent[0].split(
            grant.url)[0]  # the token rides only in the link
        assert manager.validate_token("not-the-token") \
            is None
        assert manager.validate_token(token) is grant
        # the audit-facing surface never carries the token
        active = manager.active_grants()
        assert active and token not in str(active)
        # Done kills the link
        manager.mark_done(grant)
        assert bridge.done() is True
        assert manager.validate_token(token) is None
        # a fresh grant dies at its TTL
        assert bridge.hook(_ask()) is True
        token2 = bridge.grant.url.rsplit("/", 1)[-1]
        assert manager.validate_token(token2) is not None
        now[0] += 3600.0
        assert manager.validate_token(token2) is None
    finally:
        manager.shutdown()

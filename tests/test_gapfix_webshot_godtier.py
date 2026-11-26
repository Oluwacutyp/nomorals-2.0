"""Webshot god-tier (gapfix109): the missing rungs, pinned
offline against a fake Playwright page that models the
real driver's API — no live internet, no real browser in
this file (the real-Chromium proof lives in prove.py).

- scrolling capture: a stitched composite of real viewport
  segments, in order, at the page's true height.
- flow capture: landing + click/goto steps, gates re-run
  per stop, an unfollowable step stops honestly with the
  shots so far.
- long capture: /webrec long (300 s ceiling) + frames
  mode (stills at intervals + contact sheet).
- bot/captcha evidence: a refusal carries the challenge
  page (screenshot + DOM excerpt); the handler delivers
  it labelled as a block, never as the shot.
- age-gate click fixes (consent-doc §2): the original
  label is what gets clicked (curly apostrophes included),
  a class-only hint never clicks the wrong button, a
  hidden dialog is not a standing gate, a slow clear gets
  a bounded wait, ARIA role=button controls are seen,
  remembered consent never suppresses a live gate, and a
  click failure keeps its reason.
- no new brain tool: the registry stays at 789.
"""
from __future__ import annotations

import io
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from nomorals.agents.partner.runtime_group import RuntimeGroupMixin
from nomorals.social.chat import webnav, webshot
from nomorals.social.chat.base import ChatKind
from nomorals.social.chat.webnav import (
    BlockEvidence,
    GateCandidate,
    GateMemory,
    NavOutcome,
    WallBlocked,
    _SessionOwner,
    _ThreadPage,
    capture_block_evidence,
    detect_gates,
    run_interstitial_pass,
)
from nomorals.social.chat.webshot import (
    CaptureResult,
    ShotOptions,
    WebshotUnavailable,
)
from nomorals.tools.registry import ToolRegistry


def _png(width=8, height=8, color=(200, 30, 30)):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, "PNG")
    return buf.getvalue()


# ── the fake Playwright page ──────────────────────────────


class FakeClock:
    """Monotonic clock the fake page advances: waits are
    instant in fake time, so duration tests run at full
    speed without spinning on the real clock."""

    def __init__(self, t0: float = 1000.0):
        self.now = t0

    def __call__(self) -> float:
        return self.now


class FakeLocator:
    """A Playwright locator over fake controls."""

    def __init__(self, page, controls):
        self._page = page
        self._controls = list(controls)

    @property
    def first(self):
        return FakeLocator(self._page, self._controls[:1])

    def count(self):
        return len(self._controls)

    def is_visible(self):
        return any(c.get("visible", True)
                   for c in self._controls)

    def click(self, timeout=None):
        for c in self._controls:
            if c.get("visible", True):
                self._page.clicked.append(c["name"])
                if c.get("on_click") is not None:
                    c["on_click"]()
                return
        raise RuntimeError("no visible control to click")

    def screenshot(self, timeout=None):
        return _png(8, 8)

    def inner_text(self):
        if not self._controls:
            return ""
        return self._controls[0].get("name", "")


class FakePage:
    """A Playwright page over a modelled tall page: the
    pixels encode the y-band, so a stitched composite can
    be verified band by band, not just by size."""

    def __init__(self, *, page_height=3000, viewport=900,
                 viewport_w=390, scale=1, url="https://x.example/",
                 title="Real Page", html="<html></html>",
                 controls=(), routes=None, clock=None):
        self._page_height = page_height
        self._viewport = viewport
        self._viewport_w = viewport_w
        self._scale = scale
        self._url = url
        self._title = title
        self._html = html
        self.controls = list(controls)
        self.routes = dict(routes or {})
        self.clicked: list[str] = []
        self.waits: list[float] = []
        self.scrolls: list[float] = []
        self._scroll_y = 0.0
        self._clock = clock
        self._tall = self._build_tall()
        self.main_frame = self
        self.frames = [self]
        self.video = None

    def _build_tall(self):
        img = Image.new(
            "RGB", (self._viewport_w * self._scale,
                    max(1, self._page_height * self._scale)))
        px = img.load()
        for y in range(img.height):
            css_y = y // self._scale
            color = (css_y % 256, (css_y // 256) % 256, 128)
            for x in range(img.width):
                px[x, y] = color
        return img

    @property
    def url(self):
        return self._url

    def title(self):
        return self._title

    def content(self):
        return self._html

    def wire_links(self):
        """Give link controls their navigation: a click on
        a named link follows it, the way a real page does."""
        link_map = getattr(self, "link_map", {})
        for c in self.controls:
            href = link_map.get(c["name"])
            if href is not None:
                c["on_click"] = (
                    lambda href=href: self.goto(href))

    def goto(self, url, wait_until=None, timeout=None):
        self._url = url
        route = self.routes.get(url)
        if route is not None:
            self._title = route.get("title", self._title)
            self._html = route.get("html", self._html)
            self._page_height = route.get(
                "page_height", self._page_height)
            self.controls = list(route.get("controls", []))
            self._tall = self._build_tall()
            self.wire_links()
        self._scroll_y = 0.0

    def reload(self, wait_until=None, timeout=None):
        pass

    def wait_for_timeout(self, ms):
        self.waits.append(ms)
        if self._clock is not None:
            self._clock.now += ms / 1000.0

    def wait_for_load_state(self, state, timeout=None):
        pass

    def evaluate(self, js):
        if "scrollHeight" in js and "innerHeight" in js \
                and "outerHTML" not in js:
            return [self._page_height, self._viewport]
        if "outerHTML" in js:
            return [len(self._html), self._page_height]
        if "scrollHeight" in js:
            return self._page_height
        if "scrollY" in js:
            return self._scroll_y
        if "innerWidth" in js:
            return {"w": self._viewport_w,
                    "h": self._viewport}
        if "scrollTo" in js:
            import re

            m = re.search(r"scrollTo\(0,\s*([0-9.]+)\)", js)
            if m:
                self._scroll_y = max(
                    0.0, min(float(m.group(1)),
                             max(0.0, self._page_height
                                 - self._viewport)))
                self.scrolls.append(self._scroll_y)
            return None
        return None

    def screenshot(self, **kwargs):
        if kwargs.get("full_page"):
            buf = io.BytesIO()
            self._tall.save(buf, "PNG")
            return buf.getvalue()
        top = int(self._scroll_y * self._scale)
        bottom = min(top + self._viewport * self._scale,
                     self._tall.height)
        buf = io.BytesIO()
        self._tall.crop(
            (0, top, self._tall.width, bottom)).save(
                buf, "PNG")
        return buf.getvalue()

    def pdf(self, **kwargs):
        return b"%PDF-fake"

    def _match(self, controls, role, name):
        want = (name or "").lower()
        out = []
        for c in controls:
            if role is not None and c.get("role") != role:
                continue
            if want and want not in c["name"].lower():
                continue
            out.append(c)
        return out

    def get_by_role(self, role, name=None, exact=False):
        return FakeLocator(
            self, self._match(self.controls, role, name))

    def locator(self, selector):
        if selector.startswith("text="):
            want = selector[len("text="):].lower()
            return FakeLocator(
                self, [c for c in self.controls
                       if want in c["name"].lower()])
        if selector.startswith("#"):
            ident = selector[1:]
            return FakeLocator(
                self, [c for c in self.controls
                       if c.get("id") == ident])
        if selector.startswith("."):
            cls = selector[1:].split(".")
            return FakeLocator(
                self, [c for c in self.controls
                       if all(x in (c.get("class") or "")
                              .split() for x in cls)])
        return FakeLocator(self, [])

    @property
    def mouse(self):
        return SimpleNamespace(
            click=lambda x, y: None,
            wheel=lambda dx, dy: None)

    @property
    def keyboard(self):
        return SimpleNamespace(type=lambda t: None,
                               press=lambda k: None)


def _fake_session(page, *, clock=None):
    """A real NavSession over the fake page: __enter__ /
    __exit__ are instance no-ops (no real browser), every
    other method is the production one."""
    session = webnav.NavSession(
        device=dict(webnav.PRESENTATIONS["android-phone"]))
    session._page = page
    session.probe = webnav._PlaywrightProbe(page)
    session._started_at = (clock.now if clock is not None
                           else time.monotonic())
    session.__enter__ = lambda *a, **k: session
    session.__exit__ = lambda *a, **k: None
    return session


# ── grammar: scroll / flow / long / frames ────────────────


def test_shot_grammar_scroll_and_flow():
    url, opts = webshot.parse_shot_tail(
        "scroll full phone https://x.example/")
    assert url == "https://x.example/"
    assert opts.scroll is True and opts.full_page is True
    assert opts.device == "phone"

    url, opts = webshot.parse_shot_tail(
        "flow https://x.example/ -> Docs -> Install")
    assert url == "https://x.example/"
    assert opts.flow_steps == ("Docs", "Install")

    url, opts = webshot.parse_shot_tail(
        "flow https://x.example/ -> "
        "https://x.example/docs")
    assert opts.flow_steps == ("https://x.example/docs",)


def test_shot_grammar_flow_errors():
    with pytest.raises(webshot.ShotUsageError):
        webshot.parse_shot_tail("flow https://x.example/")
    with pytest.raises(webshot.ShotUsageError):
        webshot.parse_shot_tail(
            "https://x.example/ -> Docs")
    with pytest.raises(webshot.ShotUsageError):
        webshot.parse_shot_tail(
            "flow https://x.example/ ->  -> Docs")


def test_rec_grammar_long_and_frames():
    _, opts = webshot.parse_rec_tail(
        "long 200s https://x.example/")
    assert opts.seconds == 200.0 and opts.long is True
    assert opts.clamped == ""
    _, opts = webshot.parse_rec_tail(
        "long 400s https://x.example/")
    assert opts.seconds == 300.0 and opts.clamped == "max"
    _, opts = webshot.parse_rec_tail(
        "frames 10s https://x.example/")
    assert opts.frames is True and opts.seconds == 10.0
    # Without long, the 60 s ceiling still stands.
    _, opts = webshot.parse_rec_tail("90s https://x.example/")
    assert opts.seconds == 60.0 and opts.clamped == "max"
    _, opts = webshot.parse_rec_tail(
        "long 90s https://x.example/")
    assert opts.seconds == 90.0 and opts.clamped == ""


# ── scrolling capture: stitched, in order, true height ───


def test_capture_scrolled_stitches_in_order():
    page = FakePage(page_height=3000, viewport=900,
                    viewport_w=390, scale=1)
    session = _fake_session(page)
    shot = session.capture_scrolled(settle_ms=0)
    assert shot.segments == 4
    assert shot.page_height == 3000
    assert shot.viewport_height == 900
    assert shot.reached_bottom is True
    # The last scroll is CLAMPED to scrollHeight −
    # viewport (2100), then the page scrolls home.
    assert page.scrolls == [0.0, 900.0, 1800.0, 2100.0, 0.0]
    width, height = webshot.image_dimensions(shot.data)
    assert (width, height) == (390, 3000)
    with Image.open(io.BytesIO(shot.data)) as img:
        # The pixel colour encodes the CSS y-band: the
        # composite must read top-to-bottom, not shuffled.
        assert img.getpixel((10, 100)) == (100, 0, 128)
        assert img.getpixel((10, 1500)) == (1500 % 256, 5, 128)
        assert img.getpixel((10, 2900)) == (2900 % 256, 11, 128)


def test_capture_scrolled_scales_device_pixels():
    page = FakePage(page_height=3000, viewport=844,
                    viewport_w=390, scale=2)
    session = _fake_session(page)
    shot = session.capture_scrolled(settle_ms=0)
    width, height = webshot.image_dimensions(shot.data)
    assert (width, height) == (780, 6000)
    assert shot.segments == 4
    assert shot.reached_bottom is True


def test_capture_scrolled_short_page_is_one_shot():
    page = FakePage(page_height=800, viewport=900,
                    viewport_w=390, scale=1)
    session = _fake_session(page)
    shot = session.capture_scrolled(settle_ms=0)
    assert shot.segments == 1
    assert shot.reached_bottom is True
    assert page.scrolls == []


# ── flow capture ──────────────────────────────────────────


def _flow_routes():
    docs_html = ("<html><head><title>Docs</title></head><body>"
                 "<h1>Docs</h1><a href='/install'>Install</a>"
                 "</body></html>")
    install_html = ("<html><head><title>Install</title></head>"
                    "<body><h1>Install</h1></body></html>")
    landing_html = ("<html><head><title>Home</title></head>"
                    "<body><h1>Home</h1></body></html>")
    return {
        "https://x.example/": {
            "title": "Home", "html": landing_html,
            "page_height": 900,
            "controls": [{"role": "link", "name": "Docs",
                          "on_click": None}]},
        "https://x.example/docs": {
            "title": "Docs", "html": docs_html,
            "page_height": 900,
            "controls": [{"role": "link", "name": "Install",
                          "on_click": None}]},
        "https://x.example/install": {
            "title": "Install", "html": install_html,
            "page_height": 900, "controls": []},
    }


def _flow_navigate(routes):
    link_map = {"Docs": "https://x.example/docs",
                "Install": "https://x.example/install"}

    def navigate(url, **kw):
        page = FakePage(
            url=url, title="Home",
            html="<html><head><title>Home</title></head>"
                 "<body><h1>Home</h1></body></html>",
            routes=routes)
        page.link_map = link_map
        # Landing controls live on the landing route.
        page.controls = list(
            routes[url].get("controls", []))
        page.wire_links()
        session = _fake_session(page)
        nav = NavOutcome(final_url=url, title="Home",
                         gates=[], wall=None, steps=[])
        return session, nav

    return navigate


def test_capture_flow_landing_then_clicks():
    routes = _flow_routes()
    url, opts = webshot.parse_shot_tail(
        "flow https://x.example/ -> Docs -> Install")
    shots = webshot.capture_flow(
        url, opts, navigate_fn=_flow_navigate(routes))
    assert [s.step for s in shots] == [
        "landing", "Docs", "Install"]
    assert [s.final_url for s in shots] == [
        "https://x.example/", "https://x.example/docs",
        "https://x.example/install"]
    assert [s.title for s in shots] == [
        "Home", "Docs", "Install"]
    assert all(s.width == 390 for s in shots)


def test_capture_flow_goto_step_opens_url():
    routes = _flow_routes()
    url, opts = webshot.parse_shot_tail(
        "flow https://x.example/ -> "
        "https://x.example/docs")
    shots = webshot.capture_flow(
        url, opts, navigate_fn=_flow_navigate(routes))
    assert [s.step for s in shots] == [
        "landing", "https://x.example/docs"]
    assert shots[1].final_url == "https://x.example/docs"


def test_capture_flow_missing_link_stops_with_shots():
    routes = _flow_routes()
    url, opts = webshot.parse_shot_tail(
        "flow https://x.example/ -> Nope")
    with pytest.raises(webnav.NavError) as err:
        webshot.capture_flow(
            url, opts, navigate_fn=_flow_navigate(routes))
    assert "Nope" in str(err.value)
    assert len(err.value.shots) == 1
    assert err.value.shots[0].step == "landing"


# ── long capture: frames mode ─────────────────────────────


def test_record_frames_stills_plus_contact_sheet(tmp_path):
    clock = FakeClock()
    page = FakePage(page_height=1800, viewport=900,
                    viewport_w=390, scale=1, clock=clock)
    warm = _fake_session(page, clock=clock)
    warm_nav = NavOutcome(
        final_url="https://x.example/", title="Real Page",
        gates=[], wall=None, steps=[],
        presentation="android-phone",
        presentations_tried=["android-phone"])

    def navigate_fn(url, **kw):
        return warm, warm_nav

    def session_factory(**kw):
        return _fake_session(page, clock=clock)

    seq = webnav.record_frames(
        "https://x.example/", seconds=5.0, interval_s=1.0,
        out_dir=tmp_path, navigate_fn=navigate_fn,
        session_factory=session_factory, clock=clock)
    assert len(seq.paths) == 6
    assert all(p.exists() and p.stat().st_size > 0
               for p in seq.paths)
    assert seq.duration_s == 5.0
    assert seq.interval_s == 1.0
    assert seq.contact_sheet is not None
    assert seq.contact_sheet.exists()
    width, height = webshot.image_dimensions(
        seq.contact_sheet.read_bytes())
    assert width == 3 * 360
    # Thumbs keep aspect: 900 CSS px at 360/390 scale.
    assert height == 2 * int(900 * 360 / 390)
    assert seq.nav is not None
    assert seq.nav.presentation == "android-phone"


def test_contact_sheet_rejects_empty():
    with pytest.raises(ValueError):
        webnav.contact_sheet_png([])


# ── block evidence ────────────────────────────────────────


class _EvidenceProbe:
    """A scripted probe that offers a screenshot."""

    def __init__(self, html, title="Security Check",
                 url="https://x.example/"):
        self._html = html
        self._title = title
        self._url = url

    def url(self):
        return self._url

    def title(self):
        return self._title

    def html(self):
        return self._html

    def screenshot_png(self):
        return _png(16, 16)


def test_block_evidence_carries_screenshot_and_dom():
    probe = _EvidenceProbe(
        "<html><body>Verify you are human</body></html>"
        * 1000)
    ev = capture_block_evidence(
        probe, kind="wall", wall="hard",
        attempts=["checkbox click"])
    assert isinstance(ev, BlockEvidence)
    assert ev.kind == "wall" and ev.wall == "hard"
    assert ev.screenshot == _png(16, 16)
    assert len(ev.html_excerpt) == 8000
    assert "Verify you are human" in ev.html_excerpt
    assert ev.attempts == ["checkbox click"]
    assert ev.url == "https://x.example/"


def test_block_evidence_survives_no_screenshot_seam():
    class BareProbe:
        def url(self):
            return "https://x.example/"

        def title(self):
            return "T"

        def html(self):
            return "<html/>"

    ev = capture_block_evidence(BareProbe(), kind="gate")
    assert ev.screenshot == b""
    assert ev.html_excerpt == "<html/>"


def test_webshot_unavailable_carries_evidence(monkeypatch):
    ev = BlockEvidence(kind="wall", wall="hard",
                       url="https://x.example/",
                       screenshot=_png(8, 8))

    def refused(url, opts, **kw):
        err = WallBlocked("bot wall (hard) needs a hand",
                          wall="hard")
        err.evidence = ev
        raise err

    monkeypatch.setattr(webshot, "nav_renderer", refused)
    with pytest.raises(WebshotUnavailable) as err:
        webshot.capture("https://x.example/", ShotOptions())
    assert err.value.evidence is ev


# ── age-gate click fixes ──────────────────────────────────


PORNHUB_CURLY_PAGE = """<html><head><title>Adult Site</title></head>
<body><div class="age-modal" role="dialog"
style="position:fixed;inset:0">
<h2>This is an adult website</h2>
<p>You must be 18 or older to enter.</p>
<button id="age-enter">I’m 18 or older</button>
<button>Exit</button></div>
<div id="cookie-note"><p>We use cookies.</p>
<button id="cookie-ok">OK</button></div></body></html>"""


def test_detect_keeps_original_age_label():
    gates = detect_gates(
        PORNHUB_CURLY_PAGE,
        "https://www.pornhub.com/view_video.php?viewkey=x")
    assert gates[0].kind == "age"
    assert gates[0].label == "i'm 18 or older"
    assert gates[0].label_raw == "I’m 18 or older"


def test_detect_sees_aria_role_button_gates():
    html = """<html><body><div class="age-gate-modal">
    <h2>This is an adult website</h2>
    <p>You must be 18 or older to enter.</p>
    <div role="button" id="age-enter">I'm 18 or older</div>
    </div></body></html>"""
    gates = detect_gates(html, "https://x.example/")
    assert [g.kind for g in gates] == ["age"]
    assert gates[0].label_raw == "I'm 18 or older"


def test_click_gate_clicks_the_named_button_not_its_classmate():
    # Both buttons share one class: the old selector-hint
    # + .first click landed on "Manage preferences".
    clicked: list[str] = []
    page = FakePage(
        html="<html></html>",
        controls=[
            {"role": "button", "name": "Manage preferences",
             "class": "cmp-action",
             "on_click": lambda: clicked.append("preferences")},
            {"role": "button", "name": "Accept all cookies",
             "class": "cmp-action",
             "on_click": lambda: clicked.append("accept")},
        ])
    gate = GateCandidate(kind="consent",
                         label="accept all cookies",
                         label_raw="Accept all cookies",
                         selector=".cmp-action",
                         host="x.example")
    assert page and webnav._PlaywrightProbe(page).click_gate(gate)
    assert clicked == ["accept"]


def test_click_gate_uses_original_curly_label():
    clicked: list[str] = []
    page = FakePage(
        html="<html></html>",
        controls=[
            {"role": "button", "name": "I’m 18 or older",
             "id": "age-enter",
             "on_click": lambda: clicked.append("age")},
        ])
    gate = GateCandidate(kind="age", label="i'm 18 or older",
                         label_raw="I’m 18 or older",
                         selector="#age-enter",
                         host="x.example")
    assert webnav._PlaywrightProbe(page).click_gate(gate)
    assert clicked == ["age"]


def test_hidden_gate_is_not_a_standing_gate():
    html = ("<html><body><div id='cookie-banner' "
            "style='display:none'><p>We use cookies.</p>"
            "<button id='cookie-accept'>Accept all cookies"
            "</button></div></body></html>")
    page = FakePage(
        html=html,
        controls=[{"role": "button",
                   "name": "Accept all cookies",
                   "id": "cookie-accept", "visible": False}])
    probe = webnav._PlaywrightProbe(page)
    actions = run_interstitial_pass(probe)
    assert actions == []
    assert page.clicked == []


def test_slow_gate_clear_gets_a_bounded_wait():
    # The dialog clears after 3 waits (a slow CMP
    # transition): the old fixed 600 ms glance called
    # this "stayed up".
    state = {"waits": 0}
    html = ("<html><body><div id='cookie-banner'><p>We use "
            "cookies.</p><button id='cookie-accept'>Accept "
            "all cookies</button></div></body></html>")
    content = "<html><body><h1>Real Page</h1></body></html>"
    page = FakePage(html=html)
    probe = webnav._PlaywrightProbe(page)
    real_wait = page.wait_for_timeout

    def wait(ms):
        state["waits"] += 1
        if state["waits"] >= 3:
            page.controls = []
            page._html = content
        real_wait(ms)

    page.wait_for_timeout = wait
    page.controls = [{
        "role": "button", "name": "Accept all cookies",
        "id": "cookie-accept",
        "on_click": lambda: None}]
    actions = run_interstitial_pass(probe)
    assert [a.kind for a in actions] == ["cookie"]
    assert state["waits"] >= 3


class _ScriptedProbe:
    """The old FakeProbe shape: no visibility seam, so the
    pass falls back to HTML detection."""

    def __init__(self, html, *, url="https://x.example/",
                 title="Real Page", after_gate=None,
                 click_works=True):
        self._html = html
        self._url = url
        self._title = title
        self._after_gate = after_gate
        self._click_works = click_works
        self.gate_clicks: list[str] = []

    def url(self):
        return self._url

    def title(self):
        return self._title

    def html(self):
        return self._html

    def reload(self):
        pass

    def wait(self, seconds):
        pass

    def click_gate(self, gate):
        self.gate_clicks.append(gate.kind)
        if not self._click_works:
            return False
        if self._after_gate is not None:
            self._html = self._after_gate
        return True

    def click_checkbox(self):
        return True


COOKIE_PAGE = """<html><head><title>Real Page</title></head>
<body><h1>The actual content</h1>
<div id="cookie-banner"><p>We use cookies to improve your
experience.</p>
<button id="cookie-accept">Accept all cookies</button>
</div></body></html>"""

CONTENT_PAGE = """<html><head><title>Real Page</title></head>
<body><h1>The actual content</h1>
<p>behind every gate.</p></body></html>"""


def test_memory_never_suppresses_a_live_gate():
    memory = GateMemory()
    memory.remember(webnav.GateAction(
        kind="cookie", label="Accept all cookies",
        host="x.example"))
    assert memory.knows("x.example", "cookie")
    probe = _ScriptedProbe(COOKIE_PAGE,
                           after_gate=CONTENT_PAGE)
    actions = run_interstitial_pass(probe, memory=memory)
    # The gate is standing NOW: it is worked NOW, whatever
    # another context cleared before.
    assert [a.kind for a in actions] == ["cookie"]
    assert probe.gate_clicks == ["cookie"]


def test_click_failure_keeps_its_reason():
    page = FakePage(html="<html></html>", controls=[])
    gate = GateCandidate(kind="cookie",
                         label="accept all cookies",
                         label_raw="Accept all cookies",
                         host="x.example")
    probe = webnav._PlaywrightProbe(page)
    assert probe.click_gate(gate) is False
    assert probe.last_click_error != ""


# ── handlers: flow + evidence delivery ────────────────────


class _Gateway:
    def __init__(self):
        self.files = []
        self.sent = []

    def send_file(self, platform, chat_key, path, caption="",
                  **kw):
        self.files.append((platform, chat_key, path, caption,
                           kw))
        return SimpleNamespace(ok=True)

    def send(self, platform, chat_key, text):
        self.sent.append((platform, chat_key, text))
        return SimpleNamespace(ok=True, message_id="1")


def _rig(tmp_path):
    rig = RuntimeGroupMixin()
    rig.context = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=str(tmp_path)),
        db=None)
    rig.gateway = _Gateway()
    rig._owner_sender_ids = {"42"}
    rig._is_operator = lambda message: str(
        message.sender_id) in {"42"}
    return rig


def _msg(*, chat_key="telegram:555", platform="telegram",
         kind=ChatKind.DM, sender_id="42", meta=None):
    return SimpleNamespace(
        chat=SimpleNamespace(platform=platform, key=chat_key,
                             chat_id=chat_key.split(":")[-1],
                             kind=kind, title=""),
        sender_id=sender_id, sender="Sender",
        sender_username="oluwacutyp", text="",
        meta=meta or {}, message_id="m1", reply_to="")


def test_webshot_handler_flow_delivers_every_stop(
        tmp_path, monkeypatch):
    rig = _rig(tmp_path)
    monkeypatch.setattr(webshot, "_host_is_private",
                        lambda host: False)
    png = _png(64, 64)
    shots = [
        CaptureResult(data=png, title="Home",
                      final_url="https://x.example/",
                      rung="playwright", width=64, height=64,
                      step="landing"),
        CaptureResult(data=png, title="Docs",
                      final_url="https://x.example/docs",
                      rung="playwright", width=64, height=64,
                      step="Docs"),
    ]
    monkeypatch.setattr(
        webshot, "capture_flow",
        lambda url, opts, **kw: shots)
    out = rig._control_webshot(
        "flow https://shots.invalid/ -> Docs", _msg())
    assert "flow:" in out
    assert "landing" in out and "Docs" in out
    assert len(rig.gateway.files) == 2
    assert "step 1/2" in rig.gateway.files[0][3]
    assert "step 2/2" in rig.gateway.files[1][3]


def test_webshot_handler_flow_stop_delivers_partial(
        tmp_path, monkeypatch):
    rig = _rig(tmp_path)
    monkeypatch.setattr(webshot, "_host_is_private",
                        lambda host: False)
    png = _png(64, 64)
    partial = [CaptureResult(
        data=png, title="Home",
        final_url="https://x.example/", rung="playwright",
        width=64, height=64, step="landing")]

    def broken(url, opts, **kw):
        err = webnav.NavError(
            "the flow stopped at 'Nope': no link or button "
            "named 'Nope' on the page")
        err.shots = partial
        raise err

    monkeypatch.setattr(webshot, "capture_flow", broken)
    out = rig._control_webshot(
        "flow https://shots.invalid/ -> Nope", _msg())
    assert "flow stopped" in out
    assert "1 stop(s) delivered" in out
    assert len(rig.gateway.files) == 1


def test_webshot_handler_block_delivers_labelled_evidence(
        tmp_path, monkeypatch):
    rig = _rig(tmp_path)
    monkeypatch.setattr(webshot, "_host_is_private",
                        lambda host: False)
    ev = BlockEvidence(kind="wall", wall="hard",
                       url="https://x.example/",
                       title="Security Check",
                       screenshot=_png(64, 64))

    def walled(url, opts, **kw):
        err = WebshotUnavailable(
            url, ["bot wall (hard) needs a human hand"])
        err.evidence = ev
        raise err

    monkeypatch.setattr(webshot, "capture", walled)
    out = rig._control_webshot("https://shots.invalid/",
                               _msg())
    assert "couldn't shoot" in out
    assert "challenge evidence attached" in out
    assert len(rig.gateway.files) == 1
    _p, _c, _path, caption, _kw = rig.gateway.files[0]
    assert "blocked" in caption
    assert "not the page asked for" in caption


def test_webrec_handler_frames_delivers_contact_sheet(
        tmp_path, monkeypatch):
    rig = _rig(tmp_path)
    monkeypatch.setattr(webshot, "_host_is_private",
                        lambda host: False)
    sheet = tmp_path / "contact.png"
    sheet.write_bytes(_png(360, 200))
    frame = tmp_path / "frame-0000.png"
    frame.write_bytes(_png(64, 64))
    seq = SimpleNamespace(
        paths=[frame], contact_sheet=sheet, duration_s=5.0,
        interval_s=5.0,
        nav=NavOutcome(final_url="https://x.example/",
                       title="Real Page", gates=[], wall=None,
                       steps=[]))
    monkeypatch.setattr(
        webshot, "record_video",
        lambda url, opts, **kw: seq)
    out = rig._control_webrec(
        "frames 5s https://shots.invalid/v", _msg())
    assert "1 frames over 5s" in out
    assert len(rig.gateway.files) == 1
    _p, _c, _path, _cap, kw = rig.gateway.files[0]
    assert kw.get("kind") == "image"
    book = rig._webshot_book()
    assert book.journal()[-1]["rung"] == "webrec-frames"


def test_webrec_handler_long_clamp_names_300s(
        tmp_path, monkeypatch):
    rig = _rig(tmp_path)
    monkeypatch.setattr(webshot, "_host_is_private",
                        lambda host: False)
    path = tmp_path / "rec.mp4"
    path.write_bytes(b"\x00" * 2048)
    outcome = SimpleNamespace(
        path=path, fmt="mp4", duration_s=300.0, plan=None,
        nav=NavOutcome(final_url="https://x.example/",
                       title="Real Page", gates=[], wall=None,
                       steps=[]),
        includes_clearing=False, converted=True)
    monkeypatch.setattr(
        webshot, "record_video",
        lambda url, opts, **kw: outcome)
    out = rig._control_webrec(
        "long 400s https://shots.invalid/v", _msg())
    assert "300s maximum" in out


# ── no new brain tool ─────────────────────────────────────


def test_tool_count_still_809_no_new_tool():
    registry = ToolRegistry(SimpleNamespace())
    registry.register_builtins()
    assert len(registry._tools) == 809
    assert "robot" in registry._tools


# ── thread affinity: the session owner ─────────────────


def _serving_owner():
    """An owner whose thread is serving, plus the thread.
    The caller must stop() it and join the thread."""
    holder: dict[str, Any] = {}

    def serve() -> None:
        holder["owner"] = _SessionOwner()
        holder["owner"].serve()

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    deadline = time.monotonic() + 5.0
    while "owner" not in holder:
        assert time.monotonic() < deadline, \
            "the owner thread never started"
        time.sleep(0.01)
    return holder["owner"], worker


def test_session_owner_call_runs_on_owner_thread():
    owner, worker = _serving_owner()
    seen: list[threading.Thread] = []
    value = owner.call(
        lambda: seen.append(threading.current_thread())
        or "done")
    assert value == "done"
    assert seen == [worker]
    # A call made ON the owner thread runs direct, never
    # queued — a nested call must not deadlock.
    assert owner.call(
        lambda: owner.call(lambda: "inner")) == "inner"
    # Errors propagate to the caller, not the executor.
    with pytest.raises(RuntimeError, match="boom"):
        owner.call(lambda: (_ for _ in ()).throw(
            RuntimeError("boom")))
    owner.stop()
    worker.join(timeout=5)
    assert not worker.is_alive()


def test_thread_page_dispatches_through_owner():
    page = FakePage(
        page_height=1800, viewport=900, viewport_w=390,
        scale=2, controls=[{"name": "Docs", "id": "docs",
                             "role": "link", "class": ""}])
    owner, worker = _serving_owner()
    try:
        proxy = _ThreadPage(page, owner)
        assert proxy.url == "https://x.example/"
        assert proxy.title() == "Real Page"
        assert proxy.evaluate(
            "() => [document.documentElement.scrollHeight,"
            " window.innerHeight]") == [1800, 900]
        raw = proxy.screenshot()
        assert Image.open(io.BytesIO(raw)).size == (780, 1800)
        loc = proxy.locator("#docs")
        assert loc.count() == 1
        assert loc.is_visible() is True
        assert loc.inner_text() == "Docs"
        loc.click()
        assert page.clicked == ["Docs"]
        # A proxied frame still compares equal to the
        # proxied main frame (identity, not a fresh id).
        assert proxy.frames[0] == proxy.main_frame
    finally:
        owner.stop()
        worker.join(timeout=5)
        assert not worker.is_alive()


def test_driven_session_marshals_to_owner_thread():
    """The full path a race winner takes: the session is
    handed an owner in the worker thread, the caller
    drives it from another thread, and closing from the
    caller tears it down on the owner thread and stops
    the executor — no dead connection, no hung thread."""
    page = FakePage(page_height=1800, viewport=900,
                    viewport_w=390, scale=2)
    session = webnav.NavSession(
        device=dict(webnav.PRESENTATIONS["android-phone"]))
    session._page = page
    session.probe = webnav._PlaywrightProbe(page)
    session._started_at = time.monotonic()
    holder: dict[str, Any] = {}

    def serve() -> None:
        session._start_owner_loop()
        holder["owner"] = session._owner
        holder["owner"].serve()

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    deadline = time.monotonic() + 5.0
    while "owner" not in holder:
        assert time.monotonic() < deadline, \
            "the owner thread never started"
        time.sleep(0.01)
    data = session.capture_still()
    assert Image.open(io.BytesIO(data)).size == (780, 1800)
    assert session.page.url == "https://x.example/"
    session.__exit__()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert session._owner is None

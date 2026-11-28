"""Login walls behind /webshot (gapfix169).

Instagram's logged-out post page covers the post with a
"Never miss a post from <handle>" sign-up modal: the
interstitial pass had no login-wall kind, so it shot the
modal as if it were the page.  These tests pin the fix:

- the modal is DETECTED as a ``login`` gate, and the pass
  dismisses it through its close control — never through
  the "Log in" / "Sign up" buttons;
- a login wall with no dismiss control is REFUSED (Devon
  does not sign in for you), never shot;
- a shot that would be dominated by a standing login
  modal is refused, not returned — and on Instagram the
  post itself is fetched through the download chain
  instead, when it comes down as an image;
- the Pornhub age-gate seed still fires on the live shape
  (verified against the live homepage 2026-11-28: the
  English button is still "I am 18 or older - Enter").

The Pornhub test is a regression guard: it passes before
the build and must still pass after.  Every login test
fails before the build (no ``login`` kind exists) and
passes after.
"""
from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
from PIL import Image

from nomorals.browser.gate_discovery import GateObservation
from nomorals.social.chat import webnav, webshot
from nomorals.social.chat.webnav import (
    GateBlocked,
    detect_gates,
)
from nomorals.social.chat.webshot import (
    CaptureResult,
    ShotOptions,
    WebshotUnavailable,
)

# ── pages ─────────────────────────────────────────────────

# The owner's screenshot, as DOM: the post (image + caption)
# behind a logged-out modal.  Instagram's real modal is a
# role=dialog layer whose close control is a role=button
# div with aria-label="Close" — the "Log in" / "Sign up"
# controls are NOT dismiss controls.
INSTAGRAM_LOGIN_PAGE = """<html><head><title>val.erhu • Instagram
</title></head>
<body><article><img src="https://example.invalid/art.jpg"
alt="Hera Granger art">
<p>Hera Granger — my Harry Potter OC</p></article>
<div class="modal" role="dialog" aria-modal="true">
<h2>Never miss a post from val.erhu</h2>
<p>Sign up to see photos and videos from your friends.</p>
<button>Sign up</button>
<a href="/accounts/login/">Log in</a>
<div role="button" aria-label="Close"><svg></svg></div>
</div></body></html>"""

GENERIC_LOGIN_PAGE = """<html><head><title>Watch</title></head>
<body><h1>Episode 4</h1>
<div class="overlay" style="position:fixed;inset:0">
<h2>Log in to continue watching</h2>
<button>Log in</button>
<button>Sign up</button>
<button>Not now</button>
</div></body></html>"""

# The same modal with no way out: no close, no "not now".
NO_DISMISS_PAGE = """<html><head><title>val.erhu • Instagram
</title></head>
<body><article><img src="https://example.invalid/art.jpg"
alt="Hera Granger art"></article>
<div class="modal" role="dialog" aria-modal="true">
<h2>Never miss a post from val.erhu</h2>
<p>Sign up to see photos and videos from your friends.</p>
<button>Sign up</button>
<a href="/accounts/login/">Log in</a>
</div></body></html>"""

# Login copy in plain body text, no modal layer: a help
# page mentioning login is not a wall.
PLAIN_LOGIN_MENTION_PAGE = """<html><head><title>Help</title>
</head><body><h1>Help</h1>
<p>To watch more, log in to continue with your account.</p>
<button>Close</button></body></html>"""

CONTENT_PAGE = """<html><head><title>Real Page</title></head>
<body><h1>The actual content</h1>
<p>behind every gate.</p></body></html>"""

# The Pornhub age gate as it renders live (verified
# 2026-11-28 against the live homepage: the English button
# is still "I am 18 or older - Enter").
PORNHUB_LIVE_PAGE = """<html><head><title>Free Porn Videos &amp;
Sex Movies - Pornhub</title></head>
<body><div class="age-modal" role="dialog"
style="position:fixed;inset:0">
<h2>This is an adult website</h2>
<p>You must be 18 or older to enter.</p>
<button>I am 18 or older - Enter</button>
<button>Exit</button></div></body></html>"""

INSTAGRAM_URL = ("https://www.instagram.com/p/"
                 "C0QYv8yO_2a/")


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buf, "PNG")
    return buf.getvalue()


class _Probe:
    """A PageProbe scripted by gate kind: kinds in
    ``refuse`` won't click; kinds in ``land`` land on the
    given page when clicked."""

    def __init__(self, html, *, url="https://x.example/",
                 refuse=(), land=None):
        self._html = html
        self._url = url
        self._refuse = set(refuse)
        self._land = dict(land or {})
        self.gate_clicks: list[tuple[str, str]] = []
        self.waits: list[float] = []

    def url(self):
        return self._url

    def title(self):
        return "page"

    def html(self):
        return self._html

    def reload(self):
        pass

    def wait(self, seconds):
        self.waits.append(seconds)

    def click_gate(self, gate):
        self.gate_clicks.append((gate.kind, gate.label))
        if gate.kind in self._refuse:
            return False
        if gate.kind in self._land:
            self._html = self._land[gate.kind]
        return True

    def click_checkbox(self):
        return True


def _login_obs() -> GateObservation:
    return GateObservation(
        kind="login", label="close", selector="#ig-close",
        host="instagram.com", blocking=True, visible=True,
        raw_label="Close")


# ── detection ─────────────────────────────────────────────


def test_detect_instagram_login_modal():
    gates = detect_gates(INSTAGRAM_LOGIN_PAGE, INSTAGRAM_URL)
    assert [g.kind for g in gates] == ["login"]
    assert gates[0].blocking is True
    # The dismiss control — never the "Log in" / "Sign up"
    # buttons.  The close div carries no id or class, so
    # there is no selector: the clicker resolves it by its
    # raw label instead (the contract's finding 2).
    assert gates[0].label == "close"
    assert gates[0].raw_label == "Close"


def test_detect_generic_login_modal():
    gates = detect_gates(GENERIC_LOGIN_PAGE,
                         "https://watch.example/ep/4")
    assert [g.kind for g in gates] == ["login"]
    assert gates[0].label == "not now"


def test_login_modal_without_dismiss_is_detected():
    gates = detect_gates(NO_DISMISS_PAGE, INSTAGRAM_URL)
    assert [g.kind for g in gates] == ["login"]
    assert gates[0].label == "login wall — no way to dismiss"
    assert gates[0].selector == ""
    assert gates[0].raw_label == ""


def test_login_copy_without_a_modal_is_not_a_gate():
    assert detect_gates(PLAIN_LOGIN_MENTION_PAGE,
                        "https://help.example/") == []


def test_pornhub_age_seed_still_fires_on_live_shape():
    gates = detect_gates(PORNHUB_LIVE_PAGE,
                         "https://www.pornhub.com/")
    assert gates and gates[0].kind == "age"
    assert gates[0].blocking is True
    assert "18 or older" in gates[0].label


# ── the pass ──────────────────────────────────────────────


def test_pass_dismisses_instagram_login_modal():
    probe = _Probe(INSTAGRAM_LOGIN_PAGE, url=INSTAGRAM_URL,
                   land={"login": CONTENT_PAGE})
    actions = webnav.run_interstitial_pass(probe)
    assert [a.kind for a in actions] == ["login"]
    # The click landed on the close control, never on
    # "Log in" or "Sign up".
    assert probe.gate_clicks == [("login", "close")]


def test_pass_refuses_undismissable_login_wall():
    probe = _Probe(NO_DISMISS_PAGE, url=INSTAGRAM_URL)
    with pytest.raises(GateBlocked) as err:
        webnav.run_interstitial_pass(probe)
    assert "login" in str(err.value)
    assert "does not sign in" in str(err.value)


def test_pass_refuses_when_dismiss_click_fails():
    probe = _Probe(INSTAGRAM_LOGIN_PAGE, url=INSTAGRAM_URL,
                   refuse={"login"})
    with pytest.raises(GateBlocked) as err:
        webnav.run_interstitial_pass(probe)
    assert "login" in str(err.value)


# ── the shot ──────────────────────────────────────────────


def test_shot_with_uncleared_login_modal_is_refused():
    def refused(url, opts):
        raise GateBlocked("a login wall is still standing",
                          gates=[_login_obs()])

    with pytest.raises(WebshotUnavailable) as err:
        webshot.capture("https://www.instagram.com/p/x/",
                        ShotOptions(), local_renderer=refused)
    assert "login" in str(err.value)


def test_shot_falls_back_to_instagram_post_when_it_lands():
    def refused(url, opts):
        raise GateBlocked("a login wall is still standing",
                          gates=[_login_obs()])

    def fetch(url):
        return (_png_bytes(), "val.erhu — Hera Granger")

    result = webshot.capture(
        "https://www.instagram.com/p/x/", ShotOptions(),
        local_renderer=refused, instagram_fetch=fetch)
    assert isinstance(result, CaptureResult)
    assert result.rung == "instagram"
    # The capture is a PNG (the delivery path accepts PNG
    # only), 8x8 like the fetched image.
    assert webshot.image_dimensions(result.data) == (8, 8)
    assert "Hera Granger" in result.title


def test_shot_converts_instagram_jpeg_to_png():
    def refused(url, opts):
        raise GateBlocked("a login wall is still standing",
                          gates=[_login_obs()])

    def fetch_jpeg(url):
        buf = io.BytesIO()
        Image.new("RGB", (8, 8), "blue").save(buf, "JPEG")
        return (buf.getvalue(), "a jpeg post")

    result = webshot.capture(
        "https://www.instagram.com/p/x/", ShotOptions(),
        local_renderer=refused, instagram_fetch=fetch_jpeg)
    assert result.rung == "instagram"
    assert result.data[:8] == b"\x89PNG\r\n\x1a\n"
    assert webshot.image_dimensions(result.data) == (8, 8)


def test_shot_refuses_when_instagram_fetch_returns_nothing():
    def refused(url, opts):
        raise GateBlocked("a login wall is still standing",
                          gates=[_login_obs()])

    with pytest.raises(WebshotUnavailable) as err:
        webshot.capture(
            "https://www.instagram.com/p/x/", ShotOptions(),
            local_renderer=refused,
            instagram_fetch=lambda url: None)
    assert "login" in str(err.value)
    assert "/instagram" in str(err.value)


def test_shot_refuses_when_instagram_fetch_is_not_an_image():
    def refused(url, opts):
        raise GateBlocked("a login wall is still standing",
                          gates=[_login_obs()])

    with pytest.raises(WebshotUnavailable) as err:
        webshot.capture(
            "https://www.instagram.com/p/x/", ShotOptions(),
            local_renderer=refused,
            instagram_fetch=lambda url: (b"not-an-image", "x"))
    assert "login" in str(err.value)


# ── the late-modal guard ──────────────────────────────────


def test_nav_renderer_refuses_login_wall_standing_at_shoot():
    """The modal mounts after the pass ran: the shot is
    refused, not taken over the modal."""
    probe = SimpleNamespace(
        html=lambda: INSTAGRAM_LOGIN_PAGE,
        url=lambda: INSTAGRAM_URL)

    class _Session:
        def __init__(self):
            self.probe = probe
            self.closed = False

        def capture_still(self, full_page=False, pdf=False):
            raise AssertionError("must not shoot over a wall")

        def __exit__(self, *exc):
            self.closed = True
            return False

    nav = SimpleNamespace(
        title="val.erhu", final_url=INSTAGRAM_URL, gates=[],
        wall="", presentation="desktop",
        presentations_tried=[], gate_projection=None)

    import nomorals.social.chat.webnav as webnav_mod
    real_navigate = webnav_mod.navigate

    def fake_navigate(*args, **kwargs):
        return _Session(), nav

    webnav_mod.navigate = fake_navigate  # type: ignore[method-assign]
    try:
        with pytest.raises(GateBlocked) as err:
            webshot.nav_renderer(INSTAGRAM_URL, ShotOptions())
    finally:
        webnav_mod.navigate = real_navigate  # type: ignore[method-assign]
    assert "login" in str(err.value)


def test_nav_renderer_shoots_when_no_login_wall_stands():
    probe = SimpleNamespace(
        html=lambda: CONTENT_PAGE,
        url=lambda: "https://x.example/")

    class _Session:
        def __init__(self):
            self.probe = probe

        def capture_still(self, full_page=False, pdf=False):
            return _png_bytes()

        def __exit__(self, *exc):
            return False

    nav = SimpleNamespace(
        title="Real Page", final_url="https://x.example/",
        gates=[], wall="", presentation="desktop",
        presentations_tried=[], gate_projection=None)

    import nomorals.social.chat.webnav as webnav_mod
    real_navigate = webnav_mod.navigate

    def fake_navigate(*args, **kwargs):
        return _Session(), nav

    webnav_mod.navigate = fake_navigate  # type: ignore[method-assign]
    try:
        result = webshot.nav_renderer("https://x.example/",
                                      ShotOptions())
    finally:
        webnav_mod.navigate = real_navigate  # type: ignore[method-assign]
    assert result.data == _png_bytes()
    assert result.rung == "playwright"

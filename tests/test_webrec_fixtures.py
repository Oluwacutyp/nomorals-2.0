"""Webnav/webrec against REAL Chromium + local fixtures.

The seam suite (test_webnav.py) pins the logic; this suite
proves the same code drives a real browser: fixture pages
behind (a) a cookie banner, (b) an age-gate dialog in the
real Pornhub shape (the "I'm 18 or older" overlay — no
'ok' gate), (b2) the age overlay standing over a cookie
bar whose button reads 'OK' (the age control must clear
first), (c) a checkbox challenge, (d) a hard
image-grid wall (unclearable → takeover requested, honest
failure; clearable-by-owner → takeover resumes), and (e) a
long scrollable page for /webrec (duration probed from the
file, scroll progression measured from frames, and the
recording provably never opens on a gate).

Skips cleanly when Playwright/Chromium is unavailable on
the host — the seam suite still pins the behavior there.
"""
from __future__ import annotations

import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pytest.importorskip("playwright")

from nomorals.social.chat import webshot  # noqa: E402
from nomorals.social.chat.webshot import RecOptions, ShotOptions  # noqa: E402

CONTENT_BODY = """
<h1>The actual content</h1>
<p>behind every gate.</p>"""

COOKIE_PAGE = f"""<html><head><title>Real Page</title></head>
<body>{CONTENT_BODY}
<div id="cookie-banner" style="position:fixed;top:0;left:0;
right:0;background:#ffe;border-bottom:2px solid #333">
<p>We use cookies to improve your experience.</p>
<button id="cookie-accept" onclick="
document.cookie='consent=yes; max-age=86400; path=/';
document.getElementById('cookie-banner').remove();"
>Accept all cookies</button></div>
</body></html>"""

# The real Pornhub shape (owner evidence, 2026-11-08):
# the gate is the age-verification overlay's "I'm 18 or
# older" enter button.  There is NO 'ok' gate on the
# page — an earlier fixture invented one, and the pass
# dutifully aimed at it while the age modal stood.
_AGE_OVERLAY = """
<div id="age-verification" class="modal age-gate"
role="dialog" aria-modal="true"
style="position:fixed;inset:0;background:#111;color:#eee;
z-index:1000">
<h2>This is an adult website</h2>
<p>You must be 18 or older to enter. This site contains
sexually explicit material.</p>
<button id="age-enter" onclick="
document.getElementById('age-verification').remove();
document.getElementById('content').style.display='block';
document.title='Real Page';
">I'm 18 or older</button>
<button>Exit</button></div>"""

AGE_PAGE = f"""<html><head><title>Adult Site</title></head>
<body><div id="content" style="display:none">
{CONTENT_BODY}</div>{_AGE_OVERLAY}
</body></html>"""

# Both gates at once: a cookie bar whose button reads
# 'OK' (not blocking — a bar along the bottom edge) and
# the age overlay standing over the whole page.  The
# pass must work the age control FIRST: the 'OK' is
# covered by the overlay and cannot be clicked at all
# until the modal clears.
AGE_OVER_COOKIE_PAGE = f"""<html><head><title>Adult Site</title>
</head>
<body><div id="content" style="display:none">
{CONTENT_BODY}</div>
<div id="cookie-banner" style="position:fixed;left:0;
right:0;bottom:0;background:#ffe;
border-top:2px solid #333;z-index:10">
<p>We use cookies to improve your experience.</p>
<button id="cookie-ok" onclick="
document.getElementById('cookie-banner').remove();"
>OK</button></div>{_AGE_OVERLAY}
</body></html>"""

# The detector keys on the widget's own markup (the
# g-recaptcha element) — carried hermetically, no live
# third-party script: the visible checkbox is the fixture's
# stand-in for the widget's checkbox, and clicking it
# completes the check in-page like the real flow.
CHECKBOX_PAGE = """<html><head><title>Security Check</title>
</head><body>
<h1>Verify you are human</h1>
<p>Complete the security check to continue.</p>
<div class="g-recaptcha" data-sitekey="x"
style="display:none"></div>
<div class="challenge-checkbox" id="cbx"
style="width:28px;height:28px;border:2px solid #333;
cursor:pointer"></div><span>I'm not a robot</span>
<script>
document.getElementById('cbx').addEventListener('click',
  function () {
    setTimeout(function () {
      document.body.innerHTML =
        '<h1>The actual content</h1>' +
        '<p>behind every gate.</p>';
      document.title = 'Real Page';
    }, 600);
  });
</script></body></html>"""

WALL_STATE = {"cleared": False}

WALL_PAGE = """<html><head><title>Security Check</title>
</head><body>
<h1>Verify you are human</h1>
<div class="rc-imageselect"><p>Select all images with
buses</p><div class="rc-image-tile-wrapper"></div></div>
<div class="g-recaptcha" data-sitekey="x"
style="display:none"></div>
<script>
setInterval(function () {
  fetch('/wall-state').then(function (r) {
    return r.json();
  }).then(function (s) {
    if (s.cleared) {
      document.body.innerHTML =
        '<h1>The actual content</h1>' +
        '<p>behind every gate.</p>';
      document.title = 'Real Page';
    }
  });
}, 400);
</script></body></html>"""

LONG_PAGE = "<html><head><title>Long Page</title></head>" \
    "<body>" + "".join(
        f'<div style="height:300px;background:'
        f'hsl({(i * 37) % 360},70%,80%)">'
        f"<h2>Section {i}</h2></div>"
        for i in range(30)) + "</body></html>"

# Like a real gated site: accepting sets the consent
# cookie (with a max-age, as real consent banners do), and
# the banner renders only while it is absent — so a
# persistent browser profile clears it exactly once.
LONG_GATED_PAGE = LONG_PAGE.replace(
    "</body>",
    """<script>
if (!document.cookie.includes('consent=yes')) {
  var b = document.createElement('div');
  b.id = 'cookie-banner';
  b.style.cssText = 'position:fixed;top:0;left:0;right:0;' +
    'height:120px;background:#cc0000;color:#fff;z-index:99';
  b.innerHTML = '<p>We use cookies to improve your ' +
    'experience.</p><button id="cookie-accept">Accept ' +
    'all cookies</button>';
  document.body.appendChild(b);
  document.getElementById('cookie-accept')
    .addEventListener('click', function () {
      document.cookie =
        'consent=yes; max-age=86400; path=/';
      b.remove();
    });
}
</script></body>""")

PAGES = {
    "/cookie": COOKIE_PAGE,
    "/age": AGE_PAGE,
    "/age-cookie": AGE_OVER_COOKIE_PAGE,
    "/checkbox": CHECKBOX_PAGE,
    "/wall": WALL_PAGE,
    "/long": LONG_PAGE,
    "/long-gated": LONG_GATED_PAGE,
}


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - http.server's name
        if self.path == "/flip":
            WALL_STATE["cleared"] = True
            body = b"ok"
        elif self.path == "/wall-state":
            import json

            body = json.dumps(WALL_STATE).encode()
        elif self.path in PAGES:
            body = PAGES[self.path].encode()
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence
        pass


@pytest.fixture(scope="module")
def base_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever,
                              daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture(autouse=True)
def _reset_wall():
    WALL_STATE["cleared"] = False
    yield
    WALL_STATE["cleared"] = False


def _frame(video: Path, at_s: float, out: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(at_s), "-i", str(video),
         "-frames:v", "1", str(out)],
        capture_output=True, check=True, timeout=30)
    return out


# ── stills through the pipeline ───────────────────────────


def test_cookie_gate_cleared_and_content_shot(base_url,
                                              tmp_path):
    result = webshot.capture(
        f"{base_url}/cookie", ShotOptions(),
        profile_dir=tmp_path / "profile")
    assert result.gates_cleared == ["cookie"]
    assert result.title == "Real Page"
    assert result.data[:8] == b"\x89PNG\r\n\x1a\n"


def test_age_gate_cleared_and_content_shot(base_url,
                                           tmp_path):
    result = webshot.capture(
        f"{base_url}/age", ShotOptions(),
        profile_dir=tmp_path / "profile")
    assert result.gates_cleared == ["age"]
    # The fixture flips its title only when the gate's own
    # button is clicked and the content is revealed.
    assert result.title == "Real Page"


def test_age_gate_outranks_the_cookie_ok(base_url,
                                         tmp_path):
    # The page carries a cookie 'OK' AND the blocking age
    # overlay.  The pass must clear the age gate first
    # (the 'OK' sits under the overlay), then the cookie.
    result = webshot.capture(
        f"{base_url}/age-cookie", ShotOptions(),
        profile_dir=tmp_path / "profile")
    assert result.gates_cleared == ["age", "cookie"]
    assert result.title == "Real Page"


def test_checkbox_wall_clicked_through(base_url, tmp_path):
    result = webshot.capture(
        f"{base_url}/checkbox", ShotOptions(),
        profile_dir=tmp_path / "profile")
    assert result.wall_via == "checkbox_click"
    assert result.title == "Real Page"


def test_hard_wall_without_takeover_fails_honestly(
        base_url, tmp_path):
    with pytest.raises(webshot.WebshotUnavailable) as err:
        webshot.capture(
            f"{base_url}/wall", ShotOptions(),
            profile_dir=tmp_path / "profile")
    assert "bot wall" in str(err.value)


def test_hard_wall_owner_takeover_resumes(base_url,
                                          tmp_path):
    import urllib.request

    def hook(ask):
        urllib.request.urlopen(f"{base_url}/flip",
                               timeout=5).read()
        return True

    result = webshot.capture(
        f"{base_url}/wall", ShotOptions(),
        profile_dir=tmp_path / "profile",
        takeover_hook=hook)
    assert result.wall_via == "owner_takeover"
    assert result.title == "Real Page"


# ── /webrec ───────────────────────────────────────────────


def test_webrec_records_scrolling_video(base_url, tmp_path):
    outcome = webshot.record_video(
        f"{base_url}/long", RecOptions(seconds=8.0),
        out_dir=tmp_path / "out",
        profile_dir=tmp_path / "profile")
    assert outcome.fmt == "mp4"
    assert outcome.duration_s >= 7.5
    assert outcome.plan is not None
    assert outcome.plan.reached_bottom is True
    first = _frame(outcome.path, 1.0, tmp_path / "f1.png")
    later = _frame(outcome.path, 5.0, tmp_path / "f2.png")
    ratio = webshot.change_ratio(first.read_bytes(),
                                 later.read_bytes())
    assert ratio > 0.05, ratio


def test_webrec_never_opens_on_a_gate(base_url, tmp_path):
    outcome = webshot.record_video(
        f"{base_url}/long-gated", RecOptions(seconds=5.0),
        out_dir=tmp_path / "out",
        profile_dir=tmp_path / "profile")
    assert outcome.nav is not None
    # The warm-up pass cleared the banner into the profile
    # (the site's consent cookie persists), so the
    # recording pass itself had nothing to clear and its
    # opening frames are the page, not the gate.
    assert outcome.includes_clearing is False
    assert outcome.nav.gates == []
    # And the frames agree: no red banner bar at the top
    # once the page has loaded.
    frame = _frame(outcome.path, 1.5, tmp_path / "g.png")
    from PIL import Image

    img = Image.open(frame).convert("RGB")
    strip = img.crop((0, 0, img.width, 40))
    pixels = list(strip.getdata())
    red = sum(1 for r, g, b in pixels
              if r > 150 and g < 80 and b < 80)
    assert red / len(pixels) < 0.2

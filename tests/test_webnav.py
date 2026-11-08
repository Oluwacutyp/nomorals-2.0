"""Webnav (§18/§20 god-tier): the navigation pipeline
behind /webshot and /webrec, pinned at the probe seam.

- gate detection is pure and covers the whole class:
  cookie banners, consent walls, age gates (generic
  heuristics + the pornhub.com site shape), and DOB forms
  (an age gate Devon will not invent an answer for).
- the interstitial pass clicks, VERIFIES, remembers, and
  fails honestly — a gate that stays up is a GateBlocked,
  never a photo of the gate.
- bot walls are typed against the repo's REAL challenge
  detector: managed → wait-out, checkbox → click, hard →
  takeover-or-honest-failure; an incidental widget on an
  open page is not a wall.
- scroll planning is eased, duration-paced, loop-pans
  short pages, and admits mid-page endings on long ones.
- the /webshot ladder treats a gate/wall refusal as
  FINAL (no provider seed shoots over it) while an
  unavailable engine still falls through to the seeds.
- /webrec's handler delivers an inline video (kind=
  "video", clean name), journals, and answers honestly
  on failure; /webshot's reply names the gates cleared.
- gate targeting aims at the BLOCKING layer's control:
  age confirms (incl. the real Pornhub shape, curly
  apostrophe and all) outrank a stray consent 'ok', and
  a click that refuses or doesn't clear is a failed
  attempt the pass moves on from — never a dead end.
- presentation is a ladder of coherent fingerprints
  (mobile first) RACED in parallel (owner directive
  2026-11-08): every fingerprint opens concurrently in
  its own session, the first to clear wins, losers
  close immediately, owner takeover stays the last
  resort (last rung, armed only when every other rung
  has failed), and the outcome journals every rung's
  verdict and the one that cleared.
"""
from __future__ import annotations

import io
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from nomorals.agents.partner.runtime_group import RuntimeGroupMixin
from nomorals.browser.challenge import detect_challenge_signal
from nomorals.social.chat import webnav, webshot
from nomorals.social.chat.base import ChatKind
from nomorals.social.chat.webnav import (
    GateBlocked,
    GateCandidate,
    GateMemory,
    NavUnavailable,
    WallBlocked,
    classify_wall,
    detect_gates,
    run_wall_pass,
    scroll_plan,
)
from nomorals.social.chat.webshot import CaptureResult, ShotOptions, WebshotUnavailable

# ── pages ─────────────────────────────────────────────────

COOKIE_PAGE = """<html><head><title>Real Page</title></head>
<body><h1>The actual content</h1>
<div id="cookie-banner"><p>We use cookies to improve your
experience.</p>
<button id="cookie-accept">Accept all cookies</button>
</div></body></html>"""

CONTENT_PAGE = """<html><head><title>Real Page</title></head>
<body><h1>The actual content</h1>
<p>behind every gate.</p></body></html>"""

CONSENT_PAGE = """<html><head><title>Real Page</title></head>
<body><h1>The actual content</h1>
<div id="onetrust-banner"><p>We value your privacy. This
site uses cookies and similar technologies (GDPR).</p>
<button id="onetrust-accept">Accept all</button>
</div></body></html>"""

AGE_PAGE = """<html><head><title>Adult Site</title></head>
<body><div class="age-gate-modal"><h2>This is an adult
website</h2><p>You must be 18 or older to enter.</p>
<button id="age-enter">I'm 18 or Older - Enter</button>
<button>Exit</button></div></body></html>"""

DOB_PAGE = """<html><head><title>Adult Site</title></head>
<body><h2>Age verification</h2><p>Confirm your age to
enter.</p><form><label>Date of birth
<input type="date" name="birth_date"></label>
<button type="submit">Submit</button></form>
</body></html>"""

PLAIN_ACCEPT_PAGE = """<html><head><title>Shop</title>
</head><body><h1>Terms updated</h1>
<button>Accept</button></body></html>"""

MANAGED_PAGE = """<html><head><title>Just a moment...
</title></head><body><h1>Checking your browser before
accessing</h1><div id="cf-challenge-running"></div>
</body></html>"""

CHECKBOX_PAGE = """<html><head><title>Security Check</title>
</head><body><h1>Verify you are human</h1>
<div class="g-recaptcha" data-sitekey="x"></div>
<iframe src="https://www.google.com/recaptcha/api2/anchor?k=x"></iframe>
</body></html>"""

GRID_PAGE = """<html><head><title>Security Check</title>
</head><body><h1>Verify you are human</h1>
<div class="rc-imageselect"><p>Select all images with
buses</p><div class="rc-image-tile-wrapper"></div></div>
<div class="g-recaptcha" data-sitekey="x"></div>
</body></html>"""

INCIDENTAL_PAGE = """<html><head><title>Blog</title></head>
<body><h1>A post</h1><p>comments below</p>
<div class="g-recaptcha" data-sitekey="x"></div>
</body></html>"""


def _png(width=8, height=8):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (200, 30, 30)).save(
        buf, "PNG")
    return buf.getvalue()


# ── the scripted probe ────────────────────────────────────


class FakeProbe:
    """A PageProbe driven by a script: click_gate moves to
    the page the gate was hiding (unless told to refuse);
    click_checkbox clears a checkbox wall when scripted."""

    def __init__(self, html, *, url="https://x.example/",
                 title="Real Page", after_gate=None,
                 click_works=True, after_checkbox=None):
        self._html = html
        self._url = url
        self._title = title
        self._after_gate = after_gate
        self._click_works = click_works
        self._after_checkbox = after_checkbox
        self.gate_clicks = []
        self.checkbox_clicks = 0
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

    def click_gate(self, gate):
        self.gate_clicks.append(gate.kind)
        if not self._click_works:
            return False
        if self._after_gate is not None:
            self._html = self._after_gate
        return True

    def click_checkbox(self):
        self.checkbox_clicks += 1
        if self._after_checkbox is not None:
            self._html = self._after_checkbox
        return True


# ── gate detection ────────────────────────────────────────


def test_detect_cookie_banner():
    gates = detect_gates(COOKIE_PAGE, "https://x.example/")
    assert [g.kind for g in gates] == ["cookie"]
    assert gates[0].selector == "#cookie-accept"


def test_detect_consent_wall():
    gates = detect_gates(CONSENT_PAGE, "https://x.example/")
    assert gates and gates[0].kind == "consent"


def test_detect_age_gate_generic():
    gates = detect_gates(AGE_PAGE, "https://videos.example/")
    assert [g.kind for g in gates] == ["age"]
    assert "18 or older" in gates[0].label


def test_detect_age_gate_pornhub_shape():
    gates = detect_gates(AGE_PAGE, "https://www.pornhub.com/")
    assert [g.kind for g in gates] == ["age"]


def test_detect_dob_form_is_its_own_kind():
    gates = detect_gates(DOB_PAGE, "https://videos.example/")
    assert [g.kind for g in gates] == ["age_dob"]


def test_bare_accept_without_gate_context_is_not_a_gate():
    assert detect_gates(PLAIN_ACCEPT_PAGE,
                        "https://shop.example/") == []
    assert detect_gates(CONTENT_PAGE, "https://x.example/") == []


def test_cookie_banner_mentioning_consent_is_a_cookie_gate():
    # The word "consent" inside a cookie assignment does
    # not make a cookie banner a GDPR consent wall.
    page = COOKIE_PAGE.replace(
        "</body>",
        "<script>document.cookie = 'consent=yes';</script>"
        "</body>")
    gates = detect_gates(page, "https://x.example/")
    assert [g.kind for g in gates] == ["cookie"]


# ── the interstitial pass ─────────────────────────────────


def test_pass_clears_cookie_gate_and_verifies():
    probe = FakeProbe(COOKIE_PAGE, after_gate=CONTENT_PAGE)
    actions = webnav.run_interstitial_pass(probe)
    assert [a.kind for a in actions] == ["cookie"]
    assert probe.gate_clicks == ["cookie"]


def test_pass_remembers_cleared_gates():
    memory = GateMemory()
    probe2 = FakeProbe(COOKIE_PAGE, after_gate=CONTENT_PAGE)
    webnav.run_interstitial_pass(probe2, memory=memory)
    assert memory.knows("x.example", "cookie")


def test_gate_that_stays_up_blocks_honestly():
    probe = FakeProbe(COOKIE_PAGE, after_gate=COOKIE_PAGE)
    with pytest.raises(GateBlocked) as err:
        webnav.run_interstitial_pass(probe)
    assert "still standing" in str(err.value)


def test_gate_that_will_not_click_blocks_honestly():
    probe = FakeProbe(COOKIE_PAGE, click_works=False)
    with pytest.raises(GateBlocked) as err:
        webnav.run_interstitial_pass(probe)
    assert "would not click" in str(err.value)


def test_dob_gate_is_named_and_never_invented():
    probe = FakeProbe(DOB_PAGE)
    with pytest.raises(GateBlocked) as err:
        webnav.run_interstitial_pass(probe)
    assert "date of birth" in str(err.value)
    assert probe.gate_clicks == []


# ── wall typing (against the real detector) ───────────────


def _wall(html, title):
    signal = detect_challenge_signal(html, "https://x.example/",
                                     title=title)
    return classify_wall(signal, html, title)


def test_wall_typing():
    assert _wall(MANAGED_PAGE, "Just a moment...") == "managed"
    assert _wall(CHECKBOX_PAGE, "Security Check") == "checkbox"
    assert _wall(GRID_PAGE, "Security Check") == "hard"
    assert _wall(INCIDENTAL_PAGE, "Blog") == ""
    assert _wall(CONTENT_PAGE, "Real Page") == ""


# ── the wall pass ─────────────────────────────────────────


def test_managed_wall_waits_out():
    probe = FakeProbe(MANAGED_PAGE, title="Just a moment...")
    state = {"t": 0.0}
    waits = {"n": 0}

    def clock():
        return state["t"]

    real_wait = probe.wait

    def wait(seconds):
        state["t"] += seconds
        waits["n"] += 1
        if waits["n"] >= 2:
            probe._html = CONTENT_PAGE
            probe._title = "Real Page"
        real_wait(seconds)

    probe.wait = wait
    via = run_wall_pass(probe, url="https://x.example/",
                        clock=clock)
    assert via == "managed_wait"


def test_checkbox_wall_clicks_and_verifies():
    probe = FakeProbe(CHECKBOX_PAGE, title="Security Check",
                      after_checkbox=CONTENT_PAGE)
    via = run_wall_pass(probe, url="https://x.example/")
    assert via == "checkbox_click"
    assert probe.checkbox_clicks == 1


def test_hard_wall_without_hook_fails_honestly():
    probe = FakeProbe(GRID_PAGE, title="Security Check")
    with pytest.raises(WallBlocked) as err:
        run_wall_pass(probe, url="https://x.example/")
    assert err.value.wall == "hard"


def test_hard_wall_undelivered_notice_fails():
    probe = FakeProbe(GRID_PAGE, title="Security Check")
    with pytest.raises(WallBlocked) as err:
        run_wall_pass(probe, url="https://x.example/",
                      takeover_hook=lambda ask: False)
    assert "could not be delivered" in str(err.value)


def test_hard_wall_owner_takeover_resumes():
    probe = FakeProbe(GRID_PAGE, title="Security Check")
    asks = []

    def hook(ask):
        asks.append(ask)
        probe._html = CONTENT_PAGE
        probe._title = "Real Page"
        return True

    via = run_wall_pass(probe, url="https://x.example/",
                        takeover_hook=hook)
    assert via == "owner_takeover"
    assert asks and asks[0].wall == "hard"


def test_hard_wall_takeover_timeout_is_honest():
    probe = FakeProbe(GRID_PAGE, title="Security Check")
    state = {"t": 0.0}
    probe.wait = lambda s: state.__setitem__(
        "t", state["t"] + s)
    with pytest.raises(WallBlocked) as err:
        run_wall_pass(probe, url="https://x.example/",
                      takeover_hook=lambda ask: True,
                      takeover_budget_s=10.0,
                      clock=lambda: state["t"])
    assert "did not clear" in str(err.value)


def test_open_page_needs_no_wall_pass():
    probe = FakeProbe(CONTENT_PAGE)
    assert run_wall_pass(probe, url="https://x.example/") is None


# ── scroll planning ───────────────────────────────────────


def test_scroll_plan_holds_then_reaches_bottom():
    plan = scroll_plan(6.0, 3000, 900)
    assert plan.reached_bottom is True
    assert plan.y_at(0.0) == 0.0
    assert plan.y_at(6.0) == pytest.approx(2100.0)
    # Within the first descent the glide is monotonic and
    # eased (slow at both ends, faster in the middle).
    ys = [plan.y_at(0.8 + t / 20) for t in range(25)]
    assert all(b >= a - 1.0 for a, b in zip(ys, ys[1:],
                                            strict=False))
    assert plan.y_at(1.0) < plan.y_at(1.45) < plan.y_at(1.9)


def test_scroll_plan_loop_pans_a_short_page():
    plan = scroll_plan(20.0, 1800, 900)
    assert plan.cycles >= 2
    assert plan.reached_bottom is True


def test_scroll_plan_long_page_ends_mid_page_honestly():
    plan = scroll_plan(5.0, 900 * 40, 900)
    assert plan.reached_bottom is False
    assert plan.y_at(5.0) < plan.distance


def test_scroll_plan_flat_page_holds():
    plan = scroll_plan(5.0, 800, 900)
    assert plan.distance == 0.0
    assert plan.y_at(2.5) == 0.0


# ── /webrec grammar ───────────────────────────────────────


def test_rec_grammar_defaults_and_variants():
    url, opts = webshot.parse_rec_tail("https://x.example/")
    assert url == "https://x.example/"
    assert opts.seconds == 10.0 and opts.clamped == ""
    url, opts = webshot.parse_rec_tail(
        "12s phone dark https://x.example/")
    assert opts.seconds == 12.0 and opts.device == "phone"
    assert opts.scheme == "dark"


def test_rec_grammar_clamps_are_recorded():
    _, opts = webshot.parse_rec_tail("2s https://x.example/")
    assert opts.seconds == 5.0 and opts.clamped == "min"
    _, opts = webshot.parse_rec_tail("90 https://x.example/")
    assert opts.seconds == 60.0 and opts.clamped == "max"


def test_rec_grammar_errors_and_reply_target():
    with pytest.raises(webshot.ShotUsageError):
        webshot.parse_rec_tail("sideways https://x.example/")
    with pytest.raises(webshot.ShotUsageError):
        webshot.parse_rec_tail("12s")
    url, _ = webshot.parse_rec_tail(
        "12s", reply_url="https://replied.example/v")
    assert url == "https://replied.example/v"


# ── the /webshot ladder discipline ────────────────────────


def test_gate_refusal_is_final_no_seed_fallthrough(
        monkeypatch):
    def refused(url, opts, **kw):
        raise GateBlocked("the cookie gate would not click",
                          gates=[GateCandidate(
                              kind="cookie", label="Accept")])

    monkeypatch.setattr(webshot, "nav_renderer", refused)
    fetch_calls = []

    def fetcher(url, **kw):
        fetch_calls.append(url)
        return 200, b"", ""

    with pytest.raises(WebshotUnavailable) as err:
        webshot.capture("https://x.example/", ShotOptions(),
                        fetcher=fetcher)
    assert "would not click" in str(err.value)
    assert fetch_calls == []


def test_unavailable_engine_falls_through_to_seeds(
        monkeypatch):
    def absent(url, opts, **kw):
        raise NavUnavailable("playwright is not installed")

    monkeypatch.setattr(webshot, "nav_renderer", absent)
    png = _png(64, 64)

    def fetcher(url, **kw):
        if "api.microlink.io" in url:
            body = (b'{"data": {"screenshot": '
                    b'{"url": "https://img.example/s.png"}}}')
            return 200, body, ""
        return 200, png, ""

    result = webshot.capture("https://x.example/",
                             ShotOptions(), fetcher=fetcher)
    assert result.rung == "microlink"
    assert result.data == png


# ── the runtime handlers ──────────────────────────────────


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


def _fake_outcome(tmp_path, *, gates=(), wall=None):
    path = Path(tmp_path) / "rec.mp4"
    path.write_bytes(b"\x00" * 2048)
    nav = webnav.NavOutcome(
        final_url="https://x.example/", title="Real Page",
        gates=[webnav.GateAction(kind=k, label="ok",
                                 host="x.example")
               for k in gates],
        wall=wall, steps=[])
    return SimpleNamespace(
        path=path, fmt="mp4", duration_s=6.0, plan=None,
        nav=nav, includes_clearing=False, converted=True)


def test_webrec_handler_delivers_inline_video(
        tmp_path, monkeypatch):
    rig = _rig(tmp_path)
    monkeypatch.setattr(webshot, "_host_is_private",
                        lambda host: False)
    monkeypatch.setattr(
        webshot, "record_video",
        lambda url, opts, **kw: _fake_outcome(tmp_path))
    out = rig._control_webrec("6s https://shots.invalid/v",
                              _msg())
    assert "🎬" in out
    assert len(rig.gateway.files) == 1
    _p, _c, _path, _cap, kw = rig.gateway.files[0]
    assert kw.get("kind") == "video"
    assert kw.get("name", "").endswith(".mp4")
    book = rig._webshot_book()
    assert book.journal()[-1]["ok"] is True
    assert book.journal()[-1]["rung"] == "webrec"


def test_webrec_handler_wall_failure_is_honest(
        tmp_path, monkeypatch):
    rig = _rig(tmp_path)
    monkeypatch.setattr(webshot, "_host_is_private",
                        lambda host: False)

    def walled(url, opts, **kw):
        raise webnav.WallBlocked("bot wall (hard) needs a "
                                 "human hand", wall="hard")

    monkeypatch.setattr(webshot, "record_video", walled)
    out = rig._control_webrec("https://shots.invalid/v",
                              _msg())
    assert "couldn't record" in out
    assert "bot wall" in out
    assert rig.gateway.files == []
    book = rig._webshot_book()
    assert book.journal()[-1]["ok"] is False


def test_webrec_handler_messageless_line():
    rig = RuntimeGroupMixin()
    out = rig._control_webrec("https://x.example/", None)
    assert out.startswith("webrec answers with a video")


# ── gate targeting: the blocking layer's control ────────

# The real Pornhub shape (owner evidence, 2026-11-08):
# the gate is the age modal's "I'm 18 or older" button —
# served with a curly apostrophe — and there is NO 'ok'
# gate; the only 'ok' on the page is a cookie chip that
# is not what blocks the content.
PORNHUB_CURLY_PAGE = """<html><head><title>Adult Site</title></head>
<body><div class="age-modal" role="dialog"
style="position:fixed;inset:0">
<h2>This is an adult website</h2>
<p>You must be 18 or older to enter.</p>
<button id="age-enter">I’m 18 or older</button>
<button>Exit</button></div>
<div id="cookie-note"><p>We use cookies.</p>
<button id="cookie-ok">OK</button></div></body></html>"""

# What the age click lands on: the content, with the
# cookie chip still sitting on it (it was never the
# blocker, and clearing the gate doesn't clear it).
CONTENT_WITH_COOKIE_PAGE = """<html><head><title>Real Page</title></head>
<body><h1>The actual content</h1>
<div id="cookie-note"><p>We use cookies.</p>
<button id="cookie-ok">OK</button></div></body></html>"""

# A blocking consent wall + an age link that merely sits
# in the footer: the wall is what covers the page.
STACKED_LAYER_PAGE = """<html><head><title>Real Page</title></head>
<body><h1>The actual content</h1>
<div class="modal" role="dialog"><p>We value your
privacy. GDPR applies to you.</p>
<button id="consent-accept">Accept all</button></div>
<footer><a href="/age-check">18 or older - enter</a>
</footer></body></html>"""

# Both splash gates on one page: a consent wall in a
# modal layer and an age confirm beside it.
CONSENT_AND_AGE_PAGE = """<html><head><title>Splash</title></head>
<body><div class="modal" role="dialog"><p>We value your
privacy. GDPR applies to you.</p>
<button id="consent-accept">Accept all</button></div>
<div class="age-box"><h2>This is an adult website</h2>
<p>You must be 18 or older to enter.</p>
<button id="age-enter">I'm 18 or older</button></div>
</body></html>"""


class KindProbe(FakeProbe):
    """FakeProbe with per-kind click physics: kinds in
    ``refuse`` won't click; kinds in ``land`` land on the
    given page when clicked."""

    def __init__(self, html, *, refuse=(), land=None,
                 **kw):
        super().__init__(html, **kw)
        self._refuse = set(refuse)
        self._land = dict(land or {})

    def click_gate(self, gate):
        self.gate_clicks.append(gate.kind)
        if gate.kind in self._refuse:
            return False
        if gate.kind in self._land:
            self._html = self._land[gate.kind]
        return True


def test_detect_pornhub_curly_age_beats_stray_ok():
    gates = detect_gates(
        PORNHUB_CURLY_PAGE,
        "https://www.pornhub.com/view_video.php?viewkey=x")
    assert gates[0].kind == "age"
    assert gates[0].blocking is True
    assert "18 or older" in gates[0].label
    assert [g.kind for g in gates] == ["age", "cookie"]


def test_detect_blocking_layer_outranks_kind_order():
    gates = detect_gates(STACKED_LAYER_PAGE,
                         "https://x.example/")
    assert gates[0].kind == "consent"
    assert gates[0].blocking is True
    assert gates[1].kind == "age"
    assert gates[1].blocking is False


def test_pass_clicks_age_not_the_stray_ok():
    probe = KindProbe(
        PORNHUB_CURLY_PAGE,
        url="https://www.pornhub.com/view_video.php?viewkey=x",
        land={"age": CONTENT_WITH_COOKIE_PAGE,
              "cookie": CONTENT_PAGE})
    actions = webnav.run_interstitial_pass(probe)
    # The age control is worked FIRST (it is the blocking
    # layer); the stray 'ok' is just the cookie chip the
    # pass tidies afterwards.
    assert [a.kind for a in actions] == ["age", "cookie"]
    assert probe.gate_clicks == ["age", "cookie"]


def test_pass_moves_on_from_a_refused_control():
    probe = KindProbe(
        CONSENT_AND_AGE_PAGE, refuse={"consent"},
        land={"age": CONTENT_PAGE})
    actions = webnav.run_interstitial_pass(probe)
    assert [a.kind for a in actions] == ["age"]
    # The refused consent control was tried ONCE, then
    # the pass moved to the next candidate.
    assert probe.gate_clicks == ["consent", "age"]


def test_pass_journals_failed_attempts_when_blocked():
    probe = KindProbe(
        CONSENT_AND_AGE_PAGE, refuse={"consent"},
        land={"age": CONSENT_AND_AGE_PAGE})
    with pytest.raises(GateBlocked) as err:
        webnav.run_interstitial_pass(probe)
    # Each control was attempted exactly once — no
    # retrying a control that already failed.
    assert probe.gate_clicks == ["consent", "age"]
    assert "would not click" in str(err.value)
    assert "stayed up" in str(err.value)


# ── the fingerprint ladder ────────────────────────────────


def _ladder_opener(pages, record, holder=None):
    """An open_one for open_with_ladder: runs the REAL
    passes against the page scripted for each
    presentation, the way NavSession.open does."""

    def open_one(pres, hook):
        record.append((pres["name"], hook is not None))
        probe = pages[pres["name"]]
        if holder is not None:
            holder["probe"] = probe
        gates = webnav.run_interstitial_pass(probe)
        wall = run_wall_pass(probe, url="https://x.example/",
                             takeover_hook=hook)
        return probe, webnav.NavOutcome(
            final_url="https://x.example/",
            title=probe.title(), gates=gates, wall=wall,
            steps=[])

    return open_one


def _grid_pages(ladder, *, clear=()):
    pages = {}
    for pres in ladder:
        if pres["name"] in clear:
            pages[pres["name"]] = FakeProbe(CONTENT_PAGE)
        else:
            pages[pres["name"]] = FakeProbe(
                GRID_PAGE, title="Security Check")
    return pages


def test_ladder_races_and_the_clearer_wins():
    # The ladder RACES (owner directive 2026-11-08):
    # every fingerprint runs concurrently in its own
    # session; the clearer wins; every rung is journaled.
    ladder = webnav.presentation_ladder(None)
    pages = _grid_pages(ladder, clear={"ios-phone"})
    record = []
    hook_calls = []

    def hook(ask):
        hook_calls.append(ask)
        return True

    _session, outcome = webnav.open_with_ladder(
        "https://x.example/", ladder,
        _ladder_opener(pages, record), takeover_hook=hook)
    assert outcome.presentation == "ios-phone"
    assert outcome.presentations_tried == [
        p["name"] for p in ladder]
    assert outcome.rung_outcomes["ios-phone"] == "cleared"
    assert outcome.rung_outcomes["android-phone"] == \
        "wall_blocked"
    # The takeover hook rides ONLY the last rung — and
    # the race's gate kept it from firing at all: a
    # clearer had already won.
    deadline = time.time() + 5
    while len(record) < len(ladder) \
            and time.time() < deadline:
        time.sleep(0.02)
    assert {name for name, _h in record} == {
        p["name"] for p in ladder}
    assert [name for name, hooked in record
            if hooked] == ["desktop-windows"]
    assert hook_calls == []


def test_ladder_takeover_is_the_last_rung():
    ladder = webnav.presentation_ladder(None)
    pages = _grid_pages(ladder)
    record = []
    holder = {}
    hook_calls = []

    def hook(ask):
        hook_calls.append(ask)
        # The hook rides the desktop rung: clear THAT
        # rung's page (under the race, holder["probe"]
        # would be whichever thread wrote last).
        pages["desktop-windows"]._html = CONTENT_PAGE
        pages["desktop-windows"]._title = "Real Page"
        return True

    _session, outcome = webnav.open_with_ladder(
        "https://x.example/", ladder,
        _ladder_opener(pages, record, holder),
        takeover_hook=hook)
    assert outcome.wall == "owner_takeover"
    assert outcome.presentation == "desktop-windows"
    assert outcome.presentations_tried == [
        p["name"] for p in ladder]
    assert [name for name, hooked in record
            if hooked] == ["desktop-windows"]
    assert len(hook_calls) == 1


def test_ladder_exhausted_journals_every_fingerprint():
    ladder = webnav.presentation_ladder(None)
    pages = _grid_pages(ladder)
    record = []
    with pytest.raises(WallBlocked) as err:
        webnav.open_with_ladder(
            "https://x.example/", ladder,
            _ladder_opener(pages, record),
            takeover_hook=lambda ask: False)
    assert "android-phone" in str(err.value)
    assert "desktop-windows" in str(err.value)
    assert "none cleared" in str(err.value)
    assert "fingerprint ios-phone" in err.value.attempts


def test_ladder_race_when_the_gate_pass_cannot_clear():
    ladder = webnav.presentation_ladder(None)
    pages = _grid_pages(ladder, clear={"ios-phone"})
    pages["android-phone"] = FakeProbe(
        COOKIE_PAGE, click_works=False)
    record = []
    _session, outcome = webnav.open_with_ladder(
        "https://x.example/", ladder,
        _ladder_opener(pages, record),
        takeover_hook=lambda ask: True)
    assert outcome.presentation == "ios-phone"
    assert outcome.presentations_tried == [
        p["name"] for p in ladder]
    assert outcome.rung_outcomes["android-phone"] == \
        "gate_blocked"


def test_ladder_dob_refusal_is_the_verdict_when_nothing_clears():
    # A date-of-birth form is never answered on ANY
    # rung.  When no fingerprint clears, the DOB refusal
    # is the verdict that surfaces — the page's demand,
    # not a fingerprint's, is the refusal.
    ladder = webnav.presentation_ladder(None)
    pages = _grid_pages(ladder)  # all walled
    pages["android-phone"] = FakeProbe(DOB_PAGE)
    record = []
    with pytest.raises(GateBlocked) as err:
        webnav.open_with_ladder(
            "https://x.example/", ladder,
            _ladder_opener(pages, record),
            takeover_hook=lambda ask: False)
    assert "date of birth" in str(err.value)
    assert {name for name, _h in record} == {
        p["name"] for p in ladder}


def test_ladder_dob_rung_never_answered_clearer_wins():
    # Racing does not dodge the rule — no rung answers
    # the form — but a fingerprint the site lets through
    # without demanding a birth date still wins the
    # race, and the DOB rung's refusal is journaled.
    ladder = webnav.presentation_ladder(None)
    pages = _grid_pages(ladder, clear={"ios-phone"})
    pages["android-phone"] = FakeProbe(DOB_PAGE)
    record = []
    _session, outcome = webnav.open_with_ladder(
        "https://x.example/", ladder,
        _ladder_opener(pages, record),
        takeover_hook=lambda ask: True)
    assert outcome.presentation == "ios-phone"
    assert outcome.rung_outcomes["android-phone"] == \
        "dob_refused"


def test_presentation_ladders_are_coherent():
    for variant in (None, "phone", "tablet", "desktop"):
        ladder = webnav.presentation_ladder(variant)
        assert ladder, variant
        for pres in ladder:
            ua = pres["user_agent"]
            assert ua
            hints = pres["client_hints"]
            assert hints["mobile"] == pres["mobile"]
            kwargs = webnav.context_kwargs(pres)
            assert kwargs["user_agent"] == ua
            assert kwargs["viewport"] == {
                "width": pres["width"],
                "height": pres["height"]}
            assert kwargs["is_mobile"] == pres["mobile"]
            assert kwargs["has_touch"] == pres["mobile"]
            if pres["mobile"]:
                # A mobile platform family in the UA
                # (Android tablets legitimately omit the
                # "Mobile" token — the family must agree,
                # not one token).
                assert "Android" in ua or "iPhone" in ua \
                    or "iPad" in ua
            else:
                assert "Mobile" not in ua
                assert "Android" not in ua
    default = webnav.presentation_ladder(None)
    # Mobile first; a coherent desktop is the last rung.
    assert default[0]["name"] == "android-phone"
    assert default[0]["mobile"] is True
    assert default[-1]["mobile"] is False
    # A phone variant never wears a desktop viewport.
    phone = webnav.presentation_ladder("phone")
    assert all(p["mobile"] and p["width"] <= 430
               for p in phone)
    desktop = webnav.presentation_ladder("desktop")
    assert all(not p["mobile"] for p in desktop)


def test_session_default_presentation_is_mobile():
    session = webnav.NavSession()
    assert session.device["name"] == "android-phone"
    assert session.device["mobile"] is True
    assert "Android" in session.device["user_agent"]


def test_device_variants_are_chosen_not_defaulted():
    _url, opts = webshot.parse_shot_tail(
        "https://x.example/")
    assert opts.device_chosen is False
    _url, opts = webshot.parse_shot_tail(
        "phone https://x.example/")
    assert opts.device == "phone"
    assert opts.device_chosen is True
    _url, opts = webshot.parse_rec_tail(
        "https://x.example/")
    assert opts.device_chosen is False
    _url, opts = webshot.parse_rec_tail(
        "desktop https://x.example/")
    assert opts.device == "desktop"
    assert opts.device_chosen is True


def test_webshot_reply_names_gates_and_wall(
        tmp_path, monkeypatch):
    rig = _rig(tmp_path)
    monkeypatch.setattr(webshot, "_host_is_private",
                        lambda host: False)

    def fake_capture(url, opts, **kw):
        return CaptureResult(
            data=_png(64, 64), title="Real Page",
            final_url=url, rung="playwright", width=64,
            height=64, gates_cleared=["cookie", "age"],
            wall_via="checkbox_click")

    monkeypatch.setattr(webshot, "capture", fake_capture)
    out = rig._control_webshot("https://shots.invalid/page",
                               _msg())
    assert "cleared on the way in: cookie, age" in out
    assert "bot check passed via checkbox_click" in out

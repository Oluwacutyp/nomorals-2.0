"""Pass #79 — captcha estate II: every remaining kind, its mechanism.

The estate after #78 had exactly one local solver (the slider). This
suite pins the rest of the owner's order ("update all captcha solver
for all sorts of captcha"):

* image_captcha → the OCR solver (tesseract wrapper as first reader,
  the vision stack as second reader below the confidence floor),
  typed + submitted in-session; the same core registered into
  captcha_flow's OCR stage so non-browser flows can use it;
* audio_captcha → the local ASR path (solve_audio_local_first)
  registered as a kind solver: payload bytes fetched through the
  session, transcribed, typed, outcome read;
* click-selection (reCAPTCHA v2 grids, hCaptcha, GeeTest v4 click)
  → the vision-grounding solver: clicks land exactly on the cells
  the grounder chose, verify is pressed, a fresh grid is a capped
  retry, a rejection is a failure, and NO vision means NO clicks;
* cloudflare → the download kit's CloudflareStateMachine driven for
  the first time (managed wait, vault-cookie injection, owner ask,
  expiry), plus the downloader-side gate;
* question_captcha → the brain-answer solver (a kind the estate
  never had: detection + solve), honest when no brain answers;
* friendly → PINNED as covered by the ladder's wait rung (no build);
* routing/status → every kind appears in the estate status map
  with a mechanism and a probed availability.

No network, no live third-party captchas: fake tabs, stubbed
readers/ASR/grounders/brains at the solvers' real injection seams.
"""
from __future__ import annotations

import io

import pytest
from PIL import Image, ImageDraw

from nomorals.browser import captcha_estate  # noqa: F401 - registrations
from nomorals.browser.audio_solver import (
    AudioCaptchaSolver,
    LocalAudioBackend,
    audio_asr_status,
)
from nomorals.browser.challenge import (
    ChallengeSolver,
    detect_challenge_signal,
)
from nomorals.browser.click_solver import (
    ClickSolver,
    solve_geetest_kind,
)
from nomorals.browser.cloudflare_solver import CloudflareSolver
from nomorals.browser.kind_solvers import kind_solver_for, registered_kinds
from nomorals.browser.ocr_solver import (
    LocalOcrBackend,
    ImageCaptchaSolver,
    ocr_status,
    read_captcha_text,
)
from nomorals.browser.question_solver import (
    QUESTION_KIND,
    QuestionSolver,
    detect_question_challenge,
)
from nomorals.media.cloudflare_gate import CloudflareGate
from nomorals.tools.captcha import CaptchaChallenge, CaptchaKind
from nomorals.tools.captcha_flow import CaptchaFlow

# ── shared fixtures ──────────────────────────────────────────────────────

ANSWER = "K7QX2"
CLEAN_HTML = ("<html><head><title>done</title></head><body>"
              "<h1>welcome in</h1></body></html>")


def make_captcha_png(text: str = ANSWER, *, distort: int = 0) -> bytes:
    """A synthetic text-captcha image: known string, PIL-rendered."""
    img = Image.new("RGB", (220, 70), "white")
    draw = ImageDraw.Draw(img)
    draw.text((25, 22), text, fill="black")
    if distort:
        px = img.load()
        for x in range(0, 220, 7):
            for y in range(70):
                if (x + y) % 13 == 0:
                    px[x, y] = (180, 180, 180)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── (a) OCR solver ───────────────────────────────────────────────────────

IMAGE_HTML = """<html><body><form action="/check" method="post">
<img id="cap-img" src="https://x.test/captcha.png" alt="captcha image">
<label>Enter the code</label>
<input type="text" id="cap-in" name="captcha">
</form></body></html>"""


class FakeImageTab:
    """One image captcha; clears iff the right text is typed + Enter."""

    def __init__(self, image: bytes, answer: str = ANSWER) -> None:
        self._image = image
        self._answer = answer
        self._pending = ""
        self.cleared = False
        self.typed: list[tuple[str, str]] = []
        self.fetched: list[str] = []

    @property
    def url(self) -> str:
        return "https://x.test/form"

    def html(self) -> str:
        return CLEAN_HTML if self.cleared else IMAGE_HTML

    def session_fetch_bytes(self, url: str) -> bytes:
        self.fetched.append(url)
        return self._image

    def element_image_bytes(self, selector: str) -> bytes:
        return self._image

    def type_text(self, target: str, text: str) -> None:
        self.typed.append((target, text))
        self._pending = text

    def press_key(self, key: str) -> None:
        if key == "Enter" and self._pending == self._answer:
            self.cleared = True


def _hi_conf_reader(_data: bytes) -> dict:
    return {"text": ANSWER, "mean_conf": 93.0}


def test_ocr_solver_types_the_stubbed_read_and_clears():
    tab = FakeImageTab(make_captcha_png())
    solver = ImageCaptchaSolver(
        read_fn=_hi_conf_reader, probe=lambda: (True, "stubbed"),
        sleeper=lambda _s: None)
    out = solver.solve_tab(tab)
    assert out["ok"], out
    assert tab.typed == [("#cap-in", ANSWER)]
    assert tab.cleared


def test_ocr_low_confidence_hands_off_to_the_second_reader():
    calls: list[str] = []
    solver = ImageCaptchaSolver(
        read_fn=lambda _d: {"text": "K7OX2", "mean_conf": 21.0},
        second_read_fn=lambda _d: calls.append("vision") or ANSWER,
        probe=lambda: (True, "stubbed"), sleeper=lambda _s: None)
    tab = FakeImageTab(make_captcha_png(distort=1))
    out = solver.solve_tab(tab)
    assert out["ok"], out
    assert calls == ["vision"]
    assert tab.typed == [("#cap-in", ANSWER)]  # the vision read, not the guess


def test_read_captcha_text_never_types_a_low_confidence_guess():
    out = read_captcha_text(
        make_captcha_png(),
        read_fn=lambda _d: {"text": "ZZZZZ", "mean_conf": 10.0},
        second_read_fn=lambda _d: "")
    assert not out["ok"]
    assert "below" in out["detail"]


def test_ocr_real_probe_is_honored_no_typing_when_unavailable():
    """The sandbox has no tesseract: the solver must SAY unavailable
    and never touch the page. (When a machine HAS tesseract, the
    probe agrees with ocr_status instead.)"""
    status = ocr_status()
    tab = FakeImageTab(make_captcha_png())
    solver = ImageCaptchaSolver(sleeper=lambda _s: None)
    available, _reason = solver.available()
    assert available == bool(status["available"])
    if not status["available"]:
        out = solver.solve_tab(tab)
        assert not out["ok"]
        assert "tesseract" in out["detail"]
        assert tab.typed == [] and not tab.cleared


def test_ocr_backend_solves_from_bytes_and_refuses_other_kinds():
    backend = LocalOcrBackend(
        solver=ImageCaptchaSolver(read_fn=_hi_conf_reader,
                                   probe=lambda: (True, "stubbed")))
    res = backend.solve(CaptchaChallenge(
        kind=CaptchaKind.IMAGE_CAPTCHA, image_bytes=make_captcha_png()))
    assert res.ok and res.text == ANSWER
    refused = backend.solve(CaptchaChallenge(kind=CaptchaKind.HCAPTCHA))
    assert not refused.ok and "image" in refused.detail


def test_flow_ocr_stage_uses_the_registered_default(monkeypatch):
    """captcha_flow stage 2 runs the estate's OCR fn by default now;
    a registered override proves the slot, and the real default
    proves the stage is attempted honestly without tesseract."""
    from nomorals.tools import captcha_flow
    monkeypatch.setattr(
        captcha_flow, "_DEFAULT_OCR_FN",
        lambda ch: {"ok": True, "text": ANSWER})
    out = CaptchaFlow().solve({"kind": "image_captcha",
                               "image_bytes": make_captcha_png(),
                               "page_url": "https://x.test/form"})
    assert out["ok"] and out["stage"] == "ocr" and out["text"] == ANSWER

    monkeypatch.setattr(captcha_flow, "_DEFAULT_OCR_FN", None)
    out2 = CaptchaFlow().solve({"kind": "image_captcha",
                                "image_bytes": make_captcha_png(),
                                "page_url": "https://x.test/form"})
    assert "ocr" in out2["stages_tried"]  # attempted, honestly failed
    assert not out2["ok"]


# ── (b) audio solver ─────────────────────────────────────────────────────

AUDIO_MAIN_HTML = """<html><body>
<iframe id="bf" title="recaptcha challenge"
 src="https://www.google.com/recaptcha/api2/bframe?k=KEY"></iframe>
</body></html>"""

AUDIO_FRAME_HTML = """<html><body>
<div class="rc-audiochallenge">
 <a id="audio-source" href="https://www.google.com/recaptcha/api2/payload/audio?p=AbC123">Download audio</a>
 <input id="audio-response" type="text" placeholder="Type what you hear">
 <button id="recaptcha-verify-button">Verify</button>
</div></body></html>"""

AUDIO_URL = "https://www.google.com/recaptcha/api2/payload/audio?p=AbC123"
BFRAME_URL = "https://www.google.com/recaptcha/api2/bframe?k=KEY"


class FakeAudioTab:
    def __init__(self) -> None:
        self.cleared = False
        self.fetched: list[str] = []
        self.set_calls: list[dict] = []
        self.verify_calls = 0
        self._heard = ""

    @property
    def url(self) -> str:
        return "https://x.test/form"

    def html(self) -> str:
        return CLEAN_HTML if self.cleared else AUDIO_MAIN_HTML

    def frames_html(self) -> list[dict]:
        if self.cleared:
            return []
        return [{"url": self.url, "html": self.html()},
                {"url": BFRAME_URL, "html": AUDIO_FRAME_HTML}]

    def session_fetch_bytes(self, url: str) -> bytes:
        self.fetched.append(url)
        return b"FAKE-AUDIO-BYTES"

    def frame_evaluate(self, url_substr: str, js: str, arg=None):
        assert "bframe" in url_substr
        if isinstance(arg, dict) and "text" in arg:
            self.set_calls.append(dict(arg))
            self._heard = str(arg.get("text") or "")
            return True
        if "verify" in js:
            self.verify_calls += 1
            if self._heard == "seven two":
                self.cleared = True
            return "#recaptcha-verify-button"
        return None


def test_audio_solver_fetches_transcribes_types_and_clears():
    tab = FakeAudioTab()
    solver = AudioCaptchaSolver(
        transcribe_fn=lambda audio: {"ok": True, "backend": "stub-asr",
                                     "text": "seven two"},
        probe=lambda: (True, "stubbed"), sleeper=lambda _s: None)
    out = solver.solve_tab(tab)
    assert out["ok"], out
    assert tab.fetched == [AUDIO_URL]  # payload fetched via the session
    assert tab.set_calls and tab.set_calls[0]["text"] == "seven two"
    assert tab.verify_calls == 1
    assert tab.cleared


def test_audio_unavailable_falls_through_without_touching_the_page():
    tab = FakeAudioTab()
    solver = AudioCaptchaSolver(
        probe=lambda: (False, "local audio solving unavailable — "
                              "pywhispercpp not installed"))
    out = solver.solve_tab(tab)
    assert not out["ok"]
    assert "pywhispercpp" in out["detail"]
    assert tab.fetched == [] and tab.set_calls == []


def test_audio_real_probe_matches_status():
    available, _ = AudioCaptchaSolver().available()
    assert available == bool(audio_asr_status()["available"])


def test_audio_backend_transcribes_challenge_bytes():
    backend = LocalAudioBackend(solver=AudioCaptchaSolver(
        transcribe_fn=lambda audio: {"ok": True, "text": "nine five"},
        probe=lambda: (True, "stubbed")))
    res = backend.solve(CaptchaChallenge(
        kind=CaptchaKind.AUDIO_CAPTCHA, image_bytes=b"AUDIO"))
    assert res.ok and res.text == "nine five"
    refused = backend.solve(CaptchaChallenge(kind=CaptchaKind.IMAGE_CAPTCHA,
                                             image_bytes=b"AUDIO"))
    assert not refused.ok


# ── (c) click solver ─────────────────────────────────────────────────────

CLICK_MAIN_HTML = """<html><body>
<iframe id="bf" title="recaptcha challenge"
 src="https://www.google.com/recaptcha/api2/bframe?k=KEY"></iframe>
</body></html>"""

CLICK_FRAME_TMPL = """<html><body>
<div class="rc-imageselect">
 <div class="rc-imageselect-desc-no-canonical">Select all images with
 <strong>buses</strong></div>
 <div class="rc-imageselect-challenge">
  <div class="rc-image-tile-wrapper"></div>
 </div>
 {extra}
</div></body></html>"""

GROUND = {"rows": 3, "cols": 3,
          "grid": {"x": 0.0, "y": 100.0, "w": 1000.0, "h": 800.0},
          "cells": [(0, 0), (1, 2)], "verify": (950.0, 950.0)}


class FakeClickTab:
    """Scripted rounds: verify-click outcomes come from `script`
    ('clear' | 'newgrid' | 'reject' | 'stay')."""

    BOX = {"x": 100.0, "y": 50.0, "width": 400.0, "height": 500.0}

    def __init__(self, script: list[str]) -> None:
        self._script = list(script)
        self.round = 0
        self.rejected = False
        self.cleared = False
        self.clicks: list[tuple[float, float]] = []
        self.captures = 0

    @property
    def url(self) -> str:
        return "https://x.test/form"

    @property
    def _image(self) -> bytes:
        return f"GRID-IMAGE-{self.round}".encode()

    def html(self) -> str:
        return CLEAN_HTML if self.cleared else CLICK_MAIN_HTML

    def frames_html(self) -> list[dict]:
        if self.cleared:
            return []
        extra = ('<div class="rc-imageselect-incorrect-response">'
                 "Please try again.</div>" if self.rejected else "")
        frame = CLICK_FRAME_TMPL.format(extra=extra)
        return [{"url": self.url, "html": self.html()},
                {"url": BFRAME_URL, "html": frame}]

    def element_image_bytes(self, selector: str) -> bytes:
        assert selector == "#bf"
        self.captures += 1
        return self._image

    def element_box(self, selector: str) -> dict:
        return dict(self.BOX)

    def click_at_px(self, x: float, y: float) -> None:
        self.clicks.append((float(x), float(y)))
        verify_at = (self.BOX["x"] + 0.95 * self.BOX["width"],
                     self.BOX["y"] + 0.95 * self.BOX["height"])
        if (x, y) == pytest.approx(verify_at):
            action = self._script[min(self.round, len(self._script) - 1)]
            if action == "clear":
                self.cleared = True
            elif action == "newgrid":
                self.round += 1
            elif action == "reject":
                self.rejected = True
            # 'stay' — nothing changes


def _grounder_factory(seen: list[bytes]):
    def ground(image: bytes, prompt: str) -> dict:
        seen.append(image)
        assert "buses" in prompt
        return dict(GROUND)
    return ground


def _expected_points() -> list[tuple[float, float]]:
    box = FakeClickTab.BOX
    pts = []
    for (r, c) in GROUND["cells"]:
        fx = (GROUND["grid"]["x"] + GROUND["grid"]["w"] * (c + 0.5) / 3) / 1000
        fy = (GROUND["grid"]["y"] + GROUND["grid"]["h"] * (r + 0.5) / 3) / 1000
        pts.append((box["x"] + fx * box["width"], box["y"] + fy * box["height"]))
    pts.append((box["x"] + 0.95 * box["width"],
                box["y"] + 0.95 * box["height"]))
    return pts


def test_click_solver_clicks_the_grounded_cells_and_verify():
    seen: list[bytes] = []
    tab = FakeClickTab(["clear"])
    solver = ClickSolver(ground_fn=_grounder_factory(seen),
                         probe=lambda: (True, "stubbed"),
                         sleeper=lambda _s: None)
    out = solver.solve_tab(tab)
    assert out["ok"], out
    assert tab.clicks == [pytest.approx(p) for p in _expected_points()]
    assert seen == [b"GRID-IMAGE-0"]


def test_click_solver_retries_a_fresh_grid_then_clears():
    seen: list[bytes] = []
    tab = FakeClickTab(["newgrid", "clear"])
    solver = ClickSolver(ground_fn=_grounder_factory(seen),
                         probe=lambda: (True, "stubbed"),
                         sleeper=lambda _s: None)
    out = solver.solve_tab(tab)
    assert out["ok"] and out["rounds"] == 2
    assert seen == [b"GRID-IMAGE-0", b"GRID-IMAGE-1"]  # re-grounded fresh


def test_click_solver_rejection_is_a_failure_not_a_retry():
    tab = FakeClickTab(["reject"])
    solver = ClickSolver(ground_fn=_grounder_factory([]),
                         probe=lambda: (True, "stubbed"),
                         sleeper=lambda _s: None)
    out = solver.solve_tab(tab)
    assert not out["ok"] and out["rounds"] == 1
    assert "rejected" in out["detail"]


def test_click_solver_unchanged_grid_is_a_rejection():
    tab = FakeClickTab(["stay"])
    solver = ClickSolver(ground_fn=_grounder_factory([]),
                         probe=lambda: (True, "stubbed"),
                         sleeper=lambda _s: None)
    out = solver.solve_tab(tab)
    assert not out["ok"]
    assert "unchanged" in out["detail"]


def test_click_solver_round_cap_is_named():
    tab = FakeClickTab(["newgrid"])  # endless fresh grids
    solver = ClickSolver(ground_fn=_grounder_factory([]),
                         probe=lambda: (True, "stubbed"),
                         sleeper=lambda _s: None)
    out = solver.solve_tab(tab)
    assert not out["ok"] and out["rounds"] == 3
    assert "cap" in out["detail"]


def test_click_solver_no_vision_means_zero_clicks():
    tab = FakeClickTab(["clear"])
    solver = ClickSolver(probe=lambda: (False, "vision unavailable: "
                                                  "no provider"),
                         sleeper=lambda _s: None)
    out = solver.solve_tab(tab)
    assert not out["ok"]
    assert "vision unavailable" in out["detail"]
    assert tab.clicks == [] and tab.captures == 0


def test_click_solver_ungroundable_image_means_zero_clicks():
    tab = FakeClickTab(["clear"])
    solver = ClickSolver(ground_fn=lambda _i, _p: None,
                         probe=lambda: (True, "stubbed"),
                         sleeper=lambda _s: None)
    out = solver.solve_tab(tab)
    assert not out["ok"]
    assert tab.clicks == []


# ── GeeTest dispatch ─────────────────────────────────────────────────────

GEETEST_SLIDER_HTML = """<html><body>
<div class="geetest_holder"><div class="geetest_widget">
<canvas class="geetest_canvas_bg"></canvas>
<canvas class="geetest_canvas_slice_bg"></canvas>
<div class="geetest_slider"><div class="geetest_slider_button"></div></div>
</div></div></body></html>"""

GEETEST_CLICK_HTML = """<html><body>
<div class="geetest_holder"><div class="geetest_box">
<div class="geetest_item"></div><div class="geetest_item"></div>
<p class="geetest_tip">Please click the matching icons</p>
</div></div></body></html>"""


class FakeGeetestTab:
    def __init__(self, html: str) -> None:
        self._html = html
        self.clicks: list[tuple[float, float]] = []
        self.cleared = False

    @property
    def url(self) -> str:
        return "https://x.test/geetest"

    def html(self) -> str:
        return CLEAN_HTML if self.cleared else self._html

    def element_image_bytes(self, selector: str) -> bytes:
        return b"GEETEST-GRID"

    def element_box(self, selector: str) -> dict:
        return {"x": 0.0, "y": 0.0, "width": 300.0, "height": 300.0}

    def click_at_px(self, x: float, y: float) -> None:
        self.clicks.append((float(x), float(y)))
        if len(self.clicks) >= 2:
            self.cleared = True


def test_geetest_dispatcher_routes_slider_markup_to_slider_solver(
        monkeypatch):
    from nomorals.browser import slider_solver
    calls: list = []
    monkeypatch.setattr(
        slider_solver, "solve_slider_kind",
        lambda tab, ch: calls.append(ch) or {"ok": True,
                                            "detail": "slider route"})
    tab = FakeGeetestTab(GEETEST_SLIDER_HTML)
    out = solve_geetest_kind(tab, {"kind": "geetest"})
    assert out["ok"] and out["detail"] == "slider route"
    assert len(calls) == 1


def test_geetest_dispatcher_routes_click_markup_to_click_solver(
        monkeypatch):
    from nomorals.browser import click_solver
    monkeypatch.setattr(
        click_solver.ClickSolver, "solve_tab",
        lambda self, tab, ch=None: {"ok": True, "backend": "local_click",
                                     "detail": "click route", "rounds": 1})
    tab = FakeGeetestTab(GEETEST_CLICK_HTML)
    out = solve_geetest_kind(tab, {"kind": "geetest"})
    assert out["ok"] and out["detail"] == "click route"


# ── (d) cloudflare ───────────────────────────────────────────────────────

CF_HTML = """<html><head><title>Just a moment...</title></head><body>
<div id="challenge-running">Checking your browser before accessing
x.test.</div><p>challenges.cloudflare.com</p></body></html>"""


class FakeStore:
    """The kit's duck-typed cookie store: get/set."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    def get(self, key: str):
        return self.data.get(key)

    def set(self, key: str, value: str) -> None:
        self.data[key] = value


class FakeCloudflareTab:
    def __init__(self, *, clears_on_wait: bool = False,
                 cookie_value: str = "") -> None:
        self.challenged = True
        self._clears_on_wait = clears_on_wait
        self._cookie_value = cookie_value
        self.added: list[dict] = []
        self.reloads = 0
        self.html_calls = 0

    @property
    def url(self) -> str:
        return "https://x.test/article"

    def html(self) -> str:
        self.html_calls += 1
        if self._clears_on_wait and self.html_calls > 1:
            self.challenged = False
        return CF_HTML if self.challenged else CLEAN_HTML

    def cookies(self) -> list[dict]:
        if self.challenged:
            return []
        return [{"name": "cf_clearance", "value": "FRESH-CLEARANCE",
                 "domain": ".x.test"}]

    def add_cookies(self, cookies: list[dict]) -> None:
        self.added.extend(cookies)
        for c in cookies:
            if c.get("name") == "cf_clearance" and \
                    c.get("value") == self._cookie_value:
                self.challenged = False

    def reload(self) -> None:
        self.reloads += 1


def _cf_solver(store=None, **kw) -> CloudflareSolver:
    kw.setdefault("wait_s", 0.0)
    kw.setdefault("sleeper", lambda _s: None)
    kw.setdefault("clock", lambda: 0.0)
    return CloudflareSolver(cookie_store=store, **kw)


def test_cloudflare_managed_clear_captures_clearance_to_vault():
    store = FakeStore()
    tab = FakeCloudflareTab(clears_on_wait=True)
    solver = _cf_solver(store)
    out = solver.solve_tab(tab)
    assert out["ok"], out
    assert out["state"] == "passed"
    entry = store.data.get("cloudflare_cookies:x.test")
    assert entry is not None
    assert entry["value"] == "FRESH-CLEARANCE"  # the kit's entry shape


def test_cloudflare_vault_cookie_injection_clears():
    store = FakeStore()
    from nomorals.media.download_kit import store_owner_cookie
    store_owner_cookie(store, "x.test", "STORED-COOKIE")
    tab = FakeCloudflareTab(cookie_value="STORED-COOKIE")
    solver = _cf_solver(store)
    out = solver.solve_tab(tab)
    assert out["ok"], out
    assert tab.added and tab.added[0]["value"] == "STORED-COOKIE"
    assert tab.reloads == 1
    assert out["transitions"][-1][1] == "passed"


def test_cloudflare_ask_path_states_the_kit_message():
    store = FakeStore()  # empty: nothing stored → the kit asks
    tab = FakeCloudflareTab()
    solver = _cf_solver(store)
    out = solver.solve_tab(tab)
    assert not out["ok"]
    assert "cf_clearance" in out["detail"]
    assert out["state"] == "waiting_cookie"
    assert tab.added == []


def test_cloudflare_expired_cookie_is_terminal_and_named():
    store = FakeStore()
    from nomorals.media.download_kit import store_owner_cookie
    store_owner_cookie(store, "x.test", "STALE", ttl_s=-5)
    tab = FakeCloudflareTab(cookie_value="STALE")
    solver = _cf_solver(store)
    out = solver.solve_tab(tab)
    assert not out["ok"]
    assert "expired" in out["detail"]
    assert tab.added == []


def test_cloudflare_gate_for_the_downloader():
    store = FakeStore()
    from nomorals.media.download_kit import store_owner_cookie
    gate = CloudflareGate(store)
    assert not gate.inspect("<html><body>an ordinary page</body></html>")
    assert gate.inspect(CF_HTML)
    ask = gate.cookie_decision("https://x.test/a")
    assert ask["action"] == "ask" and "cf_clearance" in ask["message"]
    store_owner_cookie(store, "x.test", "COOKIE-9")
    have = CloudflareGate(store)
    assert have.inspect(CF_HTML)
    decision = have.cookie_decision("https://x.test/a")
    assert decision["action"] == "inject"
    assert decision["cookie"] == "cf_clearance=COOKIE-9"
    have.resolve(True)
    assert have.state == "passed"


# ── (e) question solver ──────────────────────────────────────────────────

QUESTION_HTML = """<html><body>
<div class="security-check"><p>Captcha — security question:
What is 2 + 3?</p>
<input type="text" id="qa" name="captcha_answer">
</div></body></html>"""

PROSE_HTML = ("<html><body><article><p>What is 2 + 3? A question "
              "every child learns to answer.</p></article></body></html>")


class FakeQuestionTab:
    def __init__(self) -> None:
        self.cleared = False
        self.typed: list[tuple[str, str]] = []
        self._pending = ""

    @property
    def url(self) -> str:
        return "https://x.test/join"

    def html(self) -> str:
        return CLEAN_HTML if self.cleared else QUESTION_HTML

    def type_text(self, target: str, text: str) -> None:
        self.typed.append((target, text))
        self._pending = text

    def press_key(self, key: str) -> None:
        if key == "Enter" and self._pending == "5":
            self.cleared = True


def test_question_detection_needs_a_form_and_context():
    info = detect_question_challenge(QUESTION_HTML)
    assert info is not None
    assert info["question"] == "What is 2 + 3?"
    assert info["input_selector"] == "#qa"  # id wins, as everywhere
    assert detect_question_challenge(PROSE_HTML) is None
    assert detect_question_challenge(CLEAN_HTML) is None


def test_question_solver_brain_answers_types_and_clears():
    tab = FakeQuestionTab()
    solver = QuestionSolver(answer_fn=lambda q: "5",
                            probe=lambda: (True, "brain bound"),
                            sleeper=lambda _s: None)
    out = solver.solve_tab(tab)
    assert out["ok"], out
    assert tab.typed == [("#qa", "5")]
    assert tab.cleared


def test_question_solver_no_brain_is_honest_and_types_nothing():
    tab = FakeQuestionTab()
    solver = QuestionSolver(probe=lambda: (False, "no brain bound"))
    out = solver.solve_tab(tab)
    assert not out["ok"]
    assert tab.typed == []


def test_flow_names_the_question_kind_from_html():
    pings: list[dict] = []
    out = CaptchaFlow(ping_owner=pings.append).solve(
        {"kind": "unknown", "html": QUESTION_HTML,
         "page_url": "https://x.test/join"})
    assert out["kind"] == QUESTION_KIND or \
        (pings and pings[0]["kind"] == QUESTION_KIND)
    assert out["kind"] == QUESTION_KIND


def test_ladder_signal_names_the_question_kind():
    sig = detect_challenge_signal(QUESTION_HTML, "https://x.test/join")
    assert sig is not None and sig.kind == QUESTION_KIND


# ── (f) friendly: covered by the ladder's wait rung (pin) ────────────────

FRIENDLY_HTML = """<html><body><form>
<div class="frc-captcha" data-sitekey="FCMST8IMGNOVJCQO"></div>
</form></body></html>"""


class FakeFriendlyTab:
    def __init__(self) -> None:
        self.cleared = False

    @property
    def url(self) -> str:
        return "https://x.test/contact"

    @property
    def title(self) -> str:
        return "contact"

    def html(self) -> str:
        return CLEAN_HTML if self.cleared else FRIENDLY_HTML

    def reload(self) -> None:
        pass


def test_friendly_clears_via_the_wait_rung_without_a_local_solver():
    tab = FakeFriendlyTab()

    def sleeper(_s: float) -> None:
        tab.cleared = True  # the PoW widget finished during the wait

    solver = ChallengeSolver(sleeper=sleeper, wait_s=3.0)
    out = solver.solve(tab=tab)
    assert out.cleared, out
    assert out.via == "wait_proof_of_work"
    strategies = [a.strategy for a in out.attempts]
    assert "local_solver" not in strategies


# ── (g) routing + status ─────────────────────────────────────────────────

def test_every_kind_solver_is_registered():
    kinds = set(registered_kinds())
    assert {"slider", "geetest", "image_captcha", "audio_captcha",
            "recaptcha_v2", "hcaptcha", "cloudflare",
            QUESTION_KIND} <= kinds
    for kind in kinds:
        assert kind_solver_for(kind) is not None
    # Deliberate non-registration (see cloudflare_solver's
    # docstring): the ladder owns the "cloudflare-challenge" wall
    # natively — a registry entry would preempt its takeover path.
    assert "cloudflare-challenge" not in kinds
    assert kind_solver_for("cloudflare-challenge") is None


def test_kind_solvers_dispatch_through_the_registry():
    tab = FakeImageTab(make_captcha_png())
    fn = kind_solver_for("image_captcha")
    assert fn is not None
    out = fn(tab, {"kind": "image_captcha"})
    # Default solver, real probe: on this machine tesseract is
    # absent, so dispatch reaches the solver and reports its
    # availability truth instead of raising or faking.
    assert out["backend"] == "local_ocr"
    assert out["ok"] == bool(ocr_status()["available"] and out["ok"])


def test_estate_status_covers_every_kind_with_a_mechanism():
    status = captcha_estate.estate_status()
    kinds = status["kinds"]
    for kind in list(CaptchaKind.ALL) + ["slider", "cloudflare",
                                         "cloudflare-challenge",
                                         QUESTION_KIND]:
        assert kind in kinds, kind
        row = kinds[kind]
        assert row["mechanism"] and "available" in row and row["detail"]
    assert kinds["friendly"]["mechanism"] == "wait"
    assert "score-based" in kinds["recaptcha_v3"]["detail"]
    assert kinds["recaptcha_v3"]["local_solver"] == ""  # no fake local
    assert kinds["image_captcha"]["local_solver"] == "local_ocr"
    assert kinds["image_captcha"]["available"] == \
        bool(ocr_status()["available"])
    assert kinds["audio_captcha"]["available"] == \
        bool(audio_asr_status()["available"])
    assert kinds[QUESTION_KIND]["local_solver"] == "local_brain"
    assert kinds["cloudflare"]["registered"] is True
    assert kinds["cloudflare-challenge"]["registered"] is False
    assert "ladder" in kinds["cloudflare-challenge"]["detail"]
    assert set(status["registered_kinds"]) >= {"image_captcha",
                                               "audio_captcha"}


def test_cli_status_carries_the_kind_map(capsys):
    import argparse

    from nomorals.cmdline.commands.captcha import _cmd_captcha

    class _Ctx:
        settings = None

    args = argparse.Namespace(captcha_action="status", json=True,
                              url=None, html=None)
    rc = _cmd_captcha(args, _Ctx())
    assert rc == 0
    import json
    payload = json.loads(capsys.readouterr().out)
    assert "local_slider" in payload  # the #78 key survives
    assert "kind_map" in payload
    assert payload["kind_map"]["friendly"]["mechanism"] == "wait"
    assert "local_ocr" in payload["backends"]
    assert "local_click" in payload["backends"]


def test_clean_pages_are_unaffected():
    assert detect_challenge_signal(CLEAN_HTML, "https://x.test/") is None
    assert detect_question_challenge(CLEAN_HTML) is None
    out = CaptchaFlow().solve({"kind": "unknown", "html": CLEAN_HTML,
                               "page_url": "https://x.test/"})
    assert out["ok"] and out["kind"] == "no_challenge"

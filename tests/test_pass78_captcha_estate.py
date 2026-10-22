"""Pass #78: the captcha estate — local in-session slider solver.

Offline, no network, no live TikTok. Coverage:

(a) gap detection on SYNTHETIC puzzles (two-image + single-image
    shapes, clean + degraded);
(b) trajectory synthesis properties (human shape, seed-deterministic);
(c) routing order — local solver before service before takeover, in
    both the browser ChallengeSolver ladder and the captcha_flow
    pipeline; unavailable local solver falls through with its reason;
(d) the execution seam — a fake tab records the drag; the outcome read
    (challenge gone vs still present) is honored, retry rule respected,
    and a low-confidence gap NEVER produces a drag;
(e) detection — TikTok-style slider markup classified as a slider
    challenge; clean pages are not;
(f) the registered ``local_slider`` backend + honest status surfaces;
(g) the signup driver's injected local attempt (composition seam).
"""
from __future__ import annotations

import asyncio
import io
import json
import random

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("PIL")

from PIL import Image, ImageFilter  # noqa: E402

from nomorals.browser import slider_solver as ss  # noqa: E402
from nomorals.browser.challenge import (  # noqa: E402
    ChallengeSolver,
    detect_challenge_signal,
)
from nomorals.browser.slider_solver import (  # noqa: E402
    LocalSliderBackend,
    LocalSliderSolver,
    detect_gap,
    detect_slider_challenge,
    local_slider_status,
    synthesize_trajectory,
    vision_status,
)

# ── synthetic puzzle factory ─────────────────────────────────────────────

W, H = 340, 212
PIECE = 54
GAP_X, GAP_Y = 208, 76
PIECE_X = 18


def _photo(w: int = W, h: int = H, seed: int = 7) -> np.ndarray:
    """A photo-like RGB array: smooth colour field + texture + blobs."""
    rng = np.random.default_rng(seed)
    small = rng.integers(60, 200, size=(12, 20, 3), dtype=np.uint8)
    img = Image.fromarray(small).resize((w, h), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float64)
    # a few saturated blobs so the scene has real edges everywhere
    for _ in range(14):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        r = int(rng.integers(9, 30))
        yy, xx = np.mgrid[0:h, 0:w]
        m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        arr[m] = rng.integers(30, 235, size=3)
    arr += rng.normal(0, 5, size=arr.shape)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _piece_mask(w: int = PIECE, h: int = PIECE) -> np.ndarray:
    """Puzzle-piece silhouette: rounded square + a knob on top."""
    yy, xx = np.mgrid[0:h, 0:w]
    mask = (xx >= 3) & (xx < w - 3) & (yy >= 10) & (yy < h - 2)
    knob = (xx - w // 2) ** 2 + (yy - 9) ** 2 <= 10 ** 2
    return mask | knob


def _outline(mask: np.ndarray) -> np.ndarray:
    er = mask.copy()
    er[1:, :] &= mask[:-1, :]
    er[:-1, :] &= mask[1:, :]
    er[:, 1:] &= mask[:, :-1]
    er[:, :-1] &= mask[:, 1:]
    return mask & ~er


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _cut_notch(bg: np.ndarray, x: int, y: int, *, darken: float = 0.45,
               border: int = 238) -> np.ndarray:
    out = bg.copy()
    mask = _piece_mask()
    region = out[y:y + PIECE, x:x + PIECE].astype(np.float64)
    inner = (region * darken).astype(np.uint8)
    out[y:y + PIECE, x:x + PIECE][mask] = inner[mask]
    out[y:y + PIECE, x:x + PIECE][_outline(mask)] = border
    return out


def _piece_image(bg: np.ndarray, x: int, y: int, *,
                 canvas: tuple[int, int] = (W, H)) -> bytes:
    """RGBA canvas, transparent except the piece at (x, y)."""
    cw, ch = canvas
    rgba = np.zeros((ch, cw, 4), dtype=np.uint8)
    mask = _piece_mask()
    patch = rgba[y:y + PIECE, x:x + PIECE]
    patch[mask, :3] = bg[y:y + PIECE, x:x + PIECE][mask]
    patch[mask, 3] = 255
    rgba[y:y + PIECE, x:x + PIECE] = patch
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _blur_noise(png: bytes, *, radius: float = 2.6, sigma: float = 17.0,
                seed: int = 3) -> bytes:
    img = Image.open(io.BytesIO(png)).convert("RGB")
    img = img.filter(ImageFilter.GaussianBlur(radius))
    arr = np.asarray(img).astype(np.float64)
    rng = np.random.default_rng(seed)
    arr += rng.normal(0, sigma, size=arr.shape)
    return _png(np.clip(arr, 0, 255).astype(np.uint8))


def clean_two_image() -> tuple[bytes, bytes]:
    bg = _photo()
    return _png(_cut_notch(bg, GAP_X, GAP_Y)), _piece_image(bg, PIECE_X, GAP_Y)


def clean_single_image() -> bytes:
    return _png(_cut_notch(_photo(), GAP_X, GAP_Y))


# ── (a) gap detection ────────────────────────────────────────────────────

def test_gap_two_image_clean_within_a_few_px():
    bg, piece = clean_two_image()
    res = detect_gap(bg, piece)
    assert res.found, res.detail
    assert res.confidence >= ss.MIN_GAP_CONFIDENCE
    assert abs(res.gap_x - GAP_X) <= 6, (res.gap_x, GAP_X)


def test_gap_single_image_clean_within_a_few_px():
    res = detect_gap(clean_single_image())
    assert res.found, res.detail
    assert res.confidence >= ss.MIN_GAP_CONFIDENCE
    assert abs(res.gap_x - GAP_X) <= 6, (res.gap_x, GAP_X)


def test_gap_two_image_degraded_is_honest_no_find():
    _, piece = clean_two_image()
    # wash the notch out: near-zero contrast change + heavy blur/noise
    weak = _png(_cut_notch(_photo(), GAP_X, GAP_Y, darken=0.94, border=150))
    res = detect_gap(_blur_noise(weak), piece)
    assert (not res.found) or res.confidence < ss.MIN_GAP_CONFIDENCE, res


def test_gap_single_image_degraded_is_honest_no_find():
    weak = _png(_cut_notch(_photo(), GAP_X, GAP_Y, darken=0.95, border=148))
    res = detect_gap(_blur_noise(weak))
    assert (not res.found) or res.confidence < ss.MIN_GAP_CONFIDENCE, res


def test_gap_garbage_bytes_never_found():
    res = detect_gap(b"not an image at all")
    assert not res.found
    assert res.confidence == 0.0


def test_vision_status_reports_sandbox_stack():
    st = vision_status()
    assert st["available"] is True  # numpy + PIL are present here
    assert st["numpy"] is True
    assert "detail" in st


# ── (b) trajectory synthesis ─────────────────────────────────────────────

def test_trajectory_shape_and_determinism():
    pts = synthesize_trajectory(120.0, 460.0, 190.0, seed=42)
    assert pts[0].x == 120.0 and pts[0].y == 460.0 and pts[0].delay_ms == 0
    # ends AT the target after settle (x exact, y back home)
    assert pts[-1].x == pytest.approx(120.0 + 190.0)
    assert abs(pts[-1].y - 460.0) <= 2.0
    # overshoot exists, then settles back
    assert max(p.x for p in pts) > 120.0 + 190.0 + 2.0
    # mostly monotonic: never falls back more than the overshoot itself
    run_max = pts[0].x
    worst_drawdown = 0.0
    for p in pts:
        run_max = max(run_max, p.x)
        worst_drawdown = max(worst_drawdown, run_max - p.x)
    overshoot = max(p.x for p in pts) - (120.0 + 190.0)
    assert worst_drawdown <= overshoot + 1.0
    # human band: total duration + point count
    total_ms = sum(p.delay_ms for p in pts)
    assert 450 <= total_ms <= 1900, total_ms
    assert 25 <= len(pts) <= 170, len(pts)
    # vertical jitter present but small
    ys = [p.y for p in pts]
    assert max(ys) - min(ys) >= 0.8
    assert max(abs(y - 460.0) for y in ys) <= 4.5
    # seed-deterministic; a different seed differs
    again = synthesize_trajectory(120.0, 460.0, 190.0, seed=42)
    assert [(p.x, p.y, p.delay_ms) for p in pts] == \
        [(p.x, p.y, p.delay_ms) for p in again]
    other = synthesize_trajectory(120.0, 460.0, 190.0, seed=43)
    assert [(p.x, p.y) for p in pts] != [(p.x, p.y) for p in other]


def test_trajectory_scales_with_distance():
    short = synthesize_trajectory(0.0, 0.0, 30.0, seed=5)
    long = synthesize_trajectory(0.0, 0.0, 300.0, seed=5)
    assert short[-1].x == pytest.approx(30.0)
    assert long[-1].x == pytest.approx(300.0)
    assert max(p.x for p in short) > 30.0  # still overshoots a little


# ── (e) detection ────────────────────────────────────────────────────────

TIKTOK_HTML = """<html><head><title>sandraomara011 on TikTok</title></head>
<body><div id="app"><p>some video page content</p>
<div class="captcha_verify_container" id="captcha-verify">
  <div class="captcha_verify_img--wrapper">
    <img class="captcha_verify_img_bg" src="https://p16.tiktokcdn.test/bg.png">
    <img class="captcha_verify_img_slide" src="https://p16.tiktokcdn.test/piece.png">
  </div>
  <div class="captcha_verify_slide--slidebar">
    <div class="secsdk-captcha-drag-icon"><span></span></div>
  </div>
  <p class="captcha_verify_text">Drag the slider to fit the puzzle</p>
</div></div></body></html>"""

GEETEST_HTML = """<html><body>
<div class="geetest_holder">
  <div class="geetest_widget">
    <canvas class="geetest_canvas_bg"></canvas>
    <canvas class="geetest_canvas_slice_bg"></canvas>
    <div class="geetest_slider"><div class="geetest_slider_button"></div></div>
  </div>
</div></body></html>"""

CLEAN_HTML = ("<html><head><title>a normal page</title></head><body>"
              "<h1>hello</h1><p>no challenges here</p>"
              "<div class='slider'>a settings slider, not a captcha</div>"
              "</body></html>")

AFTER_HTML = ("<html><head><title>sandraomara011 on TikTok</title></head>"
              "<body><div id='app'><p>the video plays now</p></div>"
              "</body></html>")


def test_detect_tiktok_slider_markup():
    info = detect_slider_challenge(TIKTOK_HTML, "https://www.tiktok.com/x")
    assert info is not None
    assert info.kind == "slider"
    assert info.vendor == "tiktok"
    assert info.background_selector == ".captcha_verify_img_bg"
    assert info.piece_selector == ".captcha_verify_img_slide"
    assert "secsdk-captcha-drag-icon" in info.handle_selector
    assert info.shape == "two_image"


def test_detect_geetest_slider_markup():
    info = detect_slider_challenge(GEETEST_HTML, "https://example.test/")
    assert info is not None
    assert info.vendor == "geetest"
    assert info.handle_selector
    assert info.background_selector


def test_detect_clean_page_is_not_a_slider():
    assert detect_slider_challenge(CLEAN_HTML, "https://example.test/") is None
    assert detect_slider_challenge("", "") is None


def test_challenge_signal_names_slider_kind():
    sig = detect_challenge_signal(TIKTOK_HTML, "https://www.tiktok.com/x",
                                  title="sandraomara011 on TikTok")
    assert sig is not None
    assert sig.kind == "slider"
    assert sig.vendor == "tiktok"
    assert detect_challenge_signal(CLEAN_HTML, "https://example.test/",
                                   title="a normal page") is None


# ── (d) execution seam: fake tab with page physics ───────────────────────

BG_BOX = {"x": 100.0, "y": 200.0, "width": float(W), "height": float(H)}
HANDLE_BOX = {"x": 108.0, "y": 452.0, "width": 40.0, "height": 40.0}


class FakeSliderTab:
    """Duck-typed rendered tab. The wall clears iff a drag lands the
    piece within tolerance of the gap — real page physics, no cheating:
    the solver only gets images, boxes and the drag path."""

    def __init__(self, *, bg_png: bytes, piece_png: bytes,
                 piece_x: int = PIECE_X, gap_x: int = GAP_X,
                 scale: float = 1.0, stubborn: bool = False) -> None:
        self._bg, self._piece = bg_png, piece_png
        self._piece_x, self._gap_x = piece_x, gap_x
        self._scale = scale
        self._stubborn = stubborn
        self._html = TIKTOK_HTML
        self._solved = False
        self.drags: list[list] = []
        self.reloads = 0

    @property
    def url(self) -> str:
        return "https://www.tiktok.com/@someone/photo/1"

    @property
    def title(self) -> str:
        return "tiktok"

    def html(self) -> str:
        return AFTER_HTML if self._solved else self._html

    def reload(self) -> None:
        self.reloads += 1

    # -- slider protocol ------------------------------------------------
    def element_image_bytes(self, selector: str) -> bytes:
        if "img_slide" in selector:
            return self._piece
        return self._bg

    def element_box(self, selector: str) -> dict:
        if "drag-icon" in selector:
            return dict(HANDLE_BOX)
        if "img_slide" in selector:
            # the piece is a full-width transparent strip over the bg;
            # the visible piece sits inside it at piece_x (image coords)
            return {"x": BG_BOX["x"], "y": BG_BOX["y"],
                    "width": BG_BOX["width"] * self._scale,
                    "height": BG_BOX["height"] * self._scale}
        return {"x": BG_BOX["x"], "y": BG_BOX["y"],
                "width": BG_BOX["width"] * self._scale,
                "height": BG_BOX["height"] * self._scale}

    def mouse_drag_path(self, points) -> None:
        self.drags.append(list(points))
        if self._stubborn or not points:
            return
        moved = points[-1][0] - points[0][0]
        expected = (self._gap_x - self._piece_x) * self._scale
        if abs(moved - expected) <= 5.0:
            self._solved = True


def _solver(*, seed: int = 11, **kw) -> LocalSliderSolver:
    kw.setdefault("sleeper", lambda _s: None)
    return LocalSliderSolver(seed=seed, **kw)


class _NoLimiter:
    def check(self): pass
    def record_submit(self): pass
    def record_success(self): pass
    def record_failure(self, hard: bool = False): pass


def test_solver_drags_the_computed_distance_and_clears():
    bg, piece = clean_two_image()
    tab = FakeSliderTab(bg_png=bg, piece_png=piece)
    out = _solver(limiter=_NoLimiter()).solve_tab(tab)
    assert out.ok, out.detail
    assert len(tab.drags) == 1
    pts = tab.drags[0]
    # drag starts at the handle centre, ends one piece-travel later
    assert pts[0][0] == pytest.approx(HANDLE_BOX["x"] + 20.0)
    assert pts[-1][0] - pts[0][0] == pytest.approx(GAP_X - PIECE_X, abs=1.0)
    assert out.attempts == 1
    assert out.confidence >= ss.MIN_GAP_CONFIDENCE


def test_solver_scales_distance_to_css_pixels():
    bg, piece = clean_two_image()
    tab = FakeSliderTab(bg_png=bg, piece_png=piece, scale=0.5)
    out = _solver(limiter=_NoLimiter()).solve_tab(tab)
    assert out.ok, out.detail
    moved = tab.drags[0][-1][0] - tab.drags[0][0][0]
    assert moved == pytest.approx((GAP_X - PIECE_X) * 0.5, abs=1.0)


def test_solver_low_confidence_never_drags():
    weak = _png(_cut_notch(_photo(), GAP_X, GAP_Y, darken=0.95, border=148))
    bg = _blur_noise(weak)
    _, piece = clean_two_image()
    tab = FakeSliderTab(bg_png=bg, piece_png=piece)
    out = _solver(limiter=_NoLimiter()).solve_tab(tab)
    assert not out.ok
    assert tab.drags == [], "a low-confidence gap must never produce a drag"
    assert "confidence" in out.detail.lower()


def test_solver_stubborn_page_fails_honestly_with_capped_retries():
    bg, piece = clean_two_image()
    tab = FakeSliderTab(bg_png=bg, piece_png=piece, stubborn=True)
    out = _solver(limiter=_NoLimiter()).solve_tab(tab)
    assert not out.ok
    assert out.attempts == 2, "exactly one retry is permitted"
    assert len(tab.drags) == 2
    assert "still present" in out.detail or "not cleared" in out.detail


def test_solver_unavailable_vision_never_drags():
    bg, piece = clean_two_image()
    tab = FakeSliderTab(bg_png=bg, piece_png=piece)
    solver = _solver(limiter=_NoLimiter(),
                      vision_check=lambda: (False, "vision stack "
                                                  "unavailable: numpy "
                                                  "missing"))
    out = solver.solve_tab(tab)
    assert not out.ok
    assert "numpy missing" in out.detail
    assert tab.drags == []


def test_solver_no_slider_on_page_is_honest():
    bg, piece = clean_two_image()
    tab = FakeSliderTab(bg_png=bg, piece_png=piece)
    tab._html = CLEAN_HTML
    out = _solver(limiter=_NoLimiter()).solve_tab(tab)
    assert not out.ok
    assert tab.drags == []


# ── (c) routing: browser ladder ──────────────────────────────────────────

class FakeWallTab:
    def __init__(self, html: str = TIKTOK_HTML) -> None:
        self._html = html

    @property
    def url(self) -> str:
        return "https://www.tiktok.com/@someone/photo/1"

    @property
    def title(self) -> str:
        return "tiktok"

    def html(self) -> str:
        return self._html

    def reload(self) -> None:
        pass


def _ladder(**kw) -> ChallengeSolver:
    kw.setdefault("sleeper", lambda _s: None)
    kw.setdefault("wait_s", 0.0)
    return ChallengeSolver(**kw)


def test_ladder_tries_local_before_service_and_short_circuits():
    order: list[str] = []
    tab = FakeWallTab()

    def local(tab_, signal):
        order.append("local")
        tab._html = AFTER_HTML
        return {"ok": True, "detail": "gap 190px @0.9"}

    def service(challenges):
        order.append("service")
        raise AssertionError("service must not run after a local clear")

    out = _ladder(local_solve=local, service_solve=service).solve(tab=tab)
    assert out.cleared and out.via == "local_solver"
    assert order == ["local"]
    assert out.attempts[0].strategy == "local_solver"


def test_ladder_local_failure_falls_through_to_service():
    order: list[str] = []
    tab = FakeWallTab()

    def local(tab_, signal):
        order.append("local")
        return {"ok": False, "detail": "low confidence 0.31"}

    class _Res:
        ok = True

    def service(challenges):
        order.append("service")
        tab._html = AFTER_HTML
        return _Res()

    out = _ladder(local_solve=local, service_solve=service).solve(tab=tab)
    assert out.cleared and out.via == "captcha_service"
    assert order.index("local") < order.index("service")
    strategies = [a.strategy for a in out.attempts]
    assert strategies.index("local_solver") < \
        strategies.index("captcha_service")


def test_ladder_local_unavailable_falls_through_with_reason():
    tab = FakeSliderTab(bg_png=clean_two_image()[0],
                        piece_png=clean_two_image()[1])
    solver = LocalSliderSolver(
        limiter=_NoLimiter(), sleeper=lambda _s: None,
        vision_check=lambda: (False, "vision stack unavailable: "
                                     "numpy missing"))
    seen: list[str] = []

    class _Res:
        ok = True

    def service(challenges):
        seen.append("service")
        tab._solved = True
        return _Res()

    out = _ladder(slider_solver=solver,
                  service_solve=service).solve(tab=tab)
    assert out.cleared
    assert seen == ["service"]
    local_attempts = [a for a in out.attempts
                      if a.strategy == "local_solver"]
    assert local_attempts and not local_attempts[0].ok
    assert "numpy missing" in local_attempts[0].detail
    assert tab.drags == []


# ── (c) routing: captcha_flow pipeline ───────────────────────────────────

def test_flow_classifies_tiktok_wall_as_solvable_captcha():
    from nomorals.tools.captcha_flow import classify_challenge
    verdict = classify_challenge(200, {}, TIKTOK_HTML,
                                 "https://www.tiktok.com/x")
    assert verdict["kind"] == "captcha"


def test_flow_slider_stage_runs_before_service():
    from nomorals.tools.captcha_flow import CaptchaFlow
    order: list[str] = []

    def slider_fn(ch):
        order.append("slider")
        return {"ok": True, "detail": "cleared in-session"}

    def service_fn(ch):
        order.append("service")
        return {"ok": True, "token": "tok"}

    flow = CaptchaFlow(slider_fn=slider_fn, service_fn=service_fn)
    out = flow.solve({"kind": "slider", "page_url": "https://x.test/"})
    assert out["ok"] and out["stage"] == "local_slider"
    assert order == ["slider"]


def test_flow_slider_failure_falls_through_to_service():
    from nomorals.tools.captcha_flow import CaptchaFlow

    def slider_fn(ch):
        return {"ok": False, "detail": "low confidence"}

    def service_fn(ch):
        return {"ok": True, "token": "tok", "cost_usd": 0.002}

    flow = CaptchaFlow(slider_fn=slider_fn, service_fn=service_fn)
    out = flow.solve({"kind": "slider", "page_url": "https://x.test/"})
    assert out["ok"] and out["stage"] == "service"


def test_flow_default_slider_fn_uses_session_from_metadata(
        tmp_path, monkeypatch):
    # the default solver paces through the shared rate limiter —
    # isolate its state file so no other run's backoff leaks in
    monkeypatch.setenv("NOMORALS_CAPTCHA_DIR", str(tmp_path))
    from nomorals.tools.captcha_flow import CaptchaFlow
    bg, piece = clean_two_image()
    tab = FakeSliderTab(bg_png=bg, piece_png=piece)
    flow = CaptchaFlow()
    out = flow.solve({"kind": "slider", "page_url": tab.url,
                      "metadata": {"session": tab}})
    assert out["ok"] and out["stage"] == "local_slider", out
    assert len(tab.drags) == 1


def test_flow_slider_without_session_is_honest_not_fake():
    from nomorals.tools.captcha_flow import CaptchaFlow
    flow = CaptchaFlow(service_fn=None)
    out = flow.solve({"kind": "slider", "page_url": "https://x.test/"})
    assert not out["ok"]
    assert out["stage"] != "local_slider" or not out.get("ok")


# ── (f) registered backend + status truth ────────────────────────────────

def test_backend_registered_and_reachable():
    from nomorals.tools import captcha as captcha_mod
    assert ss.register_local_backend() is True
    assert "local_slider" in captcha_mod._BACKENDS
    backend = captcha_mod.backend_for("local_slider")
    assert isinstance(backend, LocalSliderBackend)
    assert backend.available() is True  # numpy + PIL in this sandbox


def test_backend_requires_a_live_session():
    from nomorals.tools.captcha import CaptchaChallenge
    backend = LocalSliderBackend()
    res = backend.solve(CaptchaChallenge(
        kind="slider", page_url="https://www.tiktok.com/x"))
    assert not res.ok
    assert "session" in res.detail
    assert res.backend == "local_slider"


def test_backend_solves_with_session_in_metadata():
    from nomorals.tools.captcha import CaptchaChallenge
    bg, piece = clean_two_image()
    tab = FakeSliderTab(bg_png=bg, piece_png=piece)
    backend = LocalSliderBackend(solver=_solver(limiter=_NoLimiter()))
    res = backend.solve(CaptchaChallenge(
        kind="slider", page_url=tab.url, metadata={"session": tab}))
    assert res.ok, res.detail
    assert len(tab.drags) == 1


def test_backend_refuses_non_slider_kinds():
    from nomorals.tools.captcha import CaptchaChallenge
    backend = LocalSliderBackend()
    res = backend.solve(CaptchaChallenge(kind="recaptcha_v2",
                                         sitekey="abc"))
    assert not res.ok
    assert "slider" in res.detail


def test_status_surfaces_local_availability_honestly(monkeypatch):
    monkeypatch.delenv("CAPTCHA_API_KEY", raising=False)
    st = local_slider_status()
    assert st["backend"] == "local_slider"
    assert st["available"] is True
    assert st["vision"]["available"] is True
    assert st["service_key_configured"] is False
    # the honest sentence: what CAN solve a slider right now
    assert st["can_solve_slider"] is True


def test_cli_status_includes_local_backend(capsys):
    from types import SimpleNamespace
    from nomorals.cmdline.commands.captcha import _cmd_captcha
    args = SimpleNamespace(captcha_action="status", json=True)
    ctx = SimpleNamespace(settings=None)
    rc = _cmd_captcha(args, ctx)
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert "local_slider" in data["backends"]
    assert data["local_slider"]["backend"] == "local_slider"
    assert "available" in data["local_slider"]


def _pending():
    from nomorals.accounts.creator import (
        AccountCheckpoint, AccountCheckpointPending, CheckpointKind)
    return AccountCheckpointPending(AccountCheckpoint(
        id="cp-test", kind=CheckpointKind.CAPTCHA, title="captcha",
        instructions="solve it", service="testsvc"))


# ── (g) signup driver seam (injected, composition-wired) ─────────────────

def test_signup_driver_local_attempt_precedes_checkpoint():
    from nomorals.accounts.creator import AccountCheckpointPending
    from nomorals.accounts.identity_bank import IdentityBank
    from nomorals.accounts.signup_driver import (
        SignupAttemptStore, SignupDriver)
    from nomorals.storage.db import Database

    db = Database(":memory:")
    attempts = SignupAttemptStore(db)
    persona = IdentityBank(db=Database(":memory:")).mint(
        "testsvc", rng=random.Random(42))
    attempt = attempts.create(service="testsvc", persona=persona)

    class FakeCreator:
        def __init__(self):
            self.called = False

        def attempt_captcha_solve(self, **kw):
            self.called = True
            raise _pending()

    class FakeTab:
        def check_captcha(self, *, fetch_bytes: bool = False):
            return {"challenges": [{"kind": "slider"}]}

    seen: list = []

    def slider_solve(tab, page_url=""):
        seen.append((tab, page_url))
        return {"attempted": True, "ok": True,
                "detail": "cleared in-session"}

    creator = FakeCreator()
    driver = SignupDriver(creator, attempts,
                          notify=lambda *a: None,
                          slider_solve=slider_solve)
    asyncio.run(driver._handle_captcha_wall(
        attempt, FakeTab(), "slider wall", "https://example.test/signup"))
    assert seen, "local slider attempt never ran"
    assert creator.called is False, (
        "token solver/checkpoint must not run after a local clear")


def test_signup_driver_without_local_attempt_still_checkpoints():
    from nomorals.accounts.creator import AccountCheckpointPending  # noqa: F401 - pytest.raises target
    from nomorals.accounts.identity_bank import IdentityBank
    from nomorals.accounts.signup_driver import (
        SignupAttemptStore, SignupDriver)
    from nomorals.storage.db import Database

    db = Database(":memory:")
    attempts = SignupAttemptStore(db)
    persona = IdentityBank(db=Database(":memory:")).mint(
        "testsvc", rng=random.Random(42))
    attempt = attempts.create(service="testsvc", persona=persona)

    class FakeCreator:
        def attempt_captcha_solve(self, **kw):
            raise _pending()

    class FakeTab:
        def check_captcha(self, *, fetch_bytes: bool = False):
            return {"challenges": [{"kind": "slider"}]}

    driver = SignupDriver(FakeCreator(), attempts,
                          notify=lambda *a: None)
    with pytest.raises(AccountCheckpointPending):
        asyncio.run(driver._handle_captcha_wall(
            attempt, FakeTab(), "slider wall",
            "https://example.test/signup"))


# ── adapter pass-throughs (browser service) ──────────────────────────────

def test_takeover_adapter_exposes_slider_primitives():
    from nomorals.browser.service import RenderedTabTakeoverAdapter

    class Inner:
        def element_image_bytes(self, selector):
            return b"img:" + selector.encode()

        def element_box(self, selector):
            return {"x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0}

        def mouse_drag_path(self, points):
            self.dragged = list(points)

    inner = Inner()
    adapter = RenderedTabTakeoverAdapter(inner)
    assert adapter.element_image_bytes(".bg") == b"img:.bg"
    assert adapter.element_box(".bg")["width"] == 3.0
    adapter.mouse_drag_path([(1.0, 2.0, 0)])
    assert inner.dragged == [(1.0, 2.0, 0)]


def test_takeover_adapter_missing_primitives_fail_loudly():
    from nomorals.browser.service import RenderedTabTakeoverAdapter

    class Bare:
        pass

    adapter = RenderedTabTakeoverAdapter(Bare())
    with pytest.raises(Exception):
        adapter.element_image_bytes(".bg")
    with pytest.raises(Exception):
        adapter.mouse_drag_path([(0.0, 0.0, 0)])


# ── live-browser integration (guarded) ─────────────────────────────────

def test_live_chromium_slider_fixture(tmp_path):
    """One real-browser pass, when the harness exists in this sandbox.

    The sandbox for this pass has Chromium binaries but no Playwright
    package, so this skips here — the seam tests above carry the proof.
    On a host with Playwright it drives a real DOM slider with the
    synthesized trajectory through page.mouse.
    """
    pw = pytest.importorskip("playwright.sync_api")
    import glob
    from pathlib import Path
    fixture = tmp_path / "slider.html"
    fixture.write_text(_LIVE_FIXTURE_HTML, encoding="utf-8")
    # the sandbox's Chromium may not match this Playwright's pinned
    # revision — resolve any installed chromium binary explicitly
    candidates = sorted(
        glob.glob(str(Path.home() / ".cache/ms-playwright"
                      / "chromium-*/chrome-linux*/chrome"))
        + glob.glob(str(Path.home() / ".cache/ms-playwright"
                        / "chromium_headless_shell-*"
                        / "chrome-linux*/headless_shell")))
    with pw.sync_playwright() as p:
        default_exe = p.chromium.executable_path
        if Path(default_exe).exists():
            exe = default_exe
        elif candidates:
            exe = candidates[0]
        else:
            pytest.skip("no chromium binary present")
        browser = p.chromium.launch(executable_path=exe)
        page = browser.new_page(viewport={"width": 800, "height": 600})
        page.goto(fixture.as_uri())
        page.wait_for_selector(".captcha_verify_container")

        class LiveTab:
            url = page.url
            title = "fixture"

            def html(self):
                return page.content()

            def reload(self):
                page.reload()

            def element_image_bytes(self, selector):
                # mirror RenderedTab: canvas elements export their own
                # pixels (no composite with overlapping elements)
                data_url = page.evaluate(
                    """(sel) => { const el = document.querySelector(sel);
                        return (el && el.tagName === 'CANVAS')
                            ? el.toDataURL('image/png') : ''; }""",
                    selector)
                if data_url:
                    import base64
                    return base64.b64decode(data_url.split(",", 1)[1])
                return page.locator(selector).screenshot()

            def element_box(self, selector):
                box = page.locator(selector).bounding_box()
                return dict(box)

            def mouse_drag_path(self, points):
                import time as _t
                page.mouse.move(points[0][0], points[0][1])
                page.mouse.down()
                for x, y, delay in points[1:]:
                    if delay:
                        _t.sleep(delay / 1000.0)
                    page.mouse.move(x, y)
                page.mouse.up()

        out = LocalSliderSolver(sleeper=lambda _s: None,
                                limiter=_NoLimiter()).solve_tab(LiveTab())
        solved_marker = page.locator("#solved").count() > 0
        browser.close()
    assert out.ok == solved_marker


_LIVE_FIXTURE_HTML = """<!doctype html><html><body>
<div class="captcha_verify_container" style="position:relative;width:340px">
 <div class="captcha_verify_img--wrapper" style="position:relative">
  <canvas id="bg" class="captcha_verify_img_bg" width="340" height="212"></canvas>
  <canvas id="piece" class="captcha_verify_img_slide" width="340" height="212"
          style="position:absolute;left:0;top:0"></canvas>
 </div>
 <div class="captcha_verify_slide--slidebar" style="height:44px;position:relative">
   <div class="secsdk-captcha-drag-icon" id="handle"
        style="position:absolute;left:0;top:2px;width:40px;height:40px;
               background:#888;cursor:pointer"></div>
 </div>
 <p>Drag the slider to fit the puzzle</p>
</div><div id="status"></div>
<script>
const GAP=208, PY=76, PS=54, START=18;
const bg=document.getElementById('bg').getContext('2d');
const grd=bg.createLinearGradient(0,0,340,212);
grd.addColorStop(0,'#7a4'); grd.addColorStop(1,'#47a');
bg.fillStyle=grd; bg.fillRect(0,0,340,212);
bg.fillStyle='#a73'; bg.fillRect(30,20,90,60);
bg.fillStyle='#37a'; bg.beginPath(); bg.arc(280,160,34,0,7); bg.fill();
bg.save(); bg.beginPath(); bg.rect(GAP,PY,PS,PS); bg.clip();
bg.fillStyle='rgba(0,0,0,0.55)'; bg.fillRect(GAP,PY,PS,PS); bg.restore();
bg.strokeStyle='#eee'; bg.lineWidth=2; bg.strokeRect(GAP,PY,PS,PS);
const pc=document.getElementById('piece').getContext('2d');
pc.fillStyle='#fc3'; pc.fillRect(START,PY,PS,PS);
let drag=null;
const handle=document.getElementById('handle');
handle.addEventListener('pointerdown',e=>{drag={x:e.clientX,left:parseFloat(handle.style.left||0)}; handle.setPointerCapture(e.pointerId);});
handle.addEventListener('pointermove',e=>{ if(!drag) return;
  const dx=e.clientX-drag.x; handle.style.left=(drag.left+dx)+'px';
  document.getElementById('piece').style.left=dx+'px'; });
handle.addEventListener('pointerup',e=>{ if(!drag) return;
  const dx=e.clientX-drag.x; drag=null;
  if (Math.abs((START+dx)-GAP)<=5) {
    document.querySelector('.captcha_verify_container').remove();
    const d=document.createElement('div'); d.id='solved';
    d.textContent='verified'; document.body.appendChild(d);
  }});
</script></body></html>"""

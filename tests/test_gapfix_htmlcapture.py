"""Real headless-browser HTML capture for the code-video path (lane 82).

The HTML scene path used to open a plain raw-HTTP ``Tab`` — which has
no ``evaluate`` and no ``screenshot_bytes`` — so a production render
died at frame 0, and the only test ever drove a ``_FakeTab``. These
tests pin the real thing: a real headless Chromium renders the page,
seeks it per frame, and the MP4 carries frames that are actually the
page — plus the bounds and the honest reporting around it.
"""
from __future__ import annotations

import io
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

from nomorals.browser import provision
from nomorals.media.videogen.capabilities import VideogenError
from nomorals.media.videogen.codegen import render_code_scene

SIZE = (320, 180)

#: An animated page with no text (no font-rendering nondeterminism):
#: the seek hook sweeps the background through the hue wheel and
#: slides a solid block across the frame, so every frame is visibly
#: different from the last.
ANIMATED_HTML = """<!doctype html><html><head><style>
html, body { margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; }
#box { position: absolute; top: 60px; left: 0px; width: 60px; height: 60px;
       background: #ffffff; }
</style></head><body>
<div id="box"></div>
<script>
window.__devon_seek = function (t) {
  document.body.style.background = 'hsl(' + Math.round(t * 360) + ', 80%, 50%)';
  document.getElementById('box').style.left = Math.round(t * 260) + 'px';
};
window.__devon_seek(0);
</script></body></html>"""


def _panel_centers(sheet: Path) -> list[tuple[int, int, int]]:
    """Center pixel of each panel in a 5-panel contact sheet."""
    img = Image.open(sheet).convert("RGB")
    w, h = img.size
    panels = w // 320
    return [img.getpixel((j * 320 + 160, 90)) for j in range(panels)]


def _close_enough(a: tuple[int, int, int], b: tuple[int, int, int],
                  tol: int = 24) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b, strict=True))


# ── the real browser, end to end ──────────────────────────────────────────


def test_html_scene_renders_real_frames_with_a_real_browser(tmp_path):
    scene = {"engine": "html", "html": ANIMATED_HTML}
    rec = render_code_scene(scene, tmp_path / "real.mp4",
                            duration=1.0, fps=6.0, size=SIZE,
                            narrate=False)
    assert rec["source"] == "html"
    assert rec["frames"] == 6
    assert (tmp_path / "real.mp4").is_file()
    capture = rec["capture"]
    assert capture["browser"] == "real"
    assert capture["engine"] == "playwright-chromium"
    assert capture["viewport"] == [SIZE[0], SIZE[1]]
    # The browser rendered at the scene size, not its 1366x768 default.
    assert capture["frame_size"] == [SIZE[0], SIZE[1]]
    # The contact sheet is the page, not a placeholder: frame 0 is the
    # t=0 hue (strong red) and the panels sweep through distinct hues.
    centers = _panel_centers(Path(rec["contact_sheet"]))
    assert len(centers) == 5
    assert _close_enough(centers[0], (230, 26, 26), tol=40)
    assert len(set(centers)) >= 3


def test_html_scene_render_is_deterministic(tmp_path):
    scene = {"engine": "html", "html": ANIMATED_HTML}
    first = render_code_scene(scene, tmp_path / "a.mp4",
                              duration=1.0, fps=6.0, size=SIZE,
                              narrate=False)
    second = render_code_scene(scene, tmp_path / "b.mp4",
                               duration=1.0, fps=6.0, size=SIZE,
                               narrate=False)
    assert first["render_hash"] == second["render_hash"]


# ── bounds + honest reporting ─────────────────────────────────────────────


def test_oversized_html_scene_refused_before_the_browser(tmp_path):
    from nomorals.media.videogen.codegen import _HTML_MAX_BYTES

    scene = {"engine": "html",
             "html": "<html><body>" + "x" * (_HTML_MAX_BYTES + 1) +
             "</body></html>"}
    with pytest.raises(VideogenError, match="too large"):
        render_code_scene(scene, tmp_path / "big.mp4",
                          duration=1.0, fps=6.0, size=SIZE,
                          narrate=False)
    assert not (tmp_path / "big.mp4").exists()


class _FakeTab:
    """Minimal injected tab — the factory seam, kept for the frame loop."""

    def __init__(self):
        self.seeks: list[float] = []
        self.url = ""

    def navigate(self, url):
        self.url = url
        return {"url": url}

    def evaluate(self, js, arg=None):
        self.seeks.append(float(arg))
        return True

    def screenshot_bytes(self):
        t = self.seeks[-1] if self.seeks else 0.0
        shade = int(40 + t * 30) % 256
        img = Image.new("RGB", SIZE, (shade, 80, 120))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def close(self):
        pass


class _FakeSession:
    def __init__(self):
        self.tab = _FakeTab()

    def open_tab(self, url=""):
        return self.tab

    def close(self):
        pass


def test_injected_factory_capture_is_reported_honestly(tmp_path):
    scene = {"engine": "html",
             "html": "<html><body><h1>hi</h1></body></html>"}
    session = _FakeSession()
    rec = render_code_scene(scene, tmp_path / "fake.mp4",
                            duration=1.0, fps=6.0, size=SIZE,
                            narrate=False,
                            browser_session_factory=lambda: session)
    assert rec["source"] == "html"
    assert rec["capture"]["browser"] == "injected"
    assert rec["capture"]["engine"] == "factory"
    assert len(session.tab.seeks) == 6


# ── self-provision: PEP 668 retry ──────────────────────────────────────────


def _no_pytest(monkeypatch):
    monkeypatch.delitem(sys.modules, "pytest", raising=False)


def test_ensure_playwright_retries_pep668_with_break_system_packages(
        monkeypatch):
    _no_pytest(monkeypatch)
    state = {"ok": False}
    monkeypatch.setattr(provision, "playwright_usable",
                        lambda: state["ok"])
    monkeypatch.setattr(provision, "_PROVISION_ATTEMPTED", False)
    commands: list[list[str]] = []
    calls = {"n": 0}

    def _fake_run(cmd, _timeout):
        commands.append(list(cmd))
        calls["n"] += 1
        if calls["n"] == 1:
            # Debian PEP 668: the system interpreter refuses pip.
            return types.SimpleNamespace(
                returncode=1, stdout="",
                stderr="error: externally-managed-environment\n"
                       "This environment is externally managed")
        if calls["n"] == 3:
            state["ok"] = True
        return types.SimpleNamespace(returncode=0, stderr="", stdout="")

    assert provision.ensure_playwright(runner=_fake_run) is True
    assert len(commands) == 3
    assert commands[0][1:] == ["-m", "pip", "install", "playwright"]
    assert commands[1][1:] == ["-m", "pip", "install",
                               "--break-system-packages", "playwright"]
    assert commands[2][1:] == ["-m", "playwright", "install", "chromium"]


def test_ensure_playwright_gives_up_after_the_retry(monkeypatch):
    _no_pytest(monkeypatch)
    monkeypatch.setattr(provision, "playwright_usable", lambda: False)
    monkeypatch.setattr(provision, "_PROVISION_ATTEMPTED", False)
    commands: list[list[str]] = []

    def _fake_run(cmd, _timeout):
        commands.append(list(cmd))
        return types.SimpleNamespace(
            returncode=1, stdout="",
            stderr="error: externally-managed-environment")

    assert provision.ensure_playwright(runner=_fake_run) is False
    # One retry of the pip step, then the chromium step is never reached
    # (the engine still is not installable) — and no second retry.
    assert len(commands) == 2
    assert "--break-system-packages" in commands[1]

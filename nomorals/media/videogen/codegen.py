"""Code-video backend — deterministic scene rendering, no model.

The Anthropic code-video shape (owner link 2026-10-31, prayag_sonar):
a video is code — every frame is a pure function of time, ``t = frame /
fps``. A renderer is seeked to a precise timestamp, the scene is painted,
one image is captured per frame; FFmpeg muxes the image sequence into MP4.
Deterministic: the same scene renders the same frames, every time.

Devon's native engine already has that core shape
(:mod:`nomorals.media.motion_studio` — ``frame_image`` + ``render_sequence``
+ FFmpeg), so the code backend is built on it, never on an external
renderer:

- **design scenes** (diagram / equation / chart) render through the
  motion-studio frame function — CPU-only, always available;
- **HTML scenes** render through the browser organ (``Tab.evaluate`` +
  ``Tab.screenshot_bytes``): the page is seeked to ``t`` with its own
  seek hook and captured once per frame.

The code rung is co-equal beside the diffusion rung (have-it-all): code
for crisp, exact scenes — diagrams, maps, routes, text stories, data
animation, UI walkthroughs; diffusion for organic footage. Per-job
routing lives in :mod:`.pipeline` — never a user choice between rungs.

Better than the reference, not just as good: every code render carries a
**determinism receipt** (``render_hash`` — a re-render of the same scene
produces the same hash, proven by test) and a **contact sheet** (sampled
frames beside the MP4, the motion-studio QC discipline) so the render
can be seen before it ships.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from ...core.logging_setup import get_logger
from .capabilities import VideogenError

_log = get_logger(__name__)

__all__ = [
    "render_code_scene",
    "is_code_scene",
    "CODE_DEFAULT_SIZE",
    "CODE_DEFAULT_FPS",
    "CODE_SESSION_NAME",
    "DEFAULT_SEEK_JS",
]

CODE_DEFAULT_SIZE: tuple[int, int] = (1280, 720)
CODE_DEFAULT_FPS = 24.0
CODE_SESSION_NAME = "codevideo"

#: JS evaluated once per frame in an HTML scene, with the frame time ``t``
#: (seconds) as its argument — the same function-expression + arg idiom
#: the browser organ's own ``local_storage`` helper uses. A scene may
#: carry its own ``seek`` snippet in the same shape (a function expression
#: taking ``t``). The default calls the page's own hook when one exists
#: and is a no-op otherwise, so a static HTML scene still renders.
DEFAULT_SEEK_JS = ("(t) => { if (window.__devon_seek) "
                   "window.__devon_seek(t); }")

#: frames sampled for the contact sheet (start, quarters, end).
_CONTACT_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


def is_code_scene(scene: Any) -> bool:
    """True when ``scene`` is a scene spec this backend can render."""
    return isinstance(scene, dict) and bool(scene)


def _default_out() -> Path:
    base = Path.home() / ".nomorals" / "videos"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"code-{int(time.time() * 1000)}.mp4"


def _contact_sheet(sampled: dict[int, Image.Image], out: Path) -> Path:
    """Sampled frames as one QC strip beside the MP4."""
    thumbs = [img.copy().resize((320, 180))
              for _, img in sorted(sampled.items())]
    sheet = Image.new("RGB", (320 * len(thumbs), 180), (10, 10, 14))
    for j, thumb in enumerate(thumbs):
        sheet.paste(thumb, (j * 320, 0))
    path = out.with_name(out.stem + ".contact.png")
    sheet.save(path)
    return path


def _sequence(scene_kind: str, n_frames: int, width: int, height: int,
              fps: float, out: Path, frame_fn: Callable,
              ) -> tuple[str, Path]:
    """Encode ``frame_fn(i, t)`` to MP4; return (render_hash, contact)."""
    from ..motion_studio._core import profile_defaults, render_sequence

    defaults = profile_defaults()
    digests: list[bytes] = []
    sampled: dict[int, Image.Image] = {}
    sample_idx = sorted({min(n_frames - 1, int(f * n_frames))
                         for f in _CONTACT_FRACTIONS})

    def _traced(i: int, t: float):
        import numpy as np

        img = frame_fn(i, t)
        if i in sample_idx:
            sampled[i] = img.copy()
        arr = np.asarray(img.convert("RGB"))
        digests.append(hashlib.sha256(arr.tobytes()).digest())
        return arr

    render_sequence(n_frames, width, height, float(fps), out, _traced,
                    crf=int(defaults.get("crf", 20)),
                    preset=str(defaults.get("preset", "veryfast")))
    render_hash = hashlib.sha256(b"".join(digests)).hexdigest()
    return render_hash, _contact_sheet(sampled, out)


def _render_design_scene(design: dict[str, Any], out: Path, *,
                         duration: float, size: tuple[int, int],
                         fps: float) -> dict[str, Any]:
    """An explainer design dict → MP4 via the frame function."""
    from ..motion_studio._core import MotionStudioError
    from ..motion_studio.explainers import frame_image, validate_design

    try:
        spec = validate_design(design)
    except MotionStudioError as exc:
        raise VideogenError(f"code-video scene is not valid: {exc}") from exc
    n_frames = max(1, int(round(duration * fps)))
    width, height = int(size[0]), int(size[1])

    def _frame(i: int, t: float) -> Image.Image:
        # frame_image re-validates the raw design (the validated spec
        # carries no "engine" key, so it cannot be fed back in).
        return frame_image(design, (width, height), t, duration)

    render_hash, sheet = _sequence("design", n_frames, width, height,
                                   fps, out, _frame)
    return {"engine": "code", "source": "design", "kind": spec["kind"],
            "title": spec.get("title", ""), "frames": n_frames,
            "fps": float(fps), "duration": float(duration),
            "path": str(out), "contact_sheet": str(sheet),
            "render_hash": render_hash, "deterministic": True}


def _open_browser_session(factory: Callable | None):
    """A browser session for HTML capture — injected or the real organ."""
    if factory is not None:
        try:
            return factory()
        except Exception as exc:  # noqa: BLE001 - wrapped, never hidden
            raise VideogenError(
                "code-video HTML capture needs a headless browser: "
                f"{exc}") from exc
    try:
        from ...browser.service import BrowserError, BrowserService
    except Exception as exc:  # noqa: BLE001 - wrapped, never hidden
        raise VideogenError(
            "code-video HTML capture needs the browser organ: "
            f"{exc}") from exc
    try:
        svc = BrowserService()
        try:
            return svc.get_session(CODE_SESSION_NAME)
        except BrowserError:
            return svc.open_session(CODE_SESSION_NAME)
    except Exception as exc:  # noqa: BLE001 - wrapped, never hidden
        raise VideogenError(
            "code-video HTML capture needs a headless browser: "
            f"{exc}") from exc


def _render_html_scene(scene: dict[str, Any], out: Path, *,
                       duration: float, size: tuple[int, int],
                       fps: float,
                       browser_session_factory: Callable | None
                       ) -> dict[str, Any]:
    """An HTML scene → MP4: seek the live page to ``t``, capture a frame.

    The scene is ``{"html": "<!doctype html>…", "seek": "(t) => …"}`` —
    ``seek`` is a JS function expression taking the frame time in seconds
    (the page's own ``window.__devon_seek(t)`` hook by default).
    """
    html = scene.get("html", "")
    if not isinstance(html, str) or not html.strip():
        raise VideogenError(
            "HTML scene needs a non-empty 'html' string")
    seek_js = str(scene.get("seek") or DEFAULT_SEEK_JS)
    n_frames = max(1, int(round(duration * fps)))
    width, height = int(size[0]), int(size[1])

    session = _open_browser_session(browser_session_factory)
    tab = None
    with tempfile.NamedTemporaryFile(
            "w", suffix=".html", dir=str(out.parent), delete=False,
            encoding="utf-8") as tmp:
        tmp.write(html)
        tmp_path = tmp.name
    try:
        page_url = Path(tmp_path).as_uri()
        try:
            tab = session.open_tab()
            tab.navigate(page_url)
        except Exception as exc:  # noqa: BLE001 - wrapped, never hidden
            raise VideogenError(
                "code-video HTML capture could not open the scene page: "
                f"{exc}") from exc

        def _frame(i: int, t: float) -> Image.Image:
            try:
                tab.evaluate(seek_js, t)
                raw = tab.screenshot_bytes()
            except Exception as exc:  # noqa: BLE001 - wrapped, named
                raise VideogenError(
                    "code-video HTML capture failed at frame "
                    f"{i} (t={t:.3f}s): {exc}") from exc
            try:
                img = Image.open(io.BytesIO(bytes(raw))).convert("RGB")
            except Exception as exc:  # noqa: BLE001 - wrapped, named
                raise VideogenError(
                    "code-video HTML capture returned no image at frame "
                    f"{i}: {exc}") from exc
            if img.size != (width, height):
                img = img.resize((width, height))
            return img

        render_hash, sheet = _sequence("html", n_frames, width, height,
                                       fps, out, _frame)
    finally:
        if tab is not None:
            with contextlib.suppress(Exception):
                tab.close()
        with contextlib.suppress(Exception):
            session.close()
        with contextlib.suppress(OSError):
            Path(tmp_path).unlink(missing_ok=True)
    return {"engine": "code", "source": "html", "kind": "html",
            "title": str(scene.get("title", "")), "frames": n_frames,
            "fps": float(fps), "duration": float(duration),
            "path": str(out), "contact_sheet": str(sheet),
            "render_hash": render_hash, "deterministic": True}


def render_code_scene(scene: dict[str, Any],
                      out: str | os.PathLike | None, *,
                      duration: float = 5.0,
                      size: tuple[int, int] | None = None,
                      fps: float | None = None,
                      browser_session_factory: Callable | None = None
                      ) -> dict[str, Any]:
    """Render a scene spec to MP4 — deterministic seek(t) frames.

    ``scene`` is an explainer design dict (``{"engine": "diagram" |
    "equation" | "chart", ...}``) or an HTML scene (``{"html": ...}``).
    Returns the render record — ``path``, ``contact_sheet``,
    ``render_hash`` (identical across re-renders of the same scene),
    ``frames`` / ``fps`` / ``duration``.
    """
    if not is_code_scene(scene):
        raise VideogenError(
            "code-video needs a scene spec — an explainer design dict "
            "({'engine': 'diagram' | 'equation' | 'chart', ...}) or an "
            "HTML scene ({'html': '<!doctype html>…'})")
    if duration <= 0:
        raise VideogenError(
            f"code-video needs duration > 0, got {duration}")
    if fps is not None and fps <= 0:
        raise VideogenError(f"code-video needs fps > 0, got {fps}")
    if size is not None and (len(size) != 2 or size[0] <= 0
                             or size[1] <= 0):
        raise VideogenError(f"code-video needs a positive size, got {size}")
    from ..motion_studio._core import profile_defaults

    defaults = profile_defaults()
    size = size or tuple(defaults["size"])
    fps = fps or float(defaults["fps"])
    out_path = Path(out) if out is not None else _default_out()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if "html" in scene:
        return _render_html_scene(scene, out_path, duration=duration,
                                  size=size, fps=fps,
                                  browser_session_factory=(
                                      browser_session_factory))
    return _render_design_scene(scene, out_path, duration=duration,
                                size=size, fps=fps)

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
- **HTML scenes** render through a real headless-Chromium tab
  (``RenderedTab.evaluate`` + ``RenderedTab.screenshot_bytes`` from
  the browser organ): the page is seeked to ``t`` with its own seek
  hook and captured once per frame. A plain raw-HTTP tab cannot
  render a page at all — it has neither method — so the HTML path
  never opens one.

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

#: Largest HTML scene accepted for capture (bytes of the source
#: string). A scene bigger than this is refused before any browser
#: work — a hostile or accidental giant page must not hang the render.
_HTML_MAX_BYTES = 2_000_000

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
              audio: str | os.PathLike | None = None,
              ) -> tuple[str, Path]:
    """Encode ``frame_fn(i, t)`` to MP4; return (render_hash, contact).

    ``audio`` is a wav muxed into the MP4 as its audio track — the
    narration voice when one was rendered, else None (silent).
    """
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
                    preset=str(defaults.get("preset", "veryfast")),
                    audio=audio)
    render_hash = hashlib.sha256(b"".join(digests)).hexdigest()
    return render_hash, _contact_sheet(sampled, out)


def _captioned(frame_fn: Callable, words: list[dict[str, Any]]
               ) -> Callable:
    """A frame function with the caption overlay burned in at ``t``.

    No words → the frame function unchanged (a silent or
    caption-free render draws exactly what it drew before).
    """
    if not words:
        return frame_fn
    from .captions import draw_captions

    def _wrapped(i: int, t: float) -> Image.Image:
        return draw_captions(frame_fn(i, t), words, t)

    return _wrapped


def _render_design_scene(design: dict[str, Any], out: Path, *,
                         duration: float, size: tuple[int, int],
                         fps: float,
                         audio: str | os.PathLike | None = None,
                         narration: dict[str, Any] | None = None,
                         captions: dict[str, Any] | None = None
                         ) -> dict[str, Any]:
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

    render_hash, sheet = _sequence(
        "design", n_frames, width, height, fps, out,
        _captioned(_frame, (captions or {}).get("words") or []),
        audio=audio)
    return {"engine": "code", "source": "design", "kind": spec["kind"],
            "title": spec.get("title", ""), "frames": n_frames,
            "fps": float(fps), "duration": float(duration),
            "path": str(out), "contact_sheet": str(sheet),
            "render_hash": render_hash, "deterministic": True,
            "narration": narration or {"status": "off", "backend": "",
                                       "segments": 0, "audio": "",
                                       "reason": ""},
            "captions": captions or {"status": "off", "words": [],
                                     "reason": ""}}


class _RenderedSession:
    """A session-shaped handle over one real headless-Chromium tab.

    The browser organ's rendered tabs live outside ``SessionHandle``
    (``BrowserService.open_rendered_tab``), but the video path only
    needs ``open_tab`` / ``close`` — this adapter gives the rendered
    tab that shape so ``_render_html_scene`` needs no second path.
    The viewport is the scene size, so frames render at the scene
    size natively instead of the organ's 1366x768 default.
    """

    def __init__(self, service: Any, size: tuple[int, int]) -> None:
        self._service = service
        self._size = (int(size[0]), int(size[1]))
        self._tab: Any = None

    def open_tab(self, url: str = "") -> Any:
        if self._tab is not None:
            raise VideogenError(
                "code-video HTML capture opens exactly one tab")
        self._tab = self._service.open_rendered_tab(
            CODE_SESSION_NAME, url,
            stealth={"viewport": {"width": self._size[0],
                                  "height": self._size[1]}},
            retries=0)
        return self._tab

    def close(self) -> None:
        tab, self._tab = self._tab, None
        if tab is not None:
            with contextlib.suppress(Exception):
                self._service.close_rendered_tab(tab.tab_id)
        with contextlib.suppress(Exception):
            self._service.shutdown()


def _open_browser_session(factory: Callable | None, *,
                          size: tuple[int, int] | None = None):
    """A browser session for HTML capture — injected or the real organ.

    The production path opens a REAL headless-Chromium tab through
    ``BrowserService.open_rendered_tab`` — never a plain raw-HTTP tab,
    which has no ``evaluate`` and no ``screenshot_bytes`` and cannot
    render a page at all. Opening the tab starts the playwright
    driver, which self-provisions (``browser.provision``) when the
    engine is missing on the host.
    """
    if factory is not None:
        try:
            return factory()
        except Exception as exc:  # noqa: BLE001 - wrapped, never hidden
            raise VideogenError(
                "code-video HTML capture needs a headless browser: "
                f"{exc}") from exc
    try:
        from ...browser.service import BrowserService
    except Exception as exc:  # noqa: BLE001 - wrapped, never hidden
        raise VideogenError(
            "code-video HTML capture needs the browser organ: "
            f"{exc}") from exc
    try:
        svc = BrowserService()
        return _RenderedSession(svc, size or CODE_DEFAULT_SIZE)
    except Exception as exc:  # noqa: BLE001 - wrapped, never hidden
        raise VideogenError(
            "code-video HTML capture needs a headless browser: "
            f"{exc}") from exc


def _render_html_scene(scene: dict[str, Any], out: Path, *,
                       duration: float, size: tuple[int, int],
                       fps: float,
                       browser_session_factory: Callable | None,
                       audio: str | os.PathLike | None = None,
                       narration: dict[str, Any] | None = None,
                       captions: dict[str, Any] | None = None,
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
    html_bytes = len(html.encode("utf-8"))
    if html_bytes > _HTML_MAX_BYTES:
        raise VideogenError(
            "HTML scene is too large for capture: "
            f"{html_bytes} bytes (max {_HTML_MAX_BYTES})")
    seek_js = str(scene.get("seek") or DEFAULT_SEEK_JS)
    n_frames = max(1, int(round(duration * fps)))
    width, height = int(size[0]), int(size[1])

    session = _open_browser_session(browser_session_factory, size=size)
    tab = None
    frame_size: list[int] | None = None
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
            nonlocal frame_size
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
            if frame_size is None:
                # Native size of the first raw capture — proof the
                # browser rendered at the scene size.
                frame_size = [img.width, img.height]
            if img.size != (width, height):
                img = img.resize((width, height))
            return img

        render_hash, sheet = _sequence(
            "html", n_frames, width, height, fps, out,
            _captioned(_frame, (captions or {}).get("words") or []),
            audio=audio)
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
            "render_hash": render_hash, "deterministic": True,
            "capture": {
                # "real" = a live headless-Chromium tab rendered the
                # page; "injected" = a factory-supplied session (tests)
                # — never labelled real when it was not.
                "browser": ("real" if browser_session_factory is None
                            else "injected"),
                "engine": ("playwright-chromium"
                           if browser_session_factory is None
                           else "factory"),
                "viewport": [width, height],
                "frame_size": frame_size or [width, height],
            },
            "narration": narration or {"status": "off", "backend": "",
                                       "segments": 0, "audio": "",
                                       "reason": ""},
            "captions": captions or {"status": "off", "words": [],
                                     "reason": ""}}


def render_code_scene(scene: dict[str, Any],
                      out: str | os.PathLike | None, *,
                      duration: float = 5.0,
                      size: tuple[int, int] | None = None,
                      fps: float | None = None,
                      browser_session_factory: Callable | None = None,
                      narrate: bool = True,
                      tts_fn: Callable | None = None,
                      captions: bool = True,
                      transcriber: Callable | None = None,
                      caption_model: str = "",
                      ) -> dict[str, Any]:
    """Render a scene spec to MP4 — deterministic seek(t) frames.

    ``scene`` is an explainer design dict (``{"engine": "diagram" |
    "equation" | "chart", ...}``) or an HTML scene (``{"html": ...}``).
    Returns the render record — ``path``, ``contact_sheet``,
    ``render_hash`` (identical across re-renders of the same scene),
    ``frames`` / ``fps`` / ``duration``, plus ``narration`` and
    ``captions``. HTML scenes also carry ``capture`` — ``browser``
    (``"real"`` when a live headless-Chromium tab rendered the page,
    ``"injected"`` for a factory-supplied session), ``engine``, the
    ``viewport`` the browser rendered at, and the native
    ``frame_size`` of the first capture. An HTML scene larger than
    ``_HTML_MAX_BYTES`` is refused before any browser work.

    Narration is ON by default — the video speaks its own script
    (the title and the labels of each build step, spoken through the
    TTS ladder and muxed into the MP4). ``narrate=False`` renders
    silent on purpose (``narration.status == "off"``). When narration
    was requested but no TTS backend is installed, the video still
    renders and ``narration`` reports ``status == "missing"`` with the
    reason — a silent file is never passed off as narrated. When the
    spoken words need more room than ``duration``, the render grows
    to fit the voice (the record's ``duration`` is the final one).
    ``tts_fn`` is an injectable ``(text, out_path) -> {"ok", ...}``
    seam for tests; production always uses the ladder.

    Captions are ON by default alongside narration — every word of
    the spoken script carries a start/end time and is burned into
    the frames word by word, in sync with the voice. The timings are
    measured with faster-whisper against the rendered narration
    audio when it is installed; otherwise they are split across the
    spoken segment windows and labelled ``"estimated"``, never
    dressed up as measured. ``captions=False`` renders without them
    on purpose (``captions.status == "off"``); a silent render has
    no voice to time against, so its captions are off too. An SRT
    sidecar is written beside the MP4 whenever words exist.
    ``transcriber`` is an injectable ``(path, *, language, model)``
    seam for tests; production uses the caption engine's
    faster-whisper seam.
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
    from .captions import caption_narration, captions_srt
    from .narration import scene_narration, synthesize_narration

    defaults = profile_defaults()
    size = size or tuple(defaults["size"])
    fps = fps or float(defaults["fps"])
    out_path = Path(out) if out is not None else _default_out()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    narration: dict[str, Any] = {"status": "off", "backend": "",
                                 "segments": 0, "audio": "", "reason": ""}
    audio: str | None = None
    synth_segments: list[dict[str, Any]] = []
    if narrate:
        segments = scene_narration(scene, duration=duration)
        if not segments:
            narration = {"status": "missing", "backend": "",
                         "segments": 0, "audio": "",
                         "reason": ("nothing to narrate — the scene "
                                    "carries no speakable text")}
        else:
            synth = synthesize_narration(
                segments, duration=duration, out_dir=str(out_path.parent),
                name=out_path.stem + ".narration.wav", tts_fn=tts_fn)
            if synth.get("ok"):
                duration = float(synth["duration"])
                audio = str(synth["path"])
                synth_segments = list(synth.get("segments") or [])
                narration = {"status": "spoken",
                             "backend": str(synth.get("backend") or ""),
                             "segments": len(synth_segments),
                             "audio": audio, "reason": ""}
            else:
                narration = {"status": "missing", "backend": "",
                             "segments": 0, "audio": "",
                             "reason": str(synth.get("reason") or
                                           "narration failed")}
    caption_record: dict[str, Any] = {"status": "off", "words": [],
                                      "reason": ""}
    if not narrate:
        caption_record = {"status": "off", "words": [],
                          "reason": ("captions follow the narration — "
                                     "a silent render has no voice to "
                                     "time against")}
    elif not captions:
        caption_record = {"status": "off", "words": [],
                          "reason": "captions disabled"}
    elif narration["status"] != "spoken":
        caption_record = {"status": "missing", "words": [],
                          "reason": str(narration.get("reason") or
                                        "narration missing")}
    else:
        caption_record = caption_narration(
            {"path": audio, "duration": duration,
             "segments": synth_segments},
            transcriber=transcriber, model=caption_model)
        if caption_record.get("words"):
            srt_path = out_path.with_name(out_path.stem + ".captions.srt")
            srt_path.write_text(
                captions_srt(caption_record["words"]), encoding="utf-8")
            caption_record["srt"] = str(srt_path)
    if "html" in scene:
        return _render_html_scene(scene, out_path, duration=duration,
                                  size=size, fps=fps,
                                  browser_session_factory=(
                                      browser_session_factory),
                                  audio=audio, narration=narration,
                                  captions=caption_record)
    return _render_design_scene(scene, out_path, duration=duration,
                                size=size, fps=fps, audio=audio,
                                narration=narration,
                                captions=caption_record)

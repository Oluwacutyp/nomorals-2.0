"""Screenshot capture for the vision system.

Three sources:
- Host screen: :func:`capture_screen` — captures the machine's own display
  via a strategy chain (termux-screenshot → mss → screencapture/ImageGrab →
  grim/scrot/maim/import), so the owner can say "look at my screen".
- Browser: :func:`capture_screenshot` — capture the current page of a
  browser session (needs Playwright; fails fast with a clear message when
  unavailable — e.g. on Termux).
- File: :func:`screenshot_from_file` — use an existing image path (e.g. a
  screenshot the owner sent via Telegram). Always works.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from ..core.errors import ToolError
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "capture_screenshot",
    "capture_screen",
    "screenshot_from_file",
    "ScreenshotUnavailable",
]


class ScreenshotUnavailable(ToolError):
    """Raised when a screenshot cannot be captured."""
    code = "vision.screenshot_unavailable"


def screenshot_from_file(path: str | Path) -> Path:
    """Use an existing image file as the screenshot source.

    Validates that the file exists and looks like an image.
    """
    path = Path(path)
    if not path.is_file():
        raise ScreenshotUnavailable(f"image file not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}:
        raise ScreenshotUnavailable(
            f"not an image file: {path} (suffix {suffix!r})"
        )
    if path.stat().st_size == 0:
        raise ScreenshotUnavailable(f"image file is empty: {path}")
    return path


def capture_screenshot(session: Any = None, *, dest: str | Path | None = None) -> Path:
    """Capture the current browser page as a PNG screenshot.

    Args:
        session: a ``nomorals.browser.service.BrowserTab`` (or anything with
            a ``screenshot(path)`` method). If None, tries the default
            browser tool session.
        dest: where to write the PNG. Defaults to a temp file.

    Fails fast with ScreenshotUnavailable when Playwright/Chromium is not
    available (e.g. on Termux phones).
    """
    tab = session
    if tab is None:
        tab = _default_browser_tab()
    if tab is None:
        raise ScreenshotUnavailable(
            "no browser session available for screenshots"
        )

    shot = tab.screenshot() if hasattr(tab, "screenshot") else None
    # BrowserTab.screenshot() writes into its session dir and returns paths.
    if isinstance(shot, dict):
        path = shot.get("screenshot")
        if path:
            return Path(path)
    elif isinstance(shot, (str, Path)):
        p = Path(shot)
        if p.is_file():
            return p

    raise ScreenshotUnavailable(
        "browser screenshot failed: Playwright/Chromium is required "
        "(pip install playwright && playwright install chromium). "
        "Not available on Termux/Android — send a screenshot image instead."
    )


def _default_browser_tab() -> Any | None:
    """Best-effort lookup of the current browser tab, else None."""
    try:
        from ..browser.service import get_default_tab
        return get_default_tab()
    except (ImportError, AttributeError):
        return None
    except Exception as exc:  # noqa: BLE001
        _log.debug("default browser tab lookup failed: %s", exc)
        return None


# ── host screen capture: strategy chain, best available wins ────────────

def _which(*names: str) -> str | None:
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


def _run_capture(argv: list[str], dest: Path,
                 timeout: float = 20.0) -> Path | None:
    """Run a screenshot binary; return ``dest`` when it produced a file."""
    try:
        proc = subprocess.run(argv, capture_output=True, timeout=timeout,
                              check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        _log.debug("screen capture %s failed: %s", argv[0], exc)
        return None
    if proc.returncode == 0 and dest.is_file() and dest.stat().st_size > 0:
        return dest
    _log.debug("screen capture %s exited %d: %s", argv[0], proc.returncode,
               (proc.stderr or b"")[:200])
    return None


def _screen_dest(dest: str | Path | None) -> Path:
    if dest is not None:
        return Path(dest)
    tmp = tempfile.NamedTemporaryFile(prefix="nm-screen-",
                                      suffix=".png", delete=False)
    tmp.close()
    return Path(tmp.name)


def capture_screen(*, dest: str | Path | None = None,
                   timeout: float = 20.0) -> Path:
    """Capture the machine's own display as a PNG.

    Strategy chain (best available wins, each miss logged):
    1. ``termux-screenshot`` (Termux/Android via termux-api).
    2. ``mss`` python library (cross-platform, in-process).
    3. macOS: ``screencapture -x``. Windows: Pillow ``ImageGrab``.
    4. Linux Wayland: ``grim`` (scrot/maim capture *black* on Wayland —
       never tried there). Linux X11: ``scrot``, ``maim``, or
       ImageMagick ``import -window root``.

    Returns the PNG path. Raises :class:`ScreenshotUnavailable` with the
    concrete install hint when nothing can capture (headless server,
    Termux without termux-api, …). Multi-monitor setups capture the
    composited desktop unless the backend restricts otherwise.
    """
    out = _screen_dest(dest)
    misses: list[str] = []

    def _miss(why: str) -> None:
        misses.append(why)
        _log.debug("screen capture backend missed: %s", why)

    # 1. Termux (Android) — the phone is a first-class Devon host.
    if _which("termux-screenshot"):
        got = _run_capture(["termux-screenshot", "-p", str(out)], out,
                           timeout=timeout)
        if got:
            return got
        _miss("termux-screenshot ran but produced no file "
              "(grant storage permission in the Termux:API app?)")

    # 2. mss — cross-platform, no external binary.
    try:
        import mss  # type: ignore[import]
        try:
            with mss.mss() as sct:
                sct.shot(output=str(out))
            if out.is_file() and out.stat().st_size > 0:
                return out
        except Exception as exc:  # noqa: BLE001 - e.g. no display
            _miss(f"mss failed ({exc})")
    except ImportError:
        _miss("mss not installed (pip install mss)")

    platform = sys.platform
    # 3. macOS / Windows natives.
    if platform == "darwin":
        if _which("screencapture"):
            got = _run_capture(["screencapture", "-x", str(out)], out,
                               timeout=timeout)
            if got:
                return got
            _miss("screencapture failed")
    elif platform == "win32":
        try:
            from PIL import ImageGrab
            shot = ImageGrab.grab()
            shot.save(out, format="PNG")
            return out
        except Exception as exc:  # noqa: BLE001
            _miss(f"PIL ImageGrab failed ({exc})")

    # 4. Linux: Wayland and X11 need different tools.
    if platform.startswith("linux"):
        session = (os.environ.get("XDG_SESSION_TYPE") or "").lower()
        has_display = bool(os.environ.get("DISPLAY") or
                           os.environ.get("WAYLAND_DISPLAY"))
        if session == "wayland" or (not session and
                                    os.environ.get("WAYLAND_DISPLAY")):
            if _which("grim"):
                got = _run_capture(["grim", str(out)], out, timeout=timeout)
                if got:
                    return got
                _miss("grim failed")
            else:
                _miss("Wayland session: grim not installed "
                      "(apt install grim)")
        else:
            # X11 (or unknown with a DISPLAY): scrot → maim → import.
            if not has_display:
                _miss("no DISPLAY/WAYLAND_DISPLAY — headless session")
            if _which("scrot"):
                got = _run_capture(["scrot", "-z", str(out)], out,
                                   timeout=timeout)
                if got:
                    return got
                _miss("scrot failed")
            if _which("maim"):
                got = _run_capture(["maim", str(out)], out, timeout=timeout)
                if got:
                    return got
                _miss("maim failed")
            if _which("import"):
                got = _run_capture(["import", "-window", "root", str(out)],
                                   out, timeout=timeout)
                if got:
                    return got
                _miss("ImageMagick 'import' failed (apt install imagemagick)")

    raise ScreenshotUnavailable(
        "could not capture the screen — no working backend. Tried: "
        + ("; ".join(misses) if misses else "nothing available")
        + ". Install one: Termux → termux-api app + pkg install termux-api; "
          "Linux X11 → apt install scrot; Linux Wayland → apt install grim; "
          "any OS → pip install mss."
    )

"""Screenshot capture for the vision system.

Two sources:
- Browser: capture the current page of a browser session (needs Playwright;
  fails fast with a clear message when unavailable — e.g. on Termux).
- File: use an existing image path (e.g. a screenshot the owner sent via
  Telegram). Always works.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.errors import ToolError
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "capture_screenshot",
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

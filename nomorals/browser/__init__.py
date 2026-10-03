"""Browser service (layer 4): sessions, tabs, downloads, screenshots."""

from .service import (
    BrowserError,
    BrowserService,
    DownloadResult,
    RenderedTab,
    ScreenshotResult,
    SessionHandle,
    Tab,
)

__all__ = [
    "BrowserError",
    "BrowserService",
    "DownloadResult",
    "RenderedTab",
    "ScreenshotResult",
    "SessionHandle",
    "Tab",
]

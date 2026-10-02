"""Browser service (layer 4): sessions, tabs, downloads, screenshots."""

from .service import (
    BrowserError,
    BrowserService,
    DownloadResult,
    ScreenshotResult,
    SessionHandle,
    Tab,
)

__all__ = [
    "BrowserError",
    "BrowserService",
    "DownloadResult",
    "ScreenshotResult",
    "SessionHandle",
    "Tab",
]

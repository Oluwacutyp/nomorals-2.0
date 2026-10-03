"""Browser service (layer 4): sessions, tabs, downloads, screenshots."""

from .daemon import (
    DaemonClient,
    DaemonControl,
    DaemonError,
    default_data_dir,
    republish_events,
)
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
    "DaemonClient",
    "DaemonControl",
    "DaemonError",
    "DownloadResult",
    "ScreenshotResult",
    "SessionHandle",
    "Tab",
    "default_data_dir",
    "republish_events",
]

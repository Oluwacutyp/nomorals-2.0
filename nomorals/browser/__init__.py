"""Browser service (layer 4): sessions, tabs, downloads, screenshots."""

from .daemon import (
    DaemonClient,
    DaemonControl,
    DaemonError,
    default_data_dir,
    republish_events,
)
from .errors import (
    BrowserBotDetectedError,
    BrowserError,
    BrowserNetworkError,
    BrowserSiteError,
    classify_exception,
    classify_http_status,
    describe,
    detect_challenge,
)
from .pacing import Pacing, pacing_from_env
from .service import (
    BrowserService,
    DownloadResult,
    RenderedTab,
    ScreenshotResult,
    SessionHandle,
    Tab,
)

__all__ = [
    "BrowserBotDetectedError",
    "BrowserError",
    "BrowserNetworkError",
    "BrowserSiteError",
    "BrowserService",
    "DaemonClient",
    "DaemonControl",
    "DaemonError",
    "DownloadResult",
    "Pacing",
    "RenderedTab",
    "ScreenshotResult",
    "SessionHandle",
    "Tab",
    "classify_exception",
    "classify_http_status",
    "default_data_dir",
    "describe",
    "detect_challenge",
    "pacing_from_env",
    "republish_events",
]

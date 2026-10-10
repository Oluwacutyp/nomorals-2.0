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
    summarize,
)
from .forms import (
    describe_forms,
    resolve_candidates,
)
from .liveview import LiveView, action_icon, format_elapsed
from .pacing import Pacing, pacing_from_env
from .service import (
    BrowserService,
    DownloadResult,
    RenderedTab,
    ScreenshotResult,
    SessionHandle,
    Tab,
)
from .styles import (
    download_line,
    error_card,
    pacing_line,
    snapshot_block,
    stats_block,
    tab_line,
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
    "LiveView",
    "Pacing",
    "RenderedTab",
    "ScreenshotResult",
    "SessionHandle",
    "Tab",
    "action_icon",
    "classify_exception",
    "classify_http_status",
    "default_data_dir",
    "describe",
    "describe_forms",
    "detect_challenge",
    "download_line",
    "error_card",
    "format_elapsed",
    "pacing_from_env",
    "pacing_line",
    "republish_events",
    "resolve_candidates",
    "snapshot_block",
    "stats_block",
    "summarize",
    "tab_line",
]

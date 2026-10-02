"""Browser service (layer 4): multi-session, multi-tab browsing over
``nomorals.tools.browser.BrowserSession``.

A :class:`BrowserService` owns named sessions. Each session holds tabs;
each :class:`Tab` wraps exactly one ``tools.browser.BrowserSession`` and
adds per-tab navigation history on top of it. State persists to disk
(``sessions.json``) and cookie jars persist per session, so logins survive
restarts. Downloads stream through the owning tab's cookie jar; screenshots
use headless Chromium via playwright when it is installed.
"""

from __future__ import annotations

import http.cookiejar
import importlib.util
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.errors import ToolError
from ..core.ids import ulid_now
from ..core.logging_setup import get_logger
from ..storage.artifacts import Provenance
from ..tools.browser import BrowserSession

__all__ = [
    "BrowserError",
    "Tab",
    "SessionHandle",
    "BrowserService",
    "DownloadResult",
    "ScreenshotResult",
]

_log = get_logger(__name__)

_SESSIONS_FILE = "sessions.json"
_SAVE_VERSION = 1


class BrowserError(Exception):
    """Anything the browser service refuses to fake: navigation failures,
    unknown tabs/sessions, HTTP errors on download, missing playwright."""


# ── result dataclasses ───────────────────────────────────────────────────────


@dataclass
class DownloadResult:
    path: str
    size: int
    mime: str
    artifact_uri: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size": self.size,
            "mime": self.mime,
            "artifact_uri": self.artifact_uri,
        }


@dataclass
class ScreenshotResult:
    path: str
    artifact_uri: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "artifact_uri": self.artifact_uri}


# ── tab ──────────────────────────────────────────────────────────────────────


class Tab:
    """One browser tab: exactly one ``tools.browser.BrowserSession`` plus the
    tab's own navigation history."""

    def __init__(self, tab_id: str, session_name: str, session: BrowserSession) -> None:
        self.tab_id = tab_id
        self.session_name = session_name
        self.session = session
        self.url: str = ""
        self.title: str = ""
        self.history: list[dict[str, Any]] = []
        #: last load failure that did not kill the tab (set by restore);
        #: empty when the tab loaded cleanly.
        self.error: str = ""

    # -- navigation ----------------------------------------------------------
    def _load_page(self, url: str) -> dict[str, Any]:
        """Open the page in the wrapped session; update url/title. Raises
        BrowserError on any failure (fail fast — never a half-loaded tab)."""
        url = (url or "").strip()
        if not url:
            raise BrowserError("navigate needs a url")
        try:
            result = self.session.open(url)
        except ToolError as exc:
            raise BrowserError(f"navigate {url} failed: {exc}") from exc
        if not result.get("ok", True):
            raise BrowserError(
                f"navigate {url} failed: HTTP {result.get('status')}")
        self.url = self.session.url
        self.title = self.session.title
        self.error = ""
        return result

    def navigate(self, url: str) -> dict[str, Any]:
        """Navigate and append to history AFTER the load succeeded."""
        result = self._load_page(url)
        self.history.append(
            {"url": self.url, "title": self.title, "ts": time.time()})
        return result

    def back(self) -> dict[str, Any]:
        """Re-navigate the previous history entry. The current entry is
        dropped; the previous one is refreshed in place."""
        if len(self.history) < 2:
            raise BrowserError("no previous page in tab history")
        self.history.pop()
        prev = self.history[-1]
        result = self._load_page(prev["url"])
        prev["ts"] = time.time()
        prev["title"] = self.title
        return result

    # -- page work (delegated to the wrapped BrowserSession) -----------------
    def _delegate(self, name: str, *args: Any, **kwargs: Any) -> Any:
        try:
            return getattr(self.session, name)(*args, **kwargs)
        except ToolError as exc:
            raise BrowserError(f"{name} on {self.url or '(no page)'} failed: {exc}") from exc

    def text(self, max_chars: int = 40000) -> dict[str, Any]:
        return self._delegate("text", max_chars=max_chars)

    def markdown(self, max_chars: int = 40000) -> dict[str, Any]:
        return self._delegate("markdown", max_chars=max_chars)

    def links(self) -> dict[str, Any]:
        return self._delegate("links")

    def click(self, target: str) -> dict[str, Any]:
        result = self._delegate("click", target=target)
        self.url = self.session.url
        self.title = self.session.title
        self.history.append(
            {"url": self.url, "title": self.title, "ts": time.time()})
        return result

    def fill(self, name: str, value: str) -> dict[str, Any]:
        return self._delegate("fill", name=name, value=value)

    def submit(self, target: str) -> dict[str, Any]:
        result = self._delegate("submit", target=target)
        self.url = self.session.url
        self.title = self.session.title
        self.history.append(
            {"url": self.url, "title": self.title, "ts": time.time()})
        return result

    def extract(self, target: str = "", kind: str = "") -> dict[str, Any]:
        return self._delegate("extract", target=target, kind=kind)

    def state(self) -> dict[str, Any]:
        base = self._delegate("state")
        base.update({
            "tab_id": self.tab_id,
            "url": self.url,
            "title": self.title,
            "history": [dict(h) for h in self.history],
            "error": self.error,
        })
        return base

    def close(self) -> None:
        """Persist cookies and release the wrapped session."""
        self.session.close()

    def to_dict(self) -> dict[str, Any]:
        return {
            "tab_id": self.tab_id,
            "url": self.url,
            "title": self.title,
            "history": [dict(h) for h in self.history],
        }


# ── session handle ───────────────────────────────────────────────────────────


class SessionHandle:
    """The tabs of one named browsing session."""

    def __init__(self, name: str, service: BrowserService) -> None:
        self.name = name
        self._service = service
        self._tabs: dict[str, Tab] = {}
        self._active_tab_id: str = ""

    # -- tabs ----------------------------------------------------------------
    def open_tab(self, url: str = "") -> Tab:
        tab_id = ulid_now()
        tab = self._service._make_tab(self, tab_id)
        self._tabs[tab_id] = tab
        self._active_tab_id = tab_id
        if (url or "").strip():
            try:
                tab.navigate(url)
            except BrowserError:
                # an un-navigable tab is still a tab (mirrors restore
                # tolerance); the error is recorded on the tab.
                tab.error = f"open_tab navigate failed for {url}"
                _log.warning("open_tab %s: initial navigate failed", url)
        return tab

    def close_tab(self, tab_id: str) -> None:
        tab = self._tabs.get(tab_id)
        if tab is None:
            raise BrowserError(f"unknown tab {tab_id!r} in session {self.name!r}")
        tab.close()
        del self._tabs[tab_id]
        if self._active_tab_id == tab_id:
            self._active_tab_id = next(iter(self._tabs), "")

    def switch_tab(self, tab_id: str) -> Tab:
        tab = self._tabs.get(tab_id)
        if tab is None:
            raise BrowserError(f"unknown tab {tab_id!r} in session {self.name!r}")
        self._active_tab_id = tab_id
        return tab

    @property
    def active_tab(self) -> Tab:
        tab = self._tabs.get(self._active_tab_id)
        if tab is None:
            raise BrowserError(f"session {self.name!r} has no active tab")
        return tab

    def list_tabs(self) -> list[dict[str, str]]:
        return [
            {"tab_id": t.tab_id, "url": t.url, "title": t.title}
            for t in self._tabs.values()
        ]

    def navigate(self, url: str) -> dict[str, Any]:
        return self.active_tab.navigate(url)

    def history(self) -> list[dict[str, Any]]:
        """Merged history of all tabs, oldest first."""
        merged: list[dict[str, Any]] = []
        for tab in self._tabs.values():
            for entry in tab.history:
                merged.append({"tab_id": tab.tab_id, **entry})
        merged.sort(key=lambda e: e.get("ts", 0.0))
        return merged

    # -- persistence -----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "tabs": [t.to_dict() for t in self._tabs.values()],
            "active_tab_id": self._active_tab_id,
        }


# ── service ──────────────────────────────────────────────────────────────────


class BrowserService:
    """Multi-session, multi-tab browsing built on ``tools.browser``.

    * sessions map to cookie directories (logins persist across restarts);
    * tabs track their own history;
    * downloads reuse the owning tab's cookies;
    * screenshots use real headless Chromium via playwright — never faked.
    """

    def __init__(
        self,
        *,
        data_dir: str | os.PathLike[str] | None = None,
        artifact_store: Any = None,
        mission_id: str = "",
    ) -> None:
        self.data_dir = Path(
            data_dir if data_dir is not None
            else Path.home() / ".nomorals" / "browser-service"
        ).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, SessionHandle] = {}
        self._artifact_store = artifact_store
        self._mission_id = mission_id or ""

    # -- sessions --------------------------------------------------------------
    def _cookie_dir(self, name: str) -> str:
        return str(self.data_dir / "cookies" / name)

    def _make_tab(self, handle: SessionHandle, tab_id: str) -> Tab:
        session = BrowserSession(
            name=f"{handle.name}:{tab_id}",
            session_dir=self._cookie_dir(handle.name),
        )
        return Tab(tab_id=tab_id, session_name=handle.name, session=session)

    def open_session(self, name: str) -> SessionHandle:
        name = (name or "").strip()
        if not name:
            raise BrowserError("session name must not be empty")
        if name in self._sessions:
            raise BrowserError(f"session {name!r} is already open")
        handle = SessionHandle(name, self)
        self._sessions[name] = handle
        return handle

    def close_session(self, name: str) -> None:
        handle = self._sessions.get(name)
        if handle is None:
            raise BrowserError(f"unknown session {name!r}")
        for tab_id in list(handle._tabs):
            handle.close_tab(tab_id)
        del self._sessions[name]
        self.save()

    def list_sessions(self) -> list[str]:
        return list(self._sessions)

    def get_session(self, name: str) -> SessionHandle:
        handle = self._sessions.get(name)
        if handle is None:
            raise BrowserError(f"unknown session {name!r}")
        return handle

    def find_tab(self, tab_id: str) -> Tab | None:
        for handle in self._sessions.values():
            tab = handle._tabs.get(tab_id)
            if tab is not None:
                return tab
        return None

    # -- store / mission wiring --------------------------------------------------
    def attach_store(self, store: Any) -> None:
        self._artifact_store = store

    def set_mission(self, mission_id: str) -> None:
        self._mission_id = mission_id or ""

    # -- downloads ---------------------------------------------------------------
    def download(
        self,
        tab_or_url: Tab | str,
        url: str = "",
        *,
        filename: str = "",
    ) -> DownloadResult:
        """Download ``url`` reusing the owning tab's cookies.

        ``tab_or_url`` is a :class:`Tab`, a tab id string (resolved across
        sessions), or — when no tab is involved — the URL itself, in which
        case ``url`` may be omitted.
        """
        tab: Tab | None = None
        if isinstance(tab_or_url, Tab):
            tab = tab_or_url
        elif isinstance(tab_or_url, str):
            tab = self.find_tab(tab_or_url)
            if tab is None:
                if (url or "").strip():
                    raise BrowserError(
                        f"unknown tab {tab_or_url!r} and a url was also given — ambiguous")
                url = tab_or_url
        else:
            raise BrowserError(
                f"download needs a Tab, tab id, or URL — got {type(tab_or_url).__name__}")

        url = (url or "").strip()
        if not url:
            raise BrowserError("download needs a url")
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise BrowserError(f"download: unsupported URL scheme {parsed.scheme!r}")

        session_name = tab.session_name if tab is not None else "default"
        dest_dir = self.data_dir / "downloads" / session_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        name = (filename or "").strip() or _filename_from_url(url)
        path = dest_dir / name

        request = urllib.request.Request(
            url, headers={"User-Agent": "NoMoralsBrowser/0.1 (download)"})
        if tab is not None:
            jar: http.cookiejar.CookieJar = tab.session.cookie_jar
            jar.add_cookie_header(request)
        try:
            with urllib.request.build_opener().open(request, timeout=60) as response:
                status = getattr(response, "status", 200)
                if status >= 400:
                    raise BrowserError(f"download {url} failed: HTTP {status}")
                mime = (response.headers.get("Content-Type", "") or "").split(";")[0].strip()
                data = response.read()
        except urllib.error.HTTPError as exc:
            raise BrowserError(f"download {url} failed: HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise BrowserError(f"download {url} failed: {exc.reason}") from exc

        with open(path, "wb") as fh:
            fh.write(data)

        artifact_uri: str | None = None
        if self._artifact_store is not None:
            art = self._artifact_store.put(
                data,
                type="download",
                mime=mime,
                creator="browser-service",
                mission_id=self._mission_id,
                provenance=Provenance(source_type="browser", source_id=url),
            )
            artifact_uri = art.uri
        return DownloadResult(path=str(path), size=len(data), mime=mime,
                              artifact_uri=artifact_uri)

    # -- screenshots -------------------------------------------------------------
    def screenshot(self, tab: Tab | str, *, full_page: bool = False) -> ScreenshotResult:
        """Real PNG screenshot of the tab's current URL via headless Chromium.

        Fail fast: if playwright is not installed this raises BrowserError —
        it never returns a fake or empty image.
        """
        if isinstance(tab, str):
            resolved = self.find_tab(tab)
            if resolved is None:
                raise BrowserError(f"unknown tab {tab!r}")
            tab = resolved
        if not tab.url:
            raise BrowserError("screenshot needs a tab with a loaded URL")
        if importlib.util.find_spec("playwright") is None:
            raise BrowserError(
                "screenshots require the 'playwright' package: "
                "pip install playwright && playwright install chromium")

        dest_dir = self.data_dir / "screenshots" / tab.session_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = dest_dir / f"shot-{int(time.time() * 1000)}.png"
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                try:
                    page = browser.new_page()
                    page.goto(tab.url, timeout=30_000, wait_until="load")
                    page.screenshot(path=str(path), full_page=full_page)
                finally:
                    browser.close()
        except BrowserError:
            raise
        except Exception as exc:
            raise BrowserError(f"screenshot of {tab.url} failed: {exc}") from exc

        if not path.is_file() or path.stat().st_size == 0:
            raise BrowserError(f"screenshot of {tab.url} produced no image")

        artifact_uri: str | None = None
        if self._artifact_store is not None:
            art = self._artifact_store.put(
                path.read_bytes(),
                type="screenshot",
                mime="image/png",
                creator="browser-service",
                mission_id=self._mission_id,
                provenance=Provenance(source_type="browser", source_id=tab.url),
            )
            artifact_uri = art.uri
        return ScreenshotResult(path=str(path), artifact_uri=artifact_uri)

    # -- persistence ---------------------------------------------------------------
    def save(self) -> str:
        """Atomically persist all open sessions to ``sessions.json``."""
        payload = {
            "version": _SAVE_VERSION,
            "saved_at": time.time(),
            "sessions": {
                name: handle.to_dict()
                for name, handle in self._sessions.items()
            },
        }
        path = self.data_dir / _SESSIONS_FILE
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, path)
        return str(path)

    def restore(self) -> int:
        """Reload ``sessions.json``: re-create sessions/tabs and re-navigate
        each tab's last URL.

        A tab whose re-navigation fails keeps the tab (and its saved
        history) with the error recorded in ``tab.error`` — one broken page
        never kills the restore.
        """
        path = self.data_dir / _SESSIONS_FILE
        if not path.is_file():
            return 0
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BrowserError(f"cannot read {path}: {exc}") from exc
        sessions = payload.get("sessions") or {}
        restored = 0
        for name, sdata in sessions.items():
            if name in self._sessions:
                continue  # already open — never clobber live state
            handle = SessionHandle(name, self)
            self._sessions[name] = handle
            for tdata in sdata.get("tabs", []):
                tab_id = str(tdata.get("tab_id") or ulid_now())
                tab = self._make_tab(handle, tab_id)
                handle._tabs[tab_id] = tab
                raw_history = tdata.get("history") or []
                tab.history = [
                    {"url": str(h.get("url", "")),
                     "title": str(h.get("title", "")),
                     "ts": float(h.get("ts", 0.0))}
                    for h in raw_history if h.get("url")
                ]
                last_url = tab.history[-1]["url"] if tab.history else (tdata.get("url") or "")
                if last_url:
                    try:
                        tab._load_page(last_url)
                    except BrowserError as exc:
                        tab.error = str(exc)
                        tab.url = last_url
                        _log.warning("restore: tab %s of %r failed to re-navigate: %s",
                                     tab_id, name, exc)
            if sdata.get("active_tab_id") in handle._tabs:
                handle._active_tab_id = sdata["active_tab_id"]
            elif handle._tabs:
                handle._active_tab_id = next(iter(handle._tabs))
            restored += 1
        return restored


def _filename_from_url(url: str) -> str:
    base = urllib.parse.unquote(
        urllib.parse.urlparse(url).path.rsplit("/", 1)[-1]).strip()
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in base).strip("._")
    return safe or "download.bin"

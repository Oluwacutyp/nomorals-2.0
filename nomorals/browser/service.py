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
    "RenderedTab",
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

# Playwright install hint, same shape as the screenshot() fail-fast.
_PLAYWRIGHT_HINT = (
    "rendered tabs require the 'playwright' package and a chromium build: "
    "pip install playwright && playwright install chromium"
)

#: goto timeout for rendered tabs (Cloudflare-guarded pages can be slow).
_RENDERED_GOTO_TIMEOUT_MS = 60_000


def _require_playwright_sync():
    """The playwright sync API factory, or a BrowserError that says exactly
    how to get it. Never imports playwright at module load."""
    if importlib.util.find_spec("playwright") is None:
        raise BrowserError(_PLAYWRIGHT_HINT)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise BrowserError(_PLAYWRIGHT_HINT) from exc
    return sync_playwright


class RenderedTab:
    """A playwright-backed tab with the same navigate/text/links shape as
    :class:`Tab`, rendered through real headless Chromium.

    Plain-HTTP tabs cannot pass Cloudflare managed challenges ("Just a
    moment..."); a rendered tab executes the page's JavaScript and keeps
    cookies via playwright's ``storage_state``, persisted to the session's
    cookie directory so logins survive restarts.

    The tab does NOT own the playwright driver: it receives the started
    driver object from :meth:`BrowserService.open_rendered_tab` and
    launches one browser per tab on the first ``navigate()``. The driver
    itself lives as long as the service — playwright's sync API cannot
    be stopped and restarted in one thread, so the service starts it once
    and only tears it down in :meth:`BrowserService.shutdown`.
    """

    def __init__(
        self,
        tab_id: str,
        session_name: str,
        storage_state_path: str | os.PathLike[str],
        playwright: Any,
    ) -> None:
        self.tab_id = tab_id
        self.session_name = session_name
        self.url: str = ""
        self.title: str = ""
        self.history: list[dict[str, Any]] = []
        #: last load failure (set by navigate); empty when the tab is clean.
        self.error: str = ""
        self._storage_state_path = Path(storage_state_path)
        #: started driver object (owns .chromium); owned by the service.
        self._playwright = playwright
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None

    # -- browser lifecycle ---------------------------------------------------
    def _ensure_page(self) -> Any:
        """Launch this tab's browser (once) and return the page. Fail fast:
        a launch failure raises BrowserError, never None."""
        if self._page is not None:
            return self._page
        try:
            self._browser = self._playwright.chromium.launch(headless=True)
            state = str(self._storage_state_path)
            if self._storage_state_path.is_file():
                self._context = self._browser.new_context(storage_state=state)
            else:
                self._context = self._browser.new_context()
            self._page = self._context.new_page()
        except Exception as exc:  # noqa: BLE001 - launch errors are opaque
            self._teardown_quiet()
            raise BrowserError(
                f"rendered tab could not launch chromium: {exc}. "
                f"{_PLAYWRIGHT_HINT}"
            ) from exc
        return self._page

    def _teardown_quiet(self) -> None:
        for attr in ("_page", "_context", "_browser"):
            obj = getattr(self, attr)
            setattr(self, attr, None)
            if obj is None:
                continue
            close = getattr(obj, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - teardown best effort
                    _log.debug("rendered tab teardown %s.close failed", attr)

    # -- navigation ----------------------------------------------------------
    def navigate(self, url: str) -> dict[str, Any]:
        """Render the page and append to history AFTER the load succeeded."""
        url = (url or "").strip()
        if not url:
            raise BrowserError("navigate needs a url")
        page = self._ensure_page()
        try:
            page.goto(url, wait_until="domcontentloaded",
                      timeout=_RENDERED_GOTO_TIMEOUT_MS)
        except Exception as exc:  # noqa: BLE001 - goto errors are opaque
            self.error = f"navigate {url} failed: {exc}"
            raise BrowserError(self.error) from exc
        self.url = page.url
        try:
            self.title = page.title()
        except Exception:  # noqa: BLE001 - title is cosmetic
            self.title = ""
        self.error = ""
        entry = {"url": self.url, "title": self.title, "ts": time.time()}
        self.history.append(entry)
        return dict(entry)

    def _require_loaded(self) -> Any:
        if self._page is None or not self.url:
            raise BrowserError("rendered tab has no loaded page — navigate first")
        return self._page

    # -- page work (same result shapes as Tab) --------------------------------
    def text(self, max_chars: int = 40000) -> dict[str, Any]:
        """Rendered DOM inner text of <body>."""
        page = self._require_loaded()
        try:
            content = page.inner_text("body")
        except Exception as exc:  # noqa: BLE001 - extraction errors are opaque
            raise BrowserError(f"text on {self.url} failed: {exc}") from exc
        return {"url": self.url, "title": self.title, "chars": len(content),
                "text": content[:max_chars],
                "truncated": len(content) > max_chars}

    def links(self, max_links: int = 100) -> dict[str, Any]:
        """Rendered links; ``el.href`` is already absolute per the DOM."""
        page = self._require_loaded()
        try:
            raw = page.eval_on_selector_all(
                "a[href]",
                "els => els.map(e => ({text: (e.innerText || '').trim(), "
                "href: e.href}))",
            )
        except Exception as exc:  # noqa: BLE001 - extraction errors are opaque
            raise BrowserError(f"links on {self.url} failed: {exc}") from exc
        out: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in raw or []:
            href = str((item or {}).get("href", ""))
            parsed = urllib.parse.urlparse(href)
            if parsed.scheme not in {"http", "https"}:
                continue
            key = href.split("#", 1)[0]
            if key in seen:
                continue
            seen.add(key)
            out.append({"text": str((item or {}).get("text", "")),
                        "url": href})
            if len(out) >= max_links:
                break
        return {"url": self.url, "count": len(out), "links": out}

    def html(self, max_chars: int = 2_000_000) -> dict[str, Any]:
        """Full rendered HTML (post-JavaScript DOM) for structured parsing."""
        page = self._require_loaded()
        try:
            content = page.content()
        except Exception as exc:  # noqa: BLE001 - extraction errors are opaque
            raise BrowserError(f"html on {self.url} failed: {exc}") from exc
        return {"url": self.url, "title": self.title, "chars": len(content),
                "html": content[:max_chars],
                "truncated": len(content) > max_chars}

    def state(self) -> dict[str, Any]:
        return {
            "tab_id": self.tab_id,
            "session_name": self.session_name,
            "url": self.url,
            "title": self.title,
            "history": [dict(h) for h in self.history],
            "error": self.error,
            "rendered": True,
        }

    def close(self) -> None:
        """Persist cookies (storage_state) to the session dir, then tear the
        browser down. A state-save failure is logged, never masks teardown."""
        if self._context is not None:
            try:
                self._storage_state_path.parent.mkdir(parents=True,
                                                      exist_ok=True)
                self._context.storage_state(path=str(self._storage_state_path))
            except Exception as exc:  # noqa: BLE001 - persistence best effort
                _log.warning("rendered tab %s: could not persist cookies: %s",
                             self.tab_id, exc)
        self._teardown_quiet()

    def to_dict(self) -> dict[str, Any]:
        return {
            "tab_id": self.tab_id,
            "url": self.url,
            "title": self.title,
            "history": [dict(h) for h in self.history],
            "rendered": True,
        }


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
        self._rendered_tabs: dict[str, RenderedTab] = {}
        #: playwright driver for rendered tabs: started once, lazily, and
        #: kept for the service's lifetime — the sync API cannot be
        #: stopped and restarted in one thread.
        self._pw_cm: Any = None
        self._playwright: Any = None
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
        for tab_id, tab in list(self._rendered_tabs.items()):
            if tab.session_name == name:
                tab.close()
                del self._rendered_tabs[tab_id]
        del self._sessions[name]
        self.save()

    # -- rendered tabs (real headless Chromium via playwright) -----------------
    def _rendered_state_path(self, session_name: str) -> Path:
        return Path(self._cookie_dir(session_name)) / "playwright-storage.json"

    def _driver(self) -> Any:
        """The started playwright driver object, created once and kept for
        the service's lifetime. Fail fast with the install hint when
        playwright is missing or the driver won't start."""
        if self._playwright is not None:
            return self._playwright
        sync_playwright = _require_playwright_sync()  # fail fast first
        try:
            # NOTE: sync_playwright() returns a context manager; .chromium
            # only exists on the object start() returns (what ``with``
            # binds as ``p``). Teardown is __exit__ — there is no stop().
            self._pw_cm = sync_playwright()
            self._playwright = self._pw_cm.start()
        except Exception as exc:  # noqa: BLE001 - driver errors are opaque
            self._pw_cm = None
            self._playwright = None
            raise BrowserError(
                f"rendered tabs could not start the playwright driver: "
                f"{exc}. {_PLAYWRIGHT_HINT}"
            ) from exc
        return self._playwright

    def shutdown(self) -> None:
        """Close all rendered tabs and stop the playwright driver.

        Idempotent. After this, rendered tabs cannot be opened again on
        this service (the sync API cannot restart in one thread) — only
        call it when the service is truly done.
        """
        for tab_id in list(self._rendered_tabs):
            try:
                self.close_rendered_tab(tab_id)
            except Exception as exc:  # noqa: BLE001 - shutdown must complete
                _log.warning("shutdown: closing rendered tab %s failed: %s",
                             tab_id, exc)
        cm, self._pw_cm = self._pw_cm, None
        self._playwright = None
        if cm is not None:
            try:
                cm.__exit__(None, None, None)
            except Exception as exc:  # noqa: BLE001 - shutdown best effort
                _log.warning("browser service shutdown failed: %s", exc)

    def open_rendered_tab(self, session_name: str, url: str = "") -> RenderedTab:
        """Open a playwright-backed tab in ``session_name``'s cookie space.

        For JavaScript/Cloudflare-guarded pages that plain-HTTP tabs cannot
        pass. Cookies persist via playwright storage_state in the session's
        cookie dir. Fail fast: raises BrowserError when playwright or its
        chromium build is missing, or when the initial navigate fails.
        """
        session_name = (session_name or "").strip()
        if not session_name:
            raise BrowserError("session name must not be empty")
        playwright = self._driver()  # fail fast before touching anything
        tab_id = ulid_now()
        tab = RenderedTab(
            tab_id=tab_id,
            session_name=session_name,
            storage_state_path=self._rendered_state_path(session_name),
            playwright=playwright,
        )
        self._rendered_tabs[tab_id] = tab
        try:
            if (url or "").strip():
                tab.navigate(url)
        except BrowserError:
            tab.close()
            del self._rendered_tabs[tab_id]
            raise
        return tab

    def close_rendered_tab(self, tab_id: str) -> None:
        tab = self._rendered_tabs.get(tab_id)
        if tab is None:
            raise BrowserError(f"unknown rendered tab {tab_id!r}")
        tab.close()
        del self._rendered_tabs[tab_id]

    def find_rendered_tab(self, tab_id: str) -> RenderedTab | None:
        return self._rendered_tabs.get(tab_id)

    def list_rendered_tabs(self) -> list[dict[str, str]]:
        return [
            {"tab_id": t.tab_id, "session_name": t.session_name,
             "url": t.url, "title": t.title}
            for t in self._rendered_tabs.values()
        ]

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

"""Browser service (layer 4): multi-session, multi-tab browsing over
``nomorals.tools.browser.BrowserSession``.

A :class:`BrowserService` owns named sessions. Each session holds tabs;
each :class:`Tab` wraps exactly one ``tools.browser.BrowserSession`` and
adds per-tab navigation history on top of it. State persists to disk
(``sessions.json``) and cookie jars persist per session, so logins survive
restarts. Downloads stream through the owning tab's cookie jar and are
tracked in a registry (``downloads.json``); screenshots use headless
Chromium via playwright when it is installed. Sessions can route traffic
through a proxy pool (duck-typed — see :meth:`BrowserService.attach_proxy_pool`);
cookies can be inspected/exported/imported; forms support multipart file
uploads; rendered (playwright) tabs support fill/click/submit/wait/
extract/upload/download on the live DOM.
"""

from __future__ import annotations

import http.cookiejar
import importlib.util
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.errors import ToolError
from ..core.events import Event, global_bus
from ..core.ids import ulid_now
from ..core.logging_setup import get_logger
from ..storage.artifacts import Provenance
from ..tools.browser import (
    BrowserSession,
    dom_forms,
    dom_headings,
    dom_meta,
    dom_nav,
    dom_tables,
    parse_html,
)

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


def _emit(topic: str, data: dict[str, Any]) -> None:
    """Publish a telemetry event. Best-effort: a broken bus or subscriber
    must never break browsing (fail-open telemetry, fail-closed function)."""
    try:
        global_bus.publish(Event(topic=topic, data=data, source=__name__))
    except Exception:  # noqa: BLE001 - telemetry is fail-open
        _log.debug("event %s failed", topic, exc_info=True)

_SESSIONS_FILE = "sessions.json"
_DOWNLOADS_FILE = "downloads.json"
_SAVE_VERSION = 1

#: MIME top-type -> download subfolder used when organize=True.
_MIME_CATEGORIES = {
    "image": "images",
    "video": "videos",
    "audio": "audio",
}


def _mime_category(mime: str) -> str:
    mime = (mime or "").split(";", 1)[0].strip().lower()
    if mime == "application/octet-stream":
        # generic binary bucket — unknown content, not a document
        return "other"
    top = mime.split("/", 1)[0].strip().lower()
    if top in _MIME_CATEGORIES:
        return _MIME_CATEGORIES[top]
    if top in {"text", "application"}:
        return "documents"
    return "other"


def _mask_proxy(proxy_url: str) -> str:
    """Hide the proxy password for logs/events/results."""
    try:
        parsed = urllib.parse.urlparse(proxy_url)
        if parsed.password:
            netloc = parsed.hostname or ""
            if parsed.username:
                netloc = f"{parsed.username}:***@{netloc}"
            if parsed.port:
                netloc += f":{parsed.port}"
            return urllib.parse.urlunparse(
                (parsed.scheme, netloc, parsed.path, "", "", ""))
    except Exception:  # noqa: BLE001 - masking must never raise
        pass
    return proxy_url


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
        _emit("browser.tab.navigated", {
            "session": self.session_name,
            "tab_id": self.tab_id,
            "url": self.url,
            "title": self.title,
        })
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

    def html(self, max_chars: int = 2_000_000) -> dict[str, Any]:
        """Raw page HTML (un-parsed markup; includes ld+json scripts)."""
        return self._delegate("html", max_chars=max_chars)

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
        return self._delegate("fill", name, value)

    def select(self, name: str, value: str) -> dict[str, Any]:
        """Pick a ``<select>`` dropdown option (submitted on next submit)."""
        return self._delegate("select", name, value)

    def check(self, name: str, checked: bool = True) -> dict[str, Any]:
        """Check/uncheck a checkbox, or pick a radio button."""
        return self._delegate("check", name, checked)

    def check_captcha(self, *, fetch_bytes: bool = False) -> dict[str, Any]:
        """One-call captcha scan of this tab's current page HTML."""
        from ..tools.captcha import detect
        raw = getattr(self.session, "_raw", "") or ""
        found = detect(raw, self.url, fetch_bytes=fetch_bytes)
        return {"url": self.url, "count": len(found),
                "challenges": [c.summary() for c in found]}

    def upload(self, field_name: str, file_path: str) -> dict[str, Any]:
        """Stage a file for a ``<input type="file">`` field, then submit.

        This only *stages* the path (like :meth:`fill`); the multipart
        upload happens on the next :meth:`submit`. The file must exist and
        be readable — otherwise this fails fast here, not at submit time.
        """
        field_name = (field_name or "").strip()
        path = os.path.abspath(os.path.expanduser((file_path or "").strip()))
        if not field_name:
            raise BrowserError("upload needs a field name")
        if not os.path.isfile(path):
            raise BrowserError(f"upload: not a file: {file_path!r}")
        try:
            with open(path, "rb"):
                pass
        except OSError as exc:
            raise BrowserError(
                f"upload: cannot read {file_path!r}: {exc}") from exc
        return self._delegate("fill", field_name, path)

    def submit(self, target: str = "",
               uploads: dict[str, str] | None = None) -> dict[str, Any]:
        result = self._delegate("submit", target=target, uploads=uploads)
        self.url = self.session.url
        self.title = self.session.title
        self.history.append(
            {"url": self.url, "title": self.title, "ts": time.time()})
        return result

    def cookies(self) -> list[dict[str, Any]]:
        """This tab's live cookie jar as plain dicts (inspection, not just
        persistence)."""
        out: list[dict[str, Any]] = []
        for cookie in self.session.cookie_jar:
            out.append({
                "name": cookie.name,
                "value": cookie.value,
                "domain": cookie.domain,
                "path": cookie.path,
                "secure": bool(cookie.secure),
                "expires": cookie.expires,
                "http_only": bool(cookie.has_nonstandard_attr("HttpOnly")),
            })
        out.sort(key=lambda c: (c["domain"], c["path"], c["name"]))
        return out

    def set_proxy(self, proxy_url: str = "") -> dict[str, Any]:
        """Route this tab's session through ``proxy_url`` ("" = direct).

        The tab's cookie jar survives the switch."""
        return self.session.set_proxy(proxy_url)

    def extract(self, target: str = "", kind: str = "") -> dict[str, Any]:
        return self._delegate("extract", target=target, kind=kind)

    def task(self, steps: Any = None, **kwargs: Any) -> dict[str, Any]:
        """Run a multi-step task program on this tab's session."""
        return self._delegate("task", steps=steps, **kwargs)

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


def _playwright_proxy_config(proxy_url: str) -> dict[str, Any] | None:
    """Playwright's ``launch(proxy=...)`` dict from a proxy URL (userinfo
    becomes username/password). None when no proxy is configured."""
    proxy_url = (proxy_url or "").strip()
    if not proxy_url:
        return None
    parsed = urllib.parse.urlparse(
        proxy_url if "://" in proxy_url else f"http://{proxy_url}")
    if not parsed.hostname:
        raise BrowserError(f"bad proxy URL {proxy_url!r}")
    server = f"{parsed.scheme or 'http'}://{parsed.hostname}"
    if parsed.port:
        server += f":{parsed.port}"
    cfg: dict[str, Any] = {"server": server}
    if parsed.username:
        cfg["username"] = urllib.parse.unquote(parsed.username)
    if parsed.password:
        cfg["password"] = urllib.parse.unquote(parsed.password)
    return cfg


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
        proxy: str = "",
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
        #: proxy URL for this tab's chromium ("" = direct).
        self._proxy = (proxy or "").strip()
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
            launch_kwargs: dict[str, Any] = {"headless": True}
            proxy_cfg = _playwright_proxy_config(self._proxy)
            if proxy_cfg:
                launch_kwargs["proxy"] = proxy_cfg
            self._browser = self._playwright.chromium.launch(**launch_kwargs)
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

    # -- interaction (same verbs as Tab.fill/click/submit, on the live DOM) --
    @staticmethod
    def _field_selector(name: str) -> str:
        """Match a form field by name, then id — whichever the page uses."""
        escaped = (name or "").replace('"', '\\"')
        return f'input[name="{escaped}"], textarea[name="{escaped}"], select[name="{escaped}"], [id="{escaped}"]'

    def _field_kind(self, name: str) -> dict[str, str] | None:
        """Inspect the first matching field's tag/type via the live DOM.

        Returns {"tag": ..., "type": ...} (lowercased), or None when the
        page object cannot evaluate JS (duck-typed drivers) — callers
        then fall back to the untyped behavior. Raises BrowserError when
        nothing matches, so callers fail fast instead of guessing.
        """
        page = self._require_loaded()
        evaluate = getattr(page, "evaluate", None)
        if evaluate is None:
            return None
        selector = self._field_selector(name)
        js = """(sel) => {
            const el = document.querySelector(sel);
            if (!el) return null;
            return {tag: (el.tagName || '').toLowerCase(),
                    type: ((el.getAttribute('type') || '')).toLowerCase()};
        }"""
        try:
            info = evaluate(js, selector)
        except Exception as exc:  # noqa: BLE001 - eval errors are opaque
            raise BrowserError(
                f"rendered field inspection of {name!r} on {self.url} "
                f"failed: {exc}") from exc
        if not info:
            raise BrowserError(
                f"no form field {name!r} on {self.url}")
        return {"tag": info.get("tag", ""), "type": info.get("type", "")}

    def fill(self, name: str, value: str) -> dict[str, Any]:
        """Fill a form field by name or id on the rendered page.

        Type-aware: ``<select>`` fields route to :meth:`select`,
        checkboxes/radios route to :meth:`check`, file inputs fail fast
        with a pointer to :meth:`upload` — plain ``page.fill`` only ever
        touches real text-like inputs, so a select no longer dies with an
        opaque playwright error. When the field type cannot be inspected
        the untyped ``page.fill`` path is used (previous behavior).
        """
        page = self._require_loaded()
        name = (name or "").strip()
        if not name:
            raise BrowserError("rendered fill needs a field name")
        kind = self._field_kind(name)
        if kind is not None:
            tag, ftype = kind["tag"], kind["type"]
            if tag == "select":
                return self.select(name, value)
            if tag == "input" and ftype in {"checkbox", "radio"}:
                truthy = str(value).strip().lower() not in {
                    "", "0", "false", "no", "off", "unchecked"}
                return self.check(name, checked=truthy)
            if tag == "input" and ftype == "file":
                raise BrowserError(
                    f"field {name!r} is a file input — use rendered upload, "
                    "not fill")
        selector = self._field_selector(name)
        try:
            page.fill(selector, str(value))
        except Exception as exc:  # noqa: BLE001 - selector errors are opaque
            raise BrowserError(
                f"rendered fill of {name!r} on {self.url} failed: {exc}") from exc
        return {"ok": True, "field": name, "tab_id": self.tab_id}

    def select(self, name: str, value: str, *,
               by: str = "auto") -> dict[str, Any]:
        """Pick an option of a ``<select>`` dropdown by name or id.

        ``by``: "auto" (default) tries the option *value* first, then the
        visible *label*; "value" and "label" pin the match. Fail fast when
        the field is not a ``<select>`` or the option does not exist.
        """
        page = self._require_loaded()
        name = (name or "").strip()
        if not name:
            raise BrowserError("rendered select needs a field name")
        kind = self._field_kind(name)
        if kind is None:
            raise BrowserError(
                f"cannot inspect field {name!r} on this page driver — "
                "select needs a live DOM with JS evaluation")
        if kind["tag"] != "select":
            raise BrowserError(
                f"field {name!r} is a <{kind['tag']}> "
                f"(type={kind['type'] or 'n/a'}), not a <select>")
        by = (by or "auto").strip().lower()
        if by not in {"auto", "value", "label"}:
            raise BrowserError(
                f"unknown select match {by!r} (want auto|value|label)")
        selector = self._field_selector(name)
        attempts = ([{"value": value}, {"label": value}] if by == "auto"
                    else [{"value": value}] if by == "value"
                    else [{"label": value}])
        last_exc: Exception | None = None
        for kw in attempts:
            try:
                picked = page.select_option(selector, **kw)
            except Exception as exc:  # noqa: BLE001 - opaque
                last_exc = exc
                continue
            if picked:
                return {"ok": True, "field": name, "picked": picked,
                        "tab_id": self.tab_id}
            last_exc = BrowserError(
                f"no option matching {value!r} in select {name!r}")
        raise BrowserError(
            f"rendered select of {value!r} in {name!r} on {self.url} "
            f"failed: {last_exc}") from last_exc

    def check(self, name: str, checked: bool = True) -> dict[str, Any]:
        """Check/uncheck a checkbox (or pick a radio) by name or id.

        Fail fast when the field is not a checkable input.
        """
        page = self._require_loaded()
        name = (name or "").strip()
        if not name:
            raise BrowserError("rendered check needs a field name")
        kind = self._field_kind(name)
        if kind is None:
            raise BrowserError(
                f"cannot inspect field {name!r} on this page driver — "
                "check needs a live DOM with JS evaluation")
        if kind["tag"] != "input" or kind["type"] not in {"checkbox",
                                                          "radio"}:
            raise BrowserError(
                f"field {name!r} is a <{kind['tag']}> "
                f"(type={kind['type'] or 'n/a'}), not a checkbox/radio")
        if kind["type"] == "radio" and not checked:
            raise BrowserError(
                f"field {name!r} is a radio button — radios cannot be "
                "unchecked, pick another option in the group instead")
        selector = self._field_selector(name)
        try:
            if checked:
                page.check(selector)
            else:
                page.uncheck(selector)
        except Exception as exc:  # noqa: BLE001 - opaque
            raise BrowserError(
                f"rendered {'check' if checked else 'uncheck'} of {name!r} "
                f"on {self.url} failed: {exc}") from exc
        return {"ok": True, "field": name, "checked": bool(checked),
                "tab_id": self.tab_id}

    def click(self, target: str) -> dict[str, Any]:
        """Click a link/button: CSS selector when it looks like one
        (starts with ``#``, ``.``, ``[``, or contains ``>>``), otherwise
        visible text match. The tab's URL/title/history refresh after the
        click, like :meth:`Tab.click`."""
        page = self._require_loaded()
        target = (target or "").strip()
        if not target:
            raise BrowserError("rendered click needs a target")
        selector = (target if target[:1] in {"#", ".", "["} or ">>" in target
                    else f"text={target}")
        before = page.url
        try:
            page.click(selector)
        except Exception as exc:  # noqa: BLE001 - click errors are opaque
            raise BrowserError(
                f"rendered click of {target!r} on {self.url} failed: {exc}") from exc
        try:
            after = page.url
        except Exception:  # noqa: BLE001 - url read is cosmetic
            after = before
        if after != before:
            self.url = after
            try:
                self.title = page.title()
            except Exception:  # noqa: BLE001 - title is cosmetic
                pass
            self.history.append(
                {"url": self.url, "title": self.title, "ts": time.time()})
        return {"ok": True, "target": target, "url": self.url,
                "title": self.title, "navigated": after != before}

    def submit(self, target: str = "") -> dict[str, Any]:
        """Submit a form: click ``target`` when given (button text/selector),
        else submit the page's first form directly. URL/title/history
        refresh after the submit."""
        page = self._require_loaded()
        target = (target or "").strip()
        try:
            if target:
                self.click(target)
            else:
                page.eval_on_selector("form", "f => f.submit()")
        except BrowserError:
            raise
        except Exception as exc:  # noqa: BLE001 - submit errors are opaque
            raise BrowserError(
                f"rendered submit on {self.url} failed: {exc}") from exc
        try:
            self.url = page.url
            self.title = page.title()
        except Exception:  # noqa: BLE001 - cosmetic
            pass
        self.history.append(
            {"url": self.url, "title": self.title, "ts": time.time()})
        return {"ok": True, "url": self.url, "title": self.title}

    def wait_for(self, selector: str = "", *, state: str = "visible",
                 timeout: int = 10_000) -> dict[str, Any]:
        """Wait for a selector to reach ``state`` (visible|hidden|attached|
        detached). Fail fast on timeout — never a silent pass."""
        page = self._require_loaded()
        selector = (selector or "").strip()
        if not selector:
            raise BrowserError("rendered wait_for needs a selector")
        try:
            page.wait_for_selector(selector, state=state, timeout=int(timeout))
        except Exception as exc:  # noqa: BLE001 - timeout errors are opaque
            raise BrowserError(
                f"rendered wait_for {selector!r} ({state}) on {self.url} "
                f"timed out after {timeout}ms: {exc}") from exc
        return {"ok": True, "selector": selector, "state": state}

    def wait_for_url(self, pattern: str = "",
                     timeout: int = 10_000) -> dict[str, Any]:
        """Wait until the page URL matches ``pattern`` (substring or
        ``re:``-prefixed regex). Built for SPAs, where navigation happens
        without a page load after a click/submit. Fail fast on timeout."""
        page = self._require_loaded()
        pattern = (pattern or "").strip()
        if not pattern:
            raise BrowserError("rendered wait_for_url needs a pattern")
        try:
            if pattern.startswith("re:"):
                page.wait_for_url(re.compile(pattern[3:]),
                                  timeout=int(timeout))
            else:
                page.wait_for_url(f"*{pattern}*", timeout=int(timeout))
        except Exception as exc:  # noqa: BLE001 - timeout errors are opaque
            raise BrowserError(
                f"rendered wait_for_url {pattern!r} on {self.url} timed "
                f"out after {timeout}ms: {exc}") from exc
        try:
            self.url = page.url
            self.title = page.title()
        except Exception:  # noqa: BLE001 - cosmetic
            pass
        return {"ok": True, "pattern": pattern, "url": self.url}

    def wait_for_text(self, text: str = "",
                      timeout: int = 10_000) -> dict[str, Any]:
        """Wait until ``text`` appears anywhere in the rendered page.

        The dynamic-content counterpart to :meth:`wait_for`: SPAs that
        fetch content via XHR never add new selectors, but the text shows
        up. Fail fast on timeout.
        """
        page = self._require_loaded()
        text = (text or "").strip()
        if not text:
            raise BrowserError("rendered wait_for_text needs text")
        try:
            page.get_by_text(text).first.wait_for(timeout=int(timeout))
        except Exception as exc:  # noqa: BLE001 - timeout errors are opaque
            raise BrowserError(
                f"rendered wait_for_text {text!r} on {self.url} timed out "
                f"after {timeout}ms: {exc}") from exc
        return {"ok": True, "text": text}

    def check_captcha(self, *, fetch_bytes: bool = False) -> dict[str, Any]:
        """One-call captcha scan of the live rendered page.

        Runs :func:`nomorals.tools.captcha.detect` over the current DOM —
        the bot asks "is there a captcha on this page?" without scraping
        HTML itself. Returns the same challenge summaries the captcha
        tool reports (kind, sitekey, domain, image info).
        """
        from ..tools.captcha import detect
        page = self._require_loaded()
        try:
            html = page.content()
            url = page.url
        except Exception as exc:  # noqa: BLE001 - read errors are opaque
            raise BrowserError(
                f"rendered check_captcha on {self.url} failed: {exc}") from exc
        found = detect(html, url, fetch_bytes=fetch_bytes)
        self.url = url
        return {"url": url, "count": len(found),
                "challenges": [c.summary() for c in found]}

    def screenshot(self, *, full_page: bool = False,
                   path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
        """Screenshot THIS tab's live rendered page (unlike
        :meth:`BrowserService.screenshot`, this does not reload the URL in
        a fresh context — logged-in state and JS mutations are captured).
        """
        page = self._require_loaded()
        dest_dir = Path(self._storage_state_path).parent.parent / "screenshots" / self.session_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        out = Path(path) if path else dest_dir / f"rendered-{int(time.time() * 1000)}.png"
        try:
            page.screenshot(path=str(out), full_page=full_page)
        except Exception as exc:  # noqa: BLE001 - capture errors are opaque
            raise BrowserError(
                f"rendered screenshot of {self.url} failed: {exc}") from exc
        if not out.is_file() or out.stat().st_size == 0:
            raise BrowserError(
                f"rendered screenshot of {self.url} produced no image")
        return {"path": str(out), "url": self.url, "tab_id": self.tab_id}

    def extract(self, target: str = "", kind: str = "") -> dict[str, Any]:
        """Structured extraction from the rendered DOM: ``kind`` selects
        headings|tables|forms|meta|nav (same shapes as Tab.extract), or a
        ``target`` tag/#id/.class for plain text."""
        page = self._require_loaded()
        kind = (kind or "").strip().lower()
        if kind and kind not in {"headings", "h", "tables", "table", "forms",
                                 "form", "meta", "head", "nav", "links-structured",
                                 "sitemap"}:
            raise BrowserError(
                f"unknown extract kind {kind!r}; use headings|tables|forms|meta|nav")
        try:
            content = page.content()
        except Exception as exc:  # noqa: BLE001 - extraction errors are opaque
            raise BrowserError(f"rendered extract on {self.url} failed: {exc}") from exc
        dom = parse_html(content)
        if kind in {"headings", "h"}:
            return dom_headings(dom, self.url)
        if kind in {"tables", "table"}:
            return dom_tables(dom, self.url)
        if kind in {"forms", "form"}:
            return dom_forms(dom, self.url)
        if kind in {"meta", "head"}:
            return dom_meta(dom, self.url, self.title)
        if kind in {"nav", "links-structured", "sitemap"}:
            return dom_nav(dom, self.url)
        # selector mode: reuse the plain-tab text-of-elements shape
        wanted = (target or "body").strip()
        nodes = []
        if wanted.startswith("#"):
            ident = wanted[1:]
            nodes = [n for n in dom.walk()
                     if not n.is_text and n.attrs.get("id") == ident]
        elif wanted.startswith("."):
            cls = wanted[1:]
            nodes = [n for n in dom.walk()
                     if not n.is_text and cls in (n.attrs.get("class") or "").split()]
        else:
            nodes = dom.find_all(wanted or "body")
        matches = [n.inner_text().strip()[:2000] for n in nodes[:50]
                   if n.inner_text().strip()]
        return {"url": self.url, "count": len(matches), "matches": matches}

    def cookies(self) -> list[dict[str, Any]]:
        """This tab's live cookie jar (playwright context cookies) as dicts."""
        page = self._require_loaded()
        try:
            raw = self._context.cookies()
        except Exception as exc:  # noqa: BLE001 - cookie errors are opaque
            raise BrowserError(
                f"rendered cookies on {self.url} failed: {exc}") from exc
        out = [{
            "name": c.get("name", ""),
            "value": c.get("value", ""),
            "domain": c.get("domain", ""),
            "path": c.get("path", "/"),
            "secure": bool(c.get("secure")),
            "expires": c.get("expires", -1),
            "http_only": bool(c.get("httpOnly")),
        } for c in (raw or [])]
        out.sort(key=lambda c: (c["domain"], c["path"], c["name"]))
        return out

    def upload(self, selector: str, file_path: str) -> dict[str, Any]:
        """Set a ``<input type="file">`` to a local file (playwright
        set_input_files). The file must exist — fail fast otherwise."""
        page = self._require_loaded()
        selector = (selector or "").strip()
        path = os.path.abspath(os.path.expanduser((file_path or "").strip()))
        if not selector:
            raise BrowserError("rendered upload needs a selector")
        if not os.path.isfile(path):
            raise BrowserError(f"rendered upload: not a file: {file_path!r}")
        try:
            page.set_input_files(selector, path)
        except Exception as exc:  # noqa: BLE001 - upload errors are opaque
            raise BrowserError(
                f"rendered upload of {file_path!r} on {self.url} failed: {exc}") from exc
        return {"ok": True, "selector": selector, "path": path,
                "tab_id": self.tab_id}

    def trigger_download(self, target: str,
                         dest_dir: str | os.PathLike[str]) -> dict[str, Any]:
        """Click ``target`` and capture the download it triggers
        (playwright expect_download). Returns the saved path; the caller
        (the service) registers it in the download registry."""
        page = self._require_loaded()
        target = (target or "").strip()
        if not target:
            raise BrowserError("trigger_download needs a target")
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        selector = (target if target[:1] in {"#", ".", "["} or ">>" in target
                    else f"text={target}")
        try:
            with page.expect_download() as download_info:
                page.click(selector)
            download = download_info.value
        except Exception as exc:  # noqa: BLE001 - download errors are opaque
            raise BrowserError(
                f"download trigger {target!r} on {self.url} failed: {exc}") from exc
        suggested = str(getattr(download, "suggested_filename", "") or "download.bin")
        safe = "".join(c if (c.isalnum() or c in "._-") else "_"
                       for c in suggested).strip("._") or "download.bin"
        out = dest / f"{int(time.time() * 1000)}-{safe}"
        try:
            download.save_as(str(out))
        except Exception as exc:  # noqa: BLE001 - save errors are opaque
            raise BrowserError(f"could not save download to {out}: {exc}") from exc
        if not out.is_file() or out.stat().st_size == 0:
            raise BrowserError(f"download from {self.url} produced no file")
        return {"path": str(out), "url": self.url,
                "suggested_filename": suggested, "tab_id": self.tab_id}

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
        #: per-session proxy URL (in-memory only — never persisted to
        #: sessions.json, since URLs can carry credentials).
        self._session_proxies: dict[str, str] = {}
        #: proxy pool (duck-typed — see attach_proxy_pool). The pool lives
        #: at L5 (nomorals/connectors), which L4 must not import, so the
        #: service talks to any object with a conforming rotate().
        self._proxy_pool: Any = None
        #: download registry: id -> record (persisted to downloads.json).
        self._downloads: dict[str, dict[str, Any]] = {}
        self._load_downloads()

    # -- proxies -----------------------------------------------------------------
    def attach_proxy_pool(self, pool: Any) -> dict[str, Any]:
        """Attach a proxy pool for traffic rotation.

        ``pool`` is duck-typed (the pool connector lives at L5, which this
        L4 service must not import): any object whose ``rotate()`` returns
        a mapping carrying the proxy's connection URL under
        ``"url_with_auth"`` (exactly what
        ``nomorals.connectors.proxypool.ProxyPoolConnector.rotate()``
        returns). The pool is live-probed once at attach time — a dead
        pool fails here, not on the first download.
        """
        rotate = getattr(pool, "rotate", None)
        if not callable(rotate):
            raise BrowserError(
                "proxy pool needs a rotate() method returning "
                "{'url_with_auth': ...}")
        try:
            probe = rotate()
        except Exception as exc:
            raise BrowserError(f"proxy pool rotate() failed: {exc}") from exc
        if not isinstance(probe, dict) or not probe.get("url_with_auth"):
            raise BrowserError(
                "proxy pool rotate() must return a mapping with 'url_with_auth'")
        self._proxy_pool = pool
        _emit("browser.proxy.attached", {"pool": type(pool).__name__})
        _log.info("browser service: proxy pool attached (%s)",
                  type(pool).__name__)
        return {"attached": True, "pool": type(pool).__name__}

    def detach_proxy_pool(self) -> None:
        """Detach the pool and clear every session/tab proxy (direct traffic)."""
        self._proxy_pool = None
        self._session_proxies = {}
        for handle in self._sessions.values():
            for tab in handle._tabs.values():
                tab.session.set_proxy("")
        _emit("browser.proxy.detached", {})

    def proxy_pool_attached(self) -> bool:
        return self._proxy_pool is not None

    def rotate_proxy(self, session_name: str = "") -> dict[str, Any]:
        """Pull the next healthy proxy from the pool and apply it.

        With ``session_name`` given, only that session's tabs (and future
        tabs — new tabs inherit the session proxy); empty applies to every
        open session. Returns the masked proxy URL and what it covered.
        """
        if self._proxy_pool is None:
            raise BrowserError(
                "no proxy pool attached — attach_proxy_pool() first")
        try:
            details = self._proxy_pool.rotate()
        except Exception as exc:
            raise BrowserError(f"proxy pool rotate() failed: {exc}") from exc
        proxy_url = (details or {}).get("url_with_auth", "") \
            if isinstance(details, dict) else ""
        if not proxy_url:
            raise BrowserError(
                "proxy pool rotate() returned no url_with_auth")
        names = [session_name] if session_name else list(self._sessions)
        applied = 0
        for name in names:
            handle = self._sessions.get(name)
            if handle is None:
                raise BrowserError(f"unknown session {name!r}")
            self._session_proxies[name] = proxy_url
            for tab in handle._tabs.values():
                tab.session.set_proxy(proxy_url)
                applied += 1
        _emit("browser.proxy.rotated", {
            "session": session_name or "*",
            "proxy": _mask_proxy(proxy_url),
            "tabs": applied,
        })
        return {"proxy": _mask_proxy(proxy_url), "sessions": names,
                "tabs": applied}

    def set_session_proxy(self, session_name: str,
                          proxy_url: str = "") -> dict[str, Any]:
        """Pin a session to an explicit proxy URL ("" = direct traffic)."""
        handle = self.get_session(session_name)  # fail fast: unknown session
        proxy_url = (proxy_url or "").strip()
        self._session_proxies[session_name] = proxy_url
        for tab in handle._tabs.values():
            tab.session.set_proxy(proxy_url)
        _emit("browser.proxy.set", {
            "session": session_name,
            "proxy": _mask_proxy(proxy_url) if proxy_url else "direct",
        })
        return {"session": session_name,
                "proxy": _mask_proxy(proxy_url) if proxy_url else "direct"}

    def session_proxy(self, session_name: str) -> str:
        """The proxy URL configured for a session ("" = direct). Masked? No —
        this is the real URL; callers that display it must mask it."""
        return self._session_proxies.get(session_name, "")

    # -- sessions --------------------------------------------------------------
    def _cookie_dir(self, name: str) -> str:
        return str(self.data_dir / "cookies" / name)

    def _make_tab(self, handle: SessionHandle, tab_id: str) -> Tab:
        session = BrowserSession(
            name=f"{handle.name}:{tab_id}",
            session_dir=self._cookie_dir(handle.name),
        )
        # new tabs inherit the session's proxy (rotation applies live)
        proxy = self._session_proxies.get(handle.name, "")
        if proxy:
            session.set_proxy(proxy)
        return Tab(tab_id=tab_id, session_name=handle.name, session=session)

    def open_session(self, name: str) -> SessionHandle:
        name = (name or "").strip()
        if not name:
            raise BrowserError("session name must not be empty")
        if name in self._sessions:
            raise BrowserError(f"session {name!r} is already open")
        handle = SessionHandle(name, self)
        self._sessions[name] = handle
        _emit("browser.session.opened", {"session": name})
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
        _emit("browser.session.closed", {"session": name})

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

    def open_rendered_tab(self, session_name: str, url: str = "",
                          proxy: str = "") -> RenderedTab:
        """Open a playwright-backed tab in ``session_name``'s cookie space.

        For JavaScript/Cloudflare-guarded pages that plain-HTTP tabs cannot
        pass. Cookies persist via playwright storage_state in the session's
        cookie dir. ``proxy`` overrides the session's proxy for this tab
        ("" = inherit the session proxy, which may itself be direct).
        Fail fast: raises BrowserError when playwright or its chromium
        build is missing, or when the initial navigate fails.
        """
        session_name = (session_name or "").strip()
        if not session_name:
            raise BrowserError("session name must not be empty")
        playwright = self._driver()  # fail fast before touching anything
        tab_id = ulid_now()
        tab_proxy = (proxy or "").strip() or self._session_proxies.get(session_name, "")
        tab = RenderedTab(
            tab_id=tab_id,
            session_name=session_name,
            storage_state_path=self._rendered_state_path(session_name),
            playwright=playwright,
            proxy=tab_proxy,
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
    def _load_downloads(self) -> None:
        """Load the download registry. A corrupt registry file resets with
        a warning — it must never kill the service."""
        path = self.data_dir / _DOWNLOADS_FILE
        if not path.is_file():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = payload.get("downloads") or {}
            self._downloads = {
                str(did): dict(rec) for did, rec in records.items()
                if isinstance(rec, dict)}
        except (OSError, ValueError) as exc:
            _log.warning("download registry %s unreadable (%s) — starting empty",
                         path, exc)
            self._downloads = {}

    def _save_downloads(self) -> None:
        path = self.data_dir / _DOWNLOADS_FILE
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"version": _SAVE_VERSION, "downloads": self._downloads},
                      fh, indent=2)
        os.replace(tmp, path)

    def _record_download(self, record: dict[str, Any]) -> dict[str, Any]:
        self._downloads[record["id"]] = record
        try:
            self._save_downloads()
        except OSError as exc:  # noqa: BLE001 - registry is bookkeeping
            _log.warning("could not persist download registry: %s", exc)
        return record

    def _finish_download(self, download_id: str, *,
                         status: str, path: str = "", size: int = 0,
                         mime: str = "", error: str = "") -> dict[str, Any]:
        rec = self._downloads.get(download_id)
        if rec is None:  # pragma: no cover - defensive
            raise BrowserError(f"unknown download {download_id!r}")
        rec.update({
            "status": status,
            "path": path or rec.get("path", ""),
            "size": size,
            "mime": mime or rec.get("mime", ""),
            "category": _mime_category(mime or rec.get("mime", "")),
            "finished_at": time.time(),
            "error": error,
        })
        return self._record_download(rec)

    def download(
        self,
        tab_or_url: Tab | str,
        url: str = "",
        *,
        filename: str = "",
        organize: bool = False,
    ) -> DownloadResult:
        """Download ``url`` reusing the owning tab's cookies.

        ``tab_or_url`` is a :class:`Tab`, a tab id string (resolved across
        sessions), or — when no tab is involved — the URL itself, in which
        case ``url`` may be omitted. With ``organize=True`` the file lands
        in a MIME-category subfolder (images/, videos/, audio/,
        documents/, other/) under the session's download dir; the default
        keeps the flat ``downloads/<session>/`` layout.

        Every download is registered in the download registry (see
        :meth:`list_downloads` / :meth:`wait_for_download`).
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

        download_id = ulid_now()
        self._record_download({
            "id": download_id,
            "session": session_name,
            "tab_id": tab.tab_id if tab is not None else "",
            "rendered_tab_id": "",
            "url": url,
            "path": "",
            "size": 0,
            "mime": "",
            "category": "",
            "status": "in_progress",
            "started_at": time.time(),
            "finished_at": 0.0,
            "error": "",
        })

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
        except BrowserError as exc:
            self._finish_download(download_id, status="failed", error=str(exc))
            raise
        except urllib.error.HTTPError as exc:
            self._finish_download(
                download_id, status="failed",
                error=f"download {url} failed: HTTP {exc.code}")
            raise BrowserError(f"download {url} failed: HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            self._finish_download(
                download_id, status="failed",
                error=f"download {url} failed: {exc.reason}")
            raise BrowserError(f"download {url} failed: {exc.reason}") from exc

        if organize:
            category_dir = dest_dir / _mime_category(mime)
            category_dir.mkdir(parents=True, exist_ok=True)
            path = category_dir / name
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
        rec = self._finish_download(
            download_id, status="completed", path=str(path),
            size=len(data), mime=mime)
        rec["artifact_uri"] = artifact_uri
        self._record_download(rec)
        _emit("browser.download.completed", {
            "url": url,
            "path": str(path),
            "size": len(data),
            "mime": mime,
            "artifact_uri": artifact_uri,
            "session": session_name,
            "tab_id": tab.tab_id if tab is not None else "",
            "download_id": download_id,
            "mission_id": self._mission_id,
        })
        return DownloadResult(path=str(path), size=len(data), mime=mime,
                              artifact_uri=artifact_uri)

    def list_downloads(self, session_name: str = "",
                       category: str = "", limit: int = 100) -> list[dict[str, Any]]:
        """The download registry, newest first. Filter by session and/or
        MIME category (images|videos|audio|documents|other)."""
        session_name = (session_name or "").strip()
        category = (category or "").strip().lower()
        recs = sorted(self._downloads.values(),
                      key=lambda r: r.get("started_at", 0.0), reverse=True)
        out = []
        for rec in recs:
            if session_name and rec.get("session") != session_name:
                continue
            if category and rec.get("category") != category:
                continue
            out.append(dict(rec))
            if len(out) >= max(1, limit):
                break
        return out

    def wait_for_download(self, download_id: str,
                          timeout: float = 60.0) -> dict[str, Any]:
        """Wait for a download to finish. Completed/failed records return
        immediately; an in-progress record is polled until its file stops
        growing (2s stable) or the timeout hits — then the record is
        finalized from disk. Fail fast on unknown ids."""
        rec = self._downloads.get(download_id)
        if rec is None:
            raise BrowserError(f"unknown download {download_id!r}")
        if rec.get("status") in {"completed", "failed"}:
            return dict(rec)
        deadline = time.time() + max(1.0, float(timeout))
        last_size = -1
        stable_since = time.time()
        while time.time() < deadline:
            rec = self._downloads.get(download_id) or rec
            if rec.get("status") in {"completed", "failed"}:
                return dict(rec)
            try:
                size = os.path.getsize(rec.get("path") or "")
            except OSError:
                size = -1
            now = time.time()
            if size >= 0 and size == last_size:
                if now - stable_since >= 2.0:
                    break  # file stopped growing — treat as done
            else:
                last_size = size
                stable_since = now
            time.sleep(0.5)
        rec = self._downloads.get(download_id) or rec
        if rec.get("status") == "in_progress":
            size = -1
            try:
                size = os.path.getsize(rec.get("path") or "")
            except OSError:
                size = -1  # file vanished mid-download; mark failed below
            if size and size > 0:
                rec = self._finish_download(
                    download_id, status="completed", size=size)
            else:
                rec = self._finish_download(
                    download_id, status="failed",
                    error="download did not complete within the wait")
        return dict(rec)

    def rendered_tab_download(self, tab_id: str, target: str) -> dict[str, Any]:
        """Click ``target`` in a rendered tab and capture the download it
        triggers (for JS-driven downloads plain HTTP can't see). The
        download is registered like any other."""
        tab = self._rendered_tabs.get(tab_id)
        if tab is None:
            raise BrowserError(f"unknown rendered tab {tab_id!r}")
        dest_dir = self.data_dir / "downloads" / tab.session_name
        download_id = ulid_now()
        self._record_download({
            "id": download_id,
            "session": tab.session_name,
            "tab_id": "",
            "rendered_tab_id": tab_id,
            "url": tab.url,
            "path": "",
            "size": 0,
            "mime": "",
            "category": "",
            "status": "in_progress",
            "started_at": time.time(),
            "finished_at": 0.0,
            "error": "",
        })
        try:
            result = tab.trigger_download(target, dest_dir)
        except BrowserError as exc:
            self._finish_download(download_id, status="failed", error=str(exc))
            raise
        path = result["path"]
        mime = (mimetypes.guess_type(path)[0] or "application/octet-stream")
        rec = self._finish_download(
            download_id, status="completed", path=path,
            size=os.path.getsize(path), mime=mime)
        _emit("browser.download.completed", {
            "url": tab.url, "path": path, "size": rec["size"], "mime": mime,
            "session": tab.session_name, "rendered_tab_id": tab_id,
            "download_id": download_id, "mission_id": self._mission_id,
        })
        return rec

    def screenshot_rendered(self, tab_id: str, *,
                            full_page: bool = False) -> ScreenshotResult:
        """Screenshot a rendered tab's LIVE page — logged-in state and JS
        mutations included (unlike :meth:`screenshot`, which reloads the
        URL in a fresh context)."""
        tab = self._rendered_tabs.get(tab_id)
        if tab is None:
            raise BrowserError(f"unknown rendered tab {tab_id!r}")
        shot = tab.screenshot(full_page=full_page)
        path = Path(shot["path"])
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

    # -- cookies: inspect / export / import --------------------------------------
    def _session_jar(self, session_name: str) -> http.cookiejar.CookieJar:
        """All tabs' cookies of a session merged into one jar (deduped by
        name/domain/path)."""
        handle = self.get_session(session_name)  # fail fast: unknown session
        jar = http.cookiejar.CookieJar()
        seen: set[tuple[str, str, str]] = set()
        for tab in handle._tabs.values():
            for cookie in tab.session.cookie_jar:
                key = (cookie.name, cookie.domain, cookie.path)
                if key in seen:
                    continue
                seen.add(key)
                jar.set_cookie(cookie)
        return jar

    def export_cookies(self, session_name: str,
                       path: str | os.PathLike[str],
                       format: str = "netscape") -> dict[str, Any]:
        """Export a session's cookies to ``path``.

        ``format="netscape"`` writes the classic Mozilla cookies.txt that
        curl, wget, and yt-dlp read; ``format="json"`` writes a plain list
        of cookie dicts. An empty jar still writes a valid (empty) file —
        0 cookies is an honest result, not an error.
        """
        fmt = (format or "").strip().lower()
        if fmt not in {"netscape", "json"}:
            raise BrowserError(
                f"unknown cookie format {format!r}: netscape|json")
        jar = self._session_jar(session_name)
        out = Path(path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        count = len(jar)
        try:
            if fmt == "netscape":
                mcj = http.cookiejar.MozillaCookieJar()
                for cookie in jar:
                    mcj.set_cookie(cookie)
                mcj.save(str(out), ignore_discard=True, ignore_expires=True)
            else:
                payload = [{
                    "name": c.name, "value": c.value, "domain": c.domain,
                    "path": c.path, "secure": bool(c.secure),
                    "expires": c.expires,
                    "http_only": bool(c.has_nonstandard_attr("HttpOnly")),
                } for c in jar]
                tmp = out.with_suffix(out.suffix + ".tmp")
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=2)
                os.replace(tmp, out)
        except OSError as exc:
            raise BrowserError(f"cookie export to {out} failed: {exc}") from exc
        _emit("browser.cookies.exported", {
            "session": session_name, "path": str(out), "format": fmt,
            "cookies": count})
        return {"path": str(out), "format": fmt, "cookies": count,
                "session": session_name}

    def import_cookies(self, session_name: str,
                       path: str | os.PathLike[str],
                       format: str = "netscape") -> dict[str, Any]:
        """Import cookies into every tab of a session (netscape or json, as
        written by :meth:`export_cookies`). The tabs' cookie persistence
        is flushed so the import survives restarts."""
        fmt = (format or "").strip().lower()
        if fmt not in {"netscape", "json"}:
            raise BrowserError(
                f"unknown cookie format {format!r}: netscape|json")
        handle = self.get_session(session_name)  # fail fast: unknown session
        src = Path(path).expanduser()
        if not src.is_file():
            raise BrowserError(f"cookie file not found: {src}")
        try:
            if fmt == "netscape":
                mcj = http.cookiejar.MozillaCookieJar(str(src))
                mcj.load(ignore_discard=True, ignore_expires=True)
                cookies = list(mcj)
            else:
                payload = json.loads(src.read_text(encoding="utf-8"))
                if not isinstance(payload, list):
                    raise BrowserError(
                        f"cookie file {src} is not a JSON list")
                cookies = [_cookie_from_dict(entry) for entry in payload]
        except BrowserError:
            raise
        except (OSError, ValueError,
                http.cookiejar.LoadError) as exc:
            raise BrowserError(f"cannot read cookies from {src}: {exc}") from exc
        if not handle._tabs:
            raise BrowserError(
                f"session {session_name!r} has no tabs — open a tab first")
        for tab in handle._tabs.values():
            for cookie in cookies:
                tab.session.cookie_jar.set_cookie(cookie)
            tab.session._save_cookies()
        _emit("browser.cookies.imported", {
            "session": session_name, "path": str(src), "format": fmt,
            "cookies": len(cookies)})
        return {"session": session_name, "imported": len(cookies),
                "tabs": len(handle._tabs), "format": fmt}

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


#: Cookie constructor fields (the same set tools/browser persists).
_COOKIE_FIELDS = {
    "version", "name", "value", "port", "port_specified",
    "domain", "domain_specified", "domain_initial_dot",
    "path", "path_specified", "secure", "expires", "discard",
    "comment", "comment_url", "rest", "rfc2109",
}

_COOKIE_DEFAULTS: dict[str, Any] = {
    "version": 0,
    "port": None,
    "port_specified": False,
    "domain_specified": False,
    "domain_initial_dot": False,
    "path": "/",
    "path_specified": True,
    "secure": False,
    "expires": None,
    "discard": True,
    "comment": None,
    "comment_url": None,
    "rest": {},
    "rfc2109": False,
}


def _cookie_from_dict(entry: dict[str, Any]) -> http.cookiejar.Cookie:
    """Build a Cookie from an exported dict. Fail fast on malformed entries
    — a half-formed cookie is never silently skipped into the jar."""
    if not isinstance(entry, dict):
        raise BrowserError(f"bad cookie entry: {entry!r}"[:160])
    cookie = http.cookiejar.Cookie.__new__(http.cookiejar.Cookie)
    for key, value in entry.items():
        if key in _COOKIE_FIELDS:
            setattr(cookie, key, value)
    for key, value in _COOKIE_DEFAULTS.items():
        if not hasattr(cookie, key):
            setattr(cookie, key, value)
    if not getattr(cookie, "name", "") or not getattr(cookie, "domain", ""):
        raise BrowserError(
            f"cookie entry missing name/domain: {entry!r}"[:160])
    # Cookie.__init__ normally derives _rest from rest; replicate it so
    # has_nonstandard_attr() works on imported cookies.
    cookie._rest = dict(getattr(cookie, "rest", None) or {})
    if getattr(cookie, "http_only", False):
        cookie._rest.setdefault("HttpOnly", "")
    return cookie

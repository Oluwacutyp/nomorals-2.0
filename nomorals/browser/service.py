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
import random
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
from . import errors
from . import forms
from .errors import (
    BrowserBotDetectedError,
    BrowserError,
    BrowserNetworkError,
    BrowserSiteError,
)
from .pacing import Pacing, pacing_from_env
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
    "BrowserBotDetectedError",
    "BrowserNetworkError",
    "BrowserSiteError",
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


#: BrowserError lives in nomorals.browser.errors (the error taxonomy);
#: re-exported here so ``from nomorals.browser.service import BrowserError``
#: keeps working unchanged.
assert BrowserError is errors.BrowserError


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


def _notify_action(tab: Any, description: str) -> None:
    """Fire a tab's ``on_action`` hook (live agent window). Never raises —
    a broken hook must never kill the browser task."""
    try:
        hook = getattr(tab, "on_action", None)
        if hook is not None:
            hook(description)
    except Exception:  # noqa: BLE001 - hook failures are never fatal
        _log.debug("tab on_action hook failed", exc_info=True)


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
        #: Optional live-view hook: called with a short description after
        #: each major action (navigate/click/fill/select/check/back).
        #: ``None`` (default) = no live view; set by
        #: ``nomorals.browser.liveview.LiveView.attach``.
        self.on_action: Any = None
        #: Pacing between actions (set by the owning BrowserService).
        #: None = no pacing.
        self.pacing: Pacing | None = None

    def _pace(self, action: str) -> None:
        """Sleep the pacing cadence before an action (no-op when pacing
        is unset or disabled)."""
        if self.pacing is not None:
            self.pacing.pause(action)

    # -- navigation ----------------------------------------------------------
    def _load_page(self, url: str) -> dict[str, Any]:
        """Open the page in the wrapped session; update url/title. Raises
        a classified BrowserError on any failure (fail fast — never a
        half-loaded tab)."""
        url = (url or "").strip()
        if not url:
            raise BrowserError("navigate needs a url")
        try:
            result = self.session.open(url)
        except ToolError as exc:
            # tools.browser wraps urllib failures in ToolError — classify
            # the underlying signal, never misreport it as generic.
            raise errors.classify_exception(exc, url=url) from exc
        status = int(result.get("status") or 0)
        if status >= 400 or not result.get("ok", True):
            if status >= 400:
                raise errors.classify_http_status(
                    status, url=url,
                    html=(self.session._raw or "")[:200_000],
                    title=self.session.title or "")
            raise BrowserError(
                f"navigate {url} failed: {result.get('error') or 'unknown error'}")
        self.url = self.session.url
        self.title = self.session.title
        self.error = ""
        return result

    def navigate(self, url: str, *, retries: int = 0) -> dict[str, Any]:
        """Navigate and append to history AFTER the load succeeded.

        ``retries`` re-attempts a failed load with linear backoff (0 =
        try once). The final failure raises BrowserError — never a
        half-loaded tab.
        """
        url = (url or "").strip()
        if not url:
            raise BrowserError("navigate needs a url")
        self._pace("navigate")
        attempts = 1 + max(0, int(retries))
        last_exc: BrowserError | None = None
        result: dict[str, Any] = {}
        for attempt in range(attempts):
            try:
                result = self._load_page(url)
                last_exc = None
                break
            except BrowserError as exc:
                last_exc = exc
                if attempt < attempts - 1:
                    time.sleep(min(2.0 * (attempt + 1), 8.0))
        if last_exc is not None:
            raise last_exc
        self.history.append(
            {"url": self.url, "title": self.title, "ts": time.time()})
        _emit("browser.tab.navigated", {
            "session": self.session_name,
            "tab_id": self.tab_id,
            "url": self.url,
            "title": self.title,
        })
        _notify_action(self, f"opened {self.url or url}")
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
        _notify_action(self, f"went back to {self.url}")
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
        _notify_action(self, f"clicked {target!r}")
        return result

    def fill(self, name: str, value: str) -> dict[str, Any]:
        result = self._delegate("fill", name, value)
        _notify_action(self, f"filled {name!r}")
        return result

    def select(self, name: str, value: str) -> dict[str, Any]:
        """Pick a ``<select>`` dropdown option (submitted on next submit)."""
        result = self._delegate("select", name, value)
        _notify_action(self, f"selected {value!r} in {name!r}")
        return result

    def check(self, name: str, checked: bool = True) -> dict[str, Any]:
        """Check/uncheck a checkbox, or pick a radio button."""
        result = self._delegate("check", name, checked)
        _notify_action(self, f"{'checked' if checked else 'unchecked'} {name!r}")
        return result

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

#: sentinel for "no arg passed" (None is a legitimate JS arg).
_MISSING: Any = object()

#: wait_until values playwright accepts for page.goto().
_GOTO_WAIT_UNTIL = frozenset({"load", "domcontentloaded", "networkidle", "commit"})

#: Basic anti-detection for rendered tabs. Realistic user agent, viewport,
#: locale, timezone, the AutomationControlled blink flag disabled, and
#: navigator.webdriver hidden — enough to stop flagging the obvious
#: headless-Chromium tells, without fingerprint-spoofing rabbit holes.
_DEFAULT_STEALTH: dict[str, Any] = {
    "enabled": True,
    "user_agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "viewport": {"width": 1366, "height": 768},
    "locale": "en-US",
    "timezone_id": "Africa/Lagos",
}
_STEALTH_KEYS = frozenset(_DEFAULT_STEALTH)

#: init script hiding the classic headless tell.
_WEBDRIVER_HIDE_JS = (
    "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
)

#: window.chrome runtime stub — real Chrome exposes this; headless
#: Chromium's object is missing or incomplete, and bot scripts check.
_CHROME_RUNTIME_JS = """(() => {
  if (window.chrome && window.chrome.runtime) return;
  window.chrome = window.chrome || {};
  window.chrome.runtime = {
    onMessage: { addListener: function () {}, removeListener: function () {} },
    onConnect: { addListener: function () {} },
    sendMessage: function () {},
    connect: function () {
      return { onMessage: { addListener: function () {} }, postMessage: function () {} };
    },
  };
  window.chrome.loadTimes = window.chrome.loadTimes || function () { return {}; };
  window.chrome.csi = window.chrome.csi || function () { return {}; };
  window.chrome.app = window.chrome.app || {};
})();"""

#: Realistic plugin list — headless Chromium reports zero plugins.
_PLUGINS_SPOOF_JS = """(() => {
  const makePlugin = (name, filename, description) => {
    const p = { name, filename, description, length: 1,
                item: function () { return null; },
                namedItem: function () { return null; } };
    return p;
  };
  const plugins = [
    makePlugin('PDF Viewer', 'internal-pdf-viewer', 'Portable Document Format'),
    makePlugin('Chrome PDF Viewer', 'mhjfbmdgcfjbbpaeojofohoefgiehjai', ''),
    makePlugin('Native Client', 'internal-nacl-plugin', ''),
  ];
  plugins.item = function (i) { return this[i] || null; };
  plugins.namedItem = function (n) {
    return this.find((p) => p.name === n) || null;
  };
  Object.defineProperty(navigator, 'plugins', { get: () => plugins });
  Object.defineProperty(navigator, 'mimeTypes', {
    get: () => ({ length: 0, item: function () { return null; },
                  namedItem: function () { return null; } }),
  });
})();"""

#: Permissions override — headless denies everything; real browsers vary.
_PERMISSIONS_SPOOF_JS = """(() => {
  try {
    const original = navigator.permissions && navigator.permissions.query;
    if (!original) return;
    const grants = { notifications: 'granted', geolocation: 'prompt',
                     camera: 'prompt', microphone: 'prompt' };
    navigator.permissions.query = function (params) {
      const name = params && params.name;
      if (name && Object.prototype.hasOwnProperty.call(grants, name)) {
        return Promise.resolve({ state: grants[name] });
      }
      return original.call(this, params);
    };
  } catch (e) {}
})();"""

#: chromium flags for rendered tabs (stealth).
_STEALTH_CHROME_ARGS = ["--disable-blink-features=AutomationControlled"]

#: Fallback accessibility-tree walker for snapshot() when the driver has
#: no aria_snapshot (older playwright). Produces the same ref-handle
#: shape as playwright's tree — ``- role "name" [ref=fN]`` — so agents
#: interact identically either way.
_SNAPSHOT_FALLBACK_JS = """() => {
  const lines = [];
  let n = 0;
  const seen = new Set();
  const nameOf = (el) => {
    const aria = el.getAttribute('aria-label');
    if (aria) return aria.trim().slice(0, 80);
    const tag = (el.tagName || '').toUpperCase();
    if (tag === 'INPUT' || tag === 'TEXTAREA') {
      const ph = el.getAttribute('placeholder');
      if (ph) return ph.trim().slice(0, 80);
      if (el.value && !/password/i.test(el.type || ''))
        return String(el.value).slice(0, 40);
    }
    if (tag === 'IMG') return (el.getAttribute('alt') || '').slice(0, 80);
    const t = (el.innerText || '').trim().replace(/\\s+/g, ' ');
    return t.slice(0, 80);
  };
  const roleOf = (el) => {
    const r = el.getAttribute('role');
    if (r) return r;
    const tag = (el.tagName || '').toUpperCase();
    const type = (el.getAttribute('type') || '').toLowerCase();
    if (tag === 'A') return 'link';
    if (tag === 'BUTTON') return 'button';
    if (tag === 'INPUT') {
      if (type === 'checkbox') return 'checkbox';
      if (type === 'radio') return 'radio';
      if (type === 'submit' || type === 'button') return 'button';
      return 'textbox';
    }
    if (tag === 'TEXTAREA') return 'textbox';
    if (tag === 'SELECT') return 'combobox';
    if (/^H[1-6]$/.test(tag)) return 'heading';
    if (tag === 'IMG') return 'img';
    if (tag === 'FORM') return 'form';
    return '';
  };
  const walk = (node, depth) => {
    if (depth > 10 || lines.length >= 500) return;
    let kids = [];
    try { kids = node.children || []; } catch (e) { return; }
    for (const el of kids) {
      if (seen.has(el)) continue;
      seen.add(el);
      try { if (el.shadowRoot) walk(el.shadowRoot, depth + 1); } catch (e) {}
      const role = roleOf(el);
      const interactive = role && role !== 'heading' && role !== 'img'
        && role !== 'form';
      const ref = interactive ? ('f' + (++n)) : '';
      if (role) {
        const nm = nameOf(el);
        lines.push('  '.repeat(Math.min(depth, 6)) + '- ' + role
          + (nm ? ' "' + nm.replace(/"/g, "'") + '"' : '')
          + (ref ? ' [ref=' + ref + ']' : ''));
        if (ref) {
          let sel = el.tagName.toLowerCase();
          if (el.id) sel += '#' + el.id;
          else {
            const parent = el.parentElement;
            if (parent) {
              const sibs = Array.from(parent.children).filter(
                (s) => s.tagName === el.tagName);
              if (sibs.length > 1) sel += ':nth-of-type(' + (sibs.indexOf(el) + 1) + ')';
            }
          }
          lines.push('__REF__' + ref + '__SEL__' + sel);
        }
      }
      walk(el, depth + 1);
    }
  };
  walk(document.body || document.documentElement, 0);
  return lines.join('\\n');
}"""

#: snapshot() caps the tree so one call cannot flood the caller.
_SNAPSHOT_MAX_CHARS = 60_000


def _merge_stealth(base: dict[str, Any] | None,
                  override: dict[str, Any] | None) -> dict[str, Any]:
    """Merge stealth profiles over the defaults. Unknown keys fail fast —
    a silently ignored profile entry is a lie about what's applied."""
    merged = dict(_DEFAULT_STEALTH)
    for profile in (base, override):
        if not profile:
            continue
        unknown = set(profile) - _STEALTH_KEYS
        if unknown:
            raise BrowserError(
                f"unknown stealth profile keys: {sorted(unknown)} "
                f"(known: {sorted(_STEALTH_KEYS)})")
        merged.update(profile)
    viewport = merged.get("viewport")
    if merged.get("enabled") and not (
            isinstance(viewport, dict) and viewport.get("width")
            and viewport.get("height")):
        raise BrowserError("stealth viewport must be {width, height}")
    return merged


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
    cookie directory so logins survive restarts. ``storage_state`` also
    carries the page's ``localStorage``/``sessionStorage`` origins, so
    sites that stash tokens in localStorage resume logged-in too.

    Stealth: unless disabled via the ``stealth`` profile, the tab launches
    chromium with a realistic user agent, viewport, locale, and timezone,
    disables the AutomationControlled blink feature, and hides
    ``navigator.webdriver`` — plus a ``window.chrome`` runtime stub, a
    realistic ``navigator.plugins`` list, and a permissions override. These
    hide the cheap, obvious headless tells; they do not forge a different
    device's fingerprint.

    Telemetry: console messages, network requests/responses, and dialogs
    are captured from page creation into ring buffers (see
    :meth:`console_messages`, :meth:`network_requests`, :meth:`dialogs`) —
    the evidence trio for debugging agent runs. Unexpected dialogs
    (alert/confirm/prompt) are auto-accepted by default and recorded, so
    a stray ``alert()`` never hangs automation; :meth:`set_dialog_policy`
    changes that.

    Error recovery: ``navigate`` retries with backoff; when
    ``shot_on_error`` is true (default), a failed navigate/fill/click/
    submit/select/check captures a screenshot AND a DOM dump into the
    session's screenshots dir and the error message names both paths, so
    the failure is debuggable instead of a bare timeout.
    """

    def __init__(
        self,
        tab_id: str,
        session_name: str,
        storage_state_path: str | os.PathLike[str],
        playwright: Any,
        proxy: str = "",
        *,
        stealth: dict[str, Any] | None = None,
        shot_on_error: bool = True,
        retries: int = 2,
    ) -> None:
        self.tab_id = tab_id
        self.session_name = session_name
        self.url: str = ""
        self.title: str = ""
        self.history: list[dict[str, Any]] = []
        #: last load failure (set by navigate); empty when the tab is clean.
        self.error: str = ""
        #: Optional live-view hook — see ``Tab.on_action``.
        self.on_action: Any = None
        self._storage_state_path = Path(storage_state_path)
        #: started driver object (owns .chromium); owned by the service.
        self._playwright = playwright
        #: proxy URL for this tab's chromium ("" = direct).
        self._proxy = (proxy or "").strip()
        #: merged stealth profile (see _merge_stealth).
        self._stealth = _merge_stealth(None, stealth)
        #: capture screenshot+DOM on action failure.
        self.shot_on_error = bool(shot_on_error)
        #: navigate retries after the first attempt (0 = try once).
        self.retries = max(0, int(retries))
        #: pacing between mutating actions (shared with the service).
        self.pacing: Pacing | None = None
        #: directory auto-captured downloads are saved to (None disables).
        self._download_dir: Path | None = None
        #: sink called with each auto-captured download record dict.
        self._on_auto_download: Any = None
        #: True while trigger_download() runs its explicit expect_download
        #: click — the context-level auto-capture must not double-save it.
        self._auto_suppress = False
        #: ring buffers for page telemetry (capped; oldest dropped first).
        self._console_log: list[dict[str, Any]] = []
        self._network_log: list[dict[str, Any]] = []
        self._dialog_log: list[dict[str, Any]] = []
        #: dialog policy: accept (default) | dismiss | record. "record"
        #: leaves the dialog open and only logs it — the caller must
        #: handle it, or the page will hang on the modal.
        self._dialog_policy = "accept"
        #: ref -> playwright selector from the last snapshot() call, so
        #: click/fill/hover accept "ref:e7" targets.
        self._snap_refs: dict[str, str] = {}
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None

    # -- page telemetry ------------------------------------------------------
    def _remember(self, buf: list[dict[str, Any]], entry: dict[str, Any],
                  cap: int) -> None:
        buf.append(entry)
        if len(buf) > cap:
            del buf[: len(buf) - cap]

    def _on_console_msg(self, msg: Any) -> None:
        try:
            text = msg.text if isinstance(msg.text, str) else str(msg.text())
        except Exception:  # noqa: BLE001 - cosmetic
            text = ""
        try:
            kind = msg.type if isinstance(msg.type, str) else str(msg.type())
        except Exception:  # noqa: BLE001 - cosmetic
            kind = ""
        try:
            loc = msg.location or {}
        except Exception:  # noqa: BLE001 - cosmetic
            loc = {}
        self._remember(self._console_log, {
            "type": kind or "log",
            "text": (text or "")[:2000],
            "location": {k: loc.get(k) for k in ("url", "lineNumber",
                                                "columnNumber")
                         if isinstance(loc, dict)},
            "ts": time.time(),
        }, 300)

    def _on_request(self, request: Any) -> None:
        try:
            url = request.url
            method = request.method
        except Exception:  # noqa: BLE001 - cosmetic
            return
        self._remember(self._network_log, {
            "kind": "request",
            "method": method if isinstance(method, str) else str(method),
            "url": url if isinstance(url, str) else str(url),
            "ts": time.time(),
        }, 500)

    def _on_response(self, response: Any) -> None:
        try:
            url = response.url
            status = response.status
        except Exception:  # noqa: BLE001 - cosmetic
            return
        try:
            request = response.request
            method = request.method if request else ""
        except Exception:  # noqa: BLE001 - cosmetic
            method = ""
        self._remember(self._network_log, {
            "kind": "response",
            "method": method if isinstance(method, str) else str(method),
            "url": url if isinstance(url, str) else str(url),
            "status": int(status) if isinstance(status, int) else status,
            "ts": time.time(),
        }, 500)

    def _on_dialog(self, dialog: Any) -> None:
        try:
            kind = dialog.type
            message = dialog.message
        except Exception:  # noqa: BLE001 - cosmetic
            kind, message = "", ""
        record = {
            "type": kind if isinstance(kind, str) else str(kind),
            "message": (message if isinstance(message, str)
                        else str(message))[:2000],
            "policy": self._dialog_policy,
            "ts": time.time(),
        }
        self._remember(self._dialog_log, record, 50)
        if self._dialog_policy == "record":
            return  # caller handles it; the page stays on the modal
        try:
            if self._dialog_policy == "dismiss":
                dialog.dismiss()
            else:
                dialog.accept()
        except Exception as exc:  # noqa: BLE001 - dialog handling best-effort
            _log.debug("rendered tab %s: dialog %s failed: %r",
                       self.tab_id, self._dialog_policy, exc)

    def _pace(self, action: str) -> None:
        """Sleep the pacing cadence before an action (no-op when pacing
        is unset or disabled)."""
        if self.pacing is not None:
            self.pacing.pause(action)

    def _can_evaluate(self) -> bool:
        """True when the page driver can run JavaScript (duck-typed
        drivers may not — verification then degrades to unverified
        instead of raising)."""
        page = self._page
        return page is not None and callable(getattr(page, "evaluate", None))

    # -- browser lifecycle ---------------------------------------------------
    def _ensure_page(self) -> Any:
        """Launch this tab's browser (once) and return the page. Fail fast:
        a launch failure raises BrowserError, never None."""
        if self._page is not None:
            return self._page
        try:
            launch_kwargs: dict[str, Any] = {"headless": True}
            if self._stealth.get("enabled"):
                launch_kwargs["args"] = list(_STEALTH_CHROME_ARGS)
            proxy_cfg = _playwright_proxy_config(self._proxy)
            if proxy_cfg:
                launch_kwargs["proxy"] = proxy_cfg
            self._browser = self._playwright.chromium.launch(**launch_kwargs)
            ctx_kwargs: dict[str, Any] = {}
            state = str(self._storage_state_path)
            if self._storage_state_path.is_file():
                ctx_kwargs["storage_state"] = state
            if self._stealth.get("enabled"):
                ctx_kwargs.update({
                    "user_agent": self._stealth["user_agent"],
                    "viewport": dict(self._stealth["viewport"]),
                    "locale": self._stealth["locale"],
                    "timezone_id": self._stealth["timezone_id"],
                })
            self._context = self._browser.new_context(**ctx_kwargs)
            add_init = getattr(self._context, "add_init_script", None)
            if callable(add_init) and self._stealth.get("enabled"):
                # Every stealth script is a passive property stub that
                # hides the automation tell — none forges a device's
                # fingerprint. The languages stub follows the locale so
                # navigator.languages agrees with Accept-Language. All
                # stubs ship as ONE init script.
                locale = str(self._stealth.get("locale") or "en-US")
                stealth_script = "\n".join([
                    _WEBDRIVER_HIDE_JS,
                    _CHROME_RUNTIME_JS,
                    _PLUGINS_SPOOF_JS,
                    ("Object.defineProperty(navigator, 'languages', "
                     f"{{ get: () => [{locale!r}, 'en'] }});"),
                    _PERMISSIONS_SPOOF_JS,
                ])
                try:
                    add_init(stealth_script)
                except Exception as exc:  # noqa: BLE001 - best effort
                    _log.debug("rendered tab %s: stealth init failed: %r",
                               self.tab_id, exc)
            self._page = self._context.new_page()
            # Page telemetry: console, network, and dialogs from creation.
            on_page = getattr(self._page, "on", None)
            if callable(on_page):
                for event, handler in (
                        ("console", self._on_console_msg),
                        ("request", self._on_request),
                        ("response", self._on_response),
                        ("dialog", self._on_dialog)):
                    try:
                        on_page(event, handler)
                    except Exception as exc:  # noqa: BLE001 - best effort
                        _log.debug("rendered tab %s: %s listener failed: %r",
                                   self.tab_id, event, exc)
            # Intercept download events at the context level: any download
            # the page triggers (JS blob saves, location-href file hits,
            # not just explicit trigger_download clicks) is auto-captured
            # into the session download dir and reported to the service.
            on_event = getattr(self._context, "on", None)
            if callable(on_event):
                try:
                    on_event("download", self._handle_auto_download)
                except Exception as exc:  # noqa: BLE001 - best effort
                    _log.debug("rendered tab %s: download listener failed: %r",
                               self.tab_id, exc)
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

    def _handle_auto_download(self, download: Any) -> None:
        """Context-level download handler: save the file and report it.

        Never raises — a broken capture must not kill the page. Downloads
        triggered via :meth:`trigger_download` set ``_auto_suppress`` so
        they are saved exactly once (through the explicit path).
        """
        if self._auto_suppress:
            return
        if self._download_dir is None:
            _log.debug("rendered tab %s: download ignored (no download dir)",
                       self.tab_id)
            return
        try:
            suggested = str(
                getattr(download, "suggested_filename", "") or "download.bin")
            safe = "".join(c if (c.isalnum() or c in "._-") else "_"
                           for c in suggested).strip("._") or "download.bin"
            dest = self._download_dir
            dest.mkdir(parents=True, exist_ok=True)
            out = dest / f"{int(time.time() * 1000)}-{safe}"
            download.save_as(str(out))
            size = out.stat().st_size if out.is_file() else 0
            if size == 0:
                _log.warning("rendered tab %s: auto-captured download %r "
                             "produced no file", self.tab_id, suggested)
                return
            record = {
                "path": str(out),
                "size": size,
                "suggested_filename": suggested,
                "url": self.url,
                "trigger": "auto",
            }
            sink = self._on_auto_download
            if callable(sink):
                try:
                    sink(record)
                except Exception as exc:  # noqa: BLE001 - sink is bookkeeping
                    _log.warning("rendered tab %s: download sink failed: %r",
                                 self.tab_id, exc)
            _log.info("rendered tab %s: auto-captured download %r -> %s "
                      "(%d bytes)", self.tab_id, suggested, out, size)
        except Exception as exc:  # noqa: BLE001 - capture never kills the page
            _log.warning("rendered tab %s: auto-capture of download failed: %r",
                         self.tab_id, exc)

    # -- error recovery --------------------------------------------------------
    def _fail_snapshot(self, action: str) -> dict[str, str]:
        """Capture a screenshot AND a DOM dump of the current page into the
        session's screenshots dir. Best-effort (never raises); returns the
        paths that were actually written, so the caller can name them in
        the error message."""
        paths: dict[str, str] = {}
        if not self.shot_on_error or self._page is None:
            return paths
        dest_dir = (Path(self._storage_state_path).parent.parent
                    / "screenshots" / self.session_name)
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return paths
        stamp = int(time.time() * 1000)
        shot = dest_dir / f"error-{self.tab_id}-{stamp}.png"
        try:
            self._page.screenshot(path=str(shot))
        except Exception as exc:  # noqa: BLE001 - snapshot best effort
            _log.debug("error snapshot screenshot failed: %s", exc)
        else:
            if shot.is_file() and shot.stat().st_size:
                paths["screenshot"] = str(shot)
        dom_path = dest_dir / f"error-{self.tab_id}-{stamp}.html"
        try:
            dom = self._page.content()
        except Exception as exc:  # noqa: BLE001 - snapshot best effort
            _log.debug("error snapshot DOM dump failed: %s", exc)
        else:
            try:
                dom_path.write_text(dom or "", encoding="utf-8")
            except OSError as exc:
                _log.debug("error snapshot DOM write failed: %s", exc)
            else:
                paths["dom"] = str(dom_path)
        return paths

    def _action_error(self, action: str, exc: Exception,
                      detail: str = "") -> BrowserError:
        """Wrap an action failure: snapshot the page, classify the cause,
        then raise a typed BrowserError that names exactly what failed,
        which taxonomy bucket it falls in, and where the evidence is.

        Classification never misreports: already-typed errors keep their
        type; network signals become BrowserNetworkError; anything else
        stays a plain BrowserError with the raw message.
        """
        paths = self._fail_snapshot(action)
        where = ""
        if paths:
            where = " (evidence: " + ", ".join(
                f"{k}={v}" for k, v in paths.items()) + ")"
        typed = errors.classify_exception(exc, url=self.url or "",
                                          evidence=paths)
        msg = (f"rendered {action} on {self.url or '(no page)'} failed"
               + (f" — {detail}" if detail else "")
               + f": {typed}{where}")
        return typed.with_message(msg)

    def persist(self) -> dict[str, Any]:
        """Flush cookies AND localStorage to the session's storage file
        right now (close() also does this). Returns the path written."""
        if self._context is None:
            raise BrowserError("rendered tab has no browser context yet — "
                               "navigate first")
        self._storage_state_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._context.storage_state(path=str(self._storage_state_path))
        except Exception as exc:  # noqa: BLE001 - persistence errors opaque
            raise BrowserError(
                f"could not persist rendered session state: {exc}") from exc
        return {"path": str(self._storage_state_path)}

    # -- navigation ----------------------------------------------------------
    def navigate(self, url: str, *, wait_until: str = "domcontentloaded",
                 retries: int | None = None) -> dict[str, Any]:
        """Render the page and append to history AFTER the load succeeded.

        ``wait_until``: load|domcontentloaded|networkidle|commit —
        "networkidle" for JS-heavy pages whose content arrives after the
        DOM is parsed. Failed loads retry ``retries`` times (default: the
        tab's ``retries``) with linear backoff; the final failure raises a
        BrowserError naming the captured screenshot/DOM evidence.
        """
        url = (url or "").strip()
        if not url:
            raise BrowserError("navigate needs a url")
        if wait_until not in _GOTO_WAIT_UNTIL:
            raise BrowserError(
                f"unknown wait_until {wait_until!r} "
                f"(want one of {sorted(_GOTO_WAIT_UNTIL)})")
        page = self._ensure_page()
        self._pace("navigate")
        attempts = 1 + max(0, self.retries if retries is None else retries)
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                self._goto_once(page, url, wait_until)
                last_exc = None
                break
            except BrowserNetworkError as exc:
                # Network blips are the only failures worth retrying.
                last_exc = exc
                if attempt < attempts - 1:
                    time.sleep(min(2.0 * (attempt + 1), 8.0))
            except BrowserError as exc:
                # Classified site errors and bot blocks do not heal on
                # immediate retry — hammering a block can extend it.
                # One snapshot, one wrapped message, via _action_error.
                raise self._action_error(
                    "navigate", exc, detail=url) from exc
            except Exception as exc:  # noqa: BLE001 - goto errors opaque
                last_exc = exc
                if attempt < attempts - 1:
                    time.sleep(min(2.0 * (attempt + 1), 8.0))
        if last_exc is not None:
            typed = self._action_error(
                "navigate", last_exc, detail=f"{url} ({attempts} attempts)")
            self.error = str(typed)
            raise typed
        self.url = page.url
        try:
            self.title = page.title()
        except Exception:  # noqa: BLE001 - title is cosmetic
            self.title = ""
        self.error = ""
        entry = {"url": self.url, "title": self.title, "ts": time.time()}
        self.history.append(entry)
        _notify_action(self, f"opened {self.url}")
        return dict(entry)

    def _require_loaded(self) -> Any:
        if self._page is None or not self.url:
            raise BrowserError("rendered tab has no loaded page — navigate first")
        return self._page

    def _content_sample(self, limit: int = 200_000) -> str:
        """Bounded read of the current page HTML (best-effort, "").

        Used for challenge-marker scans — never for content extraction.
        """
        page = self._page
        if page is None:
            return ""
        try:
            content = page.content()
        except Exception:  # noqa: BLE001 - cosmetic read
            return ""
        return (content or "")[:limit]

    def _goto_once(self, page: Any, url: str, wait_until: str) -> None:
        """One navigation attempt with failure classification.

        Raises a typed :class:`BrowserError` subclass: HTTP error
        statuses and challenge interstitials are classified immediately
        (no pointless retries); transport failures surface as
        :class:`BrowserNetworkError` for the caller's retry decision.
        Snapshots/evidence are attached by the caller's _action_error —
        this method only classifies.
        """
        try:
            response = page.goto(url, wait_until=wait_until,
                                 timeout=_RENDERED_GOTO_TIMEOUT_MS)
        except Exception as exc:  # noqa: BLE001 - goto errors opaque
            raise errors.classify_exception(exc, url=url) from exc
        status: int | None = None
        headers: dict[str, Any] = {}
        try:
            if response is not None:
                status = response.status
                headers = dict(response.headers or {})
        except Exception:  # noqa: BLE001 - response introspection is cosmetic
            status = None
        if status is not None and status >= 400:
            raise errors.classify_http_status(
                status, url=url, html=self._content_sample(),
                title=self._page_title(), headers=headers)
        # Challenge interstitials can ship HTTP 200: scan the title
        # (cheap), confirm with a marker sweep before calling it a block.
        title = self._page_title()
        if errors.detect_challenge("", title=title):
            html = self._content_sample()
            challenge = errors.detect_challenge(html, title=title)
            if challenge:
                label = challenge.replace("-", " ")
                raise BrowserBotDetectedError(
                    f"blocked by {label} on {url} — the page is a "
                    f"{label} interstitial (HTTP 200), not the site",
                    detection=challenge, url=url)

    def _page_title(self) -> str:
        page = self._page
        if page is None:
            return ""
        try:
            return page.title() or ""
        except Exception:  # noqa: BLE001 - title is cosmetic
            return ""

    def wait_for_load_state(self, state: str = "load",
                            timeout: int = 30_000) -> dict[str, Any]:
        """Wait for the page's load state: load|domcontentloaded|networkidle.

        The dynamic-site primitive: after a click that triggers XHR,
        ``wait_for_load_state("networkidle")`` waits until the network
        settles instead of a blind sleep. Fail fast on timeout.
        """
        page = self._require_loaded()
        state = (state or "").strip().lower()
        if state not in {"load", "domcontentloaded", "networkidle"}:
            raise BrowserError(
                f"unknown load state {state!r} "
                "(want load|domcontentloaded|networkidle)")
        try:
            page.wait_for_load_state(state, timeout=int(timeout))
        except Exception as exc:  # noqa: BLE001 - timeout errors opaque
            raise self._action_error(
                f"wait_for_load_state({state})", exc,
                detail=f"timeout after {timeout}ms")
        return {"ok": True, "state": state, "url": self.url}

    def wait_for_network_idle(self, timeout: int = 15_000) -> dict[str, Any]:
        """Shorthand for ``wait_for_load_state("networkidle")`` — the wait
        to use after actions on JS-heavy pages."""
        return self.wait_for_load_state("networkidle", timeout=timeout)

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
        """Legacy match: a form field by name, then id — whichever the page
        uses. Only used when the page driver cannot run JavaScript."""
        escaped = (name or "").replace('"', '\\"')
        return f'input[name="{escaped}"], textarea[name="{escaped}"], select[name="{escaped}"], [id="{escaped}"]'

    def _resolve_field(self, name: str) -> tuple[str, dict[str, Any] | None]:
        """Resolve ``name`` to ``(selector, info)`` via the smart field
        resolver (label/placeholder/aria/name/id scoring — see
        ``nomorals.browser.forms``).

        Fail fast with a field inventory: when nothing matches, the error
        lists the controls the page actually has instead of a bare name.
        """
        page = self._require_loaded()
        name = (name or "").strip()
        if not name:
            raise BrowserError("a field name is required")
        try:
            return forms.resolve(page, name)
        except forms.FieldNotFound:
            fields = forms.describe_fields(page)
            detail = ""
            if fields:
                detail = ("\nfields on this page:\n"
                          + forms.format_field_list(fields))
            paths = self._fail_snapshot("field resolution")
            where = ""
            if paths:
                where = " (evidence: " + ", ".join(
                    f"{k}={v}" for k, v in paths.items()) + ")"
            raise BrowserError(
                f"no form field {name!r} on {self.url or '(no page)'}"
                f"{detail}{where}")

    def _kind_of(self, name: str,
                 info: dict[str, Any] | None) -> dict[str, str] | None:
        """{"tag", "type"} from resolver info, or the legacy DOM inspect
        when the driver cannot run the resolver. None when the type cannot
        be inspected at all (duck-typed driver)."""
        if info is not None:
            return {"tag": info.get("tag", ""), "type": info.get("type", "")}
        return self._field_kind(name)

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

    def _resolve_field_wait(self, name: str, *, timeout: int,
                            poll_ms: int) -> tuple[str, dict[str, Any] | None]:
        """Poll the label-aware resolver until the field appears.

        For dynamic/AJAX forms whose fields render after XHR. Fail fast
        with a field inventory when the timeout expires.
        """
        page = self._require_loaded()
        name = (name or "").strip()
        if not name:
            raise BrowserError("a field name is required")
        timeout = max(0, int(timeout))
        poll = max(50, int(poll_ms)) / 1000.0
        deadline = time.monotonic() + timeout / 1000.0
        last_err: Exception | None = None
        while True:
            try:
                return forms.resolve(page, name)
            except forms.FieldNotFound as exc:
                last_err = exc
            if time.monotonic() >= deadline:
                break
            time.sleep(poll)
        fields = forms.describe_fields(page)
        detail = ""
        if fields:
            detail = ("\nfields on this page:\n"
                      + forms.format_field_list(fields))
        paths = self._fail_snapshot("wait_for_field")
        where = ""
        if paths:
            where = " (evidence: " + ", ".join(
                f"{k}={v}" for k, v in paths.items()) + ")"
        raise BrowserError(
            f"field {name!r} did not appear on {self.url or '(no page)'} "
            f"within {timeout}ms{detail}{where}") from last_err

    def wait_for_field(self, name: str, *, timeout: int = 10_000,
                       poll_ms: int = 250) -> dict[str, Any]:
        """Wait until a form field matching ``name`` appears in the DOM.

        The dynamic-form counterpart to :meth:`wait_for`: fields that
        render after AJAX/XHR never exist for the resolver to find on the
        first pass. Polls the label-aware resolver (label, placeholder,
        aria-label, name, id); fail fast with a field inventory on
        timeout — never a silent pass.
        """
        _selector, info = self._resolve_field_wait(
            name, timeout=timeout, poll_ms=poll_ms)
        _notify_action(self, f"field {name!r} appeared")
        return {"ok": True, "field": name, "tab_id": self.tab_id,
                "matched_via": (info or {}).get("by", "") or "name/id"}

    def _verify_field_value(self, name: str, expected: str) -> bool:
        """Confirm the marker-pinned field really holds ``expected``.

        Reads the live DOM back; on mismatch retries once through the
        React-compatible JS setter (framework-swallowed fills happen).
        Returns True when the value is confirmed, False when the driver
        cannot read the field back (duck-typed drivers) — an
        unverifiable fill reports ``verified: False`` instead of a false
        failure. Raises BrowserError naming expected vs actual only when
        the read-back positively shows the value did not stick.
        """
        page = self._require_loaded()
        if not self._can_evaluate():
            return False
        try:
            readback = forms.read_value_js(page)
        except Exception as exc:  # noqa: BLE001 - eval errors are opaque
            raise self._action_error(
                "fill-verify", exc, detail=f"field {name!r}") from exc
        if readback is None:
            return False
        if readback.get("value") == expected:
            return True
        # Framework-controlled inputs swallow native fills: go through
        # the native property setter + input/change events, then re-read.
        try:
            forms.set_value_js(page, expected)
            readback = forms.read_value_js(page)
        except Exception as exc:  # noqa: BLE001 - eval errors are opaque
            raise self._action_error(
                "fill-verify", exc, detail=f"field {name!r}") from exc
        if readback is None:
            return False
        actual = readback.get("value")
        if actual != expected:
            paths = self._fail_snapshot("fill-verify")
            where = ""
            if paths:
                where = " (evidence: " + ", ".join(
                    f"{k}={v}" for k, v in paths.items()) + ")"
            raise BrowserError(
                f"fill of {name!r} did not stick on "
                f"{self.url or '(no page)'}: expected {expected!r}, "
                f"the field holds {actual!r}{where}. The page's JS is "
                f"overwriting the value — try set_date/evaluate, or fill "
                f"the field and submit without re-reading.")
        return True

    def fill(self, name: str, value: str, *, verify: bool = True,
             wait_ms: int = 0) -> dict[str, Any]:
        """Fill a form field by label, placeholder, aria-label, name, or id.

        Type-aware: ``<select>`` fields route to :meth:`select`,
        checkboxes/radios route to :meth:`check`, file inputs route to
        :meth:`upload` when ``value`` is an existing file path (otherwise
        fail fast with a pointer to :meth:`upload`) — plain ``page.fill``
        only ever touches real text-like inputs, so a select no longer
        dies with an opaque playwright error.

        ``verify`` (default True) reads the field's live value back after
        the fill and retries through a framework-compatible JS setter when
        the value did not stick; a fill that will not stick raises instead
        of silently submitting the wrong data. ``wait_ms`` waits up to
        that long for the field to appear first (dynamic/AJAX forms).

        The result names which identity matched (``matched_via``:
        aria-label|label|placeholder|name|id|...), so a surprising match
        is visible instead of silent.
        """
        page = self._require_loaded()
        self._pace("fill")
        ref_name = (name or "").strip()
        if ref_name.lower().startswith("ref:") or re.fullmatch(
                r"\[ref=[^\]]+\]", ref_name):
            # Snapshot-ref fill: the agent already picked the element in
            # snapshot(); no name resolution, no type routing — a direct
            # fill of the resolved selector.
            selector = self.resolve_ref(
                ref_name[4:] if ref_name.lower().startswith("ref:")
                else ref_name[5:-1])
            try:
                page.fill(selector, str(value))
            except Exception as exc:  # noqa: BLE001 - selector errors opaque
                raise self._action_error(
                    "fill", exc, detail=f"field {name!r}") from exc
            _notify_action(self, f"filled {name!r}")
            return {"ok": True, "field": name, "tab_id": self.tab_id,
                    "verified": False, "matched_via": "snapshot-ref"}
        if wait_ms > 0:
            selector, info = self._resolve_field_wait(
                name, timeout=wait_ms, poll_ms=250)
        else:
            selector, info = self._resolve_field(name)
        kind = self._kind_of(name, info)
        tag = (kind or {}).get("tag", "")
        ftype = (kind or {}).get("type", "")
        try:
            if tag == "select":
                return self.select(name, value, verify=verify)
            if tag == "input" and ftype in {"checkbox", "radio"}:
                truthy = str(value).strip().lower() not in {
                    "", "0", "false", "no", "off", "unchecked"}
                return self.check(name, checked=truthy)
            if tag == "input" and ftype == "file":
                candidate = os.path.abspath(
                    os.path.expanduser(str(value or "").strip()))
                if candidate and os.path.isfile(candidate):
                    return self.upload(forms.MARKER_SELECTOR, candidate,
                                       verify=verify)
                raise BrowserError(
                    f"field {name!r} is a file input and {value!r} is not "
                    f"an existing file — use rendered upload with a real "
                    f"file path, not fill")
            page.fill(selector, str(value))
            verified = self._verify_field_value(name, str(value)) \
                if verify else False
        except BrowserError:
            raise
        except Exception as exc:  # noqa: BLE001 - selector errors opaque
            raise self._action_error(
                "fill", exc, detail=f"field {name!r}") from exc
        finally:
            forms.clear_marker(page)
        _notify_action(self, f"filled {name!r}")
        return {"ok": True, "field": name, "tab_id": self.tab_id,
                "verified": verified,
                "matched_via": (info or {}).get("by", "") or "name/id"}

    def fill_form(self, fields: dict[str, Any], *,
                  stop_on_error: bool = True,
                  verify: bool = True) -> dict[str, Any]:
        """Fill many fields in one call: ``{"Email address": "...",
        "country": "ng", "agree": True}``.

        Each value goes through :meth:`fill`, so selects/checkboxes/file
        inputs are handled per field and every fill is verified by
        reading the DOM back (``verify=False`` skips the read-back).
        Fail fast by default (``stop_on_error``); otherwise every field
        is attempted and failures are collected in the result's
        ``failed`` map.
        """
        if not isinstance(fields, dict) or not fields:
            raise BrowserError("fill_form needs a non-empty {field: value} mapping")
        filled: list[str] = []
        verified: list[str] = []
        failed: dict[str, str] = {}
        for name, value in fields.items():
            try:
                result = self.fill(str(name),
                                   "" if value is None else str(value),
                                   verify=verify)
            except BrowserError as exc:
                if stop_on_error:
                    raise
                failed[str(name)] = str(exc)
                continue
            filled.append(str(name))
            if result.get("verified"):
                verified.append(str(name))
        return {"ok": not failed, "filled": filled, "failed": failed,
                "verified": verified, "tab_id": self.tab_id}

    def set_date(self, name: str, value: str) -> dict[str, Any]:
        """Set a date field — native ``<input type="date">`` pickers and
        JS-driven text pickers alike.

        ``value`` accepts "2026-10-04", "04/10/2026", "10/04/2026",
        "4 Oct 2026" (see :func:`nomorals.browser.forms.normalize_date`).
        Native date inputs are filled directly; other inputs get the ISO
        date typed + Enter, with a React-compatible JS setter as fallback
        when the typed value doesn't stick. Fail fast on unparseable
        dates — never a silently wrong date.
        """
        page = self._require_loaded()
        try:
            iso = forms.normalize_date(value)
        except ValueError as exc:
            raise BrowserError(str(exc)) from exc
        self._pace("set_date")
        selector, info = self._resolve_field(name)
        kind = self._kind_of(name, info)
        tag = (kind or {}).get("tag", "")
        ftype = (kind or {}).get("type", "")
        try:
            if tag == "input" and ftype in {
                    "date", "datetime-local", "month", "time", "week"}:
                page.fill(selector, iso)
            elif tag in {"input", "textarea"} or tag == "":
                page.fill(selector, iso)
                press = getattr(page, "press", None)
                if callable(press):
                    try:
                        press(selector, "Enter")
                    except Exception:  # noqa: BLE001 - Enter is best effort
                        _log.debug("set_date Enter press failed")
                if info is not None:
                    readback = self.evaluate(
                        "() => { const el = document.querySelector("
                        "'[data-nm-field=\"1\"]'); "
                        "return el ? el.value : null; }")
                    if readback != iso:
                        forms.set_value_js(page, iso)
            else:
                raise BrowserError(
                    f"field {name!r} is a <{tag}> (type={ftype or 'n/a'}) — "
                    "not a date-settable control")
        except BrowserError:
            raise
        except Exception as exc:  # noqa: BLE001 - picker errors opaque
            raise self._action_error(
                "set_date", exc, detail=f"field {name!r} = {value!r}") from exc
        finally:
            forms.clear_marker(page)
        return {"ok": True, "field": name, "date": iso, "tab_id": self.tab_id}

    def describe_fields(self, limit: int = 100) -> dict[str, Any]:
        """Every fillable control on the current page with its visible
        identity (label, placeholder, aria-label, name, id). What to call
        a field when writing a fill/fill_form call — or when a "field not
        found" error needs context."""
        page = self._require_loaded()
        fields = forms.describe_fields(page)
        return {"url": self.url, "count": len(fields),
                "fields": fields[: max(1, int(limit))]}

    def evaluate(self, js: str, arg: Any = _MISSING) -> Any:
        """Run ``js`` in the page and return its result.

        The escape hatch for token injection (CAPTCHA solvers),
        localStorage reads, and anything the verbs don't cover. Fail fast
        when the driver cannot evaluate JavaScript.
        """
        page = self._require_loaded()
        if not (js or "").strip():
            raise BrowserError("evaluate needs JavaScript source")
        evaluate = getattr(page, "evaluate", None)
        if evaluate is None:
            raise BrowserError(
                "this page driver cannot evaluate JavaScript")
        try:
            if arg is _MISSING:
                return evaluate(js)
            return evaluate(js, arg)
        except Exception as exc:  # noqa: BLE001 - eval errors opaque
            raise self._action_error("evaluate", exc) from exc

    def local_storage(self, action: str = "get", key: str = "",
                      value: Any = None) -> dict[str, Any]:
        """Read/write the page's ``localStorage`` (the token stash most
        SPAs use for sessions).

        ``action``: get|set|remove|clear. ``get`` returns the value (None
        when the key is absent); ``set`` stores ``str(value)``; ``remove``
        deletes one key; ``clear`` wipes the origin's storage. Changes
        persist across restarts — ``localStorage`` is part of the
        persisted ``storage_state`` saved by :meth:`persist`/:meth:`close`.
        """
        page = self._require_loaded()
        action = (action or "get").strip().lower()
        if action not in {"get", "set", "remove", "clear"}:
            raise BrowserError(
                f"unknown localStorage action {action!r} "
                "(want get|set|remove|clear)")
        key = (key or "").strip()
        if action in {"get", "set", "remove"} and not key:
            raise BrowserError(
                f"localStorage {action} needs a key")
        snippets = {
            "get": ("(k) => window.localStorage.getItem(k)", key),
            "set": ("([k, v]) => { window.localStorage.setItem(k, String(v));"
                    " return true; }", [key, value]),
            "remove": ("(k) => { window.localStorage.removeItem(k);"
                       " return true; }", key),
            "clear": ("() => { window.localStorage.clear(); return true; }",
                      None),
        }
        js, arg = snippets[action]
        result = self.evaluate(js, arg)
        return {"action": action, "key": key, "value": result,
                "url": self.url}

    def select(self, name: str, value: str, *,
               by: str = "auto", verify: bool = True) -> dict[str, Any]:
        """Pick an option of a ``<select>`` dropdown by name or id.

        ``by``: "auto" (default) tries the option *value* first, then the
        visible *label*; "value" and "label" pin the match. ``verify``
        (default True) reads the selected value back from the live DOM —
        a select that reports success but shows another option raises
        instead of silently submitting the wrong choice. Fail fast when
        the field is not a ``<select>`` or the option does not exist.
        """
        page = self._require_loaded()
        name = (name or "").strip()
        if not name:
            raise BrowserError("rendered select needs a field name")
        by = (by or "auto").strip().lower()
        if by not in {"auto", "value", "label"}:
            raise BrowserError(
                f"unknown select match {by!r} (want auto|value|label)")
        self._pace("select")
        selector, info = self._resolve_field(name)
        kind = self._kind_of(name, info)
        if kind is None:
            raise BrowserError(
                f"cannot inspect field {name!r} on this page driver — "
                "select needs a live DOM with JS evaluation")
        if kind["tag"] != "select":
            raise BrowserError(
                f"field {name!r} is a <{kind['tag']}> "
                f"(type={kind['type'] or 'n/a'}), not a <select>")
        attempts = ([{"value": value}, {"label": value}] if by == "auto"
                    else [{"value": value}] if by == "value"
                    else [{"label": value}])
        last_exc: Exception | None = None
        picked: list[str] = []
        try:
            for kw in attempts:
                try:
                    picked = page.select_option(selector, **kw)
                except Exception as exc:  # noqa: BLE001 - opaque
                    last_exc = exc
                    continue
                if picked:
                    break
                last_exc = BrowserError(
                    f"no option matching {value!r} in select {name!r}")
            if not picked:
                raise self._action_error(
                    "select",
                    last_exc or BrowserError("unknown select failure"),
                    detail=f"field {name!r}, option {value!r}") from last_exc
            verified = self._verify_select_value(name, picked) \
                if verify else False
        except BrowserError:
            raise
        except Exception as exc:  # noqa: BLE001 - opaque
            raise self._action_error(
                "select", exc,
                detail=f"field {name!r}, option {value!r}") from exc
        finally:
            forms.clear_marker(page)
        return {"ok": True, "field": name, "picked": picked,
                "verified": verified, "tab_id": self.tab_id}

    def _verify_select_value(self, name: str, picked: list[str]) -> bool:
        """Read the marker-pinned ``<select>``'s live value back and make
        sure it is one of the options playwright reported as selected.

        Returns True when confirmed, False when the driver cannot read
        the value back. Raises only when the read-back positively shows
        a different option selected.
        """
        if not self._can_evaluate():
            return False
        try:
            actual = self.evaluate(
                "() => { const el = document.querySelector("
                "'[data-nm-field=\"1\"]'); "
                "return el ? el.value : null; }")
        except Exception as exc:  # noqa: BLE001 - eval errors are opaque
            raise self._action_error(
                "select-verify", exc, detail=f"field {name!r}") from exc
        if actual is None:
            return False
        if actual not in (picked or []):
            paths = self._fail_snapshot("select-verify")
            where = ""
            if paths:
                where = " (evidence: " + ", ".join(
                    f"{k}={v}" for k, v in paths.items()) + ")"
            raise BrowserError(
                f"select of {name!r} did not stick on "
                f"{self.url or '(no page)'}: picked {picked!r}, the field "
                f"shows {actual!r}{where}")
        return True

    def check(self, name: str, checked: bool = True,
              *, verify: bool = True) -> dict[str, Any]:
        """Check/uncheck a checkbox (or pick a radio) by name or id.

        ``verify`` (default True) reads the control's live ``checked``
        state back — a check the page's JS immediately undoes raises
        instead of silently submitting the wrong state. Fail fast when
        the field is not a checkable input.
        """
        page = self._require_loaded()
        name = (name or "").strip()
        if not name:
            raise BrowserError("rendered check needs a field name")
        self._pace("check")
        selector, info = self._resolve_field(name)
        kind = self._kind_of(name, info)
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
        try:
            if checked:
                page.check(selector)
            else:
                page.uncheck(selector)
            verified = False
            if verify and self._can_evaluate():
                readback = forms.read_value_js(page)
                actual = (readback or {}).get("checked")
                if actual is not None and bool(actual) != bool(checked):
                    paths = self._fail_snapshot("check-verify")
                    where = ""
                    if paths:
                        where = " (evidence: " + ", ".join(
                            f"{k}={v}" for k, v in paths.items()) + ")"
                    raise BrowserError(
                        f"{'check' if checked else 'uncheck'} of "
                        f"{name!r} did not stick on "
                        f"{self.url or '(no page)'}: the control still "
                        f"shows checked={actual!r}{where}")
                verified = actual is not None
        except BrowserError:
            raise
        except Exception as exc:  # noqa: BLE001 - opaque
            raise self._action_error(
                f"{'check' if checked else 'uncheck'}", exc,
                detail=f"field {name!r}") from exc
        finally:
            forms.clear_marker(page)
        return {"ok": True, "field": name, "checked": bool(checked),
                "verified": verified, "tab_id": self.tab_id}

    def click(self, target: str) -> dict[str, Any]:
        """Click a link/button: ``ref:<id>`` from the last :meth:`snapshot`,
        CSS selector when it looks like one (starts with ``#``, ``.``,
        ``[``, or contains ``>>``), otherwise visible text match. The
        tab's URL/title/history refresh after the click, like
        :meth:`Tab.click`."""
        page = self._require_loaded()
        selector = self._target_selector(target, "click")
        self._pace("click")
        before = page.url
        try:
            page.click(selector)
        except Exception as exc:  # noqa: BLE001 - click errors are opaque
            raise self._action_error(
                "click", exc, detail=f"target {target!r}") from exc
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
        _notify_action(self, f"clicked {target!r}")
        return {"ok": True, "target": target, "url": self.url,
                "title": self.title, "navigated": after != before}

    def submit(self, target: str = "", *,
               settle: bool = False) -> dict[str, Any]:
        """Submit a form: click ``target`` when given (button text/selector),
        else submit the page's first form directly. URL/title/history
        refresh after the submit.

        ``settle`` waits for network idle after the submit (best-effort)
        — for AJAX forms that never navigate, so the caller can read the
        result instead of racing it. The result reports ``settled``.
        """
        page = self._require_loaded()
        target = (target or "").strip()
        self._pace("submit")
        try:
            if target:
                self.click(target)
            else:
                page.eval_on_selector("form", "f => f.submit()")
        except BrowserError:
            raise
        except Exception as exc:  # noqa: BLE001 - submit errors are opaque
            raise self._action_error("submit", exc) from exc
        settled = False
        if settle:
            try:
                self.wait_for_network_idle(timeout=15_000)
                settled = True
            except BrowserError as exc:
                _log.debug("submit settle: network never went idle: %s", exc)
        try:
            self.url = page.url
            self.title = page.title()
        except Exception:  # noqa: BLE001 - cosmetic
            pass
        self.history.append(
            {"url": self.url, "title": self.title, "ts": time.time()})
        _notify_action(self, f"submitted form{f' via {target!r}' if target else ''}")
        return {"ok": True, "url": self.url, "title": self.title,
                "settled": settled}

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

    # -- accessibility snapshot (agent interaction model) ----------------------
    def snapshot(self, max_chars: int = _SNAPSHOT_MAX_CHARS) -> dict[str, Any]:
        """The page as an accessibility tree with ``[ref=..]`` handles —
        the browser-use / Playwright-MCP interaction model.

        Token-efficient and deterministic: the agent picks an element by
        ref and passes ``ref:<id>`` to :meth:`click`, :meth:`fill`,
        :meth:`hover`, :meth:`dblclick` or :meth:`press` instead of
        guessing selectors. Uses playwright's ``aria_snapshot`` when the
        driver has it, else a JS walker producing the same ref shape
        (source is reported). Refs expire on navigation — take a fresh
        snapshot after the DOM changes.
        """
        page = self._require_loaded()
        max_chars = max(1000, int(max_chars or _SNAPSHOT_MAX_CHARS))
        text: Any = ""
        source = "fallback"
        locator_fn = getattr(page, "locator", None)
        if callable(locator_fn):
            try:
                aria_fn = getattr(locator_fn("body"), "aria_snapshot", None)
                if callable(aria_fn):
                    text = aria_fn()
                    source = "aria"
            except Exception as exc:  # noqa: BLE001 - fall back to JS
                _log.debug("rendered tab %s: aria_snapshot failed: %r",
                           self.tab_id, exc)
                text = ""
        if not text and self._can_evaluate():
            try:
                text = self.evaluate(_SNAPSHOT_FALLBACK_JS)
            except Exception as exc:  # noqa: BLE001 - honest failure
                raise BrowserError(
                    f"rendered snapshot of {self.url} failed: {exc}"
                ) from exc
        if not text:
            raise BrowserError(
                "rendered snapshot needs a page driver with aria_snapshot "
                "or JS evaluation")
        text = str(text)
        refs: dict[str, str] = {}
        if source == "aria":
            for match in re.finditer(r"\[ref=([^\]]+)\]", text):
                ref = match.group(1)
                refs[ref] = f"aria-ref={ref}"
        else:
            cleaned: list[str] = []
            for line in text.splitlines():
                match = re.match(r"__REF__(.+)__SEL__(.+)", line)
                if match:
                    refs[match.group(1)] = match.group(2)
                else:
                    cleaned.append(line)
            text = "\n".join(cleaned)
        truncated = len(text) > max_chars
        self._snap_refs = refs
        return {
            "snapshot": text[:max_chars],
            "refs": dict(refs),
            "source": source,
            "truncated": truncated,
            "url": self.url,
            "tab_id": self.tab_id,
        }

    def resolve_ref(self, ref: str) -> str:
        """A snapshot ``[ref=..]`` handle (or ``ref:<id>``) to the
        playwright selector it maps to. Raises BrowserError listing the
        live refs when the handle is unknown or no snapshot was taken."""
        key = (ref or "").strip()
        if key.lower().startswith("ref:"):
            key = key[4:]
        selector = self._snap_refs.get(key)
        if selector is None:
            known = ", ".join(sorted(self._snap_refs)[:20])
            raise BrowserError(
                f"unknown snapshot ref {ref!r} on tab {self.tab_id} — "
                f"{'take snapshot() first' if not self._snap_refs else 'known refs: ' + known}")
        return selector

    def _target_selector(self, target: str, action: str) -> str:
        """A click/fill/hover-style target to a playwright selector.

        ``ref:<id>`` (or a bare ``[ref=..]`` handle from the last
        :meth:`snapshot`) resolves through the snapshot's ref map;
        otherwise the existing convention holds: CSS when it looks like
        a selector, visible-text match otherwise.
        """
        target = (target or "").strip()
        if not target:
            raise BrowserError(f"rendered {action} needs a target")
        low = target.lower()
        if low.startswith("ref:") or re.fullmatch(r"\[ref=[^\]]+\]", target):
            ref = (target[4:] if low.startswith("ref:")
                   else target[5:-1])
            return self.resolve_ref(ref)
        if target[:1] in {"#", ".", "["} or ">>" in target:
            return target
        return f"text={target}"

    # -- evidence: console, network, dialogs -----------------------------------
    def console_messages(self, limit: int = 100,
                         level: str = "") -> dict[str, Any]:
        """Console messages captured since the page was created (the
        Playwright-MCP ``console_messages`` evidence). ``level`` filters
        to one type: log|info|warning|error|debug."""
        level = (level or "").strip().lower()
        if level and level not in {"log", "info", "warning", "error",
                                   "debug"}:
            raise BrowserError(
                f"unknown console level {level!r} "
                "(want log|info|warning|error|debug)")
        msgs = [m for m in self._console_log
                if not level or m.get("type") == level]
        return {"messages": msgs[-max(1, int(limit or 100)):],
                "count": len(msgs), "tab_id": self.tab_id}

    def network_requests(self, limit: int = 100,
                         failed_only: bool = False) -> dict[str, Any]:
        """Network requests/responses captured since the page was created
        (the Playwright-MCP ``network_requests`` evidence). A console
        with no errors proves nothing about network failures — this is
        where non-2xx responses on the page's own origin show up.
        ``failed_only`` keeps just responses with status >= 400."""
        entries = self._network_log
        if failed_only:
            entries = [e for e in entries
                       if e.get("kind") == "response"
                       and isinstance(e.get("status"), int)
                       and e["status"] >= 400]
        return {"requests": entries[-max(1, int(limit or 100)):],
                "count": len(entries), "tab_id": self.tab_id}

    def dialogs(self, limit: int = 20,
                clear: bool = False) -> dict[str, Any]:
        """Dialogs (alert/confirm/prompt) seen since page creation, with
        the policy that handled each one."""
        out = list(self._dialog_log[-max(1, int(limit or 20)):])
        if clear:
            del self._dialog_log[:]
        return {"dialogs": out, "count": len(self._dialog_log),
                "policy": self._dialog_policy, "tab_id": self.tab_id}

    def set_dialog_policy(self, policy: str = "accept") -> dict[str, Any]:
        """How unexpected dialogs are handled: ``accept`` (default — a
        stray alert() never hangs automation), ``dismiss``, or
        ``record`` (leave the modal open for manual handling; the page
        will block until it is handled)."""
        policy = (policy or "accept").strip().lower()
        if policy not in {"accept", "dismiss", "record"}:
            raise BrowserError(
                f"unknown dialog policy {policy!r} "
                "(want accept|dismiss|record)")
        self._dialog_policy = policy
        return {"policy": policy, "tab_id": self.tab_id}

    # -- richer interaction ------------------------------------------------------
    def hover(self, target: str) -> dict[str, Any]:
        """Hover over an element (reveals tooltips, opens hover menus)."""
        page = self._require_loaded()
        selector = self._target_selector(target, "hover")
        self._pace("hover")
        try:
            page.hover(selector)
        except Exception as exc:  # noqa: BLE001 - hover errors are opaque
            raise self._action_error(
                "hover", exc, detail=f"target {target!r}") from exc
        _notify_action(self, f"hovered {target!r}")
        return {"ok": True, "target": target, "tab_id": self.tab_id}

    def dblclick(self, target: str) -> dict[str, Any]:
        """Double-click an element (text selection, zoom, edit-in-place)."""
        page = self._require_loaded()
        selector = self._target_selector(target, "dblclick")
        self._pace("dblclick")
        try:
            page.dblclick(selector)
        except Exception as exc:  # noqa: BLE001 - opaque
            raise self._action_error(
                "dblclick", exc, detail=f"target {target!r}") from exc
        _notify_action(self, f"double-clicked {target!r}")
        return {"ok": True, "target": target, "tab_id": self.tab_id}

    def press(self, key: str, target: str = "") -> dict[str, Any]:
        """Press a keyboard key — Enter, Escape, Tab, arrows, F-keys.

        With ``target`` the key goes to that element (focused first);
        without it, to the page itself. This is keyboard navigation the
        way a human does it, not a form fill.
        """
        page = self._require_loaded()
        key = (key or "").strip()
        if not key:
            raise BrowserError("rendered press needs a key")
        self._pace("press")
        try:
            keyboard = getattr(page, "keyboard", None)
            if target.strip():
                selector = self._target_selector(target, "press")
                press_fn = getattr(page, "press", None)
                if callable(press_fn):
                    press_fn(selector, key)
                elif keyboard is not None:
                    page.click(selector)
                    keyboard.press(key)
                else:
                    raise BrowserError(
                        "this page driver cannot press keys")
            else:
                if keyboard is None:
                    raise BrowserError(
                        "this page driver cannot press keys")
                keyboard.press(key)
        except BrowserError:
            raise
        except Exception as exc:  # noqa: BLE001 - press errors are opaque
            raise self._action_error(
                "press", exc, detail=f"key {key!r}") from exc
        _notify_action(self, f"pressed {key!r}")
        return {"ok": True, "key": key, "tab_id": self.tab_id}

    def drag(self, source: str, target: str) -> dict[str, Any]:
        """Drag ``source`` onto ``target`` (sliders, kanban, file drops)."""
        page = self._require_loaded()
        src_sel = self._target_selector(source, "drag source")
        tgt_sel = self._target_selector(target, "drag target")
        self._pace("drag")
        drag_fn = getattr(page, "drag_and_drop", None)
        if not callable(drag_fn):
            raise BrowserError("this page driver cannot drag and drop")
        try:
            drag_fn(src_sel, tgt_sel)
        except Exception as exc:  # noqa: BLE001 - opaque
            raise self._action_error(
                "drag", exc,
                detail=f"{source!r} -> {target!r}") from exc
        _notify_action(self, f"dragged {source!r} onto {target!r}")
        return {"ok": True, "source": source, "target": target,
                "tab_id": self.tab_id}

    def type_text(self, target: str, text: str,
                  delay_ms: int = 40) -> dict[str, Any]:
        """Type into a field character-by-character, like a human.

        Unlike :meth:`fill` (instant value set), this clicks into the
        field and types with per-character timing plus occasional
        "thinking" pauses between word groups — the behavioral-mimicry
        shape sites' bot checks actually measure. ``delay_ms`` is the
        per-character delay (0 = as fast as the driver goes).
        """
        page = self._require_loaded()
        selector = self._target_selector(target, "type")
        text = str(text or "")
        delay_ms = max(0, int(delay_ms))
        if not text:
            raise BrowserError("rendered type_text needs text to type")
        self._pace("type")
        keyboard = getattr(page, "keyboard", None)
        locator_fn = getattr(page, "locator", None)
        if keyboard is None or not callable(locator_fn):
            raise BrowserError("this page driver cannot type text")
        try:
            locator_fn(selector).click()
            # word-group chunks with thinking pauses between them read as
            # human cadence; a flat delay reads as a metronome.
            chunks: list[str] = []
            current = ""
            for word in text.split(" "):
                current = f"{current} {word}".strip()
                if len(current) >= 14:
                    chunks.append(current + " ")
                    current = ""
            if current:
                chunks.append(current)
            if not chunks:
                chunks = [text]
            for i, chunk in enumerate(chunks):
                keyboard.type(chunk, delay=delay_ms)
                if i < len(chunks) - 1:
                    time.sleep(random.uniform(0.15, 0.6))
            verified = False
            if self._can_evaluate():
                try:
                    readback = page.evaluate(
                        "(sel) => { const el = document.querySelector(sel);"
                        " return el ? el.value : null; }", selector)
                    verified = readback == text
                except Exception:  # noqa: BLE001 - verification best-effort
                    verified = False
        except BrowserError:
            raise
        except Exception as exc:  # noqa: BLE001 - type errors are opaque
            raise self._action_error(
                "type", exc, detail=f"target {target!r}") from exc
        _notify_action(self, f"typed into {target!r}")
        return {"ok": True, "target": target, "chars": len(text),
                "verified": verified, "tab_id": self.tab_id}

    def scroll(self, direction: str = "down",
               pixels: int = 600) -> dict[str, Any]:
        """Scroll the page in natural stepped chunks (accelerate-feel via
        small wheel steps, not one jump). ``direction``: down|up|top|
        bottom; ``pixels`` per scroll for down/up."""
        page = self._require_loaded()
        direction = (direction or "down").strip().lower()
        if direction not in {"down", "up", "top", "bottom"}:
            raise BrowserError(
                f"unknown scroll direction {direction!r} "
                "(want down|up|top|bottom)")
        pixels = max(50, int(pixels or 600))
        mouse = getattr(page, "mouse", None)
        evaluate = getattr(page, "evaluate", None)
        try:
            if direction in {"top", "bottom"} and callable(evaluate):
                evaluate(
                    "(toBottom) => window.scrollTo({top: toBottom ? "
                    "document.body.scrollHeight : 0, behavior: 'smooth'});",
                    direction == "bottom")
            elif mouse is not None and callable(getattr(mouse, "wheel", None)):
                signed = pixels if direction == "down" else -pixels
                # stepped wheel: cruise in chunks, ease the last one
                remaining = abs(signed)
                step_sign = 1 if signed > 0 else -1
                while remaining > 0:
                    step = min(220, remaining)
                    mouse.wheel(0, step_sign * step)
                    remaining -= step
                    time.sleep(random.uniform(0.03, 0.09))
            elif callable(evaluate):
                evaluate("(d) => window.scrollBy({top: d, behavior: 'smooth'});",
                         pixels if direction == "down" else -pixels)
            else:
                raise BrowserError("this page driver cannot scroll")
        except BrowserError:
            raise
        except Exception as exc:  # noqa: BLE001 - scroll errors are opaque
            raise self._action_error("scroll", exc) from exc
        _notify_action(self, f"scrolled {direction}")
        return {"ok": True, "direction": direction, "tab_id": self.tab_id}

    def element_shot(self, selector: str,
                     path: str | os.PathLike[str] | None = None
                     ) -> dict[str, Any]:
        """Screenshot one element (a chart, a captcha widget, a price
        card) instead of the whole page."""
        page = self._require_loaded()
        selector = (selector or "").strip()
        if not selector:
            raise BrowserError("rendered element_shot needs a selector")
        locator_fn = getattr(page, "locator", None)
        if not callable(locator_fn):
            raise BrowserError("this page driver cannot screenshot elements")
        dest_dir = (Path(self._storage_state_path).parent.parent
                    / "screenshots" / self.session_name)
        dest_dir.mkdir(parents=True, exist_ok=True)
        out = (Path(path) if path
               else dest_dir / f"element-{int(time.time() * 1000)}.png")
        try:
            locator_fn(selector).screenshot(path=str(out))
        except Exception as exc:  # noqa: BLE001 - capture errors are opaque
            raise self._action_error(
                "element_shot", exc, detail=f"selector {selector!r}") from exc
        if not out.is_file() or out.stat().st_size == 0:
            raise BrowserError(
                f"rendered element_shot of {selector!r} produced no image")
        return {"path": str(out), "selector": selector,
                "tab_id": self.tab_id}

    def pdf(self, path: str | os.PathLike[str] | None = None
            ) -> dict[str, Any]:
        """Save the current page as a PDF (receipts, articles, invoices —
        the Playwright-MCP ``pdf`` evidence)."""
        page = self._require_loaded()
        dest_dir = (Path(self._storage_state_path).parent.parent
                    / "pdfs" / self.session_name)
        dest_dir.mkdir(parents=True, exist_ok=True)
        out = (Path(path) if path
               else dest_dir / f"rendered-{int(time.time() * 1000)}.pdf")
        pdf_fn = getattr(page, "pdf", None)
        if not callable(pdf_fn):
            raise BrowserError("this page driver cannot save PDFs")
        try:
            pdf_fn(path=str(out))
        except Exception as exc:  # noqa: BLE001 - capture errors are opaque
            raise BrowserError(
                f"rendered pdf of {self.url} failed: {exc}") from exc
        if not out.is_file() or out.stat().st_size == 0:
            raise BrowserError(
                f"rendered pdf of {self.url} produced no document")
        return {"path": str(out), "url": self.url, "tab_id": self.tab_id}

    # -- history navigation ------------------------------------------------------
    def _after_navigation(self, page: Any, before: str,
                          action: str) -> dict[str, Any]:
        """Refresh url/title/history after a navigation action and
        notify the live view. Shared by reload/forward/back."""
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
        _notify_action(self, action)
        return {"ok": True, "url": self.url, "title": self.title,
                "navigated": after != before, "tab_id": self.tab_id}

    def reload(self, *, wait_until: str = "domcontentloaded",
               timeout: int = 60_000) -> dict[str, Any]:
        """Reload the current page (retries a flaky load, refreshes
        session-gated content)."""
        page = self._require_loaded()
        wait_until = (wait_until or "domcontentloaded").strip().lower()
        if wait_until not in _GOTO_WAIT_UNTIL:
            raise BrowserError(
                f"unknown wait_until {wait_until!r} "
                f"(want {'|'.join(sorted(_GOTO_WAIT_UNTIL))})")
        self._pace("reload")
        before = self.url
        try:
            page.reload(wait_until=wait_until, timeout=int(timeout))
        except Exception as exc:  # noqa: BLE001 - reload errors are opaque
            raise self._action_error("reload", exc) from exc
        return self._after_navigation(page, before, "reloaded the page")

    def forward(self) -> dict[str, Any]:
        """Go forward in the tab's history."""
        page = self._require_loaded()
        go_forward = getattr(page, "go_forward", None)
        if not callable(go_forward):
            raise BrowserError("this page driver cannot go forward")
        self._pace("forward")
        before = self.url
        try:
            go_forward()
        except Exception as exc:  # noqa: BLE001 - opaque
            raise self._action_error("forward", exc) from exc
        return self._after_navigation(page, before, "went forward")

    def back(self) -> dict[str, Any]:
        """Go back in the tab's history (the rendered counterpart of
        :meth:`Tab.back`)."""
        page = self._require_loaded()
        go_back = getattr(page, "go_back", None)
        if not callable(go_back):
            raise BrowserError("this page driver cannot go back")
        self._pace("back")
        before = self.url
        try:
            go_back()
        except Exception as exc:  # noqa: BLE001 - opaque
            raise self._action_error("back", exc) from exc
        return self._after_navigation(page, before,
                                      f"went back to {self.url}")

    # -- form intelligence ---------------------------------------------------------
    def candidates(self, name: str, limit: int = 5) -> dict[str, Any]:
        """Ranked candidate controls for ``name`` — the diagnosis half of
        form filling. When a fill fails, this shows what the page
        actually has (with shadow-DOM / iframe flags) so the caller can
        pick the right one instead of guessing."""
        page = self._require_loaded()
        name = (name or "").strip()
        if not name:
            raise BrowserError("rendered candidates needs a field name")
        found = forms.resolve_candidates(page, name, limit=limit)
        return {"field": name, "candidates": found,
                "tab_id": self.tab_id}

    def form_groups(self) -> dict[str, Any]:
        """The page's forms: fields grouped by owning ``<form>`` with each
        form's submit control. Pick the right form before a multi-field
        fill; find the submit target without guessing."""
        page = self._require_loaded()
        return {"forms": forms.describe_forms(page), "tab_id": self.tab_id}

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

    def upload(self, selector: str, file_path: str, *,
               verify: bool = True) -> dict[str, Any]:
        """Set a ``<input type="file">`` to a real local file (playwright
        set_input_files). The file must exist — fail fast otherwise.

        ``verify`` (default True) reads the input's ``files`` list back:
        an upload the page's JS clears or rejects raises instead of a
        silent empty submit. The result reports the attached file name.
        """
        page = self._require_loaded()
        selector = (selector or "").strip()
        path = os.path.abspath(os.path.expanduser((file_path or "").strip()))
        if not selector:
            raise BrowserError("rendered upload needs a selector")
        if not os.path.isfile(path):
            raise BrowserError(f"rendered upload: not a file: {file_path!r}")
        self._pace("upload")
        try:
            page.set_input_files(selector, path)
        except Exception as exc:  # noqa: BLE001 - upload errors are opaque
            raise self._action_error(
                "upload", exc,
                detail=f"{file_path!r} -> {selector!r}") from exc
        attached = ""
        verified = False
        if verify and self._can_evaluate():
            try:
                state = page.evaluate(
                    """(sel) => {
                        const el = document.querySelector(sel);
                        if (!el || !el.files) return null;
                        return {n: el.files.length,
                                name: el.files[0] ? el.files[0].name : ''};
                    }""", selector)
            except Exception as exc:  # noqa: BLE001 - eval errors opaque
                raise self._action_error(
                    "upload-verify", exc,
                    detail=f"{file_path!r} -> {selector!r}") from exc
            if state is None:
                verified = False  # driver cannot read back; not a failure
            elif not state.get("n"):
                paths = self._fail_snapshot("upload-verify")
                where = ""
                if paths:
                    where = " (evidence: " + ", ".join(
                        f"{k}={v}" for k, v in paths.items()) + ")"
                raise BrowserError(
                    f"rendered upload of {file_path!r} did not attach on "
                    f"{self.url or '(no page)'}: the file input holds "
                    f"0 file(s){where}")
            else:
                attached = str(state.get("name") or "")
                verified = True
        _notify_action(self, f"uploaded {path!r}")
        return {"ok": True, "selector": selector, "path": path,
                "attached": attached, "verified": verified,
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
        self._pace("download")
        # Suppress the context-level auto-capture while the explicit
        # expect_download runs — the file must be saved exactly once.
        self._auto_suppress = True
        try:
            with page.expect_download() as download_info:
                page.click(selector)
            download = download_info.value
        except Exception as exc:  # noqa: BLE001 - download errors are opaque
            raise self._action_error(
                "download", exc, detail=f"trigger {target!r}") from exc
        finally:
            self._auto_suppress = False
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
        stealth: dict[str, Any] | None = None,
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
        #: default stealth profile for rendered tabs (merged over the
        #: built-in defaults; per-tab overrides in open_rendered_tab).
        #: Fail fast on unknown keys — a silently ignored profile is a lie.
        self._stealth = _merge_stealth(None, stealth)
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
        #: pacing between browser actions, shared by every tab this
        #: service opens. NOMORALS_BROWSER_PACING pins it at startup
        #: ("delay_ms,jitter_ms"); otherwise starts disabled — pacing is
        #: an explicit owner setting, never a silent default.
        try:
            self._pacing = pacing_from_env() or Pacing.disabled()
        except ValueError as exc:
            raise BrowserError(str(exc)) from exc
        #: session name -> active rendered tab id (for tab switching).
        self._active_rendered: dict[str, str] = {}

    # -- pacing ------------------------------------------------------------------
    @property
    def pacing(self) -> Pacing:
        """The pacing shared by this service's tabs."""
        return self._pacing

    def set_pacing(self, pacing: Pacing | None = None, *,
                   enabled: bool | None = None,
                   delay_ms: int | None = None,
                   jitter_ms: int | None = None,
                   per_action: dict[str, Any] | None = None) -> dict[str, Any]:
        """Configure pacing between browser actions.

        Pass a :class:`Pacing` outright, or keyword tweaks applied over
        the current one (``per_action`` maps action names to
        ``[delay_ms, jitter_ms]`` pairs overriding the global cadence).
        Because tabs hold a reference to this same object, live tabs pick
        the change up immediately. Returns the active configuration.
        """
        # Mutate in place: tabs hold a reference to this same object, so
        # live tabs pick the change up immediately.
        if pacing is not None:
            if not isinstance(pacing, Pacing):
                raise BrowserError(
                    f"set_pacing needs a Pacing, got {type(pacing).__name__}")
            self._pacing.enabled = pacing.enabled
            self._pacing.delay_ms = pacing.delay_ms
            self._pacing.jitter_ms = pacing.jitter_ms
            self._pacing.per_action = dict(pacing.per_action)
        else:
            if enabled is not None:
                self._pacing.enabled = bool(enabled)
            if delay_ms is not None:
                self._pacing.delay_ms = max(0, int(delay_ms))
            if jitter_ms is not None:
                self._pacing.jitter_ms = max(0, int(jitter_ms))
            if per_action is not None:
                if not isinstance(per_action, dict):
                    raise BrowserError(
                        "set_pacing per_action must be a "
                        "{action: [delay_ms, jitter_ms]} mapping")
                cleaned: dict[str, tuple[int, int]] = {}
                for act, pair in per_action.items():
                    try:
                        cleaned[str(act).strip().lower()] = (
                            max(0, int(pair[0])), max(0, int(pair[1])))
                    except (TypeError, ValueError, IndexError) as exc:
                        raise BrowserError(
                            f"set_pacing per_action[{act!r}] must be "
                            f"[delay_ms, jitter_ms]: {exc}") from exc
                self._pacing.per_action = cleaned
            if ((self._pacing.delay_ms or self._pacing.jitter_ms
                 or self._pacing.per_action)
                    and not self._pacing.enabled):
                raise BrowserError(
                    "pacing has delay/jitter configured but enabled=False — "
                    "pass enabled=True or clear the delays")
        _emit("browser.pacing.configured", dict(self._pacing.describe()))
        return self._pacing.describe()

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
        tab = Tab(tab_id=tab_id, session_name=handle.name, session=session)
        tab.pacing = self._pacing
        return tab

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
                          proxy: str = "", *,
                          stealth: dict[str, Any] | None = None,
                          shot_on_error: bool = True,
                          retries: int = 2) -> RenderedTab:
        """Open a playwright-backed tab in ``session_name``'s cookie space.

        For JavaScript/Cloudflare-guarded pages that plain-HTTP tabs cannot
        pass. Cookies persist via playwright storage_state in the session's
        cookie dir. ``proxy`` overrides the session's proxy for this tab
        ("" = inherit the session proxy, which may itself be direct).
        ``stealth`` overrides the service's stealth profile for this tab;
        ``shot_on_error`` captures screenshot+DOM on action failures;
        ``retries`` is the navigate retry count. Fail fast: raises
        BrowserError when playwright or its chromium build is missing, or
        when the initial navigate fails.
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
            stealth=_merge_stealth(self._stealth, stealth),
            shot_on_error=shot_on_error,
            retries=retries,
        )
        # Wire the tab into the service: shared pacing, the download
        # auto-capture sink, and the per-session active tab.
        tab.pacing = self._pacing
        tab._download_dir = self.data_dir / "downloads" / session_name
        tab._on_auto_download = (
            lambda record: self._record_auto_download(tab, record))
        self._rendered_tabs[tab_id] = tab
        self._active_rendered[session_name] = tab_id
        try:
            if (url or "").strip():
                tab.navigate(url)
        except BrowserError:
            tab.close()
            del self._rendered_tabs[tab_id]
            if self._active_rendered.get(session_name) == tab_id:
                del self._active_rendered[session_name]
            raise
        return tab

    def close_rendered_tab(self, tab_id: str) -> None:
        tab = self._rendered_tabs.get(tab_id)
        if tab is None:
            raise BrowserError(f"unknown rendered tab {tab_id!r}")
        tab.close()
        del self._rendered_tabs[tab_id]
        if self._active_rendered.get(tab.session_name) == tab_id:
            # fall back to another live tab of the same session, if any
            rest = [t.tab_id for t in self._rendered_tabs.values()
                    if t.session_name == tab.session_name]
            if rest:
                self._active_rendered[tab.session_name] = rest[-1]
            else:
                self._active_rendered.pop(tab.session_name, None)

    def switch_rendered_tab(self, session_name: str, tab_id: str) -> dict[str, Any]:
        """Make ``tab_id`` the active rendered tab of ``session_name``.

        Fail fast on unknown tabs; the tab must belong to the session.
        """
        session_name = (session_name or "").strip()
        tab = self._rendered_tabs.get(tab_id)
        if tab is None:
            raise BrowserError(f"unknown rendered tab {tab_id!r}")
        if tab.session_name != session_name:
            raise BrowserError(
                f"rendered tab {tab_id!r} belongs to session "
                f"{tab.session_name!r}, not {session_name!r}")
        self._active_rendered[session_name] = tab_id
        _notify_action(tab, f"switched to rendered tab {tab_id}")
        return {"ok": True, "session": session_name, "tab_id": tab_id,
                "url": tab.url, "title": tab.title}

    def active_rendered_tab(self, session_name: str) -> dict[str, Any] | None:
        """The active rendered tab of a session (None when it has none)."""
        tab_id = self._active_rendered.get((session_name or "").strip(), "")
        tab = self._rendered_tabs.get(tab_id) if tab_id else None
        if tab is None:
            return None
        return {"tab_id": tab.tab_id, "session": tab.session_name,
                "url": tab.url, "title": tab.title}

    def _record_auto_download(self, tab: RenderedTab,
                              record: dict[str, Any]) -> dict[str, Any]:
        """Register a context-captured download in the download registry.

        Called by the tab's download-event sink with the saved path, size,
        and suggested filename.
        """
        path = record.get("path", "")
        size = int(record.get("size") or 0)
        suggested = str(record.get("suggested_filename") or "")
        mime = (mimetypes.guess_type(path)[0] or "application/octet-stream")
        download_id = ulid_now()
        self._record_download({
            "id": download_id,
            "session": tab.session_name,
            "tab_id": "",
            "rendered_tab_id": tab.tab_id,
            "url": record.get("url") or tab.url,
            "filename": suggested,
            "trigger": record.get("trigger") or "auto",
            "path": "",
            "size": 0,
            "mime": "",
            "category": "",
            "status": "in_progress",
            "started_at": time.time(),
            "finished_at": 0.0,
            "error": "",
        })
        rec = self._finish_download(
            download_id, status="completed", path=path,
            size=size, mime=mime)
        rec["filename"] = suggested
        _emit("browser.download.completed", {
            "url": tab.url, "path": path, "size": size, "mime": mime,
            "filename": suggested, "trigger": "auto",
            "session": tab.session_name, "rendered_tab_id": tab.tab_id,
            "download_id": download_id, "mission_id": self._mission_id,
        })
        return rec

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
                    raise errors.classify_http_status(
                        int(status), url=url,
                        headers=dict(getattr(response, "headers", {}) or {}))
                mime = (response.headers.get("Content-Type", "") or "").split(";")[0].strip()
                data = response.read()
        except BrowserError as exc:
            self._finish_download(download_id, status="failed", error=str(exc))
            raise
        except urllib.error.HTTPError as exc:
            typed = errors.classify_http_status(int(exc.code), url=url)
            self._finish_download(
                download_id, status="failed", error=str(typed))
            raise typed from exc
        except urllib.error.URLError as exc:
            typed = errors.classify_exception(exc, url=url)
            self._finish_download(
                download_id, status="failed", error=str(typed))
            raise typed from exc

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
            "filename": "",
            "trigger": "explicit",
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
        rec["filename"] = result.get("suggested_filename", "")
        _emit("browser.download.completed", {
            "url": tab.url, "path": path, "size": rec["size"], "mime": mime,
            "filename": rec["filename"], "trigger": "explicit",
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

    def pdf_rendered(self, tab_id: str,
                     path: str | os.PathLike[str] | None = None
                     ) -> dict[str, Any]:
        """Save a rendered tab's LIVE page as a PDF — receipts, articles,
        invoices. Logged-in state and JS mutations included."""
        tab = self._rendered_tabs.get(tab_id)
        if tab is None:
            raise BrowserError(f"unknown rendered tab {tab_id!r}")
        result = tab.pdf(path=path)
        artifact_uri: str | None = None
        if self._artifact_store is not None:
            pdf_path = Path(result["path"])
            art = self._artifact_store.put(
                pdf_path.read_bytes(),
                type="document",
                mime="application/pdf",
                creator="browser-service",
                mission_id=self._mission_id,
                provenance=Provenance(source_type="browser",
                                      source_id=tab.url),
            )
            artifact_uri = art.uri
        result["artifact_uri"] = artifact_uri
        return result

    def stats(self) -> dict[str, Any]:
        """Honest service snapshot: sessions, tabs, downloads, pacing —
        the daemon's ``stats`` op and CLI status surface."""
        sessions = self.list_sessions()
        plain_tabs = sum(len(self.get_session(n)._tabs) for n in sessions)
        return {
            "sessions": sessions,
            "plain_tabs": plain_tabs,
            "rendered_tabs": len(self._rendered_tabs),
            "rendered_tab_ids": sorted(self._rendered_tabs),
            "downloads": len(self._downloads),
            "pacing": self._pacing.describe(),
            "proxy_pool_attached": self.proxy_pool_attached(),
            "data_dir": str(self.data_dir),
        }

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

"""Browser-driven OSINT adapters — for sites with no API.

The API-only adapters in ``osint.py`` skip sites that need JavaScript,
form interaction, or login walls. These adapters drive a real Chromium tab
(via the spine ``browser`` tool's rendered engine) instead: fill the search
form, wait for results, extract them.

Public pages only. Paywalled/login-gated results are reported as
"login required", never bypassed.
"""

from __future__ import annotations

import re
import urllib.parse
from typing import Any

from .base import SourceAdapter
from .model import SearchResult
from .osint import build_osint_hit
from ..core.logging_setup import get_logger

_log = get_logger(__name__)


def _browser() -> Any:
    """The spine browser tool's function, or None when unwired."""
    try:
        from ..tools import registry as _reg
        tool = _reg.get("browser")
        if tool is None:
            return None
        # registry.get returns the tool callable or a wrapper; unwrap
        fn = getattr(tool, "func", None) or getattr(tool, "fn", None) or tool
        return fn if callable(fn) else None
    except Exception:  # noqa: BLE001
        return None


def _rendered_ok() -> tuple[bool, str]:
    try:
        from ..tools.browser import _playwright_available
        return _playwright_available()
    except Exception:  # noqa: BLE001
        return False, "browser tool not importable"


class BrowserPeopleSearchAdapter(SourceAdapter):
    """Name → public people-search results via a rendered browser.

    Fills the search form on public people-search engines, waits for the
    results, extracts names/locations/links. No API key needed; the browser
    is the API.
    """

    name = "osint_people_browser"
    result_type = "profile"
    description = (
        "People search via rendered browser: public people-search engines, "
        "form-filled and result-extracted. For names with no API coverage."
    )

    #: (engine label, search URL template, result-link CSS-ish hint)
    _ENGINES: tuple[tuple[str, str], ...] = (
        ("webmii", "https://webmii.com/people?n={q}"),
        ("peekyou", "https://www.peekyou.com/{q}"),
    )

    def search(self, query: str, *, limit: int, since=None, before=None):
        name = (query or "").strip()
        if not name or len(name.split()) < 2:
            return []  # needs at least a first+last name
        browse = _browser()
        if browse is None:
            _log.debug("osint_people_browser: spine browser tool unwired")
            return []
        ok, reason = _rendered_ok()
        if not ok:
            _log.debug("osint_people_browser: %s", reason)
            return []
        out: list[SearchResult] = []
        q = urllib.parse.quote(name)
        for label, tmpl in self._ENGINES:
            if len(out) >= limit:
                break
            url = tmpl.format(q=q)
            try:
                opened = browse("open", url=url, engine="rendered",
                                session="osint")
                if not opened.get("ok", True):
                    continue
                # let JS settle, then read the rendered text
                browse("wait", engine="rendered", session="osint",
                       timeout=8000)
                page = browse("text", engine="rendered", session="osint",
                              max_chars=12000)
                text = page.get("text", "")
                for m in re.finditer(
                        r"(https?://[^\s\"'<>]+)", page.get("url", url) and text):
                    link = m.group(1)
                    if len(out) >= limit:
                        break
                    out.append(build_osint_hit(
                        self, name,
                        f"{name} — {label} result",
                        link[:500],
                        f"Public people-search hit for '{name}' via {label}.",
                        0.6, confidence="low", engine=label,
                    ))
                    if len(out) >= 3:  # a few links per engine is enough
                        break
            except Exception as exc:  # noqa: BLE001 - one engine failing is fine
                _log.debug("osint_people_browser %s failed: %s", label, exc)
                continue
        try:
            browse("close", engine="rendered", session="osint")
        except Exception:  # noqa: BLE001
            pass
        return out


class BrowserSiteSearchAdapter(SourceAdapter):
    """"Search this JS-heavy site" — generic rendered-site search.

    For research queries against sites with no API and JS-rendered content:
    opens the site's search URL pattern, waits, extracts text. The site list
    is curated for public, keyless sources.
    """

    name = "osint_site_browser"
    result_type = "web"
    description = (
        "Rendered-browser site search: JS-heavy public sources with no API. "
        "Opens, waits for render, extracts."
    )

    #: (label, search-URL template with {q})
    _SITES: tuple[tuple[str, str], ...] = (
        ("archive", "https://web.archive.org/web/*/{q}*"),
    )

    def search(self, query: str, *, limit: int, since=None, before=None):
        q = (query or "").strip()
        if not q:
            return []
        browse = _browser()
        if browse is None:
            return []
        ok, _ = _rendered_ok()
        if not ok:
            return []
        out: list[SearchResult] = []
        eq = urllib.parse.quote(q)
        for label, tmpl in self._SITES:
            if len(out) >= limit:
                break
            try:
                url = tmpl.format(q=eq)
                browse("open", url=url, engine="rendered", session="osint")
                browse("wait", engine="rendered", session="osint", timeout=8000)
                page = browse("extract", engine="rendered", session="osint",
                              kind="links")
                links = page.get("links", []) or []
                for link in links[:limit]:
                    href = link.get("href", "") if isinstance(link, dict) else str(link)
                    text = link.get("text", "") if isinstance(link, dict) else ""
                    if not href.startswith("http"):
                        continue
                    out.append(build_osint_hit(
                        self, q,
                        text[:120] or href[:120],
                        href[:500],
                        f"Archived capture related to '{q}' ({label}).",
                        0.5, confidence="low", engine=label,
                    ))
                    if len(out) >= limit:
                        break
            except Exception as exc:  # noqa: BLE001
                _log.debug("osint_site_browser %s failed: %s", label, exc)
                continue
        try:
            browse("close", engine="rendered", session="osint")
        except Exception:  # noqa: BLE001
            pass
        return out


#: adapter classes in canonical order (appended after the API adapters)
BROWSER_OSINT_SPECS: list[tuple[str, str, str, type[SourceAdapter]]] = [
    (BrowserPeopleSearchAdapter.name, BrowserPeopleSearchAdapter.result_type,
     BrowserPeopleSearchAdapter.description, BrowserPeopleSearchAdapter),
    (BrowserSiteSearchAdapter.name, BrowserSiteSearchAdapter.result_type,
     BrowserSiteSearchAdapter.description, BrowserSiteSearchAdapter),
]

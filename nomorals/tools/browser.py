"""Browser tool: a stateful web session built on the standard library.

A :class:`BrowserSession` keeps its own cookie jar and current page, so the
agent can do what a human browser does without one — open a page, read it,
follow links, fill and submit forms — with a persistent identity across
steps. Parsing uses ``html.parser`` (stdlib) into a small DOM; no
BeautifulSoup, no Chromium.

If ``playwright`` is installed **and** ``NM_BROWSER_PLAYWRIGHT=1``, sessions
transparently use headless Chromium instead (for JS-heavy sites). Off by
default: zero extra dependencies.

One tool: ``browser(action, ...)`` behind ``NET_BROWSER``.
"""

from __future__ import annotations

import html
import http.cookiejar
import json
import mimetypes
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from html.parser import HTMLParser
from typing import Any

from ..core.errors import ToolError, classify
from ..core.logging_setup import get_logger
from ..core.policy import Capability
from ..core.trust import domain_tier

__all__ = ["BrowserSession", "Node", "register",
           "dom_headings", "dom_tables", "dom_forms", "dom_meta", "dom_nav",
           "parse_html"]

_log = get_logger(__name__)

_VOID = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "source", "track", "wbr",
})
_SKIP = frozenset({"script", "style", "noscript", "template", "svg", "iframe"})
_BLOCK = frozenset({
    "p", "div", "section", "article", "header", "footer", "main", "aside",
    "ul", "ol", "li", "table", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
    "blockquote", "pre", "form", "fieldset", "nav", "figure", "details",
})
_MAX_BODY = 5_000_000  # 5 MB per page is plenty; bigger is a PDF or a bug


# ── tiny DOM ─────────────────────────────────────────────────────────────────


class Node:
    """One node of the parsed page. Text nodes have ``tag == ""`` and carry
    their string in ``text``; element nodes have ``tag`` and ``attrs``."""

    __slots__ = ("tag", "text", "attrs", "children", "parent")

    def __init__(self, tag: str = "", attrs: dict[str, str] | None = None,
                 text: str = "", parent: "Node | None" = None) -> None:
        self.tag = tag
        self.text = text          # non-empty only on text nodes
        self.attrs = attrs or {}
        self.children: list[Node] = []
        self.parent = parent

    @property
    def is_text(self) -> bool:
        return self.tag == ""

    # -- structure ----------------------------------------------------------
    def append(self, child: "Node") -> None:
        child.parent = self
        self.children.append(child)

    def walk(self):
        """Depth-first over all nodes (including text)."""
        for child in self.children:
            yield child
            if not child.is_text:
                yield from child.walk()

    def find_all(self, tag: "str | tuple[str, ...]" = "", **attr: str) -> list["Node"]:
        """Elements matching ``tag`` (empty = any; a tuple = any of them)
        and all ``attr`` key=value."""
        tags = tag if isinstance(tag, (tuple, list, set, frozenset)) else (tag,)
        out: list[Node] = []
        for node in self.walk():
            if node.is_text:
                continue
            if tags and node.tag not in tags:
                continue
            if attr and not all(node.attrs.get(k) == v for k, v in attr.items()):
                continue
            out.append(node)
        return out

    # -- content ------------------------------------------------------------
    def inner_text(self) -> str:
        """Readable text of this subtree (skips _SKIP tags)."""
        out: list[str] = []

        def _collect(node: "Node") -> None:
            if node.is_text:
                out.append(node.text)
                return
            if node.tag in _SKIP:
                return
            if node.tag in _BLOCK and out and out[-1] != "\n":
                out.append("\n")
            for child in node.children:
                _collect(child)

        _collect(self)
        text = re.sub(r"[ \t]+", " ", "".join(out))
        text = re.sub(r"\n\s*\n+", "\n", text)
        return text.strip()


class _DOMBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node(tag="#root")
        self._stack: list[Node] = [self.root]

    def handle_starttag(self, tag, attrs) -> None:
        node = Node(tag=tag.lower(), attrs=dict(attrs), parent=self._stack[-1])
        self._stack[-1].append(node)
        if tag.lower() not in _VOID:
            self._stack.append(node)

    def handle_startendtag(self, tag, attrs) -> None:
        self._stack[-1].append(Node(tag=tag.lower(), attrs=dict(attrs), parent=self._stack[-1]))

    def handle_endtag(self, tag) -> None:
        tag = tag.lower()
        for i in range(len(self._stack) - 1, 0, -1):
            if self._stack[i].tag == tag:
                del self._stack[i:]
                break

    def handle_data(self, data) -> None:
        if data:
            self._stack[-1].append(Node(text=data, parent=self._stack[-1]))


def parse_html(markup: str) -> Node:
    builder = _DOMBuilder()
    try:
        builder.feed(markup)
        builder.close()
    except Exception:  # noqa: BLE001 - malformed HTML should degrade, not raise
        _log.debug("html parser choked mid-stream; using partial DOM")
    return builder.root


# ── markdown rendering ───────────────────────────────────────────────────────


def node_to_markdown(node: Node, *, depth: int = 0) -> str:
    """Render a DOM subtree as markdown (headings, links, lists, code)."""
    out: list[str] = []
    for child in node.children:
        if child.is_text:
            out.append(child.text)
            continue
        tag = child.tag
        if tag in _SKIP:
            continue
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(tag[1])
            out.append("\n" + "#" * level + " " + child.inner_text().strip() + "\n")
        elif tag == "a":
            inner = child.inner_text().strip()
            href = child.attrs.get("href", "")
            out.append(f"[{inner}]({href})" if href else inner)
        elif tag in {"strong", "b"}:
            inner = child.inner_text().strip()
            out.append(f"**{inner}**" if inner else "")
        elif tag in {"em", "i"}:
            inner = child.inner_text().strip()
            out.append(f"*{inner}*" if inner else "")
        elif tag == "code":
            out.append("`" + child.inner_text().strip() + "`")
        elif tag == "pre":
            out.append("\n```\n" + child.inner_text().strip() + "\n```\n")
        elif tag == "li":
            indent = "  " * max(0, depth)
            out.append(f"\n{indent}- " + child.inner_text().strip())
        elif tag == "br":
            out.append("\n")
        elif tag == "img":
            alt = child.attrs.get("alt", "")
            src = child.attrs.get("src", "")
            out.append(f"![{alt}]({src})" if src else "")
        elif tag in _BLOCK:
            out.append("\n" + node_to_markdown(child, depth=depth + (1 if tag in {"ul", "ol"} else 0)) + "\n")
        else:
            out.append(node_to_markdown(child, depth=depth))
    text = re.sub(r"[ \t]+", " ", "".join(out))
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


# ── the session ──────────────────────────────────────────────────────────────


class BrowserSession:
    """One cookie-kept web session with a current page and parsed DOM."""

    def __init__(
        self,
        *,
        user_agent: str = "NoMoralsCore/0.1 (browser tool)",
        timeout: float = 30.0,
        proxy_url: str = "",
        respect_robots: bool = True,
        name: str = "default",
        session_dir: str = "",
        retries: int = 2,
        max_task_steps: int = 12,
    ) -> None:
        self.name = name
        self.user_agent = user_agent
        self.timeout = timeout
        self.respect_robots = respect_robots
        #: wave 86: transient failures (timeout, 5xx, reset) are retried
        #: with backoff before they become errors.
        self.retries = max(0, int(retries))
        #: wave 86: cap for the multi-step ``task`` action (profile-tuned).
        self.max_task_steps = max(1, int(max_task_steps))
        #: wave 86: when set, cookies are persisted here across runs
        #: (data/browser/sessions/<name>.json) — a task that logs in once
        #: keeps the login next time.
        self.session_dir = str(session_dir or "").strip()
        self.cookie_jar = http.cookiejar.CookieJar()
        self._load_cookies()
        self.proxy_url = (proxy_url or "").strip()
        self._opener = self._build_opener()

        self.url: str = ""
        self.title: str = ""
        self.dom: Node | None = None
        self._raw: str = ""
        #: agent-supplied field overrides (fill); form fields keep their own
        #: value attribute unless overridden here
        self._form_values: dict[str, str] = {}
        self._history: list[str] = []
        #: solved captcha tokens for the current page (token field name ->
        #: token); merged into the next submit()'s form values.
        self.captcha_tokens: dict[str, str] = {}
        #: solved image/audio captcha text (kind -> text); the agent fills
        #: it into the visible captcha field itself.
        self.captcha_text: dict[str, str] = {}
        self.created_at = time.time()
        self.request_count = 0

    def _build_opener(self) -> Any:
        """The urllib opener for this session: cookie jar + proxy config.

        Rebuilt whenever the proxy changes so the SAME jar survives a
        proxy switch (cookies are not dropped on rotation).
        """
        handlers: list[Any] = [
            urllib.request.HTTPCookieProcessor(self.cookie_jar),
            urllib.request.HTTPSHandler(context=_ssl_context()),
        ]
        if self.proxy_url:
            handlers.append(urllib.request.ProxyHandler(
                {"http": self.proxy_url, "https": self.proxy_url}))
        else:
            handlers.append(urllib.request.ProxyHandler({}))
        return urllib.request.build_opener(*handlers)

    def set_proxy(self, proxy_url: str = "") -> dict[str, Any]:
        """Route this session's traffic through ``proxy_url`` ("" = direct).

        Rebuilds the opener around the existing cookie jar — a proxy
        switch never drops the session's cookies. Returns the applied
        proxy ("direct" when cleared).
        """
        self.proxy_url = (proxy_url or "").strip()
        self._opener = self._build_opener()
        _log.info("browser session %r: proxy -> %s", self.name,
                  self.proxy_url or "direct")
        return {"session": self.name,
                "proxy": self.proxy_url or "direct"}

    # -- low-level fetch ------------------------------------------------------
    def _fetch(self, url: str, *, method: str = "GET",
               form: dict[str, str] | None = None, extra_headers: dict[str, str] | None = None,
               retry: bool = True,
               files: dict[str, tuple[str, bytes, str]] | None = None,
               ) -> dict[str, Any]:
        """Fetch one URL.

        ``retry`` gates the transient-failure retry loop (timeouts, resets,
        5xx with backoff). It must be False for non-idempotent requests —
        retrying a POST that the server already processed would duplicate
        the submission. GETs retry; form POSTs never do.
        """
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ToolError(f"unsupported URL scheme: {parsed.scheme or '(none)'}")
        if self.respect_robots and not _robots_allowed(url, self.user_agent):
            raise ToolError(f"robots.txt disallows {url}")

        data = None
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        if files:
            # multipart/form-data (file upload): fields + files in one body.
            data, content_type = _encode_multipart(form or {}, files)
            headers["Content-Type"] = content_type
        elif form is not None:
            data = urllib.parse.urlencode(form).encode("utf-8")
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        if extra_headers:
            headers.update(extra_headers)
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        # wave D: transient failures (timeout, reset, 5xx) are retried with
        # backoff — but only for idempotent requests. A flaky GET gets a
        # second chance; a POST is attempted exactly once.
        attempts = self.retries + 1 if retry else 1
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                with self._opener.open(request, timeout=self.timeout) as response:
                    body = response.read(_MAX_BODY)
                    final_url = response.geturl()
                    status = getattr(response, "status", 200)
                    content_type = response.headers.get("Content-Type", "")
                break
            except urllib.error.HTTPError as exc:
                body = exc.read(_MAX_BODY) if hasattr(exc, "read") else b""
                final_url = url
                status = exc.code
                content_type = exc.headers.get("Content-Type", "") if exc.headers else ""
                if status < 500 or attempt >= attempts - 1:
                    break
                last_exc = exc  # 5xx: retry — the server is having a moment
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt >= attempts - 1:
                    raise ToolError(
                        f"{method} {url} failed: {classify(exc).message}"
                        + ("" if attempts == 1 else f" (after {attempts} attempts)")
                    ) from exc
                last_exc = exc
            time.sleep(0.5 * (2 ** attempt))
        else:  # pragma: no cover - the loop always breaks or raises
            raise ToolError(f"request failed: {classify(last_exc).message if last_exc else 'unknown'}")

        try:
            charset = (content_type.split("charset=")[-1].split(";")[0].strip()
                       or "utf-8").lower()
        except Exception:  # noqa: BLE001
            charset = "utf-8"
        if charset == "utf-8":
            text = body.decode("utf-8", errors="replace")
        else:
            try:
                text = body.decode(charset, errors="replace")
            except (LookupError, ValueError):
                text = body.decode("utf-8", errors="replace")
        self.request_count += 1
        return {
            "url": final_url, "status": status,
            "content_type": content_type, "text": text,
        }

    def fetch_bytes(self, url: str, dest_path: "str | os.PathLike[str]", *,
                    max_bytes: int = 1_073_741_824,
                    timeout: float | None = None) -> dict[str, Any]:
        """Download raw bytes through this session's opener.

        Cookies, the session proxy, and headers ride along — this is how
        the download fallback chain pulls media through an authenticated
        browser session.  Streams to ``dest_path`` (never holds the whole
        file in memory); raises :class:`ToolError` on failure.
        """
        from pathlib import Path as _Path

        parsed = urllib.parse.urlparse(url or "")
        if parsed.scheme not in {"http", "https"}:
            raise ToolError(f"unsupported URL scheme: {parsed.scheme or '(none)'}")
        if self.respect_robots and not _robots_allowed(url, self.user_agent):
            raise ToolError(f"robots.txt disallows {url}")
        dest = _Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={
            "User-Agent": self.user_agent,
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": self.url or url,
        }, method="GET")
        started = time.perf_counter()
        try:
            with self._opener.open(request,
                                   timeout=timeout or self.timeout) as response:
                status = getattr(response, "status", 200)
                content_type = response.headers.get("Content-Type", "")
                final_url = response.geturl()
                written = 0
                with open(dest, "wb") as fh:
                    while True:
                        chunk = response.read(256 * 1024)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > max_bytes:
                            raise ToolError(
                                f"download exceeds {max_bytes} bytes")
                        fh.write(chunk)
        except urllib.error.HTTPError as exc:
            raise ToolError(
                f"GET {url} -> HTTP {exc.code}: "
                f"{classify(exc).message}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ToolError(
                f"GET {url} failed: {classify(exc).message}") from exc
        self.request_count += 1
        _log.info("browser fetch_bytes url=%s status=%d bytes=%d "
                  "type=%s (%.1fs)", url, status, written,
                  content_type or "?", time.perf_counter() - started)
        return {"url": final_url, "status": status,
                "content_type": content_type, "bytes": written,
                "path": str(dest)}

    # -- public actions --------------------------------------------------------
    def do(self, action: str, **kw: Any) -> dict[str, Any]:
        """Dispatch one action. Returns a JSON-able dict; raises ToolError."""
        action = (action or "state").lower().strip()
        handler = _ACTIONS.get(action)
        if handler is None:
            raise ToolError(f"unknown browser action {action!r}; "
                            f"one of: {', '.join(sorted(_ACTIONS))}")
        return handler(self, **kw)

    def open(self, url: str = "", **_: Any) -> dict[str, Any]:
        if not (url or "").strip():
            raise ToolError("browser open needs a url")
        result = self._fetch(url.strip())
        self.url = result["url"]
        self._raw = result["text"]
        if self.url not in self._history[-20:]:
            self._history.append(self.url)
        self.dom = parse_html(self._raw)
        title_nodes = self.dom.find_all("title")
        self.title = title_nodes[0].inner_text() if title_nodes else ""
        # agent fills do not survive a navigation
        self._form_values = {}
        self.captcha_tokens = {}
        self.captcha_text = {}
        self._save_cookies()
        return {
            "ok": result["status"] < 400,
            "url": self.url,
            "status": result["status"],
            "title": self.title[:200],
            "chars": len(self._raw),
            "links": len(self.dom.find_all("a")),
            "forms": len(self.dom.find_all("form")),
            "cookies": len(self.cookie_jar),
        }

    def text(self, max_chars: int = 40000, **_: Any) -> dict[str, Any]:
        self._require_page()
        content = self.dom.inner_text()
        return {"url": self.url, "title": self.title, "chars": len(content),
                "text": content[:max_chars], "truncated": len(content) > max_chars}

    def markdown(self, max_chars: int = 40000, **_: Any) -> dict[str, Any]:
        self._require_page()
        content = node_to_markdown(self.dom)
        return {"url": self.url, "title": self.title, "chars": len(content),
                "markdown": content[:max_chars], "truncated": len(content) > max_chars}

    def html(self, max_chars: int = 2_000_000, **_: Any) -> dict[str, Any]:
        """Raw page HTML — the fetched markup before DOM parsing strips
        anything (``<script type="application/ld+json">`` blocks included).
        Used by connectors that parse structured data the DOM hides."""
        self._require_page()
        raw = self._raw
        return {"url": self.url, "title": self.title, "chars": len(raw),
                "html": raw[:max_chars], "truncated": len(raw) > max_chars}

    def links(self, max_links: int = 100, **_: Any) -> dict[str, Any]:
        self._require_page()
        out: list[dict[str, str]] = []
        seen: set[str] = set()
        for node in self.dom.find_all("a"):
            href = node.attrs.get("href", "")
            if not href:
                continue
            absolute = urllib.parse.urljoin(self.url, href)
            parsed = urllib.parse.urlparse(absolute)
            if parsed.scheme not in {"http", "https"}:
                continue
            key = absolute.split("#", 1)[0]
            if key in seen:
                continue
            seen.add(key)
            out.append({"text": node.inner_text().strip(), "url": absolute})
            if len(out) >= max_links:
                break
        return {"url": self.url, "count": len(out), "links": out}

    def click(self, target: str = "", **_: Any) -> dict[str, Any]:
        """Follow a link by text, href, or index. Returns the new page."""
        self._require_page()
        if not (target or "").strip():
            raise ToolError("browser click needs a target (link text, href, or index)")
        links = self.dom.find_all("a")
        wanted = target.strip()
        chosen_href = ""
        if wanted.isdigit():
            idx = int(wanted)
            if 0 <= idx < len(links) and links[idx].attrs.get("href"):
                chosen_href = links[idx].attrs["href"]
        else:
            for node in links:
                href = node.attrs.get("href", "")
                if not href:
                    continue
                if href == wanted or node.inner_text().strip() == wanted:
                    chosen_href = href
                    break
        if not chosen_href:
            raise ToolError(f"no link matching {wanted!r} on this page")
        return self.open(urllib.parse.urljoin(self.url, chosen_href))

    def fill(self, name: str = "", value: str = "", **_: Any) -> dict[str, Any]:
        """Store a value for a form field (submitted on the next submit)."""
        self._require_page()
        if not (name or "").strip():
            raise ToolError("browser fill needs a name (the input's name attribute)")
        if name not in _form_field_names(self.dom):
            # still allow it — dynamic forms exist — but flag it
            _log.debug("fill for unknown field %r (proceeding anyway)", name)
        self._form_values[name] = value
        return {"ok": True, "field": name,
                "pending": sorted(self._form_values)}

    def _find_field(self, name: str) -> Any:
        """First form field node with this name (or id), or None."""
        name = (name or "").strip()
        for form in self.dom.find_all("form"):
            for field in _form_fields(form):
                if field.attrs.get("name") == name \
                        or field.attrs.get("id") == name:
                    return field
        return None

    def select(self, name: str = "", value: str = "", **_: Any) -> dict[str, Any]:
        """Pick a ``<select>`` dropdown option (submitted on the next submit).

        ``value`` matches the option's ``value`` attribute first, then its
        visible text. Fail fast when the field is not a ``<select>`` or no
        option matches — a typo'd option name must not silently submit the
        dropdown's default.
        """
        self._require_page()
        field = self._find_field(name or "")
        if field is None:
            raise ToolError(f"browser select: no form field {name!r}")
        if (field.tag or "").lower() != "select":
            raise ToolError(
                f"browser select: field {name!r} is a <{field.tag}>, "
                "not a <select>")
        picked = ""
        for opt in field.find_all("option"):
            opt_value = opt.attrs.get("value", "")
            label = opt.inner_text().strip()
            if value == opt_value or (opt_value == "" and value == label) \
                    or value == label:
                picked = opt_value if "value" in opt.attrs else label
                break
        if not picked and value:
            options = [o.attrs.get("value", o.inner_text().strip()[:40])
                       for o in field.find_all("option")[:10]]
            raise ToolError(
                f"browser select: no option {value!r} in {name!r} "
                f"(options: {options})")
        self._form_values[name] = picked
        return {"ok": True, "field": name, "picked": picked,
                "pending": sorted(self._form_values)}

    def check(self, name: str = "", checked: bool = True, **_: Any) -> dict[str, Any]:
        """Check/uncheck a checkbox, or pick a radio button.

        HTTP forms only submit *checked* boxes, so unchecking removes the
        field from the pending values entirely. Fail fast when the field
        is not a checkbox/radio.
        """
        self._require_page()
        field = self._find_field(name or "")
        if field is None:
            raise ToolError(f"browser check: no form field {name!r}")
        ftype = (field.attrs.get("type") or "").lower()
        if (field.tag or "").lower() != "input" \
                or ftype not in {"checkbox", "radio"}:
            raise ToolError(
                f"browser check: field {name!r} is a <{field.tag}> "
                f"(type={ftype or 'n/a'}), not a checkbox/radio")
        if ftype == "radio" and not checked:
            raise ToolError(
                f"browser check: {name!r} is a radio button — radios "
                "cannot be unchecked, pick another option instead")
        if checked:
            self._form_values[name] = field.attrs.get("value", "on")
        else:
            self._form_values.pop(name, None)
        return {"ok": True, "field": name, "checked": bool(checked),
                "pending": sorted(self._form_values)}

    def submit(self, target: str = "", *,
               uploads: dict[str, str] | None = None, **_: Any) -> dict[str, Any]:
        """Submit a form: by index, id, or action-text match. Uses fills.

        ``uploads`` maps a file-input field name to a local file path —
        the form is then posted as multipart/form-data. A ``fill`` whose
        target is an ``<input type="file">`` is treated the same way (the
        filled value is the file path). Every upload path must exist and
        be a readable file, otherwise the submit fails fast. File uploads
        require a POST form — a GET form with files raises immediately.
        """
        self._require_page()
        forms = self.dom.find_all("form")
        if not forms:
            raise ToolError("no form on this page to submit")
        form = None
        wanted = (target or "").strip()
        if wanted.isdigit():
            idx = int(wanted)
            if 0 <= idx < len(forms):
                form = forms[idx]
        else:
            for node in forms:
                if wanted in ("", "0"):
                    form = node
                    break
                if node.attrs.get("id") == wanted or node.attrs.get("name") == wanted \
                        or node.attrs.get("action", "") == wanted:
                    form = node
                    break
        if form is None:
            raise ToolError(f"no form matching {wanted!r} (have {len(forms)}: use an index)")

        fields = _form_fields(form)
        values: dict[str, str] = {}
        file_field_names: set[str] = set()
        for field in fields:
            name = field.attrs.get("name")
            if not name:
                continue
            ftype = (field.attrs.get("type") or "").lower()
            if field.tag == "input" and ftype == "file":
                file_field_names.add(name)
                continue  # files go in the multipart body, not urlencoded
            if name in self._form_values:
                values[name] = self._form_values[name]
            else:
                values[name] = field.attrs.get("value", "")

        # solved captcha tokens (from check_captcha) ride along on the
        # submit — g-recaptcha-response / h-captcha-response /
        # cf-turnstile-response / fc-token.
        if self.captcha_tokens:
            values.update(self.captcha_tokens)

        # file uploads: explicit uploads= wins, then fills on file fields.
        upload_paths: dict[str, str] = {}
        if uploads:
            if not isinstance(uploads, dict):
                raise ToolError("submit uploads must be a {field_name: file_path} dict")
            for name, path in uploads.items():
                if not isinstance(name, str) or not isinstance(path, str):
                    raise ToolError("submit uploads must map field names to file paths")
                upload_paths[name] = path
        for name in file_field_names:
            if name not in upload_paths and name in self._form_values:
                upload_paths[name] = self._form_values[name]
        files: dict[str, tuple[str, bytes, str]] = {}
        for name, path in upload_paths.items():
            files[name] = _read_upload_file(name, path)

        method = (form.attrs.get("method") or "get").upper()
        if method not in {"GET", "POST"}:
            method = "POST"
        if files and method != "POST":
            raise ToolError(
                "file uploads require a POST form — this form uses GET")
        action = form.attrs.get("action", "") or self.url
        url = urllib.parse.urljoin(self.url, action)
        if method == "GET" and values:
            url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(values)
        # wave D: a POST submit is attempted exactly once — the server may
        # have processed it even when the response is lost, so retrying
        # would risk a duplicate submission. GET submits are idempotent
        # and keep the transient-failure retry.
        result = self._fetch(url, method=method,
                             form=values if method == "POST" else None,
                             files=files or None,
                             retry=(method == "GET"))
        self.url = result["url"]
        self._raw = result["text"]
        if self.url not in self._history[-20:]:
            self._history.append(self.url)
        self.dom = parse_html(self._raw)
        title_nodes = self.dom.find_all("title")
        self.title = title_nodes[0].inner_text() if title_nodes else ""
        self._form_values = {}
        self._save_cookies()
        return {
            "ok": result["status"] < 400,
            "url": self.url, "status": result["status"],
            "title": self.title[:200], "chars": len(self._raw),
            "uploaded": sorted(files),
        }

    def extract(self, target: str = "", kind: str = "", **_: Any) -> dict[str, Any]:
        """Structured extraction from the current page.

        Two modes:

        * ``kind`` empty — text of matching elements: a tag name, #id,
          .class, or css-lite 'tag.class' (the classic behaviour).
        * ``kind`` set — a STRUCTURED view of the page, no selector needed:

          - ``headings`` — the h1-h6 outline (level + text)
          - ``tables``   — every table as {caption, headers, rows}
          - ``forms``    — every form as {id, action, method, fields}
          - ``meta``     — title, description, og:*, canonical, favicon
          - ``nav``      — link groups: {url, text, tag}
        """
        self._require_page()
        kind = (kind or "").strip().lower()
        if kind in {"headings", "h"}:
            return self._extract_headings()
        if kind in {"tables", "table"}:
            return self._extract_tables()
        if kind in {"forms", "form"}:
            return self._extract_forms()
        if kind in {"meta", "head"}:
            return self._extract_meta()
        if kind in {"nav", "links-structured", "sitemap"}:
            return self._extract_nav()
        if kind:
            raise ToolError(f"unknown extract kind {kind!r}; "
                            f"use headings|tables|forms|meta|nav, or leave it empty for selector mode")

        wanted = (target or "body").strip()
        parsed_kind, value = _parse_selector(wanted)
        if parsed_kind == "id":
            nodes = [n for n in self.dom.walk() if not n.is_text and n.attrs.get("id") == value]
        elif parsed_kind == "class":
            nodes = [n for n in self.dom.walk()
                     if not n.is_text and value in (n.attrs.get("class") or "").split()]
        else:
            nodes = self.dom.find_all(value or "body")
        if not nodes:
            return {"url": self.url, "count": 0, "matches": [],
                    "note": f"nothing matched {wanted!r}"}
        matches = []
        for node in nodes[:50]:
            text = node.inner_text().strip()
            if text:
                matches.append(text[:2000])
        return {"url": self.url, "count": len(matches), "matches": matches}

    # -- structured extractors (wave 86 browser v2; bodies live at module
    # level as dom_* so rendered tabs can reuse them without duplication) --
    def _extract_headings(self) -> dict[str, Any]:
        return dom_headings(self.dom, self.url)

    def _extract_tables(self) -> dict[str, Any]:
        return dom_tables(self.dom, self.url)

    def _extract_forms(self) -> dict[str, Any]:
        return dom_forms(self.dom, self.url)

    def _extract_meta(self) -> dict[str, Any]:
        return dom_meta(self.dom, self.url, self.title)

    def _extract_nav(self, limit: int = 200) -> dict[str, Any]:
        return dom_nav(self.dom, self.url, limit)

    # -- advanced multi-page research walk (wave 85) --------------------------
    def walk(self, url: str = "", *, max_pages: int = 4,
             in_domain: bool = True, focus: str = "",
             max_chars: int = 12000, **_: Any) -> dict[str, Any]:
        """A breadth-first research walk — advanced browsing.

        From the current page (or ``url`` when given), repeatedly follows the
        best-scoring link and collects up to ``max_pages`` pages.  Links are
        scored by relevance (keyword overlap with ``focus`` when given) and
        by the domain's trust tier, so the walk drifts toward credible,
        on-topic pages instead of random corners of the site.

        * ``in_domain`` (default) — stay on the seed page's domain;
        * otherwise off-domain links are followed, but only from
          high-trust sources (tier ≥ 0.5);
        * dead links are recorded and skipped, never fatal.

        Returns ``{"pages": [{url, title, trust, excerpt}], "digest"}`` —
        the digest is one structured line per page, ready for the report.
        """
        if url:
            self.open(url)
        self._require_page()
        root_host = (urllib.parse.urlparse(self.url).hostname or "").lower()
        if not root_host:
            raise ToolError("cannot walk: the seed page has no host")
        max_pages = max(1, min(int(max_pages), 12))
        focus_words = {w for w in re.split(r"[^a-z0-9]+", (focus or "").lower())
                       if len(w) > 3}
        visited = {self.url.split("#", 1)[0]}
        pages: list[dict[str, Any]] = []

        def _collect_current() -> None:
            tier = domain_tier(self.url)[0]
            body = self.text(max_chars=max_chars).get("text", "")
            excerpt = " ".join(body.split())[:400]
            pages.append({"url": self.url, "title": self.title[:160],
                          "trust": tier, "chars": len(body),
                          "excerpt": excerpt})

        _collect_current()
        for _ in range(1, max_pages):
            links = self.links(max_links=80).get("links", [])
            best: tuple[float, float, str] | None = None
            for link in links:
                target = link["url"].split("#", 1)[0]
                if target in visited:
                    continue
                parsed = urllib.parse.urlparse(target)
                host = (parsed.hostname or "").lower()
                if not host or parsed.scheme not in {"http", "https"}:
                    continue
                same = host == root_host or host.endswith("." + root_host)
                if in_domain and not same:
                    continue
                tier = domain_tier(target)[0]
                if not in_domain and not same and tier < 0.5:
                    continue  # off-domain only from credible sources
                anchor = {w for w in re.split(r"[^a-z0-9]+",
                                              link.get("text", "").lower())
                          if len(w) > 2}
                relevance = len(anchor & focus_words) if focus_words else 0
                score = (relevance * 2.0, tier)
                if best is None or score > (best[0], best[1]):
                    best = (score[0], score[1], target)
            if best is None:
                break
            target = best[2]
            visited.add(target)
            try:
                self.open(target)
                _collect_current()
            except Exception as exc:  # noqa: BLE001 — a dead link ends the hop,
                pages.append({"url": target, "title": "", "trust": best[1],
                              "chars": 0, "excerpt": f"(unreadable: {exc})"})
                _log.debug("walk hop failed at %s: %s", target, exc)
                continue
        digest = "\n".join(
            f"{i}. [{p['trust']:.2f}] {p['title'] or p['url']}\n"
            f"   {p['url']}\n   {p['excerpt'][:300]}"
            for i, p in enumerate(pages, 1))
        return {"ok": True, "seed": self.url if not url else url,
                "pages": pages, "count": len(pages), "digest": digest}

    # -- multi-step tasks (wave 86 browser v2) --------------------------------
    _TASK_ACTS = {
        "open", "goto", "wait", "click", "fill", "submit", "extract",
        "text", "markdown", "links", "back", "stop",
    }
    #: wave D: only IDEMPOTENT task steps auto-retry on failure.
    #: fill/submit/click can mutate server state, so they run exactly once;
    #: a failure there surfaces immediately instead of risking a duplicate
    #: submission or a double click-through.
    _TASK_IDEMPOTENT = frozenset({
        "open", "text", "markdown", "links", "extract", "back", "wait",
    })

    def task(self, steps: Any = None, stop_on_error: bool = True,
             max_steps: int = 0, **_: Any) -> dict[str, Any]:
        """Multi-step web task: a small program of browsing actions.

        ``steps`` is a list of action dicts (or a JSON string of one),
        executed in order on ONE session — cookies and form fills carry
        across steps, so a login performed in step 2 is alive in step 10:

          [{"act": "open",    "url": "https://example.com"},
           {"act": "extract", "kind": "meta"},
           {"act": "fill",    "name": "q", "value": "no morals"},
           {"act": "submit",  "target": "0"},
           {"act": "extract", "kind": "headings"},
           {"act": "click",   "target": "next page"},
           {"act": "wait",    "seconds": 2},
           {"act": "back"},
           {"act": "stop",    "note": "done"}]

        Supported acts: open/goto | wait | click | fill | submit |
        extract | text | markdown | links | back | stop. Each IDEMPOTENT
        step (open/text/markdown/links/extract/back/wait) is retried once
        on failure; fill/submit/click NEVER auto-retry (they can mutate
        server state — a duplicate submit is worse than a failed one). A
        second failure halts the task (or continues, with
        ``stop_on_error=False``) and the report says exactly where and
        after how many attempts. Capped at ``max_steps`` (or the session's
        profile-tuned cap) so a bad plan cannot loop the network.
        """
        if isinstance(steps, str):
            try:
                steps = json.loads(steps)
            except json.JSONDecodeError as exc:
                raise ToolError(f"task steps must be a JSON list: {exc}") from exc
        if not isinstance(steps, list) or not steps:
            raise ToolError("task needs a non-empty list of step dicts")
        cap = max_steps or self.max_task_steps
        if len(steps) > cap:
            raise ToolError(f"task has {len(steps)} steps — cap is {cap} "
                            f"(profile limit; split the task or raise runtime limits)")
        report: list[dict[str, Any]] = []
        halted = False
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                raise ToolError(f"step {i} must be a dict, got {type(step).__name__}")
            act = str(step.get("act") or step.get("action") or "").strip().lower()
            if act == "goto":
                act = "open"
            if act not in self._TASK_ACTS:
                raise ToolError(f"step {i}: unknown act {act!r}; "
                                f"one of: {', '.join(sorted(self._TASK_ACTS))}")
            if act == "stop":
                report.append({"i": i, "act": "stop", "ok": True,
                               "note": str(step.get("note", "stopped by task"))[:200]})
                break
            data: Any = None
            error = ""
            attempts = 0
            # wave D: idempotent steps get one retry; mutating steps
            # (fill/submit/click) get exactly one attempt.
            max_attempts = 2 if act in self._TASK_IDEMPOTENT else 1
            for _attempt in range(max_attempts):
                attempts += 1
                try:
                    data = self._run_task_step(act, step)
                    error = ""
                    break
                except ToolError as exc:
                    error = str(exc)
                    if attempts >= max_attempts:
                        break
                except Exception as exc:  # noqa: BLE001 - report, don't crash
                    error = f"{type(exc).__name__}: {exc}"
                    if attempts >= max_attempts:
                        break
            ok = not error
            entry: dict[str, Any] = {"i": i, "act": act, "ok": ok,
                                     "attempts": attempts}
            if error:
                entry["error"] = error[:300]
            else:
                entry["data"] = self._compact_task_data(data)
            report.append(entry)
            if not ok and stop_on_error:
                halted = True
                break
        return {
            "ok": not halted,
            "session": self.name,
            "url": self.url,
            "title": self.title,
            "steps_total": len(steps),
            "steps_done": len(report),
            "steps": report,
        }

    def _run_task_step(self, act: str, step: dict[str, Any]) -> Any:
        if act == "open":
            return self.open(url=str(step.get("url", "")))
        if act == "wait":
            seconds = min(10.0, max(0.0, float(step.get("seconds", 1.0))))
            time.sleep(seconds)
            return {"waited": seconds}
        if act == "click":
            return self.click(target=str(step.get("target", "")))
        if act == "fill":
            return self.fill(name=str(step.get("name", "")),
                             value=str(step.get("value", "")))
        if act == "submit":
            return self.submit(target=str(step.get("target", "")),
                               uploads=step.get("uploads"))
        if act == "extract":
            return self.extract(target=str(step.get("target", "")),
                                kind=str(step.get("kind", "")))
        if act == "text":
            return self.text(max_chars=int(step.get("max_chars", 40000)))
        if act == "markdown":
            return self.markdown(max_chars=int(step.get("max_chars", 40000)))
        if act == "links":
            return self.links(max_links=int(step.get("max_links", 100)))
        if act == "back":
            if len(self._history) <= 1:
                raise ToolError("nowhere to go back to")
            self._history.pop()
            target_url = self._history[-1]
            return self.open(url=target_url)
        raise ToolError(f"unhandled task act {act!r}")

    @staticmethod
    def _compact_task_data(data: Any) -> Any:
        """Keep step data JSON-able and small enough for a tool result."""
        if isinstance(data, dict):
            out: dict[str, Any] = {}
            for k, v in data.items():
                if isinstance(v, list) and len(v) > 25:
                    out[k] = [BrowserSession._compact_task_data(x) for x in v[:25]]
                    out[k + "_truncated"] = len(v) - 25
                else:
                    out[k] = BrowserSession._compact_task_data(v)
            return out
        if isinstance(data, list):
            return [BrowserSession._compact_task_data(x) for x in data[:25]]
        if isinstance(data, str) and len(data) > 4000:
            return data[:4000] + f"… ({len(data)} chars total)"
        return data

    def state(self, **_: Any) -> dict[str, Any]:
        return {
            "session": self.name,
            "url": self.url,
            "title": self.title,
            "cookies": len(self.cookie_jar),
            "requests": self.request_count,
            "seconds_alive": round(time.time() - self.created_at, 1),
            "pending_form_values": sorted(self._form_values),
            "solved_captcha_tokens": sorted(self.captcha_tokens),
            "cookies_persisted_to": self._session_file(),
        }

    def close(self, **_: Any) -> dict[str, Any]:
        # Persist cookies BEFORE clearing: a task that logged in keeps the
        # login for the next session with this name.
        self._save_cookies()
        self.dom = None
        self._raw = ""
        self.url = ""
        self.title = ""
        self.cookie_jar.clear()
        self._form_values = {}
        self._history = []
        self.captcha_tokens = {}
        self.captcha_text = {}
        return {"ok": True, "closed": self.name, "cookies_persisted": bool(self.session_dir)}

    # -- captcha handling -------------------------------------------------------
    def check_captcha(self, solve: bool = True, **_: Any) -> dict[str, Any]:
        """Detect captchas on the current page and, by default, solve them.

        Detection scans the current page HTML (reCAPTCHA, hCaptcha,
        Turnstile, Cloudflare, GeeTest, Arkose, AWS WAF, image/audio
        challenges). When ``solve`` is true each challenge is run
        through the captcha solver (service backend first — the solver
        is ON by default — falling back to an owner takeover ping when
        it can't solve).

        Solved token captchas are stashed on the session and merged
        into the next ``submit`` automatically; solved image/audio text
        is returned (and stashed) so the agent can ``fill`` it into the
        visible captcha field. Challenges the solver couldn't clear come
        back with ``takeover: true`` and the owner already pinged.
        """
        self._require_page()
        from . import captcha as _cap

        challenges = _cap.detect_in_session(self)
        out: list[dict[str, Any]] = []
        # Solver on by default — same rule as the CLI/accounts:
        # explicit NM_CAPTCHA_SOLVER=0 goes straight to takeover.
        solver_on = os.environ.get("NM_CAPTCHA_SOLVER", "1") != "0"
        for ch in challenges:
            item: dict[str, Any] = ch.summary()
            if not solve:
                item["solve_attempted"] = False
                out.append(item)
                continue
            try:
                result = _cap.solve(ch, backend="auto", settings=None,
                                    solver_enabled=solver_on,
                                    notify_owner=True, context=None)
            except _cap.CaptchaError as exc:
                result = _cap.SolveResult(
                    ok=False, kind=ch.kind, backend="error",
                    detail=str(exc))
            item.update(result.to_dict())
            field = _cap.CaptchaKind.TOKEN_FIELD.get(ch.kind, "")
            if result.ok and field and result.token:
                self.captcha_tokens[field] = result.token
                item["injected_into"] = field
            elif result.ok and result.text:
                self.captcha_text[ch.kind] = result.text
                item["note"] = ("fill the captcha field with "
                                "'captcha_text'")
            out.append(item)
        return {
            "url": self.url,
            "challenges": out,
            "solved_tokens": sorted(self.captcha_tokens),
            "captcha_text": dict(self.captcha_text),
        }

    # -- cookie persistence (wave 86) ------------------------------------------
    def _session_file(self) -> str:
        if not self.session_dir:
            return ""
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", self.name or "default")
        return os.path.join(self.session_dir, f"{safe}.json")

    def _save_cookies(self) -> None:
        path = self._session_file()
        if not path:
            return
        try:
            os.makedirs(self.session_dir, exist_ok=True)
            # __getstate__ is the canonical cookie field dump — immune to
            # constructor signature drift between Python versions.
            payload = [c.__getstate__() for c in self.cookie_jar]
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, default=str)
            os.replace(tmp, path)
        except Exception as exc:  # noqa: BLE001 - persistence is best-effort
            _log.debug("cookie save failed for session %r: %s", self.name, exc)

    def _load_cookies(self) -> None:
        path = self._session_file()
        if not path or not os.path.isfile(path):
            return
        try:
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
            fields = {"version", "name", "value", "port", "port_specified",
                      "domain", "domain_specified", "domain_initial_dot",
                      "path", "path_specified", "secure", "expires", "discard",
                      "comment", "comment_url", "rest", "rfc2109"}
            count = 0
            for entry in payload:
                if not isinstance(entry, dict):
                    continue
                cookie = http.cookiejar.Cookie.__new__(http.cookiejar.Cookie)
                for key, value in entry.items():
                    if key in fields:
                        setattr(cookie, key, value)
                if not getattr(cookie, "name", "") or \
                        not getattr(cookie, "domain", ""):
                    continue  # half-formed entry: skip, keep the rest
                self.cookie_jar.set_cookie(cookie)
                count += 1
            if count:
                _log.info("browser session %r: restored %d persisted cookie(s)",
                          self.name, count)
        except Exception as exc:  # noqa: BLE001 - a corrupt file just means no cookies
            _log.debug("cookie load failed for session %r: %s", self.name, exc)

    # -- helpers ---------------------------------------------------------------
    def _require_page(self) -> None:
        if self.dom is None:
            raise ToolError("no page open — use browser open <url> first")


_ACTIONS = {
    "open": BrowserSession.open,
    "text": BrowserSession.text,
    "markdown": BrowserSession.markdown,
    "html": BrowserSession.html,
    "links": BrowserSession.links,
    "click": BrowserSession.click,
    "fill": BrowserSession.fill,
    "submit": BrowserSession.submit,
    "check_captcha": BrowserSession.check_captcha,
    "extract": BrowserSession.extract,
    "walk": BrowserSession.walk,
    "task": BrowserSession.task,
    "state": BrowserSession.state,
    "close": BrowserSession.close,
}


def _parse_selector(selector: str) -> tuple[str, str]:
    """'tag', '#id', or '.class' → (kind, value) with kind in tag|id|class."""
    s = selector.strip()
    if s.startswith("#"):
        return "id", s[1:]
    if s.startswith("."):
        return "class", s[1:]
    return "tag", s


# ── structured DOM extractors (module-level: reusable by rendered tabs) ──


def dom_headings(dom: Node, url: str) -> dict[str, Any]:
    """The h1-h6 outline of a parsed DOM as {level, text} items."""
    out: list[dict[str, Any]] = []
    for node in dom.walk():
        if node.is_text:
            continue
        tag = (node.tag or "").lower()
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            text = " ".join(node.inner_text().split())
            if text:
                out.append({"level": int(tag[1]), "text": text[:300]})
    return {"url": url, "kind": "headings", "count": len(out), "items": out[:120]}


def dom_tables(dom: Node, url: str) -> dict[str, Any]:
    """Every table as {headers, rows, row_count}."""
    out: list[dict[str, Any]] = []
    for table in dom.find_all("table")[:20]:
        trs = table.find_all("tr")
        rows: list[list[str]] = []
        for tr in trs:
            cells = [
                " ".join(td.inner_text().split())
                for td in tr.find_all(("td", "th"))
            ]
            if any(cells):
                rows.append([c[:200] for c in cells])
        if not rows:
            continue
        first_cells = trs[0].find_all(("td", "th")) if trs else []
        header_row = (
            first_cells
            and any((n.tag or "") == "th" for n in first_cells)
            and all((n.tag or "") == "th" for n in first_cells)
        )
        headers = rows[0] if header_row else []
        data_rows = rows[1:] if header_row else rows
        out.append({
            "headers": headers,
            "rows": data_rows[:100],
            "row_count": len(rows),
        })
    return {"url": url, "kind": "tables", "count": len(out), "tables": out}


def dom_forms(dom: Node, url: str) -> dict[str, Any]:
    """Every form as {id, name, action, method, fields}."""
    out: list[dict[str, Any]] = []
    for form in dom.find_all("form")[:20]:
        fields = []
        for field in _form_fields(form):
            tag = (field.tag or "").lower()
            if tag not in {"input", "select", "textarea"}:
                continue
            ftype = field.attrs.get("type", "" if tag != "input" else "text")
            fields.append({
                "tag": tag,
                "type": ftype if tag == "input" else "",
                "name": field.attrs.get("name", ""),
                "id": field.attrs.get("id", ""),
                "placeholder": field.attrs.get("placeholder", ""),
                "value": field.attrs.get("value", "")[:80],
                "required": field.attrs.get("required") is not None
                            or "aria-required" in field.attrs,
            })
        out.append({
            "id": form.attrs.get("id", ""),
            "name": form.attrs.get("name", ""),
            "action": form.attrs.get("action", ""),
            "method": (form.attrs.get("method") or "get").upper(),
            "fields": fields[:40],
        })
    return {"url": url, "kind": "forms", "count": len(out), "forms": out}


def dom_meta(dom: Node, url: str, title: str) -> dict[str, Any]:
    """Title, description, og:*, canonical, and other head metadata."""
    meta: dict[str, str] = {}
    for node in dom.walk():
        if node.is_text or (node.tag or "").lower() != "meta":
            continue
        key = (node.attrs.get("property") or node.attrs.get("name") or "").strip().lower()
        content = (node.attrs.get("content") or "").strip()
        if key and content and key not in meta:
            meta[key] = content[:300]
    canonical = ""
    for node in dom.walk():
        if not node.is_text and (node.tag or "").lower() == "link" \
                and node.attrs.get("rel") == "canonical":
            canonical = node.attrs.get("href", "")
            break
    return {
        "url": url, "kind": "meta",
        "title": title,
        "description": meta.get("description", "") or meta.get("og:description", ""),
        "canonical": canonical,
        "og": {k: v for k, v in meta.items() if k.startswith("og:")},
        "other": {k: v for k, v in meta.items()
                  if not k.startswith("og:") and k not in {"description", "viewport"}},
    }


def dom_nav(dom: Node, url: str, limit: int = 200) -> dict[str, Any]:
    """Link groups: {url, text, tag}."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for a in dom.find_all("a"):
        href = (a.attrs.get("href") or "").strip()
        if not href or href.startswith(("#", "javascript:")):
            continue
        absolute = urllib.parse.urljoin(url, href)
        text = " ".join(a.inner_text().split())[:120]
        key = (absolute, text)
        if key in seen:
            continue
        seen.add(key)
        out.append({"url": absolute, "text": text, "tag": (a.attrs.get("class") or "")[:60]})
        if len(out) >= limit:
            break
    return {"url": url, "kind": "nav", "count": len(out), "links": out}


# ── form helpers ─────────────────────────────────────────────────────────────


def _form_fields(form: Node) -> list[Node]:
    fields: list[Node] = []
    for node in form.walk():
        if node.tag in {"input", "textarea", "select"}:
            fields.append(node)
    return fields


def _form_field_names(dom: Node) -> set[str]:
    names: set[str] = set()
    for form in dom.find_all("form"):
        for field in _form_fields(form):
            name = field.attrs.get("name")
            if name:
                names.add(name)
    return names


# ── multipart file uploads ─────────────────────────────────────────────────


def _read_upload_file(field: str, path: str) -> tuple[str, bytes, str]:
    """Read an upload file off disk. Fail fast: missing/unreadable files
    are errors, never silent empty uploads."""
    p = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(p):
        raise ToolError(f"upload field {field!r}: not a file: {path!r}")
    try:
        with open(p, "rb") as fh:
            data = fh.read()
    except OSError as exc:
        raise ToolError(f"upload field {field!r}: cannot read {path!r}: {exc}") from exc
    mime, _ = mimetypes.guess_type(p)
    return os.path.basename(p), data, mime or "application/octet-stream"


def _encode_multipart(
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes, str]],
) -> tuple[bytes, str]:
    """Encode urlencoded fields + files as multipart/form-data.

    Returns (body, content_type). The boundary is random per call so two
    uploads never collide.
    """
    boundary = "----NoMoralsBoundary" + uuid.uuid4().hex
    buf = bytearray()

    def _part_headers(name: str, filename: str = "", mime: str = "") -> bytes:
        disp = f'form-data; name="{name}"'
        if filename:
            # quote the filename the way browsers do
            disp += f'; filename="{filename}"'
        head = f"--{boundary}\r\nContent-Disposition: {disp}\r\n"
        if mime:
            head += f"Content-Type: {mime}\r\n"
        return head.encode("utf-8") + b"\r\n"

    for name, value in (fields or {}).items():
        buf += _part_headers(str(name))
        buf += str(value).encode("utf-8")
        buf += b"\r\n"
    for name, (filename, data, mime) in (files or {}).items():
        buf += _part_headers(str(name), filename=filename, mime=mime)
        buf += data
        buf += b"\r\n"
    buf += f"--{boundary}--\r\n".encode("utf-8")
    return bytes(buf), f"multipart/form-data; boundary={boundary}"





# ── robots (shared with web tools) ───────────────────────────────────────────


def _robots_allowed(url: str, user_agent: str) -> bool:
    try:
        from .web import _robots

        return _robots.allowed(url, user_agent)
    except Exception:  # noqa: BLE001 - robots should never hard-block the browser
        return True


# ── ssl (one context, shared) ────────────────────────────────────────────────


def _ssl_context():
    import ssl

    return ssl.create_default_context()


# ── session registry ─────────────────────────────────────────────────────────

_sessions: dict[str, BrowserSession] = {}
_sessions_lock = threading.Lock()


def get_session(name: str = "default", **settings: Any) -> BrowserSession:
    with _sessions_lock:
        session = _sessions.get(name or "default")
        if session is None:
            session = BrowserSession(name=name or "default", **settings)
            _sessions[name or "default"] = session
        return session


def drop_session(name: str = "default") -> bool:
    with _sessions_lock:
        return _sessions.pop(name or "default", None) is not None


# ── tool registration ────────────────────────────────────────────────────────


def register(registry: Any) -> None:
    context = registry.context
    settings = getattr(context, "settings", None) if context is not None else None
    tools_settings = getattr(settings, "tools", None) if settings else None
    user_agent = getattr(tools_settings, "user_agent", "NoMoralsCore/0.1") if tools_settings else "NoMoralsCore/0.1"
    timeout = getattr(tools_settings, "http_timeout", 30.0) if tools_settings else 30.0
    respect_robots = getattr(tools_settings, "robots_txt", True) if tools_settings else True
    proxy_url = getattr(tools_settings, "proxy_url", "") if tools_settings else ""
    # wave 86: cookie persistence on disk + profile-tuned task cap
    tune = (getattr(context, "extras", None) or {}).get("tune") if context is not None else None
    session_dir = str(getattr(settings, "resolve", lambda p: p)
                      ("data/browser/sessions")) if settings is not None else ""
    tune_profile = getattr(tune, "profile", None) if tune is not None else None
    tune_kind = str(getattr(tune_profile, "kind", ""))
    max_task_steps = 6 if tune_kind in ("termux", "mobile", "embedded") else 12

    @registry.register(
        "browser",
        description=(
            "stateful web browsing (v2): open a page and read it (text/markdown), "
            "list links, click, fill and submit forms (submit takes a "
            "multipart file upload via uploads={field: path}), extract "
            "elements or "
            "STRUCTURED views (headings/tables/forms/meta/nav), walk a "
            "trust-ranked multi-page research path, or run a multi-step task "
            "(a small program of open/fill/submit/extract/click/back steps); "
            "cookies persist per session AND across restarts; transient "
            "failures on idempotent fetches are retried with backoff, "
            "form submits and other mutating steps never auto-retry; "
            "'check_captcha' detects captchas on the current page and solves "
            "them (service backend first, owner takeover as fallback — the "
            "solver is ON by default), stashing tokens that the next "
            "'submit' posts automatically"
        ),
        capability=Capability.NET_BROWSER,
        parameters={
            "action": "str — open|text|markdown|links|click|fill|submit|extract|walk|task|state|close|check_captcha",
            "solve": "bool — for check_captcha: detect+ solve (default true); false = detect only",
            "url": "str — for open/walk",
            "target": "str — for click (link text/href/index), submit (form index/id), extract (tag, #id, .class)",
            "kind": "str — for extract: headings|tables|forms|meta|nav (structured views, no selector needed)",
            "name": "str — for fill: the input name",
            "value": "str — for fill: the value",
            "session": "str (optional, default 'default') — named cookie session (persisted to disk)",
            "steps": "list|json — for task: [{act, ...}] multi-step program (open/fill/submit/extract/click/back/wait/stop)",
            "uploads": "dict (optional) — for submit: {field_name: local_file_path} multipart file upload",
            "max_chars": "int (optional) — cap for text/markdown output",
        },
    )
    def browser(
        action: str,
        url: str = "",
        target: str = "",
        name: str = "",
        value: str = "",
        kind: str = "",
        steps: Any = None,
        uploads: Any = None,
        session: str = "default",
        max_chars: int = 40000,
        solve: bool = True,
        **_: Any,
    ) -> dict[str, Any]:
        sess = get_session(session, user_agent=user_agent, timeout=timeout,
                           respect_robots=respect_robots, proxy_url=proxy_url,
                           session_dir=session_dir, max_task_steps=max_task_steps)
        try:
            return sess.do(action, url=url, target=target, name=name,
                           value=value, kind=kind, steps=steps, uploads=uploads,
                           max_chars=max_chars, solve=solve)
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001 - a browser error is a result
            raise ToolError(f"browser {action} failed: {classify(exc).message}") from exc

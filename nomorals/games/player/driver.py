"""Browser driver: fetch pages and act on interactive elements.

:class:`BrowserDriver` is the seam. :class:`HttpDriver` is the real
implementation over :class:`nomorals.core.http.HttpClient`: it GETs pages
and performs clicks (link follows) and form submissions (GET/POST) exactly
like the site's own markup asks. Anything else implementing the two-method
protocol (a CDP/Playwright driver, a test fake) drops in unchanged.
"""

from __future__ import annotations

import html as _html
import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from ...core.errors import ToolError
from ...core.http import HttpClient
from ...core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "Action",
    "BrowserDriver",
    "HttpDriver",
    "InteractiveElement",
    "Page",
    "extract_elements",
]


@dataclass
class Page:
    """One fetched page."""

    url: str
    status: int
    html: str


@dataclass
class InteractiveElement:
    """Something on the page the player can act on."""

    id: str
    kind: str  # "link" | "button" | "form"
    label: str
    url: str = ""  # for links
    form_action: str = ""
    form_method: str = "get"
    form_fields: dict[str, str] = field(default_factory=dict)


@dataclass
class Action:
    """A concrete thing to do on a page."""

    kind: str  # "goto" | "click" | "submit"
    target: str  # element id, or URL for goto
    label: str = ""
    payload: dict[str, str] = field(default_factory=dict)


class BrowserDriver(Protocol):
    """Two-method protocol every game driver implements."""

    def fetch(self, url: str) -> Page:
        """GET a page and return it."""
        ...

    def act(self, action: Action, page: Page) -> Page:
        """Perform ``action`` in the context of ``page``; return the new page."""
        ...


_TAG_RE = re.compile(r"(?s)<[^>]*>")
_WS_RE = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", _html.unescape(_TAG_RE.sub(" ", text or ""))).strip()


def extract_elements(html: str, base_url: str) -> list[InteractiveElement]:
    """Pull links, buttons and forms out of raw HTML.

    Labels come from visible text first, then ``value`` / ``aria-label`` /
    ``title`` / ``alt`` attributes. Elements without any usable label or
    target are skipped — an autopilot cannot meaningfully click "nothing".
    """
    elements: list[InteractiveElement] = []
    seen_targets: set[str] = set()

    def add(el: InteractiveElement) -> None:
        key = (el.kind, el.label.lower(), el.url or el.form_action)
        if key in seen_targets or not el.label:
            return
        seen_targets.add(key)
        el.id = f"e{len(elements)}"
        elements.append(el)

    # Links: <a href="..." ...>label</a>
    for m in re.finditer(
        r'(?is)<a\b([^>]*)>(.*?)</a>', html
    ):
        attrs, inner = m.group(1), m.group(2)
        href_m = re.search(r'''(?i)href\s*=\s*(['"])(.*?)\1''', attrs)
        if not href_m:
            continue
        href = href_m.group(2).strip()
        if not href or href.startswith(("#", "javascript:")):
            continue
        label = _clean(inner) or _clean(
            re.search(r'(?i)(?:aria-label|title)\s*=\s*([\'"])(.*?)\1', attrs).group(2)
            if re.search(r'(?i)(?:aria-label|title)\s*=\s*([\'"])', attrs)
            else ""
        )
        # image links: use alt text
        if not label:
            alt_m = re.search(r'(?i)<img\b[^>]*alt\s*=\s*([\'"])(.*?)\1', inner)
            if alt_m:
                label = _clean(alt_m.group(2))
        if not label:
            continue
        add(
            InteractiveElement(
                id="",
                kind="link",
                label=label,
                url=urllib.parse.urljoin(base_url, href),
            )
        )

    # Buttons: <button ...>label</button> and <input type=submit|button value="...">
    for m in re.finditer(r'(?is)<button\b([^>]*)>(.*?)</button>', html):
        attrs, inner = m.group(1), m.group(2)
        label = _clean(inner)
        if not label:
            v = re.search(r'''(?i)value\s*=\s*(['"])(.*?)\1''', attrs)
            label = _clean(v.group(2)) if v else ""
        if not label:
            continue
        add(InteractiveElement(id="", kind="button", label=label))

    for m in re.finditer(r'(?is)<input\b([^>]*)/?>', html):
        attrs = m.group(1)
        type_m = re.search(r'''(?i)type\s*=\s*(['"]?)(.*?)\1(\s|>|/)''', attrs)
        itype = (type_m.group(2) if type_m else "text").lower()
        if itype not in ("submit", "button", "image"):
            continue
        v = re.search(r'''(?i)value\s*=\s*(['"])(.*?)\1''', attrs)
        label = _clean(v.group(2)) if v else itype
        add(InteractiveElement(id="", kind="button", label=label))

    # Forms: capture action/method/hidden+text inputs so submits are real.
    for m in re.finditer(r'(?is)<form\b([^>]*)>(.*?)</form>', html):
        attrs, inner = m.group(1), m.group(2)
        a = re.search(r'''(?i)action\s*=\s*(['"])(.*?)\1''', attrs)
        action = urllib.parse.urljoin(base_url, a.group(2)) if a else base_url
        meth_m = re.search(r'''(?i)method\s*=\s*(['"]?)(get|post)\1''', attrs)
        method = (meth_m.group(2) if meth_m else "get").lower()
        fields: dict[str, str] = {}
        for im in re.finditer(r'(?is)<input\b([^>]*)/?>', inner):
            iattrs = im.group(1)
            n = re.search(r'''(?i)name\s*=\s*(['"])(.*?)\1''', iattrs)
            if not n:
                continue
            v = re.search(r'''(?i)value\s*=\s*(['"])(.*?)\1''', iattrs)
            fields[n.group(2)] = v.group(2) if v else ""
        # label: submit button inside, else form name/id
        label = ""
        sub = re.search(
            r'''(?is)<(?:button[^>]*>(.*?)</button|input[^>]*type\s*=\s*['"]?submit['"]?[^>]*value\s*=\s*['"](.*?)['"])''',
            inner,
        )
        if sub:
            label = _clean(sub.group(1) or sub.group(2) or "")
        if not label:
            n = re.search(r'''(?i)(?:name|id)\s*=\s*(['"])(.*?)\1''', attrs)
            label = _clean(n.group(2)).replace("_", " ") if n else "submit form"
        add(
            InteractiveElement(
                id="",
                kind="form",
                label=label,
                form_action=action,
                form_method=method,
                form_fields=fields,
            )
        )

    return elements


class HttpDriver:
    """Real driver over plain HTTP.

    Clicks on links become GETs; button clicks with no form context become
    no-ops resolved to their nearest form (handled by the executor); form
    submits become GET/POST with the form's fields. Cookies persist on the
    shared :class:`HttpClient`, so login sessions survive across moves.
    """

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        user_agent: str = "NoMoralsGamePlayer/0.1",
        client: HttpClient | None = None,
    ) -> None:
        self.client = client or HttpClient(timeout=timeout, user_agent=user_agent)

    def fetch(self, url: str) -> Page:
        response = self.client.get(url)
        if not response.ok:
            raise ToolError(f"fetch failed: {url} -> HTTP {response.status}")
        return Page(url=response.url or url, status=response.status, html=response.text)

    def act(self, action: Action, page: Page) -> Page:
        elements = {el.id: el for el in extract_elements(page.html, page.url)}
        if action.kind == "goto":
            return self.fetch(action.target)
        el = elements.get(action.target)
        if el is None:
            raise ToolError(f"unknown element {action.target!r} on {page.url}")
        if action.kind == "click" and el.kind == "link":
            return self.fetch(el.url)
        if action.kind == "submit" and el.kind == "form":
            fields = dict(el.form_fields)
            fields.update(action.payload)
            if el.form_method == "post":
                response = self.client.post_form(el.form_action, fields)
            else:
                response = self.client.get(el.form_action, params=fields)
            if not response.ok:
                raise ToolError(
                    f"form submit failed: {el.form_action} -> HTTP {response.status}"
                )
            return Page(
                url=response.url or el.form_action,
                status=response.status,
                html=response.text,
            )
        raise ToolError(
            f"cannot {action.kind} a {el.kind} element ({el.label!r}); "
            "use click for links, submit for forms"
        )

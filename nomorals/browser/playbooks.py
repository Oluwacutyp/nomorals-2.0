"""Site playbooks (layer 4): data-driven per-site browsing recipes.

One artifact, not code per site. A :class:`SitePlaybook` is a plain-data
descriptor — host match, login steps, search URL, content selectors,
pagination, rate limits, success criteria, fallback chain — that a generic
driver interprets. Adding a site means adding JSON, never a new module.

This mirrors the repo's existing data-driven patterns
(``nomorals.books.sources`` adapters, ``nomorals.media.film_sources``
chain, ``nomorals.media.download_kit`` extractor descriptors): selectors
and URL shapes live in data; the engine stays generic.

Validation is fail-fast: :meth:`SitePlaybook.validate` returns every
problem found (never just the first), and
:meth:`PlaybookRegistry.register` refuses an invalid playbook. A playbook
that passes validation is a complete recipe — a driver must never have to
guess a missing selector.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

__all__ = [
    "PlaybookError",
    "PlaybookLogin",
    "PlaybookSearch",
    "PlaybookContent",
    "PlaybookPagination",
    "PlaybookRate",
    "PlaybookSuccess",
    "PlaybookFallback",
    "SitePlaybook",
    "PlaybookRegistry",
    "PlaybookStore",
    "BUILTIN_PLAYBOOKS",
    "builtin_registry",
]

_log = get_logger(__name__)

#: ``{query}`` / ``{page}`` placeholders a driver substitutes.
_QUERY_RE = re.compile(r"\{query\}")
_PAGE_RE = re.compile(r"\{page\}")

#: Host pattern: letters/digits/dots/hyphens, optional leading ``*.``.
_HOST_RE = re.compile(r"^(?:\*\.)?[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$")


class PlaybookError(ValueError):
    """A playbook is malformed, or a store/registry operation failed."""


# ── section descriptors ────────────────────────────────────────────────────


@dataclass
class PlaybookLogin:
    """How to log in. Empty ``url`` = the site needs no login for this
    playbook. ``fields`` maps form-field names to value *keys* (never
    literal credentials — the driver resolves keys from the vault)."""

    url: str = ""
    fields: dict[str, str] = field(default_factory=dict)
    submit: str = ""
    success_marker: str = ""  # text/selector proving login worked

    def validate(self, prefix: str = "login") -> list[str]:
        errs: list[str] = []
        if self.url and not self.url.startswith(("http://", "https://", "/")):
            errs.append(f"{prefix}.url must be an http(s) URL or path, got {self.url!r}")
        if self.fields and not self.url:
            errs.append(f"{prefix}.fields given but {prefix}.url is empty")
        if self.submit and not self.url:
            errs.append(f"{prefix}.submit given but {prefix}.url is empty")
        return errs

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PlaybookLogin":
        raw = dict(raw or {})
        return cls(
            url=str(raw.get("url") or ""),
            fields={str(k): str(v) for k, v in (raw.get("fields") or {}).items()},
            submit=str(raw.get("submit") or ""),
            success_marker=str(raw.get("success_marker") or ""),
        )


@dataclass
class PlaybookSearch:
    """How to search. ``url_template`` must contain ``{query}``; the
    driver URL-encodes the query before substitution."""

    url_template: str = ""
    results_selector: str = ""
    title_selector: str = ""
    link_selector: str = ""

    def validate(self, prefix: str = "search") -> list[str]:
        errs: list[str] = []
        if not self.url_template:
            errs.append(f"{prefix}.url_template is required")
        else:
            if not self.url_template.startswith(("http://", "https://")):
                errs.append(f"{prefix}.url_template must be an http(s) URL, got {self.url_template!r}")
            if not _QUERY_RE.search(self.url_template):
                errs.append(f"{prefix}.url_template must contain a {{query}} placeholder")
        if not self.results_selector:
            errs.append(f"{prefix}.results_selector is required")
        return errs

    def build_url(self, query: str) -> str:
        """Substitute an URL-encoded query into the template."""
        import urllib.parse

        return self.url_template.replace("{query}", urllib.parse.quote_plus(query))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PlaybookSearch":
        raw = dict(raw or {})
        return cls(
            url_template=str(raw.get("url_template") or ""),
            results_selector=str(raw.get("results_selector") or ""),
            title_selector=str(raw.get("title_selector") or ""),
            link_selector=str(raw.get("link_selector") or ""),
        )


@dataclass
class PlaybookContent:
    """Where the content lives on a detail page."""

    title_selector: str = ""
    body_selector: str = ""
    meta_selectors: dict[str, str] = field(default_factory=dict)
    remove_selectors: list[str] = field(default_factory=list)

    def validate(self, prefix: str = "content") -> list[str]:
        errs: list[str] = []
        if not self.title_selector and not self.body_selector:
            errs.append(f"{prefix} needs at least one of title_selector / body_selector")
        return errs

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PlaybookContent":
        raw = dict(raw or {})
        return cls(
            title_selector=str(raw.get("title_selector") or ""),
            body_selector=str(raw.get("body_selector") or ""),
            meta_selectors={str(k): str(v) for k, v in (raw.get("meta_selectors") or {}).items()},
            remove_selectors=[str(s) for s in (raw.get("remove_selectors") or [])],
        )


@dataclass
class PlaybookPagination:
    """How to walk multi-page results. ``next_selector`` = link/button
    selector; ``url_template`` (with ``{page}``) is the alternative when
    pages are URL-addressable. Exactly one mechanism, or neither (single
    page)."""

    next_selector: str = ""
    url_template: str = ""
    max_pages: int = 1

    def validate(self, prefix: str = "pagination") -> list[str]:
        errs: list[str] = []
        if self.next_selector and self.url_template:
            errs.append(f"{prefix}: give next_selector OR url_template, not both")
        if self.url_template and not _PAGE_RE.search(self.url_template):
            errs.append(f"{prefix}.url_template must contain a {{page}} placeholder")
        if self.url_template and not self.url_template.startswith(("http://", "https://")):
            errs.append(f"{prefix}.url_template must be an http(s) URL, got {self.url_template!r}")
        if int(self.max_pages or 0) < 1:
            errs.append(f"{prefix}.max_pages must be >= 1, got {self.max_pages!r}")
        if (self.next_selector or self.url_template) and int(self.max_pages or 0) < 2:
            errs.append(f"{prefix}.max_pages must be >= 2 when pagination is configured")
        return errs

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PlaybookPagination":
        raw = dict(raw or {})
        try:
            max_pages = int(raw.get("max_pages") or 1)
        except (TypeError, ValueError):
            max_pages = 1
        return cls(
            next_selector=str(raw.get("next_selector") or ""),
            url_template=str(raw.get("url_template") or ""),
            max_pages=max_pages,
        )


@dataclass
class PlaybookRate:
    """Politeness budget. ``min_delay_ms`` is the floor between requests;
    ``max_requests_per_minute`` is the ceiling. Both must hold — a driver
    that honors only one is misconfigured."""

    min_delay_ms: int = 0
    max_requests_per_minute: int = 30
    burst: int = 1

    def validate(self, prefix: str = "rate") -> list[str]:
        errs: list[str] = []
        if int(self.min_delay_ms or 0) < 0:
            errs.append(f"{prefix}.min_delay_ms must be >= 0, got {self.min_delay_ms!r}")
        if int(self.max_requests_per_minute or 0) < 1:
            errs.append(f"{prefix}.max_requests_per_minute must be >= 1, got {self.max_requests_per_minute!r}")
        if int(self.burst or 0) < 1:
            errs.append(f"{prefix}.burst must be >= 1, got {self.burst!r}")
        return errs

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PlaybookRate":
        raw = dict(raw or {})
        def _int(key: str, default: int) -> int:
            try:
                return int(raw.get(key) if raw.get(key) is not None else default)
            except (TypeError, ValueError):
                return default
        return cls(
            min_delay_ms=_int("min_delay_ms", 0),
            max_requests_per_minute=_int("max_requests_per_minute", 30),
            burst=_int("burst", 1),
        )


@dataclass
class PlaybookSuccess:
    """What counts as a successful fetch. At least one signal is required
    — a playbook with no success criterion cannot tell a block page from
    content, and a driver must fail fast rather than guess."""

    required_selectors: list[str] = field(default_factory=list)
    required_text: list[str] = field(default_factory=list)
    forbidden_text: list[str] = field(default_factory=list)

    def validate(self, prefix: str = "success") -> list[str]:
        errs: list[str] = []
        if not self.required_selectors and not self.required_text:
            errs.append(f"{prefix} needs at least one required_selectors entry or required_text entry")
        return errs

    def check(self, *, selectors_present: list[str] | None = None,
              text: str = "") -> tuple[bool, list[str]]:
        """Evaluate success against observed evidence.

        Returns ``(ok, missing)``: ``missing`` names every required
        signal that was absent, plus any forbidden text that appeared.
        A block page fails here instead of being mistaken for content.
        """
        present = set(selectors_present or [])
        missing: list[str] = [s for s in self.required_selectors if s not in present]
        low = (text or "").lower()
        for needle in self.required_text:
            if needle.lower() not in low:
                missing.append(f"text:{needle}")
        for needle in self.forbidden_text:
            if needle.lower() in low:
                missing.append(f"forbidden-text:{needle}")
        return (not missing, missing)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PlaybookSuccess":
        raw = dict(raw or {})
        return cls(
            required_selectors=[str(s) for s in (raw.get("required_selectors") or [])],
            required_text=[str(s) for s in (raw.get("required_text") or [])],
            forbidden_text=[str(s) for s in (raw.get("forbidden_text") or [])],
        )


@dataclass
class PlaybookFallback:
    """What to do when the primary recipe fails. ``chain`` is an ordered
    list of strategies the driver tries in order; known names are
    ``retry``, ``rendered`` (playwright tab), ``proxy``, ``handoff``
    (owner takes over), ``abort``. ``abort`` must be last when present —
    nothing runs after an abort."""

    chain: list[str] = field(default_factory=list)
    max_retries: int = 1

    KNOWN = frozenset({"retry", "rendered", "proxy", "handoff", "abort"})

    def validate(self, prefix: str = "fallback") -> list[str]:
        errs: list[str] = []
        if int(self.max_retries or 0) < 0:
            errs.append(f"{prefix}.max_retries must be >= 0, got {self.max_retries!r}")
        for i, step in enumerate(self.chain):
            if step not in self.KNOWN:
                errs.append(f"{prefix}.chain[{i}] unknown strategy {step!r} (known: {sorted(self.KNOWN)})")
            if step == "abort" and i != len(self.chain) - 1:
                errs.append(f"{prefix}.chain: 'abort' must be last, found at index {i}")
        return errs

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PlaybookFallback":
        raw = dict(raw or {})
        try:
            max_retries = int(raw.get("max_retries") if raw.get("max_retries") is not None else 1)
        except (TypeError, ValueError):
            max_retries = 1
        return cls(
            chain=[str(s) for s in (raw.get("chain") or [])],
            max_retries=max_retries,
        )


# ── the playbook ─────────────────────────────────────────────────────────────


@dataclass
class SitePlaybook:
    """One site's browsing recipe, as pure data."""

    name: str = ""
    host: str = ""  # e.g. "example.com" or "*.example.com"
    description: str = ""
    login: PlaybookLogin = field(default_factory=PlaybookLogin)
    search: PlaybookSearch = field(default_factory=PlaybookSearch)
    content: PlaybookContent = field(default_factory=PlaybookContent)
    pagination: PlaybookPagination = field(default_factory=PlaybookPagination)
    rate: PlaybookRate = field(default_factory=PlaybookRate)
    success: PlaybookSuccess = field(default_factory=PlaybookSuccess)
    fallback: PlaybookFallback = field(default_factory=PlaybookFallback)
    verified: bool = False  # True only after a live run proved the selectors

    def validate(self) -> list[str]:
        """Every problem with this playbook, in one pass. [] = valid."""
        errs: list[str] = []
        if not (self.name or "").strip():
            errs.append("name is required")
        host = (self.host or "").strip().lower()
        if not host:
            errs.append("host is required")
        elif not _HOST_RE.match(host):
            errs.append(f"host {self.host!r} is not a valid hostname pattern (letters/digits/dots/hyphens, optional leading '*.')")
        elif host in {"*", "*.*"}:
            errs.append(f"host {self.host!r} matches everything — a playbook must name its site")
        errs += self.login.validate("login")
        errs += self.search.validate("search")
        errs += self.content.validate("content")
        errs += self.pagination.validate("pagination")
        errs += self.rate.validate("rate")
        errs += self.success.validate("success")
        errs += self.fallback.validate("fallback")
        return errs

    @property
    def is_valid(self) -> bool:
        return not self.validate()

    def matches_host(self, hostname: str) -> bool:
        """True when ``hostname`` falls under this playbook's host pattern.
        ``*.example.com`` matches subdomains AND the bare domain."""
        pattern = (self.host or "").strip().lower()
        host = (hostname or "").strip().lower().split(":")[0]
        if not pattern or not host:
            return False
        if pattern.startswith("*."):
            base = pattern[2:]
            return host == base or host.endswith("." + base)
        return host == pattern

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "host": self.host,
            "description": self.description,
            "login": self.login.to_dict(),
            "search": self.search.to_dict(),
            "content": self.content.to_dict(),
            "pagination": self.pagination.to_dict(),
            "rate": self.rate.to_dict(),
            "success": self.success.to_dict(),
            "fallback": self.fallback.to_dict(),
            "verified": bool(self.verified),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SitePlaybook":
        raw = dict(raw or {})
        return cls(
            name=str(raw.get("name") or ""),
            host=str(raw.get("host") or ""),
            description=str(raw.get("description") or ""),
            login=PlaybookLogin.from_dict(raw.get("login") or {}),
            search=PlaybookSearch.from_dict(raw.get("search") or {}),
            content=PlaybookContent.from_dict(raw.get("content") or {}),
            pagination=PlaybookPagination.from_dict(raw.get("pagination") or {}),
            rate=PlaybookRate.from_dict(raw.get("rate") or {}),
            success=PlaybookSuccess.from_dict(raw.get("success") or {}),
            fallback=PlaybookFallback.from_dict(raw.get("fallback") or {}),
            verified=bool(raw.get("verified") or False),
        )


# ── built-in example playbooks (data, not code) ──────────────────────────────


def _generic_search_playbook() -> SitePlaybook:
    return SitePlaybook(
        name="generic-search",
        host="*.example.com",
        description="Example: a site with a ?q= search page and result cards.",
        search=PlaybookSearch(
            url_template="https://example.com/search?q={query}",
            results_selector=".search-result",
            title_selector=".search-result h2",
            link_selector=".search-result a",
        ),
        content=PlaybookContent(
            title_selector="h1",
            body_selector="main article",
            remove_selectors=["script", "style", "nav", "footer"],
        ),
        pagination=PlaybookPagination(next_selector="a.next", max_pages=5),
        rate=PlaybookRate(min_delay_ms=500, max_requests_per_minute=30, burst=1),
        success=PlaybookSuccess(
            required_selectors=["main"],
            forbidden_text=["just a moment", "verify you are human"],
        ),
        fallback=PlaybookFallback(chain=["retry", "rendered", "handoff", "abort"]),
    )


def _generic_article_playbook() -> SitePlaybook:
    return SitePlaybook(
        name="generic-article",
        host="*.example.org",
        description="Example: a single-article page with no search or pagination.",
        search=PlaybookSearch(
            url_template="https://example.org/?s={query}",
            results_selector="article",
            title_selector="article h2",
            link_selector="article a",
        ),
        content=PlaybookContent(
            title_selector="article h1",
            body_selector="article .entry-content",
            meta_selectors={"author": ".byline", "date": "time"},
        ),
        pagination=PlaybookPagination(max_pages=1),
        rate=PlaybookRate(min_delay_ms=1000, max_requests_per_minute=20, burst=1),
        success=PlaybookSuccess(
            required_selectors=["article"],
            forbidden_text=["just a moment"],
        ),
        fallback=PlaybookFallback(chain=["retry", "rendered", "abort"]),
    )


def _generic_listing_playbook() -> SitePlaybook:
    return SitePlaybook(
        name="generic-listing",
        host="*.example.net",
        description="Example: a paginated listing (?page=N) with login-gated detail.",
        login=PlaybookLogin(
            url="https://example.net/login",
            fields={"username": "username", "password": "password"},
            submit="button[type=submit]",
            success_marker="Log out",
        ),
        search=PlaybookSearch(
            url_template="https://example.net/search?query={query}",
            results_selector=".listing-item",
            title_selector=".listing-item .title",
            link_selector=".listing-item a",
        ),
        content=PlaybookContent(
            title_selector="h1.title",
            body_selector=".detail-body",
            meta_selectors={"price": ".price", "posted": ".posted-date"},
        ),
        pagination=PlaybookPagination(
            url_template="https://example.net/list?page={page}", max_pages=10),
        rate=PlaybookRate(min_delay_ms=800, max_requests_per_minute=25, burst=1),
        success=PlaybookSuccess(
            required_selectors=[".listing-item, .detail-body"],
            required_text=[],
            forbidden_text=["sign in to continue"],
        ),
        fallback=PlaybookFallback(chain=["retry", "proxy", "handoff", "abort"], max_retries=2),
    )


BUILTIN_PLAYBOOKS: tuple[SitePlaybook, ...] = (
    _generic_search_playbook(),
    _generic_article_playbook(),
    _generic_listing_playbook(),
)


# ── registry ─────────────────────────────────────────────────────────────────


class PlaybookRegistry:
    """Named playbooks, keyed by name, matched by host."""

    def __init__(self, playbooks: list[SitePlaybook] | None = None) -> None:
        self._by_name: dict[str, SitePlaybook] = {}
        for pb in playbooks or []:
            self.register(pb)

    def register(self, playbook: SitePlaybook) -> None:
        """Add (or replace) a playbook. Refuses invalid ones — a broken
        recipe never enters the registry to fail at drive time."""
        errs = playbook.validate()
        if errs:
            raise PlaybookError(
                f"playbook {playbook.name!r} is invalid: " + "; ".join(errs))
        if playbook.name in self._by_name:
            _log.info("playbook %r replaced", playbook.name)
        self._by_name[playbook.name] = playbook

    def get(self, name: str) -> SitePlaybook | None:
        return self._by_name.get((name or "").strip())

    def remove(self, name: str) -> bool:
        return self._by_name.pop((name or "").strip(), None) is not None

    def names(self) -> list[str]:
        return sorted(self._by_name)

    def all(self) -> list[SitePlaybook]:
        return [self._by_name[n] for n in self.names()]

    def match_for_url(self, url: str) -> SitePlaybook | None:
        """Best playbook for a URL: longest host-pattern match wins, so a
        specific ``news.example.com`` beats a wildcard ``*.example.com``.
        None when nothing matches."""
        import urllib.parse

        try:
            host = (urllib.parse.urlparse((url or "").strip()).hostname or "").lower()
        except Exception:  # noqa: BLE001 - a bad URL matches nothing
            return None
        if not host:
            return None
        best: SitePlaybook | None = None
        best_len = -1
        for pb in self._by_name.values():
            if pb.matches_host(host) and len(pb.host) > best_len:
                best, best_len = pb, len(pb.host)
        return best

    def to_dict(self) -> dict[str, Any]:
        return {"version": 1, "playbooks": [pb.to_dict() for pb in self.all()]}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PlaybookRegistry":
        reg = cls()
        for item in (raw or {}).get("playbooks") or []:
            try:
                reg.register(SitePlaybook.from_dict(item))
            except PlaybookError as exc:
                # A stored file with one bad entry must not nuke the rest:
                # skip it loudly instead of failing the whole load.
                _log.warning("skipping invalid stored playbook: %s", exc)
        return reg


def builtin_registry() -> PlaybookRegistry:
    """A registry pre-loaded with the three example playbooks."""
    return PlaybookRegistry(list(BUILTIN_PLAYBOOKS))


# ── JSON store ───────────────────────────────────────────────────────────────


class PlaybookStore:
    """Atomic JSON persistence for a registry (tmp file + rename)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()

    def save(self, registry: PlaybookRegistry) -> Path:
        """Write the registry. Refuses to write an empty registry over a
        non-empty file — that shape is almost always an accidental wipe."""
        payload = registry.to_dict()
        if not payload["playbooks"] and self.path.is_file():
            try:
                existing = json.loads(self.path.read_text(encoding="utf-8"))
                if (existing or {}).get("playbooks"):
                    raise PlaybookError(
                        "refusing to overwrite a non-empty playbook store "
                        "with an empty registry")
            except PlaybookError:
                raise
            except Exception:  # noqa: BLE001 - unreadable file: overwrite is fine
                pass
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)
        return self.path

    def load(self) -> PlaybookRegistry:
        """Read the store. A missing file yields an empty registry (not an
        error); a corrupt file raises PlaybookError with the path."""
        if not self.path.is_file():
            return PlaybookRegistry()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PlaybookError(f"corrupt playbook store {self.path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise PlaybookError(f"corrupt playbook store {self.path}: top level must be an object")
        return PlaybookRegistry.from_dict(raw)

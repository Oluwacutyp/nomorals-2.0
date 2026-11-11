"""Unified media download orchestrator — a real fallback chain.

The old ``media_download`` path was single-shot: yt-dlp → direct HTTP,
and a failure died there.  This module turns every download into a
four-stage chain that keeps trying instead of erroring out:

0. **resolve** — a short link (vt.tiktok.com and the other shortener
   hosts) is expanded to its canonical URL before any stage runs:
   stages extract from the page the link MEANS, not the redirector.
   Expansion failure is a note, never a stopper.
1. **direct** — the current behavior (yt-dlp python API → CLI → direct
   HTTP), wrapped with retry + exponential backoff on transient
   (network) failures only.  A bot-block is not retried: retrying a
   bot check three times is just slower failure.  yt-dlp is handed the
   best cookies file for the URL — the owner's exported file, or
   Devon's own browser-profile cookies when she has logged in there.
1b. **kit** — the download kit (:mod:`nomorals.media.download_kit`),
   wired in by Pass #76: when the matched source descriptor (a view
   over the ONE source registry) says the URL *is* the file, the
   kit's resumable, validator-verified fetcher takes it.  The kit
   also guards the chain's edges: a disk preflight before any byte
   moves, an up-front refusal for sources the registry marks
   not-downloadable, quality-label → format translation, and an
   integrity-ledger entry for every landed download.
1c. **connector** — a source's own connector resolves the stream
   itself (the site descriptor names the connector id — today:
   SoundCloud, ``stream_url`` with the track's
   ``track_authorization`` forwarded), and the resolved stream URL
   is fetched as a first-class rung: progressive streams through
   the download kit, HLS playlists through the direct backend.
   A classified block (preview / DRM / region / login) is recorded
   with its class and the chain moves on — the connector's answer
   is evidence, never a dead end.  The chain a site runs, in
   order, comes from its site descriptor (B4); no descriptor
   means the canonical ladder above.
2. **proxy** — retry through the proxy lab (:class:`ProxyPoolConnector`
   via :meth:`healthy_proxies`, protocol-aware ordering).  A proxy that
   fails is *demoted* (``record_failure``) and the chain moves to the
   next one — a dead proxy never kills the download.
3a. **rendered** — the driven-player capture: the page opens in a
   REAL rendered tab on the browser service (her persistent cookie
   space), playback is started, and the media requests the player
   itself makes are harvested from the tab's network log, selected
   by the log's evidence (content type, extension, size), and
   fetched through the tab's own session so session-gated streams
   resolve.  This is the rung for JS players that expose no media
   URL to static scraping.  DRM-encrypted streams are recognised
   from the log's license traffic and reported as DRM — capture
   does not chase them.
3. **browser** — open the source page in a persistent
   :class:`BrowserSession` (cookies survive across downloads — the
   session is seeded from the same persisted browser profile the
   interactive browser tool uses, so a login earned there counts
   here), extract the real media URL (og:video/og:audio meta,
   ``<video>``/``<audio>``/``<source>`` tags, embedded JSON state —
   ``playAddr``/``downloadAddr``/``contentUrl``, which is what
   SPA players like TikTok actually serve — and download
   links/buttons, clicked when needed), then fetch the bytes through
   the session so the site's cookies and headers ride along.  This is
   the stage the user's complaint asked for: the download no longer
   dies when the direct fetch errors.
4. **honest failure** — a structured :class:`DownloadReport` naming every
   stage tried, what each returned, and the most likely cause.  Never a
   bare "download failed".

Every stage emits structured log entries (stage, URL, proxy used,
latency, bytes, error class).  Failures are classified with the shared
taxonomy :class:`FailureKind`: ``bot_block`` vs ``site_error`` vs
``network_failure`` vs ``no_media_found`` vs ``dependency_missing``.

Profile gating (never designing down): ``termux`` gets tighter budgets
(fewer proxies, fewer browser candidates, shorter stage timeouts);
``laptop``/``workstation`` get the full run.  The stages are the same
everywhere.

Pass #87 — the chain teaches the registry.  At the ``_succeed``
seam the facts that actually worked (winning stage, the extractor /
backend that landed the file, and — for the stages that receive a
format — the quality-label → format translation the descriptor
performed) are recorded as a LEARNED rule on the serving source.
A source with an active learned rule is consulted before the generic
chain on its next download: its learned stage is attempted first
and the learned descriptor (the registry's derived view) drives
format translation.  A failure through the learned path is recorded
against the rule (the registry stands it down / clears it on the
#86 health thresholds) and the download still falls back to the
generic chain in the SAME attempt.  A learned rule never bypasses
the descriptor gate, the disk preflight, or the integrity ledger.

``download()`` never raises — it returns a :class:`DownloadReport`.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.parse
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .cookies import BOT_COOKIE_HELP, is_bot_detection_error
from .downloader_failures import record_failure
from .site_descriptors import CHAIN_STAGES

_log = get_logger(__name__)

#: Connector factory injected by the composition root
#: (:func:`nomorals.agents.context.build_context`): ``factory(name, vault)``
#: — media is L4 and may not import connectors (L5). None means the proxy
#: pool is unavailable (the same outcome as the old import-failure path).
_CONNECTOR_FACTORY: Any = None


def set_connector_factory(factory: Any | None) -> None:
    """Inject (or clear) the connector factory used by the proxy pool."""
    global _CONNECTOR_FACTORY
    _CONNECTOR_FACTORY = factory


__all__ = [
    "FailureKind",
    "StageLog",
    "DownloadReport",
    "MediaDownloader",
    "classify_failure",
    "detect_profile",
    "set_connector_factory",
]


# ── failure taxonomy (shared with the browser work) ───────────────────


class FailureKind(str, Enum):
    """Why a download stage failed.  Shared with the browser stack so
    both sides report the same vocabulary."""

    OK = "ok"
    BOT_BLOCK = "bot_block"            # bot check / captcha / rate limit
    SITE_ERROR = "site_error"         # 404, private, DRM, login wall, 4xx
    NETWORK_FAILURE = "network_failure"  # timeout, reset, DNS, 5xx
    NO_MEDIA_FOUND = "no_media_found"  # page fetched but no media link
    DEPENDENCY_MISSING = "dependency_missing"  # yt-dlp absent, etc.


_DEPENDENCY_MARKERS = (
    "yt-dlp is not installed",
    "pip install yt-dlp",
    "not installed here",
)

_BOT_MARKERS = (
    "captcha",
    "rate limit",
    "rate-limit",
    "ratelimit",
    "429",
    "too many requests",
    "access denied",
    "cloudflare",
    "akamai",
    "datadome",
    "perimeterx",
    "unusual traffic",
    "automated access",
    "not a bot",
    "sign in to confirm",
    "confirm you're not a bot",
    "confirm you are not a bot",
)

_NO_MEDIA_MARKERS = (
    "produced no file",
    "empty file",
    "no media",
    "no video id",
    "no playable",
    "no direct media",
    "nothing to download",
)

_NETWORK_MARKERS = (
    "timed out",
    "timeout",
    "connection reset",
    "connection refused",
    "connection aborted",
    "name resolution",
    "dns",
    "temporary failure",
    "network is unreachable",
    "no route to host",
    "broken pipe",
    "socket",
    "5xx",
    " 500",
    " 502",
    " 503",
    " 504",
    "bad gateway",
    "service unavailable",
    "gateway timeout",
)


def classify_failure(exc: BaseException | str | None) -> FailureKind:
    """Map an exception/message to the shared failure taxonomy."""
    text = str(exc or "")
    low = text.lower()
    for marker in _DEPENDENCY_MARKERS:
        if marker in low:
            return FailureKind.DEPENDENCY_MISSING
    if is_bot_detection_error(text):
        return FailureKind.BOT_BLOCK
    for marker in _BOT_MARKERS:
        if marker in low:
            return FailureKind.BOT_BLOCK
    for marker in _NO_MEDIA_MARKERS:
        if marker in low:
            return FailureKind.NO_MEDIA_FOUND
    for marker in _NETWORK_MARKERS:
        if marker in low:
            return FailureKind.NETWORK_FAILURE
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return FailureKind.NETWORK_FAILURE
    return FailureKind.SITE_ERROR


def classify_http_status(status: int, body_hint: str = "") -> FailureKind:
    """Classify an HTTP status into the taxonomy."""
    if status in (403, 429) or classify_failure(body_hint) is FailureKind.BOT_BLOCK:
        return FailureKind.BOT_BLOCK
    if status in (401, 404, 410):
        return FailureKind.SITE_ERROR
    if 400 <= status < 500:
        return FailureKind.SITE_ERROR
    if 500 <= status < 600:
        return FailureKind.NETWORK_FAILURE
    return FailureKind.SITE_ERROR


# ── report types ──────────────────────────────────────────────────────


@dataclass
class StageLog:
    """One attempt inside one stage of the chain."""

    stage: str                  # direct | proxy | browser
    url: str = ""
    proxy: str = ""             # masked endpoint, e.g. https://1.2.3.4:8080
    attempt: int = 1
    latency_s: float = 0.0
    size_bytes: int = 0
    ok: bool = False
    error_class: str = ""       # FailureKind value, "" when ok
    error: str = ""
    note: str = ""


#: Sentinel for ``DownloadReport.login_block``: no login block
#: has been recorded on this report.
_LOGIN_BLOCK_NONE: Any = object()


@dataclass
class DownloadReport:
    """The outcome of ``MediaDownloader.download()`` — never raises."""

    ok: bool = False
    url: str = ""
    resolved_url: str = ""      # canonical URL when a short link expanded
    path: str = ""
    size_bytes: int = 0
    title: str = ""
    extractor: str = ""
    total_s: float = 0.0
    stages: list[StageLog] = field(default_factory=list)
    failure: FailureKind = FailureKind.OK
    likely_cause: str = ""
    hint: str = ""
    #: Archive outcome when the landed file was an archive container
    #: (WS1-P6 wire): the ``extract_for_downloader`` result dict —
    #: ``extracted``/``dest``/``files`` on a clean pass, or
    #: ``blocked: True`` + ``reason`` when the archive was refused
    #: (hostile members, unreadable, unextractable).  None when the
    #: download was not an archive at all.  The archive itself stays
    #: at ``path`` either way — extraction is additive, never a
    #: replacement (see ``MediaDownloader._unpack_archive``).
    archive: dict[str, Any] | None = None
    #: Structured block class when the failure is one the site
    #: itself classified (the resolver's vocabulary: preview / drm
    #: / region / login / private) — ``"login"`` when the
    #: authenticated rung (B5) failed the download for a missing
    #: session, or a login-gated track had no session to retry
    #: with and no later rung served it.  "" when the failure is
    #: unclassified — callers render the class, never a guessed
    #: sentence.
    block_reason: str = ""
    #: Pending login block (B5): the session outcome recorded
    #: when a connector reported a classified ``login`` block and
    #: no session resolved.  The chain CONTINUES — a later rung
    #: (the rendered tab runs with the profile's own cookies)
    #: may still serve the download — and only when the chain
    #: exhausts does the verdict become the LOGIN failure.  The
    #: stored value is the resolver's session outcome, or None
    #: when no resolver existed at all.  ``_LOGIN_BLOCK_NONE``
    #: when no login block was recorded.
    login_block: Any = _LOGIN_BLOCK_NONE

    def tried(self) -> list[str]:
        """Stage names attempted, in order, deduped."""
        out: list[str] = []
        for s in self.stages:
            if s.stage not in out:
                out.append(s.stage)
        return out

    def summary(self) -> str:
        """Human-readable failure report.  Never a bare 'download failed'."""
        if self.ok:
            head = (f"downloaded {self.title or self.url} "
                    f"({self.size_bytes} bytes) via "
                    f"{' → '.join(self.tried()) or 'direct'}")
            if self.archive is not None:
                if self.archive.get("blocked"):
                    head += (f"\n  archive: NOT extracted — "
                             f"{self.archive.get('reason') or 'refused'}")
                else:
                    head += (f"\n  archive: extracted "
                             f"{self.archive.get('extracted', 0)} file(s) "
                             f"to {self.archive.get('dest', '')}")
            return head
        from ..core.text import strip_ansi
        lines = [f"download failed: {self.url or '(no url)'}"]
        for i, s in enumerate(self.stages, 1):
            detail = f"{s.error_class}: {s.error}" if s.error else (
                s.note or "ok")
            detail = strip_ansi(detail)
            proxy = f" [proxy {s.proxy}]" if s.proxy else ""
            lines.append(
                f"  {i}. {s.stage}{proxy} — {detail[:160]} "
                f"({s.latency_s:.1f}s)")
        if self.likely_cause:
            lines.append(f"  likely cause: {self.likely_cause}")
        if self.hint:
            lines.append(f"  hint: {self.hint}")
        return "\n".join(lines)


# ── profile gating ────────────────────────────────────────────────────


#: Stage budgets per deployment profile.  Same stages everywhere —
#: termux just gets tighter budgets, never fewer stages.
_PROFILE_BUDGETS = {
    "termux": {"max_proxies": 2, "browser_candidates": 6,
               "stage_timeout_s": 300.0, "direct_retries": 2},
    "laptop": {"max_proxies": 3, "browser_candidates": 12,
               "stage_timeout_s": 900.0, "direct_retries": 3},
    "workstation": {"max_proxies": 5, "browser_candidates": 20,
                    "stage_timeout_s": 1800.0, "direct_retries": 3},
}


def detect_profile() -> str:
    """Deployment profile: termux | laptop | workstation.

    Termux is detected from the runtime (PEP 738: ``sys.platform ==
    'android'`` on Python 3.13+, or the Termux ``$PREFIX``).  Everything
    else defaults to ``laptop`` unless the machine looks beefy
    (``NM_PROFILE=workstation`` override, or 8+ CPUs).
    """
    override = os.environ.get("NM_PROFILE", "").strip().lower()
    if override in _PROFILE_BUDGETS:
        return override
    if sys.platform == "android":
        return "termux"
    prefix = os.environ.get("PREFIX", "")
    if prefix.startswith("/data/data/com.termux"):
        return "termux"
    try:
        if (os.cpu_count() or 0) >= 8:
            return "workstation"
    except Exception as e:
        record_failure("detect_profile", e)
    return "laptop"


_MEDIA_EXTS = (
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus", ".aac", ".wma",
    ".aiff", ".mid", ".midi", ".mp4", ".webm", ".mkv", ".mov", ".avi",
)

_DOWNLOAD_LINK_RE = re.compile(
    r"download|⬇|save|get\s+(the\s+|this\s+)?(mp3|audio|video|song|track|file)",
    re.IGNORECASE)

_SRC_TAG_RE = re.compile(
    r"<(?:video|audio|source)[^>]+src\s*=\s*[\"']([^\"']+)[\"']",
    re.IGNORECASE)


def _mask_proxy(proxy_url: str) -> str:
    """Proxy endpoint without credentials for logs."""
    try:
        p = urllib.parse.urlparse(proxy_url)
        return f"{p.scheme}://{p.hostname}:{p.port}"
    except Exception as e:
        record_failure("_mask_proxy", e)
        return "(proxy)"


def _err_text(exc: Any, limit: int = 400) -> str:
    """Stage-error text: the exception message with terminal escapes
    stripped (yt-dlp colorizes its errors; a StageLog is a chat-bound
    report, not a terminal)."""
    from ..core.text import strip_ansi
    return strip_ansi(str(exc or ""))[:limit]


# ── short links (stage zero) ─────────────────────────────────────────

#: Hosts whose URLs are redirectors, not pages: the link only MEANS
#: its destination.  Extraction attempted on the short URL itself
#: fails in instructive-looking but useless ways (the TikTok app-link
#: interstitial exposes no media at all), so these expand first.
_SHORTENER_HOSTS = frozenset({
    "vt.tiktok.com", "vm.tiktok.com", "tiktok.com",
    "bit.ly", "bitly.is", "t.co", "tinyurl.com", "goo.gl", "is.gd",
    "v.gd", "rb.gy", "buff.ly", "cutt.ly", "shorturl.at", "s.id",
    "ow.ly", "dlvr.it", "lnkd.in", "shorte.st", "bc.vc", "rebrand.ly",
    "bl.ink", "short.io", "trib.al", "yourls.org",
})


def is_short_link(url: str) -> bool:
    """True when *url*'s host is a link shortener (or a TikTok /t/
    share path, which is a short link wearing the main domain)."""
    try:
        parsed = urllib.parse.urlparse(str(url or ""))
        host = (parsed.hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        if host in _SHORTENER_HOSTS:
            # tiktok.com itself is a content host — only its /t/
            # share paths are short links.
            if host == "tiktok.com":
                return parsed.path.startswith("/t/")
            return True
        return False
    except Exception as e:
        record_failure("is_short_link", e)
        return False


def expand_short_url(url: str) -> str:
    """Follow a short link to its canonical URL.  Returns the final
    URL, or the input unchanged when expansion fails (dead shortener,
    no redirect) — expansion is step zero, never a new way to die.
    Never raises."""
    try:
        from ..core.http import HttpClient
        client = HttpClient(timeout=10.0)
        for method in ("head", "get"):
            try:
                response = getattr(client, method)(url)
            except Exception:  # noqa: BLE001 - try the other verb
                continue
            final = str(getattr(response, "url", "") or "")
            if final and final != url:
                return final
        return url
    except Exception as e:
        record_failure("expand_short_url", e)
        return url


# ── embedded-JSON media mining (browser stage) ───────────────────────

#: JSON keys under which SPA players (TikTok's rehydration state,
#: JSON-LD VideoObject, most embedded players) carry the actual media
#: URL.  og tags are the exception on such pages, not the rule.
_EMBEDDED_URL_KEYS = (
    "playAddr", "downloadAddr", "playUrl", "contentUrl", "videoUrl",
    "audioUrl", "sourceUrl",
)

_EMBEDDED_KV_RE = re.compile(
    r'"(' + "|".join(_EMBEDDED_URL_KEYS) + r')"\s*:\s*"((?:[^"\\]|\\.)*)"')

_LD_JSON_RE = re.compile(
    r'(?is)<script[^>]+type=["\']application/ld\+json["\'][^>]*>'
    r'(.*?)</script>')


def _embedded_media_urls(markup: str) -> list[str]:
    """Media URLs embedded in a page's JSON state: JSON-LD
    ``contentUrl`` plus ``playAddr``-style keys in hydration blobs.
    Values are JSON-unescaped (``\\/`` → ``/``).  Order-preserving,
    deduped, http(s) only.  Never raises."""
    out: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        if isinstance(value, str) and \
                value.startswith(("http://", "https://")) \
                and value not in seen:
            seen.add(value)
            out.append(value)

    if not markup:
        return out
    for block in _LD_JSON_RE.finditer(markup):
        try:
            data = json.loads(block.group(1))
        except Exception as e:  # a broken block is skipped
            record_failure("_embedded_media_urls", e)
            continue
        stack = [data]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                for key, value in node.items():
                    if key in _EMBEDDED_URL_KEYS:
                        add(value)
                    elif isinstance(value, (dict, list)):
                        stack.append(value)
            elif isinstance(node, list):
                stack.extend(node)
    for match in _EMBEDDED_KV_RE.finditer(markup):
        raw = match.group(2)
        try:
            add(json.loads(f'"{raw}"'))
        except Exception:  # noqa: BLE001 - hand-unescape the common case
            add(raw.replace("\\/", "/").replace("\\u0026", "&"))
    return out


def _accepts_progress(fn: Any) -> bool:
    """True when *fn* can take a ``progress`` keyword — the real
    kit downloader can; injected test doubles with the historical
    two-argument shape keep being called exactly as before."""
    try:
        import inspect
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if "progress" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD
               for p in params.values())


#: Archive container formats a landed download is unpacked for
#: (WS1-P6): mirrors ``nomorals.archives._CONTAINER_FORMATS`` — the
#: formats that hold members.  Single-file compression streams
#: (gzip/bzip2/xz of one file) are NOT unpacked: they are media
#: payloads in this pipeline, not bundles.
_ARCHIVE_CONTAINER_FORMATS = ("zip", "tar", "tar.gz", "tar.bz2",
                              "tar.xz", "7z", "rar")


class MediaDownloader:
    """Four-stage download orchestrator.  ``download()`` never raises."""

    def __init__(self, context: Any = None, *,
                 profile: str | None = None,
                 max_proxies: int | None = None,
                 direct_retries: int | None = None,
                 connector_factory: Any | None = None,
                 extractors: Any | None = None,
                 kit_downloader: Any | None = None,
                 ledger: Any | None = None,
                 cookie_store: Any | None = None,
                 source_registry: Any | None = None,
                 artifact_store: Any | None = None,
                 soundcloud_connector: Any | None = None,
                 site_descriptors: Any | None = None,
                 site_connectors: dict[str, Any] | None = None,
                 site_auth: Any | None = None,
                 rendered_tab_factory: Any | None = None) -> None:
        self.context = context
        self.profile = profile or detect_profile()
        self._connector_factory = connector_factory
        budget = _PROFILE_BUDGETS.get(self.profile, _PROFILE_BUDGETS["laptop"])
        self.max_proxies = max_proxies if max_proxies is not None \
            else budget["max_proxies"]
        self.direct_retries = direct_retries if direct_retries is not None \
            else budget["direct_retries"]
        self.browser_candidates = budget["browser_candidates"]
        self.stage_timeout = budget["stage_timeout_s"]
        # Download-kit wiring (Pass #76).  All three default to the
        # kit's real objects, built lazily so constructing a
        # MediaDownloader stays side-effect free; tests may inject.
        self._extractors = extractors        # → download_kit.DEFAULT_EXTRACTORS
        self._kit_downloader = kit_downloader  # → download_kit.ResumeDownloader
        self._ledger = ledger                # → download_kit.IntegrityLedger
        # Cloudflare gate vault (Pass #79): the kit's duck-typed
        # cookie store (get/set). An injected store wins; when none
        # is injected, _cf_cookie_store() lazily defaults to the
        # accounts-vault adapter (Pass #81) whenever this context
        # carries a database — None only where there is no vault
        # to back it, and there the gate only detects, as before.
        self._cookie_store = cookie_store
        # Source registry (Pass #86): the ONE registry (#76) this
        # downloader credits outcomes to, intakes discovery candidates
        # into, and derives its extractor view from.  Defaults to the
        # shared DEFAULT_REGISTRY; tests may inject their own.
        self._source_registry_injected = source_registry
        self._source_registry_cache: Any = None
        # Artifact store (WS4 P5): where a landed download registers
        # as a WS1 artifact so the exploration loop can cite the
        # media Devon actually fetched.  An injected store wins;
        # when none is injected, _artifact_store() lazily builds the
        # runtime's construction over the context's database — None
        # only where there is no database to register into.
        self._artifact_store_injected = artifact_store
        self._artifact_store_cache: Any = None
        # Connector stage (B2): an injected SoundCloud connector
        # wins; otherwise the stage builds one through the injected
        # connector factory + vault, exactly like the proxy pool.
        # B4: the stage is descriptor-driven — the site descriptor
        # names the connector id, and any connector id resolves
        # through the same path (injected ``site_connectors`` win,
        # then the legacy injected SoundCloud connector, then the
        # factory).  The per-id cache below is the old SoundCloud
        # cache generalized.
        self._soundcloud_connector_injected = soundcloud_connector
        self._site_connectors_injected: dict[str, Any] = dict(
            site_connectors or {})
        self._connector_cache: dict[str, Any] = {}
        self._connector_tried: set[str] = set()
        # Site descriptors (B4): the per-site download behaviour
        # as data.  An injected registry wins; otherwise the
        # default (SoundCloud first) is resolved lazily.
        self._site_descriptors_injected = site_descriptors
        self._site_descriptors_cache: Any = None
        # Authenticated rung (B5): the session resolver.  An
        # injected resolver wins (tests hand in one over a fake
        # vault store + a tmp estate); otherwise it is built
        # lazily over the context's database.
        self._site_auth_injected = site_auth
        self._site_auth_cache: Any = None
        # Rendered capture stage (B3): an injected tab factory wins
        # (tests hand in a fake tab); the production default opens a
        # real rendered tab on the browser service.  The factory
        # takes the page URL and returns a tab-shaped object
        # (network_requests / session_fetch_bytes / click / close).
        self._rendered_tab_factory = rendered_tab_factory

    # ── download-kit accessors (Pass #76) ─────────────────────────────

    def source_registry(self) -> Any:
        """The ONE source registry (Pass #86), bound to the context's
        KV store so its learned state (outcomes, health, grown
        sources, candidates) persists across restarts — the same
        ``KVStore(db)`` pattern ``core.trust.SourceTrust`` uses.
        Binding is lazy and best-effort: a context without a database
        keeps the registry memory-only, exactly as before."""
        if self._source_registry_cache is not None:
            return self._source_registry_cache
        if self._source_registry_injected is not None:
            self._source_registry_cache = self._source_registry_injected
            return self._source_registry_cache
        from .source_registry import DEFAULT_REGISTRY
        reg = DEFAULT_REGISTRY
        try:
            db = getattr(self.context, "db", None) \
                if self.context is not None else None
            if db is not None:
                from ..storage.kv import KVStore
                reg.bind_store(KVStore(db))
        except Exception as e:  # persistence is an
            # enhancement; its absence never sinks a download
            record_failure("source_registry", e)
        self._source_registry_cache = reg
        return reg

    def _extractor_registry(self) -> Any:
        """The extractor view over the ONE source registry.  Derived
        fresh from the registry (the PlayService pattern) so sources
        promoted by the health loop are visible without a restart."""
        if self._extractors is not None:
            return self._extractors
        from .download_kit import ExtractorRegistry
        return ExtractorRegistry(registry=self.source_registry())

    def _site_descriptor_registry(self) -> Any:
        """The per-site download descriptors (B4): an injected
        registry wins, otherwise the default — SoundCloud first.
        Descriptors are validated at load, so this accessor never
        raises on a bad descriptor (construction already did)."""
        if self._site_descriptors_cache is not None:
            return self._site_descriptors_cache
        if self._site_descriptors_injected is not None:
            self._site_descriptors_cache = \
                self._site_descriptors_injected
            return self._site_descriptors_cache
        from .site_descriptors import DEFAULT_SITE_DESCRIPTORS
        self._site_descriptors_cache = DEFAULT_SITE_DESCRIPTORS
        return self._site_descriptors_cache

    def _site_descriptor_for(self, url: str) -> Any | None:
        """The site descriptor claiming *url*; None when no site
        descriptor claims it — the engine then runs the canonical
        chain.  Best-effort: a registry failure is never a
        download failure."""
        try:
            return self._site_descriptor_registry().for_url(url)
        except Exception as e:
            record_failure("_site_descriptor_for", e)
            return None

    # ── authenticated rung (B5) ───────────────────────────────────

    def _site_auth_resolver(self) -> Any | None:
        """The authenticated rung's session resolver: an injected
        resolver wins; otherwise one is built lazily over the
        context's database.  None when there is no database for a
        vault / estate to open over — the gate then reports the
        vault unavailable, never a fabricated session."""
        if self._site_auth_cache is not None:
            return self._site_auth_cache
        if self._site_auth_injected is not None:
            self._site_auth_cache = self._site_auth_injected
            return self._site_auth_cache
        try:
            from .site_auth import SiteAuthResolver
            db = getattr(self.context, "db", None) \
                if self.context is not None else None
            self._site_auth_cache = SiteAuthResolver(
                db=db, context=self.context)
        except Exception as e:  # a resolver that cannot be built
            record_failure("_site_auth_resolver", e)  # is no resolver
            self._site_auth_cache = None
        return self._site_auth_cache

    def _resolve_site_auth(self, desc: Any, *,
                         require: bool = False) -> Any | None:
        """Resolve *desc*'s session through the B5 rung.  None
        when no resolver exists (no database) — the caller
        reports the vault unavailable, it never proceeds
        anonymously.  ``require`` forwards to the resolver: the
        login-block retry already knows a session is needed, so
        an ``auth == "none"`` descriptor must resolve a real
        session (or fail) instead of answering ok empty."""
        try:
            resolver = self._site_auth_resolver()
            if resolver is None:
                return None
            return resolver.resolve(desc, require=require)
        except Exception as e:  # resolution never breaks the chain
            record_failure("_resolve_site_auth", e)
            return None

    def _login_block_outcome(self, desc: Any, auth: Any | None
                             ) -> tuple[str, str]:
        """(status, owner action) for a login block that found no
        session.  ``auth`` None means no resolver existed at all.
        Never a value, never a guess at the credential itself."""
        if auth is None:
            return ("unavailable",
                    "the credential vault could not be opened — no "
                    "database is attached to this context; unlock "
                    "the vault and retry.")
        return (str(getattr(auth, "status", "") or "missing"),
                str(getattr(auth, "owner_action", "") or ""))

    def _login_block_note(self, report: DownloadReport, desc: Any,
                          auth: Any | None) -> None:
        """Append the auth stage log for a login block that found
        no session.  The log names the credential state, never a
        value.  No verdict fields are set here — the caller
        decides whether the LOGIN failure stands now (the
        up-front gate) or only if the chain exhausts (the
        login-block retry)."""
        site = str(getattr(desc, "site", "") or "")
        status, _ = self._login_block_outcome(desc, auth)
        report.stages.append(StageLog(
            stage="auth", url=report.url,
            error_class=FailureKind.SITE_ERROR.value,
            error=f"no {site} session stored ({status})",
            note="login"))
        self._log_stage("auth", report.url, "", 0.0, 0,
                        FailureKind.SITE_ERROR,
                        f"no {site} session stored ({status})",
                        note="login")

    def _login_failure(self, report: DownloadReport, desc: Any,
                       auth: Any | None, *, noted: bool = False
                       ) -> None:
        """Fill *report* as a found-but-blocked LOGIN failure: the
        site needs a signed-in account and no session resolved.
        ``auth`` None means no resolver existed at all.  The hint
        carries the exact owner action.  ``noted`` means the auth
        stage log was already appended (the deferred login-block
        path) — only the verdict fields are set."""
        site = str(getattr(desc, "site", "") or "")
        status, action = self._login_block_outcome(desc, auth)
        if not noted:
            self._login_block_note(report, desc, auth)
        report.extractor = site
        report.failure = FailureKind.SITE_ERROR
        report.block_reason = "login"
        report.likely_cause = (
            f"{site} wants a signed-in account for this content — "
            f"no session resolved ({status})")
        report.hint = action

    def _auth_retry_after_login_block(
            self, report: DownloadReport, url: str, target_dir: Path,
            audio_only: bool, format_spec: str, timeout: float,
            budget: dict[str, float]) -> bool:
        """The login-block retry (B5): the connector stage just
        reported a classified ``login`` block — the site itself
        says this content needs a signed-in account.  Resolve the
        session (estate, then vault — the resolver upserts it
        into the estate) and retry the direct stage once,
        authenticated through the estate's cookies file.  True
        when the retry landed.  False when no session resolved —
        but the block is recorded as PENDING (``login_block``),
        not a verdict: the chain continues, because a later rung
        (the rendered tab runs with the profile's own cookies)
        may still serve the download.  Only when the chain
        exhausts does the LOGIN failure stand (download()'s
        post-loop handling).  False with nothing recorded when
        the connector failure was not a login block at all."""
        last = report.stages[-1] if report.stages else None
        if last is None or last.stage != "connector" \
                or "block: login" not in (last.note or ""):
            return False
        desc = self._site_descriptor_for(url)
        if desc is None or not tuple(
                getattr(desc, "auth_domains", ()) or ()):
            return False
        auth = self._resolve_site_auth(desc, require=True)
        if auth is None or not getattr(auth, "ok", False):
            # No session: record the block as pending and let the
            # chain continue — the rendered / browser rungs run
            # with the profile's own cookies and may still serve
            # the download.  The LOGIN verdict is imposed only if
            # the chain exhausts.
            report.login_block = auth
            self._login_block_note(report, desc, auth)
            return False
        return self._stage_direct(report, url, target_dir, audio_only,
                                  format_spec, timeout, budget)

    # ── source growth: outcomes + discovery (Pass #86) ────────────────

    def _record_download_outcome(self, report: DownloadReport, *,
                                 ok: bool, error_class: str = "",
                                 latency_s: float = 0.0) -> None:
        """Credit/blame the source that served this attempt, through
        the registry's own outcome API — a source that keeps failing
        degrades in the registry's ranking, a source that serves is
        credited.  Bookkeeping is best-effort: it never turns a real
        download outcome into a different one."""
        try:
            reg = self.source_registry()
            url = report.resolved_url or report.url
            desc = reg.for_url(url)
            if desc is not None:
                reg.record_outcome(desc, ok, latency_s=latency_s,
                                   error_class=error_class)
        except Exception as e:  # bookkeeping never breaks downloads
            record_failure("_record_download_outcome", e)

    def _intake_candidate(self, url: str, *, via: str) -> None:
        """Discovery intake: a host encountered on the real path that
        the registry does not know is noted as an UNTRUSTED candidate
        (it serves nothing until the health loop verifies it)."""
        try:
            reg = self.source_registry()
            if reg.for_url(url) is None:
                reg.note_candidate(url, via=via)
        except Exception as e:  # intake never breaks downloads
            record_failure("_intake_candidate", e)

    def _intake_exploration(self, report: DownloadReport) -> None:
        """Exploration intake (WS4 P1): a video the owner actually
        landed notes its channel/title topics into the ExploreStore
        as UNPROMOTED watching candidates — his own consumption is a
        seed signal for the exploration loop, the content analogue
        of :meth:`_intake_candidate`.  Best-effort bookkeeping: it
        never turns a real download outcome into a different one."""
        try:
            from .explore import explore_for_context, note_owner_activity
            store, _engine, _beats = explore_for_context(self.context)
            note_owner_activity(
                store, title=report.title or "",
                url=report.resolved_url or report.url or "",
                via="download")
        except Exception as e:  # intake never breaks downloads
            record_failure("_intake_exploration", e)

    # ── artifact registration (WS4 P5) ─────────────────────────────

    def _artifact_store(self) -> Any | None:
        """The WS1 artifact store over the context's database — the
        runtime's construction (the context's database plus a
        BlobStore in a ``blobs`` directory beside the db file),
        built lazily and best-effort like :meth:`source_registry`:
        an injected store wins; a context without a database has
        nowhere to register and yields None.  The documents
        body-text extractor is NOT wired here (media may not import
        the documents engine's layer) — the store backfills document
        bodies lazily through any stronger instance's searches,
        exactly as its own contract states."""
        if self._artifact_store_cache is not None:
            return self._artifact_store_cache
        if self._artifact_store_injected is not None:
            self._artifact_store_cache = self._artifact_store_injected
            return self._artifact_store_cache
        try:
            db = getattr(self.context, "db", None) \
                if self.context is not None else None
            if db is None:
                return None
            from ..storage.artifacts import ArtifactStore
            from ..storage.blob import BlobStore
            db_path = getattr(db, "path", None)
            blob_dir = (Path(db_path).parent / "blobs" if db_path
                        else Path("data/blobs"))
            self._artifact_store_cache = ArtifactStore(
                db, BlobStore(db, blob_dir))
        except Exception as e:  # registration is bookkeeping; its
            # absence never sinks a download
            record_failure("_artifact_store", e)
            return None
        return self._artifact_store_cache

    def _register_download_artifact(self, report: DownloadReport) -> None:
        """Artifact registration (WS4 P5): a landed download becomes
        a WS1 artifact, so the exploration loop can cite the media
        Devon actually fetched — feed citations resolve through
        this store, and an unregistered download is invisible to it.

        Follows the store's producer idiom (the chat-inbound
        registration, WS1-P6): dedup is BY CONTENT within the
        download origin — the blob layer already dedups bytes, and
        re-downloading identical bytes must not mint a duplicate
        artifact row either; the first registration wins, and a row
        the same bytes earned under another origin does not block
        this origin's own record.  Only completed downloads
        register: this runs at the :meth:`_succeed` seam, which
        failed chains never reach.  Best-effort bookkeeping: a
        store failure is journaled via ``record_failure`` and never
        turns a real download outcome into a different one."""
        try:
            if not report.ok or not report.path:
                return
            path = Path(report.path)
            if not path.is_file():
                return
            store = self._artifact_store()
            if store is None:
                return
            digest = _sha256_of(path)
            if not digest:
                return
            for existing in store.find_by_content_hash(digest):
                if (existing.metadata or {}).get(
                        "origin") == "download":
                    return  # first registration wins
            import mimetypes
            filename = path.name
            mime = mimetypes.guess_type(filename)[0] or ""
            kind = "file"
            for prefix, label in (("video/", "video"),
                                  ("audio/", "audio"),
                                  ("image/", "image")):
                if mime.startswith(prefix):
                    kind = label
                    break
            winner = next(
                (s.stage for s in reversed(report.stages) if s.ok), "")
            metadata: dict[str, Any] = {
                "origin": "download",
                "filename": filename,
                "name": filename,
                "title": report.title or filename,
                "source_url": report.url,
                "source_path": str(path),
                "host": urllib.parse.urlsplit(report.url or "").netloc,
                "size_bytes": report.size_bytes,
                "sha256": digest,
                "extractor": report.extractor or "",
                "stage": winner,
                "ts": time.time(),
            }
            if report.resolved_url and \
                    report.resolved_url != report.url:
                metadata["resolved_url"] = report.resolved_url
            store.put_file(
                path, type=kind, creator="media_downloader",
                mime=mime, metadata=metadata,
                provenance={
                    "source_type": "tool",
                    "source_id": "media_downloader",
                    "source_timestamp": time.time(),
                })
            _log.info("download registered as artifact url=%s "
                      "sha256=%s", report.url, digest[:16])
        except Exception as e:  # bookkeeping never breaks downloads
            record_failure("_register_download_artifact", e)

    # ── rule learning (Pass #87) ──────────────────────────────────

    #: Stage → the registry's extraction-rule vocabulary, for the
    #: stages whose mechanism the stage itself defines.  The direct
    #: and proxy stages are refined by the backend that landed the
    #: file (see :meth:`_learn_from_success`).
    _STAGE_RULES = {"kit": "direct", "browser": "scrape",
                    "connector": "api", "rendered": "capture"}

    def _learned_for(self, url: str) -> Any | None:
        """(descriptor, learned rule) when the registry holds an
        ACTIVE learned rule that is *url*'s source's effective rule;
        None otherwise.  Best-effort — the chain proceeds exactly as
        before when there is nothing learned."""
        try:
            return self.source_registry().learned_for_url(url)
        except Exception as e:  # learning never gates downloads
            record_failure("_learned_for", e)
            return None

    def _note_rule_failure(self, desc: Any) -> None:
        """Record one failure through *desc*'s learned rule (the
        registry degrades / retires it on the #86 thresholds)."""
        try:
            self.source_registry().note_rule_failure(desc)
        except Exception as e:  # bookkeeping never breaks downloads
            record_failure("_note_rule_failure", e)

    def _learn_from_success(self, report: DownloadReport,
                            format_facts: dict[str, str] | None) -> None:
        """Teach the registry what just worked, at the same seam
        #86 hooked outcome recording (``_succeed``).

        Only facts actually exercised are recorded: the winning
        stage (the last stage log marked ok), the extractor/backend
        that landed the file, the extraction rule that stage's
        mechanism implies, and — only when the winning stage is one
        that receives a format (direct / proxy) — the quality→format
        translation that landed.  Sources the registry does not know
        (candidate-only hosts) learn nothing: there is no source to
        attach a rule to until the health loop promotes them."""
        try:
            winner = next((s for s in reversed(report.stages) if s.ok),
                          None)
            if winner is None or winner.stage not in (
                    "direct", "kit", "connector", "proxy", "rendered",
                    "browser"):
                return
            reg = self.source_registry()
            desc = reg.for_url(report.resolved_url or report.url)
            if desc is None:
                return
            extractor = report.extractor or ""
            if not extractor and "extractor=" in (winner.note or ""):
                extractor = winner.note.split("extractor=", 1)[1].strip()
            rule = self._STAGE_RULES.get(winner.stage, "")
            if winner.stage in ("direct", "proxy"):
                rule = "direct" if extractor == "direct" else "ytdlp"
            facts = format_facts or {}
            reg.learn_rule(
                desc, stage=winner.stage, extractor=extractor,
                extraction_rule=rule,
                quality_label=facts.get("quality_label", "")
                if winner.stage in ("direct", "proxy") else "",
                format_spec=facts.get("format_spec", "")
                if winner.stage in ("direct", "proxy") else "")
        except Exception as e:  # bookkeeping never breaks downloads
            record_failure("_learn_from_success", e)

    def _attempt_stage(self, stage: str, report: DownloadReport,
                       url: str, target_dir: Path, audio_only: bool,
                       format_spec: str, timeout: float,
                       budget: dict[str, float],
                       page_url: str = "") -> bool:
        """Dispatch one chain stage by name (the chain's real stage
        vocabulary).  Used for the canonical order and for the
        learned-path-first attempt (Pass #87)."""
        tracker = self._tracker()
        if tracker is not None:
            from ..core.progress_events import CONNECTING
            count = getattr(self, "_chain_stage_count", 0)
            if count > 0:
                # A route switch is a new attempt of this source: the
                # previous route's percentage never pretends to
                # describe the new transfer.
                tracker.begin_attempt(note=f"route: {stage}")
            self._chain_stage_count = count + 1
            tracker.set_stage(CONNECTING, note=f"via {stage}")
        if stage == "direct":
            return self._stage_direct(report, url, target_dir,
                                      audio_only, format_spec, timeout,
                                      budget)
        if stage == "kit":
            return self._stage_kit(report, url, target_dir, audio_only,
                                   budget)
        if stage == "connector":
            if self._stage_connector(report, url, target_dir,
                                     audio_only, budget):
                return True
            # A classified login block from the connector is the
            # site itself saying this content needs a signed-in
            # account — the authenticated rung retries the direct
            # stage once with the resolved session, or records
            # the LOGIN failure and stops the chain.
            return self._auth_retry_after_login_block(
                report, url, target_dir, audio_only, format_spec,
                timeout, budget)
        if stage == "proxy":
            return self._stage_proxy(report, url, target_dir, audio_only,
                                     format_spec, timeout, budget)
        if stage == "rendered":
            return self._stage_rendered(report, url, target_dir,
                                        audio_only, page_url or url,
                                        budget)
        if stage == "browser":
            return self._stage_browser(report, url, target_dir,
                                       audio_only, page_url or url,
                                       budget)
        return False

    def _descriptor_for(self, url: str) -> Any | None:
        """Best extractor descriptor for *url*; None when the
        registry view itself is unavailable (chain proceeds as
        before — descriptors advise, they never gate by absence)."""
        try:
            return self._extractor_registry().best_for(url)
        except Exception as e:
            record_failure("_descriptor_for", e)
            return None

    def _cf_cookie_store(self) -> Any:
        """The Cloudflare gate's cookie store.

        Pass #81: an explicitly injected store wins; otherwise the
        production default is the accounts-vault adapter, built
        lazily (like the other kit accessors — construction stays
        side-effect free, and the vault itself opens only when the
        gate actually consults the store). Contexts without a
        database keep the old truth: no store, the gate detects
        and asks only."""
        if self._cookie_store is None:
            try:
                from ..accounts.cf_cookie_store import (
                    default_cf_cookie_store)
                self._cookie_store = default_cf_cookie_store(
                    self.context, actor="downloader-cloudflare")
            except Exception as e:  # store is an
                # enhancement; its absence never sinks a download
                record_failure("_cf_cookie_store", e)
                self._cookie_store = None
        return self._cookie_store

    def _kit(self) -> Any:
        if self._kit_downloader is None:
            from .download_kit import ResumeDownloader
            self._kit_downloader = ResumeDownloader()
        return self._kit_downloader

    def _integrity_ledger(self) -> Any:
        if self._ledger is None:
            from .download_kit import IntegrityLedger
            self._ledger = IntegrityLedger()
        return self._ledger

    def _succeed(self, report: DownloadReport,
                 started: float,
                 format_facts: dict[str, str] | None = None) -> DownloadReport:
        """Record a landed download in the kit's integrity ledger,
        teach the registry the rule that worked (Pass #87), then
        finish.  Bookkeeping is best-effort — a ledger failure never
        turns a real download into a reported failure."""
        tracker = self._tracker()
        if tracker is not None:
            from ..core.progress_events import VERIFYING
            tracker.set_stage(VERIFYING, note="checking the landed file")
        self._record_download_outcome(
            report, ok=True, latency_s=time.perf_counter() - started)
        self._learn_from_success(report, format_facts)
        self._intake_exploration(report)
        self._register_download_artifact(report)
        try:
            if report.ok and report.path:
                digest = _sha256_of(Path(report.path))
                if digest:
                    winner = next(
                        (s.stage for s in reversed(report.stages) if s.ok),
                        "")
                    self._integrity_ledger().append(
                        url=report.url, sha256=digest,
                        size_bytes=report.size_bytes,
                        source=winner or report.extractor or "downloader",
                        dest=report.path)
        except Exception as e:  # bookkeeping never breaks downloads
            record_failure("_succeed", e)
        self._unpack_archive(report)
        return self._finish(report, started)

    def _unpack_archive(self, report: DownloadReport) -> None:
        """Unpack a landed download that IS an archive (WS1-P6).

        The wire ``archives.extract_for_downloader`` was built for:
        identify by magic, pre-flight EVERY member name, extract
        only when all are safe — a downloaded archive with one
        traversal member is refused whole, never partially
        extracted.  The extraction lands beside the archive in
        ``<name>.extracted/`` under the Archivist's own bomb budgets
        (20k files / 2 GB cumulative, nesting included).

        The original archive is RETAINED at ``report.path``:
        the downloader's contract is that the landed file is the
        deliverable (the integrity ledger above hashes it, the
        orchestrator's acceptance checks read it) — extraction is
        an additive digest of it, not a replacement.  The outcome
        rides ``report.archive`` either way: extracted (with the
        file list) or blocked (with the reason named).  A failure
        here never turns a landed download into a reported
        failure — but it is never silent either: it is recorded,
        logged, and printed by ``DownloadReport.summary()``.

        One honest limit: the Archivist resolves paths inside the
        workspace sandbox, so a download a caller deliberately
        landed OUTSIDE the workspace gets a blocked verdict naming
        the escape instead of an extraction.
        """
        if not report.ok or not report.path:
            return
        try:
            from ..archives import (
                Archivist,
                detect_format,
                extract_for_downloader,
            )
        except Exception as e:  # the wire never breaks a download
            record_failure("download.archive_import", e)
            return
        try:
            fmt = detect_format(report.path)
        except Exception as e:  # noqa: BLE001 — sniff failure: not an archive
            record_failure("download.archive_detect", e)
            return
        if fmt not in _ARCHIVE_CONTAINER_FORMATS:
            return
        try:
            result = extract_for_downloader(
                Archivist(self.context), report.path)
        except Exception as e:  # noqa: BLE001 — named on the report
            record_failure("download.archive_extract", e)
            result = {"path": report.path, "format": fmt,
                      "extracted": 0, "blocked": True,
                      "reason": f"extraction failed: "
                                f"{type(e).__name__}: {e}"}
        report.archive = result
        if result.get("blocked"):
            _log.warning("download archive NOT extracted %s: %s",
                         report.path, result.get("reason") or "refused")
        else:
            _log.info("download archive extracted %s → %s (%s files)",
                      report.path, result.get("dest", ""),
                      result.get("extracted", 0))

    # ── public API ────────────────────────────────────────────────────

    def download(self, url: str, *, dest_dir: str | Path | None = None,
                 audio_only: bool = False,
                 format_spec: str = "bestvideo*+bestaudio/best",
                 page_url: str = "", timeout: float = 1800.0,
                 title_hint: str = "") -> DownloadReport:
        """Run the fallback chain.  Never raises — returns a report."""
        started = time.perf_counter()
        report = DownloadReport(url=url or "")
        if not (url or "").strip():
            report.failure = FailureKind.SITE_ERROR
            report.likely_cause = "no URL to download"
            return report
        target_dir = self._dest_dir(dest_dir)
        budget = {"deadline": started + min(timeout, self.stage_timeout * 3)}

        # Measured progress (P1): the resolving stage covers link
        # expansion, the descriptor gate and preflight.  When no
        # caller already opened a source for this operation (the chat
        # fall-through loop does, per candidate), this download IS the
        # first source.  This layer never marks delivery — retrieval
        # complete is not delivered.
        self._chain_stage_count = 0
        tracker = self._tracker()
        if tracker is not None:
            from ..core.progress_events import RESOLVING
            if tracker.source_index == 0:
                tracker.begin_source("")
            tracker.set_stage(RESOLVING, note="resolving the source")

        _log.info("download start url=%s profile=%s audio_only=%s",
                  url, self.profile, audio_only)
        # Kit preflight (Pass #76): no point expanding links or
        # fetching a byte when the destination cannot hold anything.
        # Silent when there is room — the report gains a stage only
        # when the preflight is what stops the download.
        try:
            from .download_kit import disk_preflight
            pre = disk_preflight(0, target_dir)
            if not pre.ok:
                report.stages.append(StageLog(
                    stage="preflight", url=url,
                    error_class=FailureKind.SITE_ERROR.value,
                    error=pre.message))
                report.failure = FailureKind.SITE_ERROR
                report.likely_cause = pre.message
                report.hint = ("free disk space or choose another "
                               "destination folder")
                return self._finish(report, started)
        except Exception as e:  # preflight never breaks the chain
            record_failure("download.preflight", e)
        # Stage zero: a short link expands to the page it MEANS before
        # any extractor sees it.  Failure to expand is a note in the
        # report — the chain continues with the original URL.
        work_url = url
        if is_short_link(url):
            resolved = expand_short_url(url)
            if resolved and resolved != url:
                report.resolved_url = resolved
                report.stages.append(StageLog(
                    stage="resolve", url=url, ok=True,
                    note=f"short link expanded → {resolved}"))
                _log.info("download stage=resolve url=%s resolved=%s",
                          url, resolved)
                work_url = resolved
            else:
                report.stages.append(StageLog(
                    stage="resolve", url=url,
                    note="short link could not be expanded; continuing "
                         "with the original URL"))
        # Discovery intake (Pass #86): hosts the registry does not
        # know — the download URL's own host, and a short link's
        # redirect target — are noted as untrusted candidates.  A
        # candidate changes nothing about THIS download (the owner's
        # explicit request flows through the generic fallbacks as
        # always); it only enters the registry's growth pipeline.
        self._intake_candidate(work_url, via="download-url")
        if report.resolved_url:
            self._intake_candidate(report.resolved_url, via="redirect")
        # Descriptor truth (Pass #76): the matched source descriptor —
        # a view over the ONE source registry — speaks before any
        # fetch.  A source that declares itself not downloadable
        # (Spotify, Boomplay) is refused here with its own reason
        # instead of burning every stage to learn the same thing; a
        # caller's quality LABEL is translated to the descriptor's
        # yt-dlp format for this source.
        # The quality→format translation that lands, tracked as
        # evidence (Pass #87): when the matched descriptor translates
        # the caller's label, (label, format) is the pair a success
        # will teach; an audio-only request lands bestaudio through
        # the direct stage's own override, which is the translation
        # of record there.
        format_facts = {"quality_label": "", "format_spec": format_spec}
        if audio_only:
            format_facts = {"quality_label": "audio",
                            "format_spec": "bestaudio/best"}
        descriptor = self._descriptor_for(work_url)
        if descriptor is not None:
            if format_spec:
                mapped = ""
                try:
                    mapped = descriptor.format_for(format_spec)
                except Exception as e:
                    record_failure("download.mapped", e)
                    mapped = ""
                if mapped and mapped != format_spec:
                    _log.info("download format label %r → %r via "
                              "descriptor %s", format_spec, mapped,
                              descriptor.name)
                    if not audio_only:
                        format_facts = {
                            "quality_label": format_spec,
                            "format_spec": mapped}
                    format_spec = mapped
            if not descriptor.downloadable:
                report.extractor = descriptor.name
                report.stages.append(StageLog(
                    stage="extractor", url=work_url,
                    error_class=FailureKind.SITE_ERROR.value,
                    error=f"{descriptor.name}: {descriptor.notes}",
                    note="source registry marks this source "
                         "not downloadable"))
                report.failure = FailureKind.SITE_ERROR
                report.likely_cause = (
                    f"{descriptor.name} does not serve downloads — "
                    f"{descriptor.notes}")
                report.hint = ("find the same title on a downloadable "
                               "source, or play it in the source's own app")
                return self._finish(report, started)
        # Learned rule first (Pass #87): when this source has an
        # active learned rule whose stage is not the chain's head,
        # that stage is attempted BEFORE the generic order — the
        # rule was earned by a real landing.  Its failure is recorded
        # against the rule, the stage is not retried below, and the
        # generic chain still runs in the SAME attempt (no regression
        # vs the unlearned chain).  The descriptor gate and the disk
        # preflight above already ran — a learned rule bypasses
        # neither, and the integrity ledger still records the landing
        # in ``_succeed``.
        learned_pair = self._learned_for(work_url)
        rule_failure_recorded = False
        skip_stages: set[str] = set()
        # Per-site chain (B4): a site descriptor names the rungs
        # that apply to its URLs, in order; no descriptor means the
        # canonical ladder.  A learned rule's stage is only tried
        # first when the site's chain actually runs it.
        site_desc = self._site_descriptor_for(work_url)
        chain = tuple(site_desc.chain) if site_desc is not None \
            else CHAIN_STAGES
        # A site that declares itself not downloadable is refused
        # here with its own reason (the same shape as the
        # extractor-descriptor gate above) instead of burning every
        # stage to learn the same thing.
        if site_desc is not None and not site_desc.downloadable:
            report.extractor = site_desc.site
            report.stages.append(StageLog(
                stage="extractor", url=work_url,
                error_class=FailureKind.SITE_ERROR.value,
                error=f"{site_desc.site}: "
                      f"{site_desc.notes or 'does not serve downloads'}",
                note="site descriptor marks this site "
                     "not downloadable"))
            report.failure = FailureKind.SITE_ERROR
            report.likely_cause = (
                f"{site_desc.site} does not serve downloads — "
                f"{site_desc.notes or 'the site descriptor says so'}")
            report.hint = ("find the same title on a downloadable "
                           "source, or play it in the source's own app")
            return self._finish(report, started)
        # Authenticated rung, up-front gate (B5): a descriptor that
        # declares auth resolves its session BEFORE the chain — a
        # missing / expired / unavailable session fails here as
        # found-but-blocked, class ``login``, and the chain never
        # runs anonymously for a site that declared it needs an
        # account.  A learned rule bypasses neither this gate nor
        # the descriptor gate above.
        if site_desc is not None and site_desc.auth_needed:
            auth = self._resolve_site_auth(site_desc)
            if auth is None or not getattr(auth, "ok", False):
                self._login_failure(report, site_desc, auth)
                return self._finish(report, started)
        try:
            if learned_pair is not None:
                learned_desc, learned_rule = learned_pair
                if learned_rule.stage in ("kit", "connector", "proxy",
                                          "rendered", "browser") \
                        and learned_rule.stage in chain:
                    _log.info(
                        "download learned rule host=%s stage=%s "
                        "rule=%s — trying the learned path first",
                        learned_desc.domain, learned_rule.stage,
                        learned_rule.extraction_rule)
                    if self._attempt_stage(
                            learned_rule.stage, report, work_url,
                            target_dir, audio_only, format_spec,
                            timeout, budget, page_url):
                        return self._succeed(report, started,
                                             format_facts)
                    self._note_rule_failure(learned_desc)
                    rule_failure_recorded = True
                    skip_stages.add(learned_rule.stage)
            for stage_name in chain:
                if report.block_reason:
                    # A classified login block already set the
                    # LOGIN failure fields (up-front gate or the
                    # connector retry) — the chain stops here and
                    # never burns another anonymous rung.
                    break
                if stage_name in skip_stages:
                    continue
                if self._attempt_stage(
                        stage_name, report, work_url, target_dir,
                        audio_only, format_spec, timeout, budget,
                        page_url):
                    return self._succeed(report, started, format_facts)
        except Exception as exc:  # noqa: BLE001 — the chain never raises
            _log.warning("downloader crashed mid-chain on %s: %s",
                         url, exc)
            report.stages.append(StageLog(
                stage="chain", url=url, error_class="site_error",
                error=f"orchestrator error: {_err_text(exc, 380)}"))
        if report.block_reason:
            # The LOGIN failure fields are already set by the auth
            # path.  No outcome recording — a missing account is
            # not the source's failure, exactly like the early
            # returns above (disk preflight, a registry-marked
            # not-downloadable source).
            return self._finish(report, started)
        if report.login_block is not _LOGIN_BLOCK_NONE:
            # A connector login block found no session, and no
            # later rung served the download — the LOGIN verdict
            # stands now.  The auth stage log was already appended
            # when the block was recorded, so only the verdict
            # fields are set (noted=True).  No outcome recording
            # — a missing account is not the source's failure.
            desc = self._site_descriptor_for(work_url)
            if desc is not None:
                self._login_failure(report, desc, report.login_block,
                                    noted=True)
                return self._finish(report, started)
        self._finish_failure(report)
        # The chain exhausted: the attempt's failure (with its error
        # class) is recorded against the source that was asked to
        # serve it.  The early returns above (disk preflight, a
        # registry-marked not-downloadable source) deliberately do
        # NOT record — neither is the source's failure.
        self._record_download_outcome(
            report, ok=False, error_class=report.failure.value,
            latency_s=time.perf_counter() - started)
        # A total failure through a chain the learned rule governed
        # (its stage ran in canonical position, or the learned-first
        # attempt already recorded it — never twice) degrades the
        # rule on the registry's health thresholds.
        if learned_pair is not None and not rule_failure_recorded:
            self._note_rule_failure(learned_pair[0])
        return self._finish(report, started)

    # ── stage 1: direct ───────────────────────────────────────────────

    def _stage_direct(self, report: DownloadReport, url: str,
                      target_dir: Path, audio_only: bool,
                      format_spec: str, timeout: float,
                      budget: dict[str, float]) -> bool:
        """Current behavior, with retry + backoff on network failures."""
        from ..tools.media import download as direct_download

        tracker = self._tracker()
        attempt = 0
        while attempt < max(1, self.direct_retries):
            attempt += 1
            if attempt > 1 and tracker is not None:
                tracker.begin_attempt(note="retrying the direct route")
            if time.perf_counter() > budget["deadline"]:
                break
            t0 = time.perf_counter()
            try:
                result = direct_download(
                    url, target_dir, format_spec=format_spec,
                    audio_only=audio_only, timeout=timeout,
                    cookies_file=self._cookies_file(url),
                    progress=self._job_progress("direct"),
                    **self._progress_kwargs(tracker))
            except Exception as exc:  # noqa: BLE001
                kind = classify_failure(exc)
                lat = time.perf_counter() - t0
                report.stages.append(StageLog(
                    stage="direct", url=url, attempt=attempt,
                    latency_s=lat, error_class=kind.value,
                    error=_err_text(exc)))
                self._log_stage("direct", url, "", lat, 0, kind,
                                _err_text(exc, 200), attempt)
                if kind is FailureKind.NETWORK_FAILURE \
                        and attempt < self.direct_retries:
                    backoff = 2.0 ** (attempt - 1)
                    _log.info("direct: network failure, retrying in %.0fs "
                              "(attempt %d/%d)", backoff, attempt + 1,
                              self.direct_retries)
                    time.sleep(backoff)
                    continue
                return False
            lat = time.perf_counter() - t0
            landed = self._verify_landed(result)
            if landed is None:
                # The stage CLAIMS success — but a claim is not a
                # file.  Nothing (or an empty stub) is on disk, so
                # this stage failed and the chain moves on.
                report.stages.append(StageLog(
                    stage="direct", url=url, attempt=attempt, latency_s=lat,
                    error_class=FailureKind.NO_MEDIA_FOUND.value,
                    error="download reported success but the file is "
                          "missing or empty on disk "
                          f"({result.get('path', '') or 'no path'})"))
                self._log_stage("direct", url, "", lat, 0,
                                FailureKind.NO_MEDIA_FOUND,
                                "reported success, nothing on disk",
                                attempt)
                return False
            _path, size = landed
            report.stages.append(StageLog(
                stage="direct", url=url, attempt=attempt, latency_s=lat,
                size_bytes=size, ok=True,
                note=f"extractor={result.get('extractor', '')}"))
            self._log_stage("direct", url, "", lat, size, FailureKind.OK,
                            "", attempt)
            self._fill_success(report, result, size)
            return True
        return False

    # ── stage 1b: download kit (Pass #76) ─────────────────────────────

    def _stage_kit(self, report: DownloadReport, url: str,
                   target_dir: Path, audio_only: bool,
                   budget: dict[str, float]) -> bool:
        """The download kit's resumable fetch, in the chain.

        Engages only when the matched source descriptor says the URL
        IS the file (extraction rule ``direct`` — a direct media
        link): yt-dlp has no extractor work to do there, and the
        kit's :class:`ResumeDownloader` is the stronger fetcher —
        validator-verified resume across attempts plus a sha256 of
        what landed.  Every other rule keeps flowing to proxy /
        browser exactly as before.  A kit failure is a stage log and
        the chain moves on — never a swallowed success, never the
        end of the chain."""
        descriptor = self._descriptor_for(url)
        if descriptor is None or descriptor.extraction_rule != "direct":
            return False
        if time.perf_counter() > budget["deadline"]:
            return False
        dest = target_dir / _safe_name(url, audio_only)
        t0 = time.perf_counter()
        try:
            kit = self._kit()
            tracker = self._tracker()
            if tracker is not None and _accepts_progress(kit.download):
                result = kit.download(
                    url, dest,
                    progress=lambda received, expected:
                        tracker.report_bytes(received, expected))
            else:
                result = kit.download(url, dest)
        except Exception as exc:  # noqa: BLE001 - a stage reports, never raises
            result = None
            error = _err_text(exc)
        else:
            error = "" if result.ok else (result.error or "kit fetch failed")
        lat = time.perf_counter() - t0
        if result is None or not result.ok:
            kind = classify_failure(error)
            report.stages.append(StageLog(
                stage="kit", url=url, latency_s=lat,
                error_class=kind.value, error=error))
            self._log_stage("kit", url, "", lat, 0, kind,
                            _err_text(error, 200))
            return False
        landed = self._verify_landed(
            {"path": result.path, "bytes": result.size_bytes})
        if landed is None:
            # Same phantom guard as the direct stage: the kit CLAIMS
            # bytes — the disk decides.
            report.stages.append(StageLog(
                stage="kit", url=url, latency_s=lat,
                error_class=FailureKind.NO_MEDIA_FOUND.value,
                error="kit reported success but the file is missing "
                      f"or empty on disk ({result.path or 'no path'})"))
            self._log_stage("kit", url, "", lat, 0,
                            FailureKind.NO_MEDIA_FOUND,
                            "reported success, nothing on disk")
            return False
        _path, size = landed
        report.stages.append(StageLog(
            stage="kit", url=url, latency_s=lat, size_bytes=size,
            ok=True,
            note=f"download-kit resumable fetch "
                 f"(resumed={result.resumed} restarted={result.restarted})"))
        self._log_stage("kit", url, "", lat, size, FailureKind.OK, "")
        self._fill_success(report, {
            "path": _path, "bytes": size, "title": "",
            "extractor": "download-kit"}, size)
        return True

    # ── stage 1c: connector streams (B2) ──────────────────────────

    def _site_connector(self, connector_id: str) -> Any | None:
        """The connector for the connector stage, by id: an injected
        one wins (the ``site_connectors`` mapping, then the legacy
        injected SoundCloud connector for id ``"soundcloud"``);
        otherwise it is built through the injected connector
        factory + the accounts vault, exactly like the proxy pool.
        None when no factory is wired — the stage then records its
        absence and the chain moves on."""
        connector_id = str(connector_id or "")
        if not connector_id:
            return None
        if connector_id in self._site_connectors_injected:
            return self._site_connectors_injected[connector_id]
        if connector_id == "soundcloud" \
                and self._soundcloud_connector_injected is not None:
            return self._soundcloud_connector_injected
        if connector_id in self._connector_tried:
            return self._connector_cache.get(connector_id)
        self._connector_tried.add(connector_id)
        factory = self._connector_factory or _CONNECTOR_FACTORY
        if factory is None:
            return None
        try:
            from ..accounts.vault import open_vault
            vault = open_vault(getattr(self.context, "db", None),
                               context=self.context,
                               actor=f"downloader-{connector_id}")
            self._connector_cache[connector_id] = factory(
                connector_id, vault)
        except Exception as exc:  # noqa: BLE001 — connector optional
            record_failure("_site_connector", exc)
            self._connector_cache[connector_id] = None
        return self._connector_cache.get(connector_id)

    def _soundcloud_connector(self) -> Any | None:
        """The SoundCloud connector (the B2 seam, kept): the
        descriptor-driven stage resolves it by id."""
        return self._site_connector("soundcloud")

    def _stage_connector(self, report: DownloadReport, url: str,
                         target_dir: Path, audio_only: bool,
                         budget: dict[str, float]) -> bool:
        """The connector's own resolved stream as a download rung.

        Engages only when the matched site descriptor declares a
        connector (B4 — no per-site host check lives here any
        more): the connector resolves the track and hands back the
        stream URL (progressive MP3 or an HLS playlist, with the
        track's ``track_authorization`` forwarded by the connector
        itself for SoundCloud).  Progressive streams are fetched
        through the download kit (validator-verified, like the kit
        stage); HLS playlists go to the direct backend, which
        speaks HLS.  A classified block (preview / DRM / region /
        login — one of the descriptor's ``block_classes``) is
        journaled WITH its class and the chain moves on — later
        rungs (proxy, rendered capture) still get their turn, and
        true DRM is named, never chased.
        """
        desc = self._site_descriptor_for(url)
        if desc is None or not desc.connector:
            return False
        connector = self._site_connector(desc.connector)
        if connector is None:
            report.stages.append(StageLog(
                stage="connector", url=url,
                note=f"{desc.site} connector unavailable (no connector "
                     "factory wired)"))
            _log.info("download stage=connector url=%s skipped: no "
                      "connector factory", url)
            return False
        if time.perf_counter() > budget["deadline"]:
            return False
        t0 = time.perf_counter()
        try:
            stream = connector.stream_url(url)
        except Exception as exc:  # noqa: BLE001 — a stage reports
            lat = time.perf_counter() - t0
            kind = classify_failure(exc)
            block = str(getattr(exc, "reason", "") or "")
            report.stages.append(StageLog(
                stage="connector", url=url, latency_s=lat,
                error_class=kind.value, error=_err_text(exc),
                note=f"{desc.site} block: {block}" if block else ""))
            self._log_stage("connector", url, "", lat, 0, kind,
                            _err_text(exc, 200))
            return False
        final = str((stream or {}).get("url", "") or "")
        protocol = str((stream or {}).get("protocol", "") or "")
        track = (stream or {}).get("track") or {}
        title = str(track.get("title", "") or "")
        if not final:
            lat = time.perf_counter() - t0
            report.stages.append(StageLog(
                stage="connector", url=url, latency_s=lat,
                error_class=FailureKind.NO_MEDIA_FOUND.value,
                error="connector resolved the track but returned no "
                      "stream URL"))
            self._log_stage("connector", url, "", lat, 0,
                            FailureKind.NO_MEDIA_FOUND,
                            "no stream URL")
            return False
        if protocol == "hls":
            return self._connector_hls(report, final, target_dir,
                                       title, t0, budget, desc)
        dest = target_dir / _safe_name(
            str(track.get("permalink_url") or url), audio_only=True)
        try:
            kit = self._kit()
            tracker = self._tracker()
            if tracker is not None and _accepts_progress(kit.download):
                result = kit.download(
                    final, dest,
                    progress=lambda received, expected:
                        tracker.report_bytes(received, expected))
            else:
                result = kit.download(final, dest)
        except Exception as exc:  # noqa: BLE001 - a stage reports
            result = None
            error = _err_text(exc)
        else:
            error = "" if result.ok else (
                result.error or "kit fetch failed")
        lat = time.perf_counter() - t0
        if result is None or not result.ok:
            kind = classify_failure(error)
            report.stages.append(StageLog(
                stage="connector", url=url, latency_s=lat,
                error_class=kind.value, error=error,
                note="connector stream fetch failed"))
            self._log_stage("connector", url, "", lat, 0, kind,
                            _err_text(error, 200))
            return False
        landed = self._verify_landed(
            {"path": result.path, "bytes": result.size_bytes})
        if landed is None:
            report.stages.append(StageLog(
                stage="connector", url=url, latency_s=lat,
                error_class=FailureKind.NO_MEDIA_FOUND.value,
                error="connector stream fetch reported success but "
                      "the file is missing or empty on disk "
                      f"({result.path or 'no path'})"))
            self._log_stage("connector", url, "", lat, 0,
                            FailureKind.NO_MEDIA_FOUND,
                            "reported success, nothing on disk")
            return False
        _path, size = landed
        report.stages.append(StageLog(
            stage="connector", url=url, latency_s=lat,
            size_bytes=size, ok=True,
            note=f"{desc.site} connector stream (progressive) via "
                 "download kit"))
        self._log_stage("connector", url, "", lat, size,
                        FailureKind.OK, "")
        self._fill_success(report, {
            "path": _path, "bytes": size, "title": title,
            "extractor": f"{desc.connector}-connector"}, size)
        return True

    def _connector_hls(self, report: DownloadReport, playlist_url: str,
                       target_dir: Path, title: str, t0: float,
                       budget: dict[str, float], desc: Any) -> bool:
        """An HLS stream URL from the connector, fetched through the
        direct backend (yt-dlp/ffmpeg speak HLS natively)."""
        from ..tools.media import download as direct_download

        if time.perf_counter() > budget["deadline"]:
            return False
        try:
            result = direct_download(
                playlist_url, target_dir, format_spec="bestaudio/best",
                audio_only=True, timeout=self.stage_timeout,
                cookies_file="",
                progress=self._job_progress("connector"),
                **self._progress_kwargs(self._tracker()))
        except Exception as exc:  # noqa: BLE001 — a stage reports
            lat = time.perf_counter() - t0
            kind = classify_failure(exc)
            report.stages.append(StageLog(
                stage="connector", url=playlist_url, latency_s=lat,
                error_class=kind.value, error=_err_text(exc),
                note="connector HLS fetch failed"))
            self._log_stage("connector", playlist_url, "", lat, 0,
                            kind, _err_text(exc, 200))
            return False
        lat = time.perf_counter() - t0
        landed = self._verify_landed(result)
        if landed is None:
            report.stages.append(StageLog(
                stage="connector", url=playlist_url, latency_s=lat,
                error_class=FailureKind.NO_MEDIA_FOUND.value,
                error="connector HLS fetch reported success but the "
                      "file is missing or empty on disk "
                      f"({result.get('path', '') or 'no path'})"))
            self._log_stage("connector", playlist_url, "", lat, 0,
                            FailureKind.NO_MEDIA_FOUND,
                            "reported success, nothing on disk")
            return False
        _path, size = landed
        report.stages.append(StageLog(
            stage="connector", url=playlist_url, latency_s=lat,
            size_bytes=size, ok=True,
            note=f"{desc.site} connector stream (HLS) via direct "
                 "backend"))
        self._log_stage("connector", playlist_url, "", lat, size,
                        FailureKind.OK, "")
        merged = dict(result)
        merged["title"] = title or str(result.get("title", "") or "")
        merged["extractor"] = f"{desc.connector}-connector"
        self._fill_success(report, merged, size)
        return True

    # ── context-derived helpers ──────────────────────────────────────

    def _session_dir(self) -> str:
        """The browser-profile sessions dir, resolved exactly like
        the browser tool resolves it ("" when no settings are wired —
        persistence is then off, as before)."""
        settings = getattr(self.context, "settings", None) \
            if self.context is not None else None
        resolve = getattr(settings, "resolve", None)
        if callable(resolve):
            try:
                return str(resolve("data/browser/sessions"))
            except Exception as e:  # persistence is optional
                record_failure("_session_dir", e)
                return ""
        return ""

    def _cookies_file(self, url: str) -> str:
        """Best cookies file for *url*: the owner's exported file, or
        Devon's own browser-profile export when she holds a login for
        the host (see :mod:`nomorals.media.cookies`).  "" when none."""
        try:
            from .cookies import cookies_file_for
            return cookies_file_for(url, self._session_dir()) or ""
        except Exception as e:  # cookies are an enhancement
            record_failure("_cookies_file", e)
            return ""

    # ── stage 2: proxy lab ────────────────────────────────────────────

    def _proxy_pool(self) -> Any | None:
        """The proxy lab connector, or None when unavailable."""
        factory = self._connector_factory or _CONNECTOR_FACTORY
        if factory is None:
            return None
        try:
            from ..accounts.vault import open_vault
            vault = open_vault(getattr(self.context, "db", None),
                               context=getattr(self, "context", None),
                               actor="downloader-proxy")
            return factory("proxypool", vault)
        except Exception as exc:  # noqa: BLE001 — pool optional
            _log.debug("downloader: proxy pool unavailable: %s", exc)
            return None

    # ── route authority (doc 17 P2/P3) ────────────────────────────────

    def _route_authority(self) -> Any | None:
        """The one route authority (:mod:`nomorals.core.proxy_routing`),
        journaling to ``<home>/proxies/route_receipts.jsonl`` — the same
        journal the lab rotation manager and the browser write to.
        Lazy + cached; None when the authority cannot be built —
        receipts are evidence, never a gate on downloading."""
        if getattr(self, "_authority_tried", False):
            return self._authority_cache
        self._authority_tried = True
        self._authority_cache = None
        try:
            from ..core.proxy_routing import (
                RouteAuthority,
                RouteReceiptJournal,
                default_receipts_path,
            )
            home = getattr(self.context, "home", None) \
                if self.context is not None else None
            path = default_receipts_path()
            # A concrete str/Path home wins (production + tests point
            # it at the real Devon home).  The check must be concrete:
            # os.PathLike's subclasshook accepts any object sporting
            # __fspath__ — including test doubles — and a relative
            # "home" is never the Devon home.
            if isinstance(home, (str, Path)):
                candidate = Path(home).expanduser()
                if candidate.is_absolute():
                    path = candidate / "proxies" / "route_receipts.jsonl"
            self._authority_cache = RouteAuthority(
                RouteReceiptJournal(path))
        except Exception as exc:  # noqa: BLE001 - receipts never break
            record_failure("_route_authority", exc)
        return self._authority_cache

    def _record_route_decision(self, proxy: dict[str, Any],
                               proxy_url: str, url: str, *,
                               candidates: int,
                               candidate_tiers: dict[str, int],
                               position: int, total: int,
                               scheme: str) -> str:
        """Journal one proxy-stage attempt through the authority
        (consumer ``downloader``) and return its decision_id — ``""``
        when journaling is unavailable, so the attempt proceeds
        unreceipted rather than failing.  The receipt is built only
        from the proxy's public fields + a boolean ``authenticated``
        flag: the credentialed URL is the connect string handed to
        the fetch, never receipt material."""
        auth = self._route_authority()
        if auth is None:
            return ""
        try:
            from ..core.proxy_routing import ROUTE_FREE, ROUTE_PAID
            from ..core.proxy_tiers import normalize_tier
            try:
                tier = normalize_tier(proxy.get("tier", "free"))
            except ValueError:
                tier = "free"
            receipt = auth.record(
                consumer="downloader",
                route=tier if tier in (ROUTE_PAID, ROUTE_FREE)
                else ROUTE_FREE,
                reason_code="healthy_order",
                reason=f"healthy pool candidate {position} of {total} "
                       f"in protocol-aware order for {scheme}",
                host=urllib.parse.urlparse(url).hostname or "",
                source="proxypool",
                tier=tier,
                proxy_id=str(proxy.get("id", "")),
                protocol=str(proxy.get("protocol", "") or ""),
                country=str(proxy.get("country", "") or ""),
                candidates=candidates,
                candidate_tiers=candidate_tiers,
                selected_by="healthy_order",
                authenticated=bool(
                    urllib.parse.urlparse(proxy_url).username))
            return receipt.decision_id
        except Exception as exc:  # noqa: BLE001 - never break the stage
            record_failure("_record_route_decision", exc)
            return ""

    def _record_route_outcome(self, decision_id: str, outcome: str,
                              detail: str = "") -> None:
        """Append the observed outcome of a journaled decision.  No
        decision_id (journaling unavailable at decision time) → no
        outcome is fabricated."""
        if not decision_id:
            return
        auth = self._route_authority()
        if auth is None:
            return
        try:
            auth.record_outcome(decision_id, outcome, detail)
        except Exception as exc:  # noqa: BLE001 - reporting never breaks
            record_failure("_record_route_outcome", exc)

    @staticmethod
    def _socks_usable() -> bool:
        from .. import compat
        return bool(compat.available("socks"))

    def _order_proxies(self, proxies: list[dict[str, Any]],
                       scheme: str) -> list[dict[str, Any]]:
        """Protocol-aware ordering: same-scheme forward proxies first,
        then plain http forward proxies, then SOCKS variants."""
        def rank(p: dict[str, Any]) -> int:
            proto = str(p.get("protocol", "http") or "http").lower()
            if proto == scheme:
                return 0
            if proto == "http":
                return 1
            if proto in ("socks5", "socks"):
                return 2
            if proto in ("socks4a", "socks4"):
                return 3
            if proto == "https":
                return 1
            return 4
        return sorted(proxies, key=rank)

    def _stage_proxy(self, report: DownloadReport, url: str,
                     target_dir: Path, audio_only: bool,
                     format_spec: str, timeout: float,
                     budget: dict[str, float]) -> bool:
        """Retry through the proxy lab.  A failing proxy is demoted;
        the download moves on to the next one."""
        from ..tools.media import download as direct_download

        pool = self._proxy_pool()
        if pool is None:
            report.stages.append(StageLog(
                stage="proxy", url=url,
                note="proxy pool unavailable "
                     "(vault locked or no stored proxies)"))
            _log.info("download stage=proxy url=%s skipped: pool "
                      "unavailable", url)
            return False
        try:
            proxies = pool.healthy_proxies()
        except Exception as exc:  # noqa: BLE001
            report.stages.append(StageLog(
                stage="proxy", url=url,
                note=f"could not list healthy proxies: "
                     f"{_err_text(exc, 200)}"))
            return False
        if not proxies:
            report.stages.append(StageLog(
                stage="proxy", url=url,
                note="proxy pool has no healthy proxies"))
            _log.info("download stage=proxy url=%s skipped: pool empty",
                      url)
            return False
        scheme = (urllib.parse.urlparse(url).scheme or "https").lower()
        ordered = self._order_proxies(proxies, scheme)[:self.max_proxies]
        candidate_tiers: dict[str, int] = {}
        from ..core.proxy_tiers import normalize_tier
        for row in proxies:
            try:
                row_tier = normalize_tier(row.get("tier", "free"))
            except Exception:  # noqa: BLE001 - unlabelled = free
                row_tier = "free"
            candidate_tiers[row_tier] = \
                candidate_tiers.get(row_tier, 0) + 1
        socks_ok = self._socks_usable()
        for position, proxy in enumerate(ordered, start=1):
            if time.perf_counter() > budget["deadline"]:
                break
            proto = str(proxy.get("protocol", "http") or "http").lower()
            pid = str(proxy.get("id", ""))
            masked = _mask_proxy(proxy.get("proxy_url", "") or
                                 proxy.get("url", ""))
            if proto.startswith("socks") and not socks_ok:
                report.stages.append(StageLog(
                    stage="proxy", url=url, proxy=masked,
                    note=f"skipped {proto} proxy — no SOCKS support "
                         "in this runtime (not a failure, not demoted)"))
                _log.info("download stage=proxy url=%s proxy=%s skipped: "
                          "socks unsupported", url, masked)
                continue
            proxy_url = proxy.get("proxy_url") or proxy.get("url_with_auth") \
                or proxy.get("url", "")
            if not proxy_url:
                continue
            # One route decision per attempt, minted through the route
            # authority (doc 17 P3): the journal gains the receipt and
            # the fetch gets the proxy EXPLICITLY (tools.media's
            # ``proxy=`` — yt-dlp's own option/flag, HttpClient's
            # proxy_url).  The retired ``_proxy_env`` process-wide env
            # window is gone: no *_PROXY key is written, so no
            # unrelated in-process reader (pumpdesk, apkforge, bare
            # urllib) can inherit this download's route.
            decision_id = self._record_route_decision(
                proxy, proxy_url, url, candidates=len(proxies),
                candidate_tiers=candidate_tiers, position=position,
                total=len(ordered), scheme=scheme)
            t0 = time.perf_counter()
            try:
                result = direct_download(
                    url, target_dir, format_spec=format_spec,
                    audio_only=audio_only, timeout=timeout,
                    cookies_file=self._cookies_file(url),
                    progress=self._job_progress("proxy"),
                    proxy=proxy_url,
                    **self._progress_kwargs(self._tracker()))
            except Exception as exc:  # noqa: BLE001 — demote, don't die
                kind = classify_failure(exc)
                from ..core.proxy_routing import OUTCOME_FAILED
                self._record_route_outcome(
                    decision_id, OUTCOME_FAILED,
                    f"{kind.value}: {_err_text(exc, 160)}")
                lat = time.perf_counter() - t0
                try:
                    pool.record_failure(
                        pid, f"{kind.value}: {_err_text(exc, 280)}")
                except Exception as demote_exc:  # noqa: BLE001
                    _log.debug("proxy demote failed for %s: %s",
                               pid, demote_exc)
                report.stages.append(StageLog(
                    stage="proxy", url=url, proxy=masked, latency_s=lat,
                    error_class=kind.value, error=_err_text(exc),
                    note=f"proxy {pid or '?'} ({proto}) demoted; "
                         f"trying next"))
                self._log_stage("proxy", url, masked, lat, 0, kind,
                                _err_text(exc, 200))
                continue
            lat = time.perf_counter() - t0
            landed = self._verify_landed(result)
            if landed is None:
                # The fetch through this proxy claimed success but
                # landed nothing usable.  That is a content failure,
                # not proxy ill-health: the proxy is neither credited
                # nor demoted, and the chain tries the next one.  The
                # route receipt likewise gets NO outcome append: the
                # fetch completed through the route, so neither
                # "used" (nothing landed) nor "failed" (the route
                # carried it) is attributable — the decision row
                # keeps outcome None rather than blaming the route
                # for the content.
                report.stages.append(StageLog(
                    stage="proxy", url=url, proxy=masked, latency_s=lat,
                    error_class=FailureKind.NO_MEDIA_FOUND.value,
                    error="download reported success but the file is "
                          "missing or empty on disk "
                          f"({result.get('path', '') or 'no path'})",
                    note="proxy neither credited nor demoted — "
                         "content failure, trying next"))
                self._log_stage("proxy", url, masked, lat, 0,
                                FailureKind.NO_MEDIA_FOUND,
                                "reported success, nothing on disk")
                continue
            _path, size = landed
            try:
                pool.record_success(pid, latency_ms=lat * 1000)
            except Exception as e:  # bookkeeping only
                record_failure("_stage_proxy.bookkeeping", e)
            from ..core.proxy_routing import OUTCOME_USED
            self._record_route_outcome(
                decision_id, OUTCOME_USED,
                f"extractor={result.get('extractor', '')}")
            report.stages.append(StageLog(
                stage="proxy", url=url, proxy=masked, latency_s=lat,
                size_bytes=size, ok=True,
                note=f"extractor={result.get('extractor', '')}"))
            self._log_stage("proxy", url, masked, lat, size,
                            FailureKind.OK, "")
            self._fill_success(report, result, size)
            return True
        return False

    # ── stage 3a: rendered capture (B3) ─────────────────────────────

    def _open_rendered_tab(self, page_url: str) -> Any:
        """A rendered tab for the capture stage.

        An injected factory wins (tests).  The production default
        opens a real tab on the browser service in the shared
        ``downloader`` session — the service's own persistent cookie
        space, so a login the browser earned counts here — on THIS
        thread (Playwright is thread-bound; a fresh service per
        download mirrors ``/browser run``'s per-job service).  Raises
        when the engine is unavailable: the stage classifies and
        journals that, it never fakes a tab.
        """
        if self._rendered_tab_factory is not None:
            return self._rendered_tab_factory(page_url)
        from ..browser.service import BrowserService

        data_dir: Any = None
        settings = getattr(self.context, "settings", None) \
            if self.context is not None else None
        resolve = getattr(settings, "resolve", None)
        if callable(resolve):
            try:
                data_dir = resolve("data/browser-service")
            except Exception as e:  # service default is fine
                record_failure("_open_rendered_tab.resolve", e)
        service = BrowserService(data_dir=data_dir) \
            if data_dir else BrowserService()
        try:
            tab = service.open_rendered_tab("downloader", page_url)
        except Exception:
            try:
                service.shutdown()
            except Exception as e:  # teardown is best-effort
                record_failure("_open_rendered_tab.shutdown", e)
            raise
        return _RenderedCaptureTab(service, tab)

    def _stage_rendered(self, report: DownloadReport, url: str,
                        target_dir: Path, audio_only: bool,
                        page_url: str, budget: dict[str, float]) -> bool:
        """The driven-player capture stage.

        Opens the page in a real rendered tab, starts playback,
        harvests the media requests the player itself made from the
        tab's network log (selected by the log's evidence — content
        type, extension, size), and fetches the winner through the
        tab's own session, so session-gated streams resolve with
        the session's cookies.  HLS playlists are walked segment by
        segment through the same session.  A log carrying DRM
        license traffic ends the stage with the DRM reason — the
        segments are encrypted and no rung downloads them.  Any
        failure falls through to the static browser stage and the
        honest report, exactly like a sibling stage.
        """
        page = page_url or url
        page_path = urllib.parse.urlparse(page).path.lower()
        if page_path.endswith(_MEDIA_EXTS):
            # The target already names a media file — there is no
            # player to drive.  Not engaged, like the kit stage on a
            # non-direct descriptor: silent, no stage log.
            return False
        t0 = time.perf_counter()
        try:
            tab = self._open_rendered_tab(page)
        except Exception as exc:  # noqa: BLE001 — a stage reports
            lat = time.perf_counter() - t0
            low = str(exc).lower()
            kind = FailureKind.DEPENDENCY_MISSING \
                if "playwright" in low else classify_failure(exc)
            report.stages.append(StageLog(
                stage="rendered", url=page, latency_s=lat,
                error_class=kind.value, error=_err_text(exc),
                note="rendered tab unavailable"))
            self._log_stage("rendered", page, "", lat, 0, kind,
                            _err_text(exc, 200))
            return False
        try:
            self._try_start_playback(tab)
            candidates, drm_url = self._poll_capture(
                tab, budget, audio_only)
            lat = time.perf_counter() - t0
            if drm_url:
                report.stages.append(StageLog(
                    stage="rendered", url=page, latency_s=lat,
                    error_class=FailureKind.SITE_ERROR.value,
                    error="the page's player uses DRM-encrypted "
                          "streams (license traffic in the tab's "
                          f"network log: {drm_url[:120]}) — capture "
                          "cannot take DRM media, and no rung "
                          "downloads it",
                    note="drm detected in network log"))
                self._log_stage("rendered", page, "", lat, 0,
                                FailureKind.SITE_ERROR,
                                "drm license traffic in network log")
                return False
            if not candidates:
                report.stages.append(StageLog(
                    stage="rendered", url=page, latency_s=lat,
                    error_class=FailureKind.NO_MEDIA_FOUND.value,
                    error="the rendered player made no media "
                          "requests the network log could capture",
                    note="page played (or tried to) but exposed no "
                         "media requests"))
                self._log_stage("rendered", page, "", lat, 0,
                                FailureKind.NO_MEDIA_FOUND,
                                "no media in network log")
                return False
            report.stages.append(StageLog(
                stage="rendered", url=page, latency_s=lat,
                note=f"network log captured {len(candidates)} media "
                     f"candidate(s); fetching through the tab "
                     f"session"))
            for cand in candidates:
                if time.perf_counter() > budget["deadline"]:
                    break
                if self._fetch_capture_candidate(
                        report, tab, cand, target_dir, audio_only,
                        page):
                    return True
            return False
        except Exception as exc:  # noqa: BLE001 — a stage reports
            lat = time.perf_counter() - t0
            kind = classify_failure(exc)
            report.stages.append(StageLog(
                stage="rendered", url=page, latency_s=lat,
                error_class=kind.value, error=_err_text(exc),
                note="rendered capture failed"))
            self._log_stage("rendered", page, "", lat, 0, kind,
                            _err_text(exc, 200))
            return False
        finally:
            try:
                tab.close()
            except Exception as e:  # teardown never masks the stage
                record_failure("_stage_rendered.close", e)

    def _try_start_playback(self, tab: Any) -> None:
        """Best-effort play press: players expose the control under
        many shapes; each miss is expected, never an error.  A
        player that starts on its own is equally fine — the network
        log is the evidence either way."""
        click = getattr(tab, "click", None)
        if callable(click):
            for target in ("Play", "[aria-label='Play']",
                           "button[aria-label*='play' i]",
                           ".playButton", "#play"):
                try:
                    click(target)
                    break
                except Exception as exc:  # noqa: BLE001 — try next
                    _log.debug("rendered capture: play target %r "
                               "missed: %s", target, exc)
        idle = getattr(tab, "wait_for_network_idle", None)
        if callable(idle):
            try:
                idle(8_000)
            except Exception as exc:  # noqa: BLE001 — settling is
                # best-effort; the poll below is the real wait
                _log.debug("rendered capture: network-idle wait "
                           "ended early: %s", exc)

    def _poll_capture(self, tab: Any, budget: dict[str, float],
                      audio_only: bool) -> tuple[list[str], str]:
        """Poll the tab's network log for media evidence.

        Returns ``(candidates, drm_url)``: the selected media URLs
        in preference order, and the DRM license URL when the log
        carries one (candidates are then meaningless — encrypted
        segments are not a download).  Waits out a bounded window
        inside the chain's deadline: players fetch media lazily,
        so the first empty read is not the verdict.
        """
        soft_deadline = min(budget["deadline"],
                            time.perf_counter() + 25.0)
        while True:
            raw = tab.network_requests(limit=500)
            entries = raw.get("requests", []) \
                if isinstance(raw, dict) else []
            drm_url = _drm_evidence(entries)
            if drm_url:
                return [], drm_url
            candidates = _capture_candidates(
                entries, audio_only=audio_only)
            if candidates:
                return candidates, ""
            if time.perf_counter() >= soft_deadline:
                return [], ""
            time.sleep(1.5)

    def _fetch_capture_candidate(self, report: DownloadReport,
                                 tab: Any, media_url: str,
                                 target_dir: Path, audio_only: bool,
                                 page_url: str) -> bool:
        """Fetch one captured media URL through the tab's session
        (``session_fetch_bytes`` — the browser context's own request
        stack, cookies riding along) and verify what landed."""
        fetch = getattr(tab, "session_fetch_bytes", None)
        if not callable(fetch):
            report.stages.append(StageLog(
                stage="rendered", url=media_url,
                error_class=FailureKind.DEPENDENCY_MISSING.value,
                error="the rendered tab cannot fetch through its "
                      "session"))
            return False
        is_hls = urllib.parse.urlparse(
            media_url).path.lower().endswith(".m3u8")
        name = _safe_name(page_url if is_hls else media_url,
                          audio_only)
        if is_hls and "." not in name.rsplit("/", 1)[-1]:
            name += ".mp3" if audio_only else ".ts"
        dest = target_dir / name
        t0 = time.perf_counter()
        try:
            if is_hls:
                self._fetch_hls_via_session(tab, media_url, dest)
            else:
                data = fetch(media_url)
                if not data:
                    raise ValueError("session fetch returned no bytes")
                dest.write_bytes(bytes(data))
        except Exception as exc:  # noqa: BLE001 — try the next one
            lat = time.perf_counter() - t0
            kind = classify_failure(exc)
            report.stages.append(StageLog(
                stage="rendered", url=media_url, latency_s=lat,
                error_class=kind.value, error=_err_text(exc),
                note="captured media fetch failed"))
            self._log_stage("rendered", media_url, "", lat, 0, kind,
                            _err_text(exc, 200))
            try:
                dest.unlink(missing_ok=True)
            except OSError as e:
                record_failure("_fetch_capture_candidate.unlink", e)
            return False
        lat = time.perf_counter() - t0
        # Byte-level landing truth, same as the browser stage: an
        # HTML page wearing a media name is an error page, and the
        # size on disk is the only size that counts.
        from ..tools.media import _looks_like_html

        try:
            actual = dest.stat().st_size if dest.is_file() else 0
            with dest.open("rb") as fh:
                head = fh.read(1024)
        except OSError:
            actual, head = 0, b""
        if actual <= 0 or _looks_like_html(head):
            why = ("session fetch reported bytes but no file is on "
                   "disk" if actual <= 0 else
                   "landed bytes are an HTML page, not media")
            report.stages.append(StageLog(
                stage="rendered", url=media_url, latency_s=lat,
                size_bytes=actual,
                error_class=FailureKind.NO_MEDIA_FOUND.value,
                error=why, note="captured candidate was not media"))
            self._log_stage("rendered", media_url, "", lat, actual,
                            FailureKind.NO_MEDIA_FOUND, why)
            try:
                dest.unlink(missing_ok=True)
            except OSError as e:
                record_failure("_fetch_capture_candidate.unlink#2", e)
            return False
        report.stages.append(StageLog(
            stage="rendered", url=media_url, latency_s=lat,
            size_bytes=actual, ok=True,
            note="captured from the tab's network log; fetched "
                 "through the tab session"))
        self._log_stage("rendered", media_url, "", lat, actual,
                        FailureKind.OK, "")
        self._fill_success(report, {
            "path": str(dest), "bytes": actual, "title": dest.stem,
            "extractor": "rendered-capture"}, actual)
        return True

    def _fetch_hls_via_session(self, tab: Any, playlist_url: str,
                               dest: Path) -> None:
        """Walk an HLS playlist through the tab's session: master →
        media playlist → segments in order, concatenated.  Raises
        on any failure — a partial concatenation is deleted by the
        caller, never delivered as the track."""
        fetch = tab.session_fetch_bytes
        raw = fetch(playlist_url)
        text = bytes(raw).decode("utf-8", errors="replace")
        if "#EXT-X-STREAM-INF" in text:
            variant = _hls_variant_url(text, playlist_url)
            if not variant:
                raise ValueError("HLS master playlist names no "
                                 "variant")
            raw = fetch(variant)
            text = bytes(raw).decode("utf-8", errors="replace")
            playlist_url = variant
        segments: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("#EXT-X-MAP:"):
                m = re.search(r'URI="([^"]+)"', line)
                if m:
                    segments.append(urllib.parse.urljoin(
                        playlist_url, m.group(1)))
                continue
            if line.startswith("#"):
                continue
            segments.append(urllib.parse.urljoin(playlist_url, line))
        if not segments:
            raise ValueError("HLS playlist carried no segments")
        if len(segments) > 4096:
            raise ValueError(
                f"HLS playlist carries {len(segments)} segments — "
                "beyond the capture cap")
        with dest.open("wb") as fh:
            for seg in segments:
                fh.write(bytes(fetch(seg)))

    # ── stage 3: browser context ──────────────────────────────────────

    def _browser_session(self) -> Any | None:
        try:
            from ..tools.browser import get_session, \
                seed_session_from_persisted
            session_dir = self._session_dir()
            if session_dir:
                # Same persisted profile the interactive browser tool
                # uses: this session restores its own saved cookies,
                # and adopts the whole profile's — a login earned in
                # ANY session serves the download.
                session = get_session("downloader",
                                      session_dir=session_dir)
                adopted = seed_session_from_persisted(session, session_dir)
                if adopted:
                    _log.debug("downloader: browser session adopted %d "
                               "persisted cookie(s)", adopted)
                return session
            return get_session("downloader")
        except Exception as exc:  # noqa: BLE001
            _log.debug("downloader: browser session unavailable: %s", exc)
            return None

    def _media_candidates(self, session: Any) -> list[str]:
        """Real media URLs from the current page: meta tags, media
        elements, and download links — in that order of trust."""
        cands: list[str] = []
        seen: set[str] = set()

        def add(raw: str) -> None:
            if not raw:
                return
            abs_url = urllib.parse.urljoin(session.url, raw.strip())
            if not abs_url.lower().startswith(("http://", "https://")):
                return
            key = abs_url.split("#", 1)[0]
            if key not in seen:
                seen.add(key)
                cands.append(key)

        try:
            meta = session.extract(kind="meta") or {}
        except Exception as e:
            record_failure("_media_candidates.meta", e)
            meta = {}
        og = (meta.get("og") or {}) if isinstance(meta, dict) else {}
        for key in ("og:video", "og:video:url", "og:video:secure_url",
                    "og:audio", "og:audio:url"):
            add(str(og.get(key) or ""))
        other = (meta.get("other") or {}) if isinstance(meta, dict) else {}
        add(str(other.get("twitter:player:stream") or ""))
        try:
            raw_html = (session.html(max_chars=2_000_000) or {}).get("html",
                                                                    "")
        except Exception as e:
            record_failure("_media_candidates.raw_html", e)
            raw_html = ""
        for m in _SRC_TAG_RE.finditer(raw_html or ""):
            add(m.group(1))
        # Embedded JSON state: SPA players (TikTok above all) ship the
        # media URL inside hydration JSON / JSON-LD, with no og:video
        # and no <video src> anywhere in the served HTML.
        for media_url in _embedded_media_urls(raw_html or ""):
            add(media_url)
        try:
            links = (session.links(max_links=200) or {}).get("links", [])
        except Exception as e:
            record_failure("_media_candidates.links", e)
            links = []
        for link in links or []:
            href = str((link or {}).get("url") or "")
            text = str((link or {}).get("text") or "")
            if href.split("?", 1)[0].lower().endswith(_MEDIA_EXTS):
                add(href)
            elif _DOWNLOAD_LINK_RE.search(text) and href:
                # click-to-download button — follow it and harvest the
                # landing page's media links next.
                cands.append(f"click:{href}")
        return cands[:self.browser_candidates]

    def _cloudflare_gate(self, report: DownloadReport, session: Any,
                         page_url: str) -> None:
        """Pass #79: the download kit's Cloudflare state machine,
        wired at last (media/cloudflare_gate.py is the decision
        seam). An opened page that is a CF interstitial feeds the
        machine; a stored vault clearance is injected into the
        session's cookie jar and the page re-opened once; an ask or
        an expiry lands on the report (stage note + hint) so the
        next rung sees the kit's own words. Never raises — the gate
        advises the browser stage, it never sinks it.
        """
        try:
            body = str((session.html() or {}).get("html") or "")
            if not body:
                return
            from .cloudflare_gate import CloudflareGate
            gate = CloudflareGate(self._cf_cookie_store())
            if not gate.inspect(body):
                return
            decision = gate.cookie_decision(page_url)
            note = f"cloudflare interstitial: {decision['message']}"
            cleared = False
            if decision["action"] == "inject":
                import http.cookiejar as _cj
                import urllib.parse as _up
                host = _up.urlparse(page_url).hostname or ""
                jar = getattr(session, "cookie_jar", None)
                value = str(decision["cookie"]).split("=", 1)[-1]
                if jar is not None and host:
                    jar.set_cookie(_cj.Cookie(
                        0, "cf_clearance", value, None, False,
                        host, True, False, "/", True, False, None,
                        True, None, None, {}, False))
                    reopened = session.open(page_url)
                    cleared = bool(reopened.get("ok"))
                    if cleared:
                        body2 = str((session.html() or {}).get(
                            "html") or "")
                        cleared = not gate.inspect(body2)
                    gate.resolve(cleared)
                    note += ("; vault clearance injected, page "
                             "re-opened — wall cleared"
                             if cleared else
                             "; vault clearance injected but the "
                             "wall is still up (gate state "
                             f"{gate.state})")
                else:
                    note += "; the session has no cookie jar to " \
                            "inject into"
            report.stages.append(StageLog(stage="browser",
                                          url=page_url, note=note))
            if not cleared:
                report.hint = ((report.hint + " ") if report.hint
                               else "") + note
        except Exception as exc:  # noqa: BLE001 - gate never sinks
            _log.info("cloudflare gate failed soft: %s", exc)

    def _stage_browser(self, report: DownloadReport, url: str,
                       target_dir: Path, audio_only: bool, page_url: str,
                       budget: dict[str, float]) -> bool:
        """Open the source page in the browser session, extract the real
        media URL, and download the bytes through the session so the
        site's cookies/headers ride along."""
        session = self._browser_session()
        if session is None:
            report.stages.append(StageLog(
                stage="browser", url=page_url,
                note="browser session unavailable"))
            return False
        t0 = time.perf_counter()
        try:
            opened = session.open(page_url)
        except Exception as exc:  # noqa: BLE001
            kind = classify_failure(exc)
            lat = time.perf_counter() - t0
            report.stages.append(StageLog(
                stage="browser", url=page_url, latency_s=lat,
                error_class=kind.value, error=_err_text(exc),
                note="could not open the source page in the browser"))
            self._log_stage("browser", page_url, "", lat, 0, kind,
                            _err_text(exc, 200))
            return False
        if not opened.get("ok"):
            lat = time.perf_counter() - t0
            report.stages.append(StageLog(
                stage="browser", url=page_url, latency_s=lat,
                error_class=classify_http_status(
                    int(opened.get("status") or 0)).value,
                error=f"page returned HTTP {opened.get('status')}",
                note="source page did not load in the browser"))
            self._log_stage("browser", page_url, "",
                            lat, 0,
                            classify_http_status(
                                int(opened.get("status") or 0)),
                            f"HTTP {opened.get('status')}")
            return False
        self._log_stage("browser", page_url, "", time.perf_counter() - t0,
                        0, FailureKind.OK, "", note="page opened")
        self._cloudflare_gate(report, session, page_url)
        candidates = self._media_candidates(session)
        # Discovery intake (Pass #86): media hosts embedded in the
        # page (alternate CDNs / mirrors the extractor mined) that
        # the registry does not know become untrusted candidates.
        for cand in candidates:
            self._intake_candidate(
                cand[len("click:"):] if cand.startswith("click:") else cand,
                via="browser-page")
        report.stages.append(StageLog(
            stage="browser", url=page_url,
            note=f"page opened; {len(candidates)} media candidate(s) "
                 f"extracted"))
        if not candidates:
            report.stages.append(StageLog(
                stage="browser", url=page_url,
                error_class=FailureKind.NO_MEDIA_FOUND.value,
                error="page has no og:video/og:audio, media tags, "
                      "embedded media JSON, or download links"))
            return False
        fetch = getattr(session, "fetch_bytes", None)
        if not callable(fetch):
            report.stages.append(StageLog(
                stage="browser", url=page_url,
                error_class=FailureKind.DEPENDENCY_MISSING.value,
                error="browser session has no fetch_bytes"))
            return False
        for cand in candidates:
            if time.perf_counter() > budget["deadline"]:
                break
            target = cand
            if cand.startswith("click:"):
                # click-to-download button: follow it, then re-harvest
                try:
                    clicked = session.click(cand[len("click:"):])
                    target = str(clicked.get("url") or session.url or "")
                except Exception as exc:  # noqa: BLE001
                    _log.info("browser: click-to-download failed: %s",
                              exc)
                    continue
                if not target.split("?", 1)[0].lower().endswith(
                        _MEDIA_EXTS):
                    # landing page, not a file — harvest its media links
                    for sub in self._media_candidates(session):
                        if sub.startswith("click:"):
                            continue
                        if self._fetch_candidate(
                                report, session, sub, target_dir,
                                audio_only, url):
                            return True
                    continue
            if self._fetch_candidate(report, session, target, target_dir,
                                     audio_only, url):
                return True
        return False

    def _fetch_candidate(self, report: DownloadReport, session: Any,
                         media_url: str, target_dir: Path,
                         audio_only: bool, orig_url: str) -> bool:
        """Fetch one candidate media URL through the browser session."""
        fetch = getattr(session, "fetch_bytes", None)
        name = _safe_name(media_url, audio_only)
        dest = target_dir / name
        try:
            from ..core.jobs import report_progress
            report_progress(f"browser: fetching {media_url[:80]}")
        except Exception as e:  # progress is best-effort
            record_failure("_fetch_candidate.progress", e)
        t0 = time.perf_counter()
        try:
            got = fetch(media_url, dest)
        except Exception as exc:  # noqa: BLE001
            kind = classify_failure(exc)
            lat = time.perf_counter() - t0
            report.stages.append(StageLog(
                stage="browser", url=media_url, latency_s=lat,
                error_class=kind.value, error=_err_text(exc)))
            self._log_stage("browser", media_url, "", lat, 0, kind,
                            _err_text(exc, 200))
            return False
        lat = time.perf_counter() - t0
        size = int(got.get("bytes") or 0)
        ctype = str(got.get("content_type") or "")
        status = int(got.get("status") or 0)
        if status >= 400 or size == 0:
            kind = classify_http_status(status) if status >= 400 \
                else FailureKind.NO_MEDIA_FOUND
            report.stages.append(StageLog(
                stage="browser", url=media_url, latency_s=lat,
                size_bytes=size, error_class=kind.value,
                error=f"HTTP {status}, content-type {ctype or '?'}",
                note="not a media file"))
            self._log_stage("browser", media_url, "", lat, size, kind,
                            f"HTTP {status}")
            try:
                dest.unlink(missing_ok=True)
            except OSError as e:
                record_failure("_fetch_candidate.unlink", e)
            return False
        if ctype and not (ctype.startswith(("audio/", "video/",
                                            "application/octet-stream"))
                          or "mpeg" in ctype):
            report.stages.append(StageLog(
                stage="browser", url=media_url, latency_s=lat,
                size_bytes=size,
                error_class=FailureKind.NO_MEDIA_FOUND.value,
                error=f"content-type {ctype} is not media",
                note="candidate was a page, not a file"))
            try:
                dest.unlink(missing_ok=True)
            except OSError as e:
                record_failure("_fetch_candidate.unlink#2", e)
            return False
        # Byte-level landing truth: an absent Content-Type (or a
        # generic octet-stream) proves nothing about what landed.
        # Sniff the actual bytes — an HTML page here is an error page
        # wearing a media filename — and trust the size on disk over
        # the fetch's claimed byte count.
        from ..tools.media import _looks_like_html

        try:
            actual = dest.stat().st_size if dest.is_file() else 0
            with dest.open("rb") as fh:
                head = fh.read(1024)
        except OSError:
            actual, head = 0, b""
        if actual <= 0 or _looks_like_html(head):
            why = ("browser fetch reported bytes but no file is on "
                   "disk" if actual <= 0 else
                   "landed bytes are an HTML page, not media "
                   f"(content-type {ctype or 'absent'})")
            report.stages.append(StageLog(
                stage="browser", url=media_url, latency_s=lat,
                size_bytes=actual,
                error_class=FailureKind.NO_MEDIA_FOUND.value,
                error=why, note="candidate was a page, not a file"))
            self._log_stage("browser", media_url, "", lat, actual,
                            FailureKind.NO_MEDIA_FOUND, why)
            try:
                dest.unlink(missing_ok=True)
            except OSError as e:
                record_failure("_fetch_candidate.unlink#3", e)
            return False
        report.stages.append(StageLog(
            stage="browser", url=media_url, latency_s=lat,
            size_bytes=actual, ok=True,
            note=f"content-type={ctype or '?'} via browser session"))
        self._log_stage("browser", media_url, "", lat, actual,
                        FailureKind.OK, "")
        report.ok = True
        report.failure = FailureKind.OK
        report.path = str(dest)
        report.size_bytes = actual
        report.title = dest.stem
        report.extractor = "browser-session"
        return True

    # ── failure + finishing ───────────────────────────────────────────

    def _finish_failure(self, report: DownloadReport) -> None:
        kinds = [s.error_class for s in report.stages if s.error_class]
        notes = " ".join(s.note for s in report.stages).lower()
        errors = " ".join(s.error for s in report.stages).lower()
        browser_opened = any(
            s.stage == "browser" and "page opened" in (s.note or "")
            for s in report.stages)
        if "drm-encrypted" in errors or "drm detected" in notes:
            # The rendered player itself proved the stream is
            # DRM-encrypted (license traffic in its network log):
            # that is the cause, stated plainly — no rung downloads
            # DRM media, and more trying would not change it.
            report.failure = FailureKind.SITE_ERROR
            report.likely_cause = (
                "the media is DRM-encrypted — the site's own player "
                "asked a license server for decryption keys, and no "
                "download rung can take DRM-protected streams")
            report.hint = ("play it in the source's own app, or find "
                           "the same title on a source that serves "
                           "the file")
        elif FailureKind.BOT_BLOCK.value in kinds:
            report.failure = FailureKind.BOT_BLOCK
            report.likely_cause = (
                "the site is blocking automated downloads (bot check / "
                "rate limit) on the direct, proxy, and browser paths")
            report.hint = BOT_COOKIE_HELP
        elif browser_opened and FailureKind.NO_MEDIA_FOUND.value in kinds:
            # The most informative outcome: the page itself loaded fine
            # in the browser session, so this is not a network problem —
            # the page simply exposes no downloadable media link.
            report.failure = FailureKind.NO_MEDIA_FOUND
            report.likely_cause = (
                "the source page loaded in the browser but exposed no "
                "downloadable media link (JS-rendered player or "
                "login-walled stream)")
            report.hint = ("the browser stage already ran with your "
                           "saved browser-session cookies; if the page "
                           "needs a FRESH login, open it once in the "
                           "browser (`nm browser`) and retry — new "
                           "cookies are picked up automatically")
        elif "proxy pool has no healthy proxies" in notes \
                or "proxy pool unavailable" in notes:
            if kinds and all(k == FailureKind.NETWORK_FAILURE.value
                             for k in kinds):
                report.failure = FailureKind.NETWORK_FAILURE
                report.likely_cause = (
                    "network unreachable from here (and no proxies "
                    "configured to route around it)")
                report.hint = ("add proxies to the proxy pool "
                               "(`nm proxylab`) so downloads can route "
                               "around network blocks")
            else:
                report.failure = kinds_to_failure(kinds)
                report.likely_cause = self._cause_from_kinds(kinds)
        else:
            report.failure = kinds_to_failure(kinds)
            report.likely_cause = self._cause_from_kinds(kinds)
        if not report.hint and report.failure is FailureKind.DEPENDENCY_MISSING:
            report.hint = ("install yt-dlp (`pip install yt-dlp`) — it "
                           "unlocks the direct stage for 1000+ sites")
        _log.warning("download failed url=%s cause=%s stages=%s",
                     report.url, report.likely_cause, report.tried())

    @staticmethod
    def _cause_from_kinds(kinds: list[str]) -> str:
        uniq = [k for k in dict.fromkeys(kinds) if k]
        if not uniq:
            return "unknown — no stage recorded an error"
        if len(uniq) == 1:
            return {
                "bot_block": "the site blocked the request",
                "site_error": "the site refused or the media is gone/"
                             "protected",
                "network_failure": "network-level failure reaching the site",
                "no_media_found": "no downloadable media found",
                "dependency_missing": "a required downloader is missing",
            }.get(uniq[0], uniq[0])
        return ("mixed failures across stages: " + ", ".join(uniq))

    # ── helpers ───────────────────────────────────────────────────────

    def _dest_dir(self, dest_dir: str | Path | None) -> Path:
        if dest_dir:
            p = Path(dest_dir).expanduser()
            p.mkdir(parents=True, exist_ok=True)
            return p
        try:
            from ..tools.filesystem import safe_path
            d = safe_path(self.context, "media")
            d.mkdir(parents=True, exist_ok=True)
            return Path(str(d))
        except Exception as e:
            record_failure("_dest_dir", e)
            fallback = Path.home() / "workspace" / "media"
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback

    @staticmethod
    def _verify_landed(result: dict[str, Any]) -> tuple[str, int] | None:
        """(path, size on disk) when a stage's success claim is real.

        A stage result dict is a claim, not evidence: the file must
        exist, be a regular file, and be non-empty.  Returns None
        when the claim does not hold — the stage failed, whatever it
        said.  The size that survives is the size on disk, never the
        claimed byte count.
        """
        path = str(result.get("path") or "")
        if not path:
            return None
        try:
            p = Path(path)
            if not p.is_file():
                return None
            size = p.stat().st_size
        except OSError:
            return None
        if size <= 0:
            return None
        return path, size

    @staticmethod
    def _job_progress(stage: str):
        """Progress callback forwarding a stage's real figures to the
        running background job (``core.jobs.report_progress`` — a
        no-op outside a job context, so foreground downloads are
        unaffected)."""
        def _cb(message: str) -> None:
            try:
                from ..core.jobs import report_progress
                report_progress(f"{stage}: {message}")
            except Exception as e:  # progress is best-effort
                record_failure("_cb", e)
        return _cb

    @staticmethod
    def _tracker() -> Any | None:
        """The measured-progress tracker bound to this call, if any
        (Play & Downloads P1 — ``core.progress_events``; unbound
        downloads are exactly as before)."""
        try:
            from ..core.progress_events import current_tracker
            return current_tracker()
        except Exception as e:  # progress never breaks a download
            record_failure("downloader._tracker", e)
            return None

    @staticmethod
    def _progress_kwargs(tracker: Any | None) -> dict[str, Any]:
        """Structured byte/segment callbacks for the tools-media
        backends, fed from the bound tracker (empty when unbound)."""
        if tracker is None:
            return {}
        return {
            "progress_event": lambda done, total: tracker.report_bytes(
                done, total),
            "progress_segments": lambda d, t: tracker.report_segments(d, t),
        }

    @staticmethod
    def _fill_success(report: DownloadReport,
                      result: dict[str, Any],
                      size_bytes: int | None = None) -> None:
        report.ok = True
        report.path = str(result.get("path", ""))
        report.size_bytes = int(
            size_bytes if size_bytes is not None
            else (result.get("bytes") or 0))
        report.title = str(result.get("title", "") or "")
        report.extractor = str(result.get("extractor", "") or "")
        report.failure = FailureKind.OK

    @staticmethod
    def _finish(report: DownloadReport, started: float) -> DownloadReport:
        report.total_s = round(time.perf_counter() - started, 2)
        if report.ok:
            _log.info("download ok path=%s bytes=%d stages=%s total=%.1fs",
                      report.path, report.size_bytes, report.tried(),
                      report.total_s)
        return report

    @staticmethod
    def _log_stage(stage: str, url: str, proxy: str, latency: float,
                   size: int, kind: FailureKind, error: str,
                   attempt: int = 1, note: str = "") -> None:
        _log.info(
            "download stage=%s attempt=%d url=%s proxy=%s latency=%.2fs "
            "bytes=%d error_class=%s error=%s note=%s",
            stage, attempt, url, proxy or "-", latency, size,
            kind.value if isinstance(kind, FailureKind) else kind,
            (error or "-")[:200], (note or "-")[:200])


def kinds_to_failure(kinds: list[str]) -> FailureKind:
    """Dominant failure kind from a list of stage error classes."""
    order = (FailureKind.BOT_BLOCK, FailureKind.NETWORK_FAILURE,
             FailureKind.SITE_ERROR, FailureKind.NO_MEDIA_FOUND,
             FailureKind.DEPENDENCY_MISSING)
    for kind in order:
        if kind.value in kinds:
            return kind
    return FailureKind.SITE_ERROR


def _sha256_of(path: Path) -> str:
    """Streaming sha256 of a landed file; "" when it cannot be read."""
    try:
        import hashlib
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception as e:
        record_failure("_sha256_of", e)
        return ""


def _safe_name(url: str, audio_only: bool) -> str:
    """Filesystem-safe filename for a browser-stage download."""
    from ..core.http import url_filename
    name = url_filename(url)
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" .")[:120]
    if not name or "." not in name:
        name = (name or "media") + (".mp3" if audio_only else ".bin")
    return name or "media.bin"


# ── rendered capture: network-log evidence (B3) ─────────────────────


class _RenderedCaptureTab:
    """A live rendered tab plus the browser service that owns it.

    The capture stage talks to the duck-typed tab surface
    (``network_requests`` / ``session_fetch_bytes`` / ``click`` /
    ``close``); this wrapper adds the service teardown a bare tab
    cannot do itself, so one download never leaks a browser.
    """

    def __init__(self, service: Any, tab: Any) -> None:
        self._service = service
        self._tab = tab

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tab, name)

    def close(self) -> None:
        try:
            self._tab.close()
        finally:
            self._service.shutdown()


#: URL markers of DRM license acquisition in a network log: the
#: player asked a license server for decryption keys, which means
#: the media segments themselves are encrypted.
_DRM_URL_MARKERS = ("widevine", "fairplay", "playready")

_CAPTURE_AUDIO_EXTS = (".mp3", ".m4a", ".aac", ".ogg", ".opus",
                       ".wav", ".flac")
_CAPTURE_VIDEO_EXTS = (".mp4", ".webm", ".mov", ".mkv")


def _drm_evidence(entries: list[dict[str, Any]]) -> str:
    """The first DRM license URL in a tab network log, or "".

    A Widevine/FairPlay/PlayReady license request is the log's own
    proof that the stream is DRM-encrypted: capture reports it as
    DRM instead of fetching segments it cannot decrypt.
    """
    for entry in entries or []:
        url = str((entry or {}).get("url", "") or "")
        low = url.lower()
        if any(marker in low for marker in _DRM_URL_MARKERS):
            return url
    return ""


def _capture_candidates(entries: list[dict[str, Any]], *,
                        audio_only: bool = False) -> list[str]:
    """Media URLs from a tab network log, in preference order.

    The log's own evidence decides: response entries with a media
    content type, or a media extension on the URL path, with a
    successful status when the entry carries one.  Ranking prefers
    the kind the caller asked for (audio files, then HLS playlists,
    then video files for an audio request; mirrored otherwise) and
    the larger declared content length inside a class — a segment
    is not the stream when the whole file is also there.  Preview
    URLs are excluded: a preview is not the track, on any rung.
    """
    audio_files: list[tuple[int, str]] = []
    video_files: list[tuple[int, str]] = []
    playlists: list[tuple[int, str]] = []
    seen: set[str] = set()
    for entry in entries or []:
        url = str((entry or {}).get("url", "") or "")
        if not url.lower().startswith(("http://", "https://")):
            continue
        status = (entry or {}).get("status")
        if isinstance(status, int) and status >= 400:
            continue
        key = url.split("#", 1)[0]
        if key in seen:
            continue
        path = urllib.parse.urlparse(url).path.lower()
        if "/preview/" in path:
            continue
        if _drm_evidence([entry]):
            continue
        ctype = str((entry or {}).get("content_type", "") or "").lower()
        try:
            size = int((entry or {}).get("content_length") or 0)
        except (TypeError, ValueError):
            size = 0
        if path.endswith(".m3u8") or "mpegurl" in ctype:
            playlists.append((size, key))
        elif path.endswith(_CAPTURE_AUDIO_EXTS) \
                or ctype.startswith("audio/"):
            audio_files.append((size, key))
        elif path.endswith(_CAPTURE_VIDEO_EXTS) \
                or ctype.startswith("video/"):
            video_files.append((size, key))
        else:
            continue
        seen.add(key)
    audio_files.sort(key=lambda pair: -pair[0])
    video_files.sort(key=lambda pair: -pair[0])
    playlists.sort(key=lambda pair: -pair[0])
    if audio_only:
        ordered = audio_files + playlists + video_files
    else:
        ordered = video_files + playlists + audio_files
    return [url for _size, url in ordered[:8]]


def _hls_variant_url(master_text: str, base_url: str) -> str:
    """The highest-bandwidth variant URL from an HLS master
    playlist (the first variant when bandwidths are absent)."""
    best_url = ""
    best_bw = -1
    pending_bw = 0
    for line in master_text.splitlines():
        line = line.strip()
        if line.startswith("#EXT-X-STREAM-INF:"):
            m = re.search(r"BANDWIDTH=(\d+)", line)
            pending_bw = int(m.group(1)) if m else 0
        elif line and not line.startswith("#"):
            if pending_bw > best_bw:
                best_bw = pending_bw
                best_url = urllib.parse.urljoin(base_url, line)
            pending_bw = 0
    return best_url

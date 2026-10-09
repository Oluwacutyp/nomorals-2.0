"""Unified media download orchestrator — a real fallback chain.

The old ``media_download`` path was single-shot: yt-dlp → direct HTTP,
and a failure died there.  This module turns every download into a
four-stage chain that keeps trying instead of erroring out:

1. **direct** — the current behavior (yt-dlp python API → CLI → direct
   HTTP), wrapped with retry + exponential backoff on transient
   (network) failures only.  A bot-block is not retried: retrying a
   bot check three times is just slower failure.
2. **proxy** — retry through the proxy lab (:class:`ProxyPoolConnector`
   via :meth:`healthy_proxies`, protocol-aware ordering).  A proxy that
   fails is *demoted* (``record_failure``) and the chain moves to the
   next one — a dead proxy never kills the download.
3. **browser** — open the source page in a persistent
   :class:`BrowserSession` (cookies survive across downloads), extract
   the real media URL (og:video/og:audio meta, ``<video>``/``<audio>``/
   ``<source>`` tags, download links/buttons — clicked when needed),
   then fetch the bytes through the session so the site's cookies and
   headers ride along.  This is the stage the user's complaint asked
   for: the download no longer dies when the direct fetch errors.
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

``download()`` never raises — it returns a :class:`DownloadReport`.
"""

from __future__ import annotations

import os
import re
import sys
import time
import urllib.parse
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .cookies import BOT_COOKIE_HELP, is_bot_detection_error

_log = get_logger(__name__)

__all__ = [
    "FailureKind",
    "StageLog",
    "DownloadReport",
    "MediaDownloader",
    "classify_failure",
    "detect_profile",
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


@dataclass
class DownloadReport:
    """The outcome of ``MediaDownloader.download()`` — never raises."""

    ok: bool = False
    url: str = ""
    path: str = ""
    size_bytes: int = 0
    title: str = ""
    extractor: str = ""
    total_s: float = 0.0
    stages: list[StageLog] = field(default_factory=list)
    failure: FailureKind = FailureKind.OK
    likely_cause: str = ""
    hint: str = ""

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
            return (f"downloaded {self.title or self.url} "
                    f"({self.size_bytes} bytes) via "
                    f"{' → '.join(self.tried()) or 'direct'}")
        lines = [f"download failed: {self.url or '(no url)'}"]
        for i, s in enumerate(self.stages, 1):
            detail = f"{s.error_class}: {s.error}" if s.error else (
                s.note or "ok")
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
    except Exception:  # noqa: BLE001
        pass
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
    except Exception:  # noqa: BLE001
        return "(proxy)"


@contextmanager
def _proxy_env(proxy_url: str):
    """Route subprocess/yt-dlp traffic through a proxy, then restore.

    yt-dlp (python API and CLI) honors ``*_PROXY`` env vars; this scopes
    the override to the wrapped call so no global state leaks.
    """
    keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
            "http_proxy", "https_proxy", "all_proxy")
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ[k] = proxy_url
        yield
    finally:
        for k in keys:
            if saved[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = saved[k]


class MediaDownloader:
    """Four-stage download orchestrator.  ``download()`` never raises."""

    def __init__(self, context: Any = None, *,
                 profile: str | None = None,
                 max_proxies: int | None = None,
                 direct_retries: int | None = None) -> None:
        self.context = context
        self.profile = profile or detect_profile()
        budget = _PROFILE_BUDGETS.get(self.profile, _PROFILE_BUDGETS["laptop"])
        self.max_proxies = max_proxies if max_proxies is not None \
            else budget["max_proxies"]
        self.direct_retries = direct_retries if direct_retries is not None \
            else budget["direct_retries"]
        self.browser_candidates = budget["browser_candidates"]
        self.stage_timeout = budget["stage_timeout_s"]

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

        _log.info("download start url=%s profile=%s audio_only=%s",
                  url, self.profile, audio_only)
        try:
            if self._stage_direct(report, url, target_dir, audio_only,
                                  format_spec, timeout, budget):
                return self._finish(report, started)
            if self._stage_proxy(report, url, target_dir, audio_only,
                                 format_spec, timeout, budget):
                return self._finish(report, started)
            if self._stage_browser(report, url, target_dir, audio_only,
                                   page_url or url, budget):
                return self._finish(report, started)
        except Exception as exc:  # noqa: BLE001 — the chain never raises
            _log.warning("downloader crashed mid-chain on %s: %s",
                         url, exc)
            report.stages.append(StageLog(
                stage="chain", url=url, error_class="site_error",
                error=f"orchestrator error: {exc}"))
        self._finish_failure(report)
        return self._finish(report, started)

    # ── stage 1: direct ───────────────────────────────────────────────

    def _stage_direct(self, report: DownloadReport, url: str,
                      target_dir: Path, audio_only: bool,
                      format_spec: str, timeout: float,
                      budget: dict[str, float]) -> bool:
        """Current behavior, with retry + backoff on network failures."""
        from ..tools.media import download as direct_download

        attempt = 0
        while attempt < max(1, self.direct_retries):
            attempt += 1
            if time.perf_counter() > budget["deadline"]:
                break
            t0 = time.perf_counter()
            try:
                result = direct_download(
                    url, target_dir, format_spec=format_spec,
                    audio_only=audio_only, timeout=timeout)
            except Exception as exc:  # noqa: BLE001
                kind = classify_failure(exc)
                lat = time.perf_counter() - t0
                report.stages.append(StageLog(
                    stage="direct", url=url, attempt=attempt,
                    latency_s=lat, error_class=kind.value,
                    error=str(exc)[:400]))
                self._log_stage("direct", url, "", lat, 0, kind,
                                str(exc)[:200], attempt)
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
            size = int(result.get("bytes") or 0)
            report.stages.append(StageLog(
                stage="direct", url=url, attempt=attempt, latency_s=lat,
                size_bytes=size, ok=True,
                note=f"extractor={result.get('extractor', '')}"))
            self._log_stage("direct", url, "", lat, size, FailureKind.OK,
                            "", attempt)
            self._fill_success(report, result)
            return True
        return False

    # ── stage 2: proxy lab ────────────────────────────────────────────

    def _proxy_pool(self) -> Any | None:
        """The proxy lab connector, or None when unavailable."""
        passphrase = os.environ.get("NM_VAULT_PASSPHRASE", "")
        if not passphrase:
            return None
        try:
            from ..accounts.vault import CredentialVault
            from ..connectors.registry import create_connector
            vault = CredentialVault(getattr(self.context, "db", None),
                                    master_passphrase=passphrase)
            return create_connector("proxypool", vault)
        except Exception as exc:  # noqa: BLE001 — pool optional
            _log.debug("downloader: proxy pool unavailable: %s", exc)
            return None

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
                     "(no NM_VAULT_PASSPHRASE or no stored proxies)"))
            _log.info("download stage=proxy url=%s skipped: pool "
                      "unavailable", url)
            return False
        try:
            proxies = pool.healthy_proxies()
        except Exception as exc:  # noqa: BLE001
            report.stages.append(StageLog(
                stage="proxy", url=url,
                note=f"could not list healthy proxies: {exc}"))
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
        socks_ok = self._socks_usable()
        for proxy in ordered:
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
            t0 = time.perf_counter()
            try:
                with _proxy_env(proxy_url):
                    result = direct_download(
                        url, target_dir, format_spec=format_spec,
                        audio_only=audio_only, timeout=timeout)
            except Exception as exc:  # noqa: BLE001 — demote, don't die
                kind = classify_failure(exc)
                lat = time.perf_counter() - t0
                try:
                    pool.record_failure(pid, f"{kind.value}: {exc}"[:300])
                except Exception as demote_exc:  # noqa: BLE001
                    _log.debug("proxy demote failed for %s: %s",
                               pid, demote_exc)
                report.stages.append(StageLog(
                    stage="proxy", url=url, proxy=masked, latency_s=lat,
                    error_class=kind.value, error=str(exc)[:400],
                    note=f"proxy demoted; trying next"))
                self._log_stage("proxy", url, masked, lat, 0, kind,
                                str(exc)[:200])
                continue
            lat = time.perf_counter() - t0
            size = int(result.get("bytes") or 0)
            try:
                pool.record_success(pid, latency_ms=lat * 1000)
            except Exception:  # noqa: BLE001 — bookkeeping only
                pass
            report.stages.append(StageLog(
                stage="proxy", url=url, proxy=masked, latency_s=lat,
                size_bytes=size, ok=True,
                note=f"extractor={result.get('extractor', '')}"))
            self._log_stage("proxy", url, masked, lat, size,
                            FailureKind.OK, "")
            self._fill_success(report, result)
            return True
        return False

    # ── stage 3: browser context ──────────────────────────────────────

    def _browser_session(self) -> Any | None:
        try:
            from ..tools.browser import get_session
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
        except Exception:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
            raw_html = ""
        for m in _SRC_TAG_RE.finditer(raw_html or ""):
            add(m.group(1))
        try:
            links = (session.links(max_links=200) or {}).get("links", [])
        except Exception:  # noqa: BLE001
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
                error_class=kind.value, error=str(exc)[:400],
                note="could not open the source page in the browser"))
            self._log_stage("browser", page_url, "", lat, 0, kind,
                            str(exc)[:200])
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
        candidates = self._media_candidates(session)
        report.stages.append(StageLog(
            stage="browser", url=page_url,
            note=f"page opened; {len(candidates)} media candidate(s) "
                 f"extracted"))
        if not candidates:
            report.stages.append(StageLog(
                stage="browser", url=page_url,
                error_class=FailureKind.NO_MEDIA_FOUND.value,
                error="page has no og:video/og:audio, media tags, or "
                      "download links"))
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
        t0 = time.perf_counter()
        try:
            got = fetch(media_url, dest)
        except Exception as exc:  # noqa: BLE001
            kind = classify_failure(exc)
            lat = time.perf_counter() - t0
            report.stages.append(StageLog(
                stage="browser", url=media_url, latency_s=lat,
                error_class=kind.value, error=str(exc)[:400]))
            self._log_stage("browser", media_url, "", lat, 0, kind,
                            str(exc)[:200])
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
            except OSError:
                pass
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
            except OSError:
                pass
            return False
        report.stages.append(StageLog(
            stage="browser", url=media_url, latency_s=lat,
            size_bytes=size, ok=True,
            note=f"content-type={ctype or '?'} via browser session"))
        self._log_stage("browser", media_url, "", lat, size,
                        FailureKind.OK, "")
        report.ok = True
        report.failure = FailureKind.OK
        report.path = str(dest)
        report.size_bytes = size
        report.title = dest.stem
        report.extractor = "browser-session"
        return True

    # ── failure + finishing ───────────────────────────────────────────

    def _finish_failure(self, report: DownloadReport) -> None:
        kinds = [s.error_class for s in report.stages if s.error_class]
        notes = " ".join(s.note for s in report.stages).lower()
        browser_opened = any(
            s.stage == "browser" and "page opened" in (s.note or "")
            for s in report.stages)
        if FailureKind.BOT_BLOCK.value in kinds:
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
            report.hint = ("if the page needs login, sign in once in the "
                           "browser session (`nm browser`) so its cookies "
                           "persist, then retry")
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
        except Exception:  # noqa: BLE001
            fallback = Path.home() / "workspace" / "media"
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback

    @staticmethod
    def _fill_success(report: DownloadReport,
                      result: dict[str, Any]) -> None:
        report.ok = True
        report.path = str(result.get("path", ""))
        report.size_bytes = int(result.get("bytes") or 0)
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


def _safe_name(url: str, audio_only: bool) -> str:
    """Filesystem-safe filename for a browser-stage download."""
    from ..core.http import url_filename
    name = url_filename(url)
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" .")[:120]
    if not name or "." not in name:
        name = (name or "media") + (".mp3" if audio_only else ".bin")
    return name or "media.bin"

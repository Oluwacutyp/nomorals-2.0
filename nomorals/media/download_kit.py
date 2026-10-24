"""Step 5 — Downloader kit: extractors, resumable fetch, Cloudflare,
disk preflight, integrity ledger.

Companion to :mod:`nomorals.media.downloader` (the 4-stage orchestrator:
direct → proxy → browser → honest failure).  That module owns the
CHAIN; this module owns the PIECES the chain was missing:

* :class:`ExtractorDescriptor` / :class:`ExtractorRegistry` — one
  descriptor per site family: URL pattern, auth need, extraction rule,
  quality map.  yt-dlp covers ~1800 sites through its generic
  descriptor; the rest get explicit descriptors (Audiomack, Boomplay
  metadata-only, NetNaija scrape, Spotify no-download, ...).  No
  per-domain ``if`` in the download path — descriptors decide.
  (Pass #76: descriptors are DERIVED from
  :mod:`nomorals.media.source_registry` — the one source of truth for
  sources; this registry is a view over it, not a second list.)
* :class:`ResumeDownloader` — resume with ETag / Last-Modified
  verification.  INVARIANT: if the validators changed, restart from
  byte 0 — NEVER splice a stale partial onto fresh bytes.
* :class:`CloudflareStateMachine` — named states incl. the terminal
  ``cloudflare_blocked``.  When a challenge needs a cookie, the machine
  asks the owner ONCE (vault-stored, with expiry) and never retries
  the challenge in a loop.
* :func:`disk_preflight` — need X, have Y, fail early before a byte
  moves.
* :class:`IntegrityLedger` — content-hash + source URL per download;
  :meth:`IntegrityLedger.verify` recomputes the hash and catches
  tampering / truncation.

Never raises out of the public functions (except where a docstring
says otherwise) — every outcome is a report dataclass.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from ..core.logging_setup import get_logger
from .source_registry import DEFAULT_REGISTRY, Licence, SourceRegistry
from .download_kit_failures import record_failure

_log = get_logger(__name__)

__all__ = [
    "ExtractorDescriptor",
    "ExtractorRegistry",
    "descriptors_for_registry",
    "DEFAULT_EXTRACTORS",
    "CloudflareState",
    "CloudflareStateMachine",
    "ResumeReport",
    "ResumeDownloader",
    "Preflight",
    "disk_preflight",
    "IntegrityLedger",
    "get_or_request_cookie",
    "store_owner_cookie",
]


# ── extractor descriptors ─────────────────────────────────────────────

@dataclass(frozen=True)
class ExtractorDescriptor:
    """One site family: how to extract media from its URLs."""

    name: str = ""
    url_patterns: tuple[str, ...] = ()  # regexes matched against the URL
    auth_needed: bool = False
    extraction_rule: str = "ytdlp"  # ytdlp | scrape | direct | api | none
    quality_map: tuple[tuple[str, str], ...] = ()  # (label, ytdlp format)
    licence: Licence = Licence.UNKNOWN
    notes: str = ""
    downloadable: bool = True

    def matches(self, url: str) -> bool:
        try:
            return any(re.search(p, url or "", re.I)
                       for p in self.url_patterns)
        except Exception as e:
            record_failure("matches", e)
            return False

    def format_for(self, quality: str) -> str:
        """yt-dlp format string for a quality label; "" = default."""
        try:
            for label, fmt in self.quality_map:
                if label.lower() == (quality or "").lower():
                    return fmt
            return ""
        except Exception as e:
            record_failure("format_for", e)
            return ""


#: Quality map shared by the structural fallback descriptors.
_Q = (("360p", "best[height<=360]"), ("720p", "best[height<=720]"),
      ("1080p", "best[height<=1080]"), ("audio", "bestaudio/best"))


def _structural_fallbacks() -> list[ExtractorDescriptor]:
    """Descriptors that are NOT sources — URL-shape fallbacks closing
    every extractor view: a direct media file, then the yt-dlp generic
    catch-all (LAST in order — every specific descriptor wins)."""
    return [
        ExtractorDescriptor(
            name="direct-media", url_patterns=(r"\.(mp3|m4a|ogg|opus|wav|flac|mp4|webm|mkv|mov)(\?|$)",),
            extraction_rule="direct", quality_map=(("audio", "bestaudio/best"),),
            licence=Licence.UNKNOWN,
            notes="Direct media file URL — plain HTTP fetch"),
        ExtractorDescriptor(
            name="ytdlp-generic", url_patterns=(r"^https?://",),
            extraction_rule="ytdlp", quality_map=_Q,
            licence=Licence.UNKNOWN,
            notes="yt-dlp generic: routes ~1800 sites dynamically"),
    ]


def descriptors_for_registry(registry: SourceRegistry) -> list[ExtractorDescriptor]:
    """Derive extractor descriptors from a source registry (Pass #76).

    The source registry (:mod:`nomorals.media.source_registry`) is
    the ONE source of truth for download sources: every source that
    declares extraction facts (rule / patterns / quality map /
    downloadable) becomes a descriptor here.  Sources without facts
    (news, discovery, search) get no descriptor and fall through to
    the structural fallbacks — the same behaviour they had while this
    list was hand-maintained separately (and drifted).

    Pass #87: the facts read here are the registry's EFFECTIVE facts
    (``effective_extraction_rule`` / ``effective_quality_map``) — a
    source whose hand-written rule is empty but which LEARNED a rule
    from real download evidence surfaces here as its own descriptor,
    ahead of the generic fallback.  Still a pure view: precedence
    and evidence live in the registry, never in this module."""
    out: list[ExtractorDescriptor] = []
    try:
        for src in registry.all():
            rule = src.extraction_rule
            qmap = src.quality_map
            eff_rule = getattr(registry, "effective_extraction_rule", None)
            if callable(eff_rule):
                try:
                    rule = eff_rule(src) or ""
                except Exception as e:  # view degrades to seed facts
                    record_failure("descriptors_for_registry:effective_rule", e)
                    rule = src.extraction_rule
            if not rule:
                continue
            if not qmap:
                eff_qmap = getattr(registry, "effective_quality_map", None)
                if callable(eff_qmap):
                    try:
                        qmap = eff_qmap(src) or ()
                    except Exception as e:
                        record_failure("descriptors_for_registry:effective_qmap", e)
                        qmap = src.quality_map
            out.append(ExtractorDescriptor(
                name=(src.name or src.domain).lower().replace(" ", ""),
                url_patterns=src.url_patterns or (re.escape(src.domain),),
                auth_needed=src.auth_needed,
                extraction_rule=rule,
                quality_map=qmap,
                licence=src.licence,
                notes=src.notes,
                downloadable=src.downloadable))
    except Exception as e:
        record_failure("descriptors_for_registry", e)
    return out + _structural_fallbacks()


def _default_extractors() -> list[ExtractorDescriptor]:
    return descriptors_for_registry(DEFAULT_REGISTRY)


class ExtractorRegistry:
    """A VIEW over a :class:`SourceRegistry` (Pass #76 convergence):
    descriptors in priority order, derived from the registry's
    extraction facts plus the structural fallbacks.  Pass
    ``descriptors=`` for an explicit list (tests, one-off callers) or
    ``registry=`` to view a different registry instance.
    :meth:`for_url` returns every matching descriptor, specific
    first, generic last."""

    def __init__(self, descriptors: list[ExtractorDescriptor] | None = None,
                 *, registry: SourceRegistry | None = None) -> None:
        if descriptors is not None:
            self._descs: list[ExtractorDescriptor] = list(descriptors)
        elif registry is not None:
            self._descs = descriptors_for_registry(registry)
        else:
            self._descs = _default_extractors()

    def register(self, desc: ExtractorDescriptor) -> None:
        try:
            if desc.name and all(d.name != desc.name for d in self._descs):
                # Specific descriptors stay ahead of the generic catch-all.
                generic = [d for d in self._descs if d.name == "ytdlp-generic"]
                rest = [d for d in self._descs if d.name != "ytdlp-generic"]
                self._descs = rest + [desc] + generic
        except Exception as e:
            record_failure("ExtractorRegistry.register", e)

    def for_url(self, url: str) -> list[ExtractorDescriptor]:
        try:
            return [d for d in self._descs if d.matches(url or "")]
        except Exception as e:
            record_failure("ExtractorRegistry.for_url", e)
            return []

    def best_for(self, url: str) -> ExtractorDescriptor | None:
        found = self.for_url(url)
        return found[0] if found else None


DEFAULT_EXTRACTORS = ExtractorRegistry()


# ── Cloudflare state machine ──────────────────────────────────────────

class CloudflareState(str, Enum):
    """Named states.  Terminal: PASSED, BLOCKED, FAILED, COOKIE_EXPIRED."""

    IDLE = "idle"
    CHALLENGE_DETECTED = "challenge_detected"
    WAITING_COOKIE = "waiting_cookie"      # asked owner once, awaiting cookie
    COOKIE_PROVIDED = "cookie_provided"
    PASSED = "passed"                      # terminal: challenge cleared
    BLOCKED = "cloudflare_blocked"         # terminal: still blocked
    FAILED = "failed"                      # terminal: non-CF failure
    COOKIE_EXPIRED = "cookie_expired"      # terminal: stored cookie stale


_TERMINAL = frozenset({
    CloudflareState.PASSED, CloudflareState.BLOCKED,
    CloudflareState.FAILED, CloudflareState.COOKIE_EXPIRED,
})

_CF_MARKERS = ("cloudflare", "cf_chl", "cf_clearance", "__cf_bm",
               "attention required", "checking your browser",
               "just a moment", "verify you are human")


def is_cloudflare_challenge(text: str) -> bool:
    try:
        low = (text or "").lower()
        return any(m in low for m in _CF_MARKERS)
    except Exception as e:
        record_failure("is_cloudflare_challenge", e)
        return False


class CloudflareStateMachine:
    """Explicit state machine — no boolean soup, no retry loops.

    Transitions: IDLE → CHALLENGE_DETECTED (on challenge signal) →
    WAITING_COOKIE (ask owner once) → COOKIE_PROVIDED → PASSED | BLOCKED.
    A second challenge after a cookie → COOKIE_EXPIRED (terminal) or
    BLOCKED — never a second ask, never a retry loop."""

    def __init__(self) -> None:
        self.state = CloudflareState.IDLE
        self.cookie_asked = False
        self.transitions: list[tuple[str, str]] = []

    @property
    def terminal(self) -> bool:
        return self.state in _TERMINAL

    def _move(self, nxt: CloudflareState) -> CloudflareState:
        self.transitions.append((self.state.value, nxt.value))
        self.state = nxt
        return self.state

    def feed(self, signal: str, *, detail: str = "") -> CloudflareState:
        """Feed one signal: "challenge" | "cookie_provided" | "success" |
        "blocked" | "failed" | "cookie_expired".  Illegal transitions
        are ignored (state stays), never raise."""
        try:
            sig = (signal or "").lower()
            if self.terminal:
                return self.state
            if sig == "challenge":
                if self.state is CloudflareState.COOKIE_PROVIDED:
                    return self._move(CloudflareState.COOKIE_EXPIRED)
                if self.state in (CloudflareState.IDLE,
                                  CloudflareState.CHALLENGE_DETECTED):
                    return self._move(CloudflareState.CHALLENGE_DETECTED)
                return self.state
            if sig == "cookie_provided":
                if self.state in (CloudflareState.WAITING_COOKIE,
                                  CloudflareState.CHALLENGE_DETECTED):
                    return self._move(CloudflareState.COOKIE_PROVIDED)
                return self.state
            if sig == "success":
                return self._move(CloudflareState.PASSED)
            if sig == "blocked":
                return self._move(CloudflareState.BLOCKED)
            if sig == "failed":
                return self._move(CloudflareState.FAILED)
            if sig == "cookie_expired":
                return self._move(CloudflareState.COOKIE_EXPIRED)
            return self.state
        except Exception as e:
            record_failure("CloudflareStateMachine.feed", e)
            return self.state

    def ask_owner(self) -> str:
        """Ask the owner for a cookie — exactly ONCE.  Returns the ask
        message, or "" when already asked (caller must not re-ask)."""
        try:
            if self.cookie_asked:
                return ""
            self.cookie_asked = True
            self._move(CloudflareState.WAITING_COOKIE)
            return ("Cloudflare is challenging this site. Paste a fresh "
                    "`cf_clearance` cookie value (from your own browser, "
                    "your own session) and I'll store it in the vault with "
                    "an expiry — I won't ask again for this download.")
        except Exception as e:
            record_failure("CloudflareStateMachine.ask_owner", e)
            return ""


# ── owner cookie (vault, expiry) ──────────────────────────────────────

#: Vault key namespace for owner-provided clearance cookies.
_COOKIE_NS = "cloudflare_cookies"


def store_owner_cookie(store: Any, domain: str, cookie_value: str,
                       ttl_s: float = 3600.0) -> bool:
    """Store an OWNER-PROVIDED cookie in the vault with an expiry.
    Never touches real accounts, never scrapes a login.  Returns False
    on any problem.  ``store`` is a mapping-like vault (get/set)."""
    try:
        if not cookie_value or not domain:
            return False
        store.set(f"{_COOKIE_NS}:{domain.lower()}",
                  {"value": cookie_value, "expires_at": time.time() + ttl_s,
                   "domain": domain.lower()})
        return True
    except Exception as e:
        record_failure("store_owner_cookie", e)
        return False


def get_or_request_cookie(store: Any, domain: str,
                          now: float = 0.0) -> tuple[str, str]:
    """("have", value) when a live vault cookie exists;
    ("ask", message) when the owner must provide one once;
    ("expired", message) when the stored cookie is stale.
    Never raises; never fabricates a cookie."""
    try:
        now = now or time.time()
        entry = None
        try:
            entry = store.get(f"{_COOKIE_NS}:{domain.lower()}")
        except Exception as e:
            record_failure("get_or_request_cookie:vault_read", e)
            entry = None
        if isinstance(entry, dict) and entry.get("value"):
            if float(entry.get("expires_at") or 0) > now:
                return "have", str(entry["value"])
            return "expired", (
                f"the stored Cloudflare cookie for {domain} expired — "
                "paste a fresh cf_clearance value from your own browser.")
        return "ask", (
            f"no Cloudflare cookie stored for {domain} — paste a fresh "
            "cf_clearance value from your own browser (your own session), "
            "and I'll store it in the vault with an expiry.")
    except Exception as e:
        record_failure("get_or_request_cookie", e)
        return "ask", "paste a fresh cf_clearance cookie value."


# ── disk preflight ────────────────────────────────────────────────────

@dataclass
class Preflight:
    ok: bool = False
    need_bytes: int = 0
    have_bytes: int = 0
    dest: str = ""
    message: str = ""


def disk_preflight(need_bytes: int, dest_dir: str | Path,
                   *, reserve_bytes: int = 10 * 1024 * 1024,
                   disk_usage: Callable[[str], Any] | None = None) -> Preflight:
    """Need X, have Y, fail early.  ``reserve_bytes`` keeps a safety
    margin so a download never fills the disk to zero.  Never raises."""
    try:
        dest = str(dest_dir or "")
        usage = disk_usage or shutil.disk_usage
        total, _used, free = usage(dest or "/")
        need = max(0, int(need_bytes or 0)) + int(reserve_bytes)
        ok = free >= need
        return Preflight(
            ok=ok, need_bytes=need, have_bytes=int(free), dest=dest,
            message=(f"disk ok: need {need} bytes, have {free} bytes"
                     if ok else
                     f"disk full: need {need} bytes, have {free} bytes — "
                     "free space or pick a smaller download first"))
    except Exception as exc:  # noqa: BLE001
        return Preflight(ok=False, need_bytes=int(need_bytes or 0),
                         dest=str(dest_dir or ""),
                         message=f"disk check failed: {exc}")


# ── resumable download ────────────────────────────────────────────────

@dataclass
class ResumeReport:
    ok: bool = False
    path: str = ""
    size_bytes: int = 0
    resumed: bool = False
    restarted: bool = False   # True when validators changed → fresh start
    validators: dict[str, str] = field(default_factory=dict)
    error: str = ""
    sha256: str = ""


class ResumeDownloader:
    """Resume with ETag / Last-Modified verification.

    INVARIANT: if the validators on the server changed since the partial
    was written, the partial is DISCARDED and the download restarts from
    byte 0 — never splice.  A splice across a changed file is silent
    corruption; this class would rather re-download than corrupt.

    Transports are injectable (``head`` / ``get_range``) so tests can
    drive the validator logic without a network."""

    def __init__(self, *,
                 head: Callable[[str], dict[str, Any]] | None = None,
                 get_range: Callable[..., bytes] | None = None) -> None:
        self._head = head or self._default_head
        self._get_range = get_range or self._default_get_range

    # ── default transports (urllib) ──
    @staticmethod
    def _default_head(url: str) -> dict[str, Any]:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "DevonBot/1.0"})
        with urllib.request.urlopen(req, timeout=20.0) as resp:
            return {"status": resp.status,
                    "etag": resp.headers.get("ETag", ""),
                    "last_modified": resp.headers.get("Last-Modified", ""),
                    "content_length": resp.headers.get("Content-Length", ""),
                    "accept_ranges": resp.headers.get("Accept-Ranges", "")}

    @staticmethod
    def _default_get_range(url: str, start: int, end: int | None,
                           headers: dict[str, str] | None = None) -> bytes:
        hdrs = {"User-Agent": "DevonBot/1.0"}
        if start or end is not None:
            if end is None:
                hdrs["Range"] = f"bytes={start}-"
            else:
                hdrs["Range"] = f"bytes={start}-{end}"
        hdrs.update(headers or {})
        req = urllib.request.Request(url, headers=hdrs)
        with urllib.request.urlopen(req, timeout=60.0) as resp:
            return resp.read()

    # ── main path ──
    def download(self, url: str, dest: str | Path,
                 *, chunk_size: int = 1 << 20) -> ResumeReport:
        try:
            dest_p = Path(dest)
            dest_p.parent.mkdir(parents=True, exist_ok=True)
            meta_p = dest_p.with_suffix(dest_p.suffix + ".resume.json")
            try:
                head = self._head(url)
            except Exception as exc:  # noqa: BLE001
                return ResumeReport(error=f"HEAD failed: {exc}")
            validators = {"etag": str(head.get("etag") or ""),
                          "last_modified": str(head.get("last_modified") or "")}
            total = 0
            try:
                total = int(head.get("content_length") or 0)
            except (TypeError, ValueError):
                total = 0
            partial = dest_p.read_bytes() if dest_p.is_file() else b""
            saved: dict[str, Any] = {}
            if meta_p.is_file():
                try:
                    saved = json.loads(meta_p.read_text(encoding="utf-8"))
                except Exception as e:
                    record_failure("ResumeDownloader.download:meta_read", e)
                    saved = {}
            validators_changed = bool(partial) and bool(saved) and (
                saved.get("etag") != validators["etag"]
                or saved.get("last_modified") != validators["last_modified"])
            # No saved validators + a partial = cannot prove the partial
            # matches the current file → restart (never splice blind).
            if partial and not saved:
                validators_changed = True
            report = ResumeReport(validators=validators)
            if validators_changed:
                _log.info("resume: validators changed for %s — restarting, "
                          "not splicing", url[:80])
                partial = b""
                report.restarted = True
            start = len(partial)
            if total and start >= total and partial:
                # Already complete — verify hash only if we stored one.
                digest = hashlib.sha256(partial).hexdigest()
                dest_p.write_bytes(partial)
                return ResumeReport(ok=True, path=str(dest_p),
                                    size_bytes=len(partial), resumed=False,
                                    validators=validators, sha256=digest)
            try:
                chunk = self._get_range(url, start, None)
            except Exception as exc:  # noqa: BLE001
                # Save the partial + validators so the NEXT attempt can
                # resume honestly.
                try:
                    dest_p.write_bytes(partial)
                    meta_p.write_text(json.dumps(validators), encoding="utf-8")
                except OSError as e:
                    record_failure("ResumeDownloader.download:partial_save", e)
                return ResumeReport(error=f"fetch failed: {exc}",
                                    validators=validators,
                                    size_bytes=len(partial))
            data = partial + (chunk or b"")
            dest_p.write_bytes(data)
            try:
                meta_p.unlink(missing_ok=True)
            except OSError as e:
                record_failure("ResumeDownloader.download:meta_unlink", e)
            digest = hashlib.sha256(data).hexdigest()
            return ResumeReport(ok=True, path=str(dest_p),
                                size_bytes=len(data),
                                resumed=start > 0 and not report.restarted,
                                restarted=report.restarted,
                                validators=validators, sha256=digest)
        except Exception as exc:  # noqa: BLE001 — never raises
            return ResumeReport(error=f"resume download failed: {exc}")


# ── integrity ledger ──────────────────────────────────────────────────

class IntegrityLedger:
    """Content-hash + source URL per download.  JSONL, append-only.
    :meth:`verify` recomputes the hash — tampering / truncation is
    caught, not assumed away."""

    def __init__(self, path: str | Path | None = None) -> None:
        try:
            if path:
                self.path = Path(path).expanduser()
            else:
                self.path = (Path.home() / ".cache" / "nomorals"
                             / "integrity-ledger.jsonl")
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            # tempfile.gettempdir() honours TMPDIR/TEMP/TMP on every OS;
            # a hardcoded "/tmp" is C:\\tmp on Windows (often unwritable).
            record_failure("IntegrityLedger.__init__", e)
            self.path = Path(tempfile.gettempdir()) / "nomorals-integrity-ledger.jsonl"

    def append(self, *, url: str, sha256: str, size_bytes: int,
               source: str = "", dest: str = "") -> bool:
        try:
            row = {"url": url, "sha256": sha256, "size_bytes": int(size_bytes),
                   "source": source, "dest": dest, "ts": time.time()}
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            return True
        except Exception as e:
            record_failure("IntegrityLedger.append", e)
            return False

    def verify(self, dest: str | Path, *, url: str = "") -> tuple[bool, str]:
        """Recompute the hash of *dest* and compare with the ledger.
        Returns (ok, reason).  Never raises."""
        try:
            p = Path(dest)
            if not p.is_file():
                return False, "file missing"
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            rows = []
            try:
                for line in self.path.read_text(encoding="utf-8").splitlines():
                    try:
                        rows.append(json.loads(line))
                    except Exception as e:
                        record_failure("IntegrityLedger.verify:bad_line", e)
                        continue
            except OSError:
                return False, "no ledger entries"
            candidates = [r for r in rows
                          if (not url or r.get("url") == url)
                          and r.get("dest") == str(p)]
            if not candidates:
                candidates = [r for r in rows
                              if r.get("sha256") == digest]
            if not candidates:
                return False, "no ledger entry for this file"
            latest = candidates[-1]
            if latest.get("sha256") != digest:
                return False, "hash mismatch — file changed since download"
            if int(latest.get("size_bytes") or 0) != p.stat().st_size:
                return False, "size mismatch — file truncated or extended"
            return True, "hash + size match ledger"
        except Exception as exc:  # noqa: BLE001
            return False, f"verify failed: {exc}"
